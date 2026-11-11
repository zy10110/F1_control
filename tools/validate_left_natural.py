"""Replay mirrored left-hand validation takes through the native F1 executor."""
import json
import numpy as np
from hand_kinematics import FINGERS
from g7_teleop import BASE
from urdf_a_model import NativeHand
from urdf_a_follow import Follower
from tools.calibrate_angle_map import write


def replay(g,profile,frames,missing):
    m=Follower(g,profile,middle_unavailable='middle' in missing)
    digits=[f for f in FINGERS[1:] if f not in missing]
    m.command(frames[0])
    for _ in range(40):m.advance(.025)
    previous=None;errors=[];sources=[];actual=[];clearances=[];blocked=0
    for frame in frames:
        stamp=frame['timestamp_ns']
        dt=.025 if previous is None else (stamp-previous)*1e-9
        if dt<=0:raise ValueError('验证记录时间戳倒退')
        previous=stamp;before=m.u.copy()
        m.command(frame);m.advance(dt);g.validate(m.u)
        clearance=g.clearance();clearances.append(clearance)
        if clearance<.6-1.e-7 or np.max(abs(np.degrees(m.u-before)))>m.config['max_joint_speed_deg_s']*dt+1.e-5:
            raise ValueError('左手回放越过间隙或速度限制')
        blocked+=int(m.blocked)
        points=np.asarray(frame['positions_m'])
        sources.append([1000*np.linalg.norm(points[4]-points[BASE[f]+3]) for f in digits])
        actual.append([g.gaps(f)['pad_mm'] for f in digits])
        for finger in ['thumb',*digits]:
            v=np.diff(m.chains[finger],axis=0)@m.display_rotation.T
            v/=np.linalg.norm(v,axis=1,keepdims=True)
            errors.extend(np.degrees(np.arccos(np.clip((g.bone_directions(finger)*v).sum(1),-1,1))))
    src=np.asarray(sources);dst=np.asarray(actual)
    keep=(src>=8)&(src<=30);touch=src<=3
    # Take medians over all qualified finger-frame observations; include counts
    # so an absent contact or an unrelated finger cannot be reported as a hit.
    return dict(frames=len(frames),clearance_min_mm=min(clearances),blocked_frames=blocked,
                mean_direction_deg=float(np.mean(errors)),near_samples=int(keep.sum()),
                near_error_mm=float(np.median(abs(src[keep]-dst[keep]))) if keep.any() else None,
                false_closures=int(np.sum(keep&(dst<3))),contact_samples=int(touch.sum()),
                contact_pad_mm=float(np.median(dst[touch])) if touch.any() else None)


def run(package,profile,manifest):
    session=json.loads(manifest.read_bytes());baseline=manifest.parent/'baseline_profile.json'
    if session['input_hand']!='left' or json.loads(profile.read_bytes()).get('input_hand')!='left':
        raise ValueError('左手补录与档案手别不符')
    g=NativeHand(package); entries=session['validation']
    if {entry['stage'] for entry in entries}!={'near_pass','touch_release'}:raise ValueError('左手补录验证动作不全')
    result={'supplement_validation':{},'issues':[]}
    for entry in entries:
        frames=[json.loads(line) for line in (manifest.parent/entry['file']).read_text(encoding='utf8').splitlines()]
        if not frames or not all(f.get('source_hand')==1 and f.get('hand')==2 and f.get('pose_space')=='mirrored_right' for f in frames):
            raise ValueError('左手验证记录缺少镜像坐标标记')
        old,new=[replay(g,p,frames[::3],session.get('unavailable_fingers',[])) for p in (baseline,profile)]
        result['supplement_validation'][entry['stage']]=dict(previous=old,refitted=new)
        if entry['stage']=='near_pass' and not new['near_samples']:
            result['issues'].append('留缝验证记录缺少 8–30 mm 的观测')
        measures=[('mean_direction_deg',2.),('blocked_frames',0),('false_closures',0)]
        measures.append(('near_error_mm',1.))
        if entry['stage']=='touch_release':measures.append(('contact_pad_mm',1.))
        for key,tolerance in measures:
            if new[key] is not None and new[key]>old[key]+tolerance:
                stage_name={'near_pass':'留缝擦过','touch_release':'轻碰再松开'}[entry['stage']]
                label={'near_error_mm':'间距误差 (mm)','contact_pad_mm':'轻碰间距 (mm)',
                       'mean_direction_deg':'指节方向误差 (°)','blocked_frames':'受阻帧数',
                       'false_closures':'留缝时误合拢次数'}[key]
                result['issues'].append(f"{stage_name}的{label}从 {old[key]:.2f} 增至 {new[key]:.2f}")
    result['accepted']=not result['issues']
    write(manifest.parent/'validation.json',result)
    if result['issues']:raise ValueError('左手留缝和松开回放：'+'；'.join(result['issues']))
    return result
