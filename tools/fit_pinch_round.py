"""Incremental, angle-only fitting from independent HumanHand-style pinch holds."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from hand_kinematics import FINGERS, ROOT
from g7_official_udp import open_reference
from g7_teleop import BASE
from urdf_a_follow import Follower
from urdf_a_model import NativeHand
from tools.calibrate_angle_map import write


def take_gaps(rows,finger):
    points=np.asarray([r['positions_m'] for r in rows])
    return np.linalg.norm(points[:,4]-points[:,BASE[finger]+3],axis=1)*1000


def check_take(rows,finger,split,unavailable_fingers=()):
    gaps=take_gaps(rows,finger)
    if np.any(~np.isfinite(gaps)):
        raise ValueError('指尖位置无效，请重新采集')
    if split=='train':
        open_reference(rows)
        if np.median(gaps)>32:
            raise ValueError(f'官方手模两指尖间距约 {np.median(gaps):.0f} mm。请确认目标手指，摆好后重采。')
        for other in FINGERS[1:]:
            if other in unavailable_fingers:continue
            if other!=finger and np.median(take_gaps(rows,other))+8<np.median(gaps):
                raise ValueError('另一根手指比指定手指更靠近拇指，请放松其余手指后重采')
    else:
        low=np.percentile(gaps,15)
        n=max(1,len(gaps)//10)
        if low>32 or min(np.median(gaps[:n]),np.median(gaps[-n:]))<low+12:
            raise ValueError('请从张开开始，轻贴约三秒，最后完全松开，再采一次。')


def read_take(folder,entry):
    return [json.loads(s) for s in (folder/entry['file']).read_text(encoding='utf8').splitlines()]


def validate_mirrored_take(rows):
    if not rows or not all(r.get('source_hand')==1 and r.get('hand')==2 and r.get('pose_space')=='mirrored_right' for r in rows):
        raise ValueError('左手对指记录缺少镜像关节位姿')


def spacing_error(result):
    source=result['source_tip_mm']
    if source<=3.:
        return max(0.,result['hold_pad_mm']-3.)
    return abs(result['hold_pad_mm']-source)


def fit_holds(g,old,record,folder,manifest):
    by_digit={f:[] for f in FINGERS}
    mirrored=record.get('input_hand')=='left'
    for entry in manifest['train']:
        f=entry['stage']
        rows=read_take(folder,entry);check_take(rows,f,'train',manifest.get('unavailable_fingers',()))
        if mirrored:validate_mirrored_take(rows)
        frame=open_reference(rows)
        pad_target=max(2.5,float(np.median(take_gaps(rows,f)))) if mirrored else 1.1
        base=old.command(frame).copy()
        feat,chains=old.source.read(frame)
        directions={}
        for digit in ('thumb',f):
            vectors=np.diff(chains[digit],axis=0)@old.display_rotation.T
            directions[digit]=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
        ids=np.r_[g.cols['thumb'],g.cols[f]]
        def pose(x):
            u=base.copy();u[ids]=x;return u
        def objective(x):
            g.assign(pose(x));a,n=g.pad_pose('thumb');b,m=g.pad_pose(f)
            shape=sum(np.sum((g.bone_directions(d)-target)**2) for d,target in directions.items())
            return np.sum((a-b+pad_target*n)**2)+16*np.sum((n+m)**2)+100*shape+np.sum((np.r_[[6.]*5,[9.]*3]*(x-base[ids]))**2)
        def clearance(x):
            g.assign(pose(x));return g.clearance()-1.05
        budget=np.radians(24.)
        result=minimize(objective,base[ids],method='SLSQP',
            bounds=list(zip(np.maximum(g.lo[ids],base[ids]-budget),np.minimum(g.hi[ids],base[ids]+budget))),
            constraints=[dict(type='ineq',fun=clearance)],options=dict(maxiter=120,ftol=1e-6))
        if not np.isfinite(result.x).all() or clearance(result.x)<-.02:
            raise ValueError('本轮标定姿态未通过 A 版间隙检查 '+f)
        for digit in ('thumb',f):
            z,b,W=old.blocks[digit];x=feat[digit]-z
            previous=(base if mirrored else old.raw_angles(feat))[g.cols[digit]]
            desired=pose(result.x)[g.cols[digit]]
            by_digit[digit].append((x,np.clip(desired-previous,-budget,budget)))
    # Add smooth digit-local angular residuals to the live profile. No
    # runtime state or cross-finger distance is used to activate a correction.
    for f,samples in by_digit.items():
        if not samples:continue
        if mirrored:
            block=record.setdefault('pinch_angle_updates',{}).setdefault(f,dict(
                centers=[],coefficients=[],width=float(np.radians(14))))
        else:
            block=record['angular_corrections'][f]
        centers=np.array([x for x,_ in samples]);targets=np.array([v for _,v in samples])
        width=block['width']
        def kernel(x):
            return np.exp(-.5*np.sum(((x[:,None]-centers[None,:])/width)**2,axis=2))
        K=kernel(centers)
        previous=kernel(np.asarray(block['centers']).reshape(-1,centers.shape[1]))
        protection=previous*(1-previous.max(axis=1))[:,None]
        system=K.T@K+protection.T@protection/max(1,len(previous))+.025*np.eye(len(K))
        coef=np.linalg.solve(system,K.T@targets)
        block['centers']=np.vstack((np.asarray(block['centers']).reshape(-1,centers.shape[1]),centers)).tolist()
        block['coefficients']=np.vstack((np.asarray(block['coefficients']).reshape(-1,coef.shape[1]),coef)).tolist()
    record['pinch_round']=str((folder/'manifest.json').resolve())


def replay(g,path,rows,f=None,middle_unavailable=None):
    m=Follower(g,path,middle_unavailable=middle_unavailable);last=None;gaps=[];shape=[];clear=[];blocked=0
    frames=rows[::3]
    m.command(frames[0])
    for _ in range(30):m.advance(.025)
    for frame in frames:
        stamp=frame['timestamp_ns'];dt=.025 if last is None else (stamp-last)*1e-9;last=stamp
        m.command(frame);previous=m.u.copy();m.advance(dt)
        clearance=g.clearance();clear.append(clearance)
        if clearance<.6-1e-7 or np.max(np.abs(np.degrees(m.u-previous)))>m.config['max_joint_speed_deg_s']*dt+.0001:
            raise ValueError('回放超出间隙或速度限制')
        if m.middle_unavailable and not np.allclose(m.u[g.cols['middle']],m.held_middle):
            raise ValueError('停用的中指出现运动')
        blocked+=int(m.blocked)
        for d in FINGERS:
            if d=='middle' and m.middle_unavailable:continue
            v=np.diff(m.chains[d],axis=0)@m.display_rotation.T
            v/=np.linalg.norm(v,axis=1,keepdims=True)
            shape.extend(np.degrees(np.arccos(np.clip((g.bone_directions(d)*v).sum(1),-1,1))))
        if f:gaps.append(g.gaps(f)['pad_mm'])
    result=dict(frames=len(frames),mean_shape_deg=float(np.mean(shape)),min_clearance_mm=min(clear),blocked_frames=blocked)
    if f:
        source=take_gaps(frames,f)
        mask=source<=np.percentile(source,15)+3.
        result.update(hold_frames=int(mask.sum()),source_tip_mm=float(np.median(source[mask])),
                      hold_pad_mm=float(np.median(np.array(gaps)[mask])),
                      hold_pad_p90_mm=float(np.percentile(np.array(gaps)[mask],90)),
                      hold_under_3mm_fraction=float(np.mean(np.array(gaps)[mask]<=3)),
                      released_pad_mm=float(np.median(gaps[-8:])))
    return result


def fit_round(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    base_path=folder/'baseline_profile.json'
    record=json.loads(base_path.read_text(encoding='utf8'))
    mirrored=record.get('input_hand')=='left'
    if mirrored!=(manifest.get('input_hand')=='left'):
        raise ValueError('对指档案与本轮手别不一致')
    expected=[f for f in FINGERS[1:] if not (f=='middle' and record.get('middle_unavailable'))]
    for split in ('train','validation'):
        if [e['stage'] for e in manifest[split]]!=expected:
            raise ValueError('本轮逐指采集不完整')
    g=NativeHand(ROOT/'assets/urdf_a')
    fit_holds(g,Follower(g,base_path),record,folder,manifest)
    candidate=folder/'profile.json';write(candidate,record)


def validate_round(folder):
    manifest=json.loads((folder/'manifest.json').read_bytes())
    base_path=folder/'baseline_profile.json'
    candidate=folder/'profile.json'
    record=json.loads(candidate.read_bytes())
    mirrored=record.get('input_hand')=='left'
    g=NativeHand(ROOT/'assets/urdf_a')
    report=dict(validation={},previous_natural={},accepted=False);issues=[]
    for entry in manifest['validation']:
        f=entry['stage'];rows=read_take(folder,entry);check_take(rows,f,'validation',manifest.get('unavailable_fingers',()))
        if mirrored:validate_mirrored_take(rows)
        a,b=[replay(g,path,rows,f) for path in (base_path,candidate)]
        report['validation'][f]=dict(previous=a,refitted=b)
        gap_worse=(spacing_error(b)>spacing_error(a)+.5 if mirrored else b['hold_pad_mm']>a['hold_pad_mm']+1.5)
        if (gap_worse or b['released_pad_mm']<b['hold_pad_mm']+8 or b['blocked_frames']>a['blocked_frames']
                or b['mean_shape_deg']>a['mean_shape_deg']+2.):
            issues.append(f+' 对指间隙、松开或手形回归')
    gain=np.mean([v['previous']['hold_pad_mm']-v['refitted']['hold_pad_mm'] for v in report['validation'].values()])
    if mirrored:
        errors=[spacing_error(v['refitted']) for v in report['validation'].values()]
        old_errors=[spacing_error(v['previous']) for v in report['validation'].values()]
        report['mean_spacing_error_mm']=dict(previous=np.mean(old_errors),refitted=np.mean(errors))
        if np.mean(errors)>.5 and np.mean(errors)>=np.mean(old_errors):
            issues.append('本轮对指与手套留缝的匹配误差未改善')
    elif gain<.3:issues.append('本轮平均指腹距离未改善')
    earlier=Path(record['supplement']) if record.get('supplement') else None
    if earlier and earlier.is_file():
        previous_manifest=json.loads(earlier.read_text(encoding='utf8'))
        missing_middle='middle' in previous_manifest.get('unavailable_fingers',[])
        for e in previous_manifest['validation']:
            if mirrored and previous_manifest.get('input_hand')=='left':
                from tools.validate_left_natural import replay as replay_spacing
                rows=read_take(earlier.parent,e)
                validate_mirrored_take(rows)
                a,b=[replay_spacing(g,p,rows[::3],previous_manifest.get('unavailable_fingers',[])) for p in (base_path,candidate)]
                report['previous_natural'][e['stage']]=dict(previous=a,refitted=b)
                for key,tolerance in [('mean_direction_deg',2.),('blocked_frames',0),('false_closures',0),('near_error_mm',1.),('contact_pad_mm',1.)]:
                    if b[key] is not None and b[key]>a[key]+tolerance:
                        issues.append(f"已有左手动作 {e['stage']} 的 {key} 退步")
                continue
            rows=read_take(earlier.parent,e);a,b=[replay(g,path,rows,middle_unavailable=missing_middle) for path in (base_path,candidate)]
            report['previous_natural'][e['stage']]=dict(previous=a,refitted=b)
            if b['mean_shape_deg']>a['mean_shape_deg']+2.:issues.append('已有自然动作误差增加超过 2°')
    report.update(average_pad_gain_mm=gain,accepted=not issues,issues=issues)
    write(folder/'validation.json',report)
    if issues:raise ValueError('；'.join(issues))
    return record


def fit_and_validate(folder):
    fit_round(folder)
    return validate_round(folder)
