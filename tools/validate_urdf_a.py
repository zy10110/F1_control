"""Run the exact live retargeter, filter and swept guard on independent validation data."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hand_kinematics import FINGERS
from urdf_a_model import NativeHand
from urdf_a_follow import Follower
from g7_joint_mapping import rotations
from g7_teleop import BASE
from tools.calibrate_angle_map import write


def frames(folder,stage):
    return [json.loads(line) for line in (folder/('validation_'+stage+'.jsonl')).read_text().splitlines()]


def interpolate(a,b,n,only_middle=False):
    # Interpolate parent-relative rotations and rebuild each wrist-relative chain.
    # Isolating middle keeps the other four digits at their starting pose.
    ra,rb=rotations(a),rotations(b)
    parents=np.array([0]+[0 if (i-1)%4==0 else i-1 for i in range(1,21)])
    la,lb=ra[parents].inv()*ra,rb[parents].inv()*rb
    delta=(la.inv()*lb).as_rotvec()
    if only_middle:
        keep=delta[9:13].copy();delta[:]=0.;delta[9:13]=keep
    for t in np.linspace(0,1,n+1)[1:]:
        locals_=la*Rotation.from_rotvec(t*delta)
        result=[ra[0]]
        for i in range(1,21):
            result.append(result[parents[i]]*locals_[i])
        yield dict(hand=2,quaternions_wxyz=Rotation.concatenate(result).as_quat()[:,[3,0,1,2]].tolist())


def run(package,folder,profile=None,supplement=None):
    if supplement and json.loads(supplement.read_bytes()).get('input_hand')=='left':
        from tools.validate_left_natural import run as validate_left
        return validate_left(package,profile,supplement)
    g=NativeHand(package);profile=profile or folder/'profile_urdf_a_natural.json'
    poses={}
    for f in ('open','fist',*FINGERS[1:]):
        take=frames(folder,f)
        poses[f]=take[len(take)//2]
    dt=1/40
    def tick(m,frame,interval):
        m.command(frame);before=m.u.copy();u=m.advance(interval)
        clearance=g.clearance()
        speed=np.max(np.abs(np.degrees(u-before)))/interval
        if not m.sweep.safe(m.u) or speed>m.config['max_joint_speed_deg_s']+.001:
            raise ValueError('实际轨迹越过间隙或速度约束')
        return dict(clearance=clearance,speed=speed,blocked=int(m.blocked))
    def warm(m,fr):
        for _ in range(30):tick(m,fr,dt)
    def stats(rows):
        return dict(frames=len(rows),min_clearance_mm=min(r['clearance'] for r in rows),
                    max_speed_deg_s=max(r['speed'] for r in rows),blocked_frames=sum(r['blocked'] for r in rows))
    report={'geometry_id':g.geometry_id,'validation_takes':{},'sequences':{}}
    for path in sorted(folder.glob('validation_*.jsonl')):
        m=Follower(g,profile,middle_unavailable=False);stage=path.stem[11:];rows=[];gaps=[];last=None;shape=[]
        for frame in frames(folder,stage)[::3]:
            stamp=frame['timestamp_ns'];interval=dt if last is None else (stamp-last)*1e-9;last=stamp
            rows.append(tick(m,frame,interval))
            if stage in FINGERS[1:]:gaps.append(g.gaps(stage)['pad_mm'])
            for f in FINGERS:
                d=np.diff(m.chains[f],axis=0)@m.display_rotation.T
                d/=np.linalg.norm(d,axis=1,keepdims=True)
                shape.append(np.degrees(np.arccos(np.clip((d*g.bone_directions(f)).sum(1),-1,1))).mean())
        result=stats(rows);result['mean_bone_error_deg']=np.mean(shape)
        if gaps:result['last_half_pad_median_mm']=np.median(gaps[len(gaps)//2:])
        report['validation_takes'][stage]=result
    m=Follower(g,profile,middle_unavailable=False);warm(m,poses['open'])
    close=list(interpolate(poses['open'],poses['fist'],120,True))
    steps=close+list(reversed(close))+[poses['open']]
    rows=[];angles=[];goals=[]
    for fr in steps:
        rows.append(tick(m,fr,dt));angles.append(np.degrees(m.u[9:11]));goals.append(np.degrees(m.goal[9:11]))
    actual=np.array(angles);dg=np.diff(goals,axis=0);da=np.diff(actual,axis=0)
    stalled=(abs(dg).max(1)>.12)&(abs(da).max(1)<.04)
    longest=max((len(group) for group in ''.join('1' if s else ' ' for s in stalled).split()),default=0)
    report['middle_slow']=dict(stats(rows),actual_range_deg=np.ptp(actual,axis=0),longest_stall_ms=longest*dt*1000,
                               peak_middle_step_deg=abs(da).max(),stalled_frames=int(stalled.sum()),
                               reversal_count=[int(np.count_nonzero(np.diff(np.sign(v[abs(v)>.04])))) for v in da.T])
    for name,route in [('release',['index','open','middle','open','ring','open','pinky','open']),
                       ('switch',['index','middle','ring','pinky','open'])]:
        m=Follower(g,profile,middle_unavailable=False);warm(m,poses['open']);old='open';entries=[];rows=[]
        for f in route:
            moved=list(interpolate(poses[old],poses[f],28));hold=[poses[f]]*18
            gap=[];steps_out=[];release_trace=[]
            for fr in moved+hold:
                before=m.u.copy();rows.append(tick(m,fr,dt))
                steps_out.append(float(np.max(np.abs(np.degrees(m.u-before)))))
                if f!='open':gap.append(g.gaps(f)['pad_mm'])
                if old!='open':release_trace.append(g.gaps(old)['pad_mm'])
            e=dict(stage=f,settled_tracking_deg=np.max(np.abs(np.degrees(m.u-m.goal))),
                   peak_step_deg=max(steps_out),hold_last_step_deg=steps_out[-1])
            if f!='open': e['settled_pad_mm']=gap[-1]
            else: e['released_pad_mm']=g.gaps(old)['pad_mm']
            if release_trace:
                reached=np.flatnonzero(np.array(release_trace)>10.)
                e['previous_pair_above_10mm_ms']=(int(reached[0])+1)*dt*1000 if len(reached) else None
            entries.append(e);old=f
        report['sequences'][name]=dict(stats(rows),stages=entries)
    # Fixed long fingers, a thumb sweeping from one side of the recorded
    # middle-pinch posture to the other. Construct in wrist coordinates.
    m=Follower(g,profile,middle_unavailable=False)
    neutral=poses['middle']
    q=rotations(neutral)
    wrist=q[0]
    sweep_frames=[]
    for angle in np.linspace(-32.,32.,129):
        twist=wrist*Rotation.from_rotvec(np.radians([0.,0.,angle]))*wrist.inv()
        changed=q.as_quat()
        changed[1:5]=(twist*q[1:5]).as_quat()
        sweep_frames.append(dict(hand=2,quaternions_wxyz=changed[:,[3,0,1,2]].tolist()))
    def distances(m):
        return np.linalg.norm(m.chains['thumb'][-1]-m.chains['middle'][-1]),g.gaps('middle')['pad_mm']
    warm(m,sweep_frames[0]);targets=[];executed=[];glove=[];robot=[];rows=[]
    for fr in sweep_frames:
        rows.append(tick(m,fr,dt))
        targets.append(np.degrees(m.goal).copy())
        executed.append(np.degrees(m.u).copy())
        x,y=distances(m);glove.append(x);robot.append(y)
    stop_frame=int(np.argmin(abs(np.array(glove[:np.argmin(glove)])-12.)))
    warm(m,sweep_frames[stop_frame]);start=m.u.copy();before=distances(m)
    for _ in range(30):tick(m,sweep_frames[stop_frame],dt)
    after=distances(m)
    report['thumb_brush']=dict(stats(rows),source_min_mm=min(glove),source_endpoints_mm=[glove[0],glove[-1]],
        fixed_long_finger_goal_span_deg=np.ptp(targets,axis=0)[5:].max(),thumb_goal_span_deg=np.ptp(targets,axis=0)[:5].max(),
        fixed_long_finger_executed_span_deg=np.ptp(executed,axis=0)[5:].max(),
        f1_pad_at_start_closest_input_end_mm=[robot[0],robot[np.argmin(glove)],robot[-1]],
        stop_short_source_mm=after[0],stop_short_f1_mm=after[1],hold_drift_deg=np.max(abs(np.degrees(m.u-start))),
        hold_gap_change_mm=after[1]-before[1])
    if (min(glove)>5. or min(glove[0],glove[-1])<12. or
            report['thumb_brush']['fixed_long_finger_goal_span_deg']>1e-6 or
            report['thumb_brush']['thumb_goal_span_deg']<5. or abs(after[1]-before[1])>.25):
        raise ValueError('近距离擦过或留缝保持时的独立跟随验证失败')
    if len(report['validation_takes'])!=15:
        raise ValueError('验证动作记录不完整')
    middle=report['middle_slow']
    if (middle['blocked_frames'] or middle['longest_stall_ms']>50 or
            min(middle['actual_range_deg'])<40 or middle['reversal_count']!=[1,1]):
        raise ValueError('中指慢弯伸存在停顿、额外反向或行程不足')
    for route in report['sequences'].values():
        for e in route['stages']:
            f=e['stage']
            if f=='open':
                if e['released_pad_mm']<20:
                    raise ValueError('松开后仍锁在对指位置')
            if e['hold_last_step_deg']>.15:
                raise ValueError('保持阶段未收敛')
    if supplement is not None:
        report['supplement_validation']={}
        entries=json.loads(supplement.read_text(encoding='utf8'))['validation']
        if {e['stage'] for e in entries}!={'near_pass','touch_release'}:
            raise ValueError('补录验证动作不全')
        missing=json.loads(supplement.read_text(encoding='utf8')).get('unavailable_fingers',[])
        for entry in entries:
            lines=(supplement.parent/entry['file']).read_text(encoding='utf8').splitlines()[::3]
            comparison={}
            for label,path in [('previous',supplement.parent/'baseline_profile.json'),('refitted',profile)]:
                m=Follower(g,path,middle_unavailable='middle' in missing);rows=[];errors=[];last=None
                for line in lines:
                    fr=json.loads(line);stamp=fr['timestamp_ns']
                    rows.append(tick(m,fr,dt if last is None else (stamp-last)*1e-9));last=stamp
                    for f in FINGERS:
                        if f in missing:continue
                        v=np.diff(m.chains[f],axis=0)@m.display_rotation.T
                        v/=np.linalg.norm(v,axis=1,keepdims=True)
                        errors.extend(np.degrees(np.arccos(np.clip((v*g.bone_directions(f)).sum(1),-1,1))))
                comparison[label]=dict(stats(rows),mean_direction_deg=float(np.mean(errors)))
            report['supplement_validation'][entry['stage']]=comparison
            if comparison['refitted']['mean_direction_deg']>comparison['previous']['mean_direction_deg']+2.:
                raise ValueError('独立补录动作的手形方向误差增加超过 2°')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('package','folder','output'):parser.add_argument(k,type=Path)
    args=parser.parse_args();write(args.output,run(args.package,args.folder))

