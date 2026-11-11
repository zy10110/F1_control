"""Continuity regression checks for the recorded-data middle refinement."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from hand_kinematics import ROOT, FINGERS
from tools.calibrate_angle_map import write
from urdf_a_model import NativeHand
from urdf_a_follow import Follower


def trace(g,path,frames,dt=.025):
    m=Follower(g,path);m.command(frames[0])
    for _ in range(100):m.advance(dt)
    commands=[];actual=[];speed=[];clear=[];blocked=0
    for frame in frames:
        commands.append(m.command(frame).copy())
        previous=m.u.copy();m.advance(dt)
        speed.append(np.degrees((m.u-previous)/dt))
        actual.append(m.u.copy());clear.append(g.clearance());blocked+=int(m.blocked)
        assert np.all(m.u>=g.lo-1e-10) and np.all(m.u<=g.hi+1e-10)
        np.testing.assert_allclose(g.data.qpos,g.C@m.u,atol=1e-12)
    commands=np.asarray(commands);actual=np.asarray(actual);speed=np.asarray(speed)
    reverse=np.array([m.command(f).copy() for f in reversed(frames)])[::-1]
    np.testing.assert_allclose(commands,reverse,atol=1e-12)
    report=dict(frames=len(frames),blocked_frames=blocked,min_clearance_mm=min(clear),
        max_speed_deg_s=float(np.max(abs(speed))),
        p95_acceleration_deg_s2=float(np.percentile(abs(np.diff(speed,axis=0)/dt),95)),
        max_target_step_deg=float(np.max(abs(np.degrees(np.diff(commands,axis=0))))),
        final_tracking_error_deg=float(np.max(abs(np.degrees(actual[-1]-commands[-1])))),
        angle_map_history_independent=True)
    assert report['min_clearance_mm']>=.6-1e-7
    assert report['max_speed_deg_s']<=m.config['max_joint_speed_deg_s']+1e-6
    return report


def run(folder,candidate,source=None):
    source=source or folder/'profile.json'
    old=json.loads(source.read_text(encoding='utf8'))
    new=json.loads(candidate.read_text(encoding='utf8'))
    assert old.get('joint_angle_context')==new.get('joint_angle_context'), 'Use the whole-hand context replay audit for context changes'
    for key in ('blocks','middle_widths','geometry_id','neutral','wrist_to_model'):
        assert old[key]==new[key],key
    for finger in FINGERS:
        if finger!='middle':assert old['angular_corrections'][finger]==new['angular_corrections'][finger]
    g=NativeHand(ROOT/'assets/urdf_a')
    base=Path(json.loads((folder/'manifest.json').read_text(encoding='utf8'))['base_session'])
    opened=json.loads((base/'validation_open.jsonl').read_text().splitlines()[40])
    closed=json.loads((base/'validation_fist.jsonl').read_text().splitlines()[40])
    q0=np.asarray(opened['quaternions_wxyz']);q1=np.asarray(closed['quaternions_wxyz'])
    # Captures can have different wrist orientations. Put the closed chain in
    # the open wrist frame before interpolating an isolated finger movement.
    wrist_alignment=Rotation.from_quat(q0[0,[1,2,3,0]])*Rotation.from_quat(q1[0,[1,2,3,0]]).inv()
    q1=(wrist_alignment*Rotation.from_quat(q1[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
    t=np.r_[np.linspace(0,1,161),np.ones(40),np.linspace(1,0,161),np.zeros(40)]
    interpolated=[]
    r0=Rotation.from_quat(q0[:,[1,2,3,0]])
    r1=Rotation.from_quat(q1[:,[1,2,3,0]])
    # Interpolate each parent-relative joint. Independent world-space SLERP
    # can make a distal joint turn the long way through 180 degrees even when
    # both recorded end poses are ordinary flexion.
    parent_world=Rotation.from_quat(np.tile(q0[0,[1,2,3,0]],(len(t),1)))
    for j in range(9,13):
        parent=0 if j==9 else j-1
        local=Rotation.concatenate([r0[parent].inv()*r0[j],r1[parent].inv()*r1[j]])
        parent_world=parent_world*Slerp([0,1],local)(t)
        interpolated.append(parent_world.as_quat()[:,[3,0,1,2]])
    frames=[]
    for i in range(len(t)):
        frame=copy.deepcopy(opened)
        for j in range(9,13):frame['quaternions_wxyz'][j]=interpolated[j-9][i].tolist()
        frames.append(frame)
    report=dict(only_middle_mapping_changed=True,unchanged_kinematics=True,sequences={})
    sequences={'isolated_middle_slow':frames}
    for f in FINGERS[1:]:
        sequences[f+'_release']=[json.loads(s) for s in (folder/('validation_'+f+'.jsonl')).read_text().splitlines()][::3]
    if old.get('supplement'):
        supplement=Path(old['supplement'])
        supplement_manifest=json.loads(supplement.read_text(encoding='utf8'))
        if 'middle' not in supplement_manifest.get('unavailable_fingers',[]):
            for name in ('train_middle_slow','train_middle_approach','validation_near_pass','validation_touch_release'):
                path=supplement.parent/(name+'.jsonl')
                if path.exists():sequences[name]=[json.loads(s) for s in path.read_text().splitlines()][::3]
    issues=[]
    for name,frames in sequences.items():
        before,after=[trace(g,p,frames) for p in (source,candidate)]
        report['sequences'][name]=dict(previous=before,refined=after)
        if after['blocked_frames']>before['blocked_frames']:issues.append(name+' new blocked frames')
        if after['max_target_step_deg']>before['max_target_step_deg']*1.1+.1:issues.append(name+' target jump regression')
        if name=='isolated_middle_slow' and after['max_target_step_deg']>3:issues.append(name+' discontinuous target')
        if after['p95_acceleration_deg_s2']>before['p95_acceleration_deg_s2']*1.15+1:issues.append(name+' acceleration regression')
        if after['final_tracking_error_deg']>before['final_tracking_error_deg']+.5:issues.append(name+' release tracking regression')
        print(name,json.dumps(report['sequences'][name]),flush=True)
    report.update(accepted=not issues,issues=issues)
    write(candidate.parent/'continuity.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('session',type=Path);parser.add_argument('profile',type=Path)
    parser.add_argument('--base-profile',type=Path)
    args=parser.parse_args()
    result=run(args.session,args.profile,args.base_profile)
    print('accepted:',result['accepted'],'issues:',result['issues'],flush=True)
