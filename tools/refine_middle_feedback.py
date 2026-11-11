"""Offline fit of smooth, digit-local angle residuals to recorded feedback."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from hand_kinematics import ROOT, FINGERS
from tools.calibrate_angle_map import write
from tools.fit_pinch_round import replay, take_gaps
from tools.check_middle_refinement import trace
from urdf_a_model import NativeHand
from urdf_a_follow import Follower


def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]


def dense(rows):
    """Interpolate sparse diagnostic snapshots to 120 Hz for shared replay."""
    stamps=np.array([r['timestamp_ns'] for r in rows],dtype=np.int64)
    t=(stamps-stamps[0])*1e-9
    times=np.arange(0,t[-1],1/120)
    q=np.asarray([r['quaternions_wxyz'] for r in rows])
    pos=np.asarray([r['positions_m'] for r in rows])
    quats=np.stack([Slerp(t,Rotation.from_quat(q[:,j][:,[1,2,3,0]]))(times).as_quat()[:,[3,0,1,2]] for j in range(21)],axis=1)
    positions=np.stack([np.interp(times,t,pos[:,j,k]) for j in range(21) for k in range(3)],axis=1).reshape(-1,21,3)
    return [dict(rows[0],timestamp_ns=int(stamps[0]+ts*1e9),quaternions_wxyz=qq.tolist(),positions_m=pp.tolist())
            for ts,qq,pp in zip(times,quats,positions)]


def fit(folder):
    source=folder/'baseline_profile.json'
    record=json.loads(source.read_text(encoding='utf8'));g=NativeHand(ROOT/'assets/urdf_a')
    m=Follower(g,source);baseline=Follower(g,source)
    rows=read(folder/'motion.jsonl')
    held=[r for r,gap in zip(rows,take_gaps(rows,'middle')) if gap<8]
    if len(held)<12:raise ValueError('反馈中有效拇中接触姿态不足')
    training=held[::max(1,len(held)//12)]
    ids=np.r_[g.cols['thumb'],g.cols['middle']]
    correction={}
    for digit in ('thumb','middle'):
        x=np.array([m.source.read(r)[0][digit]-m.blocks[digit][0] for r in training])
        correction[digit]=(np.mean(x,axis=0)[None,:],np.zeros((1,len(g.cols[digit]))),np.radians(12))
    if m.local_corrections:raise ValueError('已有局部修正，请使用该轮原始基线，避免重复叠加')
    m.local_corrections=correction
    actual=[]
    for frame in training:
        goal=baseline.command(frame).copy()
        for _ in range(100):baseline.advance(.025)
        actual.append((baseline.u.copy(),goal))
    natural=Path(record['supplement']).parent
    protect=[]
    for name in ('train_rest_again','train_middle_slow','train_index_approach','train_ring_approach','train_pinky_approach'):
        seq=read(natural/(name+'.jsonl'))
        protect.extend(seq[::max(1,len(seq)//10)])
    release=read(folder/'release.jsonl');protect.extend(release[::max(1,len(release)//10)])
    old=[baseline.command(r).copy() for r in protect]
    def residual(x):
        correction['thumb'][1][0]=x[:5];correction['middle'][1][0]=x[5:]
        result=[]
        for frame,(achieved,goal) in zip(training,actual):
            g.assign(np.clip(achieved+m.command(frame)-goal,g.lo,g.hi))
            a,n=g.pad_pose('thumb');b,_=g.pad_pose('middle')
            result.extend((a-b+n)/np.sqrt(len(training)))
            for digit in ('thumb','middle'):
                v=np.diff(m.chains[digit],axis=0)@m.display_rotation.T;v/=np.linalg.norm(v,axis=1,keepdims=True)
                result.extend((30*(g.bone_directions(digit)-v)/np.sqrt(len(training))).ravel())
            result.append(18*max(0,.85-g.clearance())/np.sqrt(len(training)))
        for frame,goal in zip(protect,old):
            result.extend(35*(m.command(frame)[ids]-goal[ids])/np.sqrt(len(protect)))
        result.extend(8*x)
        return np.asarray(result)
    budget=np.radians(8)
    opt=least_squares(residual,np.zeros(8),bounds=(-budget,budget),diff_step=1e-4,
                      max_nfev=180,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    if not opt.success:raise ValueError(opt.message)
    correction['thumb'][1][0]=opt.x[:5];correction['middle'][1][0]=opt.x[5:]
    record['local_angular_corrections']={f:dict(centers=c.tolist(),coefficients=v.tolist(),width=w) for f,(c,v,w) in correction.items()}
    record['feedback_refinement']=dict(folder=str(folder.resolve()),digits=['thumb','middle'],
        method='smooth_own_angle_residual',coefficients_deg=np.degrees(opt.x).tolist(),width_deg=12,
        runtime_contact_trigger=False,evaluation='recorded_data_regression')
    path=folder/'profile.json';write(path,record)
    print(record['feedback_refinement'],flush=True)
    return g,source,path,natural


def validate(folder,g,source,path,natural):
    report=dict(feedback={},natural={},opposition={},issues=[],accepted=False,evaluation='recorded_data_regression')
    issues=report['issues']
    for name in ('motion','release'):
        rows=dense(read(folder/(name+'.jsonl')))
        a,b=[replay(g,p,rows,'middle') for p in (source,path)]
        report['feedback'][name]=dict(previous=a,refined=b)
        print(name,report['feedback'][name],flush=True)
        if b['mean_shape_deg']>a['mean_shape_deg']+1 or b['blocked_frames']>a['blocked_frames']:issues.append(name+' shape/continuity regression')
    a,b=report['feedback']['motion'].values()
    if b['hold_pad_mm']>a['hold_pad_mm']-3:issues.append('feedback improvement below 3 mm')
    for name in ('train_middle_slow','train_middle_approach','validation_near_pass','validation_touch_release'):
        rows=read(natural/(name+'.jsonl'))
        a,b=[replay(g,p,rows,'middle' if name=='train_middle_approach' else None) for p in (source,path)]
        ca,cb=[trace(g,p,rows[::3]) for p in (source,path)]
        report['natural'][name]=dict(previous=a,refined=b,continuity_previous=ca,continuity_refined=cb)
        if b['mean_shape_deg']>a['mean_shape_deg']+.7 or b['blocked_frames']>a['blocked_frames']:issues.append(name+' shape/continuity regression')
        if cb['max_target_step_deg']>ca['max_target_step_deg']*1.15+.2:issues.append(name+' target step regression')
        if cb['p95_acceleration_deg_s2']>ca['p95_acceleration_deg_s2']*1.2+1:issues.append(name+' acceleration regression')
        if name=='train_middle_approach' and b['hold_pad_mm']>a['hold_pad_mm']+1:issues.append(name+' gap regression')
    opposition=ROOT/'calibration/opposition_round_20261110_174958_468941'
    for finger in FINGERS[1:]:
        rows=read(opposition/('validation_'+finger+'.jsonl'))
        a,b=[replay(g,p,rows,finger) for p in (source,path)]
        report['opposition'][finger]=dict(previous=a,refined=b)
        if b['mean_shape_deg']>a['mean_shape_deg']+.7 or b['blocked_frames']>a['blocked_frames']:issues.append(finger+' shape/continuity regression')
        if b['hold_pad_mm']>a['hold_pad_mm']+1:issues.append(finger+' gap regression')
        if b['released_pad_mm']<max(b['hold_pad_mm']+8,a['released_pad_mm']-3):issues.append(finger+' release regression')
    report['accepted']=not issues;write(folder/'validation.json',report)
    print('accepted',report['accepted'],'issues',issues,flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    g,source,path,natural=fit(args.folder)
    validate(args.folder,g,source,path,natural)
