"""Compare the personal coupled mapping with URDF-A using the original native geometry.

Run offline. All pad coordinates are nominal mesh samples; no hardware interface.
"""
import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_kinematics import FINGERS, HandGeometry, serializable
from g7_official_udp import official_config
from g7_teleop import load_g7_config
from calibrated_angle_mapping import CalibratedAngleRetargeter


from urdf_a_model import NativeHand, vec


def rigid_fit(a, b):
    u, _, vt = np.linalg.svd((a-a.mean(0)).T @ (b-b.mean(0)))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1., 1., sign]) @ u.T


def compare_axes(old, new):
    old.assign(old.data, old.zero)
    new.assign(np.zeros(17))
    def old_joint(f):
        return old.model.joint(old.cfg['fingers'][f]['columns'][0][0][0]).id
    x = np.array([old.data.xanchor[old_joint(f)] for f in FINGERS[1:]])
    y = np.array([new.data.xanchor[new.root_joints[f]] for f in FINGERS[1:]])
    rot = rigid_fit(x,y)
    signs, angles = [], []
    for i in range(17):
        # World angular velocity of the distal phalanx from an infinitesimal
        # independent-coordinate increment, with all upstream axes expressed
        # in the aligned palms. Includes the two DIP rows in the PIP column.
        def axis(model, data, c):
            v = (c[model.jnt_qposadr,i,None]*data.xaxis).sum(0)
            return v/np.linalg.norm(v)
        dot = np.clip((rot@axis(old.model,old.data,old.C))@axis(new.model,new.data,new.C),-1.,1.)
        signs.append(1 if dot>=0 else -1)
        angles.append(np.degrees(np.arccos(abs(dot))))
    err = (x-x.mean(0))@rot.T - (y-y.mean(0))
    return np.array(signs),angles,np.sqrt(np.mean(np.sum(err**2,axis=1)))*1000,rot


def fit_zero_offsets(old, new, signs, rot):
    # Align the *directions* of axes using each digit's upstream joint;
    # exclude descending flexion axes from the set of adjustable offsets.
    # This is a cross-model neutral alignment, not a measured human neutral.
    independent = np.array([1,5,8,11,14])
    def axes(model, data, C):
        j = (data.xaxis.T @ C[model.jnt_qposadr]).T
        return j / np.linalg.norm(j,axis=1,keepdims=True)
    old.assign(old.data, old.zero)
    target = (axes(old.model,old.data,old.C) @ rot.T) * signs[:,None]
    def residual(x):
        u = np.zeros(17);u[independent]=x
        new.assign(u)
        return (axes(new.model,new.data,new.C)-target).ravel()
    fit = least_squares(residual,np.zeros(len(independent)),bounds=(new.lo[independent],new.hi[independent]))
    bias = np.zeros(17);bias[independent]=fit.x
    deviations = 2*np.arcsin(np.clip(np.linalg.norm(residual(fit.x).reshape(17,3),axis=1)/2,0,1))
    return bias,np.degrees(deviations)


def verify_fk(package, new):
    """Compare the imported body frames with direct URDF origin * hinge FK."""
    path, = (package/'urdf').glob('*.urdf')
    joints = ET.parse(path).getroot().findall('joint')
    edges = {}
    for j in joints:
        origin = j.find('origin')
        parent = j.find('parent').get('link')
        edges.setdefault(parent, []).append((j.find('child').get('link'),new.model.joint(j.get('name')).qposadr[0],
            vec(origin,'xyz','0 0 0'),Rotation.from_euler('xyz',vec(origin,'rpy','0 0 0')),vec(j.find('axis'),'xyz','1 0 0')))
    worst_pos = worst_angle = 0.
    samples = [np.zeros(17),new.lo,new.hi,*np.random.default_rng(2707).uniform(new.lo,new.hi,(12,17))]
    for u in samples:
        new.assign(u)
        pending=[('base',np.zeros(3),Rotation.identity())]
        while pending:
            parent,p,R=pending.pop()
            for child,address,offset,fixed,axis in edges.get(parent,[]):
                child_p=p+R.apply(offset)
                child_R=R*fixed*Rotation.from_rotvec(axis*new.data.qpos[address])
                b=new.model.body(child).id
                worst_pos=max(worst_pos,1000*np.linalg.norm(child_p-new.data.xpos[b]))
                actual=Rotation.from_matrix(new.data.xmat[b].reshape(3,3))
                worst_angle=max(worst_angle,np.degrees((actual.inv()*child_R).magnitude()))
                pending.append((child,child_p,child_R))
    if worst_pos>1e-5 or worst_angle>1e-5:
        raise ValueError('Native FK disagrees with the supplied URDF')
    new.assign(np.zeros(17))
    return dict(poses=len(samples),max_body_position_mm=worst_pos,max_body_rotation_deg=worst_angle)


def audit(package, profile):
    old = HandGeometry('simulation_straight'); new = NativeHand(package)
    ref = np.radians(old.simulation_profile['initial_coordinates_deg'])
    mapper = CalibratedAngleRetargeter(old,ref,'right',official_config(load_g7_config()),profile)
    signs,angles,rms,rot = compare_axes(old,new)
    bias,aligned_angles = fit_zero_offsets(old,new,signs,rot)
    new.assign(np.zeros(17))
    report = dict(urdf_sha256=new.sha, profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                  fk_crosscheck=verify_fk(package,new),native_count=new.model.nq, reduction_rank=np.linalg.matrix_rank(new.C),
                  wrist_at_zero=True, root_alignment_rms_mm=rms, zero_clearance_mm=new.clearance(),
                  coordinates={label:dict(old_to_A_sign=int(s),axis_misalignment_deg=a,
                                          estimated_offset_deg=np.degrees(b),after_offset_deg=d,
                                          lower_deg=np.degrees(l),upper_deg=np.degrees(h))
                               for label,s,a,b,d,l,h in zip(old.labels,signs,angles,bias,aligned_angles,new.lo,new.hi)},
                  axis_spacings_mm={f+'_'+s:1000*np.linalg.norm(np.diff(new.data.xanchor[js],axis=0))
                                    for (f,s),js in new.pair_axes.items()},
                  pad_candidates_local_m={f:p for f,(p,_) in new.pad.items()},stages={})
    for f in FINGERS[1:]:
        def get_pose(split,offset):
            rows=(profile.parent/(split+'_'+f+'.jsonl')).read_text().splitlines()
            mapper.previous=ref.copy()
            u=mapper.solve_frame(json.loads(rows[len(rows)//2]))['goal_rad']
            return np.clip(signs*u+offset,new.lo,new.hi)
        train=get_pose('train',bias)
        ids=np.r_[0:5,new.cols[f]]
        def with_pair(x):
            u=train.copy();u[ids]=x;return u
        def objective(x):
            new.assign(with_pair(x))
            a,n=new.pad_pose('thumb');b,m=new.pad_pose(f)
            return np.sum((a-b+1.*n)**2)+16*np.sum((n+m)**2)+np.sum((np.array([5]*5+[10]*3)*(x-train[ids]))**2)
        def clear(x):
            u=with_pair(x);values=[]
            for t in (.5,.8,1.):
                new.assign(t*u);values.append(new.clearance()-.95)
            return values
        fit=minimize(objective,train[ids],method='SLSQP',bounds=list(zip(new.lo[ids],new.hi[ids])),
                     constraints=[dict(type='ineq',fun=clear)],options=dict(maxiter=100,ftol=1e-6))
        u=get_pose('validation',0.)
        new.assign(u)
        before=new.gaps(f);before['approach_min_mm']=new.approach(u)
        u=get_pose('validation',bias)
        new.assign(u)
        aligned=new.gaps(f);aligned['approach_min_mm']=new.approach(u)
        trial=u.copy()
        trial[ids]=np.clip(fit.x+.15*(u[ids]-train[ids]),new.lo[ids],new.hi[ids])
        new.assign(trial)
        after=new.gaps(f);after['approach_min_mm']=new.approach(trial)
        samples=[]
        lines=(profile.parent/('validation_'+f+'.jsonl')).read_text().splitlines()
        mapper.previous=ref.copy()
        for line in lines[::4]:
            frame=json.loads(line)
            solved=mapper.solve_frame(frame)
            old_target=solved['goal_rad']
            directions=mapper.contact_model.targets(frame)[0]
            u=np.clip(signs*old_target+bias,new.lo,new.hi)
            v=u.copy()
            v[ids]=np.clip(fit.x+.15*(u[ids]-train[ids]),new.lo[ids],new.hi[ids])
            pair_stats=[]
            for candidate in (u,v):
                new.assign(candidate)
                gap=new.gaps(f)
                cos=np.concatenate([np.sum(new.bone_directions(digit)*(directions[digit]@rot.T),axis=1)
                                    for digit in FINGERS])
                pair_stats.append([gap['pad_mm'],new.clearance(),np.mean(np.degrees(np.arccos(np.clip(cos,-1.,1.))))])
            samples.append(pair_stats)
        values=np.asarray(samples)
        held={k:dict(pad_median_mm=np.median(values[:,i,0]),pad_max_mm=values[:,i,0].max(),
                     min_thumb_clearance_mm=values[:,i,1].min(),mean_bone_error_deg=values[:,i,2].mean())
              for i,k in enumerate(('sign_and_offset','refit_trial'))}
        report['stages'][f]=dict(sign_only=before,sign_and_offset=aligned,refit_trial=after,optimizer_success=bool(fit.success),
                                 sampled_validation_frames=len(values),validation_hold=held,
                                 train_anchor_approach_mm=new.approach(with_pair(fit.x)))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path)
    parser.add_argument('profile',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=audit(args.package,args.profile)
    args.output.write_text(json.dumps(report,default=serializable,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


