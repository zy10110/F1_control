"""F1 pinch alignment with legacy offsets or local paired-finger assistance.

Official glove calibration is unchanged. Distal surface points are nominal pad
landmarks, not physically measured contacts. Joint imitation remains in charge
outside the enabled pinch task.
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from hand_kinematics import ROOT, HandRuntime, serializable
from palm_calibration import PadGeometry, build_pad_candidates, write_json

PAIRS=('index','middle','ring','pinky')
BASE=dict(index=5,middle=9,ring=13,pinky=17)


def relative_rotations(frame):
    q=np.asarray(frame['quaternions_wxyz'],float)
    if q.shape!=(21,4) or not np.isfinite(q).all() or np.min(np.linalg.norm(q,axis=1))<1e-6:
        raise ValueError('对指标定需要21个有效骨段四元数')
    r=Rotation.from_quat(q[:,[1,2,3,0]])
    return r[0].inv()*r[1:]


class F1PinchAlignment:
    def __init__(self,geometry,path,hand,reference,open_frame=None):
        self.g=geometry;self.path=str(path)
        self.record=json.loads(Path(path).read_text(encoding='utf8'))
        r=self.record
        if r['hand']!=hand or r['mode']!=geometry.mode or r['source_sha256']!=geometry.cfg['source_sha256']:
            raise ValueError('F1 对指标定与手别、模型或模式不一致')
        if not np.allclose(r['reference_rad'],reference,atol=1e-10):
            raise ValueError('F1 对指标定初始姿态不一致')
        # Four-finger zero must be exactly the one used when fitting anchors.
        if open_frame is not None:
            d=relative_rotations(open_frame).inv()*relative_rotations(r['open_frame'])
            if np.max(d.magnitude())>np.radians(.05):
                raise ValueError('F1 对指样本对应另一份张开参考，请停用对指修正或恢复本轮参考')
        self.anchors=[]
        self.open_rotations=relative_rotations(r['open_frame'])
        self.pad=PadGeometry(geometry,r['f1_pads'])
        self.data=mujoco.MjData(geometry.model)
        for f in PAIRS:
            a=r['anchors'][f]
            delta=np.asarray(a['thumb_delta_rad'],float)
            if delta.shape!=(5,) or not np.isfinite(delta).all():raise ValueError('对指修正角无效')
            self.anchors.append((f,relative_rotations(a['frame']),delta))
        self.contact_assist=None
        if r.get('contact_assist',{}).get('enabled'):
            from g7_f1_contact_assist import ContactAssist
            self.contact_assist=ContactAssist(self)

    def apply(self,frame,desired,lower,upper):
        """Use legacy pose offsets or the enabled proximity-gated local task."""
        if self.contact_assist is not None:
            return self.contact_assist.apply(frame,desired,lower,upper)
        r=relative_rotations(frame);weights=[];scores=[]
        for f,anchor,delta in self.anchors:
            # Include every finger to distinguish contact gestures and avoid
            # moving the thumb merely because one flexion angle happens to match.
            degrees=np.degrees((anchor.inv()*r).magnitude())
            ids=np.r_[0:3,BASE[f]-1:BASE[f]+2]
            local=float(np.sqrt(np.mean(degrees[ids]**2)))
            other=float(np.sqrt(np.mean(degrees**2)))
            distance=np.sqrt(.8*local**2+.2*other**2)
            radius=float(self.record.get('support_radius_deg',35.))
            t=max(0.,1.-(distance/radius)**2)
            weights.append(t*t);scores.append(distance)
        weights=np.asarray(weights)
        # Exact-anchor interpolation: other anchors vanish when one distance
        # is zero; the support envelope still fades correction to zero outside.
        if np.min(scores)<1e-5:
            weights[:]=0.;weights[int(np.argmin(scores))]=1.
        else:
            inverse=weights/(np.asarray(scores)**4+1e-9)
            if inverse.sum()>0:weights=inverse/inverse.sum()*weights.max()
        opening=np.degrees((self.open_rotations.inv()*r).magnitude())
        open_distance=float(np.sqrt(np.mean(opening**2)))
        release=float(np.clip((open_distance-3.)/7.,0.,1.))
        weights*=release*release*(3.-2.*release)
        result=np.asarray(desired,float).copy()
        correction=sum((w*a[2] for w,a in zip(weights,self.anchors)),start=np.zeros(5))
        result[:5]=np.clip(result[:5]+correction,lower[:5],upper[:5])
        return result,dict(kind='four labelled F1 pinches; local thumb-only angle offsets',
            profile=self.path,weights=dict(zip(PAIRS,weights.tolist())),
            source_pose_distance_deg=dict(zip(PAIRS,scores)),
            correction_deg=np.degrees(result[:5]-desired[:5]).tolist(),
            four_finger_commands_unchanged=True,
            scope='nominal F1 pad landmarks; no measured physical contact guarantee')

    def gaps(self,u):
        self.g.assign(self.data,u)
        thumb=self.pad.pose(self.data,'thumb')['position_mm']
        return {f:float(np.linalg.norm(thumb-self.pad.pose(self.data,f)['position_mm'])) for f in PAIRS}


def fit_session(folder,reference_capture):
    from g7_joint_mapping import G7JointRetargeter
    from g7_teleop import load_g7_config
    from g7_official_udp import official_config
    folder=Path(folder);samples=json.loads((folder/'samples.json').read_text(encoding='utf8'))
    if any(f not in samples['samples'] for f in PAIRS):raise ValueError('四组对指尚未采齐')
    frames={f:samples['samples'][f]['frame'] for f in PAIRS}
    # Refuse indistinguishable labels instead of fitting four contradictory poses.
    for i,f in enumerate(PAIRS):
        for h in PAIRS[:i]:
            delta=(relative_rotations(frames[f]).inv()*relative_rotations(frames[h])).magnitude()
            if np.degrees(delta).max()<5.:raise ValueError(f'{f}/{h} 两组姿态无法区分，请只重采这两组')
    runtime=HandRuntime('simulation_straight');g=runtime.g
    config=official_config(dict(load_g7_config(),official_app_reference_capture=str(reference_capture)))
    config.pop('f1_pinch_profile',None)
    mapping=G7JointRetargeter(g,runtime.initial_u,samples['hand'],config)
    pads=build_pad_candidates(g,runtime.initial_u);pad=PadGeometry(g,pads)
    data=mujoco.MjData(g.model);guard=mapping.collision
    record=dict(schema_version=1,created_unix=time.time(),hand=samples['hand'],mode=g.mode,
        source_sha256=g.cfg['source_sha256'],reference_rad=runtime.initial_u,
        open_reference_capture=str(reference_capture),open_frame=mapping.open_frame,
        support_radius_deg=35.,f1_pads=pads,anchors={},hardware_enabled=False,
        scope='F1 visual pinch adjustment; original official profile and four-finger mapping unchanged')
    rng=np.random.default_rng(31103)
    pool=np.vstack((mapping.thumb_samples,rng.uniform(mapping.lower[:5],mapping.upper[:5],(350,5))))
    for finger in PAIRS:
        mapping.previous=runtime.initial_u.copy()
        out=mapping.solve_frame(frames[finger]);base=out['goal_rad'].copy()
        g.assign(data,base);target_pose=pad.pose(data,finger)
        target=target_pose['position_mm']+.8*target_pose['normal_world']
        target_normal=-target_pose['normal_world']
        def expand(v):
            u=base.copy();u[:5]=v;return u
        def metrics(v):
            g.assign(data,expand(v));pose=pad.pose(data,'thumb')
            return pose['position_mm'],pose['normal_world']
        def score(v):
            point,normal=metrics(v)
            return float(np.sum((point-target)**2)+16.*np.sum((normal-target_normal)**2)+
                         4.*np.sum((v-base[:5])**2))
        before_point,_=metrics(base[:5]);before=float(np.linalg.norm(before_point-target_pose['position_mm']))
        candidates=[]
        seeds=np.vstack((base[:5],runtime.initial_u[:5],pool))
        ranked=sorted(seeds,key=score)
        for v in ranked:
            if guard.safe(expand(v)):
                candidates.append(np.asarray(v).copy())
                if len(candidates)>=4:break
        if not candidates:raise ValueError(f'{finger}：当前四指姿态下未找到安全拇指初值')
        feasible=list(candidates)
        for v in candidates[:3]:
            fit=minimize(score,v,method='SLSQP',bounds=list(zip(mapping.lower[:5],mapping.upper[:5])),
                constraints=[dict(type='ineq',fun=lambda x:guard.distances(expand(x))-.65)],
                options=dict(maxiter=65,ftol=1e-5))
            v=np.clip(fit.x,mapping.lower[:5],mapping.upper[:5])
            if guard.safe(expand(v)):feasible.append(v)
        if guard.safe(base):feasible.append(base[:5])
        result=min(feasible,key=score);u=expand(result);point,normal=metrics(result)
        gap=float(np.linalg.norm(point-target_pose['position_mm']))
        # Never enable a fitted anchor that worsens the actual nominal pad gap.
        if gap>before:
            result=base[:5].copy();u=base.copy();point,normal=metrics(result);gap=before
        report=guard.report(u)
        record['anchors'][finger]=dict(frame=frames[finger],baseline_rad=base,
            baseline_desired_rad=out['desired_joint_rad'],
            calibrated_rad=u,thumb_delta_rad=result-out['desired_joint_rad'][:5],before_pad_gap_mm=before,
            after_pad_gap_mm=gap,clearance=report,
            normal_opposition_dot=float(np.dot(normal,target_pose['normal_world'])),
            nominal_contact_reached=bool(gap<3.),
            sample_quality=samples['samples'][finger]['p90_spread_deg'],
            four_finger_commands_unchanged=bool(np.array_equal(base[5:],u[5:])))
        print(f'{finger}: nominal pad gap {before:.2f} -> {gap:.2f} mm; hull clearance {report["clearance_mm"]:.2f} mm',flush=True)
    path=folder/'f1_pinch_profile.json';write_json(path,record)
    return path,record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder');parser.add_argument('--reference',required=True)
    args=parser.parse_args()
    folder=args.folder or json.loads((ROOT/'calibration/current_g7_f1_pinch.json').read_text(encoding='utf8'))['path']
    path,_=fit_session(folder,args.reference);print(path)
