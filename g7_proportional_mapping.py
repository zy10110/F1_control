"""Official solved hand -> per-bone length retargeting -> bounded F1 shape IK.

No wearer or F1 pinch samples are fitted or loaded. Geometry landmarks and
optional contact seeds come from the F1 model, not a hand-pose calibration.
Each instance owns all MuJoCo data; commands retain HandGeometry's reduction.
"""
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from hand_kinematics import ROOT, FINGERS
from g7_teleop import BASE, reference_frame, load_g7_config
from thumb_orientation_mapping import basis
from thumb_collision import ThumbCollisionGuard
from palm_calibration import PadGeometry
from g7_f1_contact_assist import PAIRS
from g7_joint_mapping import joint_rotations,twist


def normalized(values):
    values=np.asarray(values,float)
    length=np.linalg.norm(values,axis=-1,keepdims=True)
    if not np.isfinite(values).all() or np.any(length<1e-7):
        raise ValueError('手套或F1骨段长度无效')
    return values/length


def source_points(frame,hand):
    if frame.get('hand') != (1 if hand=='left' else 2):
        raise ValueError('官方21点手别与F1映射不一致')
    p=np.asarray(frame.get('positions_m'),float)
    if p.shape!=(21,3) or not np.isfinite(p).all():
        raise ValueError('比例重定向需要官方21点位置，单位米')
    lengths=np.array([np.linalg.norm(np.diff(p[i:i+4],axis=0),axis=1) for i in BASE.values()])
    if np.min(lengths)<.005 or np.max(lengths)>.15:
        raise ValueError('官方21点骨段长度无效，保持上一姿态')
    return p*1000.


class FingerLandmarks:
    """Four semantic points, using both centers of each rolling interface.

    An effective hinge center is the midpoint of its two original URDF axes;
    it is a kinematic landmark, not a newly introduced fixed physical hinge.
    Jacobians include the motion of both axes through the original reduction.
    """
    def __init__(self,g):
        self.g=g;self.data=mujoco.MjData(g.model);self.interfaces={}
        for f in FINGERS:
            if f=='thumb':
                groups=[('axis_2',),('axis_21','axis_3'),('axis_22','axis_4')]
            else:
                k=FINGERS.index(f)-1
                groups=[(f'axis_{6+4*k}',f'axis_{23+3*k}'),
                        (f'axis_{7+4*k}',f'axis_{24+3*k}'),
                        (f'axis_{8+4*k}',f'axis_{25+3*k}')]
            self.interfaces[f]=[[g.model.joint(name).id for name in group] for group in groups]

    def assign(self,u):self.g.assign(self.data,u)

    def points(self,f,jacobian=False):
        m,d=self.g.model,self.data;points=[];jacs=[]
        for group in self.interfaces[f]:
            points.append(np.mean(d.xanchor[group],axis=0)*1000.)
            if jacobian:
                rows=[]
                for j in group:
                    jp=np.zeros((3,m.nv))
                    mujoco.mj_jac(m,d,jp,None,d.xanchor[j],int(m.jnt_bodyid[j]))
                    rows.append(jp@self.g.C*1000.)
                jacs.append(np.mean(rows,axis=0))
        points.append(d.site_xpos[self.g.sites[f]]*1000.)
        if jacobian:
            jp=np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jp,None,self.g.sites[f])
            jacs.append(jp@self.g.C*1000.)
            return np.array(points),np.array(jacs)
        return np.array(points)


class G7ProportionalRetargeter:
    def __init__(self,geometry,reference,hand='right',config=None):
        self.g=geometry;self.reference=geometry.validate(reference).copy()
        self.previous=self.reference.copy();self.hand=hand;self.config=config or load_g7_config()
        self.fixed_indices=np.array([],dtype=int)
        # Read an existing official solved frame only for lengths/palm axes.
        # No zero-angle offsets, F1 fitted pinch anchors or SDK MLP are loaded.
        self.open_frame=reference_frame(dict(self.config,mapping_profile=None),hand)
        src=source_points(self.open_frame,hand)
        self.land=FingerLandmarks(geometry);self.land.assign(self.reference)
        self.rest={f:self.land.points(f) for f in FINGERS}
        self.lengths={f:np.linalg.norm(np.diff(p,axis=0),axis=1) for f,p in self.rest.items()}
        self.source_lengths={f:np.linalg.norm(np.diff(src[i:i+4],axis=0),axis=1) for f,i in BASE.items()}
        roots={f:self.rest[f][0] for f in FINGERS}
        target_basis=basis(roots['index']-roots['pinky'],np.mean([
            self.rest[f][-1]-roots[f] for f in PAIRS],axis=0))
        self.target_basis=target_basis
        self.lower=geometry.lower.copy();self.upper=geometry.upper.copy()
        # Keep the existing visual thumb anti-hyperextension boundary.
        self.lower[4]=max(self.lower[4],self.reference[4])
        self.upper[3]=min(self.upper[3],self.reference[3]+np.radians(
            self.config.get('thumb_mcp_visual_max_delta_deg',65.)))
        self.collision=ThumbCollisionGuard(geometry,self.config.get('thumb_clearance_mm',.6),
                                           self.config.get('collision_step_deg',.75))
        pad_record=json.loads((ROOT/'calibration/g7_f1_pad_candidates.json').read_text(encoding='utf8'))
        self.pad=PadGeometry(geometry,pad_record);self.pad_record=pad_record
        self.pad_data=mujoco.MjData(geometry.model)
        self.current_targets=None
        self.relative_weight=float(self.config.get('proportional_relative_weight',.6))
        self.temporal_weight=float(self.config.get('proportional_temporal_weight',2.))
        self.frame_step=np.radians(float(self.config.get('proportional_frame_step_deg',8.)))
        if not np.isfinite([self.relative_weight,self.temporal_weight,self.frame_step]).all() or min(
                self.relative_weight,self.temporal_weight,self.frame_step)<0 or self.frame_step==0:
            raise ValueError('连续手势参数无效')
        # Kept as a diagnostic for callers: no contact-pose seed file is read.
        self.contact_seeds={}

    def targets(self,frame):
        src=source_points(frame,self.hand)
        sb=basis(src[5]-src[17],src[[5,9,13,17]].mean(axis=0)-src[0])
        R=self.target_basis@sb.T
        targets={}
        for f,i in BASE.items():
            v=np.diff(src[i:i+4],axis=0);directions=normalized(v)@R.T
            segments=self.lengths[f][:,None]*directions
            targets[f]=dict(points_mm=np.vstack((self.rest[f][0],self.rest[f][0]+np.cumsum(segments,axis=0))),
                directions=directions,source_lengths_mm=np.linalg.norm(v,axis=1),
                length_ratios=self.lengths[f]/np.linalg.norm(v,axis=1))
        return targets,src,R

    def shape_system(self,u,f,target,ids,prior):
        self.land.assign(u);p,J=self.land.points(f,True)
        v=np.diff(p,axis=0);dv=np.diff(J,axis=0);length=np.linalg.norm(v,axis=1)
        directions=v/length[:,None]
        jd=np.array([(np.eye(3)-np.outer(n,n))@d/l for n,d,l in zip(directions,dv,length)])
        # Position differences cancel the moving virtual root. Every point and
        # direction is still evaluated with the actual rolling-joint FK.
        dp=(p[1:]-p[0])-(target['points_mm'][1:]-target['points_mm'][0])
        residual=np.r_[30.*(directions-target['directions']).ravel(),.35*dp.ravel(),.35*(u[ids]-prior[ids])]
        jac=np.vstack((30.*jd[:,:,ids].reshape(9,len(ids)),
                       .35*(J[1:]-J[0])[:,:,ids].reshape(9,len(ids)),.35*np.eye(len(ids))))
        return residual,jac

    @staticmethod
    def bounded_step(residual,jac,value,lower,upper,cap_deg=16.):
        cap=np.radians(cap_deg);n=len(value)
        return lsq_linear(np.vstack((jac,np.eye(n))),-np.r_[residual,np.zeros(n)],
            bounds=(np.maximum(lower-value,-cap),np.minimum(upper-value,cap)),
            method='bvls',tol=1e-7,max_iter=25).x

    def fit_finger(self,u,f,target):
        sl=self.g.slices[f];ids=np.arange(sl.start,sl.stop);prior=u.copy()
        seeds=[u.copy()]
        def score(v):
            r,_=self.shape_system(v,f,target,ids,prior);return float(r@r)
        best=u.copy()
        for candidate in seeds:
            test=u.copy();test[ids]=np.clip(candidate[ids],self.lower[ids],self.upper[ids])
            if score(test)<score(best):best=test
        for _ in range(6 if f=='thumb' else 4):
            r,j=self.shape_system(best,f,target,ids,prior)
            if np.linalg.norm(j.T@r)<1e-5:break
            step=self.bounded_step(r,j,best[ids],self.lower[ids],self.upper[ids])
            old=float(r@r);accepted=False
            for scale in (1.,.5,.2):
                candidate=best.copy();candidate[ids]=np.clip(best[ids]+scale*step,self.lower[ids],self.upper[ids])
                if score(candidate)<old-1e-7:best=candidate;accepted=True;break
            if not accepted:break
        return best

    def pinch_weights(self,src):
        gaps={f:float(np.linalg.norm(src[4]-src[BASE[f]+3])) for f in PAIRS}
        strength={}
        for f in PAIRS:
            length=min(np.sum(np.linalg.norm(np.diff(src[1:5],axis=0),axis=1)),
                       np.sum(np.linalg.norm(np.diff(src[BASE[f]:BASE[f]+4],axis=0),axis=1)))
            # All four objectives exist continuously. No selected contact,
            # full-closure threshold or winner-takes-all hand-pose switch.
            strength[f]=float(1./(1.+(gaps[f]/(.4*length))**4))
        return strength,gaps,None

    def relative_targets(self,src,R):
        weights,gaps,_=self.pinch_weights(src);result={}
        source_length={f:np.sum(np.linalg.norm(np.diff(src[i:i+4],axis=0),axis=1)) for f,i in BASE.items()}
        for f in PAIRS:
            scale=np.sqrt((self.lengths['thumb'].sum()/source_length['thumb'])*
                          (self.lengths[f].sum()/source_length[f]))
            result[f]=dict(vector_mm=R@(src[4]-src[BASE[f]+3])*scale,
                           weight=weights[f],source_gap_mm=gaps[f],scale=float(scale))
        return result

    def gesture_system(self,u,targets,relative,prior):
        ids=np.arange(len(u));residual=[];jac=[]
        for f in FINGERS:
            r,j=self.shape_system(u,f,targets[f],ids,prior)
            residual.append(r);jac.append(j)
        self.land.assign(u)
        points={f:self.land.points(f,True) for f in FINGERS}
        for f in PAIRS:
            weight=self.relative_weight*np.sqrt(relative[f]['weight'])
            delta=points['thumb'][0][-1]-points[f][0][-1]-relative[f]['vector_mm']
            residual.append(weight*delta)
            jac.append(weight*(points['thumb'][1][-1]-points[f][1][-1]))
        residual.append(self.temporal_weight*(u-prior));jac.append(self.temporal_weight*np.eye(len(u)))
        return np.concatenate(residual),np.vstack(jac)

    def fit_gesture(self,targets,relative):
        # One warm-started solve for every gesture, including pinches. The
        # trust box bounds the entire frame, not each inner solver iteration.
        prior=np.clip(self.previous,self.lower,self.upper);best=prior.copy()
        lo=np.maximum(self.lower,prior-self.frame_step);hi=np.minimum(self.upper,prior+self.frame_step)
        for _ in range(4):
            r,j=self.gesture_system(best,targets,relative,prior)
            step=self.bounded_step(r,j,best,lo,hi,8.)
            if np.max(np.abs(step))<1e-6:break
            old=float(r@r);accepted=False
            for scale in (1.,.5,.2):
                candidate=np.clip(best+scale*step,lo,hi)
                rr,_=self.gesture_system(candidate,targets,relative,prior)
                if float(rr@rr)<old-1e-7:best=candidate;accepted=True;break
            if not accepted:break
        return best

    def gaps(self,u):
        self.g.assign(self.pad_data,u);a=self.pad.pose(self.pad_data,'thumb')['position_mm']
        return {f:float(np.linalg.norm(a-self.pad.pose(self.pad_data,f)['position_mm'])) for f in PAIRS}

    def solve_frame(self,frame):
        started=time.perf_counter();targets,src,R=self.targets(frame)
        self.current_targets=targets
        relative_targets=self.relative_targets(src,R)
        desired=self.fit_gesture(targets,relative_targets)
        # Preserve the established clearance/path guard without jumping to a
        # distant safe pose seed, which can look like a selected hand gesture.
        goal,avoidance=self.collision.sweep_follow(self.previous,desired)
        avoidance['projected']=bool(avoidance['blocked'])
        self.previous=self.g.validate(goal).copy();self.land.assign(goal)
        fingers={};side={};angles={};shape_errors={};relative=joint_rotations(frame)
        for f in FINGERS:
            p=self.land.points(f);sl=self.g.slices[f]
            current=normalized(np.diff(p,axis=0));wanted=targets[f]['directions']
            errors=np.degrees(np.arccos(np.clip(np.sum(current*wanted,axis=1),-1.,1.)))
            requested=targets[f]['points_mm'][-1]-self.g.origin*1000.
            actual=self.g.forward(goal,f)['position_mm'].copy()
            fingers[f]=dict(success=True,coordinates_rad=goal[sl],requested_position_mm=requested,
                execution_position_mm=actual,position_error_mm=float(np.linalg.norm(actual-requested)),
                projected=bool(np.any(errors>1.)),solver='bounded per-bone shape/position least squares',
                bone_direction_errors_deg=errors.tolist(),target_landmarks_mm=(targets[f]['points_mm']-self.g.origin*1000.).tolist())
            shape_errors[f]=errors.tolist()
            i=sl.start+(2 if f=='thumb' else 0)
            side[f]=dict(actual_goal_deg=float(np.degrees(goal[i])),limits_deg=np.degrees([self.lower[i],self.upper[i]]))
            a,b,c=relative[f]
            if f=='thumb':
                angles[f]=dict(MCP=float(np.degrees(twist(b,[0.,0.,-1.]))),
                    IP=float(np.degrees(twist(c,[0.,0.,-1.]))))
            else:
                flex=twist(a,[1.,0.,0.]);remaining=a*Rotation.from_rotvec([-flex,0.,0.])
                angles[f]=dict(MCP=float(np.degrees(flex)),PIP=float(np.degrees(twist(b,[1.,0.,0.]))),
                    DIP=float(np.degrees(twist(c,[1.,0.,0.]))),MCP_AA=float(np.degrees(twist(remaining,[0.,0.,1.]))))
        report=dict(kind='continuous bone shape and four relative tip vectors',profile='model_geometry_only',
            weights={f:v['weight'] for f,v in relative_targets.items()},active_pair=None,
            source_tip_gaps_mm={f:v['source_gap_mm'] for f,v in relative_targets.items()},
            target_tip_gaps_mm={f:float(np.linalg.norm(v['vector_mm'])) for f,v in relative_targets.items()},
            paired_finger_enabled=False,forced_closure=False,pose_switching=False,
            command_pad_gaps_mm=self.gaps(goal),
            scope='pad gaps are diagnostics, not fixed closure targets')
        return dict(goal_rad=goal,desired_joint_rad=desired,fingers=fingers,angles_deg=angles,
            model_angles_deg=np.degrees(goal),hand=self.hand,retargeting_kind='proportional_shape',
            glove_fk_api='g7_proportional',glove_project='Forsense official solved 21-point UDP',
            side_swing=dict(enabled=True,method='joint direction/position constrained IK',fingers=side),
            thumb_avoidance=avoidance,thumb_visual_limits=dict(ip_straight_coordinate_deg=np.degrees(self.reference[4]),
                ip_flexion_range_deg=np.degrees([0.,self.upper[4]-self.reference[4]]),hyperextension_allowed=False),
            joint_mapping_gains={},thumb_root_mapping={},thumb_calibrated_independently=False,
            f1_pinch_alignment=report,calibrated_to_wearer=False,
            vendor_calibration_reused=True,f1_extra_calibration_used=False,
            pad_geometry_record=self.pad_record,
            calibration_scope='vendor calibration reused; no fitted F1 pose offsets or new user samples',
            length_retargeting={f:dict(f1_lengths_mm=self.lengths[f].tolist(),source_lengths_mm=targets[f]['source_lengths_mm'].tolist(),
                ratios=targets[f]['length_ratios'].tolist(),bone_direction_errors_deg=shape_errors[f]) for f in FINGERS},
            shape_only_coordinates_deg=None,
            gesture_objective='continuous_shape_and_relative_vectors',
            amplitude_mode='per_bone_length',model_mode=self.g.mode,
            pipeline='official solved 21 points -> per-bone directions and relative tip vectors -> continuous bounded coupled IK -> sampled avoidance',
            root_frame='anatomical palm; wrist motion removed',hardware_enabled=False,
            solve_ms=(time.perf_counter()-started)*1000,
            projection_method='bounded local nearest shape approximation; not global optimum')

    def mapping_record(self):
        return dict(kind='proportional_shape',hand=self.hand,model_mode=self.g.mode,
            f1_source_sha256=self.g.cfg['source_sha256'],reference_rad=self.reference,
            f1_lengths_mm=self.lengths,source_lengths_mm=self.source_lengths,
            ratios={f:self.lengths[f]/self.source_lengths[f] for f in FINGERS},
            effective_centers='midpoint of the two original rolling-interface axes',
            calibration='official solved stream; no fitted F1 pose calibration',
            objective='continuous bone shape + four scaled relative tip vectors; no contact pose selection',
            limitations='local bounded projection and nominal model pads; no exact/global contact claim')
