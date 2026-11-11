"""Official joint rotations -> F1 angles, with bounded posture assistance.

The saved official open frame defines the four-finger software zero.
Thumb angles use anatomical bone/hinge zeros, with unit angular gain.
Neither vendor calibration nor captured F1 contact corrections are fitted.
Thumb root rotations are decomposed analytically in the fixed F1 axis order.
No whole-hand endpoint IK, contact class or fitted contact pose is used.
Optional posture projection is continuous, bounded and speed limited.
"""
import time
from types import SimpleNamespace
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hand_kinematics import FINGERS
from g7_joint_mapping import joint_rotations,twist
from g7_teleop import load_g7_config,reference_frame
from thumb_orientation_mapping import basis
from thumb_collision import ThumbCollisionGuard


def decode_relative_angles(frame, hand, open_rotations):
    if frame['hand']!=(1 if hand=='left' else 2):
        raise ValueError('G7手别与关节映射不一致')
    rel=joint_rotations(frame);result={}
    for f in FINGERS:
        delta=[a.inv()*b for a,b in zip(open_rotations[f],rel[f])]
        if f=='thumb':
            root=np.degrees(delta[0].as_rotvec())
            result[f]=dict(CMC_OPP=float(root[0]),CMC_PS=float(root[1]),MCP_AA=float(root[2]),
                MCP=np.degrees(twist(rel[f][1],[0.,0.,-1.])),
                IP=np.degrees(twist(rel[f][2],[0.,0.,-1.])))
        else:
            flex=twist(delta[0],[1.,0.,0.])
            remaining=delta[0]*Rotation.from_rotvec([-flex,0.,0.])
            result[f]=dict(MCP_AA=np.degrees(twist(remaining,[0.,0.,1.])),MCP=np.degrees(flex),
                PIP=np.degrees(twist(delta[1],[1.,0.,0.])),DIP=np.degrees(twist(delta[2],[1.,0.,0.])))
    return result


class G7DirectJointRetargeter:
    def __init__(self,geometry,reference,hand='right',config=None,open_frame=None):
        cfg=dict(config or load_g7_config(),mapping_profile=None,f1_pinch_profile=None,
                 joint_amplitude_mode='angle',thumb_flexion_mode='absolute_joint',
                 thumb_root_frame_mode='anatomical_angular_axes',joint_deadband_deg=0.)
        self.g=geometry;self.reference=geometry.validate(reference).copy()
        self.previous=self.reference.copy();self.hand=hand;self.config=cfg
        self.fixed_indices=np.array([],dtype=int)
        self.alignment=None;self.f1_pinch_alignment=None
        self.open_frame=reference_frame(cfg,hand) if open_frame is None else open_frame
        self.open_rotations=joint_rotations(self.open_frame)
        if self.open_frame['hand']!=(1 if hand=='left' else 2):
            raise ValueError('已有张开参考与手套手别不一致')
        self.lower=geometry.lower.copy();self.upper=geometry.upper.copy()
        self.thumb_ip_zero=float(self.reference[4])
        self.lower[4]=max(self.lower[4],self.thumb_ip_zero)
        self.upper[3]=min(self.upper[3],self.reference[3]+np.radians(cfg.get('thumb_mcp_visual_max_delta_deg',65.)))
        if cfg.get('thumb_assist_method') in ('constrained','contact_graph') and cfg.get('thumb_linear_assist_enabled',False):
            # The display initial pose is not a mechanical lower limit.
            self.lower=geometry.lower.copy();self.upper=geometry.upper.copy()
        self.side_sign=float(cfg.get('joint_side_gain',-1.))
        if self.side_sign not in (-1.,1.):
            raise ValueError('1:1侧摆映射的方向只能为+1或-1')
        self.gains={f:dict(mcp=1.,distal=1.) for f in FINGERS}
        self.collision=ThumbCollisionGuard(geometry,cfg.get('thumb_clearance_mm',.6),cfg.get('collision_step_deg',.75))
        # Establish the palm basis once from the saved reference. No live
        # endpoint, direction fitting, wearer calibration or solver is loaded.
        data=mujoco.MjData(geometry.model);geometry.assign(data,self.reference)
        p=np.asarray(self.open_frame['positions_m'],float)
        if p.shape!=(21,3) or not np.isfinite(p).all():
            raise ValueError('已有掌心参考需要21个有限位置')
        source_basis=basis(p[5]-p[17],p[[5,9,13,17]].mean(axis=0)-p[0])
        roots=[];rays=[]
        for f in FINGERS[1:]:
            joint=geometry.model.joint(geometry.cfg['fingers'][f]['columns'][0][0][0]).id
            root=data.xanchor[joint].copy();roots.append(root)
            rays.append(data.site_xpos[geometry.sites[f]]-root)
        palm=Rotation.from_matrix(basis(roots[0]-roots[-1],np.mean(rays,axis=0))@source_basis.T)
        if cfg.get('thumb_assist_method') in ('constrained','contact_graph') and cfg.get('thumb_linear_assist_enabled',False):
            from constrained_gesture_assist import fixed_palm_basis
            palm=Rotation.from_matrix(fixed_palm_basis(geometry)@source_basis.T)
        body=geometry.model.body('f1_link_3').id
        self.thumb_root_mapper=SimpleNamespace(data=data,body=body,palm_rotation=palm,
            target_root=Rotation.from_matrix(data.xmat[body].reshape(3,3)))
        # Native F1 reference axes already include the assembly transforms.
        # Only the angular Jacobian at this reference is used, once at setup.
        mp=self.thumb_root_mapper;jr=np.zeros((3,self.g.model.nv))
        mujoco.mj_jacBody(self.g.model,mp.data,None,jr,mp.body)
        self.root_axes=jr@self.g.C[:,:3]
        if np.linalg.cond(self.root_axes)>1e4:
            raise ValueError('F1拇指参考转轴退化，不能建立直接角度映射')
        self.root_gains=np.ones(3)
        # Match bone-longitudinal and hinge-normal conventions once. A saved
        # open hand need not have a straight thumb, so its bend is not zero.
        base=geometry.model.joint('axis_2').id
        mcp=geometry.model.joint('axis_21').id;ip=geometry.model.joint('axis_22').id
        ray=data.xanchor[mcp]-data.xanchor[base]
        model_local=basis(mp.target_root.inv().apply(ray),mp.target_root.inv().apply(data.xaxis[mcp]))
        source_local=basis([0.,-1.,0.],[0.,0.,-1.])
        self.thumb_root_frame_offset=Rotation.from_matrix(source_local@model_local.T)
        proximal=geometry.model.body('f1_link_5').id
        distal_base=geometry.model.body('f1_link_6').id;distal=geometry.model.body('f1_link_7').id
        bone_rays=[ray,data.xpos[distal_base]-data.xpos[proximal],data.site_xpos[geometry.sites['thumb']]-data.xpos[distal]]
        bone_rays=[v/np.linalg.norm(v) for v in bone_rays]
        def signed(a,b,axis):
            return np.degrees(np.arctan2(axis@np.cross(a,b),a@b))
        self.thumb_flexion_offsets_deg=np.array([signed(bone_rays[0],bone_rays[1],data.xaxis[mcp]),
                                                signed(bone_rays[1],bone_rays[2],data.xaxis[ip])])
        self.linear_assist=None
        self.imitation_assist=None
        if cfg.get('thumb_linear_assist_enabled',False):
            method=cfg.get('thumb_assist_method','linear')
            if method=='contact_graph':
                from contact_graph_assist import ContactGraphAssist
                self.imitation_assist=ContactGraphAssist(geometry,cfg,self.reference,hand)
            elif method=='constrained':
                from constrained_gesture_assist import ConstrainedGestureAssist
                self.imitation_assist=ConstrainedGestureAssist(geometry,cfg,self.reference,hand)
            elif method=='imitation':
                from joint_imitation_assist import JointImitationAssist
                self.imitation_assist=JointImitationAssist(geometry,cfg,self.reference,hand)
            elif method=='hybrid':
                from hybrid_thumb_assist import HybridThumbAssist
                self.linear_assist=HybridThumbAssist(geometry,cfg,self.reference,hand)
            elif method=='linear':
                from linear_thumb_assist import LinearThumbAssist
                self.linear_assist=LinearThumbAssist(geometry,cfg)
            else:raise ValueError('未知的拇指连续修正方法')
        self.mapping_version=('posture_v8_contact_graph' if getattr(self.imitation_assist,'kind',None)=='contact_graph' else
                              'posture_v7_constrained_gesture' if getattr(self.imitation_assist,'kind',None)=='constrained_gesture' else
                              'posture_v6_joint_imitation' if self.imitation_assist else
                              'unity_v5_hybrid_thumb' if getattr(self.linear_assist,'kind',None)=='hybrid_thumb'
                              else 'unity_v4_linear_thumb' if self.linear_assist else 'unity_v3_thumb_anatomical')

    def decode(self,frame):
        return decode_relative_angles(frame, self.hand, self.open_rotations)

    @staticmethod
    def positive(degrees):
        return np.radians(max(0.,degrees))

    def root_coordinates(self,source_root):
        # R = Ra(alpha) Rb(beta) Rc(gamma), for the original serial axes.
        # a.R.c = K*cos(beta) + H*sin(beta) + D isolates the middle angle.
        # This closed-form angular decomposition avoids a large-angle linear
        # approximation and does not evaluate model FK or an optimizer.
        mp=self.thumb_root_mapper
        absolute_target=mp.palm_rotation*source_root*self.thumb_root_frame_offset
        target=absolute_target*mp.target_root.inv()
        a,b,c=self.root_axes.T
        D=float((a@b)*(b@c));K=float(a@c-D);H=float(a@np.cross(b,c))
        phase=np.arctan2(H,K)
        middle=np.arccos(np.clip((a@target.apply(c)-D)/np.hypot(K,H),-1.,1.))
        def wrap(v):return (v+np.pi)%(2*np.pi)-np.pi
        candidates=[]
        for raw_beta in (phase-middle,phase+middle):
            beta=wrap(raw_beta)
            rb=Rotation.from_rotvec(b*beta);p=rb.apply(c);v=target.apply(c)
            alpha=np.arctan2(a@np.cross(p,v),p@v-(p@a)*(v@a))
            gamma=twist(rb.inv()*Rotation.from_rotvec(-a*alpha)*target,c)
            raw=np.array([alpha,beta,gamma])*self.root_gains
            clipped=np.clip(self.reference[:3]+raw,self.lower[:3],self.upper[:3])-self.reference[:3]
            actual=Rotation.from_rotvec(a*clipped[0])*Rotation.from_rotvec(b*clipped[1])*Rotation.from_rotvec(c*clipped[2])
            error=float((target.inv()*actual).magnitude())
            continuity=float(np.sum((clipped-(self.previous[:3]-self.reference[:3]))**2))
            candidates.append((1000.*error**2+.005*continuity,raw,error))
        _,value,error=min(candidates,key=lambda x:x[0])
        return value,dict(method='anatomical bone/hinge frame -> closed-form F1 root angles',
            source_rotation_vector_deg=np.degrees(source_root.as_rotvec()),
            orientation_residual_deg=float(np.degrees(error)),
            fingertip_position_used=False,numerical_ik_used=False)

    def solve_frame(self,frame):
        start=time.perf_counter();angles=self.decode(frame);desired=self.reference.copy()
        source_root=joint_rotations(frame)['thumb'][0]
        root_angles,root_report=self.root_coordinates(source_root)
        desired[:3]+=root_angles
        desired[3]+=self.positive(angles['thumb']['MCP']-self.thumb_flexion_offsets_deg[0])
        desired[4]+=self.positive(angles['thumb']['IP']-self.thumb_flexion_offsets_deg[1])
        for f in FINGERS[1:]:
            sl=self.g.slices[f];a=angles[f]
            desired[sl.start]+=np.radians(a['MCP_AA'])*self.side_sign
            desired[sl.start+1]+=self.positive(a['MCP'])
            desired[sl.start+2]+=self.positive(a['PIP'])
        unclipped=desired.copy();desired=np.clip(desired,self.lower,self.upper)
        direct=desired.copy();linear_report=None;imitation_report=None
        if self.linear_assist is not None:
            desired,linear_report=self.linear_assist.apply(frame,direct,self.lower,self.upper)
        if self.imitation_assist is not None:
            desired,imitation_report=self.imitation_assist.apply(frame,direct,self.lower,self.upper,angles)
        # Guard the final posture; never jump to a distant stored pose seed.
        if getattr(self,'angle_profile',None) is not None:
            unclipped,direct,desired=self.calibrated_command(frame,angles)
            sweep=self.collision.sweep_follow_digits if self.limit_assist else self.collision.sweep_follow
        else:
            sweep=self.collision.sweep_follow
        goal,avoidance=sweep(self.previous,desired)
        avoidance['projected']=bool(avoidance['blocked'])
        self.previous=self.g.validate(goal).copy();records={};side={}
        if imitation_report is not None:
            imitation_report['guarded_correction_deg']=np.degrees(goal-direct)
            imitation_report['avoidance_delta_deg']=np.degrees(goal-desired)
            imitation_report['path_blocked']=bool(avoidance['blocked'])
            if getattr(self.imitation_assist,'kind',None)=='contact_graph':
                imitation_report['guarded_contacts']=self.imitation_assist.metrics(goal,self.imitation_assist.targets(frame))
        if linear_report is not None:
            linear_report['guarded_correction_deg']=np.degrees(goal[:5]-direct[:5])
            linear_report['path_blocked']=bool(avoidance['blocked'])
        for f in FINGERS:
            sl=self.g.slices[f];command=self.g.forward(desired,f)['position_mm'].copy()
            actual=self.g.forward(goal,f)['position_mm'].copy()
            records[f]=dict(success=True,coordinates_rad=goal[sl],source_joint_angles_deg=angles[f],
                requested_position_mm=command,execution_position_mm=actual,
                position_error_mm=float(np.linalg.norm(actual-command)),
                position_error_scope='FK of angle commands only, not glove endpoint tracking',
                projected=bool(not np.allclose(goal[sl],unclipped[sl])),
                solver=('angle-prior bounded posture projection; sampled avoidance' if imitation_report else
                        'direct angular mapping; bounded coupling and sampled avoidance'))
            i=sl.start+(2 if f=='thumb' else 0)
            side[f]=dict(actual_goal_deg=float(np.degrees(goal[i])),
                         limits_deg=np.degrees([self.lower[i],self.upper[i]]))
        root_report.update(requested_coordinates_deg=np.degrees(unclipped[:3]),
            reference_axes_world=self.root_axes,
            at_limit=bool(np.any(np.abs(unclipped[:3]-direct[:3])>1e-8)),
            limitation='root-angle limits and different anatomy can prevent equal gestures')
        return dict(goal_rad=goal,desired_joint_rad=desired,direct_joint_rad=direct,unclipped_joint_rad=unclipped,
            thumb_linear_assist=linear_report,joint_imitation=imitation_report,
            fingers=records,angles_deg=angles,model_angles_deg=np.degrees(goal),hand=self.hand,
            retargeting_kind='joint_visual',control_mode='direct_joint',glove_fk_api='g7_direct_joint',
            glove_project='Forsense official calibrated joint rotations',
            side_swing=dict(enabled=True,method='direct signed angular channels',fingers=side),
            thumb_avoidance=avoidance,thumb_visual_limits=dict(ip_straight_coordinate_deg=np.degrees(self.thumb_ip_zero),
                ip_flexion_range_deg=np.degrees([0.,self.upper[4]-self.thumb_ip_zero]),hyperextension_allowed=False),
            joint_mapping_gains=self.gains,thumb_root_mapping=root_report,
            thumb_calibrated_independently=False,thumb_independent_calibration=False,
            thumb_flexion_mode='absolute_joint',thumb_root_frame_mode='anatomical_angular_axes',
            thumb_model_flexion_offsets_deg=self.thumb_flexion_offsets_deg,
            joint_mapping_version=self.mapping_version,coupling_driver=('PIP_DIP_posture_projection' if imitation_report else 'PIP'),
            joint_zero='four_finger_saved_open_thumb_anatomical',
            f1_pinch_alignment=None,f1_pinch_profile=None,calibrated_to_wearer=False,
            vendor_calibration_reused=True,f1_extra_calibration_used=False,
            amplitude_mode='angle',model_mode=self.g.mode,hardware_enabled=False,
            calibration_scope='official calibration reused; four-finger saved open, thumb bone/hinge anatomical zeros; no new collection',
            pipeline='official relative joint rotations -> angular baseline -> bounded whole-hand posture projection (bone directions + DIP observations) -> sampled avoidance -> F1 angles'
                if imitation_report else
                'official relative joint rotations -> signed angular baseline -> bounded thumb direction/relative-vector correction -> sampled avoidance -> F1 angles'
                if self.mapping_version=='unity_v5_hybrid_thumb' else
                'official relative joint rotations -> signed angular channels -> bounded optional linear thumb correction -> sampled avoidance -> F1 angles',
            root_frame='parent-relative rotations; anatomical thumb frame and closed-form angular decomposition',
            projection_method=('joint-angle prior and all-bone directions within limits/coupling; weak relative vectors; no forced pinch' if imitation_report else
                'angle limits/coupling; optional thumb-only local geometry correction; no forced pinch'),
            solve_ms=(time.perf_counter()-start)*1000.)

    def mapping_record(self):
        return dict(kind='direct_joint',hand=self.hand,model_mode=self.g.mode,
            source='official calibrated 21 bone quaternions',
            reference_capture=self.config['reference_capture'],
            reference_definition='saved open for four fingers; thumb anatomical bone/hinge zeros; no fresh sample or vendor changes',
            thumb_reference_axes_world=self.root_axes,thumb_angular_gains=self.root_gains,
            joint_mapping_version=self.mapping_version,joint_zero='four_finger_saved_open_thumb_anatomical',
            thumb_linear_assist_enabled=self.linear_assist is not None,
            thumb_model_flexion_offsets_deg=self.thumb_flexion_offsets_deg,
            thumb_root_frame_offset_xyzw=self.thumb_root_frame_offset.as_quat(),
            finger_side_sign=self.side_sign,coupling_driver=('PIP_DIP_posture_projection' if self.imitation_assist else 'PIP'),glove_dip_controls_f1=self.imitation_assist is not None,
            f1_source_sha256=self.g.cfg['source_sha256'],
            vendor_calibration_reused=True,f1_extra_calibration_used=False,
            fingertip_ik=False,whole_hand_ik=self.imitation_assist is not None,whole_hand_endpoint_ik=False,
            thumb_local_geometry_correction=self.linear_assist is not None,
            whole_hand_posture_projection=self.imitation_assist is not None,
            bone_direction_fit=self.imitation_assist is not None or self.mapping_version=='unity_v5_hybrid_thumb',forced_pinch=False,
            coupling='original half-angle reduction and DIP=.8*PIP',
            limitations='different anatomy and angle limits; equal angles do not guarantee fingertip contact')
