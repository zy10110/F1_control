"""G7 relative joint rotations -> F1 coupled joint coordinates.

Visual imitation, independent of finger lengths. Optional labelled F1 pinch
anchors add local thumb-only angle offsets; no live Cartesian IK is run.
SDK quaternions describe wrist-frame bone orientations (wxyz); relative
rotations isolate each joint before applying wearer zero offsets.
"""
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

from hand_kinematics import FINGERS
from g7_teleop import BASE, load_g7_config, reference_frame
from thumb_collision import ThumbCollisionGuard
from thumb_orientation_mapping import ThumbOrientationMapper,ThumbBoneFrameMapper


def rotations(frame):
    q=np.asarray(frame.get('quaternions_wxyz'),float)
    if q.shape!=(21,4) or not np.isfinite(q).all() or np.min(np.linalg.norm(q,axis=1))<1e-6:
        raise ValueError('G7关节映射需要21个有效wxyz四元数')
    return Rotation.from_quat(q[:,[1,2,3,0]])


def joint_rotations(frame):
    r=rotations(frame)
    return {f:(r[0].inv()*r[i],r[i].inv()*r[i+1],r[i+1].inv()*r[i+2])
            for f,i in BASE.items()}


def twist(rotation,axis):
    """Signed swing/twist component, invariant to quaternion sign."""
    q=rotation.as_quat();a=np.asarray(axis,float);a=a/np.linalg.norm(a)
    angle=2*np.arctan2(np.dot(q[:3],a),q[3])
    return float((angle+np.pi)%(2*np.pi)-np.pi)


def thumb_hinge_axis(changes):
    """Reject longitudinal bone twist and weak/ambiguous bend samples.

    The SDK bone points along local -Y. Thumb hinge flexion is transverse,
    nominally about -Z; a small Y rotation must never redefine flexion.
    """
    nominal=np.array([0.,0.,-1.]);candidates=[]
    for value in changes:
        transverse=np.asarray(value,float).copy();transverse[1]=0.
        amplitude=np.linalg.norm(transverse)
        if amplitude<np.radians(12):continue
        axis=transverse/amplitude
        if abs(np.dot(axis,nominal))<.8:continue
        if np.dot(axis,nominal)<0:axis=-axis
        candidates.append((amplitude,axis))
    return max(candidates,key=lambda pair:pair[0])[1] if candidates else nominal


class G7JointRetargeter:
    def __init__(self,geometry,reference,hand='right',config=None):
        self.g=geometry;self.reference=self.g.validate(reference).copy()
        self.previous=self.reference.copy();self.hand=hand;self.config=config or load_g7_config()
        self.fixed_indices=np.array([],dtype=int)
        self.open_frame=reference_frame(self.config,hand)
        self.open_rotations=joint_rotations(self.open_frame)
        self.alignment=None
        if self.config.get('mapping_profile'):
            self.alignment=json.loads(Path(self.config['mapping_profile']).read_text(encoding='utf8'))
            if self.alignment['hand']!=hand or self.alignment['source_sha256']!=self.g.cfg['source_sha256']:
                raise ValueError('关节映射标定与手别或模型不匹配')
        self.thumb_reference_frame=self.open_frame
        self.thumb_calibration=(self.alignment or {}).get('thumb_joint_calibration')
        if self.thumb_calibration:
            self.thumb_reference_frame=self.thumb_calibration['reference_frame']
            self.open_rotations['thumb']=joint_rotations(self.thumb_reference_frame)['thumb']
        self.thumb_axes={1:np.array([0.,0.,-1.]),2:np.array([0.,0.,-1.])}
        # Only the thumb's oblique MCP/IP hinge directions are estimated.
        # Finger proximal rotations use the SDK bone-local X/Z convention.
        if self.alignment:
            for j in (1,2):
                changes=[]
                for key in ('fist_frame','pinch_frame'):
                    rel=joint_rotations(self.alignment[key])['thumb'][j]
                    changes.append((self.open_rotations['thumb'][j].inv()*rel).as_rotvec())
                self.thumb_axes[j]=thumb_hinge_axis(changes)
        if self.thumb_calibration:
            for j in (1,2):
                axis=np.asarray(self.thumb_calibration['hinge_axes'][str(j)],float)
                if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<.99 or abs(axis[1])>1e-5:
                    raise ValueError('拇指独立标定的屈曲轴无效')
                self.thumb_axes[j]=axis/np.linalg.norm(axis)
        self.thumb_flexion_mode=self.config.get('thumb_flexion_mode','relative_delta')
        if self.thumb_flexion_mode not in ('relative_delta','absolute_joint'):
            raise ValueError('未知拇指屈曲角定义')
        for key,raw in self.config.get('thumb_hinge_axes',{}).items():
            axis=np.asarray(raw,float)
            if key not in ('1','2') or axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-6:
                raise ValueError('拇指屈曲轴配置无效')
            self.thumb_axes[int(key)]=axis/np.linalg.norm(axis)
        self.thumb_root_frame_mode=self.config.get('thumb_root_frame_mode','rotation_delta')
        root_type=ThumbBoneFrameMapper if self.thumb_root_frame_mode=='bone_and_bend_plane' else ThumbOrientationMapper
        self.thumb_root_mapper=root_type(self.g,self.reference,self.open_frame,self.open_rotations['thumb'][0])
        self.lower=self.g.lower.copy()
        self.upper=self.g.upper.copy()
        # The original strict thumb IP has an assembly offset. Its geometric
        # straight pose is ~86.407 independent degrees, not coordinate zero.
        self.thumb_ip_zero=float(self.reference[4])
        self.lower[4]=max(self.lower[4],self.thumb_ip_zero)
        self.upper[3]=min(self.upper[3],self.reference[3]+np.radians(self.config.get('thumb_mcp_visual_max_delta_deg',65.)))
        self.amplitude_mode=self.config.get('joint_amplitude_mode','angle')
        if self.amplitude_mode not in ('angle','calibrated_range'):raise ValueError('未知关节幅度映射方式')
        self.decoder_reference=self.decode(self.open_frame)
        closed=self.decode(self.alignment['fist_frame']) if self.alignment else None
        self.gains={};self.deadband=np.radians(self.config.get('joint_deadband_deg',1.))
        for f in FINGERS:
            # Visual imitation copies angular motion. A weak closed sample
            # cannot amplify a small gesture into an artificial full fist.
            if self.amplitude_mode=='angle':
                self.gains[f]=dict(mcp=1.,distal=1.)
                continue
            sl=self.g.slices[f]
            mcp=sl.start+(3 if f=='thumb' else 1);distal=mcp+1
            source=closed[f] if closed else dict(MCP=75.,PIP=85.,DIP=68.,IP=45.)
            distal_source=source['IP'] if f=='thumb' else self.coupled_pip(source)
            desired_mcp=min(np.radians(65 if f=='thumb' else 75),self.g.upper[mcp]-self.reference[mcp])
            desired_distal=min(np.radians(55 if f=='thumb' else 85),self.g.upper[distal]-self.reference[distal])
            self.gains[f]=dict(mcp=min(2.5,desired_mcp/np.radians(max(30.,source['MCP']))),
                               distal=min(2.5,desired_distal/np.radians(max(30.,distal_source))))
        self.collision=ThumbCollisionGuard(self.g,self.config.get('thumb_clearance_mm',.6),
                                            self.config.get('collision_step_deg',.75))
        self.thumb_samples=np.array(list(itertools.product(*[np.linspace(self.lower[i],self.upper[i],3) for i in range(5)])))
        self.f1_pinch_alignment=None
        if self.config.get('f1_pinch_profile'):
            from g7_f1_pinch_alignment import F1PinchAlignment
            self.f1_pinch_alignment=F1PinchAlignment(self.g,self.config['f1_pinch_profile'],
                self.hand,self.reference,self.open_frame)

    def decode(self,frame):
        if frame['hand']!=(1 if self.hand=='left' else 2):raise ValueError('G7手别与关节映射不一致')
        rel=joint_rotations(frame);result={}
        for f in FINGERS:
            delta=[a.inv()*b for a,b in zip(self.open_rotations[f],rel[f])]
            if f=='thumb':
                root=np.degrees(delta[0].as_rotvec())
                bends=rel[f] if self.thumb_flexion_mode=='absolute_joint' else delta
                result[f]=dict(CMC_OPP=float(root[0]),CMC_PS=float(root[1]),MCP_AA=float(root[2]),
                    MCP=np.degrees(twist(bends[1],self.thumb_axes[1])),
                    IP=np.degrees(twist(bends[2],self.thumb_axes[2])))
            else:
                flex=twist(delta[0],[1,0,0])
                remaining=delta[0]*Rotation.from_rotvec([-flex,0,0])
                result[f]=dict(MCP_AA=np.degrees(twist(remaining,[0,0,1])),MCP=np.degrees(flex),
                    PIP=np.degrees(twist(delta[1],[1,0,0])),DIP=np.degrees(twist(delta[2],[1,0,0])))
        return result

    def coupled_pip(self,angles):
        # Closest weighted angle pair on DIP=.8*PIP; the glove DIP is observed,
        # but the F1 DIP cannot be commanded independently.
        w=float(self.config.get('joint_dip_weight',.25));ratio=.8
        return (max(0.,angles['PIP'])+w*ratio*max(0.,angles['DIP']))/(1+w*ratio**2)

    def positive(self,degrees):
        a=max(0.,np.radians(degrees))
        return 0. if a<self.deadband else a

    def solve_frame(self,frame):
        started=time.perf_counter();angles=self.decode(frame);desired=self.reference.copy()
        for f in FINGERS:
            sl=self.g.slices[f];a=angles[f];gain=self.gains[f]
            if f=='thumb':
                if self.thumb_root_frame_mode=='bone_and_bend_plane':
                    desired[:3],root_report=self.thumb_root_mapper.solve_frame(frame,self.previous)
                    offsets=self.thumb_root_mapper.flexion_offsets_deg
                else:
                    desired[:3],root_report=self.thumb_root_mapper.solve(joint_rotations(frame)['thumb'][0],self.previous)
                    offsets=np.zeros(2)
                desired[3]+=gain['mcp']*self.positive(a['MCP']-offsets[0])
                desired[4]+=gain['distal']*self.positive(a['IP']-offsets[1])
            else:
                desired[sl.start]+=np.radians(a['MCP_AA'])*self.config.get('joint_side_gain',1.)
                desired[sl.start+1]+=gain['mcp']*self.positive(a['MCP'])
                desired[sl.start+2]+=gain['distal']*self.positive(self.coupled_pip(a))
        desired=np.clip(desired,self.lower,self.upper)
        if self.thumb_root_frame_mode=='bone_and_bend_plane':
            desired[:5],shape_report=self.thumb_root_mapper.fit_bone_directions(
                frame,desired,self.previous,self.lower,self.upper)
            root_report=dict(method='bounded thumb bone-direction retargeting',
                seed_alignment=root_report,shape_fit=shape_report,
                metacarpal_direction_error_deg=shape_report['bone_direction_errors_deg'][0],
                mcp_bend_axis_error_deg=shape_report['mcp_bend_axis_error_deg'],
                at_limit=shape_report['at_limit'],fingertip_position_used=False)
        pinch_report=None
        if self.f1_pinch_alignment is not None:
            desired,pinch_report=self.f1_pinch_alignment.apply(frame,desired,self.lower,self.upper)
        goal,avoidance=self.collision.project(desired,self.previous,self.reference,self.thumb_samples,
                                             lower_override=self.lower,upper_override=self.upper)
        if pinch_report is not None:
            pinch_report['command_pad_gaps_mm']=self.f1_pinch_alignment.gaps(goal)
        self.previous=self.g.validate(goal).copy();records={};side={}
        for f in FINGERS:
            sl=self.g.slices[f];command=self.g.forward(desired,f)['position_mm'].copy()
            actual=self.g.forward(goal,f)['position_mm'].copy()
            records[f]=dict(success=True,source_joint_delta_deg=angles[f],coordinates_rad=goal[sl],
                requested_position_mm=command,execution_position_mm=actual,
                position_error_mm=float(np.linalg.norm(actual-command)),
                position_error_scope='FK diagnostics of joint commands only; not fingertip IK or glove tracking accuracy',
                projected=bool(not np.allclose(goal[sl],desired[sl])),solver='relative joint rotations + bounded coupled angle mapping')
            i=sl.start+(2 if f=='thumb' else 0)
            side[f]=dict(actual_goal_deg=float(np.degrees(goal[i])),limits_deg=np.degrees([self.g.lower[i],self.g.upper[i]]))
        return dict(goal_rad=goal,desired_joint_rad=desired,fingers=records,angles_deg=angles,
            model_angles_deg=np.degrees(goal),hand=self.hand,retargeting_kind='joint_visual',
            glove_fk_api='g7_joint',glove_project='Forsense G7 SDK',
            side_swing=dict(enabled=True,method='relative joint rotation mapping',fingers=side),
            thumb_avoidance=avoidance,thumb_visual_limits=dict(ip_straight_coordinate_deg=np.degrees(self.thumb_ip_zero),
                ip_flexion_range_deg=np.degrees([0.,self.g.upper[4]-self.thumb_ip_zero]),hyperextension_allowed=False),
            joint_mapping_gains=self.gains,calibrated_to_wearer=bool(self.alignment),
            thumb_root_mapping=root_report,thumb_calibrated_independently=bool(self.thumb_calibration),
            f1_pinch_alignment=pinch_report,
            thumb_flexion_mode=self.thumb_flexion_mode,
            thumb_root_frame_mode=self.thumb_root_frame_mode,
            f1_pinch_profile=self.config.get('f1_pinch_profile'),
            amplitude_mode=self.amplitude_mode,model_mode=self.g.mode,
            thumb_root_method='common palm-frame bone orientation fitted to three native root axes',
            thumb_independent_calibration=bool(self.thumb_calibration),
            calibration_scope='existing wearer joint-zero and range adaptation; visual imitation only',
            pipeline='G7 bone orientations -> parent-relative rotations -> zero/range/coupling mapping -> F1 joint angles',
            root_frame='relative joint frames',hardware_enabled=False,solve_ms=(time.perf_counter()-started)*1000,
            projection_method='joint angle limits/coupling and thumb collision guard; no Cartesian IK')

    def mapping_record(self):
        return dict(kind='joint_visual',hand=self.hand,source='G7 wrist-frame wxyz bone orientations',
            reference_profile=self.config.get('mapping_profile'),thumb_hinge_axes={str(k):v for k,v in self.thumb_axes.items()},
            amplitude_mode=self.amplitude_mode,model_mode=self.g.mode,
            angle_gains=self.gains,thumb_ip_straight_coordinate_deg=np.degrees(self.thumb_ip_zero),
            thumb_flexion_mode=self.thumb_flexion_mode,
            thumb_root_frame_mode=self.thumb_root_frame_mode,
            scope='visual joint imitation; no fingertip targets or length scaling')
