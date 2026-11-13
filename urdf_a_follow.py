"""Angle-only personal retargeting and the shared URDF-A execution step."""
import json
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from calibrated_angle_mapping import joint_features
from g7_direct_joint import decode_relative_angles
from g7_joint_mapping import rotations, joint_rotations
from g7_teleop import BASE, load_g7_config
from glove_teleop import follow_digits
from hand_kinematics import FINGERS
from thumb_collision import ThumbCollisionGuard


class SourceHand:
    def __init__(self, neutral):
        self.open_rotations = joint_rotations(neutral)
        r = rotations(neutral)
        points = np.asarray(neutral['positions_m'])*1000
        self.roots = {f:r[0].inv().apply(points[i]-points[0]) for f,i in BASE.items()}
        self.bones = {f:r[i:i+3].inv().apply(np.diff(points[i:i+4],axis=0)) for f,i in BASE.items()}

    def read(self, frame):
        angles = decode_relative_angles(frame,'right',self.open_rotations)
        features = {f:joint_features(angles,f) for f in FINGERS}
        r = rotations(frame)
        chains = {}
        for f,i in BASE.items():
            bones = (r[0].inv()*r[i:i+3]).apply(self.bones[f])
            chains[f] = self.roots[f]+np.vstack([np.zeros(3),np.cumsum(bones,axis=0)])
        return features,chains


class Sweep(ThumbCollisionGuard):
    def __init__(self, model, middle_fixed=False):
        self.g = model
        self.active=np.ones(17,dtype=bool)
        if middle_fixed:self.active[model.cols['middle']]=False
        self.margin = .6
        self.step = np.radians(.75)
        self.floors = {}
        self.pinch_rows = {}
        self.overlap_mm = 0.
        self.long_surfaces = None

    def contact_floors(self, u, weights):
        requested = {k:self.margin*(1.-w) for k,w in weights.items()}
        for row,weight in self.pinch_rows.items():
            requested[row]=min(requested.get(row,self.margin),self.margin-(self.margin+self.overlap_mm)*weight)
        self.g.assign(u)
        # Restore the old margin progressively on release. A pair that starts
        # below it may separate, but cannot use that recovery to move deeper.
        for k,old in self.floors.items():
            want=requested.get(k,self.margin)
            if old < want:
                a,b=self.g.pairs[k]
                gap=1000.*mujoco.mj_geomDistance(self.g.model,self.g.data,a,b,.001,None)
                requested[k]=min(want,max(old,gap-1.e-7))
        self.floors={k:v for k,v in requested.items() if v<self.margin}
        self.step=np.radians(.05 if self.floors else .75)

    def report(self,u):
        self.g.assign(u)
        long_safe=self.long_surfaces is None or self.long_surfaces.clear()
        if self.floors:
            g=self.g
            a,b=g.pairs.T
            lower=np.linalg.norm(g.data.geom_xpos[a]-g.data.geom_xpos[b],axis=1)-g.model.geom_rbound[a]-g.model.geom_rbound[b]
            gap=10.;safe=True
            for k in np.flatnonzero(lower<.01):
                d=1000.*mujoco.mj_geomDistance(g.model,g.data,a[k],b[k],.01,None)
                gap=min(gap,d)
                safe &= d>=self.floors.get(k,self.margin)
            return dict(clearance_mm=gap,safe=bool(safe and long_safe))
        gap = self.g.clearance()
        return dict(clearance_mm=gap,safe=gap>=self.margin and long_safe)

    def safe(self,u):
        # Intermediate samples only need a pass/fail decision. Most hulls
        # are outside the active clearance margin even when fingers cluster.
        g=self.g
        g.assign(u)
        a,b=g.pairs.T
        lower=np.linalg.norm(g.data.geom_xpos[a]-g.data.geom_xpos[b],axis=1)-g.model.geom_rbound[a]-g.model.geom_rbound[b]
        for k in np.flatnonzero(lower<=self.margin*.001):
            distance=mujoco.mj_geomDistance(g.model,g.data,a[k],b[k],.01,None)
            if distance*1000.<self.floors.get(k,self.margin):
                return False
        return self.long_surfaces is None or self.long_surfaces.clear()

    def release_step(self,current,weights,cap):
        """Lift off a retired contact before following a new tangential goal."""
        g=self.g
        rows=[];rhs=[]
        g.assign(current)
        for k in self.floors:
            if weights.get(k,0.)>.001:continue
            a,b=g.pairs[k];points=np.zeros(6)
            gap=1000.*mujoco.mj_geomDistance(g.model,g.data,a,b,.001,points)
            if abs(gap)<1.e-8 or gap>=self.margin:continue
            ja=np.zeros((3,g.model.nv));jb=np.zeros_like(ja)
            mujoco.mj_jac(g.model,g.data,ja,None,points[:3],g.model.geom_bodyid[a])
            mujoco.mj_jac(g.model,g.data,jb,None,points[3:],g.model.geom_bodyid[b])
            rows.append(((points[3:]-points[:3])*1.e6/gap)@(jb-ja)@g.C)
            rhs.append(.85-gap)
        if not rows:return None
        A=np.vstack((rows,2.*np.eye(17)))
        step=lsq_linear(A[:,self.active],np.r_[rhs,np.zeros(17)],
                        bounds=(np.maximum(g.lo-current,-cap)[self.active],
                                np.minimum(g.hi-current,cap)[self.active]),method='bvls').x
        command=current.copy()
        command[self.active]+=step
        return command

    def approach_step(self, current, proposed, max_step):
        """Allow intervening fingers to yield before the thumb reaches a hull."""
        from urdf_a_spacing import clearance_linearization
        g = self.g
        g.assign(current)
        rows, shortfall = clearance_linearization(g, 3.)
        move = proposed-current
        required = shortfall-(3.-.85)
        approaching = rows@move < np.minimum(0.,required)-.001
        if not approaching.any():
            return proposed
        rows = rows[approaching]
        # Retain thumb travel while sharing the normal correction with the
        # approaching digit. The nonlinear sweep still checks the result.
        weights = np.r_[np.full(5,2.5),np.ones(12)]
        matrix = np.vstack((4.*rows,np.diag(weights)))
        rhs = np.r_[4.*required[approaching],weights*move]
        bounds = (np.maximum(g.lo-current,-max_step),np.minimum(g.hi-current,max_step))
        return current+lsq_linear(matrix,rhs,bounds=bounds,method='bvls').x

    def slide(self,u,step,max_step=None):
        """Project the proposed small motion out of nearby thumb hulls."""
        g=self.g
        g.assign(step)
        m,d=g.model,g.data
        a,b=g.pairs.T
        bound=np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b],axis=1)-m.geom_rbound[a]-m.geom_rbound[b]
        rows=[]; errors=[]
        for k in np.flatnonzero(bound<.0009):
            ga,gb=g.pairs[k]
            floor=self.floors.get(k,self.margin)
            reserve=floor+(.06 if floor<0. else floor/3.)
            points=np.zeros(6)
            distance=mujoco.mj_geomDistance(m,d,ga,gb,.001,points)
            if distance*1000>=reserve or abs(distance)<1e-10:
                continue
            direction=(points[3:]-points[:3])/distance
            ja=np.zeros((3,m.nv)); jb=np.zeros_like(ja)
            mujoco.mj_jac(m,d,ja,None,points[:3],int(m.geom_bodyid[ga]))
            mujoco.mj_jac(m,d,jb,None,points[3:],int(m.geom_bodyid[gb]))
            rows.append(1000*direction@(jb-ja)@g.C)
            errors.append(reserve-1000*distance)
        if self.long_surfaces is not None:
            from urdf_a_spacing import clearance_linearization
            extra,shortfall=clearance_linearization(g,.12,self.long_surfaces.pairs,self.long_surfaces.offsets_mm)
            rows.extend(4.*extra)
            errors.extend(4.*shortfall)
        if rows:
            budget=np.max(np.abs(step-u))
            if budget>1e-9:
                # Solve within the available range on each axis. Clipping an
                # unconstrained correction after the solve can trap the thumb
                # when a required rotation already sits at its hard stop.
                lower=np.maximum(g.lo-step,-budget)
                upper=np.minimum(g.hi-step,budget)
                if max_step is not None:
                    # A speed clamp after projection can undo the clearance
                    # correction. Allocate it inside the reachable box.
                    lower=np.maximum(lower,u-step-max_step)
                    upper=np.minimum(upper,u-step+max_step)
                regularizer=np.diag([1.5]*5+[8.]*12)
                result=lsq_linear(np.vstack((rows,regularizer))[:,self.active],np.r_[errors,np.zeros(17)],
                                  bounds=(lower[self.active],upper[self.active]),method='bvls',max_iter=25)
                step=step.copy()
                step[self.active]+=result.x
                step=np.clip(step,g.lo,g.hi)
        return step


def angular_adjustment(x, centers, coefficients, width):
    """Smooth per-digit correction fitted to residuals in joint-angle space."""
    distances=np.sum(((centers-x)/width)**2,axis=1)
    return np.exp(-.5*distances)@coefficients


class Follower:
    def __init__(self, model, profile, middle_unavailable=None, *, overlap_limit_mm=None):
        self.g=model
        self.overlap_stop=None
        if overlap_limit_mm is not None:
            from urdf_a_overlap import OverlapStop
            self.overlap_stop=OverlapStop(model,overlap_limit_mm)
        payload=Path(profile).read_bytes()
        self.profile_sha256=hashlib.sha256(payload).hexdigest()
        record=json.loads(payload.decode('utf8'))
        if record.get('kind') not in ('urdf_a_angles_v1','urdf_a_natural_v2') or record.get('geometry_id')!=model.geometry_id:
            raise ValueError('档案与 A 版几何、耦合或指腹定义不匹配')
        self.source=SourceHand(record['neutral'])
        self.middle_unavailable=record.get('middle_unavailable',False) if middle_unavailable is None else middle_unavailable
        self.neutral_middle=np.asarray(record['neutral']['quaternions_wxyz'])[9:13]
        self.blocks={f:tuple(np.asarray(record['blocks'][f][k]) for k in ('zero','bias','matrix')) for f in FINGERS}
        self.corrections={f:(np.asarray(b['centers']),np.asarray(b['coefficients']),float(b['width']))
                          for f,b in record.get('angular_corrections',{}).items()}
        self.local_corrections={f:(np.asarray(b['centers']),np.asarray(b['coefficients']),float(b['width']))
                                for f,b in record.get('local_angular_corrections',{}).items()}
        self.pinch_corrections={f:(np.asarray(b['centers']),np.asarray(b['coefficients']),float(b['width']))
                                for f,b in record.get('pinch_angle_updates',{}).items()}
        context=record.get('joint_angle_context')
        self.official_corrections={f:(np.asarray(b['centers']),np.asarray(b['coefficients']),float(b['width']))
                                   for f,b in record.get('official_pose_updates',{}).items()}
        self.angle_context=None if context is None or self.middle_unavailable else (
            np.asarray(context['centers']),np.asarray(context['coefficients']),
            np.asarray(context['scale']),float(context['width']),float(context['max_delta']))
        self.widths=np.asarray(record['middle_widths'])
        self.spacing=record.get('angle_derived_spacing') if not self.middle_unavailable else None
        self.segment_alignment=None
        if record.get('segment_alignment') and not self.middle_unavailable:
            from urdf_a_segments import SegmentAlignment
            self.segment_alignment=SegmentAlignment(model,record['segment_alignment'])
        self.display_rotation=np.asarray(record['wrist_to_model'])
        self.joint_touch=None
        self.touch=None
        if record.get('joint_surface_touch') and not self.middle_unavailable:
            from urdf_a_joint_touch import JointTouch
            self.joint_touch=JointTouch(model,record['joint_surface_touch'])
        self.sweep=Sweep(model,self.middle_unavailable)
        # The right-hand G7 profile was validated with compliant fingertip
        # pinches. Mirrored profiles retain their fitted clearances by default.
        if record.get('input_hand','right')=='right' and self.joint_touch:
            self.sweep.overlap_mm=.30
        self.u=np.zeros(17)
        if self.middle_unavailable:
            # Use a known, bounded target. Neither the stream nor clearance
            # projection can command these unavailable axes.
            self.u[model.cols['middle']]=np.clip(self.blocks['middle'][1],model.lo[model.cols['middle']],model.hi[model.cols['middle']])
        self.held_middle=self.u[model.cols['middle']].copy()
        self.vel=None
        self.config=load_g7_config()
        self.response = record.get('fine_motion_response')
        if self.response is not None:
            transition = float(self.response['transition_deg'])
            ratio = float(self.response['max_rate_ratio'])
            if not (np.isfinite(transition) and transition > 0 and np.isfinite(ratio) and ratio >= 1):
                raise ValueError('精细跟随响应参数无效')
            self.response=dict(transition_deg=transition,max_rate_ratio=ratio)
        if self.spacing and 'aperture_range_mm' in self.spacing:
            radius = float(self.spacing['aperture_range_mm'])
            if not (np.isfinite(radius) and radius > 0):
                raise ValueError('精细跟随间距范围无效')
            self.spacing['aperture_range_mm']=radius
        self.chains=None
        self.contact_sample=None
        self.contact_velocity=None
        self.thumb_proximity=0.
        self.approach_sample=None
        self.approach_weight=0.
        self.goal_stamp=None
        self.goal_velocity=None
        from urdf_a_pairs import PairHold, LongFingerSurfaces
        self.pair_hold=PairHold(model)
        self.pad_rows={f:int(np.flatnonzero(np.all(model.pairs==[model.geoms['thumb'],model.geoms[f]],axis=1))[0])
                       for f in FINGERS[1:]}
        if self.joint_touch and self.response:
            self.sweep.long_surfaces=LongFingerSurfaces(model)
            self.sweep.long_surfaces.overlap_mm=self.sweep.overlap_mm
        self.blocked=False
        self.goal=self.u.copy()
        if not self.sweep.safe(self.u):
            raise ValueError('A 版起始姿态未通过间隙检查')

    def raw_angles(self,features):
        raw=np.concatenate([b+W@(features[f]-z) for f,(z,b,W) in self.blocks.items()])
        for blocks in (self.corrections,self.local_corrections,self.pinch_corrections,self.official_corrections):
            for f,parameters in blocks.items():
                raw[self.g.cols[f]]+=angular_adjustment(features[f]-self.blocks[f][0],*parameters)
        if self.angle_context is not None:
            centers,coef,scale,width,cap=self.angle_context
            x=np.concatenate([features[f]-self.blocks[f][0] for f in FINGERS])*scale
            delta=angular_adjustment(x,centers,coef,width)
            raw+=cap*np.tanh(delta/cap)
        return raw

    def contact_frame(self,frame):
        """Suppress small, rapid changes near contact in the wrist coordinate frame."""
        self.thumb_proximity=0.
        stamp=frame.get('timestamp_ns')
        if stamp is None or 'positions_m' not in frame:
            self.contact_sample=None
            self.pair_hold.reset()
            return frame
        points=np.asarray(frame['positions_m'],float)
        if points.shape!=(21,3) or not np.isfinite(points).all():
            raise ValueError('官方指节位置无效')
        r=rotations(frame)
        wrist=r[0]
        hand=wrist.inv().apply(points[1:]-points[0]).reshape(5,4,3)*1000.
        bones=np.diff(hand,axis=1)
        lengths=np.linalg.norm(bones,axis=2)
        if np.min(lengths)<1.:
            raise ValueError('官方指节长度无效')
        from urdf_a_pairs import PAIRS,pair_distances
        distances=pair_distances(hand)
        radius=max(22.,self.joint_touch.settings.get('index_radius_mm',0.))
        strength=np.clip((np.where(PAIRS[:,0]==0,radius,12.)+5.-distances)/5.,0.,1.)
        proximity=np.zeros(5)
        np.maximum.at(proximity,PAIRS[:,0],strength)
        np.maximum.at(proximity,PAIRS[:,1],strength)
        self.thumb_proximity=float(proximity[0])
        if not proximity.any():
            self.contact_sample=None
            self.pair_hold.reset()
            return frame
        observed=wrist.inv()*r[1:]
        if self.overlap_stop is None:
            self.pair_hold.observe(stamp,hand,observed,distances,self.u)
        previous=self.contact_sample
        filtered=observed
        filtered_velocity=None
        if previous is not None:
            last_stamp,last_hand,last_rot,last_filtered=previous
            dt=(stamp-last_stamp)*1.e-9
            if dt==0. and np.array_equal(hand,last_hand) and np.array_equal(observed.as_quat(),last_rot.as_quat()):
                filtered=last_filtered
                filtered_velocity=self.contact_velocity
            elif 0.<dt<=.1:
                # Respond independently to deliberate motion of each digit.
                # Include axial twist even when all four landmarks stay still.
                linear=hand-last_hand
                angular=(last_rot.inv()*observed).as_rotvec().reshape(5,4,3)
                velocity=np.concatenate((linear,20.*angular),axis=1)/dt
                speed_scale,decay=32.,.050
                if self.sweep.overlap_mm:
                    # A single reversed sample can have high speed in a held
                    # pinch. Use signed velocity over time for the adaptive
                    # cutoff, including axial twist and intentional sliding.
                    previous_velocity=self.contact_velocity if self.contact_velocity is not None else np.zeros_like(velocity)
                    filtered_velocity=previous_velocity+(-np.expm1(-dt/.075))*(velocity-previous_velocity)
                    velocity=filtered_velocity
                    speed_scale,decay=42.,.070
                speed=np.linalg.norm(velocity,axis=2).max(axis=1)
                fast=np.clip((speed-8.)/speed_scale,0.,1.)
                memory=np.exp(-dt/decay)*proximity*(1.-fast*fast*(3.-2.*fast))
                fraction=np.repeat(1.-memory,4)
                delta=(last_filtered.inv()*observed).as_rotvec()
                filtered=last_filtered*Rotation.from_rotvec(fraction[:,None]*delta)
        self.contact_sample=(stamp,hand,observed,filtered)
        self.contact_velocity=filtered_velocity
        if filtered is observed:
            return frame
        # Rotate the measured bone vectors with their filtered parent poses.
        # Rebuilding the chain keeps points and quaternions in agreement.
        parents=np.arange(20).reshape(5,4)[:,:3].ravel()
        v=(filtered[parents]*observed[parents].inv()).apply(bones.reshape(-1,3)).reshape(5,3,3)
        local=hand.copy()
        local[:,1:]=local[:,:1]+np.cumsum(v,axis=1)
        pos=np.vstack((points[0],points[0]+wrist.apply(local.reshape(20,3))*.001))
        q=Rotation.concatenate([wrist,wrist*filtered]).as_quat()[:,[3,0,1,2]]
        return dict(frame,quaternions_wxyz=q,positions_m=pos)

    def pinch_goal(self, nominal, observed):
        """Permit a small indentation only for a demonstrated fingertip pinch."""
        sweep=self.sweep
        sweep.pinch_rows={}
        if not sweep.overlap_mm:
            return nominal
        gaps=np.array([np.linalg.norm(observed[f][-1]-observed['thumb'][-1]) for f in FINGERS[1:]])
        weights=np.clip((13.-gaps)/7.,0.,1.)
        if self.overlap_stop:
            # The recorded little-finger pinch retains ~10 mm between the
            # SDK landmarks even when the two physical fingertips touch.
            weights[-1]=np.clip((17.-gaps[-1])/7.,0.,1.)
        weights=weights**2*(3.-2.*weights)
        if self.touch is not None:
            # Crossfade the two objectives when tipping onto a phalanx.
            shaft=np.clip(self.touch.weight/.3,0.,1.)
            weights*=1.-shaft*shaft*(3.-2.*shaft)
        if not weights.any():
            return nominal
        g=self.g
        q=nominal.copy()
        for iteration in range(3):
            g.assign(q)
            jac=[];errors=[]
            for f,w in zip(FINGERS[1:],weights):
                if not w:continue
                row=self.pad_rows[f]
                a,b=g.pairs[row];points=np.zeros(6)
                gap=1000.*mujoco.mj_geomDistance(g.model,g.data,a,b,.008,points)
                reach_mm=(14. if f=='pinky' else 7.) if self.overlap_stop else 4.
                if gap>=reach_mm:continue
                # Fade out the correction near the reach boundary.
                reach=np.clip((reach_mm-gap)/2.,0.,1.)
                if iteration==0:
                    sweep.pinch_rows[row]=w*reach*reach*(3.-2.*reach)
                strength=sweep.pinch_rows.get(row,0.)
                # The pad can settle anywhere in a narrow compression band.
                # There is no corrective normal displacement inside the band.
                target=np.clip(gap,-.8*sweep.overlap_mm,-.35*sweep.overlap_mm)
                if self.overlap_stop:
                    # Bring demonstrated, reachable pinches together. Once
                    # they overlap, only the execution depth stop applies.
                    if gap<=0.:continue
                    target=-.20
                if abs(gap)<1.e-6 or gap==target or not strength:continue
                jac.append(strength*self.joint_touch.distance_row(row,gap,points,np.zeros(3)))
                errors.append(strength*(target-gap))
            if not jac:break
            ridge=7.
            cap=np.radians(4.+5.*weights[-1] if self.overlap_stop else 4.)
            lower=np.maximum(g.lo,nominal-cap)-q
            upper=np.minimum(g.hi,nominal+cap)-q
            step=lsq_linear(np.vstack([jac,ridge*np.eye(17)]),np.r_[errors,ridge*(nominal-q)],
                            bounds=(lower,upper),method='bvls').x
            q+=.8*step
            if np.max(abs(step))<1.e-5:break
        return q

    def approach_vectors(self,frame,chains):
        """Track the speed of official relative tip motion in the F1 palm frame."""
        stamp=frame.get('timestamp_ns')
        vectors=np.array([chains[f][-1]-chains['thumb'][-1] for f in FINGERS[1:]])@self.display_rotation.T
        old=self.approach_sample
        self.approach_sample=(stamp,vectors)
        if old is None or stamp is None or old[0] is None:
            self.approach_weight=0.
        elif stamp!=old[0]:
            dt=(stamp-old[0])*1.e-9
            if dt<=0. or dt>.12:
                self.approach_weight=0.
            else:
                speed=np.max(np.linalg.norm(vectors-old[1],axis=1))/dt
                t=np.clip((speed-60.)/100.,0.,1.)
                gain=t*t*(3.-2.*t)
                tau=.018 if gain>self.approach_weight else .085
                self.approach_weight+=-np.expm1(-dt/tau)*(gain-self.approach_weight)
        return vectors

    def command(self,frame):
        if self.joint_touch and self.response:
            frame=self.contact_frame(frame)
        if self.middle_unavailable:
            q=np.array(frame['quaternions_wxyz'],dtype=float,copy=True)
            q[9:13]=self.neutral_middle
            frame=dict(frame,quaternions_wxyz=q)
        features,chains=self.source.read(frame)
        self.chains=chains
        if self.contact_sample is not None:
            # The reference skeleton in the viewer stays on the measured input.
            self.chains=dict(zip(FINGERS,self.contact_sample[1]))
        raw=self.raw_angles(features)
        base=np.clip(raw,self.g.lo,self.g.hi)
        x=raw[9:11];lo=self.g.lo[9:11];hi=self.g.hi[9:11]
        wl,wh=self.widths[:2],self.widths[2:]
        low=np.where(x<lo+wl,lo+np.maximum(0,x-lo+wl)**2/(4*wl),x)
        base[9:11]=np.where(x>hi-wh,hi-np.maximum(0,hi+wh-x)**2/(4*wh),low)
        # Optional calibrated context uses joint angles only: no live endpoint
        # distances, contact classification, hysteresis or snap-to-pose state.
        if self.spacing or self.segment_alignment or self.joint_touch:
            from urdf_a_spacing import spacing_goal
            try:
                if self.segment_alignment:
                    base=self.segment_alignment.solve(base,chains,self.display_rotation)
                bone_rotations = None
                axis = None
                if self.joint_touch:
                    observed=chains
                    if 'positions_m' in frame:
                        points=np.asarray(frame['positions_m'],float)
                        if points.shape!=(21,3) or not np.isfinite(points).all():
                            raise ValueError('官方指节位置无效')
                        bone_rotations = rotations(frame)
                        points=bone_rotations[0].inv().apply(points-points[0])*1000.
                        if np.min(np.linalg.norm(np.diff(points[1:].reshape(5,4,3),axis=1),axis=2))<1.:
                            raise ValueError('官方指节长度无效')
                        observed={f:points[i:i+4] for f,i in BASE.items()}
                    previous_touch=self.touch
                    self.touch=self.joint_touch.observe(observed,self.display_rotation)
                    if (self.overlap_stop is None and previous_touch and self.touch and self.touch.finger==previous_touch.finger
                            and self.pair_hold.locked[0] and self.pair_hold.locked[FINGERS.index(self.touch.finger)]):
                        self.touch=previous_touch
                if self.segment_alignment and self.segment_alignment.thumb_bend_plane:
                    if bone_rotations is None:
                        bone_rotations = rotations(frame)
                    proximal = bone_rotations[0].inv() * bone_rotations[BASE['thumb']+1]
                    axis = self.display_rotation @ proximal.apply([0., 0., -1.])
                    base = self.segment_alignment.align_thumb(base, chains, self.display_rotation, axis,
                                                              self.touch.weight if self.touch else 0.)
                if self.spacing:
                    vectors=self.approach_vectors(frame,observed) if self.joint_touch and self.response else None
                    gain=self.approach_weight*(1.-self.touch.weight) if self.touch else self.approach_weight
                    base=spacing_goal(self.g,base,chains,self.spacing,source_vectors=vectors,motion_weight=gain,
                                      avoid_shells=self.overlap_stop is None)
                if self.touch is not None:
                    base=self.joint_touch.solve(base,self.touch,axis,landmark_only=self.overlap_stop is not None)
                if self.sweep.long_surfaces is not None:
                    hand=self.contact_sample[1] if self.contact_sample is not None else None
                    base=self.pinch_goal(base,observed)
                    base=self.sweep.long_surfaces.goal(base,hand,self.u,attract_only=self.overlap_stop is not None)
                    if self.overlap_stop is None:
                        self.pair_hold.hold(base,self.u)
                    # Nearby contacts can yield several quite different IK
                    # solutions on adjacent frames. Smooth the thumb target
                    # before the geometric guard, using the source clock.
                    stamp=frame.get('timestamp_ns')
                    if self.sweep.overlap_mm and self.goal_stamp is not None and stamp is not None:
                        dt=(stamp-self.goal_stamp)*1.e-9
                        if 0.<dt<=.12 and self.thumb_proximity:
                            thumb=self.g.slices['thumb']
                            memory=self.thumb_proximity*np.exp(-dt/(.068 if self.overlap_stop else .046))
                            base[thumb]+=memory*(self.goal[thumb]-base[thumb])
            finally:
                self.g.assign(self.u)
        stamp=frame.get('timestamp_ns')
        self.goal_velocity=None
        if self.approach_weight and stamp is not None and self.goal_stamp is not None:
            dt=(stamp-self.goal_stamp)*1.e-9
            if 0.<dt<=.12:
                speed=np.radians(self.config['max_joint_speed_deg_s'])
                self.goal_velocity=self.approach_weight*np.clip((base-self.goal)/dt,-speed,speed)
        self.goal_stamp=stamp
        self.goal=base
        if self.middle_unavailable:self.goal[self.g.cols['middle']]=self.held_middle
        return self.goal

    def control_step(self, dt, wrist, fingers_fresh=True):
        """Execute a bounded wrist and the latest safe finger pose in one tick."""
        if fingers_fresh:
            self.advance(dt)
        self.g.set_control_pose(self.u,wrist)
        return self.g.control_data.qpos.copy()

    def advance(self,dt):
        if dt<=0:return self.u
        prop=follow_digits(self.u,self.goal,dt,self.u,self.g.slices,self.config,self.vel,
                           response=self.response,goal_velocity=self.goal_velocity)
        self.goal_velocity=None
        if self.overlap_stop is not None:
            nxt=self.overlap_stop.limit(self.u,prop)
            self.blocked=bool(self.overlap_stop.stopped.any())
            self.vel=(nxt-self.u)/dt
            self.u=nxt
            self.g.assign(nxt)
            return nxt
        cap=np.radians(self.config['max_joint_speed_deg_s'])*min(dt,.1)
        # On a fast reversal the straight target can go through a fingertip
        # hull. Recompute the local sliding direction along large increments,
        # keeping each increment inside its share of the frame speed budget.
        delta=prop-self.u
        steps=max(1,int(np.ceil(np.max(np.abs(delta))/np.radians(2.)))) if self.response else 1
        nxt=self.u
        self.blocked=False
        path_floors={}
        waypoints=[]
        bent_path=steps>1
        for _ in range(steps):
            departure=None
            if self.touch is not None or self.sweep.floors or self.sweep.pinch_rows:
                self.g.assign(nxt)
                weights=self.joint_touch.contact_rows(self.touch) if self.touch else {}
                self.sweep.contact_floors(nxt,weights)
                if self.sweep.floors:
                    departure=self.sweep.release_step(nxt,{**weights,**self.sweep.pinch_rows},cap/steps)
                for k,floor in self.sweep.floors.items():
                    path_floors[k]=min(path_floors.get(k,self.sweep.margin),floor)
            command=departure if departure is not None else np.clip(nxt+delta/steps,self.g.lo,self.g.hi)
            if self.sweep.long_surfaces is not None:
                command=self.sweep.long_surfaces.tangent_step(nxt,command,cap/steps)
            if self.sweep.long_surfaces is not None and self.touch is None and not self.sweep.floors:
                command=self.sweep.approach_step(nxt,command,cap/steps)
            command=self.sweep.slide(nxt,command,cap/steps if self.response else None)
            command=nxt+np.clip(command-nxt,-cap/steps,cap/steps)
            if departure is None and self.touch is not None and self.touch.weight>=1.-1.e-8:
                command=self.joint_touch.project(command,self.touch,np.maximum(self.g.lo,nxt-cap/steps),
                                                 np.minimum(self.g.hi,nxt+cap/steps))
            nxt,report=self.sweep.sweep_follow_digits(nxt,command)
            self.blocked |= report['blocked']
            bent_path |= report.get('route')!='simultaneous'
            waypoints.append(nxt)
        if bent_path and (path_floors or self.sweep.long_surfaces is not None):
            # The guard above follows the piecewise sliding route. A viewer
            # interpolating tick endpoints takes a chord; validate it too.
            end_floors=self.sweep.floors
            self.sweep.floors=path_floors
            self.sweep.step=np.radians(.05 if path_floors else .5)
            requested=nxt
            nxt,report=self.sweep.sweep(self.u,requested)
            if report['blocked']:
                # Move along the verified curved route. Repeatedly taking a
                # straight prefix at the blocking normal can pin the hand.
                for waypoint in reversed(waypoints[:-1]):
                    candidate,check=self.sweep.sweep(self.u,waypoint)
                    if not check['blocked']:
                        closer=np.linalg.norm(candidate-requested)<np.linalg.norm(nxt-requested)
                        stalled=np.linalg.norm(nxt-self.u)<.2*np.linalg.norm(requested-self.u)
                        if closer or stalled:
                            nxt=candidate
                        break
            self.blocked |= report['blocked']
            if not report['blocked']:
                self.sweep.floors=end_floors
            self.sweep.step=np.radians(.05 if self.sweep.floors else .75)
        if self.sweep.long_surfaces is not None:
            requested=nxt
            nxt,limited=self.sweep.long_surfaces.limit_chord(self.u,requested)
            if limited:
                # The displayed chord may differ from the internal routed
                # steps. Check the accepted prefix against the thumb too.
                previous_step=self.sweep.step
                self.sweep.step=np.radians(.05)
                try:
                    nxt,_=self.sweep.sweep(self.u,nxt)
                    slide=self.sweep.long_surfaces.tangent_step(self.u,requested,cap,recover=True)
                    slide,_=self.sweep.long_surfaces.limit_chord(self.u,slide)
                    slide,_=self.sweep.sweep(self.u,slide)
                    stalled=np.linalg.norm(nxt-self.u)<.1*np.linalg.norm(requested-self.u)
                    if (stalled and np.linalg.norm(slide-self.u)>1.e-7) or np.linalg.norm(requested-slide)<np.linalg.norm(requested-nxt):
                        nxt=slide
                finally:self.sweep.step=previous_step
                self.blocked=True
        self.vel=(nxt-self.u)/dt
        self.u=nxt
        self.g.assign(nxt)
        return nxt


