"""Angle-prior thumb retargeting with bone directions and relative vectors.

Only the five thumb coordinates may change. Official solved data and the F1
model supply all targets; no wearer fit, contact class, pose seed or new zero
capture is used. Surface points remain nominal model candidates.
"""
import numpy as np
from linear_thumb_assist import LinearThumbAssist, PAIRS
from g7_proportional_mapping import FingerLandmarks, normalized, source_points, BASE
from thumb_orientation_mapping import basis


class HybridThumbAssist(LinearThumbAssist):
    kind='hybrid_thumb'
    def __init__(self, g, config, reference, hand='right'):
        super().__init__(g, config)
        self.hand=hand
        self.land=FingerLandmarks(g)
        self.land.assign(reference)
        rest={f:self.land.points(f) for f in ('thumb',*PAIRS)}
        self.lengths={f:np.linalg.norm(np.diff(p,axis=0),axis=1) for f,p in rest.items()}
        self.target_basis=basis(rest['index'][0]-rest['pinky'][0],
            np.mean([rest[f][-1]-rest[f][0] for f in PAIRS],axis=0))
        self.shape_weight=float(config.get('thumb_hybrid_shape_weight',20.))
        self.relative_weight=float(config.get('thumb_hybrid_relative_weight',1.))
        self.vector_weight=float(config.get('thumb_hybrid_vector_weight',.35))
        self.normal_weight=float(config.get('thumb_hybrid_normal_weight',7.))
        self.damping=np.diag(config.get('thumb_hybrid_angle_prior',[12.,12.,12.,28.,28.]))
        if (not np.isfinite([self.shape_weight,self.relative_weight,self.normal_weight,self.vector_weight]).all()
                or min(self.shape_weight,self.relative_weight,self.normal_weight,self.vector_weight)<0
                or self.damping.shape!=(5,5) or not np.isfinite(self.damping).all()
                or np.min(np.diag(self.damping))<=0):
            raise ValueError('拇指混合重定向参数无效')

    def targets(self,frame):
        source=source_points(frame,self.hand)
        source_basis=basis(source[5]-source[17],source[[5,9,13,17]].mean(axis=0)-source[0])
        rotation=self.target_basis@source_basis.T
        directions=normalized(np.diff(source[1:5],axis=0))@rotation.T
        source_lengths={f:np.linalg.norm(np.diff(source[i:i+4],axis=0),axis=1).sum()
                        for f,i in BASE.items()}
        relative={}
        for f in PAIRS:
            scale=np.sqrt(self.lengths['thumb'].sum()/source_lengths['thumb']
                          *self.lengths[f].sum()/source_lengths[f])
            relative[f]=rotation@(source[4]-source[BASE[f]+3])*scale
        return directions,relative

    def shape_residual(self,u,target):
        self.land.assign(u)
        points,jac=self.land.points('thumb',True)
        segments=np.diff(points,axis=0);dsegments=np.diff(jac,axis=0)[:,:,:5]
        lengths=np.linalg.norm(segments,axis=1)
        directions=segments/lengths[:,None]
        jd=np.array([(np.eye(3)-np.outer(n,n))@j/l
                     for n,j,l in zip(directions,dsegments,lengths)])
        return self.shape_weight*(directions-target).ravel(),self.shape_weight*jd.reshape(9,5)

    def system(self,u,base,target,relative,weights,contact):
        r,j=self.shape_residual(u,target)
        residual=[r,self.damping@(u[:5]-base[:5])]
        jac=[j,self.damping]
        if contact:
            self.g.assign(self.data,u)
            thumb,jt,jnt=self.pose_jac('thumb')
            for f,w in zip(PAIRS,weights):
                if w<1e-12:continue
                other=self.pad.pose(self.data,f);sw=np.sqrt(w)
                # Keep the broad approach task and add the live normalized
                # vector as a secondary shape cue. No saved closed pose is
                # loaded; the pad normal supplies a nominal 1 mm clearance.
                delta=thumb['position_mm']-other['position_mm']-other['normal_world']
                residual.extend([sw*self.relative_weight*delta,
                    sw*self.vector_weight*(delta-relative[f]),
                    sw*self.normal_weight*(thumb['normal_world']+other['normal_world'])])
                jac.extend([sw*self.relative_weight*jt,sw*self.vector_weight*jt,sw*self.normal_weight*jnt])
        return np.concatenate(residual),np.vstack(jac)

    def fit(self,initial,base,target,relative,weights,contact,lo,hi):
        u=initial.copy()
        for _ in range(5):
            r,j=self.system(u,base,target,relative,weights,contact)
            step=np.linalg.solve(j.T@j+np.eye(5)*1e-8,-j.T@r)
            step=np.clip(step,-np.radians(4),np.radians(4))
            if np.max(np.abs(step))<1e-7:break
            score=float(r@r)
            for fraction in (1.,.5,.2):
                proposed=u.copy();proposed[:5]=np.clip(u[:5]+fraction*step,lo,hi)
                rr,_=self.system(proposed,base,target,relative,weights,contact)
                if float(rr@rr)<score-1e-9:
                    u=proposed;break
            else:break
        return u

    def contact_seed(self,base,weights,lo,hi):
        """Retain v4's broad approach geometry as the position-task anchor."""
        u=base.copy()
        for _ in range(5):
            self.g.assign(self.data,u);thumb,jt,jnt=self.pose_jac('thumb')
            rows=[];rhs=[]
            for f,w in zip(PAIRS,weights):
                if w<1e-12:continue
                other=self.pad.pose(self.data,f);sw=np.sqrt(w)
                rows.extend([sw*jt,sw*self.normal_weight*jnt])
                rhs.extend([sw*(thumb['position_mm']-other['position_mm']-other['normal_world']),
                    sw*self.normal_weight*(thumb['normal_world']+other['normal_world'])])
            j=np.vstack(rows+[self.damping]);r=np.concatenate(rhs+[self.damping@(u[:5]-base[:5])])
            step=np.linalg.solve(j.T@j+np.eye(5)*1e-8,-j.T@r)
            u[:5]=np.clip(u[:5]+np.clip(step,-np.radians(4),np.radians(4)),lo,hi)
        return u

    def refine_contact(self,seed,base,target,relative,weights,lo,hi):
        """Improve orientation while preserving the existing approach point.

        A high-priority position residual plus a 0.5 mm trust envelope prevents
        bone fitting from trading a large new gap for a better direction score.
        This anchor is computed from the current frame, never a recorded pose.
        """
        self.g.assign(self.data,seed)
        anchor=self.pad.pose(self.data,'thumb')['position_mm'].copy()
        u=seed.copy()
        def system(value):
            # The position task has already applied the angle prior. During
            # refinement, regularize around that live solution so the new
            # direction term can use the remaining orientation freedom.
            r,j=self.system(value,seed,target,relative,weights,True)
            self.g.assign(self.data,value);pose,jp,_=self.pose_jac('thumb')
            return (np.r_[r,50.*(pose['position_mm']-anchor)],np.vstack([j,50.*jp]),
                    float(np.linalg.norm(pose['position_mm']-anchor)))
        for _ in range(5):
            r,j,_=system(u);step=np.linalg.solve(j.T@j+np.eye(5)*1e-8,-j.T@r)
            step=np.clip(step,-np.radians(3),np.radians(3));score=float(r@r)
            for fraction in (1.,.5,.2):
                proposed=u.copy();proposed[:5]=np.clip(u[:5]+fraction*step,lo,hi)
                rr,_,distance=system(proposed)
                if distance<=.5 and float(rr@rr)<score-1e-9:u=proposed;break
            else:break
        return u

    def apply(self,frame,base,lower,upper):
        # Validate every source-dependent feature before consuming time/state.
        strengths,gaps,starts=self.weights(frame)
        target,relative=self.targets(frame)
        intensity=float(np.max(strengths));total=float(strengths.sum())
        weights=strengths/total if total>1e-12 else np.zeros(4)
        lo=np.maximum(lower[:5],base[:5]-self.cap)
        hi=np.minimum(upper[:5],base[:5]+self.cap)
        shape=self.fit(base,base,target,relative,weights,False,lo,hi)
        contact=(self.refine_contact(self.contact_seed(base,weights,lo,hi),base,target,relative,weights,lo,hi)
                 if total>1e-12 else shape)
        # A broad affine blend avoids a near-contact gain jump. Direction and
        # angle objectives exist for every pose, including ordinary flexion.
        raw=(shape[:5]-base[:5])+intensity*(contact[:5]-shape[:5])
        dt=self.elapsed(frame)
        correction=self.last+np.clip(raw-self.last,-self.speed*dt,self.speed*dt)
        correction=np.clip(correction,lo-base[:5],hi-base[:5])
        result=base.copy();result[:5]+=correction
        self.last=correction.copy()
        self.g.assign(self.data,result);thumb=self.pad.pose(self.data,'thumb')
        self.land.assign(result);directions=normalized(np.diff(self.land.points('thumb'),axis=0))
        errors=np.degrees(np.arccos(np.clip(np.sum(directions*target,axis=1),-1,1)))
        return result,dict(enabled=True,kind='hybrid_thumb',
            method='angle prior + palm-normalized bone directions + continuous relative vectors',
            strengths=dict(zip(PAIRS,strengths)),weights=dict(zip(PAIRS,weights)),
            source_gaps_mm=dict(zip(PAIRS,gaps)),ramp_start_mm=dict(zip(PAIRS,starts)),
            intensity=intensity,dt_s=dt,raw_correction_deg=np.degrees(raw),
            shape_correction_deg=np.degrees(shape[:5]-base[:5]),correction_deg=np.degrees(correction),
            cap_deg=np.degrees(self.cap),correction_speed_deg_s=np.degrees(self.speed),
            thumb_bone_errors_deg=errors,target_relative_vectors_mm=relative,
            four_fingers_unchanged=True,active_pair=None,forced_closure=False,extra_calibration=False,
            command_pad_gaps_mm={f:float(np.linalg.norm(thumb['position_mm']-self.pad.pose(self.data,f)['position_mm'])) for f in PAIRS},
            scope='local thumb-only projection; nominal rolling-center landmarks and unconfirmed surface candidates')
