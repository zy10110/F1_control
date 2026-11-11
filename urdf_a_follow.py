"""Angle-only personal retargeting and the shared URDF-A execution step."""
import json
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
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

    def report(self,u):
        self.g.assign(u)
        gap = self.g.clearance()
        return dict(clearance_mm=gap,safe=gap>=self.margin)

    def safe(self,u):
        return self.report(u)['safe']

    def slide(self,u,step):
        """Project the proposed small motion out of nearby thumb hulls."""
        g=self.g
        g.assign(step)
        m,d=g.model,g.data
        a,b=g.pairs.T
        bound=np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b],axis=1)-m.geom_rbound[a]-m.geom_rbound[b]
        rows=[]; errors=[]
        for ga,gb in g.pairs[bound<.0009]:
            points=np.zeros(6)
            distance=mujoco.mj_geomDistance(m,d,ga,gb,.001,points)
            if distance>=.0008 or abs(distance)<1e-10:
                continue
            direction=(points[3:]-points[:3])/distance
            ja=np.zeros((3,m.nv)); jb=np.zeros_like(ja)
            mujoco.mj_jac(m,d,ja,None,points[:3],int(m.geom_bodyid[ga]))
            mujoco.mj_jac(m,d,jb,None,points[3:],int(m.geom_bodyid[gb]))
            rows.append(1000*direction@(jb-ja)@g.C)
            errors.append(.8-1000*distance)
        if rows:
            budget=np.max(np.abs(step-u))
            if budget>1e-9:
                # Solve within the available range on each axis. Clipping an
                # unconstrained correction after the solve can trap the thumb
                # when a required rotation already sits at its hard stop.
                lower=np.maximum(g.lo-step,-budget)
                upper=np.minimum(g.hi-step,budget)
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
    def __init__(self, model, profile, middle_unavailable=None):
        self.g=model
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
        context=record.get('joint_angle_context')
        self.angle_context=None if context is None or self.middle_unavailable else (
            np.asarray(context['centers']),np.asarray(context['coefficients']),
            np.asarray(context['scale']),float(context['width']),float(context['max_delta']))
        self.widths=np.asarray(record['middle_widths'])
        self.spacing=record.get('angle_derived_spacing') if not self.middle_unavailable else None
        self.display_rotation=np.asarray(record['wrist_to_model'])
        self.sweep=Sweep(model,self.middle_unavailable)
        self.u=np.zeros(17)
        if self.middle_unavailable:
            # Use a known, bounded target. Neither the stream nor clearance
            # projection can command these unavailable axes.
            self.u[model.cols['middle']]=np.clip(self.blocks['middle'][1],model.lo[model.cols['middle']],model.hi[model.cols['middle']])
        self.held_middle=self.u[model.cols['middle']].copy()
        self.vel=None
        self.config=load_g7_config()
        self.chains=None
        self.blocked=False
        self.goal=self.u.copy()
        if not self.sweep.safe(self.u):
            raise ValueError('A 版起始姿态未通过间隙检查')

    def raw_angles(self,features):
        raw=np.concatenate([b+W@(features[f]-z) for f,(z,b,W) in self.blocks.items()])
        for blocks in (self.corrections,self.local_corrections):
            for f,parameters in blocks.items():
                raw[self.g.cols[f]]+=angular_adjustment(features[f]-self.blocks[f][0],*parameters)
        if self.angle_context is not None:
            centers,coef,scale,width,cap=self.angle_context
            x=np.concatenate([features[f]-self.blocks[f][0] for f in FINGERS])*scale
            delta=angular_adjustment(x,centers,coef,width)
            raw+=cap*np.tanh(delta/cap)
        return raw

    def command(self,frame):
        if self.middle_unavailable:
            q=np.array(frame['quaternions_wxyz'],dtype=float,copy=True)
            q[9:13]=self.neutral_middle
            frame=dict(frame,quaternions_wxyz=q)
        features,chains=self.source.read(frame)
        self.chains=chains
        raw=self.raw_angles(features)
        base=np.clip(raw,self.g.lo,self.g.hi)
        x=raw[9:11];lo=self.g.lo[9:11];hi=self.g.hi[9:11]
        wl,wh=self.widths[:2],self.widths[2:]
        low=np.where(x<lo+wl,lo+np.maximum(0,x-lo+wl)**2/(4*wl),x)
        base[9:11]=np.where(x>hi-wh,hi-np.maximum(0,hi+wh-x)**2/(4*wh),low)
        # Optional calibrated context uses joint angles only: no live endpoint
        # distances, contact classification, hysteresis or snap-to-pose state.
        if self.spacing:
            from urdf_a_spacing import spacing_goal
            try:
                base=spacing_goal(self.g,base,chains,self.spacing)
            finally:
                self.g.assign(self.u)
        self.goal=base
        if self.middle_unavailable:self.goal[self.g.cols['middle']]=self.held_middle
        return self.goal

    def advance(self,dt):
        if dt<=0:return self.u
        prop=follow_digits(self.u,self.goal,dt,self.u,self.g.slices,self.config,self.vel)
        prop=self.sweep.slide(self.u,prop)
        cap=np.radians(self.config['max_joint_speed_deg_s'])*min(dt,.1)
        prop=self.u+np.clip(prop-self.u,-cap,cap)
        nxt,report=self.sweep.sweep_follow_digits(self.u,prop)
        self.blocked=report['blocked']
        self.vel=(nxt-self.u)/dt
        self.u=nxt
        self.g.assign(nxt)
        return nxt

