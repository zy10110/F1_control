"""Pair-local settling in wrist coordinates and long-finger shell distances."""
from itertools import combinations

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from hand_kinematics import FINGERS
from urdf_a_spacing import clearance_linearization


PAIRS = np.array(list(combinations(range(5), 2)))


def pair_distances(hand):
    """Tip to opposing shaft, plus side contact between middle/distal bones."""
    a,b = PAIRS.T
    def tip_to_shaft(tips, fingers):
        bones = np.diff(fingers, axis=1)
        delta = tips[:,None,:] - fingers[:,:-1]
        t = np.clip(np.sum(delta*bones,axis=2)/np.sum(bones*bones,axis=2),0.,1.)
        return np.linalg.norm(delta-t[:,:,None]*bones,axis=2).min(axis=1)
    distances=np.minimum(tip_to_shaft(hand[a,-1],hand[b]),tip_to_shaft(hand[b,-1],hand[a]))
    # The nearest points can both be in the interior of a bone. Exclude
    # palm roots, which stay close to their neighbours even in an open hand.
    p=hand[a,1:-1,None];q=hand[b,None,1:-1]
    u=np.diff(hand[a,1:],axis=1)[:,:,None];v=np.diff(hand[b,1:],axis=1)[:,None,:]
    w=p-q
    uu=np.sum(u*u,axis=-1);vv=np.sum(v*v,axis=-1);uv=np.sum(u*v,axis=-1)
    uw=np.sum(u*w,axis=-1);vw=np.sum(v*w,axis=-1)
    det=uu*vv-uv**2
    s=np.divide(uv*vw-vv*uw,det,out=np.zeros_like(det),where=det>1.e-8)
    s=np.clip(s,0.,1.)
    t=np.clip((uv*s+vw)/vv,0.,1.)
    s=np.clip((uv*t-uw)/uu,0.,1.)
    side=np.linalg.norm(w+s[:,:,:,None]*u-t[:,:,:,None]*v,axis=-1).min(axis=(1,2))
    return np.minimum(distances,side)


class PairHold:
    """Rest a pair only after both the source and executed joints have settled."""
    def __init__(self,g):
        self.cols=[g.cols[f] for f in FINGERS]
        self.reset()

    def reset(self):
        self.stamp=None
        self.links=np.zeros((5,5),bool)
        self.locked=np.zeros(5,bool)
        self.keep=np.zeros(5,bool)
        self.memory=np.zeros(5)
        self.last_goal=None

    def observe(self,stamp,hand,rot,dist,pose):
        a,b=PAIRS.T
        limit=np.where(a==0,16.,12.)
        # Two non-collinear markers register twists about a stationary bone.
        matrix=rot.as_matrix().reshape(5,4,3,3)
        marks=np.concatenate((hand,24.*matrix[:,:,:,0],24.*matrix[:,:,:,2]),axis=1)
        if stamp==self.stamp and np.allclose(marks,self.last_marks,atol=1.e-9,rtol=0.):
            return
        dt=0. if self.stamp is None else (stamp-self.stamp)*1.e-9
        restart=self.stamp is None or dt<=0. or dt>.12
        if restart:self.reset()
        active=dist < limit+3.*self.links[a,b]
        links=np.zeros((5,5),bool)
        links[a,b]=links[b,a]=active
        if restart:
            self.mean=marks.copy();self.anchor=marks.copy();self.robot=pose.copy()
            self.since=np.full(5,stamp,dtype=np.int64)
            self.settled=self.since.copy()
            self.moving_until=self.since.copy()
        alpha=-np.expm1(-max(dt,0.)/.045)
        change=alpha*(marks-self.mean)
        self.mean+=change
        def excursion(x):
            return np.linalg.norm(x-self.anchor,axis=2).max(axis=1)
        moved=(excursion(self.mean)>.38)|(excursion(marks)>1.25)
        self.moving_until[moved]=stamp+50_000_000
        changed=moved | (links!=self.links).any(axis=1) | ~links.any(axis=1)
        self.anchor[changed]=self.mean[changed]
        self.since[changed]=stamp
        for i,c in enumerate(self.cols):
            if changed[i] or np.max(np.abs(pose[c]-self.robot[c]))>np.radians(.35):
                self.settled[i]=stamp
                self.robot[c]=pose[c]
        ready=((stamp-self.since)>=100_000_000)&((stamp-self.settled)>=75_000_000)
        self.locked=ready & links.any(axis=1) & ~(links & ~ready[None,:]).any(axis=1)
        stable=(~changed)&(stamp>=self.moving_until)
        self.keep=stable & links.any(axis=1) & ~(links & ~stable[None,:]).any(axis=1)
        # Blend out of a rest on a slow deliberate change. A fast gesture
        # or a different partner receives its current goal immediately.
        speed=np.linalg.norm(change,axis=2).max(axis=1)/max(dt,.001)
        speed=np.maximum(speed,(links*speed[None,:]).max(axis=1))
        fast=np.clip((speed-6.)/22.,0.,1.)
        same=links.any(axis=1) & (links==self.links).all(axis=1)
        self.memory=np.exp(-max(dt,0.)/.065)*(1.-fast)*same if not restart else np.zeros(5)
        self.links=links
        self.stamp=stamp
        self.last_marks=marks

    def hold(self,goal,current):
        for i,c in enumerate(self.cols):
            if self.locked[i] and self.last_goal is not None:
                # A collision elsewhere can pause this digit mid-approach.
                # Keep chasing the settled target until the actuator arrives.
                self.locked[i]=np.max(abs(current[c]-self.last_goal[c]))<np.radians(.65)
            if self.locked[i]:
                goal[c]=current[c]
            elif self.keep[i] and self.last_goal is not None:
                goal[c]=self.last_goal[c]
            elif self.last_goal is not None:
                goal[c]+=self.memory[i]*(self.last_goal[c]-goal[c])
        self.last_goal=goal.copy()


class LongFingerSurfaces:
    """Moving phalanx shells for the six pairs of non-thumb fingers."""
    def __init__(self,g):
        self.g=g
        geoms=[g.model.body_geomadr[g.chains[f][2:]] for f in FINGERS[1:]]
        self.pairs=np.array([(a,b) for i,j in combinations(range(4),2)
                             for a in geoms[i] for b in geoms[j]])
        self.tips=np.array([(s[-1],t[-1]) for s,t in combinations(geoms,2)])
        self.tip_rows=np.array([np.flatnonzero((self.pairs==pair).all(axis=1))[0] for pair in self.tips])
        self.offsets_mm=np.zeros(len(self.pairs))
        self.overlap_mm=0.
        self.previous=None
        # Bound how far any point of each shell can travel on a straight
        # joint-space chord, including both hinges of every coupled joint.
        m=g.model
        arms=np.zeros((m.ngeom,len(g.lo)))
        for geom in np.unique(self.pairs):
            body=m.geom_bodyid[geom]
            radius=np.linalg.norm(m.geom_pos[geom])+m.geom_rbound[geom]
            while body:
                for joint in range(m.body_jntadr[body],m.body_jntadr[body]+m.body_jntnum[body]):
                    arms[geom]+=(radius+np.linalg.norm(m.jnt_pos[joint]))*np.abs(g.C[joint])
                radius+=np.linalg.norm(m.body_pos[body])
                body=m.body_parentid[body]
        self.motion_bound=arms[self.pairs[:,0]]+arms[self.pairs[:,1]]

    def limit_chord(self,start,end):
        """Certify the interpolated long-finger shells, including routed ticks."""
        g=self.g;delta=end-start
        travel=self.motion_bound@np.abs(delta)
        if not np.any(travel>1.e-9):return end,False
        # Distance may decrease by at most travel * fraction. A bounding
        # sphere eliminates far meshes, and an exact distance sets the step.
        t=0.;ids=np.flatnonzero(travel>1.e-9)
        for _ in range(64):
            g.assign(start+t*delta)
            a,b=self.pairs[ids].T
            lower=np.linalg.norm(g.data.geom_xpos[a]-g.data.geom_xpos[b],axis=1)-g.model.geom_rbound[a]-g.model.geom_rbound[b]
            ids=ids[lower<=travel[ids]*(1.-t)+.000015]
            step=1.-t
            remaining=[]
            for k in ids:
                a,b=self.pairs[k]
                cutoff=travel[k]*(1.-t)+.000001
                gap=mujoco.mj_geomDistance(g.model,g.data,a,b,cutoff,None)
                if gap<cutoff:
                    remaining.append(k)
                    step=min(step,(gap-.000001-self.offsets_mm[k]*.001)/travel[k])
            if step>=1.-t:return end,False
            if step<1.e-6:break
            t+=.99*step
            ids=np.asarray(remaining,dtype=int)
        return start+t*delta,True

    def tangent_step(self,start,target,cap,*,recover=False):
        """Continue along a blocking shell without stopping the other digits."""
        g=self.g;g.assign(start)
        desired=target[5:]-start[5:]
        if np.max(abs(desired))<1.e-9:return target
        rows,short=clearance_linearization(g,2.,self.pairs,self.offsets_mm)
        J=rows[:,5:]
        required=short-(2.-.22)
        predicted=J@desired
        constrained=(predicted<required)&((predicted<-.001)|recover)
        if not np.any(constrained):return target
        local=min(cap,np.radians(1.2))
        desired=np.clip(desired,-local,local)
        A=np.vstack((5.*J[constrained],np.eye(12)))
        rhs=np.r_[5.*required[constrained],desired]
        bounds=(np.maximum(g.lo[5:]-start[5:],-local),np.minimum(g.hi[5:]-start[5:],local))
        step=lsq_linear(A,rhs,bounds=bounds,method='bvls').x
        return np.r_[target[:5],start[5:]+step]

    def clear(self):
        g=self.g;a,b=self.pairs.T
        bounds=np.linalg.norm(g.data.geom_xpos[a]-g.data.geom_xpos[b],axis=1)-g.model.geom_rbound[a]-g.model.geom_rbound[b]
        return all(mujoco.mj_geomDistance(g.model,g.data,*self.pairs[k],.0001,None)>=(.015+self.offsets_mm[k])*.001
                   for k in np.flatnonzero(bounds<.000015))

    def goal(self,nominal,hand,current=None,*,attract_only=False):
        g=self.g
        # Preserve a calibrated thumb opposition when it is the closest task.
        tip=hand[:,-1] if hand is not None else None
        weights=np.zeros(6)
        if tip is not None:
            a,b=PAIRS[4:].T
            d=np.linalg.norm(tip[a]-tip[b],axis=1)
            w=np.clip((13.-d)/6.,0.,1.)
            thumb=np.clip((np.linalg.norm(tip[1:]-tip[0],axis=1).min()-9.)/9.,0.,1.)
            weights=w*w*(3.-2.*w)*thumb*thumb*(3.-2.*thumb)
        # Only assist closure already within reach of the nominal posture.
        g.assign(nominal)
        for i in np.flatnonzero(weights):
            a,b=self.tips[i]
            gap=1000*mujoco.mj_geomDistance(g.model,g.data,a,b,.02,None)
            t=np.clip((15.-gap)/8.,0.,1.)
            weights[i]*=t*t*(3.-2.*t)
        if self.overlap_mm and current is not None and not attract_only:
            # Restore clearance from the executed gap when the source releases.
            # Only the six distal shell pairs ever get a negative allowance.
            requested=-self.overlap_mm*weights
            g.assign(current)
            for i,row in enumerate(self.tip_rows):
                if requested[i]>self.offsets_mm[row]:
                    gap=1000.*mujoco.mj_geomDistance(g.model,g.data,*self.tips[i],.001,None)
                    requested[i]=min(requested[i],max(self.offsets_mm[row],gap-.016))
            self.offsets_mm[self.tip_rows]=requested
        pulling=np.flatnonzero(weights)
        if attract_only and not pulling.size:
            self.previous=None
            return nominal
        def linearize(x):
            g.assign(np.r_[nominal[:5],x])
            jac=[4.*np.eye(12)]
            residual=[4.*(x-nominal[5:])]
            if not attract_only:
                J,short=clearance_linearization(g,.25,self.pairs,self.offsets_mm)
                jac.append(18.*J[:,5:])
                residual.append(-18.*short)
            for i in pulling:
                a,b=self.tips[i];points=np.empty(6)
                gap=mujoco.mj_geomDistance(g.model,g.data,a,b,.4,points)
                if abs(gap)<1.e-10 or (attract_only and gap<=0.):continue
                ja=np.empty((3,g.model.nv));jb=np.empty_like(ja)
                mujoco.mj_jac(g.model,g.data,ja,None,points[:3],g.model.geom_bodyid[a])
                mujoco.mj_jac(g.model,g.data,jb,None,points[3:],g.model.geom_bodyid[b])
                normal=(points[3:]-points[:3])/gap
                row=1000.*normal@(jb-ja)@g.C[:,5:]
                pull=4.*weights[i]
                jac.append(pull*row[None,:])
                target=0. if attract_only else .18+self.offsets_mm[self.tip_rows[i]]
                residual.append([pull*(1000.*gap-target)])
            return np.vstack(jac),np.concatenate(residual)
        x=nominal[5:].copy()
        J,r=linearize(x)
        if r@r<1.e-9:
            self.previous=None
            return nominal
        if self.previous is not None:
            old_nom,old_pose=self.previous
            guess=np.clip(x+old_pose-old_nom,g.lo[5:],g.hi[5:])
            j0,r0=linearize(guess)
            if r0@r0<r@r:x,J,r=guess,j0,r0
        for _ in range(8):
            lo=np.maximum(g.lo[5:]-x,-.09);hi=np.minimum(g.hi[5:]-x,.09)
            step=lsq_linear(J,-r,bounds=(lo,hi),method='bvls').x
            if np.max(abs(step))<2.e-5:break
            for scale in (1.,.5,.25,.125):
                trial=np.clip(x+scale*step,g.lo[5:],g.hi[5:])
                j1,r1=linearize(trial)
                if r1@r1 < r@r-1.e-7:
                    x,J,r=trial,j1,r1
                    break
            else:break
        self.previous=nominal[5:].copy(),x.copy()
        return np.r_[nominal[:5],x]
