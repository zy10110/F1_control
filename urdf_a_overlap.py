"""Kinematic joint stops for the visual teleoperation simulator."""
from itertools import combinations

import mujoco
import numpy as np

from hand_kinematics import FINGERS


class OverlapStop:
    def __init__(self, hand, limit_mm):
        if not np.isfinite(limit_mm) or limit_mm <= 0:
            raise ValueError('重叠阈值必须为正数，单位 mm')
        self.g=hand
        self.limit_mm=float(limit_mm)
        model=hand.model
        pairs=[(model.body_geomadr[i],model.body_geomadr[j])
               for f,h in combinations(FINGERS,2) for i in hand.chains[f] for j in hand.chains[h]]
        # Thumb housings at the base attach to the palm. Keep the established
        # thumb/palm scope; all inter-digit link pairs are independent.
        pairs.extend((model.body_geomadr[i],model.body_geomadr[hand.palm]) for i in hand.chains['thumb'][2:])
        self.a,self.b=np.asarray(pairs,dtype=int).T
        self.radii=model.geom_rbound[self.a]+model.geom_rbound[self.b]
        self.stopped=np.zeros(len(hand.lo),bool)

    def violation(self,q):
        g=self.g
        g.assign(q)
        centers=g.data.geom_xpos[self.a]-g.data.geom_xpos[self.b]
        floor=-self.limit_mm*.001
        near=np.flatnonzero(np.einsum('ij,ij->i',centers,centers)<np.maximum(0.,self.radii+floor)**2)
        for k in near:
            points=np.empty(6)
            gap=mujoco.mj_geomDistance(g.model,g.data,self.a[k],self.b[k],.001,points)
            if gap<floor:
                return k,gap,points
        return None

    def limit(self,start,requested):
        """Clip only inward components, then recheck the actual displayed chord."""
        self.stopped.fill(False)
        end=requested.copy()
        g=self.g
        for _ in range(len(start)+1):
            delta=end-start
            travel=np.max(abs(delta))
            if travel<1.e-10:return start.copy()
            steps=max(1,int(np.ceil(travel/np.radians(.35))))
            for i in range(1,steps+1):
                hit=self.violation(start+(i/steps)*delta)
                if hit is not None:break
            else:
                return end
            low,high=(i-1)/steps,i/steps
            # Find the last admissible position of the joint(s) that close
            # this pair. Other fingers keep their full commanded increment.
            for _bisect in range(8):
                mid=(low+high)*.5
                if self.violation(start+mid*delta) is None:low=mid
                else:high=mid
            hit=self.violation(start+high*delta)
            k,gap,points=hit
            ja=np.zeros((3,g.model.nv));jb=np.zeros_like(ja)
            mujoco.mj_jac(g.model,g.data,ja,None,points[:3],g.model.geom_bodyid[self.a[k]])
            mujoco.mj_jac(g.model,g.data,jb,None,points[3:],g.model.geom_bodyid[self.b[k]])
            derivative=((points[3:]-points[:3])/gap)@(jb-ja)@g.C
            inward=(derivative*delta < -1.e-13) & ~self.stopped
            if not inward.any():
                # Degenerate witness or second-order contact. Limit only the
                # moving axes on the two involved kinematic chains.
                involved=np.any(abs((jb-ja)@g.C)>1.e-10,axis=0)
                inward=involved & (abs(delta)>1.e-12)
                if not inward.any():return start+low*delta
            self.stopped|=inward
            end[inward]=start[inward]+low*delta[inward]
        # Numerical fallback: retain the verified prefix of the final chord.
        for i in range(1,steps+1):
            if self.violation(start+(i/steps)*(end-start)) is not None:
                return start+((i-1)/steps)*(end-start)
        return end
