"""Retarget the official thumb's location along a long finger to URDF-A hulls."""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from hand_kinematics import FINGERS

# Positive numerical reserve for straight interpolation of displayed poses.
CONTACT_GAP_MM = .03


def smooth(x):
    t = np.clip(x, 0., 1.)
    return t*t*(3.-2.*t)


@dataclass
class Touch:
    finger: str
    segment: int
    fraction: float
    weight: float
    side: np.ndarray
    rows: np.ndarray


class JointTouch:
    def __init__(self, model, settings):
        self.g = model
        self.settings = settings
        for name in ('radius_mm','index_radius_mm'):
            if name in settings and not 0.<settings[name]<=35.:
                raise ValueError('指节触摸源半径无效')
        self.rows = {}
        for finger in FINGERS[1:]:
            bodies = model.chains[finger]
            self.rows[finger] = np.array([np.flatnonzero(
                (model.pairs[:,0] == model.geoms['thumb']) &
                (model.pairs[:,1] == model.model.body_geomadr[b]))[0] for b in bodies])

    def observe(self, chains, rotation):
        fingers = np.stack([chains[f] for f in FINGERS[1:]])
        tip = chains['thumb'][-1]
        bones = np.diff(fingers, axis=1)
        offsets = tip-fingers[:,:3]
        t = np.clip(np.sum(offsets*bones,axis=2)/np.sum(bones*bones,axis=2),0.,1.)
        radial = offsets-t[:,:,None]*bones
        distances = np.linalg.norm(radial,axis=2)
        segments = distances.argmin(axis=1)
        ids = np.arange(4)
        d = distances[ids,segments]
        tips = np.linalg.norm(tip-fingers[:,-1],axis=1)
        # Require evidence for the shaft over *every* fingertip. Ordinary
        # opposition and gliding past an open gap retain their existing map.
        radius = np.full(4, self.settings['radius_mm'])
        radius[0] = self.settings.get('index_radius_mm',radius[0])
        score = smooth((radius+5.-d)/5.)*smooth((tips.min()-d-7.)/10.)
        # An ambiguous nearest digit contributes progressively less.
        competing=np.broadcast_to(d,(4,4)).copy()
        np.fill_diagonal(competing,np.inf)
        score *= smooth((competing.min(axis=1)-d)/6.)
        i = int(np.argmax(score))
        weight = float(score[i])
        if weight < 1.e-5:
            return None
        finger = FINGERS[i+1]
        j = int(segments[i])
        fraction = float(t[i,j])
        # Include the joint housing where two shaft segments meet.
        indices = [2*j+1,2*j+2]
        if fraction < .32:
            indices.append(2*j)
        if fraction > .68 and j < 2:
            indices.extend((2*j+3,2*j+4))
        side = rotation @ radial[i,j]
        return Touch(finger,j,fraction,weight,side/max(np.linalg.norm(side),1.e-8),
                     self.rows[finger][indices])

    def landmarks(self, touch):
        g = self.g
        f,j = touch.finger,touch.segment
        axes = [g.pair_axes[f,s] for s in ('mcp','pip','dip')]
        a = g.data.xanchor[axes[j]].mean(axis=0)*1000.
        body = g.chains[f][2*j+2]
        if j < 2:
            b = g.data.xanchor[axes[j+1]].mean(axis=0)*1000.
        else:
            b = (g.data.xpos[body]+g.data.xmat[body].reshape(3,3)@g.tip[f])*1000.
        axis = (b-a)/np.linalg.norm(b-a)
        normal = touch.side-axis*np.dot(touch.side,axis)
        normal /= max(np.linalg.norm(normal),1.e-8)
        return a+touch.fraction*(b-a),axis,normal,body

    def surface(self, touch):
        g = self.g
        best = np.inf
        for row in touch.rows:
            points = np.zeros(6)
            a,b = g.pairs[row]
            gap = 1000.*mujoco.mj_geomDistance(g.model,g.data,a,b,.35,points)
            if gap < best:
                best, nearest, witness = gap, int(row), points
        return best, nearest, witness

    def contact_rows(self,touch):
        """Only permit contact at the requested side and longitudinal region."""
        g=self.g
        anchor,axis,normal,_=self.landmarks(touch)
        floors={}
        for row in touch.rows:
            points=np.zeros(6)
            a,b=g.pairs[row]
            gap=mujoco.mj_geomDistance(g.model,g.data,a,b,.005,points)
            if gap>=.005:
                continue
            offset=1000.*points[3:]-anchor
            floors[int(row)]=self.local_weight(touch,offset,axis,normal)
        return floors

    @staticmethod
    def local_weight(touch,offset,axis,normal):
        return touch.weight*smooth((14.-abs(axis@offset))/5.)*smooth((normal@offset+1.)/4.)

    def point_jac(self, body, point):
        g = self.g
        jac = np.zeros((3,g.model.nv))
        mujoco.mj_jac(g.model,g.data,jac,None,point,int(body))
        return 1000.*jac@g.C

    def distance_row(self, row, gap, points, fallback):
        g = self.g
        a,b = g.pairs[row]
        normal = (points[3:]-points[:3])*1000./gap if abs(gap)>1.e-6 else fallback
        jac = self.point_jac(g.model.geom_bodyid[b],points[3:])-self.point_jac(g.model.geom_bodyid[a],points[:3])
        return normal@jac

    def solve(self, nominal, touch, bend_axis=None, *, landmark_only=False):
        g = self.g
        ids = np.r_[g.cols['thumb'],g.cols[touch.finger]]
        root_weight=(1.-smooth((touch.segment+touch.fraction)/.55))*touch.weight
        cap = np.radians([52.+18.*root_weight]*5+[20.+8.*root_weight]*3)
        lo = np.maximum(g.lo[ids],nominal[ids]-cap)
        hi = np.minimum(g.hi[ids],nominal[ids]+cap)
        q = nominal.copy()
        if landmark_only:
            q=self.fit(q,nominal,touch,ids,lo,hi,bend_axis,landmark_only=True)
            return nominal+touch.weight*(q-nominal)
        q=self.fit(q,nominal,touch,ids,lo,hi,bend_axis)
        g.assign(q)
        anchor,axis,normal,_=self.landmarks(touch)
        gap,_,points=self.surface(touch)
        # A folded thumb can put a shaft between its pad and the requested
        # joint. Try an extended distal thumb in that constrained case.
        def cost(pose):
            g.assign(pose)
            anchor,axis,normal,_=self.landmarks(touch)
            gap,_,points=self.surface(touch)
            off=points[3:]*1000.-anchor
            return (axis@off)**2+(.7*np.cross(axis,normal)@off)**2+gap**2+max(0.,-g.clearance())**2*100
        if abs(axis@(points[3:]*1000.-anchor))>10. or gap>2. or g.clearance() < -.5:
            seed=nominal.copy()
            seed[0]=np.clip(seed[0]+np.radians(20.),lo[0],hi[0])
            seed[3:5]=np.clip(seed[3:5]-np.radians(35.),lo[3:5],hi[3:5])
            alternative=self.fit(seed,nominal,touch,ids,lo,hi,bend_axis)
            if cost(alternative)<cost(q):
                q=alternative
        q=nominal+touch.weight*(q-nominal)
        lower,upper=g.lo.copy(),g.hi.copy()
        lower[ids],upper[ids]=lo,hi
        q=self.project(q,touch,lower,upper)
        if touch.weight>.95:
            q=self.finish(q,touch,ids,lower,upper)
        return q

    def finish(self,q,touch,ids,lower,upper):
        """Close residual shell gaps while protecting neighbouring joints."""
        g=self.g
        origin=q.copy()
        for _ in range(6):
            g.assign(q)
            gap,row,points=self.surface(touch)
            anchor,axis,normal,body=self.landmarks(touch)
            offset=points[3:]*1000.-anchor
            weight=self.local_weight(touch,offset,axis,normal)
            target=CONTACT_GAP_MM+.8*(1.-weight)
            if abs(gap-target)<.005 and g.clearance()>=min(target-.005,.6):
                break
            if weight<.3 and gap>.8:
                break
            j=self.point_jac(g.distal['thumb'],points[:3])-self.point_jac(body,points[3:])
            rows=[4.*self.distance_row(row,gap,points,-normal),axis@j,np.cross(axis,normal)@j]
            errors=[4.*(gap-target),axis@offset,np.cross(axis,normal)@offset]
            a,b=g.pairs.T
            bounds=np.linalg.norm(g.data.geom_xpos[a]-g.data.geom_xpos[b],axis=1)-g.model.geom_rbound[a]-g.model.geom_rbound[b]
            for k in np.flatnonzero(bounds<.001):
                if k==row:continue
                pts=np.zeros(6)
                d=1000.*mujoco.mj_geomDistance(g.model,g.data,a[k],b[k],.001,pts)
                if d<.8 and abs(d)>1.e-6:
                    rows.append(14.*self.distance_row(k,d,pts,-normal))
                    errors.append(14.*(d-.8))
            A=np.vstack((np.array(rows)[:,ids],4.*np.eye(len(ids))))
            rhs=-np.r_[errors,4.*(q[ids]-origin[ids])]
            step=lsq_linear(A,rhs,bounds=(lower[ids]-q[ids],upper[ids]-q[ids]),method='bvls').x
            q[ids]+=.7*np.clip(step,-.12,.12)
        return self.project(q,touch,lower,upper)

    def fit(self,q,nominal,touch,ids,lo,hi,bend_axis=None,*,landmark_only=False):
        g=self.g
        for iteration in range(12):
            g.assign(q)
            anchor,axis,normal,body = self.landmarks(touch)
            pad,pad_normal = g.pad_pose('thumb')
            # Track both where along the shaft and on which side the thumb
            # lies. The independent normal residual uses actual mesh witnesses.
            # Move the closest thumb-shell point along the finger. A pad
            # marker alone can remain near a joint while the actual contact
            # patch drifts several millimetres along the neighbouring shaft.
            # First approach the intended spot with the thumb pad. A
            # distant hull witness often belongs to the opposite end.
            point=pad
            if not landmark_only:
                gap,row,points=self.surface(touch)
                blend=.80*smooth((12.-abs(gap))/10.)
                point=(1.-blend)*pad+blend*points[:3]*1000.
            J = self.point_jac(g.distal['thumb'],point*.001)-self.point_jac(body,anchor*.001)
            lateral = np.cross(axis,normal)
            origin = (anchor+35.*normal)*.001
            rays=[mujoco.mj_rayMesh(g.model,g.data,g.pairs[k,1],origin,-normal) for k in touch.rows]
            hit=min((d for d in rays if 0<=d<.035),default=.027)
            target=1000.*origin-1000.*hit*normal
            rows = [1.6*axis@J, lateral@J, .65*normal@J]
            errors = [1.6*axis@(point-target),lateral@(point-target),.65*normal@(point-target)]
            # On the shaft, use spare rotation to follow the glove plane.
            # The knuckle region already constrains the thumb pad normal.
            plane_weight = 4.*smooth((min(touch.fraction, 1.-touch.fraction)-.25)/.15)
            if bend_axis is not None and plane_weight > 0.:
                hinge = g.pair_axes['thumb','mcp'][0]
                current = g.data.xaxis[hinge]
                jr = np.empty((3,g.model.nv))
                mujoco.mj_jacBody(g.model,g.data,None,jr,g.model.jnt_bodyid[hinge])
                rows.extend(plane_weight*np.cross((jr@g.C).T,current).T)
                errors.extend(plane_weight*(current-bend_axis))
            # Around a knuckle, seat the pad itself on the approach face.
            # Shaft sliding remains free to roll on the thumb shell.
            at_joint=1.-smooth(min(touch.fraction,1.-touch.fraction)/.28)
            if at_joint>0.:
                arm=self.point_jac(g.distal['thumb'],pad*.001)-self.point_jac(body,target*.001)
                rows.extend(at_joint*arm)
                errors.extend(at_joint*(pad-target))
                jr=np.zeros((3,g.model.nv))
                mujoco.mj_jacBody(g.model,g.data,None,jr,g.distal['thumb'])
                rows.extend(9.*at_joint*np.cross((jr@g.C).T,pad_normal).T)
                errors.extend(9.*at_joint*(pad_normal+normal))
            along_error=abs(axis@(point-target))
            strength = (.5 if iteration<4 else 3.)/(1.+(along_error/10.)**2)
            if not landmark_only:
                rows.append(strength*self.distance_row(row,gap,points,-normal))
                errors.append(strength*gap)
            # Other thumb links, the palm and unrelated fingers retain the
            # original guard. Include their local constraints in the IK.
            if not landmark_only:
                a,b = g.pairs.T
                bound = np.linalg.norm(g.data.geom_xpos[a]-g.data.geom_xpos[b],axis=1)-g.model.geom_rbound[a]-g.model.geom_rbound[b]
                for k in np.flatnonzero(bound < .0014):
                    if k in touch.rows:
                        continue
                    pts = np.zeros(6)
                    gap2 = 1000.*mujoco.mj_geomDistance(g.model,g.data,a[k],b[k],.0014,pts)
                    if gap2 < 1.1 and abs(gap2)>1.e-6:
                        rows.append(8.*self.distance_row(k,gap2,pts,-normal))
                        errors.append(8.*(gap2-1.1))
            prior = np.array([2.]*5+[5.]*3)
            A = np.vstack((np.asarray(rows)[:,ids],np.diag(prior)))
            rhs = -np.r_[errors,prior*(q[ids]-nominal[ids])]
            step = lsq_linear(A,rhs,bounds=(lo-q[ids],hi-q[ids]),method='bvls',max_iter=20).x
            q[ids] += .85*step
            if np.max(np.abs(step)) < 1.e-5:
                break
        return q

    def project(self,q,touch,lower,upper):
        """Track the local sliding surface inside the available joint travel."""
        g=self.g
        ids=np.r_[g.cols['thumb'],g.cols[touch.finger]]
        q=q.copy()
        for _ in range(4):
            g.assign(q)
            gap,row,points=self.surface(touch)
            if gap>.7:
                break
            anchor,axis,normal,_=self.landmarks(touch)
            weight=self.local_weight(touch,points[3:]*1000.-anchor,axis,normal)
            # Free-space tracking is allowed to leave the surface; only a
            # confident, correctly located contact attracts the thumb.
            target=CONTACT_GAP_MM+.8*(1.-weight)
            if (weight<.999 and gap>=target) or abs(gap-target)<.0003:
                break
            jac=self.distance_row(row,gap,points,-normal)[ids]
            error=target-gap
            free=((jac*error>0)&(q[ids]<upper[ids]-1.e-10))|((jac*error<0)&(q[ids]>lower[ids]+1.e-10))
            jac=np.where(free,jac,0.)
            delta=np.clip(error*jac/(jac@jac+.01),-.025,.025)
            q[ids]=np.clip(q[ids]+delta,lower[ids],upper[ids])
        return q
