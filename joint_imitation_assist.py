"""Bounded posture projection around live joint commands, for simulation.

Bone directions (all phalanges), joint angles and coupled DIP observations
define the task. Relative tip vectors are a weak optional secondary cue;
there is no absolute endpoint target, contact snap or recorded pose lookup.
"""
import time
import numpy as np
from hand_kinematics import FINGERS
from g7_proportional_mapping import FingerLandmarks, source_points, normalized, BASE
from thumb_orientation_mapping import basis


class JointImitationAssist:
    kind = 'joint_imitation'

    def __init__(self, g, config, reference, hand='right'):
        self.g, self.hand = g, hand
        self.reference = reference.copy()
        self.land = FingerLandmarks(g)
        self.land.assign(reference)
        rest = {f: self.land.points(f) for f in FINGERS}
        self.lengths = {f: np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                        for f, p in rest.items()}
        self.target_basis = basis(rest['index'][0]-rest['pinky'][0],
            np.mean([rest[f][-1]-rest[f][0] for f in FINGERS[1:]], axis=0))
        self.cap = np.radians(config.get('imitation_cap_deg', [25,25,25,12,12]+[8,12,12]*4))
        self.prior = np.asarray(config.get('imitation_angle_prior', [12,12,12,28,28]+[28,28,28]*4), float)
        self.shape_weight = float(config.get('imitation_shape_weight', 24.))
        self.relative_weight = float(config.get('imitation_relative_weight', .08))
        self.dip_weight = float(config.get('imitation_dip_weight', 14.))
        self.speed = np.radians(float(config.get('imitation_correction_speed_deg_s', 60.)))
        if (self.cap.shape != reference.shape or self.prior.shape != reference.shape
                or not np.isfinite(self.cap).all() or not np.isfinite(self.prior).all()
                or np.any(self.cap <= 0) or np.any(self.prior <= 0)
                or not np.isfinite([self.shape_weight,self.relative_weight,self.dip_weight,self.speed]).all()
                or min(self.shape_weight,self.speed) <= 0 or min(self.relative_weight,self.dip_weight) < 0):
            raise ValueError('整手模仿参数无效')
        # Read the actual reduction, including each finger's configured ratio.
        self.ratios = {}
        for f in FINGERS[1:]:
            # The public geometry configuration is authoritative for coupling.
            self.ratios[f] = float(g.cfg['fingers'][f]['dip_pip_ratio'])
        self.last = np.zeros_like(reference)
        self.last_stamp = self.last_time = None
        self.warm_start = bool(config.get('imitation_warm_start',False))
        self.iterations=6

    def targets(self, frame):
        p = source_points(frame, self.hand)
        rotation = self.target_basis @ basis(p[5]-p[17],p[[5,9,13,17]].mean(axis=0)-p[0]).T
        directions = {f: normalized(np.diff(p[i:i+4],axis=0)) @ rotation.T for f,i in BASE.items()}
        lengths = {f: np.linalg.norm(np.diff(p[i:i+4],axis=0),axis=1).sum() for f,i in BASE.items()}
        relative = {f: rotation @ (p[4]-p[BASE[f]+3]) *
                    np.sqrt(self.lengths['thumb']*self.lengths[f]/(lengths['thumb']*lengths[f]))
                    for f in FINGERS[1:]}
        return directions, relative

    def system(self, u, base, targets, angles):
        directions, relative = targets
        self.land.assign(u)
        residual = [self.prior*(u-base)]
        jacobian = [np.diag(self.prior)]
        points, jacs = {}, {}
        for f in FINGERS:
            points[f],jacs[f] = self.land.points(f,True)
            v = np.diff(points[f],axis=0)
            dv = np.diff(jacs[f],axis=0)
            lengths = np.linalg.norm(v,axis=1)
            unit = v/lengths[:,None]
            jd = np.array([(np.eye(3)-np.outer(n,n))@j/l for n,j,l in zip(unit,dv,lengths)])
            residual.append(self.shape_weight*(unit-directions[f]).ravel())
            jacobian.append(self.shape_weight*jd.reshape(9,len(u)))
            if f != 'thumb':
                k = self.g.slices[f].start+2
                ratio = self.ratios[f]
                residual.append(np.array([self.dip_weight*(ratio*(u[k]-self.reference[k])-
                    np.radians(angles[f]['DIP']))]))
                row = np.zeros((1,len(u)));row[0,k] = self.dip_weight*ratio
                jacobian.append(row)
        for f in FINGERS[1:]:
            residual.append(self.relative_weight*(points['thumb'][-1]-points[f][-1]-relative[f]))
            jacobian.append(self.relative_weight*(jacs['thumb'][-1]-jacs[f][-1]))
        return np.concatenate(residual),np.vstack(jacobian)

    def metrics(self, u, targets):
        self.land.assign(u)
        p = {f:self.land.points(f) for f in FINGERS}
        return dict(bone_errors_deg={f:np.degrees(np.arccos(np.clip(np.sum(
            normalized(np.diff(p[f],axis=0))*targets[0][f],axis=1),-1,1))) for f in FINGERS},
            relative_errors_mm={f:float(np.linalg.norm(p['thumb'][-1]-p[f][-1]-targets[1][f]))
                                for f in FINGERS[1:]})

    def apply(self, frame, base, lower, upper, angles):
        # All input validation precedes timing and persistent-state updates.
        targets = self.targets(frame)
        if not all(np.isfinite(list(a.values())).all() for a in angles.values()):
            raise ValueError('整手模仿关节角无效')
        lo,hi = np.maximum(lower,base-self.cap),np.minimum(upper,base+self.cap)
        u = base.copy()
        initial,_ = self.system(u,base,targets,angles)
        if self.warm_start:
            seed=np.clip(base+self.last,lo,hi)
            rr,_=self.system(seed,base,targets,angles)
            if rr@rr<initial@initial:u=seed
        for _ in range(self.iterations):
            r,j = self.system(u,base,targets,angles)
            gradient = j.T@r
            free = ~(((u <= lo+1e-9)&(gradient>0)) | ((u >= hi-1e-9)&(gradient<0)))
            step = np.zeros_like(u)
            jf = j[:,free]
            step[free] = np.linalg.solve(jf.T@jf+np.eye(free.sum())*1e-7,-gradient[free])
            step = np.clip(step,-np.radians(5),np.radians(5))
            if np.max(np.abs(step)) < 1e-7: break
            for fraction in (1.,.5,.2):
                candidate = np.clip(u+fraction*step,lo,hi)
                rr,_ = self.system(candidate,base,targets,angles)
                if rr@rr < r@r-1e-10:
                    u = candidate
                    break
            else: break
        final,_ = self.system(u,base,targets,angles)
        now = time.monotonic();stamp = frame.get('timestamp_ns')
        dt = 1/120 if self.last_time is None else now-self.last_time
        valid_stamp = isinstance(stamp,(int,float)) and np.isfinite(stamp)
        if valid_stamp and self.last_stamp is not None and stamp > self.last_stamp:
            dt = (stamp-self.last_stamp)*1e-9
        dt = float(np.clip(dt,1/1000,1/30))
        correction = self.last+np.clip(u-base-self.last,-self.speed*dt,self.speed*dt)
        bounded = np.clip(correction,lo-base,hi-base)
        result = base+bounded
        self.last = bounded.copy();self.last_time = now
        self.last_stamp = stamp if valid_stamp else None
        return result,dict(kind=self.kind,enabled=True,correction_deg=np.degrees(bounded),
            raw_correction_deg=np.degrees(u-base),cap_deg=np.degrees(self.cap),dt_s=dt,
            correction_speed_deg_s=float(np.degrees(self.speed)),
            limit_overrode_correction_rate=bool(np.max(np.abs(bounded-correction))>1e-9),
            baseline_objective=float(initial@initial),fitted_objective=float(final@final),
            baseline=self.metrics(base,targets),command=self.metrics(result,targets),
            dip_coupling_observation_residual_deg={f:float(np.degrees(self.ratios[f]*(
                result[self.g.slices[f].start+2]-self.reference[self.g.slices[f].start+2]))-angles[f]['DIP'])
                for f in FINGERS[1:]},
            at_limit_indices=np.flatnonzero((result<=lower+1e-7)|(result>=upper-1e-7)).tolist(),
            absolute_endpoint_tracking=False,forced_closure=False,extra_calibration=False)
