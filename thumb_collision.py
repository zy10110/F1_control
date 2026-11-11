"""Conservative mesh-hull clearance and sampled motion checks, in mm.

Uses the original MuJoCo mesh convex hulls, not a friction/contact dynamics
simulation. Adjacent thumb hinges are excluded; no mesh or URDF is changed.
Each guard owns MjData and must be used by one thread.
"""
import numpy as np
import mujoco
from scipy.optimize import minimize


class ThumbCollisionGuard:
    def __init__(self, geometry, margin_mm=.6, step_deg=.75):
        self.g = geometry
        self.data = mujoco.MjData(geometry.model)
        self.margin = float(margin_mm)
        self.step = np.radians(step_deg)
        model = self.g.model
        by_link = {model.body(int(model.geom_bodyid[i])).name: i
                   for i in range(model.ngeom)}
        # Root hinges 1/2 are attached to the palm; exclude those assemblies.
        self.pairs = [(by_link[f'f1_link_{a}'], by_link[f'f1_link_{b}'])
                      for a in range(3, 8) for b in (0, *range(8, 36))]
        self.i = np.asarray([p[0] for p in self.pairs])
        self.j = np.asarray([p[1] for p in self.pairs])

    def distances(self, u):
        self.g.assign(self.data, u)
        m, d = self.g.model, self.data
        # Bounding spheres only reject distant pairs; exact hull distance is
        # still evaluated whenever a pair could enter the clearance band.
        lower = (np.linalg.norm(d.geom_xpos[self.i]-d.geom_xpos[self.j], axis=1)
                 - m.geom_rbound[self.i] - m.geom_rbound[self.j])*1000
        values = np.full(len(self.pairs), 10.)
        for k in np.flatnonzero(lower < 10.):
            a, b = self.pairs[k]
            values[k] = mujoco.mj_geomDistance(m, d, a, b, .01, None)*1000
        return values

    def report(self, u):
        ds = self.distances(u); k = int(np.argmin(ds)); a, b = self.pairs[k]
        return dict(clearance_mm=float(ds[k]), required_clearance_mm=self.margin,
                    safe=bool(ds[k] >= self.margin-1e-5),
                    pair=[self.g.model.body(int(self.g.model.geom_bodyid[x])).name for x in (a,b)],
                    geometry='original mesh convex hulls', path_step_deg=float(np.degrees(self.step)))

    def safe(self, u):
        return bool(np.min(self.distances(u)) >= self.margin-1e-5)

    def project(self, desired, previous, reference, samples=None, target=None, lower_override=None, upper_override=None):
        """Keep other fingers' requested angles whenever a safe thumb exists.

        Local constrained optimization plus finite safe seeds; this is not a
        globally optimal collision-free workspace projection.
        """
        lower=self.g.lower if lower_override is None else np.maximum(self.g.lower,lower_override)
        upper=self.g.upper if upper_override is None else np.minimum(self.g.upper,upper_override)
        desired = np.clip(self.g.validate(desired),lower,upper).copy()
        if self.safe(desired): return desired, dict(self.report(desired), projected=False)
        idx = np.arange(self.g.slices['thumb'].start, self.g.slices['thumb'].stop)
        def expand(v):
            u = desired.copy(); u[idx] = v; return u
        def score(u):
            posture=float(np.sum((u[idx]-desired[idx])**2))
            if target is None:return posture
            position=self.g.forward(u,'thumb')['position_mm']
            return float(np.sum((position-target)**2)/2500+.02*posture)
        seeds = [previous[idx], reference[idx]]
        if samples is not None:
            distance = np.linalg.norm(samples-desired[idx], axis=1)
            seeds.extend(samples[np.argsort(distance)[:32]])
        feasible = []
        for seed in seeds:
            candidate = expand(np.clip(seed,lower[idx],upper[idx]))
            if self.safe(candidate):
                feasible.append(candidate)
                if len(feasible) >= 3: break
        # Start on the feasible side of the obstacle and approach the desired
        # angle vector. Hard acceptance below is independent of solver success.
        if feasible:
            start = min(feasible, key=score)
            fit = minimize(lambda v: score(expand(v)), start[idx], method='SLSQP',
                bounds=list(zip(lower[idx],upper[idx])),
                constraints=[dict(type='ineq',fun=lambda v: self.distances(expand(v))-self.margin)],
                options=dict(maxiter=9,ftol=1e-5))
            candidate = expand(np.clip(fit.x,lower[idx],upper[idx]))
            if self.safe(candidate): feasible.append(candidate)
            result = min(feasible,key=score)
        else:
            # A moving index may invalidate a formerly safe thumb. Do not move
            # the other fingers through it when no safe thumb seed was found.
            result = previous.copy()
            if np.any(result<lower-1e-8) or np.any(result>upper+1e-8) or not self.safe(result):
                raise ValueError('当前姿态已在拇指碰撞区，请停止跟随并复位到初始姿态')
        return result, dict(self.report(result),projected=True,
                            held_other_fingers=bool(not np.allclose(result[5:],desired[5:])))

    def sweep(self, current, proposed):
        """Return last accepted sample; never jump over an unsafe sample."""
        current = self.g.validate(current).copy(); proposed = self.g.validate(proposed)
        if not self.safe(current):
            return current, dict(self.report(current),blocked=True,accepted_fraction=0.)
        count=max(1,int(np.ceil(np.max(np.abs(proposed-current))/self.step)))
        accepted=current.copy()
        for k in range(1,count+1):
            u=current+(proposed-current)*(k/count)
            if not self.safe(u):
                return accepted,dict(self.report(accepted),blocked=True,accepted_fraction=(k-1)/count)
            accepted=u
        return accepted,dict(self.report(accepted),blocked=False,accepted_fraction=1.)

    def sweep_follow(self, current, proposed):
        """Try safe joint groups when a simultaneous teleop step is blocked.

        Each segment uses the same sampled clearance check. No joint takes a
        larger step than requested; four-finger commands and limits are intact.
        This local fallback does not claim to find a route around every obstacle.
        """
        direct, report = self.sweep(current, proposed)
        if not report['blocked'] or not self.safe(current):
            return direct, dict(report, route='simultaneous')
        current=self.g.validate(current).copy()
        proposed=self.g.validate(proposed).copy()
        candidates=[(direct, report, 'simultaneous')]
        for first, second, label in ((slice(5,None),slice(0,5),'four_then_thumb'),
                                     (slice(0,5),slice(5,None),'thumb_then_four')):
            intermediate=current.copy();intermediate[first]=proposed[first]
            accepted, first_report=self.sweep(current,intermediate)
            end=accepted.copy();end[second]=proposed[second]
            accepted, second_report=self.sweep(accepted,end)
            candidates.append((accepted, second_report, label))
        accepted, final, route=min(candidates,key=lambda item:float(np.linalg.norm(item[0]-proposed)))
        complete=bool(np.allclose(accepted,proposed,atol=1e-10,rtol=0))
        return accepted,dict(final,blocked=not complete,route=route,
            partial_progress=bool(np.max(np.abs(accepted-current))>1e-8),
            accepted_fraction=None,
            thumb_limited=bool(not np.allclose(accepted[:5],proposed[:5],atol=1e-10,rtol=0)),
            four_fingers_limited=bool(not np.allclose(accepted[5:],proposed[5:],atol=1e-10,rtol=0)))

    def sweep_follow_digits(self,current,proposed):
        """After a blocked group step, finish reachable digits one at a time."""
        accepted,report=self.sweep_follow(current,proposed)
        if not report['blocked']:return accepted,report
        moved=[]
        for finger,sl in self.g.slices.items():
            if finger=='thumb' or np.max(np.abs(proposed[sl]-accepted[sl]))<1e-7:continue
            candidate=accepted.copy()
            candidate[sl]=proposed[sl]
            end,_=self.sweep(accepted,candidate)
            if np.max(np.abs(end-accepted))>1e-7:
                accepted=end;moved.append(finger)
        if not moved:return accepted,report
        return accepted,dict(self.report(accepted),route=report['route']+'+individual',
            blocked=not np.allclose(accepted,proposed,rtol=0,atol=1e-10),
            recovered_digits=moved,accepted_fraction=None,partial_progress=True,
            thumb_limited=not np.allclose(accepted[:5],proposed[:5],rtol=0,atol=1e-10),
            four_fingers_limited=not np.allclose(accepted[5:],proposed[5:],rtol=0,atol=1e-10))
