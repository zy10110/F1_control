"""Local pinch assistance on top of joint imitation; all distances are mm.

Source tip separation gates this task. It does not require the other three
fingers to reproduce a calibration pose. The returned command is still passed
through the existing collision projection, speed filter and sampled path guard.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from thumb_collision import ThumbCollisionGuard

PAIRS = ('index', 'middle', 'ring', 'pinky')
TIPS = dict(index=8, middle=12, ring=16, pinky=20)


class PairedFingerGuard(ThumbCollisionGuard):
    """Add the moving finger's distal links against palm and other fingers.

    Its own adjacent assemblies retain the exclusions of the source model.
    Existing collisions in the unmodified four-finger mapping are not certified.
    """
    def __init__(self, geometry, finger, margin_mm=.65):
        super().__init__(geometry, margin_mm)
        self.thumb_pair_count=len(self.pairs)
        m = geometry.model
        by_link = {m.body(int(m.geom_bodyid[i])).name: i for i in range(m.ngeom)}
        start = 8+7*PAIRS.index(finger)
        other = [0]+[b for b in range(8,36) if not start <= b < start+7]
        extra = [(by_link[f'f1_link_{a}'], by_link[f'f1_link_{b}'])
                 for a in range(start+2, start+7) for b in other]
        self.pairs = list(dict.fromkeys(self.pairs+extra))
        self.i = np.asarray([p[0] for p in self.pairs])
        self.j = np.asarray([p[1] for p in self.pairs])

    def sweep(self, current, proposed):
        """Protect clear pairs; report inherited overlaps without freezing.

        The source four-finger mapping can already contain convex-hull overlap.
        Those pairs are allowed to separate; they become guarded once clear.
        Thumb pairs remain covered by the original .6 mm execution guard.
        """
        current = self.g.validate(current).copy()
        proposed = self.g.validate(proposed)
        distances = self.distances(current)
        if np.min(distances[:self.thumb_pair_count]) < self.margin-1e-5:
            return current,dict(self.report(current),blocked=True,accepted_fraction=0.,
                                reason='original thumb clearance violated')
        inherited = distances < 0.
        accepted = current.copy()
        count = max(1,int(np.ceil(np.max(np.abs(proposed-current))/self.step)))
        for k in range(1,count+1):
            u = current+(proposed-current)*(k/count)
            ds = self.distances(u)
            protected = ~inherited
            if np.any(ds[protected] < np.minimum(distances[protected],self.margin)-1e-5):
                return accepted,dict(self.report(accepted),blocked=True,accepted_fraction=(k-1)/count,
                                     inherited_overlap_pairs=int(inherited.sum()))
            accepted, distances = u, ds
            inherited &= ds < 0.
        return accepted,dict(self.report(accepted),blocked=False,accepted_fraction=1.,
                             inherited_overlap_pairs=int(inherited.sum()))

    def sweep_follow(self, current, proposed):
        direct, report = self.sweep(current, proposed)
        if not report['blocked']:
            return direct,dict(report,route='simultaneous')
        candidates = [(direct,report,'simultaneous')]
        for first,second,label in ((slice(5,None),slice(0,5),'four_then_thumb'),
                                   (slice(0,5),slice(5,None),'thumb_then_four')):
            waypoint = np.asarray(current).copy()
            waypoint[first] = proposed[first]
            u,_ = self.sweep(current,waypoint)
            end = u.copy();end[second] = proposed[second]
            u,r = self.sweep(u,end)
            candidates.append((u,r,label))
        u,r,label = min(candidates,key=lambda x:np.linalg.norm(x[0]-proposed))
        return u,dict(r,blocked=bool(not np.allclose(u,proposed,atol=1e-10,rtol=0)),route=label)


def source_tip_gaps(frame):
    p = np.asarray(frame.get('positions_m', []), float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    # Reject placeholder positions; orientations alone cannot establish touch.
    lengths = [sum(np.linalg.norm(p[i+1]-p[i]) for i in range(root, root+3))
               for root in (1, 5, 9, 13, 17)]
    if min(lengths) < .015 or max(lengths) > .35:
        return None
    return {f: float(np.linalg.norm(p[4]-p[i])*1000) for f, i in TIPS.items()}


class ContactAssist:
    def __init__(self, owner):
        self.owner = owner
        self.g, self.pad = owner.g, owner.pad
        self.data = mujoco.MjData(self.g.model)
        self.cfg = owner.record['contact_assist']
        self.cooperative = bool(self.cfg.get('paired_finger_enabled', False))
        self.guard = ThumbCollisionGuard(self.g, .65)
        self.pair_guards = {f: PairedFingerGuard(self.g, f) for f in PAIRS}
        self.last = None
        self.last_pair = None
        self.contacts = {}
        self.thresholds = {}
        for f in PAIRS:
            gaps = source_tip_gaps(owner.record['anchors'][f]['frame'])
            if gaps is None:
                raise ValueError('对指辅助需要官方输出的有效21点位置')
            self.contacts[f] = gaps[f]
            default_full = gaps[f]+3.
            full = float(self.cfg.get('full_assistance_mm',{}).get(f,default_full))
            start = float(self.cfg.get('approach_start_mm',{}).get(f,
                default_full+float(self.cfg.get('approach_band_mm',28.))))
            if not np.isfinite([full,start]).all() or full<0. or start<=full:
                raise ValueError('对指接近阈值需要 0 <= 完全辅助距离 < 开始辅助距离')
            self.thresholds[f] = dict(full=full,start=start)

    def weights(self, frame):
        gaps = source_tip_gaps(frame)
        weights = {f: 0. for f in PAIRS}
        if gaps is None:
            return weights, None, None
        strengths = {}
        for f in PAIRS:
            full = self.thresholds[f]['full']
            start = self.thresholds[f]['start']
            t = float(np.clip((start-gaps[f])/(start-full), 0., 1.))
            strengths[f] = t*t*(3.-2.*t)
        pair = max(strengths, key=strengths.get)
        # Retain the current pair when two near contacts have almost equal
        # strength; this reduces switching in a relaxed fist.
        if self.last_pair and strengths[self.last_pair] >= strengths[pair]-.08:
            pair = self.last_pair
        if strengths[pair] <= 0.:
            return weights, gaps, None
        weights[pair] = strengths[pair]
        return weights, gaps, pair

    def _pose_jac(self, finger):
        pose = self.pad.pose(self.data, finger)
        jp = np.zeros((3, self.g.model.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jac(self.g.model, self.data, jp, jr,
                       pose['position_world_m'], self.pad.ids[finger])
        angular = jr @ self.g.C
        normal_jac = np.cross(angular.T, pose['normal_world']).T
        return pose, jp @ self.g.C * 1000., normal_jac

    def system(self, u, finger, ids, prior):
        self.g.assign(self.data, u)
        t, jt, jnt = self._pose_jac('thumb')
        f, jf, jnf = self._pose_jac(finger)
        gap = float(self.cfg.get('target_gap_mm', .9))
        delta = t['position_mm']-f['position_mm']-gap*f['normal_world']
        normal = 1.5*(t['normal_world']+f['normal_world'])
        residual = np.r_[delta, normal, .45*(u[ids]-prior[ids])]
        jac = np.vstack(((jt-jf-gap*jnf)[:, ids],
                         1.5*(jnt+jnf)[:, ids], .45*np.eye(len(ids))))
        return residual, jac

    def apply(self, frame, desired, lower, upper):
        weights, source_gaps, finger = self.weights(frame)
        report = dict(kind='source pinch proximity + bounded local pad alignment',
                      profile=self.owner.path, weights=weights,
                      source_tip_gaps_mm=source_gaps,
                      source_thresholds_mm=self.thresholds,
                      paired_finger_enabled=self.cooperative,
                      four_finger_commands_unchanged=True,
                      scope='nominal surface landmarks; no physical contact guarantee')
        if finger is None:
            self.last = None
            self.last_pair = None
            report['correction_deg'] = [0.]*5
            return np.asarray(desired).copy(), report
        ids = np.arange(5)
        if self.cooperative:
            sl = self.g.slices[finger]
            ids = np.r_[ids, np.arange(sl.start, sl.stop)]
        guard = self.pair_guards[finger] if self.cooperative else self.guard
        lo, hi = lower.copy(), upper.copy()
        if self.cooperative:
            bound = np.radians(self.cfg.get('paired_delta_limit_deg', [5.,20.,20.]))
            lo[ids[5:]] = np.maximum(lo[ids[5:]], desired[ids[5:]]-bound)
            hi[ids[5:]] = np.minimum(hi[ids[5:]], desired[ids[5:]]+bound)
        anchor = self.owner.record['anchors'][finger]
        preferred = np.asarray(desired, float).copy()
        calibrated = np.asarray(anchor['calibrated_rad'], float)
        baseline = np.asarray(anchor.get('baseline_desired_rad', anchor['baseline_rad']), float)
        preferred[ids] += calibrated[ids]-baseline[ids]
        preferred = np.clip(preferred, lo, hi)
        absolute = np.asarray(desired).copy()
        absolute[ids] = np.clip(calibrated[ids], lo[ids], hi[ids])
        seeds = [preferred, absolute, np.clip(desired, lo, hi)]
        if self.last is not None and finger == self.last_pair:
            warm = np.asarray(desired).copy()
            warm[ids] = np.clip(self.last[ids], lo[ids], hi[ids])
            seeds.insert(0, warm)
        feasible = [u for u in seeds if guard.safe(u)]
        if feasible:
            score = lambda u: float(np.sum(self.system(u, finger, ids, preferred)[0]**2))
            best = min(feasible, key=score).copy()
            iterations=int(self.cfg.get('iterations_by_pair',{}).get(finger,self.cfg.get('iterations',5)))
            for _ in range(iterations):
                residual, jac = self.system(best, finger, ids, preferred)
                if finger in self.cfg.get('bounded_step_pairs',[]):
                    # Solve the local step WITH bounds. Solving an unconstrained
                    # step then clipping can stall at CMC/MCP limits even when
                    # the remaining free axes can reduce the pad gap.
                    cap=np.radians(12.)
                    step=lsq_linear(np.vstack((jac,np.sqrt(1.5)*np.eye(len(ids)))),
                        -np.r_[residual,np.zeros(len(ids))],
                        bounds=(np.maximum(lo[ids]-best[ids],-cap),
                                np.minimum(hi[ids]-best[ids],cap)),
                        method='bvls',tol=1e-8,max_iter=30).x
                else:
                    step = -np.linalg.solve(jac.T@jac+1.5*np.eye(len(ids)), jac.T@residual)
                    step *= min(1., np.radians(12.)/max(np.max(np.abs(step)), 1e-12))
                old = float(residual@residual)
                accepted = False
                for scale in (1., .5, .2):
                    candidate = best.copy()
                    candidate[ids] = np.clip(best[ids]+scale*step, lo[ids], hi[ids])
                    if score(candidate) < old-1e-8 and guard.safe(candidate):
                        best = candidate
                        accepted = True
                        break
                if not accepted:
                    break
        else:
            # The normal projection stage owns the fallback; do not invent
            # a collision-free seed or force a joint through an obstacle.
            best = np.asarray(desired).copy()
        self.last, self.last_pair = best.copy(), finger
        self.g.assign(self.data,best)
        candidate_gap=float(np.linalg.norm(self.pad.pose(self.data,'thumb')['position_mm']-
                                         self.pad.pose(self.data,finger)['position_mm']))
        strength = weights[finger]
        result = np.asarray(desired).copy()
        result[ids] += strength*(best[ids]-result[ids])
        # Endpoint screening. Actual small filtered steps are checked by an
        # independent PairedFingerGuard on the UI/execution thread.
        baseline_distances = guard.distances(desired)
        floor = np.minimum(baseline_distances, guard.margin)-1e-5
        limited = bool(np.any(guard.distances(result) < floor))
        if limited:
            result = np.asarray(desired).copy()
        report.update(active_pair=finger, correction_deg=np.degrees(result[:5]-desired[:5]).tolist(),
                      four_finger_commands_unchanged=bool(np.array_equal(result[5:], desired[5:])),
                      paired_correction_deg=(np.degrees(result[self.g.slices[finger]]-
                                                desired[self.g.slices[finger]]).tolist()),
                      feasible_local_seed=bool(feasible),
                      full_assistance_candidate_gap_mm=candidate_gap,
                      correction_path_limited=limited,
                      local_clearance=guard.report(result))
        return result, report
