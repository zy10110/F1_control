"""Match finger segment directions in palm coordinates within native A limits."""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from hand_kinematics import FINGERS


def thumb_segments(values):
    cmc, mcp1, mcp2, ip1, ip2, tip = values
    return np.diff([cmc, (mcp1+mcp2)/2, (ip1+ip2)/2, tip], axis=0)


class SegmentAlignment:
    def __init__(self,g,settings):
        self.g=g
        self.cap=np.radians(settings['max_change_deg'])
        self.ridge=settings['posture_weight']
        self.distal=[g.distal[f] for f in FINGERS]
        self.tips=np.stack([g.tip[f] for f in FINGERS])
        thumb_axis=g.model.joint('r_thumb_mcp_abd').id
        self.anchors=np.array([[[thumb_axis,thumb_axis],g.pair_axes['thumb','mcp'],g.pair_axes['thumb','ip']],
                               *[[g.pair_axes[f,s] for s in ('mcp','pip','dip')] for f in FINGERS[1:]]])
        self.columns=[g.cols[f] for f in FINGERS]
        self.thumb_bend_plane = settings.get('thumb_bend_plane', False)

    def align_thumb(self, nominal, chains, rotation, bend_axis, shaft_weight):
        """Match the G7 thumb's three bone directions and signed flexion axis."""
        gap = min(np.linalg.norm(chains[f][-1] - chains['thumb'][-1]) for f in FINGERS[1:])
        # The existing aperture and surface calibration owns close approaches.
        t = np.clip((gap - 30.) / 30., 0., 1.)
        weight = t*t*(3.-2.*t) * (1.-shaft_weight)
        if weight <= 0.:
            return nominal
        g = self.g
        cols = g.cols['thumb']
        coupling = g.C[:, cols]
        target = np.diff(chains['thumb'], axis=0) @ rotation.T
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        flex = g.pair_axes['thumb', 'mcp']
        ip = g.pair_axes['thumb', 'ip']
        root = g.root_joints['thumb']
        q = nominal.copy()
        for _ in range(4):
            g.assign(q)
            # The SDK's first segment starts at CMC. Use the two physical
            # hinge centers for each coupled joint and the actual distal tip.
            distal = g.distal['thumb']
            tip = g.data.xpos[distal] + g.data.xmat[distal].reshape(3, 3) @ g.tip['thumb']
            points = [g.data.xanchor[root], *g.data.xanchor[flex], *g.data.xanchor[ip], tip]
            bodies = [g.model.jnt_bodyid[j] for j in (root, *flex, *ip)] + [distal]
            pos_jac = []
            for point, body in zip(points, bodies):
                jac = np.empty((3, g.model.nv))
                mujoco.mj_jac(g.model, g.data, jac, None, point, body)
                pos_jac.append(jac @ coupling)
            bones = thumb_segments(points)
            lengths = np.linalg.norm(bones, axis=1)
            dirs = bones / lengths[:, None]
            proj = np.eye(3) - dirs[:, :, None] * dirs[:, None, :]
            direction_jac = (proj @ thumb_segments(pos_jac)) / lengths[:, None, None]
            axis = g.data.xaxis[flex[0]]
            rot_jac = np.empty((3, g.model.nv))
            mujoco.mj_jacBody(g.model, g.data, None, rot_jac, bodies[1])
            axis_jac = np.cross((rot_jac @ coupling).T, axis).T
            # A soft preference in the last degrees of travel lets alternate
            # root axes share the bend. The native hard stops remain available.
            reserve = np.radians(8.)
            inside = np.clip(q[cols[:3]], g.lo[cols[:3]]+reserve, g.hi[cols[:3]]-reserve)
            barrier = inside - q[cols[:3]]
            A = np.vstack((direction_jac.reshape(9, 5), axis_jac, .20*np.eye(5),
                           .8*(barrier != 0.)[:,None]*np.eye(3, 5)))
            rhs = np.r_[(target-dirs).ravel(), bend_axis-axis, .20*(nominal[cols]-q[cols]), .8*barrier]
            lo, hi = g.lo[cols]-q[cols], g.hi[cols]-q[cols]
            step = lsq_linear(A, rhs, bounds=(lo, hi), method='bvls').x
            q[cols] += np.clip(step, lo, hi)
        return nominal + weight*(q-nominal)

    def directions(self,q):
        g=self.g;g.assign(q)
        joints=g.data.xanchor[self.anchors].mean(axis=2)
        tip=g.data.xpos[self.distal]+np.einsum('fij,fj->fi',g.data.xmat[self.distal].reshape(5,3,3),self.tips)
        bones=np.diff(np.concatenate((joints,tip[:,None,:]),axis=1),axis=1)
        return bones/np.linalg.norm(bones,axis=2,keepdims=True)

    def solve(self,nominal,chains,rotation):
        target=np.stack([np.diff(chains[f],axis=0)@rotation.T for f in FINGERS])
        target/=np.linalg.norm(target,axis=2,keepdims=True)
        desired=self.shape_features(target)
        gaps=np.array([np.linalg.norm(chains[f][-1]-chains['thumb'][-1]) for f in FINGERS[1:]])
        # Preserve the calibrated pinch pair while still matching the shape
        # of each free finger. Each input joint trajectory stays continuous.
        blend=np.clip((np.r_[gaps.min(),gaps]-10.)/30.,0.,1.)
        blend=blend**2*(3.-2.*blend)
        g=self.g;q=nominal.copy();eps=1e-4
        for _ in range(2):
            current=self.shape_features(self.directions(q))
            J=np.zeros((5,current.shape[1],5))
            # The five chains have disjoint actuators, so perturb a column
            # of every chain together. Five FK evaluations cover all 17 axes.
            for k in range(5):
                digits=[i for i,c in enumerate(self.columns) if k<len(c)]
                ids=[self.columns[i][k] for i in digits]
                dq=np.where(q[ids]+eps<=g.hi[ids],eps,-eps)
                shifted=q.copy();shifted[ids]+=dq
                moved=self.shape_features(self.directions(shifted))
                J[digits,:,k]=(moved-current)[digits]/dq[:,None]
            for i,c in enumerate(self.columns):
                if blend[i]==0:continue
                A=J[i,:,:len(c)]
                lower=np.maximum(g.lo[c],nominal[c]-self.cap*blend[i])-q[c]
                upper=np.minimum(g.hi[c],nominal[c]+self.cap*blend[i])-q[c]
                rhs=A.T@(desired[i]-current[i])+self.ridge*(nominal[c]-q[c])
                H=A.T@A+self.ridge*np.eye(len(c))
                # Re-solve on the remaining axes when an axis saturates.
                free=np.ones(len(c),bool);step=np.zeros(len(c))
                for _axis in c:
                    step[free]=np.linalg.solve(H[np.ix_(free,free)],rhs[free]-H[np.ix_(free,~free)]@step[~free])
                    bounded=np.clip(step,lower,upper)
                    viol=np.abs(bounded-step)
                    if np.max(viol)<1e-10:break
                    hit=np.argmax(viol);step[hit]=bounded[hit];free[hit]=False
                q[c]+=np.clip(step,lower,upper)
        return q

    @staticmethod
    def shape_features(v):
        # Changes in orientation between consecutive segments describe curl
        # independently of the finger's overall direction at its root.
        turn=np.cross(v[:,:-1],v[:,1:])
        dot=np.sum(v[:,:-1]*v[:,1:],axis=2)
        return np.column_stack((v.reshape(5,9),.7*turn.reshape(5,6),.7*dot))
