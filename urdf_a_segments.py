"""Match finger segment directions in palm coordinates within native A limits."""
import numpy as np
from hand_kinematics import FINGERS


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
