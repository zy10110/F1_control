"""Gesture similarity inside the original F1 coordinates and limits.

Match local bending and bending planes as well as directions, and retain
continuous thumb/finger relationships. No virtual joints or contact snapping.
"""
import numpy as np
from hand_kinematics import FINGERS
from joint_imitation_assist import JointImitationAssist
from thumb_orientation_mapping import basis


def fixed_palm_basis(g):
    """Palm axes from fixed finger mounting anchors, not flexed phalanges."""
    import mujoco
    data=mujoco.MjData(g.model);g.assign(data,g.zero)
    roots=np.array([data.xanchor[g.model.joint(g.cfg['fingers'][f]['columns'][0][0][0]).id]
                    for f in FINGERS[1:]])
    return basis(roots[0]-roots[-1],roots.mean(axis=0)-g.origin)


class ConstrainedGestureAssist(JointImitationAssist):
    kind='constrained_gesture'

    def __init__(self,g,config,reference,hand='right'):
        cfg=dict(config,imitation_cap_deg=np.degrees(g.upper-g.lower).tolist(),
            imitation_angle_prior=[6,6,6,8,8]+[8,8,8]*4,
            imitation_shape_weight=20.,imitation_dip_weight=0.,
            imitation_relative_weight=0.,imitation_warm_start=True,
            imitation_correction_speed_deg_s=90.)
        super().__init__(g,cfg,reference,hand)
        self.target_basis=fixed_palm_basis(g)
        self.bend_weight=24.
        self.plane_weight=14.

    def system(self,u,base,targets,angles):
        # All features below come from the same source skeleton; these weights
        # specify a geometric objective, not independent sensor likelihoods.
        directions,relative=targets
        self.land.assign(u);points={};jacs={};res=[self.prior*(u-base)];jac=[np.diag(self.prior)]
        for f in FINGERS:
            p,jp=self.land.points(f,True);points[f]=p;jacs[f]=jp
            v=np.diff(p,axis=0);dv=np.diff(jp,axis=0);length=np.linalg.norm(v,axis=1)
            n=v/length[:,None];dn=np.array([(np.eye(3)-np.outer(a,a))@b/c for a,b,c in zip(n,dv,length)])
            target=directions[f]
            # Give visibly bent fingers their own weight instead of letting
            # numerous nearly straight bones dominate the score.
            bend=float(np.linalg.norm(np.cross(target[0],target[1])))
            weight=self.shape_weight*np.sqrt(1.+bend)
            res.append(weight*(n-target).ravel());jac.append(weight*dn.reshape(9,len(u)))
            for k in (0,1):
                res.append(np.array([self.bend_weight*(n[k]@n[k+1]-target[k]@target[k+1])]))
                jac.append(self.bend_weight*(n[k+1]@dn[k]+n[k]@dn[k+1])[None,:])
                # Unnormalised cross products vanish smoothly for straight
                # bones, where a bending-plane normal would be undefined.
                res.append(self.plane_weight*(np.cross(n[k],n[k+1])-np.cross(target[k],target[k+1])))
                jac.append(self.plane_weight*(np.cross(dn[k].T,n[k+1])+np.cross(n[k],dn[k+1].T)).T)
        for f in FINGERS[1:]:
            reach=min(self.lengths['thumb'],self.lengths[f])
            approach=float(np.clip(1.-np.linalg.norm(relative[f])/reach,0.,1.))
            w=.08+.65*approach*approach
            res.append(w*(points['thumb'][-1]-points[f][-1]-relative[f]))
            jac.append(w*(jacs['thumb'][-1]-jacs[f][-1]))
        return np.concatenate(res),np.vstack(jac)

    def metrics(self,u,targets):
        result=super().metrics(u,targets)
        result['bend_errors_deg']={}
        for f in FINGERS:
            p=self.land.points(f);v=np.diff(p,axis=0);n=v/np.linalg.norm(v,axis=1)[:,None]
            target=targets[0][f]
            actual=np.degrees(np.arccos(np.clip(np.sum(n[:-1]*n[1:],axis=1),-1,1)))
            source=np.degrees(np.arccos(np.clip(np.sum(target[:-1]*target[1:],axis=1),-1,1)))
            result['bend_errors_deg'][f]=actual-source
        result['worst_finger_mean_direction_error_deg']=float(max(np.mean(x) for x in result['bone_errors_deg'].values()))
        return result

    def apply(self,*args,**kwargs):
        u,report=super().apply(*args,**kwargs)
        report.update(kind=self.kind,original_model_bounds=True,
            constraint_scope='active F1 model limits and original C; no added DOF',
            objective='joint prior + bone directions + local bends + bend planes + continuous relative vectors')
        return u,report
