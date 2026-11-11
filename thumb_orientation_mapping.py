"""Convert a glove thumb's root rotation to the actual F1 root joint axes.

No fingertip-position objective or bone-length scaling. The three bounded
root coordinates represent a rotation in a common anatomical palm frame.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def unit(value):
    length=np.linalg.norm(value)
    if not np.isfinite(length) or length<1e-8:raise ValueError('拇指映射的掌心基底退化')
    return np.asarray(value)/length


def basis(radial,distal):
    x=unit(radial);y=unit(distal-x*np.dot(x,distal));z=unit(np.cross(x,y))
    return np.column_stack((x,y,z))


class ThumbOrientationMapper:
    def __init__(self,geometry,reference,source_frame,source_root):
        self.g=geometry;self.reference=reference.copy();self.data=mujoco.MjData(geometry.model)
        p=np.asarray(source_frame['positions_m'],float)
        if p.shape!=(21,3) or not np.isfinite(p).all():raise ValueError('拇指掌心对齐需要21个有效参考点')
        source_basis=basis(p[5]-p[17],p[[5,9,13,17]].mean(axis=0)-p[0])
        self.g.assign(self.data,self.reference);roots=[];rays=[]
        for f in ('index','middle','ring','pinky'):
            name=self.g.cfg['fingers'][f]['columns'][0][0][0]
            root=self.data.xanchor[self.g.model.joint(name).id].copy();roots.append(root)
            rays.append(self.data.site_xpos[self.g.sites[f]]-root)
        target_basis=basis(roots[0]-roots[-1],np.mean(rays,axis=0))
        self.palm_rotation=Rotation.from_matrix(target_basis@source_basis.T)
        self.source_root=source_root
        self.body=self.g.model.body('f1_link_3').id
        self.target_root=Rotation.from_matrix(self.data.xmat[self.body].reshape(3,3))

    def target_rotation(self,source_root):
        delta=source_root*self.source_root.inv()
        return self.palm_rotation*delta*self.palm_rotation.inv()*self.target_root

    def solve(self,source_root,previous):
        target=self.target_rotation(source_root)
        return self.solve_target(target,previous)

    def solve_target(self,target,previous):
        def residual(value):
            u=self.reference.copy();u[:3]=value;self.g.assign(self.data,u)
            current=Rotation.from_matrix(self.data.xmat[self.body].reshape(3,3))
            return (target.inv()*current).as_rotvec()
        start=np.clip(previous[:3],self.g.lower[:3]+1e-10,self.g.upper[:3]-1e-10)
        fit=least_squares(residual,start,bounds=(self.g.lower[:3],self.g.upper[:3]),
                          max_nfev=14,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        value=np.clip(fit.x,self.g.lower[:3],self.g.upper[:3])
        error=float(np.degrees(np.linalg.norm(residual(value))))
        return value,dict(method='palm-frame bone orientation -> bounded actual thumb-root axes',
                          orientation_residual_deg=error,at_limit=bool(np.any(np.minimum(
                              value-self.g.lower[:3],self.g.upper[:3]-value)<np.radians(.1))),
                          fingertip_position_used=False)


class ThumbBoneFrameMapper(ThumbOrientationMapper):
    """Match the metacarpal ray and physical MCP bending plane.

    CAD link frames have arbitrary orientation. A wrist-frame rotation delta
    alone cannot align the human and F1 hinge planes. This frame uses two
    physical directions, with no fingertip position or length objective.
    """
    def __init__(self,geometry,reference,source_frame,source_root):
        super().__init__(geometry,reference,source_frame,source_root)
        self.base_joint=self.g.model.joint('axis_2').id
        self.mcp_joint=self.g.model.joint('axis_21').id
        self.ip_joint=self.g.model.joint('axis_22').id
        self.proximal=self.g.model.body('f1_link_5').id
        self.distal_base=self.g.model.body('f1_link_6').id
        self.distal=self.g.model.body('f1_link_7').id
        self.g.assign(self.data,self.reference)
        ray=self.data.xanchor[self.mcp_joint]-self.data.xanchor[self.base_joint]
        axis=self.data.xaxis[self.mcp_joint]
        self.local_frame=basis(self.target_root.inv().apply(ray),self.target_root.inv().apply(axis))
        self.flexion_offsets_deg=self.geometric_flexion(self.reference)

    def rays(self):
        return (self.data.xanchor[self.mcp_joint]-self.data.xanchor[self.base_joint],
                self.data.xpos[self.distal_base]-self.data.xpos[self.proximal],
                self.data.site_xpos[self.g.sites['thumb']]-self.data.xpos[self.distal])

    def geometric_flexion(self,u):
        self.g.assign(self.data,u)
        a,b,c=(unit(v) for v in self.rays())
        def signed(v,w,axis):
            return np.degrees(np.arctan2(np.dot(axis,np.cross(v,w)),np.dot(v,w)))
        return np.array([signed(a,b,self.data.xaxis[self.mcp_joint]),
                         signed(b,c,self.data.xaxis[self.ip_joint])])

    def solve_frame(self,frame,previous):
        q=np.asarray(frame['quaternions_wxyz'],float)
        if q.shape!=(21,4) or not np.isfinite(q).all() or np.min(np.linalg.norm(q,axis=1))<1e-6:
            raise ValueError('拇指骨段对齐需要21个有效四元数')
        rotations=Rotation.from_quat(q[:,[1,2,3,0]])
        wrist=rotations[0].inv()
        ray=wrist.apply(rotations[1].apply([0.,-1.,0.]))
        # The MCP hinge normal is carried by the proximal bone: rotating
        # about local Z leaves this direction invariant, including axial roll.
        axis=wrist.apply(rotations[2].apply([0.,0.,-1.]))
        source_frame=basis(ray,axis)
        target=self.palm_rotation*Rotation.from_matrix(source_frame@self.local_frame.T)
        value,report=self.solve_target(target,previous)
        u=self.reference.copy();u[:3]=value;self.g.assign(self.data,u)
        actual_ray=unit(self.rays()[0]);actual_axis=self.data.xaxis[self.mcp_joint]
        wanted_ray=self.palm_rotation.apply(ray);wanted_axis=self.palm_rotation.apply(axis)
        def error(a,b):return float(np.degrees(np.arccos(np.clip(np.dot(unit(a),unit(b)),-1.,1.))))
        report.update(method='anatomical metacarpal ray + MCP bend plane -> bounded F1 root axes',
            metacarpal_direction_error_deg=error(actual_ray,wanted_ray),
            mcp_bend_axis_error_deg=error(actual_axis,wanted_axis),
            f1_flexion_zero_offsets_deg=self.flexion_offsets_deg.tolist())
        return value,report

    def fit_bone_directions(self,frame,initial,previous,lower,upper):
        """Bounded thumb-only shape projection using three unit bone rays.

        Human MCP/IP bend planes can differ, while F1's two hinges are
        parallel. Fit the whole shape when an exact angle copy is impossible.
        All FK evaluations use the existing native half-angle reduction.
        """
        q=np.asarray(frame['quaternions_wxyz'],float)
        rotations=Rotation.from_quat(q[:,[1,2,3,0]])
        wrist=rotations[0].inv()
        wanted=np.asarray([self.palm_rotation.apply((wrist*rotations[i]).apply([0.,-1.,0.]))
                           for i in (1,2,3)])
        wanted_axis=self.palm_rotation.apply((wrist*rotations[2]).apply([0.,0.,-1.]))
        preferred=initial[:5].copy()
        def residual(value):
            u=initial.copy();u[:5]=value;self.g.assign(self.data,u)
            rays=np.asarray([unit(v) for v in self.rays()])
            return np.r_[(rays-wanted).ravel(),
                .10*(self.data.xaxis[self.mcp_joint]-wanted_axis),
                .025*(value-preferred)]
        seeds=[np.clip(initial[:5],lower[:5]+1e-10,upper[:5]-1e-10),
               np.clip(previous[:5],lower[:5]+1e-10,upper[:5]-1e-10)]
        start=min(seeds,key=lambda v:np.linalg.norm(residual(v)))
        fit=least_squares(residual,start,bounds=(lower[:5],upper[:5]),max_nfev=18,
            ftol=1e-6,xtol=1e-6,gtol=1e-6)
        value=np.clip(fit.x,lower[:5],upper[:5]);residual(value)
        rays=np.asarray([unit(v) for v in self.rays()])
        errors=np.degrees(np.arccos(np.clip(np.sum(rays*wanted,axis=1),-1.,1.)))
        return value,dict(method='three unit thumb bone directions; no endpoint or length target',
            bone_direction_errors_deg=errors.tolist(),iterations=int(fit.nfev),
            mcp_bend_axis_error_deg=float(np.degrees(np.arccos(np.clip(
                np.dot(self.data.xaxis[self.mcp_joint],wanted_axis),-1.,1.)))),
            at_limit=bool(np.any(np.minimum(value-lower[:5],upper[:5]-value)<np.radians(.1))),
            geometric_bends_deg=self.geometric_flexion(np.r_[value,initial[5:]]).tolist())
