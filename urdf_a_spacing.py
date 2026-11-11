"""Soft, angle-derived finger spacing for the native A simulation."""
import mujoco
import numpy as np

from hand_kinematics import FINGERS


def spacing_goal(g, nominal, chains, settings):
    """Refine a nominal joint pose, without a nearest-finger or pinch state.

    Near touch, a smaller nominal gap may already be correct (the glove
    reports bone tips; the robot uses surface landmarks). Profiles calibrated
    for open gaps also correct fingers that pass too close to each other.
    All four pairs participate continuously. Retain the measured nonzero
    separation and solve in the 17 native semantic axes, not the 37 hinges.
    """
    src=np.array([np.linalg.norm(chains[f][-1]-chains['thumb'][-1]) for f in FINGERS[1:]])
    # Smooth maximum of observed spacing and a nonzero pad-marker allowance.
    target=(src**4+settings['pad_allowance_mm']**4)**.25
    weight=1/(1+(src/settings['range_mm'])**6)
    fade=np.clip(2-src/settings['range_mm'],0.,1.)
    weight*=fade**2*(3-2*fade)
    cap=np.radians(settings['max_change_deg'])
    damping=settings['posture_mm_per_rad']
    q=nominal.copy()
    lo=np.maximum(g.lo,nominal-cap);hi=np.minimum(g.hi,nominal+cap)
    influence=None
    for iteration in range(settings['iterations']):
        g.assign(q)
        pads=np.array([g.pad_pose(f)[0] for f in FINGERS])
        vectors=pads[1:]-pads[0]; lengths=np.linalg.norm(vectors,axis=1)
        if influence is None:
            difference=lengths-target
            excess=np.maximum(difference,0.)
            influence=weight*excess**2/(excess**2+1.)
            if settings.get('preserve_open_gap'):
                # The thumb is shared. As it approaches any finger, taper
                # opening corrections so another digit cannot pull it away.
                t=np.clip((src.min()-16.)/10.,0.,1.)
                shortfall=np.maximum(-difference,0.)
                influence+=weight*t*t*(3.-2.*t)*shortfall**2/(shortfall**2+1.)
            if not np.any(influence):break
        jacs=[]
        for f,pt in zip(FINGERS,pads):
            jp=np.zeros((3,g.model.nv))
            mujoco.mj_jac(g.model,g.data,jp,None,pt*.001,g.distal[f])
            jacs.append(1000*jp@g.C)
        J=np.array(jacs[1:])-jacs[0]
        normals=np.array([g.pad_pose(f)[1] for f in FINGERS])
        facing=normals[0]-normals[1:]
        facing/=np.maximum(np.linalg.norm(facing,axis=1,keepdims=True),1.e-8)
        if iteration==0:
            radial=vectors/np.maximum(lengths[:,None],1.e-8)
        # Near opposition, bring opposing pad surfaces face to face. Farther
        # away, keep the nominal direction and the wearer's positive gap.
        mix=1/(1+(src/8.)**4)
        direction=mix[:,None]*facing+(1-mix[:,None])*radial
        direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1.e-8)
        A=(influence[:,None,None]*J).reshape(-1,len(q))
        residual=(influence[:,None]*(vectors-direction*target[:,None])).ravel()
        # Add linearized clearance residuals before the execution guard: a
        # smaller nominal gap behind a fingertip shell cannot be executed.
        a,b=g.pairs.T;model,data=g.model,g.data
        bounds=np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b],axis=1)-model.geom_rbound[a]-model.geom_rbound[b]
        rows=[];errors=[]
        for ga,gb in g.pairs[bounds<.0012]:
            points=np.zeros(6);dist=mujoco.mj_geomDistance(model,data,ga,gb,.0012,points)
            if dist>=.00105 or abs(dist)<1.e-10:continue
            ja=np.zeros((3,model.nv));jb=np.zeros_like(ja)
            mujoco.mj_jac(model,data,ja,None,points[:3],model.geom_bodyid[ga])
            mujoco.mj_jac(model,data,jb,None,points[3:],model.geom_bodyid[gb])
            strength=3.*np.max(influence)
            rows.append(strength*1000*((points[3:]-points[:3])/dist)@(jb-ja)@g.C)
            errors.append(strength*(1000*dist-1.05))
        if rows:
            A=np.vstack((A,rows));residual=np.r_[residual,errors]
        pull=nominal-q
        # Damped least squares; freeze outward hard-stop axes and
        # re-solve so another available axis can contribute.
        free=np.ones(len(q),dtype=bool)
        for _ in range(4):
            B=A[:,free]
            step=np.zeros_like(q)
            step[free]=np.linalg.solve(B.T@B+np.eye(B.shape[1])*damping**2,damping**2*pull[free]-B.T@residual)
            blocked=((q<=lo+1e-9)&(step<0))|((q>=hi-1e-9)&(step>0))
            if not np.any(blocked&free):break
            free &= ~blocked
        q=np.clip(q+.8*step,lo,hi)
    return q
