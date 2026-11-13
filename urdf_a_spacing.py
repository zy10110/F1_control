"""Soft, angle-derived finger spacing for the native A simulation."""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from hand_kinematics import FINGERS


def clearance_linearization(g, clearance_mm, pairs=None, offsets_mm=None):
    """Signed gap and derivative at the closest witnesses of each near pair."""
    pairs = g.pairs if pairs is None else pairs
    a,b = pairs.T
    model,data = g.model,g.data
    radius = clearance_mm*.001
    bounds = np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b],axis=1)-model.geom_rbound[a]-model.geom_rbound[b]
    rows, shortfall = [], []
    for index in np.flatnonzero(bounds < radius):
        ga,gb=pairs[index]
        points = np.zeros(6)
        dist = mujoco.mj_geomDistance(model,data,ga,gb,radius,points)
        target=clearance_mm if offsets_mm is None else clearance_mm+offsets_mm[index]
        if dist >= target*.001 or abs(dist) < 1.e-10:
            continue
        ja=np.empty((3,model.nv));jb=np.empty_like(ja)
        mujoco.mj_jac(model,data,ja,None,points[:3],model.geom_bodyid[ga])
        mujoco.mj_jac(model,data,jb,None,points[3:],model.geom_bodyid[gb])
        rows.append(1000*((points[3:]-points[:3])/dist)@(jb-ja)@g.C)
        shortfall.append(target-1000.*dist)
    return np.asarray(rows).reshape(-1,len(g.lo)), np.asarray(shortfall)


def spacing_goal(g, nominal, chains, settings, *, source_vectors=None, motion_weight=0., avoid_shells=True):
    """Refine a nominal joint pose, without a nearest-finger or pinch state.

    Near touch, a smaller nominal gap may already be correct (the glove
    reports bone tips; the robot uses surface landmarks). Profiles calibrated
    for open gaps also correct fingers that pass too close to each other.
    All four pairs participate continuously. Retain the measured nonzero
    separation and solve in the 17 native semantic axes, not the 37 hinges.
    """
    src=np.array([np.linalg.norm(chains[f][-1]-chains['thumb'][-1]) for f in FINGERS[1:]])
    second = np.partition(src,1)[1]
    crowd = np.clip((settings['range_mm']+8.-second)/12., 0., 1.)
    crowd = crowd*crowd*(3.-2.*crowd)
    # Smooth maximum of observed spacing and a nonzero pad-marker allowance.
    target=(src**4+settings['pad_allowance_mm']**4)**.25
    weight=1/(1+(src/settings['range_mm'])**6)
    fade=np.clip(2-src/settings['range_mm'],0.,1.)
    weight*=fade**2*(3-2*fade)
    # A demonstrated fingertip gap is a scalar measurement. Let the pads
    # slide tangentially while tracking it, especially during a near pass.
    # Blend back into the calibrated facing constraint at very small gaps.
    aperture_range = settings.get('aperture_range_mm')
    if aperture_range is not None:
        t = np.clip((src.min() - 3.) / 7., 0., 1.)
        aperture_blend = t*t*(3.-2.*t)
        aperture_weight = aperture_blend / (1.+(src/aperture_range)**6)
        taper = np.clip(2.-src/aperture_range, 0., 1.)
        aperture_weight *= taper*taper*(3.-2.*taper)
    # A scalar aperture cannot distinguish opposite approach directions.
    # During transitions also match the wrist-relative SDK tip vectors.
    direction_weight = np.zeros(4)
    if source_vectors is not None and aperture_range is not None and motion_weight>1.e-4:
        t = np.clip((src.min()-4.)/14.,0.,1.)
        band = t*t*(3.-2.*t)*motion_weight
        direction_weight = band / (1.+(src/100.)**4)
    cap=np.radians(settings['max_change_deg']+24.*direction_weight.max())
    direction_weight *= .9
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
            # At a shared contact a millimetre of nominal pad error must
            # not switch one finger's pull on/off relative to the others.
            influence = (1.-crowd)*influence + crowd*weight
            if aperture_range is not None:
                influence *= 1.-aperture_blend
            if not np.any(direction_weight) and not np.any(influence) and (aperture_range is None or not np.any(aperture_weight)):break
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
        clearance_weight = np.max(influence)
        if aperture_range is not None:
            unit = vectors / np.maximum(lengths[:, None], 1.e-8)
            # Coincident markers have no radial derivative. The facing
            # direction supplies a separation direction at that pose.
            unit = np.where(lengths[:, None] > 1.e-8, unit, facing)
            radial_jac = np.einsum('fi,fij->fj', unit, J)
            A = np.vstack((A, aperture_weight[:, None]*radial_jac))
            residual = np.r_[residual, aperture_weight*(lengths-target)]
            clearance_weight = max(clearance_weight, np.max(aperture_weight))
        if direction_weight.any():
            A=np.vstack((A,(direction_weight[:,None,None]*J).reshape(-1,len(q))))
            residual=np.r_[residual,(direction_weight[:,None]*(vectors-source_vectors)).ravel()]
        # Add linearized clearance residuals before the execution guard: a
        # smaller nominal gap behind a fingertip shell cannot be executed.
        if avoid_shells and crowd < 1.:
            rows, shortfall = clearance_linearization(g, 1.05)
            strength = 3.*clearance_weight*(1.-crowd)
            A=np.vstack((A,strength*rows));residual=np.r_[residual,-strength*shortfall]
        pull=nominal-q
        # Damped least squares; freeze outward hard-stop axes and
        # re-solve so another available axis can contribute.
        step=lsq_linear(np.vstack((A,damping*np.eye(len(q)))),np.r_[-residual,damping*pull],
                        bounds=(lo-q,hi-q),method='bvls').x
        q=np.clip(q+.8*step,lo,hi)
    if avoid_shells and crowd > 0.:
        # With several assigned pads the aperture objectives can demand an
        # intersecting pose. Project the target itself before executing it.
        unprojected = q.copy()
        regularizer = np.diag([1.5]*5+[8.]*12)
        for _ in range(6):
            g.assign(q)
            rows, shortfall = clearance_linearization(g, .85)
            if not shortfall.size or shortfall.max() < .005:
                break
            step = lsq_linear(np.vstack((rows,regularizer)), np.r_[shortfall,np.zeros(len(q))],
                              bounds=(g.lo-q,g.hi-q), method='bvls').x
            q += np.clip(step,-.12,.12)
        q=unprojected+crowd*(q-unprojected)
    return q
