"""Fit the A geometry only on training takes; keep the legacy profile intact."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_urdf_a_mapping import compare_axes, fit_zero_offsets, verify_fk
from tools.calibrate_angle_map import load_groups, check_split, write
from urdf_a_model import NativeHand
from urdf_a_follow import SourceHand, angular_adjustment
from hand_kinematics import FINGERS, HandGeometry
from calibrated_angle_mapping import CalibratedAngleRetargeter
from g7_official_udp import official_config
from g7_teleop import load_g7_config, BASE


def left_supplement_settings(original):
    settings={k:original[k] for k in ('input_hand','source_geometry','angle_reference','mirror_base_sha256',
        'pinch_angle_updates','pinch_round','retained_contact_records','official_pose_updates','official_reference_session','segment_alignment','fine_motion_response','joint_surface_touch') if k in original}
    if 'angle_derived_spacing' in original:
        settings['angle_derived_spacing']=dict(original['angle_derived_spacing'],preserve_open_gap=True)
    return settings


def update_left_natural_fit(record,previous):
    """Limit how far one additional natural-motion take moves the existing fit."""
    prior_manifest=Path(previous['supplement']) if previous.get('supplement') else None
    if not prior_manifest or not prior_manifest.is_file() or json.loads(prior_manifest.read_bytes()).get('input_hand')!='left':
        return record
    fraction=.15
    for finger,block in record['blocks'].items():
        old=previous['blocks'][finger]
        offset=np.asarray(block['zero'])-old['zero']
        matrix=np.asarray(old['matrix'])
        block['bias']=(1-fraction)*(old['bias']+matrix@offset)+fraction*np.asarray(block['bias'])
        block['matrix']=(1-fraction)*matrix+fraction*np.asarray(block['matrix'])
        # Combine Gaussian residuals in the new zero's coordinates. Sum
        # coefficients at repeated sample centers so they are evaluated once.
        a=previous['angular_corrections'][finger];b=record['angular_corrections'][finger]
        if not np.isclose(a['width'],b['width']):raise ValueError('两轮关节修正的角度带宽不一致')
        centers=np.vstack((np.asarray(a['centers'])-offset,b['centers']))
        values=np.vstack(((1-fraction)*np.asarray(a['coefficients']),fraction*np.asarray(b['coefficients'])))
        unique,index=np.unique(centers,axis=0,return_inverse=True)
        coef=np.zeros((len(unique),values.shape[1]));np.add.at(coef,index,values)
        b.update(centers=unique,coefficients=coef)
    return record


def fit(package,folder,supplement=None):
    manifest=folder/'manifest.json'
    check_split(manifest)
    groups=load_groups(manifest,'train')
    missing=[]
    if supplement is not None:
        missing=json.loads(supplement.read_text(encoding="utf8")).get("unavailable_fingers",[])
        check_split(supplement)
        groups += [(dict(e,supplemental=True),frames) for e,frames in load_groups(supplement,'train')]
    stage_frames={e['stage']:frames[0] for e,frames in groups}
    neutral=stage_frames['open']
    migrated={}
    if supplement and json.loads(supplement.read_bytes()).get('input_hand')=='left':
        original=json.loads((supplement.parent/'baseline_profile.json').read_bytes())
        if original.get('input_hand')!='left':raise ValueError('左手补录缺少独立镜像基准')
        neutral=original['neutral']
        migrated=left_supplement_settings(original)
    elif supplement and (supplement.parent/'baseline_profile.json').is_file():
        original=json.loads((supplement.parent/'baseline_profile.json').read_bytes())
        if original.get('fine_motion_response') and original.get('input_hand','right')=='right':
            migrated={k:original[k] for k in ('fine_motion_response','angle_derived_spacing','segment_alignment','joint_surface_touch') if k in original}
    source=SourceHand(neutral)
    g=NativeHand(package);fk=verify_fk(package,g)
    old=HandGeometry('simulation_straight')
    ref=np.radians(old.simulation_profile['initial_coordinates_deg'])
    prior=CalibratedAngleRetargeter(old,ref,'right',official_config(load_g7_config()),folder/'profile.json')
    signs,_,_,R=compare_axes(old,g)
    bias,_=fit_zero_offsets(old,g,signs,R)
    wrist_to_model=R@prior.wrist_to_f1
    zeros,_=source.read(neutral)
    # Scale relative distances by the ratio of palm widths at the four
    # finger roots, a dimension common to the wrist-relative skeletons.
    g.assign(np.zeros(17))
    width=np.linalg.norm(g.data.xanchor[g.root_joints['index']]-g.data.xanchor[g.root_joints['pinky']])*1000
    size_ratio=width/np.linalg.norm(source.roots['index']-source.roots['pinky'])
    data={f:[] for f in FINGERS}
    baseline=json.loads((supplement.parent/'baseline_profile.json').read_text(encoding='utf8')) if missing else None
    blocks={}
    corrections={}
    for entry,frames in groups:
        digits=[f for f in FINGERS if not (entry.get('supplemental') and f in missing)]
        others=digits[1:]
        active=np.concatenate([g.cols[f] for f in digits])
        for fr in frames:
            if 'middle' not in digits:
                q=np.array(fr['quaternions_wxyz'],copy=True)
                q[9:13]=np.asarray(neutral['quaternions_wxyz'])[9:13]
                fr=dict(fr,quaternions_wxyz=q)
            features,chains=source.read(fr)
            desired=np.clip(prior.angle_profile.raw(prior.decode(fr))*signs+bias,g.lo,g.hi)
            if 'middle' not in digits:
                desired[g.cols['middle']]=np.clip(baseline['blocks']['middle']['bias'],g.lo[g.cols['middle']],g.hi[g.cols['middle']])
            target_dirs={}
            for f in digits:
                bones=np.diff(chains[f],axis=0)@wrist_to_model.T
                target_dirs[f]=bones/np.linalg.norm(bones,axis=1,keepdims=True)
            # Use the measured 3-D relationships as soft training observations
            # for *all* poses. The target preserves direction and separation.
            # Only each digit's own angular channels are used by the live fit.
            tip_vectors=np.array([(chains[f][-1]-chains['thumb'][-1])@wrist_to_model.T for f in others])
            def residual(active_angles):
                x=desired.copy();x[active]=active_angles
                g.assign(x)
                shape=np.concatenate([(g.bone_directions(f)-target_dirs[f]).ravel() for f in digits])
                thumb=g.pad_pose('thumb')[0]
                delta=np.array([g.pad_pose(f)[0]-thumb for f in others])-size_ratio*tip_vectors
                vector_error=delta/(9.+.52*np.linalg.norm(tip_vectors,axis=1)[:,None])
                clearance_error=max(0.,1.1-g.clearance())/3.
                return np.r_[shape,vector_error.ravel(),.24*(x-desired),clearance_error]
            sol=least_squares(residual,desired[active],bounds=(g.lo[active],g.hi[active]),max_nfev=40)
            fitted=desired.copy();fitted[active]=sol.x
            for f in digits:
                data[f].append((features[f]-zeros[f],fitted[g.cols[f]],min(1.,6/len(frames)),entry['stage']))
    for f,samples in data.items():
        X=np.array([s[0] for s in samples])
        Y=np.array([s[1] for s in samples])
        initial=signs[g.cols[f],None]*prior.angle_profile.blocks[f][2]
        w=np.sqrt([s[2] for s in samples])[:,None]
        A=np.column_stack((np.ones(len(X)),X))*w
        prior_bias=next(s[1] for s in samples if s[3]=='open')
        lam=np.diag([3.,*([.25]*X.shape[1])])
        coef=np.linalg.solve(A.T@A+lam,A.T@(Y*w)+lam@np.vstack((prior_bias,initial.T)))
        blocks[f]=dict(zero=zeros[f],bias=coef[0],matrix=coef[1:].T)
        # Broad, regularized residuals account for the change in rotation axes
        # across the workspace. All poses use the same digit-local function.
        width=np.radians(42.)
        K=np.exp(-.5*np.sum(((X[:,None]-X[None,:])/width)**2,axis=2))
        residual=Y-(coef[0]+X@coef[1:])
        coeff=np.linalg.solve(K+np.diag(.12/w[:,0]**2),residual)
        corrections[f]=dict(centers=X,coefficients=coeff,width=width)
    def raw(fr):
        features,_=source.read(fr)
        values=[]
        for f,b in blocks.items():
            x=features[f]-b['zero']
            values.append(b['bias']+b['matrix']@x+angular_adjustment(x,**corrections[f]))
        return np.concatenate(values)
    flex=np.array([raw(fr)[9:11] for _,frames in groups for fr in frames])
    widths=np.clip(np.r_[g.lo[9:11]-flex.min(0),flex.max(0)-g.hi[9:11]]+np.radians(4),np.radians(6),np.radians(18))
    if 'middle' in missing:
        blocks['middle']=baseline['blocks']['middle']
        corrections['middle']=baseline['angular_corrections']['middle']
        widths=baseline['middle_widths']
    record=dict(**migrated,middle_unavailable='middle' in missing,kind='urdf_a_natural_v2',geometry_id=g.geometry_id,urdf_sha256=g.sha,neutral=neutral,
                blocks=blocks,angular_corrections=corrections,middle_widths=widths,wrist_to_model=wrist_to_model,
                supplement=str(supplement.resolve()) if supplement is not None else None,
                training_manifest=str(manifest.resolve()),training_stages=list(stage_frames),fk_crosscheck=fk)
    return update_left_natural_fit(record,original) if migrated.get('input_hand')=='left' else record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path);parser.add_argument('folder',type=Path)
    parser.add_argument('--supplement',type=Path)
    args=parser.parse_args()
    write(args.folder/'profile_urdf_a_natural.json',fit(args.package,args.folder,args.supplement))

