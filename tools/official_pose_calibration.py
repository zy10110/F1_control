"""Calibrate the simulated F1 from the solved HumanHand skeleton."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from g7_joint_mapping import rotations
from g7_teleop import BASE
from hand_kinematics import FINGERS, ROOT
from tools.calibrate_angle_map import load_groups, write
from tools.contact_regression import check
from tools.validate_left_natural import replay
from urdf_a_follow import Follower
from urdf_a_model import NativeHand


TAKES=[
    ('train','open_static','手掌平放张开',2,'让官方手模五指自然伸开，保持。'),
    ('train','fist_static','静态握拳',2,'让官方手模握成一个舒服的拳，保持。'),
    ('train','repeated_fist','重复握拳',5,'从张开开始，反复握拳、松开，让官方手模走过完整行程。'),
    ('train','thumb_motion','拇指活动',5,'四指放松，拇指从掌边展开，再扫过掌心，来回活动。'),
    *[('train',f,'拇指贴'+name,2,'看官方手模，让拇指与'+name+'轻贴，保持。')
      for f,name in zip(FINGERS[1:],('食指','中指','无名指','小指'))]
]
MOVING={'repeated_fist','thumb_motion'}


def fit_reference(g,old,record,groups):
    samples={f:[] for f in FINGERS}
    for entry,frames in groups:
        for frame in frames:
            points=np.asarray(frame['positions_m'])*1000.
            points=rotations(frame)[0].inv().apply(points-points[0])@old.display_rotation.T
            directions=[];separations=[]
            for f in FINGERS:
                i=BASE[f]
                bones=np.diff(points[i:i+4],axis=0)
                lengths=np.linalg.norm(bones,axis=1)
                if np.min(lengths)<1.:raise ValueError('官方指节位置不完整，请重采该项')
                directions.append(bones/lengths[:,None])
                if f!='thumb':separations.append(np.linalg.norm(points[i+3]-points[4]))
            directions=np.array(directions);separations=np.array(separations)
            nominal=old.command(frame).copy()
            features,_=old.source.read(frame)
            def residual(q):
                g.assign(q)
                bones=np.array([g.bone_directions(f) for f in FINGERS])
                pads=np.array([g.pad_pose(f)[0] for f in FINGERS])
                gaps=np.linalg.norm(pads[1:]-pads[0],axis=1)
                return np.r_[(bones-directions).ravel(),(gaps-np.maximum(2.5,separations))/(8.+.6*separations),
                             .9*(q-nominal),max(0.,.9-g.clearance())]
            budget=np.radians(8.)
            q=least_squares(residual,nominal,bounds=(np.maximum(g.lo,nominal-budget),np.minimum(g.hi,nominal+budget)),
                            max_nfev=20).x
            for f in FINGERS:
                samples[f].append((features[f]-old.blocks[f][0],(q-nominal)[g.cols[f]],1./len(frames)))
    for f,items in samples.items():
        centers=np.array([i[0] for i in items]);values=np.array([i[1] for i in items])
        weights=np.array([i[2] for i in items])
        block=record.setdefault('official_pose_updates',{}).setdefault(f,dict(width=float(np.radians(22)),centers=[],coefficients=[]))
        K=np.exp(-.5*np.sum(((centers[:,None]-centers[None,:])/block['width'])**2,axis=2))
        coef=np.linalg.solve(K.T@(weights[:,None]*K)+.2*np.eye(len(K)),K.T@(weights[:,None]*values))
        block['centers'].extend(centers.tolist());block['coefficients'].extend(coef.tolist())
    return record


def fit_round(folder):
    manifest=json.loads((folder/'manifest.json').read_bytes())
    if [e['stage'] for e in manifest['train']]!=[t[1] for t in TAKES]:raise ValueError('官方手模八段动作尚未采完')
    base=folder/'baseline_profile.json';dest=folder/'profile.json'
    record=json.loads(base.read_bytes())
    groups=load_groups(folder/'manifest.json','train')
    if record.get('input_hand')!='left' or any(fr.get('pose_space')!='mirrored_right' or fr.get('source_hand')!=1 for _,frames in groups for fr in frames):
        raise ValueError('本轮需要左手官方解算结果')
    g=NativeHand(ROOT/'assets/urdf_a')
    fit_reference(g,Follower(g,base),record,groups)
    record['official_reference_session']=str((folder/'manifest.json').resolve());write(dest,record)


def validate_round(folder):
    base=folder/'baseline_profile.json'
    dest=folder/'profile.json'
    record=json.loads(dest.read_bytes())
    g=NativeHand(ROOT/'assets/urdf_a')
    natural=Path(record['supplement'])
    prior=json.loads(natural.read_bytes())
    if prior.get('input_hand')!='left':raise ValueError('缺少左手独立回放记录')
    report={};issues=[]
    for entry in prior['validation']:
        rows=[json.loads(s) for s in (natural.parent/entry['file']).read_text().splitlines()][::3]
        a,b=[replay(g,p,rows,[]) for p in (base,dest)]
        report[entry['stage']]=dict(previous=a,refitted=b)
        for key,slack in [('mean_direction_deg',1.),('near_error_mm',1.),('contact_pad_mm',1.),('false_closures',0),('blocked_frames',0)]:
            if a[key] is not None and b[key]>a[key]+slack:issues.append(entry['stage']+' '+key)
    if not issues:
        try:check(ROOT/'assets/urdf_a',base,dest)
        except ValueError as exc:issues.append(str(exc))
    write(folder/'validation.json',dict(recording=str(natural),validation=report,issues=issues,accepted=not issues))
    if issues:raise ValueError('独立回放退步：'+'、'.join(issues))
    return json.loads(dest.read_bytes())


def fit_and_validate(folder):
    fit_round(folder)
    return validate_round(folder)
