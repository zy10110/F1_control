"""Example-based simulation retargeting; no inferred physical glove angles.

Open/fist recordings define per-finger flexion features. Labeled thumb-index
examples provide local pose corrections, fitted against nominal F1 pad sites.
This is an approximate action map, not calibrated human fingertip tracking.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from hand_kinematics import ROOT, FINGERS, serializable
from palm_calibration import PadGeometry, build_pad_candidates, write_json


def delta(a,b):
    return (np.asarray(a,float)-np.asarray(b,float)+180.)%360.-180.


def kernel(distance,radius=.85):
    z=np.maximum(0.,1.-np.asarray(distance)/radius)
    return z**4*(5.-4.*z)


class ApproximateRetargeter:
    def __init__(self,g,reference,glove,config):
        if glove.hand!='left': raise ValueError('已有近似数据来自左手，不能用于右手')
        self.g=g;self.glove=glove;self.reference=g.validate(reference).copy()
        self.config=config;self.previous=self.reference.copy()
        path=config.get('approximate_session')
        if path:
            self.folder=(ROOT/path).resolve()
        else:
            self.folder=Path(json.loads((ROOT/'calibration/current_pinch_session.json').read_text(encoding='utf8'))['path'])
        session_file=self.folder/'session.json'
        self.record=json.loads(session_file.read_text(encoding='utf8'))
        if self.record['hand']!='left' or self.record['f1_source_sha256']!=g.cfg['source_sha256']:
            raise ValueError('近似样本的手侧或 F1 模型版本不匹配')
        self.samples=self.record['samples']
        for key in ('open','fist'):
            if key not in self.samples:raise ValueError('已有数据缺少张开/握拳参考')
        self.open=np.asarray(self.samples['open']['median_feedback_deg'],float)
        self.closed=np.asarray(self.samples['fist']['median_feedback_deg'],float)
        if self.open.shape!=(5,4) or self.closed.shape!=(5,4) or not np.isfinite([self.open,self.closed]).all():
            raise ValueError('近似参考必须是有限的5×4反馈角')
        self.span=delta(self.closed,self.open)
        noise=np.maximum(np.asarray(self.samples['open']['p90_span_deg']),
                         np.asarray(self.samples['fist']['p90_span_deg']))
        self.weights=np.where(np.abs(self.span)>=8.,np.abs(self.span)/(noise+2.),0.)
        self.weights[:4,0]=0.  # Side swing is not inferred from flexion spans.
        if np.any(self.weights.sum(axis=1)==0):raise ValueError('至少一指的已有屈伸跨度不足')
        self.safe_span=np.where(self.weights>0,self.span,1.)
        self.rows=dict(index=0,middle=1,ring=2,pinky=3,thumb=4)
        profile=g.simulation_profile
        self.fist=(np.clip(np.radians(profile['fist_coordinates_deg']),g.lower,g.upper)
                   if profile else self.reference.copy())
        if profile is None:
            for f,sl in g.slices.items():
                self.fist[sl.stop-2:sl.stop]=self.reference[sl.stop-2:sl.stop]+.8*(g.upper[sl.stop-2:sl.stop]-self.reference[sl.stop-2:sl.stop])
        # Model pad points are geometric candidates, never measured human pads.
        self.pads=PadGeometry(g,build_pad_candidates(g,self.reference))
        self.feature_indices=np.r_[np.arange(4),np.arange(16,20)]
        self.feature_scale=np.maximum(np.abs(self.span.ravel()[self.feature_indices]),25.)
        self.nodes=[];self.corrections=[];self.anchor_records=[]
        for key in ('open','fist'):
            self.nodes.append(self.features(self.samples[key]['median_feedback_deg']))
            self.corrections.append(np.zeros(len(reference)))
        self.contact_indices=np.r_[np.arange(g.slices['thumb'].start,g.slices['thumb'].stop),
                                   np.arange(g.slices['index'].start,g.slices['index'].stop)]
        for key,s in self.samples.items():
            if s.get('role')!='fit' or s.get('finger')!='index':continue
            q=np.asarray(s['median_feedback_deg'],float);base,_=self.base_goal(q)
            goal,info=self.fit_contact(base)
            info.update(label=key,source_raw_file=s['raw_file'],goal_rad=goal)
            self.anchor_records.append(info)
            if info['enabled']:
                self.nodes.append(self.features(q));self.corrections.append(goal-base)
        self.nodes=np.asarray(self.nodes)
        distances=np.linalg.norm(self.nodes[:,None]-self.nodes[None,:],axis=2)/np.sqrt(len(self.feature_indices))
        regularizer=np.diag(np.r_[0.,0.,np.full(len(self.nodes)-2,1e-3)])
        system=kernel(distances)+regularizer
        self.coefficients=np.linalg.solve(system,self.corrections)
        self.side_enabled=bool(config.get('approximate_side_swing_enabled',False))
        self.side_profile=None;self.side_indices={}
        if self.side_enabled:
            side_path=ROOT/config.get('side_swing_profile','calibration/left_side_swing_20261028.json')
            self.side_profile=json.loads(side_path.read_text(encoding='utf8'))
            if self.side_profile['hand']!='left' or self.side_profile['units']!='deg':
                raise ValueError('近似侧摆配置的手侧或单位不匹配')
            for f,sl in g.slices.items():
                self.side_indices[f]=sl.start+(2 if f=='thumb' else 0)
                c=self.side_profile['channels'][f]
                if (not np.isfinite([c['direction_sign'],c['gain'],c['deadband_deg']]).all()
                        or abs(c['direction_sign'])!=1 or c['gain']<=0 or c['deadband_deg']<0):
                    raise ValueError('近似侧摆方向、增益或死区无效')
            # Predict side-channel motion already present in the accepted poses.
            # The remaining signal drives side swing, so pure flexion/pinch
            # references retain their original target angles.
            residuals=[np.zeros(5),np.zeros(5)]
            for a in self.anchor_records:
                if not a['enabled']:continue
                q=np.asarray(self.samples[a['label']]['median_feedback_deg'],float)
                _,progress=self.base_goal(q)
                expected=self.expected_side_feedback(progress)
                residuals.append(delta(q[:,0],expected))
            self.side_root_coefficients=np.linalg.solve(system,residuals)
            self.side_profile_sha256=hashlib.sha256(side_path.read_bytes()).hexdigest()
        self.fixed_indices=(np.array([],int) if self.side_enabled else
                            np.array([g.slices[f].start for f in ('middle','ring','pinky')],int))
        self.profile_id='approx_'+self.record['id']+'_'+g.mode
        self.raw={}
        for key in ['open','fist']+[x['label'] for x in self.anchor_records]:
            raw=self.folder/self.samples[key]['raw_file']
            self.raw[key]=np.array([json.loads(line)['angles_deg'] for line in raw.read_text(encoding='utf8').splitlines() if line.strip()])
        self.replay_keys=['open','fist','open']
        for a in self.anchor_records:self.replay_keys.extend([a['label'],'open'])
        provenance=dict(schema_version=1,id=self.profile_id,source_session=str(self.folder),
            source_hand='left',source_f1_sha256=self.record['f1_source_sha256'],
            source_samples={k:dict(raw_file=self.samples[k]['raw_file'],sha256=hashlib.sha256((self.folder/self.samples[k]['raw_file']).read_bytes()).hexdigest()) for k in self.raw},
            anchors=self.anchor_records,
            side_swing=dict(enabled=self.side_enabled,profile=self.side_profile,
                reference='current session open/fist and pinch feedback; old profile zero not used',
                profile_sha256=getattr(self,'side_profile_sha256',None)),
            scope='approximate action mapping; nominal F1 pad candidates; no human metric calibration, collision or cable validation',
            hardware_enabled=False,glove_FK_used=False)
        write_json(ROOT/('results/approximate_profile_'+g.mode+'.json'),provenance)

    def features(self,q):
        return delta(q,self.open).ravel()[self.feature_indices]/self.feature_scale

    def base_goal(self,q):
        q=np.asarray(q,float)
        if q.shape!=(5,4) or not np.isfinite(q).all():raise ValueError('近似输入需要有限5×4反馈角')
        fractions=np.clip(delta(q,self.open)/self.safe_span,0.,1.)
        progress=(fractions*self.weights).sum(axis=1)/self.weights.sum(axis=1)
        u=self.reference.copy()
        for f,sl in self.g.slices.items():u[sl]+=progress[self.rows[f]]*(self.fist[sl]-self.reference[sl])
        return self.g.validate(u),{f:float(progress[self.rows[f]]) for f in FINGERS}

    def expected_side_feedback(self,progress):
        fractions=np.array([progress[f] for f in ('index','middle','ring','pinky','thumb')])
        return self.open[:,0]+fractions*self.span[:,0]

    def apply_side_swing(self,q,u,progress,k):
        if not self.side_enabled:return u,None
        expected=self.expected_side_feedback(progress)+k@self.side_root_coefficients
        residual=delta(q[:,0],expected);side={};goal=u.copy()
        for f,idx in self.side_indices.items():
            row=self.rows[f];c=self.side_profile['channels'][f]
            # Floor set from saved open/fist noise, without fitting a noise gain.
            noise=max(self.samples['open']['p90_span_deg'][row][0],
                      self.samples['fist']['p90_span_deg'][row][0])
            deadband=max(float(c['deadband_deg']),1.25*float(noise))
            motion=np.sign(residual[row])*max(0.,abs(residual[row])-deadband)
            offset=np.radians(c['direction_sign']*c['gain']*motion)
            requested=u[idx]+offset;goal[idx]=np.clip(requested,self.g.lower[idx],self.g.upper[idx])
            side[f]=dict(coordinate=self.g.labels[idx],feedback_residual_deg=float(residual[row]),
                deadband_deg=deadband,requested_delta_deg=float(np.degrees(offset)),
                applied_delta_deg=float(np.degrees(goal[idx]-u[idx])),
                actual_goal_deg=float(np.degrees(goal[idx])),
                limits_deg=np.degrees([self.g.lower[idx],self.g.upper[idx]]),
                clipped=bool(abs(requested-goal[idx])>1e-10))
        return goal,dict(enabled=True,method='approximate feedback residual; bounded nominal direction/gain',
            calibrated=False,profile_id=self.side_profile['id'],fingers=side,
            thumb_scope='MCP-AA offset; existing CMC and flexion mapping retained')

    def fit_contact(self,base):
        g=self.g;idx=self.contact_indices;scale=np.maximum(g.upper[idx]-g.lower[idx],.1)
        def residual(x):
            u=base.copy();u[idx]=x;g.assign(g.data,u)
            a=self.pads.pose(g.data,'thumb');b=self.pads.pose(g.data,'index')
            return np.r_[a['position_mm']-b['position_mm'],
                         .12*(x-base[idx])/scale,.15*(a['normal_world']+b['normal_world'])]
        starts=[base[idx].copy()]
        for t in (.15,.5,.85):
            u=base.copy();sl=g.slices['thumb'];u[sl.start]=g.lower[sl.start]+t*(g.upper[sl.start]-g.lower[sl.start])
            starts.append(u[idx])
        fits=[least_squares(residual,np.clip(x,g.lower[idx]+1e-8,g.upper[idx]-1e-8),
                bounds=(g.lower[idx],g.upper[idx]),max_nfev=140,ftol=1e-7,xtol=1e-7,gtol=1e-7) for x in starts]
        fit=min(fits,key=lambda v:np.linalg.norm(residual(v.x)))
        goal=base.copy();goal[idx]=fit.x;g.assign(g.data,goal)
        a=self.pads.pose(g.data,'thumb');b=self.pads.pose(g.data,'index')
        gap=float(np.linalg.norm(a['position_mm']-b['position_mm']))
        return goal,dict(nominal_pad_gap_mm=gap,enabled=bool(gap<3.),
            surface_contact_validated=False,normal_dot=float(a['normal_world']@b['normal_world']))

    def solve(self,angles_deg):
        started=time.perf_counter();q=np.asarray(angles_deg,float)
        base,progress=self.base_goal(q);features=self.features(q)
        d=np.linalg.norm(self.nodes-features,axis=1)/np.sqrt(len(features))
        k=kernel(d);u=np.clip(base+k@self.coefficients,self.g.lower,self.g.upper)
        u[self.fixed_indices]=self.reference[self.fixed_indices]
        u,side=self.apply_side_swing(q,u,progress,k)
        self.g.assign(self.g.data,u);records={}
        for f in FINGERS:
            p=self.g.pose(self.g.data,f)['position_mm']
            records[f]=dict(success=True,requested_position_mm=p.copy(),execution_position_mm=p.copy(),
                position_error_mm=0.,error_scope='internal generated target only; not human fingertip error',
                coordinates_rad=u[self.g.slices[f]],solver='example_synergy')
        gap=float(np.linalg.norm(self.pads.pose(self.g.data,'thumb')['position_mm']-self.pads.pose(self.g.data,'index')['position_mm']))
        self.previous=u.copy()
        return dict(goal_rad=u,fingers=records,angles_deg=q.copy(),model_angles_deg=None,
            solve_ms=(time.perf_counter()-started)*1000,hand='left',root_frame=None,
            retargeting_kind='approximate_examples',progress=progress,nominal_index_pinch_gap_mm=gap,
            side_swing=side,
            nearest_example_distance=float(d.min()),contact_correction_active=bool(np.max(np.abs(u-base))>1e-5),
            example_labels=['open','fist']+[x['label'] for x in self.anchor_records if x['enabled']],
            feedback_calibration_id=self.profile_id,feedback_zero_direction_adapted=False,
            calibrated_to_wearer=False,hardware_enabled=False,
            pipeline='recorded relative feedback -> per-finger normalized flexion + local example correction + optional residual side swing -> bounded coupled F1 coordinates -> nominal fingertip positions',
            calibration_scope='approximate action mapping; no calibrated human fingertip FK or physical pad contact claim')

    def replay_frame(self,t):
        # Real raw frames at holds; minimum-jerk interpolation between medians.
        duration=3.5;index=int(t//duration)%len(self.replay_keys);phase=t%duration
        previous=self.replay_keys[(index-1)%len(self.replay_keys)];key=self.replay_keys[index]
        a=np.asarray(self.samples[previous]['median_feedback_deg']);b=np.asarray(self.samples[key]['median_feedback_deg'])
        if phase<2.:
            s=phase/2.;s=10*s**3-15*s**4+6*s**5
            q=a+s*delta(b,a);label=previous+' → '+key+'（数据插值）'
        else:
            rows=self.raw[key];q=rows[int((phase-2.)*100)%len(rows)];label=key+'（已采原始帧）'
        return q,label
