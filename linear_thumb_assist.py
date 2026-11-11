"""Continuous thumb-only angular correction; no contact class or pose seed.

Official calibration is reused. Surface landmarks remain model candidates.
The ordinary execution guard remains responsible for clearance and paths.
"""
import json
import time
import mujoco
import numpy as np
from hand_kinematics import ROOT
from palm_calibration import PadGeometry

PAIRS = ('index','middle','ring','pinky')


class LinearThumbAssist:
    def __init__(self,g,config):
        self.g=g
        self.pad=PadGeometry(g,json.loads((ROOT/'calibration/g7_f1_pad_candidates.json').read_text(encoding='utf8')))
        self.data=mujoco.MjData(g.model)
        self.cap=np.radians(config.get('thumb_linear_cap_deg',[20.,20.,20.,12.,12.]))
        self.speed=np.radians(float(config.get('thumb_linear_speed_deg_s',35.)))
        self.start=float(config.get('thumb_linear_start_ratio',.8))
        if self.cap.shape!=(5,) or not np.isfinite(self.cap).all() or np.any(self.cap<=0) or not np.isfinite([self.speed,self.start]).all() or min(self.speed,self.start)<=0:
            raise ValueError('拇指连续修正参数无效')
        self.last=np.zeros(5);self.last_stamp=None;self.last_time=None

    def weights(self,frame):
        p=np.asarray(frame.get('positions_m'),float)
        if p.shape!=(21,3) or not np.isfinite(p).all():
            raise ValueError('拇指连续修正需要21个有效位置')
        lengths=[float(np.linalg.norm(np.diff(p[i:i+4],axis=0),axis=1).sum()) for i in (1,5,9,13,17)]
        if min(lengths)<.015 or max(lengths)>.4:
            raise ValueError('官方骨段长度无效')
        gaps=np.linalg.norm(p[4]-p[[8,12,16,20]],axis=1)
        starts=self.start*np.minimum(lengths[0],lengths[1:])
        # One affine ramp from a broad approach distance to zero gap. No
        # close-contact band, smoothstep power, argmax or pose switch.
        values=np.clip(1.-gaps/starts,0.,1.)
        return values,gaps*1000,starts*1000

    def elapsed(self,frame):
        now=time.monotonic();stamp=frame.get('timestamp_ns')
        dt=1/120 if self.last_time is None else now-self.last_time
        if isinstance(stamp,(int,float)) and np.isfinite(stamp):
            if self.last_stamp is not None and stamp>self.last_stamp:dt=(stamp-self.last_stamp)*1e-9
            self.last_stamp=stamp
        self.last_time=now
        # Do not release a stored correction in one jump after a pause.
        return float(np.clip(dt,1/1000,1/30))

    def pose_jac(self,f):
        pose=self.pad.pose(self.data,f)
        jp=np.zeros((3,self.g.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(self.g.model,self.data,jp,jr,pose['position_world_m'],self.pad.ids[f])
        angular=jr@self.g.C[:,:5]
        jnormal=np.cross(angular.T,pose['normal_world']).T
        return pose,jp@self.g.C[:,:5]*1000,jnormal

    def apply(self,frame,base,lower,upper):
        strengths,gaps,starts=self.weights(frame);dt=self.elapsed(frame)
        intensity=float(np.max(strengths));total=float(strengths.sum())
        raw=np.zeros(5)
        if total>1e-12:
            weights=strengths/total
            lo=np.maximum(lower[:5],base[:5]-self.cap)
            hi=np.minimum(upper[:5],base[:5]+self.cap)
            u=base.copy()
            # Deterministic damped local steps from the live angle baseline;
            # never jump to a fitted contact pose or a different seed.
            for _ in range(5):
                self.g.assign(self.data,u)
                thumb,jt,jnt=self.pose_jac('thumb');rows=[];rhs=[]
                for f,w in zip(PAIRS,weights):
                    if w<1e-12:continue
                    other=self.pad.pose(self.data,f);sw=np.sqrt(w)
                    residual=thumb['position_mm']-other['position_mm']-1.*other['normal_world']
                    rows.extend([sw*jt,sw*7.*jnt])
                    rhs.extend([sw*residual,sw*7.*(thumb['normal_world']+other['normal_world'])])
                # Prefer root adjustment; retain the live MCP/IP shape more.
                damping=np.diag([12.,12.,12.,28.,28.])
                j=np.vstack(rows+[damping]);r=np.concatenate(rhs+[damping@(u[:5]-base[:5])])
                step=np.linalg.solve(j.T@j+np.eye(5)*1e-8,-j.T@r)
                u[:5]=np.clip(u[:5]+np.clip(step,-np.radians(4),np.radians(4)),lo,hi)
            # In a fixed, single-pair configuration the amplitude is exactly
            # proportional to source approach progress, until limits intervene.
            raw=(u[:5]-base[:5])*intensity
        else:weights=np.zeros(4)
        correction=self.last+np.clip(raw-self.last,-self.speed*dt,self.speed*dt)
        correction=np.clip(correction,lower[:5]-base[:5],upper[:5]-base[:5])
        self.last=correction.copy();result=base.copy();result[:5]+=correction
        self.g.assign(self.data,result)
        t=self.pad.pose(self.data,'thumb')
        return result,dict(enabled=True,method='linear approach ramp + thumb-only damped correction',
            strengths=dict(zip(PAIRS,strengths)),weights=dict(zip(PAIRS,weights)),
            source_gaps_mm=dict(zip(PAIRS,gaps)),ramp_start_mm=dict(zip(PAIRS,starts)),
            intensity=intensity,dt_s=dt,raw_correction_deg=np.degrees(raw),
            correction_deg=np.degrees(correction),cap_deg=np.degrees(self.cap),
            correction_speed_deg_s=np.degrees(self.speed),four_fingers_unchanged=True,
            active_pair=None,forced_closure=False,extra_calibration=False,
            command_pad_gaps_mm={f:float(np.linalg.norm(t['position_mm']-self.pad.pose(self.data,f)['position_mm'])) for f in PAIRS},
            scope='unconfirmed model pad points; contact is not guaranteed')
