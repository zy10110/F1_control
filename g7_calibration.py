"""Vendor SDK calibration, serialized on the SDK polling thread.

SDK writes only the session's ASCII working profile. Original HumanHand
assets and imported snapshots are never modified. Joint-point captures are
diagnostics / F1 alignment inputs, not replacements for SDK raw observations.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation

from hand_kinematics import ROOT

GESTURES = (
    ('平放张开', '五指自然伸直、舒适分开，手掌放平，保持不动。', 2.),
    ('握拳静止', '握到舒适拳姿，拇指自然收拢，不用力攥紧。', 2.),
    ('重复握拳', '在张开和舒适握拳之间连续、缓慢往返，采集期间保持动作。', 5.),
    ('拇指活动', '四指放松，拇指舒适地外展、内收、对掌、屈伸，连续活动。', 5.),
    ('拇食轻触', '拇指和食指指腹轻触，不用力捏紧，保持不动。', 2.),
    ('拇中轻触', '拇指和中指指腹轻触，其余手指放松，保持不动。', 2.),
    ('拇无名轻触', '拇指和无名指指腹轻触，不要挤压传感器，保持不动。', 2.),
    ('拇小轻触', '拇指和小指指腹轻触，到舒适位置即可，保持不动。', 2.),
)
ALIGNMENT = (
    ('F1 张开基准', '五指自然伸展，拇指舒适展开，手腕放稳；不要重新调零。', 1.5),
    ('F1 屈曲方向', '舒适握拳，拇指自然收拢，不用力攥紧，手腕放稳。', 1.5),
    ('F1 拇食轻触', '自然做放松的 OK 手势，用拇食指腹轻触，不用力捏紧。', 1.5),
)


def write_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    temporary.replace(path)


def bind_calibration(dll):
    for name,args in (('hh_sdk_calib_start',[ct.c_void_p,ct.c_int]),
                      ('hh_sdk_calib_stop',[ct.c_void_p]),
                      ('hh_sdk_calib_solve',[ct.c_void_p])):
        fn=getattr(dll,name); fn.argtypes=args; fn.restype=ct.c_int
    dll.hh_sdk_calib_suggested_seconds.argtypes=[ct.c_int]
    dll.hh_sdk_calib_suggested_seconds.restype=ct.c_double


class CalibrationCapture:
    def __init__(self,dll,handle,profile,hand,profile_info):
        self.dll=dll; self.handle=handle; self.profile=Path(profile); self.hand=hand
        self.directory=ROOT/'calibration/g7_sessions'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        self.directory.mkdir(parents=True)
        self.original=self.profile.read_bytes()
        (self.directory/'vendor_before.bin').write_bytes(self.original)
        self.state=dict(phase='ready',message='等待选择标定步骤',completed=None)
        self.samples={}; self.active=None; self.vendor_changed=False
        self.profile_info=dict(profile_info)
        write_json(self.directory/'session.json',dict(hand=hand,source_profile=profile_info,
            vendor_gestures=GESTURES,scope='SDK 8 gestures + separate F1 Cartesian alignment'))

    def start(self,key,gesture=None,duration=1.5):
        if self.active: raise ValueError('当前采集尚未结束')
        if gesture is not None:
            rc=self.dll.hh_sdk_calib_start(self.handle,gesture)
            if rc: raise RuntimeError(f'标定开始失败：{rc}')
            duration=float(self.dll.hh_sdk_calib_suggested_seconds(gesture))
        self.active=dict(key=key,gesture=gesture,end=time.monotonic()+duration,
                         duration=duration,frames=[])
        self.state=dict(phase='capturing',key=key,remaining_s=duration,message='采集中',completed=None)

    def tick(self,frame=None):
        if not self.active:return
        a=self.active
        if frame is not None:
            p=np.asarray(frame['positions_m'])
            if p.shape==(21,3) and np.isfinite(p).all(): a['frames'].append(frame)
        remaining=max(0.,a['end']-time.monotonic())
        self.state=dict(self.state,remaining_s=remaining,frame_count=len(a['frames']))
        if remaining>0:return
        if a['gesture'] is not None:
            rc=self.dll.hh_sdk_calib_stop(self.handle)
            if rc:
                self.active=None;raise RuntimeError(f'标定停止失败：{rc}')
        self.active=None
        if len(a['frames'])<20:
            self.state=dict(phase='failed',message='有效数据不足，请检查信号后重采本步骤',completed=None)
            return
        p=np.asarray([f['positions_m'] for f in a['frames']])
        median=np.median(p,axis=0)
        record=dict(a['frames'][-1],positions_m=median.tolist(),sample_count=len(p),
                    p95_deviation_mm=float(np.percentile(np.linalg.norm(p-median,axis=2),95)*1000))
        # Joint mapping needs a representative orientation, not an arbitrary
        # last frame. Rotation.mean handles equivalent q / -q samples.
        if all('quaternions_wxyz' in f for f in a['frames']):
            quats=np.asarray([f['quaternions_wxyz'] for f in a['frames']],float)
            if quats.shape!=(len(p),21,4) or not np.isfinite(quats).all() or np.min(np.linalg.norm(quats,axis=2))<1e-6:
                self.state=dict(phase='failed',message='关节姿态数据无效，请重采当前步骤',completed=None)
                return
            record['quaternions_wxyz']=[Rotation.from_quat(quats[:,j,[1,2,3,0]]).mean().as_quat()[[3,0,1,2]].tolist()
                                         for j in range(21)]
            record['orientation_aggregation']='quaternion rotation mean, sign invariant'
        self.samples[a['key']]=record
        write_json(self.directory/(a['key']+'.json'),record)
        (self.directory/(a['key']+'.jsonl')).write_text('\n'.join(json.dumps(f) for f in a['frames']),encoding='utf8')
        self.state=dict(phase='done',completed=a['key'],message=f'已采集 {len(p)} 帧',record=record)

    def solve_vendor(self):
        if self.active: raise ValueError('请先完成当前采集')
        if any('vendor_'+str(i) not in self.samples for i in range(1,9)):
            raise ValueError('八段手势未采齐，不能计算个人标定')
        self.state=dict(phase='solving',message='SDK 正在计算个人标定',completed=None)
        rc=self.dll.hh_sdk_calib_solve(self.handle)
        if rc: raise RuntimeError(f'SDK 标定解算失败：{rc}；未切换配置')
        target=self.directory/'vendor_calibrated.bin';target.write_bytes(self.profile.read_bytes())
        self.vendor_changed=True
        self.profile_info.update(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        self.state=dict(phase='done',completed='vendor_solved',message='个人标定已生成，继续采集 F1 对齐姿态')
        return str(target)

    def build_alignment(self,mode,reference,source_sha256,method='cartesian'):
        if any(k not in self.samples for k in ('open','fist','pinch')):
            raise ValueError('请完成张开、握拳和拇食轻触三个对齐步骤')
        opening=self.samples['open'];fist=self.samples['fist'];pinch=self.samples['pinch']
        p=np.asarray(opening['positions_m']);q=np.asarray(pinch['positions_m'])
        lengths=np.linalg.norm(p[[4,8,12,16,20]]-p[[1,5,9,13,17]],axis=1)*1000
        if method=='joint':
            from g7_joint_mapping import rotations
            for frame in (opening,fist,pinch):rotations(frame)
        elif np.any(lengths<15):raise ValueError('张开基准有指长不足15mm，请检查佩戴或信号，仅重采张开步骤')
        record=dict(schema_version=1,hand=self.hand,mode=mode,source_sha256=source_sha256,
            reference_coordinates_rad=np.asarray(reference).tolist(),vendor_profile=self.profile_info,
            reference_frame=opening,fist_frame=fist,pinch_frame=pinch,
            pinch_closed_mm=float(np.linalg.norm(q[4]-q[8])*1000),
            retargeting_method=method,
            scope=('wearer relative-joint zero/range alignment; fingertip lengths unused' if method=='joint' else
                   'wearer Cartesian open/fist/pinch alignment; not measured physical pad calibration'))
        path=self.directory/'f1_alignment.json';write_json(path,record)
        self.state=dict(phase='done',completed='alignment_ready',message='F1 对齐已保存，尚未启动运动',path=str(path))
        return str(path)

    def cancel(self):
        if self.active and self.active['gesture'] is not None:
            self.dll.hh_sdk_calib_stop(self.handle)
        self.active=None
        self.state=dict(phase='cancelled',completed=None,message='已结束采集；模型保持静止')

    def build_thumb_alignment(self,config,source_sha256):
        """Replace only the thumb zero/hinges; retain four-finger samples."""
        if any(k not in self.samples for k in ('thumb_open','thumb_flex')):
            raise ValueError('请先采拇指展开和屈曲两个姿态')
        from g7_joint_mapping import joint_rotations
        opened=self.samples['thumb_open'];flexed=self.samples['thumb_flex']
        a=joint_rotations(opened)['thumb'];b=joint_rotations(flexed)['thumb']
        axes={};amplitudes={}
        for j,name in ((1,'MCP'),(2,'IP')):
            delta=(a[j].inv()*b[j]).as_rotvec();delta[1]=0.
            amplitude=np.linalg.norm(delta);amplitudes[name]=float(np.degrees(amplitude))
            if amplitude<np.radians(5):
                raise ValueError(f'拇指{name}有效弯曲仅{np.degrees(amplitude):.1f}°；只需重采第2步，让两个指节都舒适弯曲')
            # The explicitly labelled flexion gesture defines the sign.
            # Do not force the polarity of an earlier SDK/profile convention.
            axes[str(j)]=(delta/amplitude).tolist()
        record=json.loads(Path(config['mapping_profile']).read_text(encoding='utf8'))
        if record['hand']!=self.hand or record['source_sha256']!=source_sha256:
            raise ValueError('原关节标定与手别或F1模型不一致')
        record['thumb_joint_calibration']=dict(reference_frame=opened,flexion_frame=flexed,
            hinge_axes=axes,transverse_flexion_deg=amplitudes,source='explicit thumb-only flexion; no vendor recalibration')
        path=self.directory/'f1_alignment.json';write_json(path,record)
        self.state=dict(phase='done',completed='alignment_ready',message='拇指映射已保存；四指标定未改',path=str(path))
        return str(path)
