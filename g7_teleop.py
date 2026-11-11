"""G7 Cartesian retargeting. Device calibration belongs to the vendor SDK.

F1 limits/reduction are consumed unchanged. The position projection is a
bounded numerical approximation; pinch adds a relative-position objective.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import socket
import queue
import threading
import time

import numpy as np
from scipy.optimize import least_squares

from hand_kinematics import ROOT, FINGERS, HandGeometry, serializable
from glove_teleop import FlexionRetargeter, unit

TIP = dict(zip(FINGERS, (4, 8, 12, 16, 20)))
BASE = dict(zip(FINGERS, (1, 5, 9, 13, 17)))


def load_g7_config():
    config=json.loads((ROOT/'g7_config.json').read_text(encoding='utf8'))
    for key in ('mapping_profile','active_vendor_profile','official_f1_pinch_profile','official_app_reference_capture'):
        if config.get(key) and not Path(config[key]).is_absolute():
            config[key]=str((ROOT/config[key]).resolve())
    return config


def load_reader(config):
    path = (ROOT/config['sdk_reader']).resolve()
    spec = importlib.util.spec_from_file_location('f1_g7_vendor_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_frame(config, hand):
    if config.get('mapping_profile'):
        record=json.loads(Path(config['mapping_profile']).read_text(encoding='utf8'))
        if record['hand']!=hand: raise ValueError('F1 对齐文件的手别不一致')
        return record['reference_frame']
    path = (ROOT/config['reference_capture']).resolve()
    with path.open(encoding='utf8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['hand'] == (1 if hand == 'left' else 2):
                return record
    raise ValueError('没有对应手别的 G7 参考帧；不能复用另一只手的坐标适配')


def positions(frame):
    p = np.asarray(frame['positions_m'], float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError('G7 需要21个有限 XYZ，单位米')
    return p * 1000.


class G7Retargeter(FlexionRetargeter):
    """Finger-length adaptation in anatomical frames plus paired pinch IK."""
    def __init__(self, geometry, reference, hand='right', config=None):
        self.g = geometry
        self.reference = self.g.validate(reference).copy()
        self.previous = self.reference.copy()
        self.hand = hand
        self.config = config or load_g7_config()
        self.alignment=None
        if self.config.get('mapping_profile'):
            self.alignment=json.loads(Path(self.config['mapping_profile']).read_text(encoding='utf8'))
            if (self.alignment['hand']!=hand or self.alignment['mode']!=self.g.mode or
                    self.alignment['source_sha256']!=self.g.cfg['source_sha256'] or
                    not np.allclose(self.alignment['reference_coordinates_rad'],self.reference,atol=1e-8)):
                raise ValueError('F1 对齐文件与手别、模型、模式或初始姿态不一致，请重新做 F1 对齐')
        src = positions(reference_frame(self.config, hand))
        self.active = {}; self.grids = {}; self.frames = {}
        self.fixed_indices = np.array([], dtype=int)
        self.g.assign(self.g.data, self.reference)
        roots = {}
        for f in FINGERS:
            column = 2 if f == 'thumb' else 0
            name = self.g.cfg['fingers'][f]['columns'][column][0][0]
            roots[f] = (self.g.data.xanchor[self.g.model.joint(name).id]-self.g.origin)*1000
        radial = unit(roots['index']-roots['pinky'])
        for f in FINGERS:
            sl = self.g.slices[f]
            self.active[f] = np.arange(sl.start, sl.stop)
            neutral = self.g.forward(self.reference, f)['position_mm'].copy()
            ray = neutral-roots[f]
            extension = unit(ray)
            bend = self.g.position_jacobian(f)[:, -2]
            bend = unit(bend-extension*np.dot(bend, extension))
            side = unit(np.cross(extension, bend))
            if np.dot(side, radial) < 0:
                side = -side
            destination_basis = np.column_stack((side, extension, bend))
            source_extension = unit(src[TIP[f]]-src[BASE[f]])
            source_bend = np.array([0., 0., -1.])
            if self.alignment:
                closed=positions(self.alignment['fist_frame'])
                movement=(closed[TIP[f]]-closed[BASE[f]])-(src[TIP[f]]-src[BASE[f]])
                transverse=movement-source_extension*np.dot(movement,source_extension)
                if np.linalg.norm(transverse)>5: source_bend=transverse
            source_bend = unit(source_bend-source_extension*np.dot(source_bend, source_extension))
            source_side = unit(np.cross(source_extension, source_bend))
            source_basis = np.column_stack((source_side, source_extension, source_bend))
            source_length = np.linalg.norm(src[TIP[f]]-src[BASE[f]])
            if source_length < 15:
                raise ValueError('G7 参考帧有效指长不足：'+f)
            scale = float(np.linalg.norm(ray)/source_length)
            self.frames[f] = dict(source_root_mm=src[BASE[f]].copy(),
                destination_root_mm=roots[f], scale=scale,
                transform=destination_basis@source_basis.T,
                destination_basis=destination_basis, source_basis=source_basis,
                source_length_mm=float(source_length), destination_length_mm=float(np.linalg.norm(ray)))
            n = int(self.config['thumb_grid_size'] if f == 'thumb' else self.config['grid_size'])
            values = np.array(list(itertools.product(*[np.linspace(self.g.lower[i], self.g.upper[i], n)
                                                      for i in self.active[f]])))
            points = []
            for v in values:
                u = self.reference.copy(); u[self.active[f]] = v
                points.append(self.g.forward(u, f)['position_mm'].copy())
            self.grids[f] = values, np.asarray(points)
        self.active_indices = np.arange(len(reference))
        # Shared palm map keeps coincident source points coincident. It is used
        # for the pinch midpoint, while free fingers retain their own lengths.
        source_x = unit(src[BASE['index']]-src[BASE['pinky']])
        source_y = np.array([0., 1. if hand == 'left' else -1., 0.])
        source_y = unit(source_y-source_x*np.dot(source_y, source_x))
        source_z = unit(np.cross(source_x, source_y))
        if source_z[2] > 0: source_z = -source_z
        dest_x = radial
        dest_y = self.frames['middle']['destination_basis'][:, 1]
        dest_y = unit(dest_y-dest_x*np.dot(dest_y, dest_x))
        dest_z = unit(np.cross(dest_x, dest_y))
        if np.dot(dest_z, self.frames['middle']['destination_basis'][:, 2]) < 0: dest_z = -dest_z
        width_scale = np.linalg.norm(roots['index']-roots['pinky'])/np.linalg.norm(src[5]-src[17])
        length_scale = float(np.mean([self.frames[f]['scale'] for f in FINGERS if f != 'thumb']))
        self.common_transform = np.column_stack((dest_x,dest_y,dest_z))@np.diag(
            [width_scale,length_scale,length_scale])@np.column_stack((source_x,source_y,source_z)).T
        self.common_source_root = src[9].copy()
        self.common_destination_root = roots['middle'].copy()
        self.mean_scale = length_scale
        from thumb_collision import ThumbCollisionGuard
        self.collision=ThumbCollisionGuard(self.g,
            self.config.get('thumb_clearance_mm',.6),self.config.get('collision_step_deg',.75))
        if self.alignment:
            self.config=dict(self.config,pinch_closed_mm=self.alignment['pinch_closed_mm'],
                             pinch_start_mm=self.alignment['pinch_closed_mm']+25.)

    def mapping_record(self):
        return dict(hand=self.hand, source_units='mm', destination_frame='F1 P / mm',
            reference_capture=self.config['reference_capture'], frames=self.frames,
            shared_palm_transform=self.common_transform,
            source_palm_origin_mm=self.common_source_root,
            destination_palm_origin_mm=self.common_destination_root,
            scope='existing calibrated G7 reference; nominal F1 geometry adaptation, not measured extrinsics or pad calibration')

    def map_frame(self, frame):
        if frame['hand'] != (1 if self.hand == 'left' else 2):
            raise ValueError('G7 实时手别与坐标适配不一致')
        p = positions(frame)
        targets = {}
        for f in FINGERS:
            m = self.frames[f]
            targets[f] = m['destination_root_mm']+m['scale']*m['transform']@(
                p[TIP[f]]-p[BASE[f]])
        gap = float(np.linalg.norm(p[4]-p[8]))
        a, b = float(self.config['pinch_start_mm']), float(self.config['pinch_closed_mm'])
        t = float(np.clip((a-gap)/(a-b), 0., 1.))
        weight = t*t*(3.-2.*t)
        # Both target midpoint and separation are expressed in the shared frame.
        mid = self.common_destination_root+self.common_transform@(
            (p[4]+p[8])*.5-self.common_source_root)
        relative = self.common_transform@(p[4]-p[8])
        for f, sign in (('thumb', 1.), ('index', -1.)):
            targets[f] = (1.-weight)*targets[f]+weight*(mid+sign*.5*relative)
        return targets, dict(source_gap_mm=gap, weight=weight,
                             requested_relative_mm=targets['thumb']-targets['index'])

    def _pinch(self, goal, targets, pinch):
        if pinch['weight'] < .01:
            return goal
        idx = np.r_[self.active['thumb'], self.active['index']]
        base = goal.copy()
        w = np.sqrt(float(self.config['pinch_relative_weight'])*pinch['weight'])
        wanted = targets['thumb']-targets['index']
        def residual(v):
            u = base.copy(); u[idx] = v
            a = self.g.forward(u,'thumb')['position_mm'].copy()
            b = self.g.forward(u,'index')['position_mm'].copy()
            return np.r_[a-targets['thumb'], b-targets['index'], w*(a-b-wanted)]
        def jac(v):
            u = base.copy(); u[idx] = v
            self.g.forward(u,'thumb'); ja = self.g.position_jacobian('thumb')
            self.g.forward(u,'index'); jb = self.g.position_jacobian('index')
            A = np.c_[ja,np.zeros((3,3))]; B = np.c_[np.zeros((3,5)),jb]
            return np.vstack((A,B,w*(A-B)))
        starts = [goal[idx], self.previous[idx]]
        candidates = [(np.linalg.norm(residual(goal[idx])),goal[idx].copy())]
        for start in starts:
            fit = least_squares(residual,np.clip(start,self.g.lower[idx]+1e-10,self.g.upper[idx]-1e-10),
                jac=jac,bounds=(self.g.lower[idx],self.g.upper[idx]),max_nfev=35,
                ftol=1e-5,xtol=1e-5,gtol=1e-5)
            candidates.append((np.linalg.norm(residual(fit.x)),fit.x.copy()))
        base[idx] = min(candidates,key=lambda x:x[0])[1]
        return base

    def solve_frame(self, frame):
        started = time.perf_counter()
        targets, pinch = self.map_frame(frame)
        goal = self.reference.copy(); records = {}
        for f in FINGERS:
            u, rec = self.solve_position(f, targets[f])
            goal[self.active[f]] = u[self.active[f]]
            rec['independent_projection_error_mm'] = rec['position_error_mm']
            records[f] = rec
        goal = self._pinch(goal, targets, pinch)
        goal, avoidance=self.collision.project(goal,self.previous,self.reference,self.grids['thumb'][0],targets['thumb'])
        for f in FINGERS:
            position = self.g.forward(goal,f)['position_mm'].copy()
            records[f].update(execution_position_mm=position,
                position_error_mm=float(np.linalg.norm(position-targets[f])),coordinates_rad=goal[self.g.slices[f]],
                projected=bool(np.linalg.norm(position-targets[f]) > self.g.cfg['position_tolerance_mm']),
                solver='G7 bounded position projection'+(' + paired pinch objective' if f in ('thumb','index') and pinch['weight']>.01 else ''))
        pinch['execution_gap_mm'] = float(np.linalg.norm(records['thumb']['execution_position_mm']-
                                                        records['index']['execution_position_mm']))
        self.previous = self.g.validate(goal).copy()
        side = {}
        for f in FINGERS:
            i = self.g.slices[f].start+(2 if f == 'thumb' else 0)
            side[f] = dict(actual_goal_deg=float(np.degrees(goal[i])),
                           limits_deg=np.degrees([self.g.lower[i],self.g.upper[i]]))
        return dict(goal_rad=goal,fingers=records,pinch=pinch,hand=self.hand,
            angles_deg=None,model_angles_deg=None,glove_fk_api='g7_cartesian',
            glove_project='Forsense G7 SDK',side_swing=dict(enabled=True,method='G7 fingertip position IK',fingers=side),
            thumb_avoidance=avoidance,
            calibrated_to_wearer=bool(self.alignment),calibration_scope=('G7 vendor calibration + wearer open/fist/pinch F1 alignment' if self.alignment else 'G7 vendor calibration; F1 reference not independently calibrated'),
            pipeline='G7 calibrated wrist XYZ -> frame/size retargeting -> bounded coupled position IK -> angles',
            root_frame='shared wrist/palm',hardware_enabled=False,
            solve_ms=(time.perf_counter()-started)*1000,
            projection_method='workspace samples + bounded multistart; pinch uses weighted relative-position objective')


class G7Session:
    """One SDK owner, separate newest-frame IK worker; Tk never reads serial."""
    def __init__(self,mode,reference,hand='right',source='serial',port='COM7',config=None):
        self.mode=mode; self.reference=np.asarray(reference).copy(); self.hand=hand
        self.source=source; self.port=port; self.config=config or load_g7_config()
        if source=='g7_udp':
            from g7_official_udp import official_config
            self.config=official_config(self.config)
        self.stop_event=threading.Event(); self.lock=threading.Lock()
        self.error=None; self.state='initializing'; self.latest_output=None; self.latest_frame=None
        self.last_frame_time=None; self.frame_count=0; self.solve_count=0
        self.retargeter=None; self.reader_thread=None
        self.rejected_frames=0;self.last_frame_error=None
        self.profile_info=None
        self.commands=queue.Queue();self.calibration=None;self.calibration_error=None
        self.motion_enabled=True;self.mapping_revision=0;self.loaded_revision=0
        self.udp_alignment=None;self.udp_receiver=None
        if source=='g7_udp':
            self.motion_enabled=bool(self.config.get('official_app_resume_reference') and
                self.config.get('official_app_reference_capture') and
                Path(self.config['official_app_reference_capture']).is_file())
            if self.config.get('retargeting_method')=='proportional':
                # Directions are absolute in the anatomical palm; no fresh
                # wearer zero or additional F1 alignment is required.
                self.motion_enabled=True
            if self.config.get('retargeting_method')=='direct_joint':
                self.motion_enabled=bool(self.config.get('official_app_reference_capture') and
                    Path(self.config['official_app_reference_capture']).is_file())
            if self.config.get('calibrated_angle_profile'):
                self.motion_enabled=True   # The profile contains its own open reference.
        self.thread=threading.Thread(target=self._run,name='G7-F1-IK',daemon=True)

    def start(self): self.thread.start()

    def make_retargeter(self):
        if self.source=='g7_udp' and self.config.get('calibrated_angle_profile'):
            from calibrated_angle_mapping import CalibratedAngleRetargeter
            return CalibratedAngleRetargeter(HandGeometry(self.mode),self.reference,self.hand,
                self.config,self.config['calibrated_angle_profile'])
        if self.config.get('retargeting_method')=='direct_joint':
            from g7_direct_joint import G7DirectJointRetargeter
            return G7DirectJointRetargeter(HandGeometry(self.mode),self.reference,self.hand,self.config)
        if self.config.get('retargeting_method')=='proportional':
            from g7_proportional_mapping import G7ProportionalRetargeter
            return G7ProportionalRetargeter(HandGeometry(self.mode),self.reference,self.hand,self.config)
        if self.config.get('retargeting_method')=='joint':
            from g7_joint_mapping import G7JointRetargeter
            return G7JointRetargeter(HandGeometry(self.mode),self.reference,self.hand,self.config)
        return G7Retargeter(HandGeometry(self.mode),self.reference,self.hand,self.config)

    def _publish(self, frame):
        with self.lock:
            self.frame_count+=1; self.last_frame_time=time.monotonic()
            self.latest_frame=(self.frame_count,frame,self.last_frame_time)

    def _read_sdk(self):
        handle=None; dll=None
        try:
            reader=load_reader(self.config); dll=reader.load_sdk()
            import serial
            with serial.Serial(self.port,921600,timeout=.1): pass
            hand_id=dll.hh_sdk_probe_hand(self.port.encode('ascii'),921600)
            expected=1 if self.hand=='left' else 2
            if hand_id!=expected: raise ValueError(f'串口手别为 {hand_id}，期望 {expected}；未启动解算')
            source=(ROOT/self.config['calibration_snapshot']/f'assets/users/默认用户/hand_profile_{self.hand}.bin')
            if self.config.get('active_vendor_profile'):source=Path(self.config['active_vendor_profile'])
            work=Path.home()/'Documents/Codex/g7_f1_runtime'
            work.mkdir(parents=True,exist_ok=True)
            profile=work/f'hand_profile_{self.hand}.bin'; shutil.copy2(source,profile)
            shutil.copytree(ROOT/self.config['calibration_snapshot']/'param',work/'param',dirs_exist_ok=True)
            digest=hashlib.sha256(profile.read_bytes()).hexdigest()
            self.profile_info=dict(path=str(source),sha256=digest,sdk_version=dll.hh_sdk_version().decode())
            cfg=reader.Config(self.port.encode('ascii'),921600,hand_id,
                              str(profile).encode('ascii'),str(work/'param').encode('ascii'))
            handle=dll.hh_sdk_create(ct.byref(cfg))
            if not handle: raise RuntimeError('SDK 打开失败，检查串口是否被上位机占用')
            from g7_calibration import bind_calibration, CalibrationCapture, write_json
            bind_calibration(dll)
            while not self.stop_event.is_set():
                try:
                    command=self.commands.get_nowait()
                except queue.Empty:command=None
                if command:
                    try:
                        self.calibration_error=None
                        action=command['action']
                        if action=='begin':
                            self.calibration=CalibrationCapture(dll,handle,profile,self.hand,self.profile_info)
                            self.calibration_error=None
                        elif action=='capture':
                            self.calibration.start(command['key'],command.get('gesture'),command.get('duration',1.5))
                        elif action=='solve_vendor':
                            self.calibration.solve_vendor();self.profile_info=self.calibration.profile_info
                        elif action=='apply':
                            path=self.calibration.build_alignment(self.mode,self.reference,self.retargeter.g.cfg['source_sha256'],
                                                                  self.config.get('retargeting_method','cartesian'))
                            new=dict(self.config,mapping_profile=path,
                                     active_vendor_profile=self.calibration.profile_info['path'])
                            write_json(ROOT/'g7_config.json',new)
                            with self.lock:self.config=new;self.mapping_revision+=1
                        elif action=='apply_thumb':
                            path=self.calibration.build_thumb_alignment(self.config,self.retargeter.g.cfg['source_sha256'])
                            new=dict(self.config,mapping_profile=path)
                            write_json(ROOT/'g7_config.json',new)
                            with self.lock:self.config=new;self.mapping_revision+=1
                        elif action=='cancel':self.calibration.cancel()
                    except Exception as exc:
                        self.calibration_error=str(exc)
                frame=reader.Frame()
                rc=dll.hh_sdk_poll(handle,ct.byref(frame))
                if rc: raise RuntimeError('G7 SDK poll 返回 '+str(rc))
                if frame.status!=0:
                    if self.calibration:self.calibration.tick()
                    self.stop_event.wait(.001); continue
                confidence=(ct.c_float*6)()
                confidence_rc=dll.hh_sdk_coil_confidence(handle,confidence)
                payload=dict(hand=hand_id,seq=int(frame.seq),timestamp_ns=int(frame.timestamp_ns),
                    positions_m=[list(p) for p in frame.pos_m],quaternions_wxyz=[list(q) for q in frame.quat_wxyz],
                    coil_confidence=list(confidence) if confidence_rc==0 else None)
                self._publish(payload)
                if self.calibration:self.calibration.tick(payload)
        except Exception as exc:
            with self.lock: self.error='G7读取失败：'+str(exc); self.state='error'
            self.stop_event.set()
        finally:
            if self.calibration and self.calibration.active:self.calibration.cancel()
            if handle: dll.hh_sdk_destroy(handle)

    def _read_replay(self):
        try:
            path=ROOT/self.config['reference_capture']
            frames=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]
            while not self.stop_event.is_set():
                for frame in frames:
                    if self.stop_event.is_set(): return
                    self._publish(frame); self.stop_event.wait(1/120)
        except Exception as exc:
            with self.lock: self.error='G7回放失败：'+str(exc)
            self.stop_event.set()

    def _read_udp(self):
        from g7_official_udp import OfficialUDPReceiver, save_open_reference
        try:
            reader=load_reader(self.config)
            receiver=OfficialUDPReceiver(reader.decode_udp,self.hand,
                port=self.config.get('official_app_udp_port',15001))
            self.udp_receiver=receiver
            self.profile_info=dict(source='Forsense HumanHand solved joint UDP',
                calibration_owner='running official app; no SDK recalibration',
                address=f'127.0.0.1:{receiver.port}')
            samples=[];start=None;end=None
            def publish(frame):
                self._publish(frame)
                if start is not None and start<=time.monotonic()<=end:samples.append(frame)
            def tick():
                nonlocal samples,start,end
                try:command=self.commands.get_nowait()
                except queue.Empty:command=None
                if command and command['action']=='align_udp_open':
                    samples=[];start=time.monotonic()+5.;end=start+1.
                    self.udp_alignment=dict(phase='prepare',remaining_s=5.)
                    self.calibration_error=None
                if start is None:return
                now=time.monotonic()
                if now<start:self.udp_alignment=dict(phase='prepare',remaining_s=start-now)
                elif now<end:self.udp_alignment=dict(phase='capture',remaining_s=end-now)
                else:
                    try:
                        path=save_open_reference(samples,self.hand)
                        with self.lock:
                            # New four-finger zero invalidates local pinch
                            # anchors fitted with a previous open reference.
                            self.config=dict(self.config,reference_capture=path,
                                f1_pinch_profile=None,official_f1_pinch_profile=None)
                            self.mapping_revision+=1
                        self.udp_alignment=dict(phase='ready',reference_capture=path)
                    except Exception as exc:
                        self.calibration_error=str(exc);self.udp_alignment=dict(phase='error')
                    start=end=None
            receiver.receive(self.stop_event,publish,tick)
        except Exception as exc:
            with self.lock:self.error='官方上位机 UDP 接收失败：'+str(exc);self.state='error'
            self.stop_event.set()

    def _run(self):
        try:
            self.retargeter=self.make_retargeter()
            mapping=self.retargeter.mapping_record()
            (ROOT/'results/g7_frame_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2,default=serializable),encoding='utf8')
            self.state='waiting'
            target={'serial':self._read_sdk,'g7_replay':self._read_replay,'g7_udp':self._read_udp}[self.source]
            self.reader_thread=threading.Thread(target=target,
                                                name='G7-device',daemon=True)
            self.reader_thread.start()
            seen=0
            while not self.stop_event.is_set():
                if self.loaded_revision!=self.mapping_revision:
                    self.retargeter=self.make_retargeter()
                    (ROOT/'results/g7_frame_mapping.json').write_text(json.dumps(self.retargeter.mapping_record(),
                        ensure_ascii=False,indent=2,default=serializable),encoding='utf8')
                    self.loaded_revision=self.mapping_revision
                    with self.lock:self.latest_output=None
                    if self.source=='g7_udp' and self.udp_alignment and self.udp_alignment['phase']=='ready':
                        self.motion_enabled=True
                if not self.motion_enabled:
                    self.stop_event.wait(.01);continue
                with self.lock: frame=self.latest_frame
                if frame is None or frame[0]==seen:
                    self.stop_event.wait(.003); continue
                seen,payload,received=frame
                try:
                    result=self.retargeter.solve_frame(payload)
                except ValueError as exc:
                    if self.config.get('retargeting_method') not in ('proportional','direct_joint'):raise
                    with self.lock:
                        self.rejected_frames+=1;self.last_frame_error=str(exc)
                    # Hold on malformed/degenerate positions. Later valid
                    # frames may recover without restarting the receiver.
                    continue
                self.last_frame_error=None
                result.update(sequence=seen,device_timestamp=payload['timestamp_ns'],device_seq=payload['seq'],
                    received_monotonic=received,solved_monotonic=time.monotonic(),profile_info=self.profile_info,
                    source_label={'serial':'G7 SDK 实时','g7_udp':'Forsense 官方手模 UDP','g7_replay':'G7 已采数据回放'}[self.source],
                    source_positions_m=payload['positions_m'],coil_confidence=payload.get('coil_confidence'))
                result['source_quaternions_wxyz']=payload['quaternions_wxyz']
                with self.lock:
                    self.latest_output=result; self.solve_count+=1; self.state='streaming'
        except Exception as exc:
            with self.lock: self.error='G7映射失败：'+str(exc); self.state='error'
            self.stop_event.set()

    def poll(self):
        with self.lock:
            return dict(state=self.state,error=self.error,frame_count=self.frame_count,solve_count=self.solve_count,
                frame_age_s=None if self.last_frame_time is None else time.monotonic()-self.last_frame_time,
                output=self.latest_output,calibration=(None if self.calibration is None else dict(self.calibration.state)),
                calibration_error=self.calibration_error,motion_enabled=self.motion_enabled,
                udp_alignment=self.udp_alignment,
                udp_receiver=None if self.udp_receiver is None else self.udp_receiver.stats(),
                rejected_frames=self.rejected_frames,last_frame_error=self.last_frame_error,
                mapping_ready=self.mapping_revision==self.loaded_revision)

    def calibration_command(self,action,**kwargs):
        if self.source!='serial':raise ValueError('SDK 标定只支持实时 G7 连接')
        self.motion_enabled=False
        self.commands.put(dict(action=action,**kwargs))

    def align_udp_open(self):
        if self.source!='g7_udp':raise ValueError('此入口只适用于官方上位机 UDP')
        self.motion_enabled=False
        self.commands.put(dict(action='align_udp_open'))

    def resume(self):
        if self.mapping_revision!=self.loaded_revision:raise ValueError('坐标映射正在加载，请稍候')
        if (self.calibration and self.calibration.vendor_changed and
                (not self.config.get('mapping_profile') or self.calibration.state.get('completed')!='alignment_ready')):
            raise ValueError('新个人标定需要先完成 F1 对齐')
        self.motion_enabled=True

    def stop(self): self.stop_event.set()

    def join(self,timeout=3.):
        self.thread.join(timeout)
        if self.reader_thread: self.reader_thread.join(timeout)
