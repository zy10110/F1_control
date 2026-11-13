"""Receive HumanHand's solved poses without opening or recalibrating COM7."""
import json
import socket
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hand_kinematics import ROOT


class OfficialUDPReceiver:
    def __init__(self, decoder, hand, bind='127.0.0.1', port=15001, imu_sink=None):
        if bind != '127.0.0.1':
            raise ValueError('官方上位机入口仅绑定本机 127.0.0.1')
        self.imu_sink = imu_sink
        self.decoder = decoder
        self.hand = 1 if hand == 'left' else 2
        self.bind, self.port = bind, int(port)
        self.last_seq = None
        self.last_arrival = None
        self.invalid = self.duplicates = self.other_hand = 0
        self.packet_count = 0
        self.last_packet_bytes = 0
        self.last_error = None

    def accept(self, packet):
        if self.imu_sink is not None and packet.startswith(b"MINF"):
            self.imu_sink(packet)
            return None
        self.packet_count += 1
        self.last_packet_bytes = len(packet)
        try:
            frame = self.decoder(packet)
            q = np.asarray(frame['quaternions_wxyz'], float)
            if q.shape != (21, 4) or not np.isfinite(q).all() or np.min(np.linalg.norm(q, axis=1)) < 1e-6:
                raise ValueError('无效骨段四元数')
        except (ValueError, KeyError, TypeError) as exc:
            self.invalid += 1
            self.last_error = str(exc)
            return None
        if frame['hand'] != self.hand:
            self.other_hand += 1
            self.last_error = '收到另一手别的关节位姿'
            return None
        if not frame['new_solution']:
            self.duplicates += 1
            return None
        # Accept forward/wrapped sequence numbers. A timestamp reset permits
        # restarting the upstream app without reconnecting the F1 receiver.
        seq = int(frame['seq'])
        stamp = int(frame['timestamp_ns'])
        now=time.monotonic()
        if self.last_seq is not None:
            advance = (seq - self.last_seq) & 0xffffffff
            restarted=now-self.last_arrival>.5 and stamp>self.last_stamp
            if not restarted and (advance == 0 or (advance > 0x80000000 and stamp >= self.last_stamp)):
                self.duplicates += 1
                return None
        self.last_seq, self.last_stamp = seq, stamp
        self.last_arrival=now
        self.last_error = None
        return frame

    def stats(self):
        return dict(packet_count=self.packet_count,invalid=self.invalid,duplicates=self.duplicates,
            other_hand=self.other_hand,last_packet_bytes=self.last_packet_bytes,last_error=self.last_error)

    def receive(self, stop, publish, tick=lambda: None):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.bind, self.port))
            sock.settimeout(.02)
            while not stop.is_set():
                tick()
                try:
                    packet, sender = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                if sender[0] != '127.0.0.1':
                    continue
                frame = self.accept(packet)
                if frame is not None:
                    publish(frame)


def open_reference(frames):
    """Average one short open-hand sample, rejecting substantial motion."""
    if len(frames) < 20:
        raise ValueError('有效数据不足，请先开启官方关节位姿 UDP')
    qs = np.asarray([f['quaternions_wxyz'] for f in frames], float)
    rotations = [Rotation.from_quat(qs[:, i, [1, 2, 3, 0]]) for i in range(21)]
    means = [r.mean() for r in rotations]
    spread = max(float(np.percentile((m.inv()*r).magnitude(), 90)) for m, r in zip(means, rotations))
    if spread > np.radians(8):
        raise ValueError('张开采样时仍在明显运动，请再点一次张开对齐')
    frame = dict(frames[-1])
    frame['quaternions_wxyz'] = np.asarray([m.as_quat()[[3, 0, 1, 2]] for m in means]).tolist()
    frame['positions_m'] = np.median([f['positions_m'] for f in frames], axis=0).tolist()
    return frame


def official_config(config):
    """Never reuse a different SDK session's wearer calibration/zero pose."""
    result = dict(config, mapping_profile=None, active_vendor_profile=None,
        f1_pinch_profile=config.get('official_f1_pinch_profile'),
        thumb_flexion_mode='absolute_joint',
        thumb_root_frame_mode='bone_and_bend_plane',
        thumb_hinge_axes={'1':[0.,0.,-1.],'2':[0.,0.,-1.]})
    result['retargeting_method']=config.get('official_retargeting_method',config.get('retargeting_method','joint'))
    if result['retargeting_method']=='proportional':
        result['f1_pinch_profile']=None
    if result['retargeting_method']=='direct_joint':
        result.update(f1_pinch_profile=None,thumb_flexion_mode='absolute_joint',
                      thumb_root_frame_mode='anatomical_angular_axes',joint_amplitude_mode='angle')
    saved = config.get('official_app_reference_capture')
    if saved and not Path(saved).is_absolute():
        saved=str((ROOT/saved).resolve())
    if not saved and config.get('official_f1_pinch_profile'):
        # Retain the existing solved-stream reference for geometric lengths;
        # proportional mode does not consume the fitted F1 pinch offsets.
        profile=json.loads(Path(config['official_f1_pinch_profile']).read_text(encoding='utf8'))
        saved=profile['open_reference_capture']
    if not saved and result['f1_pinch_profile']:
        profile=json.loads(Path(result['f1_pinch_profile']).read_text(encoding='utf8'))
        saved=profile['open_reference_capture']
    if saved and Path(saved).is_file():
        result['reference_capture'] = saved
    return result


def save_open_reference(frames, hand):
    frame = open_reference(frames)
    folder = ROOT/'calibration'/'g7_official_app'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'{time.time_ns()}_{hand}_open.jsonl'
    path.write_text(json.dumps(frame, ensure_ascii=False)+'\n', encoding='utf8')
    return str(path)
