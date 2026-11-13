"""G7 source IMU in transmitter axes and bounded F1 wrist motion (radians)."""
import struct
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


IMU_MAGIC = 0x464E494D
HEADER = struct.Struct('<IBBHIQI')


def decode_imu(packet):
    magic, version, hand, _, seq, stamp, order = HEADER.unpack_from(packet)
    if (magic != IMU_MAGIC or (version, len(packet)) not in ((2, 240), (3, 384))
            or hand not in (1, 2) or order != 0x77787A79):
        raise ValueError('无效 G7 FNIM 数据包')
    accel, gyro = np.array(struct.unpack_from('<6f', packet, len(packet)-24)).reshape(2, 3)
    if not np.isfinite([accel, gyro]).all() or not (accel.any() or gyro.any()):
        raise ValueError('无有效磁源 IMU 数据')
    slot = 24 + 5*(32 if version == 2 else 56)
    forearm = None
    if packet[slot+(28 if version == 2 else 52)] == 1:
        q = np.array(struct.unpack_from('<4f', packet, slot+12))
        if np.isfinite(q).all() and .8 < np.linalg.norm(q) < 1.2:
            forearm = Rotation.from_quat(q[[1,2,3,0]]).inv()
    return hand, seq, stamp, accel, gyro, forearm


class WristMotion:
    def __init__(self, model, display_rotation, hand):
        self.lock = threading.Lock()
        self.hand = 1 if hand == 'left' else 2
        # Transmitter axes are RFU on both hands. Right-hand joint packets
        # use left/back/up. Left-hand joint packets are mirrored by F1.
        self.basis = display_rotation @ np.diag([1,-1,1] if self.hand == 1 else [-1,-1,1])
        ids = [model.joint(name).id for name in ('r_wrist_abd','r_wrist_flex_joint')]
        self.qadr = model.jnt_qposadr[ids]
        self.limits = model.jnt_range[ids].copy()
        self.invalid = 0
        self.count = 0
        self.reset()

    def reset(self):
        with self.lock:
            self.stamp = self.seq = self.arrival = None
            self.attitude = Rotation.identity()
            self.gravity = self.forearm_zero = None
            self.angles = np.zeros(3)  # base roll, wrist abduction, wrist flexion
            self.using_forearm = False
            self.handover = Rotation.identity()

    def model_rotation(self, rotation):
        return Rotation.from_matrix(self.basis @ rotation.as_matrix() @ self.basis.T)

    def accept(self, packet):
        try:
            hand, seq, stamp, accel, gyro, forearm = decode_imu(packet)
        except (ValueError, struct.error):
            with self.lock:
                self.invalid += 1
            return
        if hand != self.hand:
            return
        with self.lock:
            now = time.monotonic()
            stale = self.arrival is None or now-self.arrival > .5
            if not stale and ((seq-self.seq) & 0xffffffff) not in range(1, 0x80000000):
                return
            dt = 0. if stale else (stamp-self.stamp)*1.e-9
            self.seq, self.stamp, self.arrival = seq, stamp, now
            self.count += 1
            if 0. < dt <= .12:
                self.attitude *= Rotation.from_rotvec(gyro*dt)
            norm = np.linalg.norm(accel)
            if abs(norm-9.80665) < .9:
                up = accel/norm
                if self.gravity is None:
                    self.gravity = self.attitude.apply(up)
                elif 0. < dt <= .12:
                    predicted = self.attitude.inv().apply(self.gravity)
                    self.attitude *= Rotation.from_rotvec(np.cross(up,predicted)*(-np.expm1(-dt/.8)))
            total = self.model_rotation(self.attitude)
            wrist = total
            had_forearm = self.using_forearm
            self.using_forearm = forearm is not None
            if forearm is not None:
                if self.forearm_zero is None:
                    # If the module first appears mid-gesture, preserve the
                    # two angles already shown in the viewer.
                    prior = Rotation.from_euler('YZ',[-self.angles[1],self.angles[2]])
                    local = Rotation.from_matrix(self.basis.T @ prior.as_matrix() @ self.basis)
                    self.forearm_zero = forearm * local.inv()
                wrist = self.model_rotation(self.forearm_zero.inv()*forearm)
            if had_forearm != self.using_forearm:
                prior = Rotation.from_euler('XYZ',[self.angles[0],-self.angles[1],self.angles[2]])
                self.handover = prior*wrist.inv()
            if 0. < dt <= .12:
                self.handover = Rotation.from_rotvec(self.handover.as_rotvec()*np.exp(-dt/.16))
            wrist = self.handover*wrist
            r = wrist.as_matrix()
            # Intrinsic X, -Y, Z about the native zero-pose axes. The X
            # rotation is a simulated forearm, outside the 37 URDF hinges.
            abd = -np.arctan2(r[0,2],np.hypot(r[0,0],r[0,1]))
            flex = np.arctan2(-r[0,1],r[0,0])
            self.angles[1:] = np.clip([abd,flex], self.limits[:,0], self.limits[:,1])
            if forearm is None:
                tr = wrist.as_matrix()
                roll = np.arctan2(-tr[1,2],tr[2,2])
            else:
                # Project the residual orientation onto the forearm X axis.
                q = (total*Rotation.from_euler('YZ',[-self.angles[1],self.angles[2]]).inv()).as_quat()
                roll = 2*np.arctan2(q[0],q[3]) if np.hypot(q[0],q[3])>1.e-8 else self.angles[0]
            self.angles[0] += np.arctan2(np.sin(roll-self.angles[0]),np.cos(roll-self.angles[0]))

    def snapshot(self):
        with self.lock:
            fresh = self.arrival is not None and time.monotonic()-self.arrival < .5
            return self.angles.copy(), dict(fresh=fresh, packets=self.count, invalid=self.invalid,seq=self.seq,timestamp_ns=self.stamp,
                forearm_reference=bool(fresh and self.using_forearm), angles_deg=np.degrees(self.angles).tolist())
