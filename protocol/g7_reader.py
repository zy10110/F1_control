"""HumanHand FSNH v4 UDP decoder; no device commands or vendor SDK."""
import struct

HEADER = struct.Struct("<IBBBBIQI")
JOINT = struct.Struct("<7f")

def decode_udp(packet: bytes) -> dict:
    if len(packet) < HEADER.size:
        raise ValueError('Truncated header')
    magic, version, hand, count, status, seq, stamp, order = HEADER.unpack_from(packet)
    if magic != 0x464E4853 or version != 4 or order != 0x77787A79:
        raise ValueError('Not the documented FSNH v4/wxyz format')
    if hand not in (1, 2) or count not in (21, 22):
        raise ValueError('Invalid hand or joint count')
    if len(packet) != HEADER.size + JOINT.size * count:
        raise ValueError('Packet size disagrees with joint count')
    if status & 0xFC:
        raise ValueError('Unknown status bits')
    joints = [JOINT.unpack_from(packet, HEADER.size + JOINT.size * i) for i in range(count)]
    # Normalize forearm-root packets to the same 21-point wrist-root interface.
    if count == 22:
        import numpy as np
        from scipy.spatial.transform import Rotation
        wrist = joints[1]
        rw = Rotation.from_quat([*wrist[4:7], wrist[3]])
        poses = []
        for joint in joints[1:]:
            p = rw.inv().apply(np.asarray(joint[:3]) - wrist[:3])
            r = rw.inv() * Rotation.from_quat([*joint[4:7], joint[3]])
            q = r.as_quat()
            poses.append([*p.tolist(), float(q[3]), *q[:3].tolist()])
        joints = poses
    import math
    if not all(math.isfinite(v) for joint in joints for v in joint):
        raise ValueError('Non-finite pose')
    return {'source': 'udp', 'hand': hand, 'seq': seq, 'timestamp_ns': stamp,
            'new_solution': bool(status & 1), 'positions_m': [list(j[:3]) for j in joints],
            'quaternions_wxyz': [list(j[3:]) for j in joints]}
