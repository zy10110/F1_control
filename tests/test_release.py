"""Portable regression checks; all test frames are synthetic."""
import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g7_official_udp import OfficialUDPReceiver
from urdf_a_model import NativeHand
from urdf_a_follow import Sweep

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('udp_parser', ROOT / 'protocol/g7_reader.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


@pytest.fixture(scope='module')
def model():
    return NativeHand(ROOT / 'assets/urdf_a')


def packet(seq=1, stamp=100, hand=2, count=21, status=1):
    header = reader.HEADER.pack(0x464E4853, 4, hand, count, status, seq, stamp, 0x77787A79)
    joints = []
    for i in range(count):
        joints.append(reader.JOINT.pack(i * .001, 0, 0, 1, 0, 0, 0))
    return header + b''.join(joints)


def test_distributed_model_hashes():
    records = json.loads((ROOT / 'docs/model_sha256.json').read_text(encoding='utf8'))
    assert len(records) == 39
    for path, expected in records.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_native_limits_and_fixed_wrist_across_independent_range(model):
    g = model
    rng = np.random.default_rng(20261111)
    for u in [g.lo, g.hi, np.zeros(17), *rng.uniform(g.lo, g.hi, size=(30, 17))]:
        g.assign(g.validate(u))
        assert np.all(g.data.qpos >= g.model.jnt_range[:, 0] - 1e-9)
        assert np.all(g.data.qpos <= g.model.jnt_range[:, 1] + 1e-9)
        for name in ('r_wrist_abd', 'r_wrist_flex_joint'):
            assert g.data.qpos[g.model.joint(name).qposadr[0]] == 0
    g.assign(np.zeros(17))


def test_semantic_flexion_split_and_distal_coupling(model):
    u = np.zeros(17)
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        cols = model.cols[finger]
        u[cols[-2:]] = np.radians([30, 25])
    model.assign(u)
    for (finger, segment), joints in model.pair_axes.items():
        angles = [model.data.qpos[model.model.jnt_qposadr[j]] for j in joints]
        assert angles[0] == pytest.approx(angles[1])
        expected = 30 if segment == 'mcp' else (20 if segment == 'dip' else 25)
        assert np.degrees(sum(angles)) == pytest.approx(expected)
    model.assign(np.zeros(17))


@pytest.mark.parametrize('bad', [np.full(17, np.nan), np.zeros(16), np.full(17, np.inf)])
def test_invalid_joint_targets_are_rejected(model, bad):
    with pytest.raises(ValueError):
        model.validate(bad)


def test_swept_execution_preserves_clearance_and_limits(model):
    g = model
    guard = Sweep(g)
    u = np.zeros(17)
    assert guard.safe(u)
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        target = np.zeros(17)
        target[g.cols[finger][-2:]] = np.radians([35, 40])
        target = np.clip(target, g.lo, g.hi)
        for goal in (target, np.zeros(17)):
            for _ in range(12):
                candidate = u + np.clip(goal - u, -np.radians(5), np.radians(5))
                candidate = guard.slide(u, candidate)
                u, _ = guard.sweep_follow_digits(u, candidate)
                g.assign(g.validate(u))
                assert g.clearance() >= .6 - 1e-8
    assert np.max(abs(u)) < 1e-6


def test_udp_right_hand_sequence_wrap_restart_and_other_hand():
    receiver = OfficialUDPReceiver(reader.decode_udp, 'right')
    assert receiver.accept(packet(seq=0xffffffff)) is not None
    assert receiver.accept(packet(seq=0, stamp=101)) is not None
    assert receiver.accept(packet(seq=0, stamp=102)) is None
    assert receiver.accept(packet(seq=1, hand=1)) is None
    assert receiver.accept(packet(seq=1, status=0)) is None
    assert receiver.accept(packet(seq=2, stamp=103)) is not None
    assert receiver.accept(packet(seq=1, stamp=104)) is None
    assert receiver.accept(packet(seq=1, stamp=1)) is not None
    assert receiver.stats()['other_hand'] == 1


def test_udp_forearm_packet_becomes_wrist_relative():
    wrist_r = Rotation.from_rotvec([.2, -.3, .4])
    wrist_p = np.array([.1, -.2, .3])
    expected = np.array([[i * .002, i * .001, 0] for i in range(21)])
    q = wrist_r.as_quat()[[3, 0, 1, 2]]
    poses = [[0, 0, 0, 1, 0, 0, 0]]
    poses += [[*p, *q] for p in wrist_r.apply(expected) + wrist_p]
    header = reader.HEADER.pack(0x464E4853, 4, 2, 22, 1, 1, 100, 0x77787A79)
    frame = reader.decode_udp(header + b''.join(reader.JOINT.pack(*p) for p in poses))
    np.testing.assert_allclose(frame['positions_m'], expected, atol=3e-8)
    np.testing.assert_allclose(frame['quaternions_wxyz'], np.tile([1, 0, 0, 0], (21, 1)), atol=1e-7)


def test_malformed_udp_is_rejected():
    receiver = OfficialUDPReceiver(reader.decode_udp, 'right')
    assert receiver.accept(packet()[:-1]) is None
    corrupt = bytearray(packet())
    corrupt[:4] = b'nope'
    assert receiver.accept(corrupt) is None
    invalid = packet()[:-reader.JOINT.size] + reader.JOINT.pack(0, 0, 0, 0, 0, 0, 0)
    assert receiver.accept(invalid) is None
    assert receiver.stats()['invalid'] == 3
