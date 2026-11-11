"""Fingertip-position retargeting with side swing solved by bounded IK.

Source FK uses the installation frame before the glove root rotation. One
fixed similarity/reflection maps that XYZ to the F1 palm frame per finger.
Only XYZ enters the optimizer; no glove angle is copied to an F1 joint.
"""
import itertools
import json
import time

import numpy as np

from hand_kinematics import ROOT, FINGERS, vector
from glove_teleop import FlexionRetargeter, _rotation, unit


class SpatialPositionRetargeter(FlexionRetargeter):
    def __init__(self, flexion):
        self.g = flexion.g; self.glove = flexion.glove
        self.config = dict(flexion.config,
                           multistarts=int(flexion.config.get('spatial_multistarts', 2)))
        self.reference = flexion.reference.copy()
        self.previous = self.reference.copy()
        self.active = {}; self.grids = {}; self.frames = {}
        self.open_reference_id=None
        nominal = self.glove.neutral_angles_deg.copy()
        self.neutral_profile_id = None
        # Use the measured root NEUTRAL only to define a fixed source frame.
        # The former per-channel gain/sign/deadband do not control the hand.
        if self.glove.hand == 'left' and self.glove.fk.feedback_calibration is not None:
            path = self.config.get('side_swing_profile')
            if path:
                profile = json.loads((ROOT/path).read_text(encoding='utf8'))
                if profile['hand'] != self.glove.hand or profile['units'] != 'deg':
                    raise ValueError('指根基准的手侧或单位不匹配')
                feedback = self.glove.fk.feedback_calibration.feedback_zero.copy()
                for f in FINGERS:
                    feedback[self.glove.frames[f]['row'], 0] = profile['channels'][f]['feedback_zero_deg']
                nominal[:, 0] = self.glove.fk.model_angles(feedback)[:, 0]
                self.neutral_profile_id = profile['id']
        self.nominal_angles_deg = nominal.copy()
        tips0 = self.glove.nominal_root_fk.update(nominal)
        n = int(self.config.get('spatial_grid_size', 13))
        if n < 3: raise ValueError('三维工作区网格至少 3 点')
        for f in FINGERS:
            sl = self.g.slices[f]
            # Thumb CMC is needed for opposition; fixing it prevents most
            # fingertip pairs from meeting even in the nominal model.
            indices = np.arange(sl.start, sl.stop)
            self.active[f] = indices
            src = self.glove.frames[f]; dst = flexion.frames[f]
            R0 = _rotation(self.glove.root_axes[f], np.radians(nominal[src['row'], 0]))
            source_basis = R0@np.column_stack((src['extension'], src['bend'],
                                                unit(np.cross(src['extension'], src['bend']))))
            # Left glove -> nominal F1 hand includes a lateral reflection;
            # flexion axes and length scale are unchanged from the prior path.
            parity = -1. if self.glove.hand == 'left' else 1.
            destination_basis = np.column_stack((dst['extension'], dst['bend'],
                                parity*unit(np.cross(dst['extension'], dst['bend']))))
            self.frames[f] = dict(source_neutral_m=tips0[src['row']].copy(),
                destination_neutral_mm=self.g.forward(self.reference, f)['position_mm'].copy(),
                transform=destination_basis@source_basis.T, scale=dst['scale'],
                destination_basis=destination_basis,destination_radius_mm=dst['length_mm'])
            grid_count = int(self.config.get('spatial_thumb_grid_size', 4)) if f=='thumb' else n
            if grid_count < 3: raise ValueError('拇指工作区网格至少 3 点')
            values = np.array(list(itertools.product(*[
                np.linspace(self.g.lower[i], self.g.upper[i], grid_count) for i in indices])))
            points = []
            for angles in values:
                u = self.reference.copy(); u[indices] = angles
                points.append(self.g.forward(u, f)['position_mm'].copy())
            self.grids[f] = (values, np.asarray(points))
        self.active_indices = np.concatenate(list(self.active.values()))
        self.fixed_indices = np.array([i for i in range(len(self.reference))
                                       if i not in self.active_indices], dtype=int)
        path=self.config.get('latest_open_reference')
        if self.glove.raw_fk and path:
            self.set_open_reference(json.loads((ROOT/path).read_text(encoding='utf8')))

    def set_open_reference(self,record):
        """Align FK endpoint displacements; never alter raw feedback or FK.

        The measured root angle, root-to-MCP axis and measured endpoint ray
        determine a local frame/radius. One pose cannot calibrate fingertip
        pad contact or validate the physical joint-angle definitions.
        """
        if (not self.glove.raw_fk or record.get('schema_version')!=1 or
                record.get('hand')!=self.glove.hand or record.get('units')!='deg' or
                record.get('glove_provenance')!=self.glove.provenance):
            raise ValueError('张开参考与当前新版 FK 模型、手侧或单位不一致')
        q=np.asarray(record['angles_deg'],float)
        if q.shape!=(5,4) or not np.isfinite(q).all(): raise ValueError('张开参考角度无效')
        tips=self.glove.positions_root_origin(q); updates={}
        for f in FINGERS:
            src=self.glove.frames[f]; dst=self.frames[f]
            R0=_rotation(self.glove.root_axes[f],np.radians(q[src['row'],0]))
            axis=R0@src['axis']; anchor=R0@src['anchor_m']
            ray=tips[f]-anchor; planar=ray-axis*np.dot(ray,axis)
            radius=float(np.linalg.norm(planar))
            if not np.isfinite(radius) or radius<.01: raise ValueError('张开参考有效指长过小：'+f)
            extension=unit(planar); bend=self.glove.sign*unit(np.cross(axis,extension))
            source_basis=np.column_stack((extension,bend,unit(np.cross(extension,bend))))
            updates[f]=dict(source_neutral_m=tips[f].copy(),
                transform=dst['destination_basis']@source_basis.T,
                scale=dst['destination_radius_mm']/(1000*radius)*float(self.config['finger_scale_gain'][f]),
                source_reference_radius_m=radius,source_reference_basis=source_basis)
        for f,values in updates.items(): self.frames[f].update(values)
        self.open_reference_id=record['id']; self.previous=self.reference.copy()

    def map_positions(self, tips):
        targets = {}; diagnostics = {}
        for f in FINGERS:
            src = vector(tips[f], 3, '手套指根坐标系指尖 / m')
            frame = self.frames[f]
            target = frame['destination_neutral_mm'] + frame['scale']*frame['transform']@(
                (src-frame['source_neutral_m'])*1000.)
            targets[f] = target
            diagnostics[f] = dict(glove_tip_root_m=src, f1_requested_position_mm=target,
                length_scale=frame['scale'], transform=frame['transform'],
                source_neutral_m=frame['source_neutral_m'],
                destination_neutral_mm=frame['destination_neutral_mm'],open_reference_id=self.open_reference_id)
        return targets, diagnostics

    def solve_position(self, finger, target, previous=None):
            # All coupled position Jacobian columns are active (thumb: five).
        u, record = super().solve_position(finger, target, previous)
        record['solver'] = 'side_flexion_coupled_multistart_position_IK'
        return u, record

    def solve_tips(self, tips):
        """Controller boundary: five XYZ values, with no glove joint input."""
        started = time.perf_counter()
        targets, mapping = self.map_positions(tips)
        goal = self.reference.copy(); records = {}; side = {}
        for f in FINGERS:
            u, rec = self.solve_position(f, targets[f])
            goal[self.active[f]] = u[self.active[f]]
            records[f] = rec
            idx = self.g.slices[f].start+(2 if f=='thumb' else 0)
            side[f] = dict(actual_goal_deg=float(np.degrees(goal[idx])),
                limits_deg=np.degrees([self.g.lower[idx], self.g.upper[idx]]),
                coordinate=self.g.labels[idx],
                at_limit=bool(min(goal[idx]-self.g.lower[idx],self.g.upper[idx]-goal[idx]) < 1e-5))
        self.previous = goal.copy()
        return dict(goal_rad=goal, fingers=records, mapping=mapping,
            solve_ms=(time.perf_counter()-started)*1000,
            side_swing=dict(enabled=True, method='fingertip_position_IK', fingers=side,
                neutral_profile_id=self.neutral_profile_id,
                thumb_scope='CMC-OPP / CMC-PS / MCP-AA / MCP / IP optimized from endpoint XYZ'),
            hand=self.glove.hand, root_frame='joint_origin',
            pipeline='feedback -> calibrated glove FK -> 3D frame/size mapping -> coupled position IK -> angles -> face distances',
            projection_method='3D grid + bounded multistart; numerical approximate nearest',
            open_reference_id=self.open_reference_id,open_pose_aligned=self.open_reference_id is not None,
            calibrated_to_wearer=False, hardware_enabled=False)

    def solve(self, angles_deg):
        started = time.perf_counter()
        tips=self.glove.positions_root_origin(angles_deg)
        result = self.solve_tips(tips)
        calibration = self.glove.fk.feedback_calibration
        result.update(angles_deg=np.asarray(angles_deg).copy(),
            model_angles_deg=self.glove.fk.model_angles(angles_deg),
            feedback_calibration_id=None if calibration is None else calibration.record['id'],
            feedback_zero_direction_adapted=calibration is not None,
            calibration_scope='nominal model frames and length scales; measured feedback zero/direction only',
            solve_ms=(time.perf_counter()-started)*1000)
        if self.glove.raw_fk:
            result.update(glove_tips_root_m=tips,
                pipeline='raw feedback degrees -> glove_latest FK (joint_origin, meters) -> fixed frame/size mapping -> coupled bounded F1 fingertip IK -> joint angles -> face distances',
                calibration_scope='glove_latest raw FK + nominal per-finger frame/size mapping; no legacy feedback corrections; fingertip pads not calibrated')
            if self.open_reference_id is not None:
                result['calibration_scope']='single open-pose FK endpoint frame/radius alignment; raw angles and FK unchanged; fingertip pads not calibrated'
        return result
