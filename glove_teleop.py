"""Glove FK -> model-sized positions -> coupled F1 position IK.

Only joint kinematics are displayed. No cable length or hardware commands.
The imported glove project remains the source of its FK and serial parser.
Legacy sample mode fixes root axes. Spatial mode retains root-frame XYZ and
jointly solves side swing and flexion, without copying glove joint angles.
"""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

from hand_kinematics import ROOT, FINGERS, HandGeometry, serializable, vector


def load_config():
    return json.loads((ROOT/'glove_teleop_config.json').read_text(encoding='utf8'))


def _module(path, name):
    key = '_f1_glove_'+name+'_'+hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(key, None)
            raise
    return sys.modules[key]


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-10:
        raise ValueError('无法从模型确定屈伸方向')
    return v/n


def _rotation(axis, angle):
    x, y, z = axis
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


class GloveModel:
    def __init__(self, hand='left', config=None, apply_feedback_calibration=False):
        self.config = config or load_config()
        self.path = (ROOT/self.config['glove_project']).resolve()
        self.hand = hand
        fk_module = _module(self.path/'glove_fk.py', 'fk')
        self.stream_module = _module(self.path/'serial_reader.py', 'serial')
        self.raw_fk=self.config.get('glove_fk_api')=='raw_latest'
        # Removing the first root rotation is intentional for flexion-only use.
        self.feedback_calibration_path = None
        if not self.raw_fk and apply_feedback_calibration and hand == 'left' and self.config.get('left_feedback_calibration'):
            self.feedback_calibration_path = (ROOT/self.config['left_feedback_calibration']).resolve()
        if self.raw_fk:
            kwargs=dict(hand=hand,j17_deg=self.config.get('latest_j17_deg'))
            self.fk=RawFKAdapter(fk_module.GloveFK(root_frame='child_link',**kwargs))
            self.root_fk=RawFKAdapter(fk_module.GloveFK(root_frame='joint_origin',**kwargs))
            self.nominal_root_fk=fk_module.GloveFK(root_frame='joint_origin',**kwargs)
        else:
            self.fk = fk_module.GloveFK(hand=hand, root_frame='child_link',
                                      feedback_calibration=self.feedback_calibration_path)
            # Spatial IK must retain the first joint's rotation in the FK output.
            self.root_fk = fk_module.GloveFK(hand=hand, root_frame='joint_origin',
                                           feedback_calibration=self.feedback_calibration_path)
            self.nominal_root_fk = fk_module.GloveFK(hand=hand, root_frame='joint_origin')
        self.root_axes = {}
        self.names = ['pinky' if f == 'little' else f for f in self.fk.names]
        if set(self.names) != set(FINGERS) or len(self.names) != 5:
            raise ValueError('手套五指命名与 F1 不一致')
        cfg = json.loads((self.path/'fk_config.json').read_text(encoding='utf8'))
        urdf_path = self.path/cfg['hands'][hand]['model']
        joints = {j.get('name'): j for j in ET.parse(urdf_path).getroot().findall('joint')}
        self.frames = {}; self.neutral_angles_deg = np.zeros((5, 4))
        self.sign = float(self.config['bend_sign'][hand])
        for row, spec in enumerate(cfg['fingers']):
            finger = self.names[row]
            first = joints[spec['joints'][0]]
            self.root_axes[finger] = unit(np.fromstring(first.find('axis').get('xyz'), sep=' '))
            chain = [joints[n] for n in spec['joints'][1:]]
            origins = [fk_module._origin_transform(j.find('origin')) for j in chain]
            axes = [unit(np.fromstring(j.find('axis').get('xyz'), sep=' ')) for j in chain]
            anchor = origins[0][:3, 3]
            axis = unit(origins[0][:3, :3]@axes[0])
            # Root-to-MCP installation offset defines the nominal outward ray.
            extension = unit(anchor-axis*np.dot(anchor, axis))
            bend = self.sign*unit(np.cross(axis, extension))
            tip = np.asarray(spec['tip_local'][hand], float)
            T = np.eye(4); length = 0.
            # Construct an exactly extended *model* pose for reproducible tests.
            # These angles are never used as the measured input's zero offsets.
            for k, (origin, local_axis) in enumerate(zip(origins, axes)):
                T = T@origin
                bone = origins[k+1][:3, 3] if k < 2 else tip
                a = unit(T[:3, :3]@local_axis)
                v = T[:3, :3]@bone; v = v-a*np.dot(a, v)
                desired = extension-a*np.dot(a, extension)
                q = np.arctan2(np.dot(a, np.cross(unit(v), unit(desired))),
                               np.dot(unit(v), unit(desired)))
                self.neutral_angles_deg[row, k+1] = np.degrees(q) % 360
                motion = np.eye(4); motion[:3, :3] = _rotation(local_axis, q)
                T = T@motion
                length += np.linalg.norm(v)
            self.frames[finger] = dict(anchor_m=anchor, extension=extension,
                bend=bend, axis=axis, planar_length_m=float(length), row=row)
        self.provenance = {str(p.relative_to(self.path)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (self.path/'glove_fk.py', self.path/'serial_reader.py',
                                     self.path/'fk_config.json', urdf_path)}
        if self.feedback_calibration_path is not None:
            self.provenance[str(self.feedback_calibration_path)] = hashlib.sha256(self.feedback_calibration_path.read_bytes()).hexdigest()

    def sample(self):
        parser = self.stream_module.FrameParser()
        with (self.path/'examples/sample_serial.txt').open(encoding='utf8') as stream:
            for line in stream:
                frame = parser.feed(line)
                if frame is not None:
                    return frame
        raise ValueError('随附样例没有完整五指帧')

    def positions(self, angles_deg):
        tips = self.fk.update(angles_deg)
        return {f: tips[i].copy() for i, f in enumerate(self.names)}

    def positions_root_origin(self, angles_deg):
        tips = self.root_fk.update(angles_deg)
        return {f: tips[i].copy() for i, f in enumerate(self.names)}

    def synthetic_angles(self, fraction):
        """Declared synthetic FK exercise, not a recorded human movement."""
        q = self.neutral_angles_deg.copy()
        for f, rec in self.frames.items():
            weights = np.array([35., 48., 32.]) if f == 'thumb' else np.array([38., 55., 38.])
            q[rec['row'], 1:] += self.sign*float(fraction)*weights
        return q


class RawFKAdapter:
    """Legacy controller interface for the new unmodified raw-angle FK API."""
    feedback_calibration=None

    def __init__(self,fk): self.raw=fk
    def __getattr__(self,name): return getattr(self.raw,name)
    def model_angles(self,angles_deg):
        q=np.asarray(angles_deg,float)
        if q.shape!=(5,4) or not np.isfinite(q).all(): raise ValueError('手套角度需要有限的 5×4 数组')
        return q.copy()
    def update(self,angles_deg): return self.raw.update(angles_deg)


class FlexionRetargeter:
    """Two angle variables per finger, all root axes fixed in the optimizer."""
    def __init__(self, geometry, initial_u, glove, config=None):
        self.g = geometry; self.glove = glove; self.config = config or glove.config
        self.reference = self.g.validate(initial_u).copy()
        self.previous = self.reference.copy(); self.frames = {}; self.grids = {}
        self.active = {}
        for finger in FINGERS:
            sl = self.g.slices[finger]
            indices = np.array([sl.stop-2, sl.stop-1])
            self.active[finger] = indices
            neutral = self.g.forward(self.reference, finger)['position_mm']
            spec = self.g.cfg['fingers'][finger]
            joint = self.g.model.joint(spec['columns'][-2][0][0])
            anchor = (self.g.data.xanchor[joint.id]-self.g.origin)*1000
            axis = unit(self.g.data.xaxis[joint.id])
            ray = neutral-anchor; planar = ray-axis*np.dot(axis, ray)
            extension = unit(planar)
            deriv = self.g.position_jacobian(finger)[:, -2]
            bend = unit(deriv-extension*np.dot(deriv, extension))
            # Retain the model's constant out-of-plane fingertip offset.
            length = float(np.linalg.norm(planar))
            scale = length/(1000*glove.frames[finger]['planar_length_m'])
            gain = float(self.config['finger_scale_gain'][finger])
            if not np.isfinite(gain) or gain <= 0:
                raise ValueError('手指缩放增益必须为正有限数')
            self.frames[finger] = dict(offset_mm=neutral-length*extension,
                extension=extension, bend=bend, length_mm=length, scale=scale*gain)
            n = int(self.config['grid_size'])
            if n < 3: raise ValueError('工作区网格至少 3 点')
            values = np.array(list(itertools.product(*[np.linspace(self.g.lower[i], self.g.upper[i], n) for i in indices])))
            points = []
            for pair in values:
                u = self.reference.copy(); u[indices] = pair
                points.append(self.g.forward(u, finger)['position_mm'])
            self.grids[finger] = (values, np.asarray(points))
        self.active_indices = np.concatenate(list(self.active.values()))
        self.fixed_indices = np.array([i for i in range(len(self.reference)) if i not in self.active_indices])

    def map_positions(self, tips):
        targets = {}; diagnostics = {}
        for finger in FINGERS:
            src = self.glove.frames[finger]; dst = self.frames[finger]
            local = vector(tips[finger], 3, '手套指尖 / m')-src['anchor_m']
            xy = np.array([np.dot(local, src['extension']), np.dot(local, src['bend'])])*1000
            target = dst['offset_mm']+dst['scale']*(dst['extension']*xy[0]+dst['bend']*xy[1])
            targets[finger] = target
            diagnostics[finger] = dict(glove_tip_root_m=tips[finger],
                glove_flexion_xy_mm=xy, f1_requested_position_mm=target, length_scale=dst['scale'])
        return targets, diagnostics

    def solve_position(self, finger, target, previous=None):
        """Approximate closest position: workspace grid + bounded multistarts.

        A global optimum is not claimed. The returned error never exceeds the
        closest dense-grid candidate, including targets outside the workspace.
        """
        target = vector(target, 3, 'F1 目标 / mm')
        indices = self.active[finger]; sl = self.g.slices[finger]
        base = self.reference.copy()
        previous = self.previous if previous is None else self.g.validate(previous)
        values, points = self.grids[finger]
        distances = np.linalg.norm(points-target, axis=1)
        best = np.argsort(distances)[:int(self.config['multistarts'])]
        def residual(pair):
            u = base.copy(); u[indices] = pair
            return self.g.forward(u, finger)['position_mm']-target
        def jac(pair):
            residual(pair)
            return self.g.position_jacobian(finger)[:, indices-sl.start]
        candidates = [(float(distances[best[0]]), values[best[0]].copy())]
        starts = [previous[indices]]+list(values[best])
        for start in starts:
            fit = least_squares(residual, np.clip(start, self.g.lower[indices]+1e-10,
                self.g.upper[indices]-1e-10), jac=jac,
                bounds=(self.g.lower[indices], self.g.upper[indices]),
                max_nfev=55, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            candidates.append((float(np.linalg.norm(residual(fit.x))), fit.x.copy()))
        min_error = min(x[0] for x in candidates)
        error, pair = min((v for v in candidates if v[0] <= min_error+1e-5),
                         key=lambda v: np.linalg.norm(v[1]-previous[indices]))
        u = base.copy(); u[indices] = pair
        pose = self.g.forward(u, finger)
        return u, dict(success=True, projected=bool(error > self.g.cfg['position_tolerance_mm']),
            requested_position_mm=target, execution_position_mm=pose['position_mm'],
            position_error_mm=error, grid_nearest_error_mm=float(distances[best[0]]),
            optimized_labels=[self.g.labels[i] for i in indices],
            fixed_labels=[self.g.labels[i] for i in range(sl.start, sl.stop) if i not in indices],
            coordinates_rad=u[sl], solver='fixed_root_flexion_multistart_position_IK')

    def solve(self, angles_deg):
        started = time.perf_counter()
        tips = self.glove.positions(angles_deg)
        targets, mapping = self.map_positions(tips)
        goal = self.reference.copy(); records = {}
        for finger in FINGERS:
            u, record = self.solve_position(finger, targets[finger])
            goal[self.active[finger]] = u[self.active[finger]]
            records[finger] = record
        self.previous = goal.copy()
        return dict(goal_rad=goal, fingers=records, mapping=mapping,
            angles_deg=np.asarray(angles_deg).copy(), solve_ms=(time.perf_counter()-started)*1000,
            model_angles_deg=self.glove.fk.model_angles(angles_deg),
            feedback_calibration_id=(None if self.glove.fk.feedback_calibration is None else
                self.glove.fk.feedback_calibration.record['id']),
            hand=self.glove.hand, root_frame='child_link',
            pipeline='device feedback -> optional zero/direction adapter -> glove URDF FK -> size/frame mapping -> fixed-root angle IK -> joint angles -> face distances',
            projection_method='dense grid + bounded multistart; numerical approximate nearest',
            feedback_zero_direction_adapted=self.glove.fk.feedback_calibration is not None,
            calibrated_to_wearer=False, calibration_scope='two-pose zero/direction only; no metric fingertip calibration',
            hardware_enabled=False)


def follow_step(current, goal, dt, reference, fixed_indices, config):
    """Bounded first-order response; preserves all linear couplings via u."""
    current = np.asarray(current, float); goal = np.asarray(goal, float)
    if not np.isfinite(dt) or dt < 0: raise ValueError('跟随时间无效')
    dt = min(float(dt), .1)
    alpha = -np.expm1(-dt/float(config['filter_time_constant_s']))
    step = (goal-current)*alpha
    maxstep = np.radians(float(config['max_joint_speed_deg_s']))*dt
    size = float(np.max(np.abs(step)))
    if size > maxstep and size > 0: step *= maxstep/size
    u = current+step
    u[fixed_indices] = np.asarray(reference)[fixed_indices]
    return u


def follow_digits(current,goal,dt,reference,slices,config,previous_velocity=None,*,response=None,goal_velocity=None):
    """Critically damped tracking with the original speed ceiling per digit."""
    if not np.isfinite(dt) or dt<0:raise ValueError('跟随时间无效')
    if dt==0:return current.copy()
    interval=min(dt,.1)
    last=np.zeros_like(current) if previous_velocity is None or dt>.1 else previous_velocity
    # A 28 ms pole for the default personal profile softens reversals while
    # each digit retains its full steady movement range.
    omega=1.25/float(config['filter_time_constant_s'])
    error=goal-current
    if response is not None:
        # Shorten the response time only on digits that are falling behind.
        # Keep the original damping for sub-degree movements and settling.
        base_omega = omega
        omega = np.full_like(error, base_omega)
        transition = np.radians(response['transition_deg'])
        for sl in slices.values():
            ratio = np.max(np.abs(error[sl])) / transition
            omega[sl] *= 1.+(response['max_rate_ratio']-1.)*ratio**2/(1.+ratio**2)
    decay=np.exp(-omega*interval)
    step=error*(1.-(1.+omega*interval)*decay)+last*interval*decay
    if goal_velocity is not None:
        # Exact critically damped update in the frame of a linearly moving
        # target, expressed relative to the newest target sample.
        step+=goal_velocity*omega*interval**2*decay
    maxstep=np.radians(config['max_joint_speed_deg_s'])*interval
    for sl in slices.values():
        largest=np.max(np.abs(step[sl]))
        if largest>maxstep:step[sl]*=maxstep/largest
    # A reversal or collision can require stopping sooner. Never overshoot
    # a joint target or carry velocity through its hard limit.
    return current+np.clip(step,np.minimum(0.,goal-current),np.maximum(0.,goal-current))


class TeleopSession:
    """Independent reader + solver, one newest-frame slot, no serial writes."""
    def __init__(self, mode, reference, hand='left', source='sample', port=None, baudrate=921600, config=None):
        if source not in ('sample', 'spatial_sample', 'serial', 'feedback_replay','approximate_replay'): raise ValueError('未知手套数据源')
        if source == 'serial' and not port: raise ValueError('请选择手套串口')
        self.config = config or load_config(); self.mode = mode
        self.reference = np.asarray(reference, float).copy(); self.hand = hand
        self.source = source; self.port = port; self.baudrate = int(baudrate)
        self.stop_event = threading.Event(); self.lock = threading.Lock()
        self.latest_frame = None; self.latest_output = None; self.last_frame_time = None
        self.frame_count = 0; self.solve_count = 0; self.error = None
        self.state = 'initializing'; self.retargeter = None; self.reader_thread = None
        self.thread = threading.Thread(target=self._run, name='F1-glove-IK', daemon=True)

    def start(self): self.thread.start()

    def _publish(self, angles, stamp, label):
        with self.lock:
            self.frame_count += 1; self.last_frame_time = time.monotonic()
            self.latest_frame = (self.frame_count, np.asarray(angles).copy(), stamp, label, self.last_frame_time)

    def _read(self, glove):
        try:
            with glove.stream_module.SerialReader(self.port, baudrate=self.baudrate, timeout=.02) as reader:
                while not self.stop_event.is_set():
                    frame = reader.read_latest_frame() if glove.raw_fk else reader.read_frame()
                    if frame is not None: self._publish(frame.angles_deg, frame.timestamp, '实时手套')
        except Exception as exc:
            with self.lock: self.error = '串口读取失败：'+str(exc); self.state = 'error'
            self.stop_event.set()

    def _run(self):
        try:
            approximate=self.source=='approximate_replay' or (self.source=='serial' and self.config.get('retargeting_method')=='approximate')
            glove = GloveModel(self.hand, self.config, apply_feedback_calibration=not approximate and self.source not in ('sample','spatial_sample'))
            if self.source=='feedback_replay' and glove.fk.feedback_calibration is None:
                raise ValueError('当前手侧没有本轮反馈方向配置')
            g = HandGeometry(self.mode)
            if approximate:
                from glove_approximate import ApproximateRetargeter
                self.retargeter=ApproximateRetargeter(g,self.reference,glove,self.config)
            else:
                self.retargeter = FlexionRetargeter(g, self.reference, glove, self.config)
            if not approximate and (self.source=='spatial_sample' or (self.source=='serial' and self.config.get('side_swing_enabled',False))):
                from glove_spatial_ik import SpatialPositionRetargeter
                self.retargeter=SpatialPositionRetargeter(self.retargeter)
            sample = glove.sample()
            with self.lock: self.state = 'waiting' if self.source == 'serial' else 'sample'
            if self.source == 'serial':
                self.reader_thread = threading.Thread(target=self._read, args=(glove,), name='F1-glove-read', daemon=True)
                self.reader_thread.start()
            seen = 0; origin = time.monotonic()
            while not self.stop_event.is_set():
                if self.source=='approximate_replay':
                    angles,label=self.retargeter.replay_frame(time.monotonic()-origin)
                    self._publish(angles,'recorded_example_replay',label)
                elif self.source == 'sample':
                    t = (time.monotonic()-origin) % 12.
                    if t < 2.:
                        angles, label = glove.neutral_angles_deg, '模型伸直样例（合成）'
                    elif t < 5.:
                        angles, label = sample.angles_deg, '项目随附原始样例'
                    else:
                        fraction = .5-.5*np.cos(2*np.pi*(t-5.)/7.)
                        angles, label = glove.synthetic_angles(fraction), '模型屈伸扫描（合成）'
                    self._publish(angles, 'sample', label)
                elif self.source == 'feedback_replay':
                    profile=glove.fk.feedback_calibration.record
                    fraction=.5-.5*np.cos(2*np.pi*(time.monotonic()-origin)/9.)
                    angles=np.asarray(profile['feedback_reference_deg'])+fraction*np.asarray(profile['observed_flex_feedback_delta_deg'])
                    self._publish(angles,'feedback_pose_interpolation','本轮张开—屈曲插值回放（已适配）')
                elif self.source == 'spatial_sample':
                    t = time.monotonic()-origin
                    fraction = .5-.5*np.cos(2*np.pi*t/9.)
                    angles = glove.synthetic_angles(.6*fraction)
                    for f, amplitude in dict(thumb=-20,index=10,middle=4,ring=-10,pinky=-10).items():
                        angles[glove.frames[f]['row'],0] = amplitude*fraction
                    self._publish(angles,'synthetic_spatial_FK','三维指尖样例（合成；侧摆由IK求解）')
                with self.lock: frame = self.latest_frame
                if frame is None or frame[0] == seen:
                    self.stop_event.wait(.005); continue
                seen, angles, stamp, label, received = frame
                result = self.retargeter.solve(angles)
                result.update(sequence=seen, device_timestamp=stamp, source_label=label,
                              glove_project=str(glove.path),glove_fk_api='raw_latest' if glove.raw_fk else 'legacy',
                              glove_provenance=glove.provenance,j17_deg=getattr(glove.fk,'j17_deg',None),
                              received_monotonic=received, solved_monotonic=time.monotonic())
                with self.lock:
                    self.latest_output = result; self.solve_count += 1
                    if self.error is None: self.state = 'streaming' if self.source == 'serial' else 'sample'
                if self.source != 'serial': self.stop_event.wait(.015)
        except Exception as exc:
            with self.lock: self.error = str(exc); self.state = 'error'
            self.stop_event.set()

    def poll(self):
        with self.lock:
            age = None if self.last_frame_time is None else time.monotonic()-self.last_frame_time
            return dict(state=self.state, error=self.error, frame_count=self.frame_count,
                solve_count=self.solve_count, frame_age_s=age, output=self.latest_output)

    def stop(self):
        self.stop_event.set()

    def join(self, timeout=3.):
        self.thread.join(timeout)
        if self.reader_thread: self.reader_thread.join(timeout)


def save_session_summary(result, source, mode):
    path = ROOT/'results/glove_last_result.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(source=source, simulation_mode=mode, result=result),
        ensure_ascii=False, indent=2, default=serializable), encoding='utf8')
    return path
