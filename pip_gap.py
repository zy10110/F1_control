"""Nonlinear nominal PIP stop-face calibration; distances in mm, angles in deg.

The confirmed finite patches are measured by the same sampled surface metric
used in the survey. This is not a collision test or a physical calibration.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator


def _scalar(value, name):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(name + '必须是有限数值')
    return result


def _patch_samples(vertices, faces):
    triangles = vertices[faces]
    edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    return np.vstack([vertices, triangles.mean(axis=1), vertices[edges].mean(axis=1)])


def _closest(points, vertices, faces):
    """Vectorized closest point on a triangle mesh for each supplied sample."""
    triangles = vertices[faces]
    a, b, c = (triangles[:, i] for i in range(3))
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    norm2 = np.einsum('ij,ij->i', normal, normal)
    if np.any(norm2 < 1e-24):
        raise ValueError('测量面包含退化三角形')
    ap = points[:, None, :]-a
    projected = points[:, None, :] - (np.einsum('ptj,tj->pt', ap, normal)/norm2)[..., None]*normal
    v = projected-a
    d00 = np.einsum('ij,ij->i', ab, ab); d01 = np.einsum('ij,ij->i', ab, ac)
    d11 = np.einsum('ij,ij->i', ac, ac)
    d20 = np.einsum('ptj,tj->pt', v, ab); d21 = np.einsum('ptj,tj->pt', v, ac)
    denom = d00*d11-d01*d01
    u = (d11*d20-d01*d21)/denom; w = (d00*d21-d01*d20)/denom
    inside = (u >= -1e-12) & (w >= -1e-12) & (u+w <= 1+1e-12)
    distances = np.sum((projected-points[:, None, :])**2, axis=-1)
    distances[~inside] = np.inf
    nearest = projected.copy()
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end-start
        fraction = np.clip(np.einsum('ptj,tj->pt', points[:, None, :]-start, edge) /
                           np.einsum('ij,ij->i', edge, edge), 0, 1)
        candidate = start+fraction[..., None]*edge
        square = np.sum((candidate-points[:, None, :])**2, axis=-1)
        better = square < distances
        distances[better] = square[better]; nearest[better] = candidate[better]
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    return float(np.sqrt(distances[i, j])), points[i].copy(), nearest[i, j].copy()


def measure_patch_pair(vertices_a_mm, faces_a, vertices_b_mm, faces_b):
    a = np.asarray(vertices_a_mm, float); b = np.asarray(vertices_b_mm, float)
    fa = np.asarray(faces_a, int); fb = np.asarray(faces_b, int)
    first = _closest(_patch_samples(a, fa), b, fb)
    reverse = _closest(_patch_samples(b, fb), a, fa)
    second = (reverse[0], reverse[2], reverse[1])
    return min((first, second), key=lambda item: item[0])


class GapCalibration:
    def __init__(self, record):
        self.record = record
        self.angles_deg = np.asarray(record['pip_angles_deg'], float)
        self.gaps_mm = np.asarray(record['gaps_mm'], float)
        if (self.angles_deg.ndim != 1 or self.gaps_mm.shape != self.angles_deg.shape or
                len(self.angles_deg) < 4 or not np.isfinite(self.angles_deg).all() or
                not np.isfinite(self.gaps_mm).all() or np.any(np.diff(self.angles_deg) <= 0) or
                np.any(self.gaps_mm < 0)):
            raise ValueError('非线性间距标定表无效')
        if record.get('units') != {'length': 'mm', 'angle': 'deg'}:
            raise ValueError('间距表单位必须为 mm / deg')
        if not record.get('faces_confirmed_by_user'):
            raise ValueError('间距表测量面尚未确认')
        self.curve = PchipInterpolator(self.angles_deg, self.gaps_mm, extrapolate=False)
        extrema = self.curve.derivative().roots(extrapolate=False)
        x = np.r_[self.angles_deg, extrema[np.isfinite(extrema)]]
        self.minimum_mm, self.maximum_mm = float(np.nanmin(self.curve(x))), float(np.nanmax(self.curve(x)))

    @classmethod
    def load(cls, root, settings):
        spec = settings['distance_calibration']
        if spec.get('method') != 'sampled_pchip':
            raise ValueError('当前未启用多姿态非线性间距映射')
        path = (Path(root)/spec['table_file']).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError('间距标定文件必须位于工程目录内')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec['table_sha256']:
            raise ValueError('间距标定表校验失败，请重新生成和验证')
        return cls(json.loads(raw))

    def gap_from_angle(self, pip_deg):
        angle = _scalar(pip_deg, 'PIP 角度')
        if angle < self.angles_deg[0]-1e-9 or angle > self.angles_deg[-1]+1e-9:
            raise ValueError('PIP 角度超出非线性标定范围')
        return float(self.curve(np.clip(angle, self.angles_deg[0], self.angles_deg[-1])))

    def angles_from_gap(self, gap_mm, current_pip_deg=0., ratio=.8):
        gap = _scalar(gap_mm, '间距'); seed = _scalar(current_pip_deg, '当前 PIP 角度')
        ratio = _scalar(ratio, 'DIP/PIP 比例')
        if ratio <= 0: raise ValueError('耦合比例必须为正数')
        if gap < self.minimum_mm-1e-12 or gap > self.maximum_mm+1e-12:
            raise ValueError(f'间距超出标定范围 {self.minimum_mm:.9f}–{self.maximum_mm:.9f} mm，不允许外推')
        gap = float(np.clip(gap, self.minimum_mm, self.maximum_mm))
        roots = self.curve.solve(gap, extrapolate=False)
        roots = sorted(float(x) for x in roots if np.isfinite(x) and
                       self.angles_deg[0]-1e-8 <= x <= self.angles_deg[-1]+1e-8)
        unique = []
        for angle in roots:
            angle = float(np.clip(angle, self.angles_deg[0], self.angles_deg[-1]))
            if not unique or abs(angle-unique[-1]) > 1e-6: unique.append(angle)
        if not unique: raise ValueError('该间距在标定表中没有有效反解')
        pip = min(unique, key=lambda x: (abs(x-seed), x))
        sensitivity = float(self.curve.derivative()(pip))
        low = abs(sensitivity) < 1e-4  # Engineering warning threshold, not a sensor accuracy claim.
        return dict(pip_delta_deg=pip, dip_delta_deg=ratio*pip, ratio=ratio, gap_mm=gap,
                    candidate_pip_deg=unique, ambiguous=len(unique)>1, low_sensitivity=low,
                    sensitivity_mm_per_deg=sensitivity, method='sampled_pchip',
                    selection='nearest current PIP angle; lower angle breaks ties',
                    warning='近零区间间距变化极小，反解对尺寸扰动敏感' if low else '')


class LiveGapGauge:
    """Measure the original confirmed patches at displayed MuJoCo transforms."""
    def __init__(self, model, calibration):
        self.model = model
        self.patches = []
        for item in calibration.record['patches']:
            self.patches.append((model.body(item['mujoco_body']).id,
                                 np.asarray(item['vertices_body_mm']), np.asarray(item['faces'], int)))

    def world_patches_mm(self, data):
        out = []
        for body, vertices, faces in self.patches:
            world_mm = vertices @ data.xmat[body].reshape(3,3).T + data.xpos[body]*1000
            out.append((world_mm, faces))
        return out

    def measure(self, data):
        (a, fa), (b, fb) = self.world_patches_mm(data)
        return measure_patch_pair(a, fa, b, fb)
