"""Post-IK angle -> finite-face distance outputs for all five fingers.

Distances do not participate in IK, trajectory generation or angle assignment.
New faces are candidate engineering gauges, not confirmed physical stop faces.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator
from pip_gap import LiveGapGauge


class WholeHandDistanceOutputs:
    def __init__(self,root,geometry,initial_u):
        self.root=Path(root); self.g=geometry; self.initial_u=geometry.validate(initial_u).copy()
        manifest=json.loads((self.root/'calibration/whole_hand_distance_manifest.json').read_text(encoding='utf8'))
        for name,digest in manifest.items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('五指距离输出表版本校验失败：'+name)
        bundle=json.loads((self.root/'calibration/whole_hand_distance_maps.json').read_text(encoding='utf8'))
        if bundle['source_sha256']!=self.g.cfg['source_sha256']: raise ValueError('五指距离表URDF版本不一致')
        self.records=bundle['modes'][self.g.mode]; self.curves={}; self.gauges={}
        for finger,joints in self.records.items():
            self.curves[finger]={}; self.gauges[finger]={}
            for joint,record in joints.items():
                angles=np.asarray(record['angles_deg'],float); gaps=np.asarray(record['gaps_mm'],float)
                if (record['units']!={'length':'mm','angle':'deg'} or angles.ndim!=1 or angles.shape!=gaps.shape or
                        len(angles)<4 or np.any(np.diff(angles)<=0) or not np.isfinite(angles).all() or
                        not np.isfinite(gaps).all() or np.any(gaps<0)):
                    raise ValueError('五指距离表单位或节点无效：'+finger+'/'+joint)
                self.curves[finger][joint]=PchipInterpolator(angles,gaps,extrapolate=False)
                self.gauges[finger][joint]=LiveGapGauge(self.g.model,type('Record',(),{'record':record})())
        initial_data=mujoco.MjData(self.g.model); self.g.assign(initial_data,self.initial_u)
        self.actual_initial_distances={f:{j:gauge.measure(initial_data)[0] for j,gauge in joints.items()}
                                      for f,joints in self.gauges.items()}
        self._cache=None

    def angle_deg(self,record,u):
        return float(np.degrees(u[record['coordinate_index']])*record['angle_factor'])

    def forward(self,finger,joint,angle_deg):
        record=self.records[finger][joint]; a=float(angle_deg)
        if not np.isfinite(a) or a<record['angles_deg'][0]-1e-8 or a>record['angles_deg'][-1]+1e-8:
            raise ValueError('角度超出本模式距离输出表，禁止外推')
        return float(self.curves[finger][joint](np.clip(a,record['angles_deg'][0],record['angles_deg'][-1])))

    def output(self,u):
        u=self.g.validate(u); fingers={}
        for finger,joints in self.records.items():
            values={}
            for joint,record in joints.items():
                angle=self.angle_deg(record,u); initial=self.angle_deg(record,self.initial_u)
                d=self.forward(finger,joint,angle); d0=self.forward(finger,joint,initial)
                values[joint]=dict(angle_deg=angle,delta_angle_from_initial_deg=angle-initial,
                    distance_mm=d,initial_distance_mm=d0,delta_distance_mm=d-d0,
                    faces_confirmed_by_user=record['faces_confirmed_by_user'],face_status=record['face_status'])
            spec=self.g.cfg['fingers'][finger]; sl=self.g.slices[finger]
            angular_only={label:float(np.degrees(u[sl.start+i])) for i,label in enumerate(spec['coordinate_labels']) if label not in joints}
            fingers[finger]=dict(joints=values,angular_only_coordinates_deg=angular_only,
                dip_pip_ratio=spec['dip_pip_ratio'],ratio_status=spec['ratio_status'])
        return dict(schema_version=2,simulation_mode=self.g.mode,units=dict(length='mm',angle='deg'),
            role='post_IK_forward_distance_output_only',
            pipeline='target pose -> bounded coupled angle IK -> joint angles -> finite-face distances and delta distances',
            reference='delta_distance_mm = distance_mm - distance at this mode display initial pose',
            fingers=fingers,hardware_enabled=False,tendon_length=False,
            uncertainty='nominal mesh interpolation only; physical manufacturing, assembly and sensing uncertainties unknown')

    def status(self,data,u):
        key=data.qpos.tobytes()
        if self._cache is not None and key==self._cache[0]: return self._cache[1]
        output=self.output(u); out={}
        for finger,joints in self.records.items():
            out[finger]={}
            for joint,record in joints.items():
                value=output['fingers'][finger]['joints'][joint]
                d,a,b=self.gauges[finger][joint].measure(data)
                out[finger][joint]=dict(value,actual_gap_mm=d,table_gap_mm=value['distance_mm'],
                    actual_initial_distance_mm=self.actual_initial_distances[finger][joint],
                    actual_delta_distance_mm=d-self.actual_initial_distances[finger][joint],
                    interpolation_residual_mm=d-value['distance_mm'],point_a_mm=a,point_b_mm=b)
        self._cache=(key,out); return out
