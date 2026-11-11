"""Whole-hand bounded FK/IK. SI internally; public positions mm, angles rad.

Explicit kinematic reduction, not cable mechanics. Each solver uses private data.
"""
import hashlib
import csv
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pip_gap import GapCalibration, LiveGapGauge
from index_distance import IndexDistanceMapping
from whole_hand_distances import WholeHandDistanceOutputs

ROOT = Path(__file__).resolve().parent
FINGERS = ('thumb','index','middle','ring','pinky')
MODES = {'strict':'严格限位（原URDF）','simulation_straight':'仅仿真伸直（禁止硬件）'}


def vector(value, n, name):
    v = np.asarray(value,float)
    if v.shape != (n,) or not np.isfinite(v).all(): raise ValueError(f'{name}需要 {n} 个有限数值')
    return v


def serializable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


class HandGeometry:
    def __init__(self,mode='strict'):
        if mode not in MODES: raise ValueError('未知仿真模式')
        self.mode=mode; self.simulation_only=mode=='simulation_straight'
        self.cfg = json.loads((ROOT/'hand_config.json').read_text(encoding='utf8'))
        assets = ROOT/'assets'
        for path,digest in json.loads((assets/'manifest.json').read_text(encoding='utf8')).items():
            if hashlib.sha256((assets/path).read_bytes()).hexdigest()!=digest:
                raise ValueError('整手模型校验失败：'+path)
        xml=ET.parse(assets/'hand.xml').getroot()
        xml.find('compiler').set('meshdir','')
        files={p.name:p.read_bytes() for p in (assets/'meshes').glob('*.stl')}
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'),files)
        if (self.model.nq,self.model.nu,self.model.ntendon,self.model.neq)!=(35,0,0,0):
            raise ValueError('整手应有35个无驱动原始转轴')
        self.data = mujoco.MjData(self.model)
        self.source_native_ranges=self.model.jnt_range.copy()
        self.origin = vector(self.cfg['palm_frame_origin_m'],3,'掌心原点')
        self.reference = np.zeros(self.model.nq)
        for original,name in self.cfg['joint_names'].items():
            j = self.model.joint(name); address = j.qposadr[0]
            self.reference[address] = j.range[0] if original.endswith('-OR') else 0.
        self.source_reference=self.reference.copy(); self.simulation_profile=None
        if self.simulation_only:
            profile=json.loads((ROOT/'simulation_straight_profile.json').read_text(encoding='utf8'))
            if (profile['source_sha256']!=self.cfg['source_sha256'] or profile['units']!='deg' or
                    profile['mode']!=mode or profile['hardware_enabled'] is not False):
                raise ValueError('仅仿真伸直配置的版本、单位或用途无效')
            self.simulation_profile=profile
            self.reference[:]=np.radians(profile['default_native_reference_deg'])
            for name,angle_deg in profile['native_reference_overrides_deg'].items():
                self.reference[self.model.joint(name).qposadr[0]]=np.radians(angle_deg)
            vector(self.reference,self.model.nq,'仅仿真原生零位 / rad')
        self.slices = {}; self.labels = []; columns = []; self.lower=[]; self.upper=[]
        for finger in FINGERS:
            spec = self.cfg['fingers'][finger]; start=len(columns)
            ratio = spec['dip_pip_ratio']
            if ratio is not None and (not np.isfinite(ratio) or ratio<=0): raise ValueError('耦合比例无效')
            inverse_names = {n:o for o,n in self.cfg['joint_names'].items()}
            for label,entries in zip(spec['coordinate_labels'],spec['columns']):
                col=np.zeros(self.model.nq); lo,hi=-np.inf,np.inf
                for joint_name,factor in entries:
                    if 'DIP' in inverse_names[joint_name]: factor *= ratio/.8
                    j=self.model.joint(joint_name); address=j.qposadr[0]; col[address]=factor
                    bounds=(j.range-self.reference[address])/factor
                    lo=max(lo,float(min(bounds))); hi=min(hi,float(max(bounds)))
                if self.simulation_only and finger+':'+label in profile['independent_ranges_deg']:
                    lo,hi=np.radians(vector(profile['independent_ranges_deg'][finger+':'+label],2,'仅仿真角度范围'))
                if not np.isfinite([lo,hi]).all() or hi<=lo: raise ValueError('独立坐标限位交集为空')
                columns.append(col); self.lower.append(lo); self.upper.append(hi)
                self.labels.append(finger+':'+label)
            self.slices[finger]=slice(start,len(columns))
        self.C=np.stack(columns,axis=1); self.lower=np.array(self.lower); self.upper=np.array(self.upper)
        if self.simulation_only:
            # Expand only this fresh in-memory model. Source XML/URDF and strict
            # model instances remain byte-for-byte / numerically unchanged.
            native_lo=self.reference+np.minimum(self.C*self.lower,self.C*self.upper).sum(axis=1)
            native_hi=self.reference+np.maximum(self.C*self.lower,self.C*self.upper).sum(axis=1)
            for j in range(self.model.njnt):
                address=self.model.jnt_qposadr[j]
                self.model.jnt_range[j,0]=min(self.model.jnt_range[j,0],native_lo[address])
                self.model.jnt_range[j,1]=max(self.model.jnt_range[j,1],native_hi[address])
        self.zero=np.clip(np.zeros(len(columns)),self.lower,self.upper)
        self.sites={f:self.model.site(self.cfg['fingers'][f]['tip_site']).id for f in FINGERS}
        self.sample_cache={}
        self.assign(self.data,self.zero)

    def validate(self,u):
        u=vector(u,len(self.zero),'整手独立角度')
        if np.any(u<self.lower-1e-9) or np.any(u>self.upper+1e-9):
            raise ValueError('角度超出'+('仅仿真演示范围' if self.simulation_only else ' URDF 运动学范围'))
        return np.clip(u,self.lower,self.upper)

    def assign(self,data,u):
        u=self.validate(u); data.qpos[:]=self.reference+self.C@u
        mujoco.mj_kinematics(self.model,data); mujoco.mj_comPos(self.model,data)

    def pose(self,data,finger):
        site=self.sites[finger]; rot=Rotation.from_matrix(data.site_xmat[site].reshape(3,3))
        q=rot.as_quat()[[3,0,1,2]]
        if q[0]<0: q=-q
        return dict(position_mm=(data.site_xpos[site]-self.origin)*1000,
                    quaternion_wxyz=q,rotation=rot.as_matrix())

    def forward(self,u,finger):
        self.assign(self.data,u); return self.pose(self.data,finger)

    def position_jacobian(self,finger):
        jac=np.zeros((3,self.model.nv)); mujoco.mj_jacSite(self.model,self.data,jac,None,self.sites[finger])
        return 1000*jac@self.C[:,self.slices[finger]]

    def samples(self,finger):
        if finger not in self.sample_cache:
            sl=self.slices[finger]; n=sl.stop-sl.start
            grid=7 if n==3 else 3
            values=np.array(list(itertools.product(*[np.linspace(a,b,grid) for a,b in zip(self.lower[sl],self.upper[sl])])))
            if n==5:
                rng=np.random.default_rng(407)
                values=np.vstack([values,rng.uniform(self.lower[sl],self.upper[sl],(64,n))])
            records=[]
            for v in values:
                u=self.zero.copy(); u[sl]=v; records.append(self.forward(u,finger))
            self.sample_cache[finger]=(values,np.array([r['position_mm'] for r in records]),
                                        np.array([r['rotation'] for r in records]))
        return self.sample_cache[finger]

    def solve(self,finger,target,base_u,allow_projection=True,locked_coordinates=None):
        started=time.perf_counter(); sl=self.slices[finger]; base=self.validate(base_u).copy()
        labels=self.labels[sl]; locked=dict(locked_coordinates or {})
        if any(name not in labels for name in locked): raise ValueError('锁定轴不属于当前手指')
        fixed=base[sl].copy()
        for name,value in locked.items():
            i=labels.index(name)
            if not np.isfinite(value) or value<self.lower[sl][i]-1e-8 or value>self.upper[sl][i]+1e-8:
                raise ValueError('锁定角度超限或无效：'+name)
            # Accommodate decimal-degree UI rounding without widening bounds.
            fixed[i]=np.clip(value,self.lower[sl][i],self.upper[sl][i]); locked[name]=fixed[i]
        active=np.array([i for i,name in enumerate(labels) if name not in locked],dtype=int)
        lo,hi=self.lower[sl][active],self.upper[sl][active]
        def expand(v):
            full=fixed.copy(); full[active]=v; return full
        p=vector(target['position_mm'],3,'目标位置 / mm'); wanted=None; quat=None
        if target.get('quaternion_wxyz') is not None:
            quat=vector(target['quaternion_wxyz'],4,'四元数 wxyz'); norm=np.linalg.norm(quat)
            if norm<1e-9: raise ValueError('四元数不能全零')
            quat=quat/norm; wanted=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        weight=self.cfg['orientation_weight_mm_per_rad']
        def fk(v):
            u=base.copy(); u[sl]=expand(v); return self.forward(u,finger)
        def residual(v):
            result=fk(v); delta=result['position_mm']-p
            if wanted is not None:
                delta=np.r_[delta,weight*Rotation.from_matrix(wanted@result['rotation'].T).as_rotvec()]
            return delta
        def jac(v): fk(v); return self.position_jacobian(finger)[:,active]
        if locked:
            # Rebuild the workspace at the requested root angles. Samples from
            # the unconstrained workspace are invalid for a locked-axis IK.
            grid=13 if len(active)<=2 else 5
            samples=np.array(list(itertools.product(*[np.linspace(a,b,grid) for a,b in zip(lo,hi)])))
            records=[fk(v) for v in samples]
            points=np.array([r['position_mm'] for r in records]); rots=np.array([r['rotation'] for r in records])
        else:
            samples,points,rots=self.samples(finger)
        scores=np.sum((points-p)**2,axis=1)
        if wanted is not None:
            angles=np.arccos(np.clip((np.einsum('ij,nij->n',wanted,rots)-1)/2,-1,1))
            scores+=(weight*angles)**2
        starts=[base[sl][active],(lo+hi)/2]+list(samples[np.argsort(scores)[:8]])
        fits=[]
        if not len(active): fits.append((float(np.linalg.norm(residual(np.array([])))),0.,np.array([])))
        for start in starts if len(active) else []:
            fit=least_squares(residual,np.clip(start,lo+1e-10,hi-1e-10),
                jac=jac if wanted is None else '3-point',bounds=(lo,hi),
                max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            fits.append((float(np.linalg.norm(residual(fit.x))),float(np.linalg.norm(fit.x-base[sl][active])),fit.x.copy()))
        # Keep the best workspace sample as a fallback for projected targets.
        if len(active):
            seed=samples[np.argmin(scores)]
            fits.append((float(np.linalg.norm(residual(seed))),float(np.linalg.norm(seed-base[sl][active])),seed.copy()))
        bestscore=min(x[0] for x in fits)
        score,_,v=min((x for x in fits if x[0]<=bestscore+1e-7),key=lambda x:x[1])
        f=fk(v); error=float(np.linalg.norm(f['position_mm']-p))
        orientation=0. if wanted is None else float(np.degrees(Rotation.from_matrix(wanted@f['rotation'].T).magnitude()))
        exact=error<=self.cfg['position_tolerance_mm'] and orientation<=self.cfg['orientation_tolerance_deg']
        return dict(finger=finger,success=bool(exact or allow_projection),exact_reachable=bool(exact),
            projected=bool(not exact and allow_projection),coordinates_rad=expand(v),
            requested_position_mm=p,requested_quaternion_wxyz=quat,
            execution_position_mm=f['position_mm'],execution_quaternion_wxyz=f['quaternion_wxyz'],
            position_error_mm=error,orientation_error_deg=orientation,weighted_pose_error_mm=score,
            solve_ms=(time.perf_counter()-started)*1000,
            solver='bounded_locked_axis_IK' if locked else 'bounded_coupled_angle_IK',
            optimization_variables=[labels[i] for i in active],locked_coordinates_rad=locked,
            residual_path='independent joint angles -> coupled native axes -> FK -> target pose error')


class HandRuntime:
    def __init__(self,mode='strict'):
        self.g=HandGeometry(mode); self.model=self.g.model; self.cfg=self.g.cfg
        self.mode=mode; self.simulation_only=self.g.simulation_only
        self.initial_profile=(self.g.simulation_profile if self.simulation_only else
                              json.loads((ROOT/'initial_pose.json').read_text(encoding='utf8')))
        profile=self.initial_profile
        if (profile['source_sha256']!=self.cfg['source_sha256'] or profile['units']!='deg' or
                profile['coordinate_labels']!=self.g.labels):
            raise ValueError('初始姿态与当前模型版本、单位或坐标顺序不一致')
        self.initial_u=self.g.validate(np.radians(profile['initial_coordinates_deg'])).copy()
        self.data=mujoco.MjData(self.model); self.u=self.initial_u.copy(); self.g.assign(self.data,self.u)
        self.gap_calibration=GapCalibration.load(ROOT,self.cfg)
        if self.cfg['source_sha256'] not in self.gap_calibration.record['source_hashes'].values():
            raise ValueError('食指间距表与整手原始 URDF 版本不一致，需要重建验证')
        self.gauge=LiveGapGauge(self.model,self.gap_calibration)
        self.distance_mapping=IndexDistanceMapping(ROOT,self.g)
        self.distance_outputs=WholeHandDistanceOutputs(ROOT,self.g,self.initial_u)
        self.status='idle'; self.plan=None; self.last_result=None

    def index_gap_table(self,u):
        # The calibrated strict geometry has a different native reference.
        # Never extrapolate or pretend that table is valid for the display copy.
        if self.simulation_only: return None
        p=self.g.slices['index'].start+2
        return self.gap_calibration.gap_from_angle(np.degrees(u[p]))

    def prepare(self,targets,base_u=None,allow_projection=True):
        if not isinstance(targets,dict) or not targets or any(f not in FINGERS for f in targets):
            raise ValueError('至少指定一个有效手指目标')
        base=self.g.validate(self.u.copy() if base_u is None else base_u).copy(); goal=base.copy(); records={}
        for finger,target in targets.items():
            record=self.g.solve(finger,target,base,allow_projection)
            goal[self.g.slices[finger]]=record['coordinates_rad']; records[finger]=record
        result=self.result_from_angles(goal)
        result.update(mode='whole_hand_IK',success=all(x['success'] for x in records.values()),
            fingers=records,projection_allowed=bool(allow_projection),
            solve_pipeline='target pose -> coupled angle IK -> joint angles -> forward face-distance outputs')
        return result

    def prepare_thumb_axes(self,target,root_angles_deg,base_u=None,allow_projection=True):
        """Lock CMC-OPP/CMC-PS/MCP-AA inside IK; solve MCP/IP only.

        Angles use model coordinates, not deltas from the display initial pose.
        The other four fingers remain at base_u throughout the preview.
        """
        roots=vector(root_angles_deg,3,'拇指三轴模型角度 / deg')
        base=self.g.validate(self.u.copy() if base_u is None else base_u).copy()
        sl=self.g.slices['thumb']; names=self.g.labels[sl][:3]
        locks=dict(zip(names,np.radians(roots)))
        record=self.g.solve('thumb',target,base,allow_projection,locked_coordinates=locks)
        goal=base.copy(); goal[sl]=record['coordinates_rad']
        result=self.result_from_angles(goal)
        result.update(mode='thumb_independent_axes_IK',success=record['success'],
            fingers={'thumb':record},projection_allowed=bool(allow_projection),
            thumb_root_angles_deg=roots.copy(),
            solve_pipeline='specified three root axes + target position -> bounded MCP/IP IK -> coupled native angles -> face-distance outputs')
        return result

    def result_from_angles(self,u):
        goal=self.g.validate(u).copy()
        return dict(success=True,mode='whole_hand_FK',goal_rad=goal,goal_deg=np.degrees(goal),
            coordinate_labels=self.g.labels,initial_deg=np.degrees(self.initial_u),
            delta_from_initial_deg=np.degrees(goal-self.initial_u),
            index_target_gap_mm=self.index_gap_table(goal),
            face_distance_outputs=self.distance_outputs.output(goal),
            result_schema_version=2,execution_space='coupled_joint_angles',
            simulation_mode=self.mode,simulation_only=self.simulation_only,
            original_limits_enforced=not self.simulation_only,
            reference_native_angles_rad=self.g.reference.copy(),
            scope=self.cfg['scope'],frame=self.cfg['frame_definition'],collision_validated=False,
            cable_computation=False,hardware_enabled=False)

    def prepare_index_gap(self,gap_mm,base_u=None):
        if self.simulation_only: raise ValueError('原PIP单项反解表仅适用严格模式；仿真副本可使用独立的双距离工具')
        u=self.u.copy() if base_u is None else self.g.validate(base_u).copy()
        p=self.g.slices['index'].start+2
        inverse=self.gap_calibration.angles_from_gap(gap_mm,np.degrees(u[p]),self.cfg['fingers']['index']['dip_pip_ratio'])
        u[p]=np.radians(inverse['pip_delta_deg']); result=self.result_from_angles(u)
        result.update(mode='index_gap_in_whole_hand',gap_inverse=inverse)
        return result

    def prepare_index_distances(self,mcp_mm,pip_mm,base_u=None,branches=None,aa_deg=None):
        base=self.u.copy() if base_u is None else self.g.validate(base_u).copy()
        u,inverse=self.distance_mapping.decode(mcp_mm,pip_mm,base,branches,aa_deg)
        result=self.result_from_angles(u)
        result.update(mode='index_MCP_PIP_distance_input',distance_inverse=inverse)
        return result

    def start(self,result):
        if not result.get('success'): raise ValueError('存在不可达目标且未启用近似投影，未执行整手动作')
        if result.get('simulation_mode','strict')!=self.mode:
            raise ValueError('结果属于不同仿真模式；禁止跨模式执行旧目标')
        goal=self.g.validate(result['goal_rad']).copy()
        if result.get('result_schema_version')!=2 or result.get('execution_space')!='coupled_joint_angles':
            raise ValueError('旧距离控制结果不能按新角度路径执行，请重新求解')
        # Distance is a post-IK readout. Never decode it to modify an IK goal.
        # Recompute the readout from the authoritative angles before execution.
        result['face_distance_outputs']=self.distance_outputs.output(goal)
        duration=max(.4,1.875*np.max(abs(goal-self.u))/np.radians(self.cfg['preview_velocity_deg_s']))
        self.plan=dict(start=self.u.copy(),goal=goal,duration=float(duration),elapsed=0.,
            execution_space='coupled_joint_angles',distance_role='post_angle_forward_output_only')
        self.last_result=result; self.status='tracking'

    def demonstration(self,fingers=None):
        """Return inspectable sequential goals inside the active mode bounds.

        Thumb root 3 axes held, MCP +25 deg and IP +10 deg. Four-finger MCP
        +25 deg then PIP +30 deg (DIP +24 deg). Deltas use the display initial.
        """
        fingers=list(FINGERS if fingers is None else fingers)
        if not fingers or len(set(fingers))!=len(fingers) or any(f not in FINGERS for f in fingers):
            raise ValueError('演示手指列表无效')
        if self.simulation_only: return self.straight_fist_demonstration(fingers)
        start=self.initial_u.copy(); root=start.copy(); distal=start.copy()
        for finger in fingers:
            sl=self.g.slices[finger]
            root[sl.start+(3 if finger=='thumb' else 1)]+=np.radians(25.)
        distal=root.copy()
        for finger in fingers:
            sl=self.g.slices[finger]
            distal[sl.start+(4 if finger=='thumb' else 2)]+=np.radians(10. if finger=='thumb' else 30.)
        stages=[]
        for title,u,hold in (('1/5 回到统一初始姿态',start,1.2),
                             ('2/5 MCP根部屈曲；远端相对角保持',root,1.2),
                             ('3/5 PIP/DIP耦合屈曲；拇指IP屈曲',distal,1.5),
                             ('4/5 远端复位；MCP保持',root,1.2),
                             ('5/5 MCP复位，回到初始姿态',start,1.2)):
            result=self.result_from_angles(u)
            result.update(mode='staged_flexion_demo',stage=title,demo_fingers=fingers)
            stages.append(dict(title=title,result=result,hold_seconds=hold))
        return stages

    def straight_fist_demonstration(self,fingers):
        """Open -> graded nominal fist -> open, display-only joint kinematics."""
        if not self.simulation_only: raise ValueError('伸直握拳演示只允许在仅仿真模式使用')
        start=self.initial_u.copy()
        fist=self.g.validate(np.radians(self.initial_profile['fist_coordinates_deg']))
        goal=start.copy()
        for finger in fingers: goal[self.g.slices[finger]]=fist[self.g.slices[finger]]
        stages=[]
        for title,fraction,hold in (
            ('1/7 五指伸直初始姿态（仅仿真）',0.,1.5),
            ('2/7 约三分之一屈曲',1/3,1.),
            ('3/7 约三分之二屈曲',2/3,1.),
            ('4/7 完整握拳展示姿态',1.,2.),
            ('5/7 缓慢松拳',2/3,.7),
            ('6/7 继续伸展',1/3,.7),
            ('7/7 回到五指伸直',0.,1.5)):
            result=self.result_from_angles(start+fraction*(goal-start))
            result.update(mode='simulation_straight_fist_demo',stage=title,demo_fingers=fingers,
                          flexion_fraction=fraction)
            stages.append(dict(title=title,result=result,hold_seconds=hold))
        return stages

    def advance(self,dt):
        if not np.isfinite(dt) or dt<0: raise ValueError('时间增量无效')
        if self.status!='tracking': return
        p=self.plan; p['elapsed']=min(p['duration'],p['elapsed']+dt); t=p['elapsed']/p['duration']
        proposed=p['start']+(10*t**3-15*t**4+6*t**5)*(p['goal']-p['start'])
        if hasattr(self,'execution_guard'):
            self.u,self.path_report=self.execution_guard.sweep(self.u,proposed)
            if self.path_report['blocked']:
                self.status='holding'
                self.last_result['path_check']=self.path_report
        else:self.u=proposed
        self.g.assign(self.data,self.u)
        if t>=1: self.status='holding'

    def snapshot(self):
        all_distances=self.distance_outputs.status(self.data,self.u)
        distances=all_distances['index']; gap=distances['PIP']['actual_gap_mm']
        return dict(coordinates_deg=np.degrees(self.u),initial_deg=np.degrees(self.initial_u),
            delta_from_initial_deg=np.degrees(self.u-self.initial_u),native_angles_rad=self.data.qpos.copy(),
            poses={f:self.g.pose(self.data,f) for f in FINGERS},index_gap_mm=gap,
            index_distances=distances,face_distances=all_distances,index_distance_mapping_enabled=True,
            distance_role='post_angle_forward_output_only',execution_space='coupled_joint_angles',
            index_gap_table_mm=self.index_gap_table(self.u),status=self.status,
            simulation_mode=self.mode,simulation_only=self.simulation_only,
            original_limits_enforced=not self.simulation_only,hardware_enabled=False)

    def save(self):
        path=ROOT/('results/simulation_last_result.json' if self.simulation_only else 'results/last_result.json')
        path.write_text(json.dumps(dict(result=self.last_result,preview=self.snapshot()),ensure_ascii=False,
                        indent=2,default=serializable),encoding='utf8')
        csv_path=path.with_name(path.stem+'_distances.csv')
        fields=['simulation_mode','finger','joint','angle_deg','delta_angle_from_initial_deg',
                'distance_mm','initial_distance_mm','delta_distance_mm','actual_gap_mm',
                'actual_initial_distance_mm','actual_delta_distance_mm','faces_confirmed_by_user']
        with csv_path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for finger,joints in self.snapshot()['face_distances'].items():
                for joint,value in joints.items():
                    row={name:value[name] for name in fields[3:]}
                    writer.writerow(dict(row,simulation_mode=self.mode,finger=finger,joint=joint))
        return path
