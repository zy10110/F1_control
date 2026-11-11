"""Finite-patch MCP/PIP distance coordinates, mm/deg at public boundaries.

Legacy manual distance inverse and branch-aware helpers; not the main IK. These are nominal
geometry measurements, not string lengths or identified physical sensors.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from pip_gap import LiveGapGauge


class JointDistanceMap:
    def __init__(self,record):
        self.record=record; self.joint=record['joint']
        if record['units']!={'length':'mm','angle':'deg'}: raise ValueError('距离表单位错误')
        self.angles=np.asarray(record['angles_deg'],float); self.gaps=np.asarray(record['gaps_mm'],float)
        if (self.angles.ndim!=1 or self.gaps.shape!=self.angles.shape or len(self.angles)<4 or
                not np.isfinite(self.angles).all() or not np.isfinite(self.gaps).all() or
                np.any(np.diff(self.angles)<=0) or np.any(self.gaps<0)):
            raise ValueError('距离标定表无效')
        self.curve=PchipInterpolator(self.angles,self.gaps,extrapolate=False)
        # A PCHIP segment cannot overshoot monotone data. Split at actual data
        # turning nodes, not every derivative zero at a flat-but-monotone node.
        signs=np.sign(np.diff(self.gaps)); signs[abs(np.diff(self.gaps))<1e-13]=0
        nonzero=np.flatnonzero(signs); turns=[]
        for a,b in zip(nonzero[:-1],nonzero[1:]):
            if signs[a]!=signs[b]: turns.append(float(self.angles[a+1]))
        self.cuts=np.unique(np.r_[self.angles[0],turns,self.angles[-1]])
        self.branches=[]
        for a,b in zip(self.cuts[:-1],self.cuts[1:]):
            ga,gb=float(self.curve(a)),float(self.curve(b))
            if abs(ga-gb)>1e-12:
                self.branches.append(dict(id=len(self.branches),angle_lo_deg=a,angle_hi_deg=b,
                    gap_lo_mm=min(ga,gb),gap_hi_mm=max(ga,gb),increasing=gb>ga))
        if not self.branches: raise ValueError('测量面对角度不敏感，不能建立距离映射')

    def forward(self,angle_deg):
        a=float(angle_deg)
        if not np.isfinite(a) or a<self.angles[0]-1e-8 or a>self.angles[-1]+1e-8:
            raise ValueError(self.joint+'角度超出本模式距离表，不允许外推')
        return float(self.curve(np.clip(a,self.angles[0],self.angles[-1])))

    def branch(self,angle_deg):
        for b in self.branches:
            if b['angle_lo_deg']-1e-8<=angle_deg<=b['angle_hi_deg']+1e-8: return b
        raise ValueError(self.joint+'角度没有可用的距离分支')

    def inverse_on_branch(self,gap_mm,branch):
        d=float(gap_mm); b=self.branches[int(branch)] if isinstance(branch,(int,np.integer)) else branch
        if not np.isfinite(d) or d<b['gap_lo_mm']-1e-10 or d>b['gap_hi_mm']+1e-10:
            raise ValueError(self.joint+'距离超出指定分支，禁止外推')
        d=float(np.clip(d,b['gap_lo_mm'],b['gap_hi_mm']))
        a,z=b['angle_lo_deg'],b['angle_hi_deg']; fa=self.forward(a)-d; fz=self.forward(z)-d
        if fa==0: return float(a)
        if fz==0: return float(z)
        return float(brentq(lambda x:float(self.curve(x))-d,a,z,xtol=1e-10))

    def candidates(self,gap_mm):
        d=float(gap_mm); candidates=[]
        for branch in self.branches:
            if branch['gap_lo_mm']-1e-12<=d<=branch['gap_hi_mm']+1e-12:
                a=self.inverse_on_branch(d,branch)
                if not any(abs(x['angle_deg']-a)<1e-6 for x in candidates):
                    candidates.append(dict(angle_deg=a,branch_id=branch['id']))
        if not candidates: raise ValueError(self.joint+'距离超出本模式标定范围，不允许外推')
        return candidates

    def inverse(self,gap_mm,current_deg,branch_id=None):
        candidates=self.candidates(gap_mm)
        if branch_id is not None:
            angle=self.inverse_on_branch(gap_mm,int(branch_id)); branch=int(branch_id)
        else:
            selected=min(candidates,key=lambda x:(abs(x['angle_deg']-current_deg),x['angle_deg']))
            angle=selected['angle_deg']; branch=selected['branch_id']
        sensitivity=float(self.curve.derivative()(angle))
        return dict(angle_deg=angle,branch_id=branch,gap_mm=float(gap_mm),candidates=candidates,
                    ambiguous=len(candidates)>1,sensitivity_mm_per_deg=sensitivity,
                    low_sensitivity=abs(sensitivity)<1e-4)

    def path(self,start_deg,goal_deg):
        if abs(start_deg-goal_deg)<1e-9: return dict(constant_deg=float(goal_deg),segments=[])
        lo,hi=sorted((start_deg,goal_deg)); points=[float(start_deg)]
        turns=[float(x) for x in self.cuts if lo+1e-8<x<hi-1e-8]
        points+=sorted(turns,reverse=goal_deg<start_deg); points.append(float(goal_deg))
        total=abs(goal_deg-start_deg); segments=[]; cursor=0.
        for a,b in zip(points[:-1],points[1:]):
            end=cursor+abs(b-a)/total; branch=self.branch((a+b)/2)
            segments.append(dict(start_fraction=cursor,end_fraction=end,start_deg=a,goal_deg=b,
                start_gap_mm=self.forward(a),goal_gap_mm=self.forward(b),branch_id=branch['id']))
            cursor=end
        return dict(constant_deg=None,segments=segments)

    def sample_path(self,path,fraction):
        if not path['segments']: return path['constant_deg'],self.forward(path['constant_deg'])
        t=float(np.clip(fraction,0,1)); seg=path['segments'][-1]
        for candidate in path['segments']:
            if t<=candidate['end_fraction']+1e-12: seg=candidate; break
        local=np.clip((t-seg['start_fraction'])/(seg['end_fraction']-seg['start_fraction']),0,1)
        smooth=10*local**3-15*local**4+6*local**5
        d=seg['start_gap_mm']+smooth*(seg['goal_gap_mm']-seg['start_gap_mm'])
        return self.inverse_on_branch(d,seg['branch_id']),float(d)


class IndexDistanceMapping:
    def __init__(self,root,geometry):
        self.root=Path(root); self.g=geometry
        manifest=json.loads((self.root/'calibration/index_distance_manifest.json').read_text(encoding='utf8'))
        for path,digest in manifest.items():
            p=(self.root/path).resolve()
            if not p.is_relative_to(self.root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
                raise ValueError('食指 MCP/PIP 距离映射版本校验失败：'+path)
        records=json.loads((self.root/'calibration/index_distance_maps.json').read_text(encoding='utf8'))
        if records['source_sha256']!=self.g.cfg['source_sha256']: raise ValueError('距离映射源URDF版本不同')
        self.maps={j:JointDistanceMap(records['modes'][self.g.mode][j]) for j in ('MCP','PIP')}
        self.gauges={j:LiveGapGauge(self.g.model,m) for j,m in self.maps.items()}
        self.sl=self.g.slices['index']; self.indices={'MCP':self.sl.start+1,'PIP':self.sl.start+2}
        self._cache=None

    def command(self,u):
        joints={}
        for name,idx in self.indices.items():
            angle=float(np.degrees(u[idx])); mapping=self.maps[name]
            joints[name]=dict(gap_mm=mapping.forward(angle),branch_id=mapping.branch(angle)['id'],
                             angle_deg=angle,faces_confirmed_by_user=mapping.record['faces_confirmed_by_user'])
        return dict(units=dict(length='mm',angle='deg'),simulation_mode=self.g.mode,
                    mcp_aa_deg=float(np.degrees(u[self.sl.start])),joints=joints,
                    distance_only_unique=all(len(self.maps[j].candidates(v['gap_mm']))==1 for j,v in joints.items()),
                    hardware_enabled=False,scope='nominal finite-patch distances, NOT tendon or actuator strokes')

    def decode(self,mcp_mm,pip_mm,base_u,branches=None,aa_deg=None):
        u=self.g.validate(base_u).copy(); records={}
        for name,d in (('MCP',mcp_mm),('PIP',pip_mm)):
            idx=self.indices[name]; hint=None if branches is None else branches.get(name)
            records[name]=self.maps[name].inverse(d,float(np.degrees(u[idx])),hint)
            u[idx]=np.radians(records[name]['angle_deg'])
        if aa_deg is not None: u[self.sl.start]=np.radians(float(aa_deg))
        return self.g.validate(u),records

    def status(self,data,u):
        key=data.qpos.tobytes()
        if self._cache is not None and key==self._cache[0]: return self._cache[1]
        out={}
        for name,mapping in self.maps.items():
            idx=self.indices[name]; angle=float(np.degrees(u[idx])); gap,a,b=self.gauges[name].measure(data)
            table=mapping.forward(angle); decoded=mapping.inverse(table,angle,mapping.branch(angle)['id'])
            out[name]=dict(actual_gap_mm=gap,table_gap_mm=table,interpolation_residual_mm=gap-table,
                point_a_mm=a,point_b_mm=b,angle_deg=angle,branch_id=decoded['branch_id'],
                ambiguous=decoded['ambiguous'],low_sensitivity=decoded['low_sensitivity'],
                sensitivity_mm_per_deg=decoded['sensitivity_mm_per_deg'],
                faces_confirmed_by_user=mapping.record['faces_confirmed_by_user'])
        self._cache=(key,out); return out

    def solve(self,target,base_u,allow_projection=True):
        """Kept only to reject accidental use of the superseded IK path."""
        raise ValueError('距离空间IK已停用；请调用HandRuntime.prepare，先求角度再输出距离')

    def trajectory(self,start,goal):
        return {name:m.path(float(np.degrees(start[self.indices[name]])),float(np.degrees(goal[self.indices[name]])))
                for name,m in self.maps.items()}

    def advance_trajectory(self,path,fraction,u):
        out=u.copy(); commands={}
        for name,mapping in self.maps.items():
            angle,d=mapping.sample_path(path[name],fraction); out[self.indices[name]]=np.radians(angle)
            commands[name]=d
        return out,commands
