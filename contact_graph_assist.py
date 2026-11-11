"""Continuous multi-pair pad approach inside unchanged F1 kinematics."""
import itertools
import json
import mujoco
import numpy as np
from hand_kinematics import ROOT,FINGERS
from constrained_gesture_assist import ConstrainedGestureAssist
from g7_proportional_mapping import source_points,BASE
from palm_calibration import PadGeometry
from thumb_collision import ThumbCollisionGuard


class ContactTargets(tuple):
    def __new__(cls,shape,edges,shafts=()):
        obj=super().__new__(cls,shape);obj.edges=edges;obj.shafts=shafts;return obj


class ContactGraphAssist(ConstrainedGestureAssist):
    kind='contact_graph'

    def __init__(self,g,config,reference,hand='right'):
        super().__init__(g,config,reference,hand)
        self.iterations=6
        self._init_contact_geometry(g,config)

    def _init_contact_geometry(self,g,config):
        self.pad=PadGeometry(g,json.loads((ROOT/'calibration/g7_f1_pad_candidates.json').read_text(encoding='utf8')))
        self.pad_data=mujoco.MjData(g.model)
        self.geoms={f:int(np.flatnonzero(g.model.geom_bodyid==self.pad.ids[f])[0]) for f in FINGERS}
        self.clearance=ThumbCollisionGuard(g)
        bodies={'thumb':[3,5,7],**{f:[10+7*k,12+7*k,14+7*k] for k,f in enumerate(FINGERS[1:])}}
        self.bone_geoms={f:[int(np.flatnonzero(g.model.geom_bodyid==g.model.body(f'f1_link_{b}').id)[0]) for b in bs] for f,bs in bodies.items()}
        self.contact_weight=float(config.get('contact_graph_weight',3.))
        if not np.isfinite(self.contact_weight) or self.contact_weight<=0:raise ValueError('接触权重无效')

    def targets(self,frame):
        shape=super().targets(frame);p=source_points(frame,self.hand)
        lengths={f:np.linalg.norm(np.diff(p[i:i+4],axis=0),axis=1).sum() for f,i in BASE.items()}
        edges=[]
        for a,b in itertools.combinations(FINGERS,2):
            gap=float(np.linalg.norm(p[BASE[a]+3]-p[BASE[b]+3]))
            length=min(lengths[a],lengths[b]);thumb=a=='thumb'
            full=(.22 if thumb else .08)*length;start=(.50 if thumb else .20)*length
            t=float(np.clip((start-gap)/(start-full),0,1));strength=t*t*(3-2*t)
            scale=np.sqrt(self.lengths[a]*self.lengths[b]/(lengths[a]*lengths[b]))
            edges.append(dict(a=a,b=b,weight=strength,source_gap_mm=gap,
                full_mm=full,start_mm=start,target_gap_mm=.8+max(0.,gap-full)*scale))
        shafts=[];samples=np.linspace(0.,1.,11)
        # Also retain thumb-shaft to finger contacts (e.g. a closed hand with
        # the thumb alongside the index). Tip-only graphs cannot represent it.
        for f in FINGERS[1:]:
            group=[];root=BASE[f]
            for a,b in itertools.product(range(3),repeat=2):
                if (a,b) in ((0,0),(2,2)):continue
                va=p[2+a]-p[1+a];vb=p[root+b+1]-p[root+b]
                aa=p[1+a]+samples[:,None]*va;bb=p[root+b]+samples[:,None]*vb
                gap=float(np.min(np.linalg.norm(aa[:,None]-bb[None,:],axis=-1)))
                full=.14*(np.linalg.norm(va)+np.linalg.norm(vb));start=full+10.
                t=float(np.clip((start-gap)/(start-full),0.,1.));strength=t*t*(3.-2.*t)
                if strength<1e-6:continue
                scale=np.sqrt(self.lengths['thumb']*self.lengths[f]/(lengths['thumb']*lengths[f]))
                group.append(dict(finger=f,thumb_segment=a,finger_segment=b,weight=strength,
                    source_gap_mm=gap,target_gap_mm=.8+max(0.,gap-full)*scale))
            total=sum(e['weight'] for e in group)
            tip_weight=next(e['weight'] for e in edges if e['a']=='thumb' and e['b']==f)
            for e in group:e['weight']*=((1.-tip_weight)/max(1.,total))
            shafts.extend(group)
        return ContactTargets(shape,edges,shafts)

    def pad_jac(self,f):
        pose=self.pad.pose(self.pad_data,f);jp=np.zeros((3,self.g.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(self.g.model,self.pad_data,jp,jr,pose['position_world_m'],self.pad.ids[f])
        return pose,jp@self.g.C*1000.,np.cross((jr@self.g.C).T,pose['normal_world']).T

    def surface(self,a,b,jacobian=False):
        return self.geom_surface(self.geoms[a],self.geoms[b],jacobian)

    def geom_surface(self,ga,gb,jacobian=False):
        endpoints=np.zeros(6)
        distance=float(mujoco.mj_geomDistance(self.g.model,self.pad_data,ga,gb,.5,endpoints))
        if not jacobian:return distance*1000.
        if abs(distance)<1e-9:
            direction=self.pad_data.geom_xpos[gb]-self.pad_data.geom_xpos[ga]
            direction/=max(np.linalg.norm(direction),1e-12)
        else:direction=(endpoints[3:]-endpoints[:3])/distance
        j=[]
        for geom,p in ((ga,endpoints[:3]),(gb,endpoints[3:])):
            jp=np.zeros((3,self.g.model.nv))
            mujoco.mj_jac(self.g.model,self.pad_data,jp,None,p,int(self.g.model.geom_bodyid[geom]));j.append(jp@self.g.C*1000.)
        return distance*1000.,direction@(j[1]-j[0])

    def clearance_rows(self,u):
        # Keep the original thumb hull constraints visible to the optimizer,
        # instead of discovering all conflicts only in the final path guard.
        d=self.pad_data;m=self.g.model;res=[];jac=[]
        for a,b in self.clearance.pairs:
            bound=np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])-m.geom_rbound[a]-m.geom_rbound[b]
            if bound>.0015:continue
            endpoints=np.zeros(6);distance=float(mujoco.mj_geomDistance(m,d,a,b,.002,endpoints))
            if distance>=.001 or abs(distance)<1e-10:continue
            direction=(endpoints[3:]-endpoints[:3])/distance;js=[]
            for geom,p in ((a,endpoints[:3]),(b,endpoints[3:])):
                jp=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,None,p,int(m.geom_bodyid[geom]));js.append(jp@self.g.C*1000.)
            res.append(12.*(1.-distance*1000.));jac.append(-12.*direction@(js[1]-js[0]))
        return np.array(res),np.array(jac).reshape(-1,len(u))

    def system(self,u,base,targets,angles):
        r,j=super().system(u,base,targets,angles);res=[r];jac=[j]
        self.g.assign(self.pad_data,u);poses={}
        for edge in targets.edges:
            if edge['weight']<1e-6:continue
            a,b=edge['a'],edge['b']
            for f in (a,b):
                if f not in poses:poses[f]=self.pad_jac(f)
            pa,ja,jna=poses[a];pb,jb,jnb=poses[b];w=self.contact_weight*np.sqrt(edge['weight'])
            distance,jdistance=self.surface(a,b,True)
            # Each pair has its own closest surface points. A thumb can touch
            # two fingers at different places, without collapsing three fixed
            # pad centres onto a single point.
            res.append(np.array([2.*w*(distance-edge['target_gap_mm'])]))
            jac.append((2.*w*jdistance)[None,:])
            w*=.15
            if a=='thumb':
                # Opposing pads approach the source-implied surface gap. All
                # four relationships coexist; no winner-take-all pinch class.
                gap=edge['target_gap_mm']
                delta=pa['position_mm']-pb['position_mm']-gap*pb['normal_world']
                res.extend([w*delta,3.*np.sqrt(edge['weight'])*(pa['normal_world']+pb['normal_world'])])
                jac.extend([w*(ja-jb-gap*jnb),3.*np.sqrt(edge['weight'])*(jna+jnb)])
            else:
                delta=pa['position_mm']-pb['position_mm'];distance=float(np.linalg.norm(delta))
                if distance<1e-7:continue
                # Adjacent fingers can touch side-by-side; their pad normals
                # need not oppose, so constrain separation only here.
                res.append(np.array([w*(distance-edge['target_gap_mm'])]))
                jac.append((w*(delta/distance)@(ja-jb))[None,:])
        for edge in targets.shafts:
            ga=self.bone_geoms['thumb'][edge['thumb_segment']];gb=self.bone_geoms[edge['finger']][edge['finger_segment']]
            distance,jdistance=self.geom_surface(ga,gb,True);w=2.*self.contact_weight*np.sqrt(edge['weight'])
            res.append(np.array([w*(distance-edge['target_gap_mm'])]));jac.append((w*jdistance)[None,:])
        cr,cj=self.clearance_rows(u)
        res.append(cr);jac.append(cj)
        return np.concatenate(res),np.vstack(jac)

    def metrics(self,u,targets):
        report=super().metrics(u,targets);self.g.assign(self.pad_data,u)
        report['contact_edges']=[]
        for edge in targets.edges:
            a,b=edge['a'],edge['b'];pa=self.pad.pose(self.pad_data,a);pb=self.pad.pose(self.pad_data,b)
            report['contact_edges'].append(dict(edge,pad_gap_mm=float(np.linalg.norm(pa['position_mm']-pb['position_mm'])),
                surface_gap_mm=self.surface(a,b)))
        report['shaft_contacts']=[dict(e,surface_gap_mm=self.geom_surface(self.bone_geoms['thumb'][e['thumb_segment']],
            self.bone_geoms[e['finger']][e['finger_segment']])) for e in targets.shafts]
        return report

    def apply(self,*args,**kwargs):
        u,report=super().apply(*args,**kwargs)
        report.update(kind=self.kind,contact_inference='continuous chain-length-scaled proximity; not measured touch',
            contact_point_scope='unconfirmed mesh surface pad candidates',multi_pair=True,forced_closure=False)
        return u,report
