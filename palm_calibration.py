"""Common-palm coordinates and explicit candidate finger-pad calibration.

No serial writes, no hardware output, no automatic activation of fitted data.
Contact residuals alone are not proof of identifiable physical point offsets.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from hand_kinematics import ROOT, FINGERS, serializable, vector
from glove_teleop import unit


def write_json(path, record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=serializable),encoding='utf8')
    tmp.replace(path)


def basis(x, outward):
    x=unit(x); y=unit(outward-x*np.dot(outward,x))
    return np.column_stack((x,y,unit(np.cross(x,y))))


def source_poses(glove, feedback):
    return {('pinky' if f=='little' else f):p for f,p in
            glove.root_fk.poses_in_base(feedback).items()}


def pad_positions(poses, local_points):
    return {f: poses[f]['body_position_m']+poses[f]['body_rotation']@vector(local_points[f],3,'指腹局部点')
            for f in FINGERS}


def build_pad_candidates(g, reference):
    """Pick distal palm-side SURFACE TRIANGLES; all remain user-review candidates."""
    import trimesh
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    tree=ET.parse(ROOT/'assets/hand.urdf').getroot()
    links={l.get('name'):l for l in tree.findall('link')}
    original={v:k for k,v in g.cfg['body_names'].items()}
    result={}
    for f in FINGERS:
        spec=g.cfg['fingers'][f]; body=spec['tip_body']; link=links[original[body]]
        g.forward(reference,f)
        bid=g.model.body(body).id; R=g.data.xmat[bid].reshape(3,3)
        ray=g.position_jacobian(f)[:,-2]
        normal=unit(R.T@ray)
        old=np.array(spec['tip_local_m'])
        along=unit(old-normal*np.dot(old,normal))
        lateral=unit(np.cross(along,normal))
        choices=[]
        for visual_index,v in enumerate(link.findall('visual')):
            mesh=v.find('geometry/mesh'); origin=v.find('origin')
            verts=trimesh.load_mesh(ROOT/'assets'/mesh.get('filename'),process=False)
            scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            offset=np.zeros(3) if origin is None else np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            rpy=np.zeros(3) if origin is None else np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
            xyz=Rotation.from_euler('xyz',rpy).apply(verts.vertices*scale)+offset
            triangles=xyz[verts.faces]
            centers=triangles.mean(axis=1)
            normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            lengths=np.linalg.norm(normals,axis=1)
            normals=normals/np.maximum(lengths[:,None],1e-15)
            progress=centers@along; high=float(np.max(xyz@along))
            valid=(lengths>1e-12)&(progress>.6*high)&(normals@normal>.55)
            if not np.any(valid): continue
            ids=np.flatnonzero(valid)
            # Distal pad center, lateral center; retain a real surface point.
            score=(progress[ids]-.82*high)**2+(centers[ids]@lateral)**2
            k=ids[np.argmin(score)]
            choices.append((float(np.min(score)),dict(body=body,point_local_m=centers[k],
                normal_local=normals[k],triangle_local_m=triangles[k],
                visual_index=visual_index,triangle_index=int(k),
                mesh=mesh.get('filename'),mesh_sha256=hashlib.sha256((ROOT/'assets'/mesh.get('filename')).read_bytes()).hexdigest(),
                status='candidate palm-side surface triangle; requires visual confirmation',
                confirmed_by_user=False)))
        if not choices: raise ValueError(f+'：没有找到可复核的掌侧指腹候选三角形')
        result[f]=min(choices,key=lambda c:c[0])[1]
    return dict(schema_version=1,units='m',source_sha256=g.cfg['source_sha256'],
                status='unconfirmed surface candidates; no physical contact validation',fingers=result)


class PadGeometry:
    def __init__(self,g,record):
        self.g=g; self.record=record
        if record['source_sha256']!=g.cfg['source_sha256']: raise ValueError('指腹候选与源模型不一致')
        self.ids={f:g.model.body(record['fingers'][f]['body']).id for f in FINGERS}

    def pose(self,data,f):
        bid=self.ids[f]; rec=self.record['fingers'][f]; R=data.xmat[bid].reshape(3,3)
        p=data.xpos[bid]+R@np.array(rec['point_local_m'])
        return dict(position_world_m=p,position_mm=(p-self.g.origin)*1000,
                    normal_world=R@np.array(rec['normal_local']))

    def jacobian(self,data,f):
        p=self.pose(data,f)['position_world_m']; J=np.zeros((3,self.g.model.nv))
        mujoco.mj_jac(self.g.model,data,J,None,p,self.ids[f])
        return J@self.g.C*1000


def build_palm_alignment(g,reference,glove,pads):
    nominal=glove.neutral_angles_deg
    poses={('pinky' if f=='little' else f):p for f,p in glove.nominal_root_fk.poses_in_base(nominal).items()}
    sg={f:p['root_position_m'] for f,p in poses.items()}
    source_out=poses['middle']['root_rotation']@glove.frames['middle']['extension']
    BG=basis(sg['index']-sg['pinky'],source_out)
    g.forward(reference,'middle'); dest={}
    for f in FINGERS:
        sl=g.slices[f]; aa=2 if f=='thumb' else 0
        name=g.cfg['fingers'][f]['columns'][aa][0][0]
        dest[f]=g.data.xanchor[g.model.joint(name).id].copy()
    pg=PadGeometry(g,pads)
    dest_out=pg.pose(g.data,'middle')['position_world_m']-dest['middle']
    BF=basis(dest['index']-dest['pinky'],dest_out)
    source_bend=poses['middle']['root_rotation']@glove.frames['middle']['bend']
    dest_bend=g.position_jacobian('middle')[:,-2]
    parity=float(np.sign(np.dot(source_bend,BG[:,2])*np.dot(dest_bend,BF[:,2])))
    if parity==0: raise ValueError('无法确定掌侧镜像方向')
    widthG=np.linalg.norm(sg['index']-sg['pinky']); widthF=np.linalg.norm(dest['index']-dest['pinky'])
    lengthG=glove.frames['middle']['planar_length_m']; lengthF=np.linalg.norm(dest_out)
    scales=np.array([widthF/widthG,lengthF/lengthG,lengthF/lengthG])
    A=BF@np.diag(scales)@np.diag([1.,1.,parity])@BG.T
    return dict(schema_version=1,hand=glove.hand,units='m',
        source_frame=dict(origin_base_m=sg['middle'],basis_base=BG),
        destination_frame=dict(origin_world_m=dest['middle'],basis_world=BF),
        shared_map=dict(linear=A,source_origin_m=sg['middle'],destination_origin_m=dest['middle'],
                        anatomical_scales=scales,lateral_reflection=parity),
        status='nominal common mapping; shared affine size adaptation, not measured extrinsics',
        source_provenance=glove.provenance,f1_source_sha256=g.cfg['source_sha256'],
        J17_status='user-confirmed physically fixed; model installation angle retained')


def map_common_positions(positions,alignment):
    m=alignment['shared_map']; A=np.array(m['linear'])
    return {f:np.array(m['destination_origin_m'])+A@(
                 vector(p,3,'公共基座指腹点')-m['source_origin_m']) for f,p in positions.items()}


def fit_pad_offsets(glove,samples,initial_points,offset_bound_mm=30.):
    """Fit local point offsets from LABELED touches, with bounds and a prior.

    Rank is computed on DATA ONLY, before adding the regularizer. A low fitted
    residual never upgrades a deficient system to a physical calibration.
    """
    contacts=[s for s in samples if s.get('role')=='fit' and s.get('finger') in FINGERS[1:]]
    active=['thumb']+[f for f in FINGERS[1:] if any(s['finger']==f for s in contacts)]
    if len(active)<2: raise ValueError('需要至少一组已标注的拇指轻触样本')
    slots={f:slice(i*3,i*3+3) for i,f in enumerate(active)}
    rows=[]; rhs=[]; labels=[]
    for s in contacts:
        poses=source_poses(glove,np.array(s['median_feedback_deg']))
        points=pad_positions(poses,initial_points); f=s['finger']
        A=np.zeros((3,len(active)*3)); A[:,slots['thumb']]=poses['thumb']['body_rotation']
        A[:,slots[f]]=-poses[f]['body_rotation']
        rows.append(A); rhs.append((points[f]-points['thumb'])*1000); labels.append(s['pose'])
    A=np.vstack(rows); b=np.concatenate(rhs)
    singular=np.linalg.svd(A,compute_uv=False)
    threshold=max(float(singular[0])*1e-5,1e-9)
    rank=int(np.sum(singular>threshold)); columns=A.shape[1]
    # A small bounded prior stabilizes approximate fits; it is not data.
    def residual(x): return np.r_[A@x-b,x/20.]
    fit=least_squares(residual,np.zeros(columns),bounds=(-offset_bound_mm,offset_bound_mm),
                      loss='soft_l1',f_scale=2.,max_nfev=200)
    points={f:np.array(p).copy() for f,p in initial_points.items()}
    for f in active: points[f]+=fit.x[slots[f]]*.001
    errors=np.linalg.norm((A@fit.x-b).reshape(-1,3),axis=1)
    at_bound=bool(np.max(np.abs(fit.x))>.95*offset_bound_mm)
    return dict(schema_version=1,units='m',pad_local_m=points,
        active_fingers=active,fit_labels=labels,residuals_mm=dict(zip(labels,errors)),
        data_rank=rank,unknowns=columns,singular_values=singular,
        identifiable=bool(rank==columns),offsets_mm={f:fit.x[slots[f]] for f in active},
        at_bound=at_bound,offset_bound_mm=offset_bound_mm,
        status=('candidate needs independent validation' if rank==columns and not at_bound else
                'approximate prior-dependent candidate; additional poses/measurements required'),
        validated=False,enabled=False,
        note='No fingertip normals, joint zeros, lengths or global frames were fitted from these touches.')
