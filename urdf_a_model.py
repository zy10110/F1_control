"""Native URDF-A geometry for calibration, replay and live FK."""
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from hand_kinematics import FINGERS


def vec(node, key, fallback):
    return np.fromstring(node.get(key, fallback) if node is not None else fallback, sep=' ')


class NativeHand:
    """37 native hinge rows, 17 semantic columns; the two wrist rows are zero."""
    def __init__(self, package):
        path, = (package/'urdf').glob('*.urdf')
        self.sha = hashlib.sha256(path.read_bytes()).hexdigest()
        xml = ET.parse(path).getroot()
        files = {}
        for mesh in xml.iter('mesh'):
            name = Path(mesh.attrib['filename']).name
            if name not in files:
                files[name] = (package/'meshes'/name).read_bytes()
            mesh.set('filename', name)
        self.model = m = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'), files)
        if (m.nq, m.nv, m.njnt) != (37, 37, 37) or not np.all(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('Expected the 37 scalar hinges of the A model')
        self.data = mujoco.MjData(m)
        self.slices = {}
        digest = hashlib.sha256(path.read_bytes())
        for name, payload in sorted(files.items()):
            digest.update(name.encode('utf8'))
            digest.update(hashlib.sha256(payload).digest())
        self.C = np.zeros((37, 17))
        self.root_joints, self.distal, self.pad, self.pair_axes = {}, {}, {}, {}
        self.chains = {}
        self.cols = {}
        self.tip = {}
        links = {node.get('name'): node for node in xml.findall('link')}
        for i, finger in enumerate(FINGERS):
            prefix = 'r_'+('little' if finger == 'pinky' else finger)+'_'
            is_thumb = finger == 'thumb'
            groups = (['cmc_inward','cmc_spin','mcp_abd'] if is_thumb else ['mcp_abd'])
            segments = ['mcp','ip'] if is_thumb else ['mcp','pip','dip']
            names = groups + [s+'_flex_'+n for s in segments for n in ('1','2')]
            chain = [m.joint(prefix+n) for n in names]
            self.chains[finger] = [int(m.jnt_bodyid[j.id]) for j in chain]
            self.root_joints[finger] = chain[0].id
            col = 0 if is_thumb else 5+(i-1)*3
            self.cols[finger] = np.arange(col, col+(5 if is_thumb else 3))
            self.slices[finger] = slice(col, col+len(self.cols[finger]))
            for k, j in enumerate(chain[:len(groups)]):
                self.C[j.qposadr[0], col+k] = 1.
            for k, segment in enumerate(segments):
                indices = [m.joint(prefix+segment+'_flex_'+n) for n in ('1','2')]
                self.pair_axes[finger,segment] = [j.id for j in indices]
                for j in indices:
                    self.C[j.qposadr[0], col+len(groups)+min(k,1)] = .4 if segment=='dip' else .5
            self.distal[finger] = self.chains[finger][-1]
            # In A, distal +X continues the phalanx, +Y faces the positive
            # flexion sweep about +Z. Use an actual outward-facing triangle.
            link = links[prefix+names[-1]]
            collision = link.find('collision')
            mesh = collision.find('geometry/mesh')
            origin = collision.find('origin')
            surface = trimesh.load_mesh(package/'meshes'/mesh.get('filename'), process=False)
            xyz = Rotation.from_euler('xyz', vec(origin,'rpy','0 0 0')).apply(
                surface.vertices * vec(mesh,'scale','1 1 1')) + vec(origin,'xyz','0 0 0')
            tri = xyz[surface.faces]
            centers = tri.mean(1)
            ns = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
            ns /= np.maximum(np.linalg.norm(ns,axis=1,keepdims=True),1e-15)
            length = xyz[:,0].max()
            self.tip[finger] = xyz[xyz[:,0] >= .97*length].mean(axis=0)
            candidates = np.flatnonzero((centers[:,0]>.6*length) & (ns[:,1]>.55))
            score = (centers[candidates,0]-.82*length)**2 + centers[candidates,2]**2
            idx = candidates[np.argmin(score)]
            self.pad[finger] = (centers[idx], ns[idx])
        self.lo, self.hi = [], []
        for col in self.C.T:
            rows = np.flatnonzero(col)
            limits = m.jnt_range[rows] / col[rows,None]
            self.lo.append(limits[:,0].max()); self.hi.append(limits[:,1].min())
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)
        self.assign(np.zeros(17))
        self.palm = m.body('r_wrist_flex_link').id
        def geom(body):
            ids = np.flatnonzero(m.geom_bodyid==body)
            if len(ids)!=1: raise ValueError('expected one collision mesh per body')
            return int(ids[0])
        self.geoms = {f:geom(b) for f,b in self.distal.items()}
        # The original monitoring scope: the five distal thumb links versus
        # the palm and all long-finger links. Exclude attached root assemblies.
        other = [self.palm] + [b for f in FINGERS[1:] for b in self.chains[f]]
        self.pairs = np.array([(geom(a),geom(b)) for a in self.chains['thumb'][2:] for b in other])

        for array in (self.C, self.lo, self.hi, *(v for pair in self.pad.values() for v in pair)):
            digest.update(np.asarray(array,dtype='<f8').tobytes())
        self.geometry_id = digest.hexdigest()

    def validate(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape != (17,) or not np.all((u >= self.lo-1e-9) & (u <= self.hi+1e-9)):
            raise ValueError('A 版独立关节角超限或无效')
        return np.clip(u,self.lo,self.hi)

    def assign(self, u):
        q = self.C @ u
        limits = self.model.jnt_range
        if not np.all((q>=limits[:,0]-1e-9)&(q<=limits[:,1]+1e-9)):
            raise ValueError('A native limit exceeded')
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def pad_pose(self, finger):
        b = self.distal[finger]
        r = self.data.xmat[b].reshape(3,3)
        p,n = self.pad[finger]
        return (self.data.xpos[b]+r@p)*1000, r@n

    def gaps(self, finger):
        a,n = self.pad_pose('thumb'); b,m = self.pad_pose(finger)
        d = mujoco.mj_geomDistance(self.model,self.data,self.geoms['thumb'],self.geoms[finger],.5,None)
        return dict(pad_mm=np.linalg.norm(a-b),normal_error_deg=np.degrees(np.arccos(np.clip(-n@m,-1,1))),shell_mm=d*1000)

    def bone_directions(self, finger):
        thumb = finger == 'thumb'
        segments = ('mcp','ip') if thumb else ('mcp','pip','dip')
        points = [self.data.xanchor[self.pair_axes[finger,s]].mean(axis=0) for s in segments]
        if thumb:
            points.insert(0,self.data.xanchor[self.model.joint('r_thumb_mcp_abd').id])
        b = self.distal[finger]
        points.append(self.data.xpos[b]+self.data.xmat[b].reshape(3,3)@self.tip[finger])
        delta = np.diff(points, axis=0)
        return delta / np.linalg.norm(delta,axis=1,keepdims=True)

    def clearance(self):
        d, m = self.data, self.model
        a,b = self.pairs.T
        bound = np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b],axis=1)-m.geom_rbound[a]-m.geom_rbound[b]
        return min((1000*mujoco.mj_geomDistance(m,d,i,j,.01,None) for i,j in self.pairs[bound<.01]),default=10.)

    def approach(self, u):
        count = max(1,int(np.ceil(np.max(np.abs(np.degrees(u)))/.75)))
        gaps = []
        for t in np.linspace(0,1,count+1):
            self.assign(t*u)
            gaps.append(self.clearance())
        return min(gaps)


