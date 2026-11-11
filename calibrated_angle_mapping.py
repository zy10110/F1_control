"""Per-finger calibration of official parent-relative angles into 17 coupled coordinates."""
import hashlib
import json
from pathlib import Path

import numpy as np
from g7_direct_joint import G7DirectJointRetargeter
from hand_kinematics import FINGERS
from g7_joint_mapping import rotations
from g7_proportional_mapping import FingerLandmarks, normalized, BASE
from constrained_gesture_assist import fixed_palm_basis
from contact_graph_assist import ContactGraphAssist
from thumb_orientation_mapping import basis

CHANNELS = {'thumb': ['CMC_OPP','CMC_PS','MCP_AA','MCP','IP'],
            **{f:['MCP_AA','MCP','PIP','DIP'] for f in FINGERS[1:]}}


def joint_features(angles, finger):
    return np.radians([angles[finger][name] for name in CHANNELS[finger]])


def geometry_id(g, reference):
    digest = hashlib.sha256(g.cfg['source_sha256'].encode())
    for array in (g.C, g.reference, g.lower, g.upper, reference):
        digest.update(np.asarray(array, dtype='<f8').tobytes())
    return digest.hexdigest()


class AngleProfile:
    def __init__(self, record, geometry, reference, hand):
        self.g = geometry
        if (record.get('version') != 1 or record.get('hand') != hand
                or record.get('geometry_id') != geometry_id(geometry, reference)
                or record.get('labels') != geometry.labels or record.get('channels') != CHANNELS
                or record.get('units') != 'rad' or geometry.C.shape != (35,17)):
            raise ValueError('映射档案与目标手别、轴序、单位、限位或耦合矩阵不符')
        self.blocks = {}
        for f, names in CHANNELS.items():
            block = record['blocks'][f]
            zero, bias, matrix = (np.asarray(block[k], float) for k in ('zero','bias','matrix'))
            width = geometry.slices[f].stop - geometry.slices[f].start
            if (zero.shape != (len(names),) or bias.shape != (width,)
                    or matrix.shape != (width,len(names))
                    or not all(np.isfinite(v).all() for v in (zero,bias,matrix))):
                raise ValueError('无效角度映射系数 '+f)
            self.blocks[f] = zero, bias, matrix

    def raw(self, angles):
        result = np.empty(17)
        for f,(zero,bias,matrix) in self.blocks.items():
            result[self.g.slices[f]] = bias + matrix @ (joint_features(angles,f)-zero)
        if not np.isfinite(result).all():
            raise ValueError('无效输入关节角')
        return result

    def predict(self, angles):
        return np.clip(self.raw(angles),self.g.lower,self.g.upper)


class CalibratedAngleRetargeter(G7DirectJointRetargeter):
    def __init__(self, geometry, reference, hand, config, path):
        self.profile_path = str(Path(path).resolve())
        profile = json.loads(Path(path).read_text(encoding='utf8'))
        angle_profile = AngleProfile(profile,geometry,reference,hand)
        super().__init__(geometry,reference,hand,dict(config,thumb_linear_assist_enabled=False),
                         open_frame=profile['open_frame'])
        self.lower, self.upper = geometry.lower.copy(), geometry.upper.copy()
        self.angle_profile = angle_profile
        self.limit_assist=config.get('calibrated_limit_assist',True)
        self.land=FingerLandmarks(geometry)
        points=np.asarray(self.open_frame['positions_m'])
        rest=rotations(self.open_frame)
        self.wrist_to_f1=fixed_palm_basis(geometry)@basis(points[5]-points[17],
            points[[5,9,13,17]].mean(axis=0)-points[0]).T@rest[0].as_matrix()
        self.bone_local={f:rest[i:i+3].inv().apply(normalized(np.diff(points[i:i+4],axis=0)))
                         for f,i in BASE.items()}
        self.roots_wrist={f:rest[0].inv().apply(points[i]-points[0])*1000 for f,i in BASE.items()}
        self.lengths_mm={f:np.linalg.norm(np.diff(points[i:i+4],axis=0),axis=1)*1000 for f,i in BASE.items()}
        self.contact_model=ContactGraphAssist(geometry,config,reference,hand)
        self.mapping_version = 'calibrated_coupled_angles_v1'
        self.opposition = {}
        self.contact_strengths = {}
        supplement = Path(path).with_suffix('.opposition.json')
        if self.limit_assist and config.get('personal_opposition', True) and supplement.is_file():
            data = json.loads(supplement.read_text(encoding='utf8'))
            if data.get('version') == 1 and data.get('profile_sha256') == hashlib.sha256(Path(path).read_bytes()).hexdigest():
                widths = np.asarray(data['middle_widths_rad'], float)
                if widths.shape != (4,) or not np.all((widths > 0) & (widths <= np.radians(15))):
                    raise ValueError('中指末段映射范围无效')
                self.middle_widths = widths
                for f in FINGERS[1:]:
                    sl = geometry.slices[f]
                    ids = np.r_[0:5, sl.start:sl.stop]
                    base, anchor = (np.asarray(data['anchors'][f][k], float) for k in ('base', 'anchor'))
                    for x in (base, anchor):
                        if x.shape != (8,) or not np.all((x >= self.lower[ids]-1e-8) & (x <= self.upper[ids]+1e-8)):
                            raise ValueError('对指标定角度无效 '+f)
                    self.opposition[f] = (ids, base, anchor)
                self.mapping_version = 'calibrated_coupled_pad_opposition_v2'

    def personal_command(self, raw, command, proximity):
        # Join the interior linear map to each hard stop with matching slope.
        # The overshoot measured during fitting still has usable output travel.
        x = raw[9:11]
        lo, hi = self.lower[9:11], self.upper[9:11]
        wl, wh = self.middle_widths[:2], self.middle_widths[2:]
        flex = np.where(x < lo + wl, lo + np.maximum(0., x-lo+wl)**2/(4*wl), x)
        command[9:11] = np.where(x > hi - wh, hi - np.maximum(0., hi+wh-x)**2/(4*wh), flex)
        origin = command.copy()
        self.contact_strengths = {}
        for f, (ids, base, anchor) in self.opposition.items():
            # Ambiguous multi-finger closure should not pull the thumb between
            # several different single-pinch anchors. Fade continuously.
            competitor = max(proximity[other] for other in self.opposition if other != f)
            strength = proximity[f] * (1.-competitor)
            self.contact_strengths[f] = float(strength)
        total = max(1., sum(self.contact_strengths.values()))
        for f, (ids, base, anchor) in self.opposition.items():
            weight = self.contact_strengths[f] / total
            if weight:
                # Preserve local angular variation around a demonstrated pinch.
                # Only the thumb and the finger approaching it share this adjustment.
                target = np.clip(anchor + .15*(origin[ids]-base), self.lower[ids], self.upper[ids])
                command[ids] += weight * (target-origin[ids])
        # A changed approach angle can put even a good pair anchor just inside
        # the safety margin. Correct along its local clearance gradient before
        # the sweep, so a held pinch does not repeatedly run into the guard.
        for f, strength in self.contact_strengths.items():
            if strength <= .65:
                continue
            ids = self.opposition[f][0]
            self.g.assign(self.contact_model.pad_data, command)
            gap, jac = self.contact_model.surface('thumb', f, True)
            error = 1.0 - gap
            grad = jac[ids].copy()
            grad[((command[ids] <= self.lower[ids]+1e-7) & (grad*error < 0)) |
                 ((command[ids] >= self.upper[ids]-1e-7) & (grad*error > 0))] = 0.
            step = np.clip(error*grad/(grad@grad + 5.**2), -np.radians(3), np.radians(3))
            command[ids] += ((strength-.65)/.35)**2 * step
        command = np.clip(command, self.lower, self.upper)
        free_weight = 1. - max(self.contact_strengths.values())
        if free_weight > 1e-6:
            command = self.clear_thumb_for_free_motion(command, free_weight)
        return command

    def clear_thumb_for_free_motion(self, command, free_weight):
        """Keep a small clearance reserve when a free finger passes the thumb."""
        model = self.contact_model
        original = command[:5].copy()
        budget = np.radians(6.) * free_weight
        for _ in range(3):
            distances = self.collision.distances(command)
            close = np.flatnonzero(distances < 1.1)
            if not len(close):
                break
            self.g.assign(model.pad_data, command)
            jac = np.array([model.geom_surface(*self.collision.pairs[k], jacobian=True)[1][:5]
                            for k in close])
            delta = np.linalg.solve(jac.T@jac + 6.**2*np.eye(5), jac.T@(1.1-distances[close]))
            trial = command.copy()
            trial[:5] = np.clip(command[:5]+delta, np.maximum(self.lower[:5], original-budget),
                                np.minimum(self.upper[:5], original+budget))
            if self.collision.distances(trial).min() <= distances.min()+1e-5:
                break
            command = trial
        return command

    def calibrated_command(self,frame,angles):
        raw=self.angle_profile.raw(angles)
        bounded=np.clip(raw,self.lower,self.upper)
        command=bounded.copy()
        if not self.limit_assist or (not self.opposition and np.max(np.abs(raw-bounded))<1e-8):
            return raw,bounded,command
        orientations=rotations(frame)
        wrist=orientations[0].inv()
        directions={f:(wrist*orientations[i:i+3]).apply(self.bone_local[f]) for f,i in BASE.items()}
        tips={f:self.roots_wrist[f]+self.lengths_mm[f]@directions[f] for f in BASE}
        proximity={}
        for f in FINGERS[1:]:
            length=min(self.lengths_mm[f].sum(),self.lengths_mm['thumb'].sum())
            ratio=np.linalg.norm(tips['thumb']-tips[f])/length
            t=np.clip((.50-ratio)/.22,0.,1.)
            proximity[f]=float(t*t*(3-2*t))
        proximity['thumb']=max(proximity.values())
        self.land.assign(bounded)
        for f,sl in self.g.slices.items():
            # The trust region opens continuously as the angular request exceeds
            # a bound. Interior poses keep the calibrated map exactly.
            cap=min(np.radians(8.),np.max(np.abs(raw[sl]-bounded[sl])))*(1.-proximity[f])
            if cap<1e-8:continue
            target=directions[f]@self.wrist_to_f1.T
            p,j=self.land.points(f,True)
            v=np.diff(p,axis=0);n=normalized(v);length=np.linalg.norm(v,axis=1)
            dn=np.array([(np.eye(3)-np.outer(a,a))@b/l for a,b,l in
                         zip(n,np.diff(j[:,:,sl],axis=0),length)])
            # Keep tip displacement and axis changes small while reducing
            # direction error. Solve only in this digit's original coordinates.
            J=dn.reshape(9,-1);tip=j[-1,:,sl]/length.sum()
            H=J.T@J+tip.T@tip+.20*np.eye(J.shape[1])
            step=np.linalg.solve(H,J.T@(target-n).ravel())
            delta=np.clip(step,np.maximum(-cap,self.lower[sl]-bounded[sl]),
                               np.minimum(cap,self.upper[sl]-bounded[sl]))
            loss=np.sum((target-n)**2)
            for fraction in (1.,.5,.25):
                candidate=bounded.copy();candidate[sl]+=fraction*delta
                self.land.assign(candidate);q=self.land.points(f)
                trial=np.sum((target-normalized(np.diff(q,axis=0)))**2)
                trial+=np.sum(((q[-1]-p[-1])/length.sum())**2)+.20*np.sum((fraction*delta)**2)
                if trial<loss:
                    command[sl]=candidate[sl];break
            self.land.assign(bounded)
        # For opposition, distribute a small correction over the involved
        # digits using actual F1 surface separation. Keep all other digits
        # independent. Backtracking accepts only a lower multi-contact cost.
        if self.opposition:
            return raw,bounded,self.personal_command(raw,command,proximity)
        active=[f for f in FINGERS[1:] if proximity[f]>.001]
        if active:
            digits=['thumb',*active]
            ids=np.concatenate([np.arange(self.g.slices[f].start,self.g.slices[f].stop) for f in digits])
            excess=np.abs(raw-bounded)
            caps={f:min(np.radians(8.),max(excess[self.g.slices[f]].max(),excess[:5].max()))*proximity[f] for f in active}
            caps['thumb']=max(caps.values())
            cap=np.concatenate([np.full(self.g.slices[f].stop-self.g.slices[f].start,caps[f]) for f in digits])
            if cap.max()>1e-7:
                origin=command.copy()
                def residual(u,derivatives=False):
                    self.g.assign(self.contact_model.pad_data,u);errors=[];rows=[]
                    for f in active:
                        item=self.contact_model.surface('thumb',f,derivatives)
                        gap,j=item if derivatives else (item,None)
                        target=.85+max(0.,np.linalg.norm(tips['thumb']-tips[f])-.22*min(
                            self.lengths_mm['thumb'].sum(),self.lengths_mm[f].sum()))
                        weight=proximity[f]
                        errors.append(weight*(gap-target))
                        if derivatives:rows.append(weight*j[ids])
                    return np.array(errors),np.array(rows)
                e,J=residual(command,True)
                # Damped minimum motion in radians; original box bounds
                # and later segment-by-segment safety checks remain active.
                delta=-J.T@np.linalg.solve(J@J.T+12.**2*np.eye(len(e)),e)
                delta=np.clip(delta,np.maximum(-cap,self.lower[ids]-origin[ids]),
                                    np.minimum(cap,self.upper[ids]-origin[ids]))
                for scale in (1.,.5,.25):
                    trial=origin.copy();trial[ids]+=scale*delta
                    new,_=residual(trial)
                    if new@new+12.**2*np.sum((scale*delta)**2)<e@e:
                        command=trial;break
        return raw,bounded,command

    def solve_frame(self, frame):
        result=super().solve_frame(frame)
        result.update(control_mode='calibrated_joint',retargeting_kind='calibrated_joint_visual',
            f1_extra_calibration_used=True,calibrated_to_wearer=True,
            coupling_driver='calibrated PIP and DIP',joint_zero='profile open frame',
            calibration_scope='per-finger angle map fitted to coupled F1 pose examples',
            calibrated_angle_profile=self.profile_path,projection_method='per-finger fitted angular gains, original coupled bounds and sampled guard',
            pipeline='parent-relative angles -> calibrated finger blocks -> original limits and collision sweep')
        result['joint_mapping_gains']={f:block[2] for f,block in self.angle_profile.blocks.items()}
        result['native_qpos_deg']=np.degrees(self.g.reference+self.g.C@result['goal_rad'])
        result['range_excess_deg']=np.degrees(result['unclipped_joint_rad']-result['direct_joint_rad'])
        result['range_compensation_deg']=np.degrees(result['desired_joint_rad']-result['direct_joint_rad'])
        result['limit_assist']=self.limit_assist
        result['personal_opposition_weights'] = self.contact_strengths.copy()
        return result

    def mapping_record(self):
        return dict(kind=self.mapping_version,hand=self.hand,model_mode=self.g.mode,
                    calibrated_angle_profile=self.profile_path,coordinate_labels=self.g.labels,
                    source_channels=CHANNELS,hardware_enabled=False,
                    coupling='q_native = q_assembly + C @ u')
