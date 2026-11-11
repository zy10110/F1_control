"""Record, fit and evaluate a right-hand angle profile for F1 simulation."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize
from calibrated_angle_mapping import CHANNELS, AngleProfile, geometry_id, joint_features
from contact_graph_assist import ContactGraphAssist
from g7_direct_joint import G7DirectJointRetargeter
from g7_official_udp import OfficialUDPReceiver, official_config, open_reference
from g7_teleop import BASE, load_g7_config, load_reader
from hand_kinematics import ROOT, FINGERS, HandGeometry, serializable

POSES = [
    ('open','自然张开','掌心朝前，手腕放松。五指轻轻伸直，指间留一点自然的空隙。','食指到小指大致朝上，拇指自然偏向旁边。'),
    ('mcp','桌面手形｜只折指根','先竖起手掌，像做“停止”的手势。食指到小指并排，从四指连着手掌的那排关节一起向掌心方向折，接近直角即可。拇指放松。','从侧面看，手掌竖直，四指像一块水平的小桌板。每根手指的中间关节和靠指甲的关节都尽量伸直。'),
    ('hook','钩拳｜只勾手指后两节','重新竖掌，指尖向上。让四指连着手掌的第一节仍向上，弯曲手指中间和靠近指甲的两个关节。拇指放松。','四个指尖像钩子，朝向各自指根。侧面看，手背和四指的第一节大致在一条线上。'),
    ('fist','自然握拳','从张开开始，四根手指连同指根一起弯，让指尖向掌心收。拇指轻搭在食指和中指外侧。','看起来像松松握住一根细杆，保持舒服的力度。'),
    ('spread','四指扇形张开','掌心朝前，伸直食指到小指，向两边分开手指，像张开一把扇子。拇指自然放松。','从掌心正面能看到各指间的空隙比自然张开时更大。'),
    ('thumb_out','拇指向掌外展开','四指并排伸直。让拇指从虎口处向侧面张开，拇指和食指形成一个宽的 L 形。','从掌心正面看，虎口打开，拇指两节自然伸展。'),
    ('thumb_in','拇指横过掌心','四指并排伸直。以虎口下方为起点，让整个拇指扫过掌心，拇指尖朝小指根部。','从掌心正面能看到拇指斜跨手掌，够到舒服的位置就好。'),
    ('thumb_mcp','弯拇指靠近手掌的关节','将拇指从食指旁稍微张开。找到拇指上靠虎口、弯起来会鼓起的关节，从这里把拇指折向掌心。','拇指靠指甲的那个关节尽量伸直。看起来是整段拇指尖一起转向掌心。'),
    ('thumb_ip','只勾拇指尖','将拇指稍微张开，保持虎口和靠手掌的那一节朝向稳定。只从最靠近拇指指甲的关节，把指尖轻轻勾向掌心。','从侧面看，带指甲的一小节弯了，前面的那一节仍朝原方向。'),
    *[(f,'拇指与'+name+'指指腹相碰',
       '让拇指和'+name+'指弯成一个圈，两指指尖有指纹的软垫轻轻相贴。其余三指放松。',
       '从侧面检查两块软垫相对，只让拇指碰这根手指。')
      for f,name in zip(FINGERS[1:],('食','中','无名','小'))],
    ('side_contact','拇指贴食指侧面','食指微弯。把拇指带指甲的那一节靠向食指朝拇指的一侧，轻贴在食指根部到中间关节之间、靠近中间关节的位置。','食指尖仍露在外面，两指尖之间留有空隙。其余三指放松。'),
    ('motion','连续动作｜按回车后开始动','从张开开始。前两秒慢慢半握、握拳、松开；接下来让食指到小指依次轻弯，最后让拇指横过掌心再打开。','整个采集约 5 秒，顺畅地做一遍即可。')
]


def session_progress(folder, resume):
    if not any(folder.iterdir()):return {'train':[],'validation':[]}
    if not resume:raise ValueError('采集目录必须为空，继续原会话请加 --resume')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    for split in ('train','validation'):
        rows=manifest[split]
        if len(rows)>len(POSES):raise ValueError('进度与当前流程不符')
        for row,pose in zip(rows,POSES):
            filename=f'{split}_{pose[0]}.jsonl'
            if row['stage']!=pose[0] or row['file']!=filename or not (folder/filename).is_file():
                raise ValueError('进度与已有采集文件不符，请检查目录')
    if manifest['validation'] and len(manifest['train'])!=len(POSES):raise ValueError('训练轮进度不完整')
    return manifest


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,default=serializable,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(path)


def record(folder,resume=False):
    folder.mkdir(parents=True,exist_ok=True)
    manifest=session_progress(folder,resume)
    reader=load_reader(load_g7_config())
    current=[];failures=[];stop=threading.Event();lock=threading.Lock()
    recording=threading.Event()
    def put(frame):
        with lock:
            if recording.is_set(): current.append(frame)
    def receive():
        try: OfficialUDPReceiver(reader.decode_udp,'right').receive(stop,put)
        except OSError as exc: failures.append(str(exc))
    worker=threading.Thread(target=receive,daemon=True);worker.start()
    try:
        for split in manifest:
            for step,(stage,title,instruction,check) in enumerate(POSES,1):
                if step<=len(manifest[split]):continue
                name=f'{split}_{stage}.jsonl'
                if (folder/name).exists():raise ValueError('发现尚未登记的样本，已保留文件：'+name)
                while True:
                    print(f'\n{"第一轮 · 标定" if split=="train" else "第二轮 · 验证"}   {step}/{len(POSES)}   {title}')
                    print('怎么摆：'+instruction+'\n自己看：'+check)
                    input('准备好后按回车，开始做连续动作：' if stage=='motion' else '摆好后按回车，用另一只手按键，右手保持 1 秒：')
                    if failures: raise RuntimeError(failures[0])
                    duration=5. if stage=='motion' else 1.
                    with lock:
                        current.clear()
                        recording.set()
                    time.sleep(duration)
                    with lock:
                        recording.clear()
                        rows=list(current)
                    if failures: raise RuntimeError(failures[0])
                    try:
                        if len(rows)<20: raise ValueError('有效帧不足 20，请检查官方 UDP 发送')
                        elapsed=(rows[-1]['timestamp_ns']-rows[0]['timestamp_ns'])*1e-9
                        if elapsed<duration*.65: raise ValueError('有效记录时间不足')
                        if stage!='motion':
                            stationary=open_reference(rows)
                            if stage in FINGERS[1:]:
                                points=np.asarray(stationary['positions_m'])
                                digit=points[BASE[stage]:BASE[stage]+4]
                                length=np.linalg.norm(np.diff(digit,axis=0),axis=1).sum()
                                if np.linalg.norm(points[4]-digit[-1])>.40*length:
                                    raise ValueError('官方模型中这两个指尖距离偏大，请核对目标手指')
                    except ValueError as exc:
                        print(exc,'，请重做此动作');continue
                    break
                (folder/name).write_text(''.join(json.dumps(v,default=serializable)+'\n' for v in rows),encoding='utf8')
                manifest[split].append(dict(stage=stage,file=name,kind='motion' if stage=='motion' else 'hold'))
                write(folder/'manifest.json',manifest)
    finally:
        stop.set();worker.join(2)


def load_groups(manifest, split):
    groups=[]
    for entry in json.loads(manifest.read_text(encoding='utf8'))[split]:
        path=(manifest.parent/entry['file']).resolve()
        raw=path.read_text(encoding='utf-8-sig')
        rows=[json.loads(raw)] if path.suffix=='.json' else [json.loads(line) for line in raw.splitlines() if line.strip()]
        if 'rows' in entry:
            lo,hi=entry['rows']
            if not 0<=lo<hi<=len(rows):raise ValueError('无效记录范围 '+str(path))
            rows=rows[lo:hi]
        if not rows or any(row.get('hand')!=2 or row.get('new_solution',True)==False for row in rows):
            raise ValueError('缺少有效右手解算记录 '+str(path))
        if entry['kind']=='hold':
            frames=[open_reference(rows)]
        elif entry['kind']=='reference' and len(rows)==1:
            frames=rows  # archived mean or explicit single labelled frame
        elif entry['kind']=='motion':
            indices=np.linspace(0,len(rows)-1,min(len(rows),int(entry.get('samples',32))),dtype=int)
            frames=[rows[i] for i in indices]
        else: raise ValueError('未知样本格式 '+str(path))
        groups.append((entry,frames))
    return groups


def check_split(manifest):
    entries=json.loads(manifest.read_text(encoding='utf8'))
    for train in entries['train']:
        for val in entries['validation']:
            if (manifest.parent/train['file']).resolve()!=(manifest.parent/val['file']).resolve():continue
            a,b=train.get('rows',(0,float('inf')))
            c,d=val.get('rows',(0,float('inf')))
            if max(a,c)<min(b,d): raise ValueError('训练与验证重复使用同一段记录')


def initial_matrix(f, mapper):
    if f=='thumb':
        axes=mapper.root_axes
        local=(mapper.thumb_root_mapper.palm_rotation*mapper.open_rotations['thumb'][0]).as_matrix()
        matrix=np.eye(5)
        matrix[:3,:3]=np.linalg.solve(axes,local)
        return matrix
    ratio=mapper.g.cfg['fingers'][f]['dip_pip_ratio']
    return np.array([[mapper.side_sign,0,0,0],[0,1,0,0],[0,0,1/(1+ratio**2),ratio/(1+ratio**2)]])


def fit(manifest):
    check_split(manifest)
    groups=load_groups(manifest,'train')
    opens=[frames[0] for e,frames in groups if e['stage']=='open']
    if len(opens)!=1 or len(groups)<5: raise ValueError('需要一个张开参考及多组独立动作')
    source=opens[0]
    cfg=official_config(load_g7_config());g=HandGeometry('simulation_straight')
    reference=np.radians(g.simulation_profile['initial_coordinates_deg'])
    mapper=G7DirectJointRetargeter(g,reference,config=dict(cfg,thumb_linear_assist_enabled=False),open_frame=source)
    teacher=ContactGraphAssist(g,cfg,reference)
    features={f:[] for f in FINGERS}; labels=[]; stages=[]; weights=[]
    for entry,frames in groups:
        for frame in frames:
            angles=mapper.decode(frame)
            for f in FINGERS: features[f].append(joint_features(angles,f))
            # Geometric calibration labels are computed offline; optional reviewer-supplied
            # labels are named, complete, coupled-coordinate poses from the simulator.
            if 'f1_deg' in entry:
                if len(frames)!=1 or set(entry['f1_deg'])!=set(g.labels):
                    raise ValueError('人工标注需给定单个姿态及全部17个按名标记的角度')
                target=g.validate(np.radians([entry['f1_deg'][name] for name in g.labels]))
                if not teacher.clearance.safe(target): raise ValueError('人工标注违反拇指间隙')
            else:
                mapper.previous=reference.copy()
                prior=mapper.solve_frame(frame)['direct_joint_rad']
                task=teacher.targets(frame)
                def objective(u):
                    r,j=teacher.system(u,prior,task,{})
                    return .5*float(r@r),j.T@r
                sol=minimize(objective,prior,jac=True,bounds=list(zip(g.lower,g.upper)),
                             method='L-BFGS-B',options={'maxiter':48,'ftol':1e-8})
                if not np.isfinite(sol.fun): raise ValueError('离线形态拟合无效')
                target,_=teacher.clearance.sweep_follow(reference,g.validate(sol.x))
            labels.append(target);stages.append(entry['stage'])
            # Weight each static pose once, a motion sequence at most six poses.
            weights.append(min(1.,6./len(frames)))
    labels=np.array(labels);w=np.sqrt(weights)[:,None]
    record=dict(version=1,hand='right',units='rad',labels=g.labels,channels=CHANNELS,
                geometry_id=geometry_id(g,reference),open_frame=source,blocks={},
                train_manifest=str(manifest.resolve()),train_samples=len(labels),stages=stages)
    for f in FINGERS:
        zero=joint_features(mapper.decode(source),f);x=np.array(features[f])-zero
        y=labels[:,g.slices[f]]
        prior=initial_matrix(f,mapper)
        # Ridge regression on physically interpretable rad/rad coefficients.
        # The open example determines the bias exactly.
        bias=y[stages.index('open')]
        A=x*w;B=(y-bias-x@prior.T)*w
        rank=int(np.sum(np.linalg.svd(A,compute_uv=False)>.2))
        if rank<2: raise ValueError('动作变化不足，无法辨识 '+f+'，请重采独立屈伸和侧摆动作')
        correction=np.linalg.solve(A.T@A+.15*np.eye(A.shape[1]),A.T@B)
        record['blocks'][f]=dict(zero=zero,bias=bias,matrix=prior+correction.T,
            span_deg=np.degrees(np.ptp(x,axis=0)),excited_rank=rank)
    AngleProfile(record,g,reference,'right')
    return record


def evaluate(manifest, profile):
    check_split(manifest)
    from calibrated_angle_mapping import CalibratedAngleRetargeter
    cfg=official_config(load_g7_config());g=HandGeometry('simulation_straight')
    ref=np.radians(g.simulation_profile['initial_coordinates_deg'])
    profile_data=json.loads(profile.read_text(encoding='utf8'))
    cal=CalibratedAngleRetargeter(g,ref,'right',cfg,profile)
    unit=G7DirectJointRetargeter(g,ref,config=dict(cfg,thumb_linear_assist_enabled=False),open_frame=profile_data['open_frame'])
    # Both baselines use the entire selected model range and identical start and guard.
    unit.lower,unit.upper=g.lower.copy(),g.upper.copy()
    metric=ContactGraphAssist(g,cfg,ref);rows=[]
    for entry,frames in load_groups(manifest,'validation'):
        cal.previous=ref.copy();unit.previous=ref.copy()
        for frame in frames:
            targets=metric.targets(frame)
            for name,solver in [('unit',unit),('calibrated',cal)]:
                out=solver.solve_frame(frame);q=g.validate(out['goal_rad'])
                assert metric.clearance.safe(q)
                m=metric.metrics(q,targets)
                rows.append(dict(stage=entry['stage'],method=name,seq=frame.get('seq'),
                    direction_deg=float(np.mean(list(m['bone_errors_deg'].values()))),
                    bend_deg=float(np.mean(np.abs(list(m['bend_errors_deg'].values())))),
                    relative_tip_mm=float(np.mean(list(m['relative_errors_mm'].values()))),
                    strong_thumb_gaps_mm=[e['surface_gap_mm'] for e in m['contact_edges'] if e['a']=='thumb' and e['weight']>.8],
                    path_blocked=out['thumb_avoidance']['blocked']))
    if not rows: raise ValueError('缺少独立验证记录')
    summary={}
    for name in ('unit','calibrated'):
        r=[v for v in rows if v['method']==name]
        summary[name]={k:float(np.mean([v[k] for v in r])) for k in ('direction_deg','bend_deg','relative_tip_mm')}
        gaps=[d for v in r for d in v['strong_thumb_gaps_mm']]
        summary[name].update(frames=len(r),strong_pairs=len(gaps),strong_gap_mm=float(np.mean(gaps)) if gaps else None)
    return dict(summary=summary,records=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    capture=sub.add_parser('record');capture.add_argument('folder',type=Path);capture.add_argument('--resume',action='store_true')
    for name in ('fit','evaluate'):
        action=sub.add_parser(name);action.add_argument('manifest',type=Path);action.add_argument('profile',type=Path)
        if name=='evaluate':action.add_argument('report',type=Path)
    args=parser.parse_args()
    if args.command=='record':record(args.folder,args.resume)
    elif args.command=='fit':write(args.profile,fit(args.manifest))
    else:
        result=evaluate(args.manifest,args.profile);write(args.report,result);print(json.dumps(result['summary'],indent=2))
