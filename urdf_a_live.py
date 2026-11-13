"""Display the personal URDF-A right hand and the wrist-relative glove skeleton."""
import json
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageTk
from scipy.spatial.transform import Rotation

from g7_official_udp import OfficialUDPReceiver
from g7_teleop import load_reader
from hand_kinematics import ROOT, FINGERS
from urdf_a_model import NativeHand
from urdf_a_follow import Follower


COLORS = [(1.,.63,.28,1.),(.21,.78,.83,1.),(.36,.64,.96,1.),(.67,.51,.88,1.),(.92,.5,.68,1.)]
NAMES = dict(zip(FINGERS,('拇指','食指','中指','无名指','小指')))


class View:
    def __init__(self,g):
        self.g=g
        m=g.model
        m.vis.global_.offwidth=880;m.vis.global_.offheight=540;m.vis.quality.offsamples=1
        for f,color in zip(FINGERS,COLORS):
            m.geom_rgba[np.isin(m.geom_bodyid,g.chains[f])]=color
        for name in ('base','r_wrist_abd'):
            try:m.geom_rgba[m.geom_bodyid==m.body(name).id,3]=0
            except KeyError:pass
        # The importer fuses the static base into world.
        m.geom_rgba[m.geom_bodyid==0,3]=0
        self.renderer=mujoco.Renderer(m,height=540,width=880,max_geom=1000)
        self.origin=g.data.xpos[g.palm].copy()
        self.yaw=.10;self.pitch=.12;self.distance=.37
        self.last_mouse=None
        self.display_data=g.control_data
        self.wrist_qadr=g.wrist_qadr
        self.roll_pivot=g.roll_pivot

    def draw(self,m,wrist_angles=None,*,use_control=False):
        g=self.g
        if wrist_angles is not None:
            # Support standalone replay viewers with no live control loop.
            digits=np.linalg.lstsq(g.C,g.data.qpos,rcond=None)[0]
            g.set_control_pose(digits,wrist_angles)
            use_control=True
        if use_control:
            mujoco.mj_forward(g.model,g.control_data)
        self.renderer.update_scene(g.control_data if use_control else g.data)
        scene=self.renderer.scene
        palm_rotation=np.eye(3)
        if use_control:
            roll=g.base_rotation
            for geom in scene.geoms[:scene.ngeom]:
                geom.pos[:]=roll@geom.pos+g.base_translation_m
                geom.mat[:]=(roll@geom.mat.reshape(3,3)).reshape(geom.mat.shape)
            palm_rotation=roll@self.display_data.xmat[g.palm].reshape(3,3)@g.data.xmat[g.palm].reshape(3,3).T
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
        orbit=Rotation.from_rotvec([self.yaw,0,0])*Rotation.from_rotvec([0,0,self.pitch])
        forward=orbit.apply([0.,-1.,0.]);up=orbit.apply([1.,0.,0.])
        right=np.cross(forward,up)
        center=self.origin+np.array([.09,0.,0.])-.085*right
        for cam in scene.camera:
            cam.pos[:]=center-forward*self.distance;cam.forward[:]=forward;cam.up[:]=up
            cam.orthographic=1
            half_height=self.distance*np.tan(np.radians(g.model.vis.global_.fovy/2))
            cam.frustum_top=half_height;cam.frustum_bottom=-half_height
        # The scene was built before we substituted the palm-facing camera.
        # Aim its headlight in the same frame so the visible pads are lit.
        for light in scene.lights[:scene.nlight]:
            light.pos[:]=center-forward*self.distance
            light.dir[:]=forward
            light.ambient[:]=[.55,.55,.55]
            light.diffuse[:]=[.65,.65,.65]
        def sphere(point,r,color):
            geom=scene.geoms[scene.ngeom];scene.ngeom+=1
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,r),point,np.eye(3).ravel(),np.array(color))
        def line(a,b,r,color):
            geom=scene.geoms[scene.ngeom];scene.ngeom+=1
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),a,np.eye(3).ravel(),np.array(color))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,r,a,b)
        if m.chains is not None:
            wrist=self.origin-.17*right
            for f,color in zip(FINGERS,COLORS):
                if f=='middle' and m.middle_unavailable:continue
                chain=wrist+m.chains[f]@m.display_rotation.T@palm_rotation.T*.001
                previous=wrist
                for point in chain:
                    line(previous,point,.0014,color)
                    sphere(point,.0027,[1.,.77,.32,1.]);previous=point
        return Image.fromarray(self.renderer.render())


def four_finger_decoder(reader,packet):
    # For FSNH v4, 22 poses prepend a forearm to the 21 wrist-root poses.
    # Neutralize just the missing chain before checking numeric validity.
    # The vendor decoder still validates header, hand, length and all other poses.
    if len(packet)>=reader.HEADER.size:
        count=reader.HEADER.unpack_from(packet)[3]
        if count in (21,22) and len(packet)==reader.HEADER.size+count*reader.JOINT.size:
            start=reader.HEADER.size+(count-21)*reader.JOINT.size
            size=reader.JOINT.size
            packet=packet[:start+9*size]+packet[start:start+size]*4+packet[start+13*size:]
    return dict(reader.decode_udp(packet),unavailable_fingers=['middle'])


class LiveWindow:
    def __init__(self,profile,port,guide=False,resume=None,input_hand='right',sdk_port=None):
        self.input_hand=input_hand
        self.hand_name='左手' if input_hand=='left' else '右手'
        profile_data=json.loads(Path(profile).read_bytes())
        if profile_data.get('input_hand','right')!=input_hand:
            raise ValueError('输入手别与档案不符：左手镜像和右手各用独立的标定档案')
        self.overlap_limit_mm=profile_data.get('sim_overlap_limit_mm',1.5) if input_hand=='right' else None
        self.g=NativeHand(ROOT/'assets/urdf_a')
        self.follower=Follower(self.g,profile,overlap_limit_mm=self.overlap_limit_mm)
        self.resume=resume
        self.root=tk.Tk();self.root.title(f'F1 灵巧手 · {self.hand_name}手套 → URDF-A 右手仿真')
        self.root.configure(bg='#182631')
        self.view=View(self.g)
        if self.follower.middle_unavailable:
            self.g.model.geom_rgba[np.isin(self.g.model.geom_bodyid,self.g.chains['middle'])]=[.45,.46,.48,1.]
        self.guide=None
        self.status=tk.StringVar(value=f'等待{self.hand_name}关节位姿')
        self.spacing_status=tk.StringVar(value='')
        title=tk.Label(self.root,text=('左手手套（镜像）' if input_hand=='left' else '右手手套')+'                         F1 · URDF-A 右手',bg='#182631',fg='#eef4f5',font=('Microsoft YaHei UI',14))
        title.pack(fill='x',pady=(8,2))
        self.canvas=tk.Label(self.root,bg='#182631',borderwidth=0)
        self.canvas.pack()
        tk.Label(self.root,textvariable=self.status,bg='#182631',fg='#a9e6dd',font=('Microsoft YaHei UI',10)).pack(pady=7)
        tk.Label(self.root,textvariable=self.spacing_status,bg='#182631',fg='#e0e8ec',font=('Microsoft YaHei UI',10)).pack()
        self.canvas.bind('<ButtonPress-1>',self.pick);self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<MouseWheel>',self.zoom)
        self.stop=threading.Event();self.latest=None;self.error=None
        self.dirty=True;self.last_seen=None;self.last_status=0.;self.prev=time.monotonic()
        self.port=port;self.profile=Path(profile).resolve()
        self.path=ROOT/'results'/('urdf_a_live_status.json' if port==15001 else f'urdf_a_status_{port}.json')
        reader=load_reader(self.follower.config)
        decoder=(lambda p:four_finger_decoder(reader,p)) if self.follower.middle_unavailable else reader.decode_udp
        from urdf_a_wrist import WristMotion
        self.wrist=WristMotion(self.g.model,self.follower.display_rotation,input_hand)
        self.last_wrist_angles=np.zeros(3)
        self.receiver=OfficialUDPReceiver(decoder,input_hand,port=port,imu_sink=self.wrist.accept)
        tk.Button(self.root,text='当前手姿设为腕部零位',command=self.wrist.reset).pack(pady=(0,4))
        self.thread=threading.Thread(target=self.receive,name='F1-A official UDP',daemon=True)
        self.thread.start()
        self.sdk_process=None
        self.sdk_error=None
        self.sdk_log=ROOT/'results'/f'g7_sdk_{port}.log'
        self.sdk_port=sdk_port
        self.sdk_state={}
        self.sdk_pending=False
        if sdk_port:
            row=tk.Frame(self.root)
            row.pack(pady=(0,4))
            self.sdk_button=tk.Button(row,text=f'连接 SDK + IMU（{sdk_port}）',command=self.connect_sdk)
            self.sdk_button.pack(side='left')
            self.vendor_menu=tk.Menubutton(row,text='G7 官方标定 ▾',relief='raised')
            menu=tk.Menu(self.vendor_menu,tearoff=False)
            menu.add_command(label='普通手型标定（4 个动作）',command=lambda:self.sdk_command('general'))
            menu.add_command(label='拇指对指标定（依次轻贴四指）',command=lambda:self.sdk_command('pinch'))
            menu.add_separator()
            menu.add_command(label='取消本次标定',command=lambda:self.sdk_command('cancel'))
            self.vendor_menu.configure(menu=menu)
            self.vendor_menu.pack(side='left')
            self.sdk_note=tk.StringVar()
            tk.Label(row,textvariable=self.sdk_note).pack(side='left')
            self.sdk_state_path=ROOT/'results'/f'g7_sdk_state_{port}.json'
            self.connect_sdk()
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.after(0,self.tick)
        if input_hand=='left' and not self.follower.middle_unavailable:
            tk.Button(self.root,text='用官方手模标定（八个动作）',command=lambda:self.open_guide(official=True)).pack(pady=(0,6))
        tk.Button(self.root,text='补录自然动作',command=self.open_guide).pack(pady=(0,6))
        tk.Button(self.root,text='对指专项标定',command=lambda:self.open_guide(True)).pack(pady=(0,6))
        if guide:self.root.after(200,lambda:self.open_guide(guide=='pinch'))

    def connect_sdk(self):
        if self.sdk_process is not None:
            if self.sdk_process.poll() is None:
                return
            self.sdk_process.stdin.close()
        self.sdk_error=None
        self.sdk_state={}
        self.sdk_pending=False
        self.sdk_state_path.unlink(missing_ok=True)
        self.sdk_note.set('等待 SDK')
        with self.sdk_log.open('wb') as log:
            self.sdk_process=subprocess.Popen(
                [sys.executable,str(ROOT/'g7_imu_bridge.py'),'--port',self.sdk_port,
                 '--hand',self.input_hand,'--udp-port',str(self.port),'--parent-pipe'],cwd=ROOT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),stdin=subprocess.PIPE,stdout=log,stderr=log)
        self.sdk_button.configure(state='disabled')

    def sdk_command(self,mode):
        process=self.sdk_process
        if process is None or process.poll() is not None or self.sdk_state.get('connection')!='open':
            self.sdk_note.set('请先连接 SDK')
            return
        busy=self.sdk_pending or self.sdk_state.get('mode')
        if (mode=='cancel')!=bool(busy):return
        if mode!='cancel' and self.guide is not None:
            self.sdk_note.set('请先结束当前 F1 动作采集')
            return
        try:
            process.stdin.write((mode+'\n').encode('ascii'))
            process.stdin.flush()
            self.sdk_pending=True
            self.sdk_note.set('等待 SDK 标定反馈')
        except OSError:
            self.sdk_note.set('SDK 连接已中断')

    def read_sdk_state(self):
        if self.sdk_process.poll() is not None:
            self.sdk_pending=False
            self.sdk_state={}
            self.sdk_note.set('SDK 已断开')
            return
        try:
            state=json.loads(self.sdk_state_path.read_bytes())
        except (OSError,ValueError):
            return
        if state==self.sdk_state:return
        self.sdk_pending=False
        self.sdk_state=state
        if state.get('connection')=='retry':
            self.sdk_note.set(state['reason']+' · 正在自动重连')
        elif state.get('mode'):
            from g7_calibration import GESTURES
            phase=state['phase']
            gesture=state['gesture']
            name=GESTURES[gesture-1][0] if 1<=gesture<=len(GESTURES) else ''
            label={1:'准备',2:'采集',3:'重录',4:'计算手型',5:'保存手型'}.get(phase,'标定')
            self.sdk_note.set(f"{state['step']}/{state['total']} {name} · {label} {state['seconds']} s")
        else:
            settings=state['settings']
            pinch='开' if settings['pinch_enabled'] else '关'
            filt='开' if settings['traj_filter'] else '关'
            self.sdk_note.set(f"{state.get('result','SDK 已连接')} {state.get('serial_port','')} · 捏合{pinch} · 轨迹滤波{filt}")

    def open_guide(self,pinch=False,official=False):
        if self.sdk_pending or self.sdk_state.get('mode'):return
        if self.guide is None:
            from urdf_a_capture import CaptureGuide
            self.guide=CaptureGuide(self,pinch,official)
        else:self.guide.window.lift()

    def pick(self,e):self.view.last_mouse=(e.x,e.y)

    def drag(self,e):
        x,y=self.view.last_mouse
        self.view.yaw+=(e.x-x)*.008;self.view.pitch=np.clip(self.view.pitch+(e.y-y)*.008,-1.3,1.3)
        self.pick(e);self.dirty=True

    def zoom(self,e):
        self.view.distance=np.clip(self.view.distance*(.90 if e.delta>0 else 1.1),.22,.85);self.dirty=True

    def receive(self):
        def publish(frame):
            if self.input_hand=='left':
                from g7_left_mirror import canonical_right
                frame=canonical_right(frame)
            stamp=time.monotonic()
            self.latest=(stamp,frame)
            guide=self.guide
            if guide is not None:guide.receive(stamp,frame)
        try:self.receiver.receive(self.stop,publish)
        except Exception as exc:self.error=str(exc)

    def tick(self):
        now=time.monotonic();dt=now-self.prev;self.prev=now
        if self.sdk_process and self.sdk_process.poll() is not None and not self.sdk_error:
            self.sdk_error='SDK 串口未连接，请释放串口后重连。详情见 '+self.sdk_log.name
            self.sdk_button.configure(state='normal')
        source=self.latest;m=self.follower
        fresh=source is not None and now-source[0]<m.config['source_timeout_s'] and not self.error
        try:
            if fresh:
                if self.last_seen!=source[0]:
                    m.command(source[1]);self.last_seen=source[0];self.dirty=True
            else:
                m.vel=None
                m.contact_sample=None
                m.pair_hold.reset()
                m.approach_sample=None
                m.approach_weight=0.
                m.goal_velocity=None
                m.goal_stamp=None
            angles,wrist_status=self.wrist.snapshot()
            self.dirty |= not np.array_equal(self.last_wrist_angles,angles)
            self.last_wrist_angles=angles
            before=m.u.copy()
            m.control_step(dt,angles,fingers_fresh=fresh)
            self.dirty |= np.max(abs(before-m.u))>1.e-8
            if self.dirty:
                self.photo=ImageTk.PhotoImage(self.view.draw(m,use_control=True))
                self.canvas.config(image=self.photo);self.dirty=False
            if now-self.last_status>.25:
                self.last_status=now
                if self.sdk_port:self.read_sdk_state()
                state=self.error or ('IMU 翻腕中 · 等待五指关节流' if wrist_status['fresh'] and not fresh else None) or (self.sdk_error if not fresh else None) or ('跟随中' if fresh else ('数据超时，保持当前姿态' if source else f'等待{self.hand_name}手套数据'))
                if not fresh and self.sdk_state.get('connection')=='retry':
                    state=f'等待{self.hand_name}手套响应 · 自动重连中'
                detail='中指传感器停用 · 中指固定' if m.middle_unavailable else f'中指 MCP / PIP  {np.degrees(m.u[9]):.1f}° / {np.degrees(m.u[10]):.1f}°'
                precision=bool(m.spacing and m.spacing.get('aperture_range_mm'))
                if precision:detail='精细跟随 · '+detail
                contact=None
                if m.joint_touch:
                    if fresh and m.touch is not None:
                        touch=m.touch
                        gap,_,points=m.joint_touch.surface(touch)
                        anchor,axis,normal,_=m.joint_touch.landmarks(touch)
                        offset=points[3:]*1000.-anchor
                        contact=dict(finger=touch.finger,segment=touch.segment,fraction=touch.fraction,
                                     signed_gap_mm=gap,along_error_mm=float(axis@offset),
                                     source_weight=touch.weight)
                        segment=('近节','中节','末节')[touch.segment]
                        detail=f'{NAMES[touch.finger]}{segment} {touch.fraction:.0%} · 壳面间距 {gap:.2f} mm · 轴向偏差 {axis@offset:+.1f} mm'
                    else:detail='指节触摸 · '+detail
                if fresh and m.segment_alignment and m.segment_alignment.thumb_bend_plane:
                    spin, splay = np.degrees(m.u[1:3])
                    detail += f'    拇指 CMC 旋转 {spin:+.1f}° · MCP 侧摆 {splay:+.1f}°'
                self.status.set(f'{state}    127.0.0.1:{self.port}    {detail}')
                distances={}
                if fresh and m.chains is not None:
                    from g7_teleop import BASE
                    points=np.asarray(source[1]['positions_m'])
                    for f in FINGERS[1:]:
                        if f=='middle' and m.middle_unavailable:continue
                        distances[f]=(float(np.linalg.norm(points[4]-points[BASE[f]+3])*1000),
                                      float(np.linalg.norm(self.g.pad_pose('thumb')[0]-self.g.pad_pose(f)[0])))
                held_pairs=[(FINGERS[i],FINGERS[j]) for i,j in zip(*np.triu(m.pair_hold.links,1).nonzero())
                            if m.pair_hold.locked[i] and m.pair_hold.locked[j]]
                spacing='距拇指 · 手套指尖 / F1 指腹 (mm)     '+'     '.join(
                    f'{NAMES[f]} {s:.1f} / {d:.1f}' for f,(s,d) in distances.items()) if distances else ''
                if held_pairs:
                    spacing+='\n指对保持 · '+'、'.join(f'{NAMES[a]}–{NAMES[b]}' for a,b in held_pairs)
                roll,abd,flex=wrist_status['angles_deg']
                source_label=('前臂参考 + IMU' if wrist_status['forearm_reference'] else '手背 IMU') if wrist_status['fresh'] else ('IMU 超时，保持' if wrist_status['packets'] else '等待 IMU')
                if self.sdk_error:source_label=f'等待 SDK 串口 {self.sdk_port}'
                self.spacing_status.set(spacing+f'\n{source_label} · 前臂翻转 {roll:+.1f}° · 腕侧摆 {abd:+.1f}° · 腕屈伸 {flex:+.1f}°')
                record=dict(time=time.time(),mode='urdf_a',input_hand=self.input_hand,model_hand='right',profile=str(self.profile),geometry_id=self.g.geometry_id,
                            loaded_profile_sha256=m.profile_sha256,source_frame=source[1] if source else None,
                            fresh=bool(fresh),state=state,port=self.port,receiver=self.receiver.stats(),
                            wrist=wrist_status,g7_sdk=self.sdk_state,
                            contact_mode='overlap_stop' if m.overlap_stop else 'surface_guard',
                            pinch_overlap_limit_mm=m.overlap_stop.limit_mm if m.overlap_stop else m.sweep.overlap_mm,
                            stopped_axes=np.flatnonzero(m.overlap_stop.stopped).tolist() if m.overlap_stop else [],
                            f1_control=dict(joint_names=self.g.control_joint_names,
                                position_rad=self.g.control_data.qpos.tolist(),forearm_roll_rad=self.g.forearm_roll,
                                base_quat_wxyz=self.g.base_quat_wxyz.tolist(),base_translation_m=self.g.base_translation_m.tolist(),
                                fingers_fresh=bool(fresh),finger_seq=source[1]['seq'] if source else None,
                                imu_fresh=wrist_status['fresh'],imu_seq=wrist_status['seq'],
                                finger_timestamp_ns=source[1].get('timestamp_ns') if source else None,
                                imu_timestamp_ns=wrist_status['timestamp_ns']),
                            actual_deg=np.degrees(m.u).tolist(),target_deg=np.degrees(m.goal).tolist(),
                            thumb_bend_plane_enabled=bool(m.segment_alignment and m.segment_alignment.thumb_bend_plane),
                            contact_stabilization_enabled=bool(m.joint_touch and m.response),
                            official_vector_tracking=bool(m.spacing and m.joint_touch and m.response and precision),
                            contact_filter_active=m.contact_sample is not None,
                            pair_stability_enabled=m.sweep.long_surfaces is not None and m.overlap_stop is None,
                            held_pairs=held_pairs,
                            retargeting=('precision_aperture' if precision else 'glove_angle_pad_spacing' if m.spacing else 'calibrated_joint_angle_context' if m.angle_context is not None else 'independent_finger_angles'),
                            joint_surface_enabled=m.joint_touch is not None,joint_surface=contact,
                            glove_tip_f1_pad_mm=distances,clearance_mm=self.g.clearance(),sweep_blocked=m.blocked,
                            middle_unavailable=m.middle_unavailable,seq=source[1]['seq'] if source else None)
                tmp=self.path.with_suffix('.tmp')
                tmp.write_text(json.dumps(record,ensure_ascii=False),encoding='utf8');tmp.replace(self.path)
        except (ValueError,RuntimeError,OSError) as exc:
            self.status.set('仿真异常：'+str(exc))
        self.root.after(max(1,round(25-1000*(time.monotonic()-now))),self.tick)

    def close(self):
        if self.sdk_process and self.sdk_process.poll() is None:
            self.sdk_process.stdin.close()
            try:self.sdk_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.sdk_process.terminate()
                self.sdk_process.wait(timeout=3)
        self.stop.set();self.thread.join(1)
        self.view.renderer.close();self.root.destroy()


def launch(profile,port=15001,guide=False,resume=None,input_hand='right',sdk_port=None):
    LiveWindow(profile,port,guide,resume,input_hand,sdk_port).root.mainloop()

