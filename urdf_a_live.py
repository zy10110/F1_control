"""Display the personal URDF-A right hand and the wrist-relative glove skeleton."""
import json
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

    def draw(self,m):
        g=self.g
        self.renderer.update_scene(g.data)
        scene=self.renderer.scene
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
                chain=wrist+m.chains[f]@m.display_rotation.T*.001
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
    def __init__(self,profile,port,guide=False,resume=None,input_hand='right'):
        self.input_hand=input_hand
        self.hand_name='左手' if input_hand=='left' else '右手'
        if json.loads(Path(profile).read_bytes()).get('input_hand','right')!=input_hand:
            raise ValueError('输入手别与档案不符：左手镜像和右手各用独立的标定档案')
        self.g=NativeHand(ROOT/'assets/urdf_a');self.follower=Follower(self.g,profile)
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
        self.receiver=OfficialUDPReceiver(decoder,input_hand,port=port)
        self.thread=threading.Thread(target=self.receive,name='F1-A official UDP',daemon=True)
        self.thread.start()
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.after(0,self.tick)
        if input_hand=='left' and not self.follower.middle_unavailable:
            tk.Button(self.root,text='用官方手模标定（八个动作）',command=lambda:self.open_guide(official=True)).pack(pady=(0,6))
        tk.Button(self.root,text='补录自然动作',command=self.open_guide).pack(pady=(0,6))
        tk.Button(self.root,text='对指专项标定',command=lambda:self.open_guide(True)).pack(pady=(0,6))
        if guide:self.root.after(200,lambda:self.open_guide(guide=='pinch'))

    def open_guide(self,pinch=False,official=False):
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
        source=self.latest;m=self.follower
        fresh=source is not None and now-source[0]<m.config['source_timeout_s'] and not self.error
        try:
            if fresh:
                if self.last_seen!=source[0]:
                    m.command(source[1]);self.last_seen=source[0];self.dirty=True
                before=m.u.copy();m.advance(dt)
                self.dirty|=np.max(abs(before-m.u))>1e-8
            else:m.vel=None
            if self.dirty:
                self.photo=ImageTk.PhotoImage(self.view.draw(m))
                self.canvas.config(image=self.photo);self.dirty=False
            if now-self.last_status>.25:
                self.last_status=now
                state=self.error or ('跟随中' if fresh else ('数据超时，保持当前姿态' if source else f'等待 HumanHand 发送{self.hand_name}数据'))
                detail='中指传感器停用 · 中指固定' if m.middle_unavailable else f'中指 MCP / PIP  {np.degrees(m.u[9]):.1f}° / {np.degrees(m.u[10]):.1f}°'
                self.status.set(f'{state}    127.0.0.1:{self.port}    {detail}')
                distances={}
                if fresh and m.chains is not None:
                    from g7_teleop import BASE
                    points=np.asarray(source[1]['positions_m'])
                    for f in FINGERS[1:]:
                        if f=='middle' and m.middle_unavailable:continue
                        distances[f]=(float(np.linalg.norm(points[4]-points[BASE[f]+3])*1000),
                                      float(np.linalg.norm(self.g.pad_pose('thumb')[0]-self.g.pad_pose(f)[0])))
                self.spacing_status.set(('距拇指 · 手套指尖 / F1 指腹 (mm)     '+ '     '.join(
                    f'{NAMES[f]} {s:.1f} / {d:.1f}' for f,(s,d) in distances.items())) if distances else '')
                record=dict(time=time.time(),mode='urdf_a',input_hand=self.input_hand,model_hand='right',profile=str(self.profile),geometry_id=self.g.geometry_id,
                            loaded_profile_sha256=m.profile_sha256,source_frame=source[1] if source else None,
                            fresh=bool(fresh),state=state,port=self.port,receiver=self.receiver.stats(),
                            actual_deg=np.degrees(m.u).tolist(),target_deg=np.degrees(m.goal).tolist(),
                            retargeting=('glove_angle_pad_spacing' if m.spacing else 'calibrated_joint_angle_context' if m.angle_context is not None else 'independent_finger_angles'),
                            glove_tip_f1_pad_mm=distances,clearance_mm=self.g.clearance(),sweep_blocked=m.blocked,
                            middle_unavailable=m.middle_unavailable,seq=source[1]['seq'] if source else None)
                tmp=self.path.with_suffix('.tmp')
                tmp.write_text(json.dumps(record,ensure_ascii=False),encoding='utf8');tmp.replace(self.path)
        except (ValueError,RuntimeError,OSError) as exc:
            self.status.set('仿真异常：'+str(exc))
        self.root.after(max(1,round(25-1000*(time.monotonic()-now))),self.tick)

    def close(self):
        self.stop.set();self.thread.join(1)
        self.view.renderer.close();self.root.destroy()


def launch(profile,port=15001,guide=False,resume=None,input_hand='right'):
    LiveWindow(profile,port,guide,resume,input_hand).root.mainloop()

