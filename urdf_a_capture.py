"""Guided supplemental capture sharing the live window's accepted UDP stream."""
import json
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk

import numpy as np
from g7_official_udp import open_reference
from hand_kinematics import ROOT
from tools.calibrate_angle_map import write


def approach(finger):
    return f'只用拇指向{finger}指腹慢慢靠近。留约一厘米，停一秒；再轻碰一下，沿着指腹侧面划开，回到张开。其余三指自然放松。'


TAKES=[
    ('train','rest_again','自然张开',2,'右手手腕放松，掌心朝前。五指自然伸开，按回车后保持原姿势。'),
    ('train','middle_slow','中指单独弯伸',7,'拇指放松停在掌边。只将中指慢慢弯向掌心，再慢慢伸直。全程做两遍，其他三根长指尽量保持原姿势。'),
    *[('train',f+'_approach',name+'与拇指',8,approach(name)) for f,name in [('index','食指'),('middle','中指'),('ring','无名指'),('pinky','小指')]],
    ('validation','near_pass','验证一 · 留缝与擦过',10,'先让拇指靠近食指，留一条缝，停一秒后退开。再让拇指从中指指尖旁边慢慢划过，回到张开。两次都无需碰到。'),
    ('validation','touch_release','验证二 · 自然轻碰、松开',12,'从张开开始，拇指依次轻碰食指、中指、无名指、小指。每碰一次就自然松开，最后五指张开。')
]


class CaptureGuide:
    def __init__(self, app, pinch=False,official=False):
        self.pinch=pinch
        self.official=official
        self.app=app;self.window=tk.Toplevel(app.root)
        self.window.title('A 版自然手形 · 补录标定')
        self.window.geometry('720x310+45+55')
        self.window.attributes('-topmost',True)
        self.index=0;self.start=None;self.end=None;self.rows=[];self.issue=None
        self.lock=threading.Lock();self.job=None;self.alive=True;self.validation_notes=[]
        self.folder=app.resume or ROOT/'calibration'/('natural_supplement_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.manifest=dict(train=[],validation=[],base_session=str(app.profile.parent),completed=False)
        self.takes=list(TAKES)
        if app.follower.middle_unavailable:
            self.takes=[t for t in self.takes if t[1] not in ('middle_slow','middle_approach','near_pass','touch_release')]
            self.takes.insert(1,('train','thumb_sweep','拇指独立活动',7,'四根长指放松。拇指先向手掌侧面展开，再慢慢从掌心前方扫向小指根，最后退回。只到舒服的位置。'))
            self.takes.extend([
                ('validation','near_pass','验证一 · 留缝与擦过',10,'让拇指靠近食指，留一条缝，停一秒后退开。再从无名指指尖旁边慢慢划过，回到张开。'),
                ('validation','touch_release','验证二 · 自然轻碰、松开',10,'从张开开始，拇指依次轻碰食指、无名指、小指。每次都自然松开，最后张开。中指一直放松。')])
        if pinch:
            self.folder=app.resume or ROOT/'calibration'/('opposition_round_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.window.title('A 版 · 对指专项标定')
            self.window.geometry('760x490+45+40')
            self.diagram=tk.Label(self.window)
            self.diagram.pack()
            fingers=[('index','食指'),('middle','中指'),('ring','无名指'),('pinky','小指')]
            if app.follower.middle_unavailable:fingers=[item for item in fingers if item[0]!='middle']
            self.takes=[('train',f,'拇指与'+name+' · 保持',2,
                f'照图让拇指尖与{name}指尖相对，轻轻贴住，指腹不要压扁。摆好后用另一只手按回车，保持到采集结束。') for f,name in fingers]
            self.takes += [('validation',f,'拇指与'+name+' · 重做验证',8,
                f'先自然张开。采集开始后，用约两秒靠近{name}，指尖轻贴并停住约三秒，再慢慢完全松开。') for f,name in fingers]
            self.manifest['method']='official_pinch_holds'
        if official:
            from tools.official_pose_calibration import TAKES as official_takes
            self.takes=list(official_takes)
            self.folder=ROOT/'calibration'/('official_pose_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.window.title('用官方手模标定 F1')
            self.manifest['method']='solved_official_poses'
        if app.resume is not None and not official:
            self.manifest=json.loads((self.folder/'manifest.json').read_text(encoding='utf8'))
            wanted={t[1] for t in self.takes}
            for split in ('train','validation'):
                self.manifest[split]=[e for e in self.manifest[split] if e['stage'] in wanted]
            app.resume=None
        self.manifest['unavailable_fingers']=['middle'] if app.follower.middle_unavailable else []
        self.manifest['base_session']=str(app.profile.parent)
        if self.manifest.get('input_hand','right')!=app.input_hand and self.manifest.get('train'):
            raise ValueError('补录手别与当前手套不一致')
        self.manifest['input_hand']=app.input_hand
        completed={e['stage'] for split in ('train','validation') for e in self.manifest[split]}
        self.index=next((i for i,t in enumerate(self.takes) if not any(e['stage']==t[1] for e in self.manifest[t[0]])),len(self.takes))
        write(self.folder/'manifest.json',self.manifest)
        if not (self.folder/'baseline_profile.json').exists():
            (self.folder/'baseline_profile.json').write_bytes(app.profile.read_bytes())
        self.heading=tk.StringVar();self.instruction=tk.StringVar();self.status=tk.StringVar()
        for var,size in ((self.heading,16),(self.instruction,12),(self.status,11)):
            tk.Label(self.window,textvariable=var,font=('Microsoft YaHei UI',size),wraplength=660,justify='left',
                     anchor='w').pack(fill='x',padx=22,pady=8)
        self.button=tk.Button(self.window,text='开始本项（回车）',command=self.begin)
        self.button.pack(anchor='w',padx=22)
        self.window.bind('<Return>',lambda _:self.begin());self.window.protocol('WM_DELETE_WINDOW',self.close)
        if self.index==len(self.takes):
            self.heading.set('正在验证已采集的动作')
            self.button.config(state='disabled')
            self.job=threading.Thread(target=self.calibrate,daemon=True);self.job.start()
        else:self.show()
        self.window.lift();self.window.focus_force();self.poll()

    def close(self):
        if self.job is not None:
            self.status.set('正在验证，请等待本轮结束。');return
        self.alive=False;self.app.guide=None;self.window.destroy()

    def show(self):
        split,key,name,duration,instruction=self.takes[self.index]
        if self.pinch:
            from PIL import Image,ImageTk
            paths=list(Path('E:/HumanHand/assets/factory/calib_guide').glob('pinch_thumb_'+key+'.*'))
            if paths:
                with Image.open(paths[0]) as image:
                    if self.app.input_hand=='left':image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    image.thumbnail((350,190));self.picture=ImageTk.PhotoImage(image.copy())
                self.diagram.config(image=self.picture)
        self.heading.set(f'{self.index+1}/{len(self.takes)}　{name}');self.instruction.set(instruction.replace('右手',self.app.hand_name))
        self.status.set(f'准备好后按回车，倒数 3 秒后开始，采 {duration} 秒。')
        self.button.config(state='normal')

    def begin(self):
        if self.index>=len(self.takes) or self.start is not None:return
        if not self.app.latest or time.monotonic()-self.app.latest[0]>.5:
            self.status.set(f'等待 HumanHand 向 127.0.0.1:{self.app.port} 发送{self.app.hand_name} IK 数据。收到数据后再按回车。');return
        with self.lock:
            self.rows=[];self.start=time.monotonic()+3.;self.end=self.start+self.takes[self.index][3]
        self.button.config(state='disabled')

    def receive(self,now,frame):
        with self.lock:
            if self.start is not None and self.start<=now<=self.end:self.rows.append(frame)

    def save(self,rows):
        split,key,name,duration,_=self.takes[self.index]
        hold=key=='rest_again' or (self.pinch and split=='train')
        stamps=np.array([r['timestamp_ns'] for r in rows],dtype=float)*1e-9
        if len(rows)<duration*15 or stamps[-1]-stamps[0]<duration*.8 or np.any(np.diff(stamps)<=0) or np.diff(stamps).max()>.4:
            raise ValueError('有效帧不足或数据中断。请检查官方发送，再重做本项。')
        if self.official:
            from tools.official_pose_calibration import MOVING
            hold=key not in MOVING
            if key in ('index','middle','ring','pinky'):
                from tools.fit_pinch_round import check_take
                check_take(rows,key,split)
            elif key not in MOVING:open_reference(rows)
            else:
                from g7_joint_mapping import joint_rotations
                digits=('thumb',) if key=='thumb_motion' else ('index','middle','ring','pinky')
                start=joint_rotations(rows[0])
                movement=[]
                for row in rows[::6]:
                    now=joint_rotations(row)
                    movement.append(max((a.inv()*b).magnitude() for f in digits for a,b in zip(start[f],now[f])))
                if np.percentile(movement,90)<np.radians(8):raise ValueError('该段关节活动太小，请在开始采集后做动作。')
        elif self.pinch:
            from tools.fit_pinch_round import check_take
            check_take(rows,key,split,self.manifest['unavailable_fingers'])
        elif key=='rest_again':open_reference(rows)
        else:
            qs=np.array([r['quaternions_wxyz'] for r in rows])
            qs/=np.linalg.norm(qs,axis=-1,keepdims=True)
            rel=np.abs(np.sum(qs*qs[0],axis=2))
            if np.percentile(np.max(np.degrees(2*np.arccos(np.clip(rel,0,1))),axis=1),90)<5:
                raise ValueError('本段只有静止手形。倒计时结束后开始动作，再做一次。')
        filename=f'{split}_{key}.jsonl'
        (self.folder/filename).write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf8')
        self.manifest[split].append(dict(file=filename,stage=key,kind='hold' if hold else 'motion',samples=16 if self.official else 24))
        write(self.folder/'manifest.json',self.manifest)

    def calibrate(self):
        package=ROOT/'assets/urdf_a'
        manifest=self.folder/'manifest.json'
        candidate=self.folder/'profile.json'
        try:
            if self.official or self.pinch:
                if self.official:
                    from tools.official_pose_calibration import fit_round, validate_round
                else:
                    from tools.fit_pinch_round import fit_round, validate_round
                fit_round(self.folder)
            else:
                from tools.fit_urdf_a import fit
                write(candidate,fit(package,self.app.profile.parent,manifest))
        except Exception as exc:
            self.issue=f'拟合失败：{exc}'
            return

        try:
            if self.official or self.pinch:
                validate_round(self.folder)
            else:
                from tools.validate_urdf_a import run
                report=run(package,self.app.profile.parent,profile=candidate,supplement=manifest)
                write(self.folder/'validation.json',report)
        except Exception as exc:
            self.validation_notes.append(str(exc))
        if not self.official:
            try:
                from tools.contact_regression import check
                check(package,self.folder/'baseline_profile.json',candidate)
            except Exception as exc:
                self.validation_notes.append(str(exc))

    def apply_profile(self):
        from urdf_a_follow import Follower
        candidate=self.folder/'profile.json'
        payload=candidate.read_bytes()
        if json.loads(payload).get('input_hand','right')!=self.app.input_hand:
            raise ValueError('标定档案与当前手套手别不一致')
        previous=self.app.follower
        try:
            updated=Follower(self.app.g,candidate)
            updated.u=previous.u.copy()
            updated.goal=updated.u.copy()
            updated.g.assign(updated.u)
            pending=self.app.profile.with_suffix('.json.tmp')
            pending.write_bytes(payload)
            pending.replace(self.app.profile)
        except Exception:
            previous.g.assign(previous.u)
            raise
        self.app.follower=updated
        self.app.last_seen=None
        return updated.profile_sha256

    def poll(self):
        if not self.alive:return
        now=time.monotonic()
        if self.start is not None:
            if now<self.start:self.status.set(f'准备　{int(np.ceil(self.start-now))} 秒')
            elif now<self.end:self.status.set(f'现在做动作　剩余 {self.end-now:.1f} 秒')
            else:
                with self.lock:rows=self.rows;self.rows=[];self.start=None
                try:self.save(rows)
                except ValueError as exc:self.show();self.status.set(str(exc))
                else:
                    self.index+=1
                    done={e['stage'] for split in ('train','validation') for e in self.manifest[split]}
                    while self.index<len(self.takes) and any(e['stage']==self.takes[self.index][1] for e in self.manifest[self.takes[self.index][0]]):self.index+=1
                    if self.index<len(self.takes):self.show()
                    else:
                        self.manifest['capture_completed']=True
                        self.manifest['application_status']='validating'
                        write(self.folder/'manifest.json',self.manifest)
                        self.heading.set('补录完成，正在拟合并验证')
                        self.instruction.set(f'可以放松{self.app.hand_name}。拟合成功后自动应用本轮档案，回放结果会显示在这里。')
                        self.status.set(str(self.folder));self.job=threading.Thread(target=self.calibrate,daemon=True);self.job.start()
        if self.job is not None and not self.job.is_alive():
            self.job=None
            self.manifest.update(capture_completed=True,application_policy='latest_fitted',
                                 validation_notes=self.validation_notes,
                                 validation_status='skipped' if self.issue else ('warning' if self.validation_notes else 'passed'))
            if self.issue is None:
                try:
                    self.manifest['loaded_profile_sha256']=self.apply_profile()
                except Exception as exc:
                    self.issue=f'加载失败：{exc}'
            if self.issue:
                self.heading.set('本轮标定未能加载')
                self.instruction.set('采集数据已保存。')
                self.status.set(self.issue)
                self.manifest.update(completed=False,application_status='failed',application_error=self.issue)
            else:
                self.manifest.update(completed=True,application_status='applied')
                self.manifest.pop('application_error',None)
                self.heading.set('已应用本轮标定')
                self.instruction.set('回放提示：'+'；'.join(self.validation_notes) if self.validation_notes else '回放检查通过。')
                self.status.set(str(self.folder))
            write(self.folder/'manifest.json',self.manifest)
            self.button.configure(text='关闭',state='normal',command=self.close)
        self.window.after(100,self.poll)

