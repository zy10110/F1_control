"""Forward solved G7 joints and source IMU from one official SDK 1.2.5 handle."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import struct
import time
import tempfile
import threading
import sys

from g7_teleop import load_g7_config, load_reader
from hand_kinematics import ROOT


class CoilPoll(ct.Structure):
    _fields_ = [('status',ct.c_int32), ('pad',ct.c_uint32), ('packet',ct.c_ubyte*384)]


class CalibrationEvent(ct.Structure):
    _fields_ = [(name,ct.c_int32) for name in ('phase','step','step_count','gesture','seconds_left')]


CalibrationCallback = ct.CFUNCTYPE(ct.c_int,ct.POINTER(CalibrationEvent),ct.c_void_p)


def load_runtime(directory):
    directory = directory.resolve()
    search = os.add_dll_directory(str(directory))
    dll = ct.CDLL(str(directory/'hand_runtime.dll'))
    dll._directory_handle = search
    dll.hh_sdk_version.restype = ct.c_char_p
    if dll.hh_sdk_abi_version() != 3 or dll.hh_sdk_version() != b'1.2.5':
        raise RuntimeError('需要 G7 SDK 1.2.5 / ABI 3')
    reader = load_reader(load_g7_config())
    dll.hh_sdk_create.argtypes = [ct.POINTER(reader.Config)]
    dll.hh_sdk_create.restype = ct.c_void_p
    dll.hh_sdk_destroy.argtypes = [ct.c_void_p]
    dll.hh_sdk_destroy.restype = None
    dll.hh_sdk_hand.argtypes = [ct.c_void_p]
    dll.hh_sdk_probe_hand.argtypes = [ct.c_char_p,ct.c_int]
    dll.hh_sdk_poll.argtypes = [ct.c_void_p, ct.POINTER(reader.Frame)]
    dll.hh_sdk_poll_coils.argtypes = [ct.c_void_p, ct.POINTER(CoilPoll)]
    for mode in ('general','pinch'):
        getattr(dll,'hh_sdk_calib_'+mode).argtypes = [ct.c_void_p,CalibrationCallback,ct.c_void_p]
    dll.hh_sdk_calib_cancel.argtypes = [ct.c_void_p]
    for setting in ('pinch_enabled','traj_filter','mag_guard'):
        getattr(dll,'hh_sdk_get_'+setting).argtypes = [ct.c_void_p,ct.POINTER(ct.c_int)]
    dll.hh_sdk_set_pinch_enabled.argtypes = [ct.c_void_p,ct.c_int]
    return dll, reader


def stream(dll, reader, handle, hand, send, stop, service=None):
    joint, coil = reader.Frame(), CoilPoll()
    packet = struct.Struct('<7f')
    coils_first = True
    last_data=time.monotonic()
    while not stop():
        if service:
            serviced=service()
            if serviced is False:return
            if serviced is True:last_data=time.monotonic()
        # The two poll APIs consume samples. Alternate first call so a busy
        # joint solver cannot starve the raw path. Each packet retains its
        # own device sequence and acquisition timestamp.
        for is_coil in (coils_first, not coils_first):
            if is_coil:
                if dll.hh_sdk_poll_coils(handle,ct.byref(coil)):
                    raise ConnectionError('G7 IMU 读取失败')
                if coil.status == 0:
                    send(bytes(coil.packet))
            else:
                if dll.hh_sdk_poll(handle,ct.byref(joint)):
                    raise ConnectionError('G7 关节读取失败')
                if joint.status == 0:
                    header = reader.HEADER.pack(0x464e4853,4,hand,21,1,joint.seq,joint.timestamp_ns,0x77787a79)
                    send(header+b''.join(packet.pack(*p,*q) for p,q in zip(joint.pos_m,joint.quat_wxyz)))
        coils_first = not coils_first
        if coil.status != 0 and joint.status != 0:
            if time.monotonic()-last_data>=5.:
                raise ConnectionError('连续 5 秒没有收到五指或 IMU 有效帧')
            time.sleep(.001)
        else:
            last_data=time.monotonic()


class Calibration:
    def __init__(self,dll,status_path):
        self.dll=dll
        self.path=status_path
        self.commands=queue.SimpleQueue()
        self.stop=threading.Event()
        self.cancel=threading.Event()
        self.lock=threading.Lock()
        self.active=None
        self.state={}

    def publish(self,**changes):
        state=dict(self.state,**changes)
        if state!=self.state:
            tmp=self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(state,ensure_ascii=False),encoding='utf8')
            tmp.replace(self.path)
            self.state=state

    def read_commands(self):
        try:
            pending=b''
            while chunk:=os.read(sys.stdin.fileno(),256):
                lines= (pending+chunk).split(b'\n')
                pending=lines.pop()
                for line in lines:
                    mode=line.strip().decode('ascii')
                    if mode in ('general','pinch'):
                        self.commands.put(mode)
                    elif mode=='cancel':
                        self.interrupt()
        finally:
            self.stop.set()
            self.interrupt()

    def interrupt(self):
        with self.lock:
            self.cancel.set()
            if self.active is not None:
                self.dll.hh_sdk_calib_cancel(self.active)

    def service(self,handle,working,saved):
        try:
            mode=self.commands.get_nowait()
        except queue.Empty:
            return None
        self.publish(attempt=self.state.get('attempt',0)+1,result='',mode=mode,
                     phase=0,step=0,total=4,gesture=0,seconds=0)
        callback_error=[]
        @CalibrationCallback
        def progress(pointer,user):
            event=pointer.contents
            try:
                self.publish(mode=mode,phase=event.phase,step=event.step,total=event.step_count,
                             gesture=event.gesture,seconds=event.seconds_left)
            except OSError as exc:
                callback_error.append(str(exc))
                return 1
            return int(self.cancel.is_set() or self.stop.is_set())

        # No polls while the vendor's blocking call owns the serial stream.
        with self.lock:
            if self.stop.is_set() or self.cancel.is_set():
                self.cancel.clear()
                self.publish(result='已取消',mode=None)
                return True
            self.active=handle
        try:
            rc=getattr(self.dll,'hh_sdk_calib_'+mode)(handle,progress,None)
        finally:
            with self.lock:
                self.active=None
                self.cancel.clear()
        if rc==0 and not callback_error:
            try:
                payload=working.read_bytes()
                if not payload: raise OSError('SDK 输出的手型为空')
                tmp=saved.with_suffix('.tmp')
                tmp.write_bytes(payload)
                tmp.replace(saved)
            except OSError as exc:
                callback_error.append(str(exc))
            else:
                self.publish(mode=None,profile_sha256=hashlib.sha256(payload).hexdigest(),
                             result='官方标定已保存并应用',phase=0)
                return True
        result={3:'需要先完成普通手型标定',4:'求解失败',6:'已取消'}.get(rc,f'SDK 错误 {rc}')
        self.publish(mode=None,phase=0,result='；'.join(callback_error) or result)
        return False  # Recreate the handle from the last committed calibration.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True,help='G7 serial port')
    parser.add_argument('--hand',choices=('left','right'),default='left')
    parser.add_argument('--udp-port',default=15001,type=int)
    parser.add_argument('--sdk-lib',type=Path,default=ROOT.parent/'G7_F1接入/sdk/G7_sdk_1.2.5_windows_x86_64/lib/windows/x86_64')
    parser.add_argument('--humanhand',type=Path,default=Path('E:/HumanHand'))
    parser.add_argument('--profile',type=Path,help='official personal hand_profile_*.bin')
    parser.add_argument('--parent-pipe',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    dll, reader = load_runtime(args.sdk_lib)
    hand = 1 if args.hand == 'left' else 2
    source = args.profile or args.humanhand/'assets/users/默认用户'/f'hand_profile_{args.hand}.bin'
    cache = args.humanhand/'param'
    if not source.is_file() or not cache.is_dir():
        parser.error('找不到个人手型或工厂参数，请指定 --profile 和 --humanhand')
    # Detect an occupied port before entering the vendor runtime or
    # touching any live HumanHand session. Only this process owns the SDK.
    import serial
    runtime = ROOT/'results'/'g7_imu_runtime'/args.hand
    runtime.mkdir(parents=True,exist_ok=True)
    original=source.read_bytes()
    source_hash=hashlib.sha256(original).hexdigest()
    saved=runtime/f'calibrated_{source_hash}.bin'
    control=Calibration(dll,ROOT/'results'/f'g7_sdk_state_{args.udp_port}.json')
    if args.parent_pipe:
        threading.Thread(target=control.read_commands,daemon=True).start()
    shutil.copytree(cache,runtime/'param',dirs_exist_ok=True)
    os.chdir(runtime)  # narrow paths in the Windows SDK; private writable copies
    with tempfile.TemporaryDirectory(dir=runtime,prefix='session_') as session:
        working=Path(session)/'profile.bin'
        while not control.stop.is_set():
            port=args.port
            if port.lower()=='auto':
                from serial.tools.list_ports import comports
                hubs=sorted(p.device for p in comports() if (p.vid,p.pid)==(0x1a86,0x55d2))
                reports=[]
                for candidate in hubs:
                    if control.stop.is_set():return
                    side=dll.hh_sdk_probe_hand(candidate.encode('ascii'),921600)
                    if side==hand:
                        port=candidate
                        break
                    reports.append(candidate+{1:' 左手',2:' 右手'}.get(side,' 未应答'))
                else:
                    control.publish(pid=os.getpid(),connection='retry',reason='、'.join(reports) or '未检测到 G7 Hub',
                                    serial_port=None,settings={},mode=None)
                    control.stop.wait(4.)
                    continue
            config=reader.Config(port.encode('ascii'),921600,hand,os.fsencode(working.relative_to(runtime)),b'param')
            payload=saved.read_bytes() if saved.exists() else original
            working.write_bytes(payload)
            handle=None
            try:
                with serial.Serial(port,921600,timeout=.1):
                    pass
            except serial.SerialException as exc:
                reason=f'无法访问 {port}：{exc}'
            else:
                handle=dll.hh_sdk_create(ct.byref(config))
                reason=f'{port} 可打开，手套未响应'
            if not handle:
                control.publish(pid=os.getpid(),connection='retry',reason=reason,serial_port=port,settings={},mode=None)
                control.stop.wait(4.)
                continue
            try:
                if dll.hh_sdk_hand(handle)!=hand:
                    raise RuntimeError('串口手别与所选标定不一致')
                settings={}
                for name in ('pinch_enabled','traj_filter','mag_guard'):
                    value=ct.c_int()
                    if getattr(dll,'hh_sdk_get_'+name)(handle,ct.byref(value)):
                        raise RuntimeError('无法读取 SDK 设置 '+name)
                    if name=='pinch_enabled' and not value.value:
                        rc=dll.hh_sdk_set_pinch_enabled(handle,1)
                        if rc or dll.hh_sdk_get_pinch_enabled(handle,ct.byref(value)) or value.value!=1:
                            raise RuntimeError(f'SDK 捏合优化未启用（设置返回 {rc}）')
                    settings[name]=value.value
                control.publish(pid=os.getpid(),connection='open',reason='',serial_port=port,settings=settings,mode=None,phase=0,
                                profile_sha256=hashlib.sha256(payload).hexdigest(),source_sha256=source_hash)
                with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
                    address=('127.0.0.1',args.udp_port)
                    stream(dll,reader,handle,hand,lambda p:sock.sendto(p,address),control.stop.is_set,
                           lambda:control.service(handle,working,saved))
            except ConnectionError as exc:
                control.publish(connection='retry',reason=str(exc),settings={},mode=None)
            finally:
                dll.hh_sdk_destroy(handle)
            if control.state.get('connection')=='retry':
                control.stop.wait(2.)


if __name__ == '__main__':
    main()
