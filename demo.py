"""Offline coupled joint sweep using the live swept-clearance guard."""
import time
import tkinter as tk
from types import SimpleNamespace

import numpy as np
from PIL import ImageTk

from hand_kinematics import ROOT, FINGERS
from urdf_a_model import NativeHand
from urdf_a_follow import Sweep
from urdf_a_live import View


def launch(seconds=None):
    g = NativeHand(ROOT / 'assets/urdf_a')
    guard = Sweep(g)
    if not guard.safe(np.zeros(17)):
        raise ValueError('Initial model clearance is below the live guard margin')
    root = tk.Tk()
    root.title('F1 URDF-A · 离线耦合演示（无手套输入）')
    view = View(g)
    label = tk.Label(root)
    label.pack()
    status = tk.StringVar()
    tk.Label(root, textvariable=status).pack(pady=6)
    state = SimpleNamespace(chains=None)
    current = np.zeros(17)
    start = previous = time.monotonic()
    closed = False

    def close():
        nonlocal closed
        if not closed:
            closed = True
            view.renderer.close()
            root.destroy()

    def tick():
        nonlocal current, previous
        now = time.monotonic()
        if seconds is not None and now - start >= seconds:
            close()
            return
        dt = min(now - previous, .1)
        previous = now
        phase = (now - start) / 4
        finger = FINGERS[int(phase) % len(FINGERS)]
        amount = np.sin(np.pi * (phase % 1)) ** 2
        target = np.zeros(17)
        cols = g.cols[finger]
        target[cols[-2:]] = np.minimum(np.radians([40., 50.]), g.hi[cols[-2:]]) * amount
        cap = np.radians(120.) * dt
        proposed = current + np.clip(target - current, -cap, cap)
        proposed = guard.slide(current, proposed)
        current, report = guard.sweep_follow_digits(current, proposed)
        g.assign(current)
        photo = ImageTk.PhotoImage(view.draw(state))
        label.configure(image=photo)
        label.image = photo
        status.set(f'{finger} · MCP/PIP 半角分配 · DIP=0.8×PIP · 间隙 {g.clearance():.2f} mm'
                   + (' · 避让保持' if report['blocked'] else ''))
        root.after(25, tick)

    root.protocol('WM_DELETE_WINDOW', close)
    root.after(0, tick)
    root.mainloop()
