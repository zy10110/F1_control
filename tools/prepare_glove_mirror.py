"""Build a separate left-input profile from solved FSNH frames (any pose).

Only rigid bone dimensions and palm roots are measured. An arbitrary incoming
pose must never become the wearer's angular zero.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from g7_left_mirror import canonical_right
from g7_joint_mapping import rotations
from g7_teleop import BASE
from urdf_a_follow import SourceHand
from tools.calibrate_angle_map import write


def prepare(profile,recording,output):
    if profile.resolve()==output.resolve() or output.exists():raise ValueError('输出必须是新的独立左手档案')
    ref=json.loads(profile.read_bytes())
    if ref.get('input_hand','right')!='right':raise ValueError('需要右手基准')
    rows=[canonical_right(json.loads(line)) for line in recording.read_text(encoding='utf8').splitlines() if line.strip()]
    if len(rows)<20:raise ValueError('至少需要 20 帧有效左手位姿')
    geometries=[SourceHand(row) for row in rows[::max(1,len(rows)//40)]]
    neutral=ref['neutral'];r=rotations(neutral);pts=np.array(neutral['positions_m'])*1000
    for f,i in BASE.items():
        root=np.median([s.roots[f] for s in geometries],axis=0)
        bones=np.median([s.bones[f] for s in geometries],axis=0)
        if not np.all((np.linalg.norm(bones,axis=1)>8)&(np.linalg.norm(bones,axis=1)<80)):raise ValueError('左手骨长无效：'+f)
        delta=r[i:i+3].apply(bones)
        pts[i:i+4]=pts[0]+r[0].apply(root)+np.vstack((np.zeros(3),np.cumsum(delta,axis=0)))
    neutral['positions_m']=(pts/1000).tolist()
    ref.update(input_hand='left',angle_reference='transferred_right',source_geometry='measured_left_mirrored',
               mirror_base_sha256=hashlib.sha256(profile.read_bytes()).hexdigest())
    write(output,ref)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('profile','recording','output'):p.add_argument(key,type=Path)
    args=p.parse_args();prepare(args.profile,args.recording,args.output)
