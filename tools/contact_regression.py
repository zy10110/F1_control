"""Keep accepted contact examples when a natural-motion calibration is refit."""
import json
from pathlib import Path

import numpy as np

from tools.calibrate_angle_map import write
from tools.fit_pinch_round import replay
from urdf_a_model import NativeHand


def contact_records(record):
    items=list(record.get('retained_contact_records',[]))
    feedback=record.get('feedback_refinement')
    if feedback:
        items.append(dict(file=str(Path(feedback['folder'])/'motion.jsonl'),finger='middle'))
    refinement=record.get('middle_refinement')
    if refinement:
        items.append(dict(file=refinement['development_motion'],finger='middle'))
    pinch=record.get('pinch_round')
    if pinch:
        manifest=Path(pinch)
        if manifest.is_file():
            for entry in json.loads(manifest.read_text(encoding='utf8'))['validation']:
                items.append(dict(file=str(manifest.parent/entry['file']),finger=entry['stage']))
    return list({(str(Path(e['file']).resolve()),e['finger']):dict(file=str(Path(e['file']).resolve()),finger=e['finger']) for e in items}.values())


def check(package,baseline,candidate):
    old=json.loads(baseline.read_text(encoding='utf8'))
    items=contact_records(old)
    report=dict(records=[],issues=[],accepted=True)
    if not items:return report
    g=NativeHand(package)
    for item in items:
        path=Path(item['file'])
        if not path.is_file():
            report['issues'].append('缺少原对指回放 '+str(path));continue
        rows=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
        stamps=np.array([r['timestamp_ns'] for r in rows],dtype=np.int64)
        if len(stamps)>1 and np.median(np.diff(stamps))>1e9/30:
            from tools.refine_middle_feedback import dense
            rows=dense(rows)
        before,after=[replay(g,p,rows,item['finger']) for p in (baseline,candidate)]
        report['records'].append(dict(**item,previous=before,refitted=after))
        reasons=[]
        gap_limit=before['hold_pad_mm']+1
        if old.get('contact_goal_mm') is not None:
            gap_limit=min(gap_limit,max(float(old['contact_goal_mm'])+.2,before['hold_pad_mm']))
        if after['hold_pad_mm']>gap_limit:
            reasons.append(f"间距 {before['hold_pad_mm']:.1f}→{after['hold_pad_mm']:.1f} mm")
        if after['mean_shape_deg']>before['mean_shape_deg']+.7:
            reasons.append(f"手形误差增加 {after['mean_shape_deg']-before['mean_shape_deg']:.2f}°")
        if after['blocked_frames']>before['blocked_frames']:
            reasons.append(f"新增阻挡 {after['blocked_frames']-before['blocked_frames']} 帧")
        if reasons:
            name={'index':'拇食','middle':'拇中','ring':'拇无名','pinky':'拇小'}.get(item['finger'],item['finger'])
            source='截图姿态' if path.name=='motion.jsonl' else '历史动作'
            report['issues'].append(name+'（'+source+'）：'+'，'.join(reasons))
    report['accepted']=not report['issues']
    write(candidate.parent/'retained_contact_validation.json',report)
    if not report['accepted']:raise ValueError('新补录未应用，保留当前标定：'+'；'.join(report['issues']))
    record=json.loads(candidate.read_text(encoding='utf8'))
    record['retained_contact_records']=items
    write(candidate,record)
    return report
