"""Read-only replay of the first formal v63 video's failed settlement RGB-D."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/scene1_act_optimization')
from scene1.rgbd import metric_candidates
from scene1.residual import ResidualAssist
from scene1.settlement import resting_observation
p=Path('/root/scene1_act_optimization/outputs/20261017-011742')
rows={r['step']:r for r in map(json.loads,(p/'steps.jsonl').open())}
cfg=json.loads((p/'run.json').read_text())['config']['cartesian_alignment']
center=rows[300]['cartesian_alignment']['placement_target_compensation']['original_target_base_m']
surfaces=[(n,s) for n,r in rows.items() if n<391 for s in r['rgbd_surfaces'] if not s.get('partial') and min(s['extent_xy_m'])>=.145]
n,surface=min(surfaces,key=lambda ns:np.linalg.norm(np.asarray(ns[1]['position_base_m'])-center))
extent=np.load(p/'sensors-0200.npz')['held_extent_m']
report={'read_only':True,'surface_source_step':n,'surface':surface,'surface_reference_distance_m':float(np.linalg.norm(np.asarray(surface['position_base_m'])-center)),'extent':extent.tolist(),'frames':[],'limitation':'Offline raw-frame geometry only; saved frames cannot prove continuous 1 second confirmation or physical success. No controller change.'}
for step in [420,440,460,480,500,520,528]:
 path=p/(f'sensors-{step:04d}.npz' if step<528 else 'failed-sensors.npz');snap=dict(np.load(path));snap['stamp']=rows[min(step,527)]['rgbd_stamp'];r={'step':step,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 for spatial in (False,True):
  snap['candidates'],snap['table_z_base_m']=metric_candidates(snap['rgb'],snap['depth_m'],snap['intrinsic'],snap['camera_to_base'],cfg,stride=2,spatial_components=spatial)
  r['spatial' if spatial else 'original']=resting_observation(snap,surface,extent,ResidualAssist._roof_normal)
 report['frames'].append(r)
 print(step,[(k,None if r[k] is None else {x:r[k][x] for x in ('top_height_m','support_height_ok','footprint_margin_m','tilt_degrees')}) for k in ('original','spatial')],flush=True)
Path('/tmp/v63-video-settlement-probe-20261017.json').write_text(json.dumps(report,indent=2)+'\n')
