#!/usr/bin/env python3
"""Short, measured upward arm motion with both grippers open. No parcel grasp."""
import json
import argparse
import time
from pathlib import Path
import numpy as np
import rospy
from run_scene1 import load_config, SensorWatchdog
from scene1.cartesian import CartesianGraspAligner
from scene1.kinematics import UrdfArmKinematics
from scene1.rgbd import RgbdScene
from scene1.motion import MotionFilter, ActionPassthrough, reset_for_manipulation

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--from-release', type=Path, help='Use a completed weighing release or verified empty-arm retreat')
parser.add_argument('--height', type=float, default=.06)
parser.add_argument('--back', type=float, default=0)
parser.add_argument('--lateral', type=float, default=0)
args = parser.parse_args()
if not 0 < args.height <= .15 or not 0 <= args.back <= .10 or abs(args.lateral) > .12:
    parser.error('Require height in (0,.15], back in [0,.10] and abs(lateral)<=.12 m')
source = None
if args.from_release is not None:
    if (args.from_release / 'tracking.jsonl').is_file():
        row = json.loads((args.from_release / 'tracking.jsonl').read_text().splitlines()[-1])
        if row.get('state') != 'tracking_probe':
            parser.error('Source must describe an empty-arm tracking probe')
        source = row['command']
    else:
        summary = json.loads((args.from_release / 'summary.json').read_text())
        if summary['reason'] != 'weighing_release_completed':
            parser.error('Source run must have completed weighing release')
        source = json.loads((args.from_release / 'steps.jsonl').read_text().splitlines()[-1])['sent_action']
    if max(source[7], source[15]) > .25:
        parser.error('Both grippers must already be open')
c, base, _ = load_config(ROOT / 'config.yaml')
base.env.head_init = [0.0, np.deg2rad(20)]
base.env.direct_to_wbc = True
rospy.init_node('kuavo_deploy', anonymous=True)
from kuavo_deploy.src.scripts.script import ArmMove
arm = ArmMove(base)
env = arm.env
topic_map = {entry['topic']: entry['msg_type'] for entry in base.env.obs_key_map.values() if 'topic' in entry}
watchdog = SensorWatchdog(rospy, topic_map, c['sensor_timeout_s'])
scene = RgbdScene(rospy, c['cartesian_alignment'])
try:
    obs, _ = reset_for_manipulation(env, keep_head=source is not None)
    env.low_pass_filter = ActionPassthrough()
    measured = obs['observation.state'].cpu().numpy().reshape(-1)
    raw = measured.copy()
    raw[[7, 15]] = 0
    if source is not None:
        raw[:7] = source[:7]
    k = UrdfArmKinematics(rospy)
    a = CartesianGraspAligner(c['cartesian_alignment'], k, env.action_space.low, env.action_space.high)
    raw = np.clip(raw, a.low, a.high)
    if source is not None:
        offset = np.asarray(source)[8:15] - measured[8:15]
        if np.max(np.abs(offset)) > c['cartesian_alignment']['max_loaded_feedforward_rad']:
            raise RuntimeError('Static empty-arm offset is too large')
        a.loaded_feedforward[8:15] = offset
        a.tracking_bias[8:15] = offset
    import time
    deadline = time.monotonic() + c['startup_timeout_s']
    while True:
        try:
            snapshot = scene.read(); break
        except Exception:
            if time.monotonic() > deadline: raise
            time.sleep(.05)
    a.weighing_target = np.array([0, 0, snapshot['table_z_base_m']])
    poses = k.fk(raw)
    goal = np.asarray(poses.right_pose.pos_xyz) + [-args.back, args.lateral, args.height]
    segments, _ = a._path('tracking_probe', poses, 1, goal, raw, 15)
    a._begin(raw, 1, segments)
    motion = MotionFilter(env.action_space.low, env.action_space.high, .8, .6)
    motion.reset(measured if source is None else source)
    output = ROOT / 'outputs' / (time.strftime('%Y%m%d-%H%M%S') + '-tracking')
    output.mkdir()
    with (output / 'tracking.jsonl').open('w') as log:
        for step in range(180):
            watchdog.check()
            try:
                snapshot = scene.read()
            except ValueError:
                snapshot = None  # empty-arm control retains the last observed table plane
            proposed, info = a.apply(raw, measured, snapshot)
            command, _ = motion.apply(proposed, .1)
            obs, *_ = env.step(command)
            measured = obs['observation.state'].cpu().numpy().reshape(-1)
            info.update(step=step, command=command.tolist(), measured=measured.tolist())
            log.write(json.dumps(info) + '\n')
            if step % 10 == 0 or a.pending is None:
                print(json.dumps(info), flush=True)
            if a.pending is None:
                print('UPWARD_TRACKING_VERIFIED', output, flush=True)
                break
        else:
            raise RuntimeError('Tracking probe did not finish')
finally:
    scene.close()
    watchdog.close()
    env.close()
