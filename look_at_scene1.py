#!/usr/bin/env python3
"""Look at the observed weighing pad; send head commands only."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from scene1.head import HeadLookController
from scene1.kinematics import UrdfArmKinematics
from scene1.rgbd import RgbdScene
from scene1.recovery import load_run_history
from scene1.cartesian import CartesianGraspAligner

ROOT = Path(__file__).resolve().parent


def main():
    import rospy
    from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot
    from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-release', type=Path, help='Use the released tool pose to identify the pad')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / 'config.yaml').read_text())
    rospy.init_node('scene1_head_look', anonymous=True)
    if not KuavoSDK().Init():
        raise RuntimeError('SDK initialization failed')
    robot = KuavoRobot()
    scene = RgbdScene(rospy, config['cartesian_alignment'])
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                snapshot = scene.read()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.05)
        k = UrdfArmKinematics(rospy)
        if args.from_release is not None:
            records = load_run_history(args.from_release)
            settled = [row for row in records if row['cartesian_alignment'].get('state') == 'settle_before_release'
                       and row['cartesian_alignment'].get('arm') == 1]
            observed_pad = [row['cartesian_alignment']['target_base_m'] for row in records
                            if row['cartesian_alignment'].get('state') == 'small_weighing'
                            and row['cartesian_alignment'].get('target_base_m') is not None]
            if observed_pad:
                anchor = observed_pad[-1]
            else:
                row = settled[-1] if settled else records[-1]
                anchor = k.fk(row['measured_state']).right_pose.pos_xyz
        else:
            message = rospy.wait_for_message('/sensors_data_raw', sensorsData, timeout=3)
            action = np.zeros(16)
            action[:7] = message.joint_data.joint_q[13:20]
            action[8:15] = message.joint_data.joint_q[20:27]
            anchor = k.fk(action).right_pose.pos_xyz
        if args.from_release is not None:
            snapshot = CartesianGraspAligner.restore_surface_view(snapshot, records)
        surfaces = snapshot.get('surfaces', [])
        if args.from_release is not None and (not surfaces or min(
                np.linalg.norm(np.asarray(s['position_base_m'])[:2] - np.asarray(anchor)[:2])
                for s in surfaces) > .20):
            surfaces = [surface for old in records for surface in old.get('rgbd_surfaces', [])
                        if min(surface['extent_xy_m']) >= .145]
        if not surfaces:
            raise RuntimeError('No visually observed weighing surface')
        surface = min(surfaces, key=lambda s: np.linalg.norm(np.asarray(s['position_base_m'])[:2] - np.asarray(anchor)[:2]))
        target = np.asarray(surface['position_base_m']) + [0, 0, .06]
        tracker = HeadLookController(config['head_camera'], rospy.get_param('/humanoid_description'))
        stable = 0
        for _ in range(70):
            snapshot = scene.read()
            command, info = tracker.update(target, scene.head_angles(), snapshot, .1)
            robot.control_head(float(command[0]), float(command[1]))
            error = np.max(np.abs(np.asarray(info['measured_degrees']) - info['desired_degrees']))
            stable = stable + 1 if error < config['head_camera']['home_tolerance_degrees'] else 0
            if stable >= config['head_camera']['stable_frames']:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Head did not reach the weighing view')
        time.sleep(.2)
        snapshot = scene.read()
        output = ROOT / 'captures' / (time.strftime('%Y%m%d-%H%M%S') + '-weighing-view')
        output.mkdir(parents=True)
        cv2.imwrite(str(output / 'head.jpg'), cv2.cvtColor(snapshot['rgb'], cv2.COLOR_RGB2BGR))
        result = dict(target_base_m=target.tolist(), head=info,
                      surfaces=snapshot.get('surfaces', []), candidates=snapshot['candidates'],
                      table_z_base_m=snapshot['table_z_base_m'],
                      controls_sent='head_only', arm_commands_sent=False, output=str(output))
        (output / 'view.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        scene.close()


if __name__ == '__main__':
    main()
