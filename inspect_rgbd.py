#!/usr/bin/env python3
"""Read RGB-D and FK only. Save parcel positions and projected end effectors."""
import json
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from scene1.rgbd import RgbdScene
from scene1.cartesian import CartesianGraspAligner
from scene1.kinematics import UrdfArmKinematics
from scene1.recovery import load_run_history, load_last_arm_command

ROOT = Path(__file__).resolve().parent


def main():
    import rospy
    from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData
    rospy.init_node("scene1_rgbd_inspect", anonymous=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-from', type=Path, help='Check a held-parcel recovery path without moving')
    parser.add_argument('--plan-regrasp-from', type=Path)
    parser.add_argument('--arm-state-from', type=Path)
    parser.add_argument('--wrist-roll-degrees', type=float, default=0.)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config.yaml").read_text())["cartesian_alignment"]
    scene = RgbdScene(rospy, config)
    deadline = time.monotonic() + 15
    try:
        while True:
            try:
                snapshot = scene.read()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        message = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=3)
        action = np.zeros(16)
        # platform 5 has one waist DOF before the 14 arm joints.
        arm_start = 12 + int(rospy.get_param('/mpc/mpcWaistDof', 0))
        action[:7] = message.joint_data.joint_q[arm_start:arm_start + 7]
        action[8:15] = message.joint_data.joint_q[arm_start + 7:arm_start + 14]
        kinematics = UrdfArmKinematics(rospy)
        poses = kinematics.fk(action)
        recovery_plan = None
        if args.plan_from is not None:
            records = load_run_history(args.plan_from)
            aligner = CartesianGraspAligner(config, kinematics)
            aligner.restore_weighing(records, action, snapshot)
            recovery_plan = {'phases': len(aligner.pending['phases']),
                             'release_position_base_m': kinematics.fk(aligner.pending['phases'][-1]['action']).right_pose.pos_xyz}
        if args.plan_regrasp_from is not None:
            records = load_run_history(args.plan_regrasp_from)
            aligner = CartesianGraspAligner(config, kinematics)
            aligner.restore_regrasp(records, action, snapshot,
                load_last_arm_command(args.arm_state_from) if args.arm_state_from else None,
                wrist_roll_degrees=args.wrist_roll_degrees)
            recovery_plan = dict(phases=len(aligner.pending['phases']),
                                 regrasp_target=aligner.pending['target'],
                                 wrist_roll_degrees=args.wrist_roll_degrees)
        image = cv2.cvtColor(snapshot["rgb"], cv2.COLOR_RGB2BGR)
        for index, candidate in enumerate(snapshot["candidates"]):
            x, y, w, h = candidate["bbox"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(image, f"parcel-{index}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        inverse = np.linalg.inv(snapshot["camera_to_base"])
        projected = {}
        for side, pose in (("left", poses.left_pose), ("right", poses.right_pose)):
            point = inverse @ np.r_[pose.pos_xyz, 1]
            uvw = snapshot["intrinsic"] @ point[:3]
            uv = uvw[:2] / uvw[2]
            projected[side] = {"position_base_m": list(pose.pos_xyz), "uv": uv.tolist()}
            if point[2] > 0 and 0 <= uv[0] < image.shape[1] and 0 <= uv[1] < image.shape[0]:
                cv2.circle(image, tuple(np.round(uv).astype(int)), 6, (0, 0, 255), -1)
                cv2.putText(image, side + "-eef", tuple(np.round(uv).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        output = ROOT / "captures" / (time.strftime("%Y%m%d-%H%M%S") + "-rgbd")
        output.mkdir(parents=True)
        cv2.imwrite(str(output / "geometry.jpg"), image)
        np.savez_compressed(output / 'rgbd.npz', rgb=snapshot['rgb'],
            depth_m=snapshot['depth_m'], intrinsic=snapshot['intrinsic'],
            camera_to_base=snapshot['camera_to_base'], measured_state=action)
        result = {"table_z_base_m": snapshot["table_z_base_m"],
                  "surfaces": snapshot.get("surfaces", []),
                  "candidates": snapshot["candidates"], "end_effectors": projected,
                  "recovery_plan": recovery_plan,
                  "controls_sent": False, "output": str(output)}
        (output / "geometry.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        scene.close()


if __name__ == "__main__":
    main()
