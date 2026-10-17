#!/usr/bin/env python3
"""Inspect ACT predictions without constructing a robot or sending commands.

Save the exact original unnormalized policy tensors for offline replay.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_scene1 import load_config
from scene1.kinematics import UrdfArmKinematics
from scene1.policy_trace import ACTChunkTrace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--urdf', type=Path, default=Path('/tmp/scene1_robot.urdf'),
                        help='Public robot URDF saved from /humanoid_description')
    args = parser.parse_args()
    if not args.urdf.is_file():
        parser.error('Public robot URDF is missing; pass --urdf or save /humanoid_description first')
    config, base, _ = load_config(args.config)
    from lerobot_patches import custom_patches  # noqa: F401
    from kuavo_deploy.utils.policy_loader import load_native_policy_bundle, inject_task_prompt
    policy, preprocess, postprocess, _ = load_native_policy_bundle(
        pretrained_path=base.inference.pretrained_path, device=torch.device(base.inference.device), strict=True)
    if args.input:
        with np.load(args.input) as data:
            observation = {key: torch.from_numpy(data[key]) for key in data.files if key.startswith('observation.')}
    else:
        import rospy
        from kuavo_deploy.utils.obs_buffer import ObsBuffer
        from kuavo_deploy.kuavo_env.KuavoBaseRosEnv import KuavoBaseRosEnv
        rospy.init_node('scene1_act_probe', anonymous=True)
        buffer = ObsBuffer(base.env)
        deadline = time.monotonic() + 15
        while not buffer.obs_buffer_is_ready():
            if time.monotonic() > deadline:
                raise RuntimeError('Observation buffer timeout')
            time.sleep(.05)
        # get_obs only reads ObsBuffer. __init__ would construct robot controls.
        reader = KuavoBaseRosEnv.__new__(KuavoBaseRosEnv)
        for name in ('frame_alignment', 'ratio', 'ros_rate', 'is_binary', 'arm_state_keys', 'which_arm'):
            setattr(reader, name, getattr(base.env, name))
        reader.obs_buffer = buffer
        observation = reader.get_obs()
        buffer.stop_subscribers()
    if not observation:
        raise ValueError('Input has no original policy tensors')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'observation.npz',
                        **{key: value.cpu().numpy() for key, value in observation.items()})
    policy.reset()
    inputs = preprocess(inject_task_prompt(dict(observation), base.inference.task_prompt))
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(inputs)
        decoded = torch.stack([postprocess(chunk[:, i].clone()) for i in range(chunk.shape[1])], dim=1)
        policy.reset()
        reference = policy.select_action(inputs).clone()
        policy.reset()
        trace = ACTChunkTrace(policy)
        traced = policy.select_action(inputs)
        trace.describe(postprocess)
        torch.testing.assert_close(traced, reference, rtol=0, atol=0)
        trace.close()
    actions = decoded.cpu().numpy()[0]
    np.save(args.output / 'chunk.npy', actions)
    kinematics = UrdfArmKinematics(description=args.urdf.read_text())
    samples = []
    for index in range(0, len(actions), 10):
        action = actions[index]
        mat = kinematics.matrix(action[8:15], 1)
        samples.append(dict(offset=index, left_grip=float(action[7]), right_grip=float(action[15]),
                            right_tcp=mat[:3, 3].tolist(),
                            right_pitch_degrees=float(np.rad2deg(np.arcsin(mat[2, 2])))))
    report = dict(read_only=True, trace_preserves_action_exactly=True,
                  temporal_ensemble_coeff=policy.config.temporal_ensemble_coeff,
                  chunk_size=policy.config.chunk_size,
                  observation_state=observation['observation.state'].cpu().numpy().tolist(), samples=samples,
                  min_right_grip=float(actions[:, 15].min()), max_left_grip=float(actions[:, 7].max()))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
