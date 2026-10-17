#!/usr/bin/env python3
"""Run the original ACT with separately switchable scene-1 extensions."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

from scene1.motion import MotionFilter, ActionPassthrough, action_vector, head_radians, reset_for_manipulation
from scene1.grasp import GraspMonitor
from scene1.servo import VisualServo
from scene1.vision import ParcelDetector, rgb_image
from scene1.rgbd import RgbdScene, metric_candidates
from scene1.settlement import resting_observation
from scene1.cartesian import ArmKinematics, CartesianGraspAligner
from scene1.residual import ResidualAssist
from scene1.kinematics import UrdfArmKinematics
from scene1.head import HeadLookController
from scene1.wrist import WristProjector
from scene1.handover import HandoverGripGuard, HandoverPhaseRefresh
from scene1.policy_trace import ACTChunkTrace
from scene1.trace_window import PolicyTraceWindow
from scene1.handover_contract import HandoverContract, HandoverContractStop
from scene1.visual_cycle import VisualCycle
from scene1.visual_pickup import VisualPickup, grasp_template, grasp_plan, restore_next_context
from scene1.pickup_resume import write_pickup_checkpoint, restore_weighing_context
from scene1.placement_gain import PlacementGain
from scene1.placement_target import PlacementTarget
from scene1.bin_layout import validate_layout
from scene1.recovery import load_run_history, load_last_arm_command, validate_act_handover_resume

ROOT = Path(__file__).resolve().parent


def read_manipulation_rgbd(scene, aligner, visual_cycle, visual_pickup, config):
    pending = getattr(aligner, 'pending', None)
    settling = (visual_cycle is not None and pending is not None
                and pending['kind'] in ('short_retreat', 'settle'))
    split_enabled = config.get('visual_cycle', {}).get('settlement_spatial_components', False)
    split = (settling and pending['kind'] == 'settle' and split_enabled
             and visual_pickup is not None and visual_pickup.active)
    snapshot = scene.read(candidate_stride=2 if settling else 4, spatial_components=split)
    if settling and pending['kind'] == 'settle' and split_enabled and not split:
        # Keep every existing first-parcel observation, including unsupported
        # ones. Only missing geometry may try the already-used 3D separation;
        # the original settlement monitor still decides support and stability.
        if resting_observation(snapshot, pending['surface'], pending['extent'],
                               aligner._roof_normal) is None:
            snapshot = snapshot.copy()
            snapshot['candidates'], snapshot['table_z_base_m'] = metric_candidates(
                snapshot['rgb'], snapshot['depth_m'], snapshot['intrinsic'],
                snapshot['camera_to_base'], config['cartesian_alignment'],
                stride=2, spatial_components=True)
            snapshot['candidate_spatial_components'] = True
    return snapshot


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    for key in ("max_steps", "log_every_steps", "snapshot_every_steps"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("sensor_timeout_s", "startup_timeout_s"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    sys.path.insert(0, config["project_root"])
    from kuavo_deploy.config import load_kuavo_config
    base = load_kuavo_config(config["base_config"])
    env = base.env
    if (env.inference_env, env.control_mode, env.which_arm, env.only_arm) != ("sim", "joint", "both", True):
        raise ValueError("This extension supports scene1 simulation, joint control, both arms, only_arm=true")
    if base.inference.policy_type != "act" or base.inference.task_prompt != "scene1":
        raise ValueError("Use the original scene1 ACT deployment configuration")
    if base.inference.async_inference:
        raise ValueError("This runner supports synchronous ACT only")
    model_dir = Path(base.inference.pretrained_path)
    if not model_dir.is_absolute():
        model_dir = Path(config["project_root"]) / model_dir
    base.inference.pretrained_path = str(model_dir)
    model = json.loads((model_dir / "config.json").read_text())
    for filename in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(model_dir / filename)
    if model["type"] != "act" or model["output_features"]["action"]["shape"] != [16]:
        raise ValueError("Expected original 16-dimensional ACT policy")
    if model["input_features"]["observation.state"]["shape"] != [16]:
        raise ValueError("Expected joint/gripper state in 16-dimensional ACT layout")
    expected_images = {}
    for key, feature in model["input_features"].items():
        if key.startswith("observation.images."):
            shape = feature["shape"]
            if shape != [3, env.image_size[1], env.image_size[0]]:
                raise ValueError(f"{key}: model {shape} does not match configured resize {env.image_size}")
            expected_images[key] = shape
            if key.split(".")[-1] not in env.obs_key_map:
                raise ValueError(f"Missing original model camera: {key}")
    if config["vision"]["image_key"] not in expected_images:
        raise ValueError("Vision image_key must be an original policy camera")
    if config["servo"]["enabled"] and config["servo"]["image_key"] not in expected_images:
        raise ValueError("Servo image_key must be an original policy camera")
    if not Path(base.inference.go_bag_path).is_file():
        raise FileNotFoundError(base.inference.go_bag_path)
    low = env.limits["joint_q"]["min"][:7] + env.limits["gripper"]["min"][:1]
    low += env.limits["joint_q"]["min"][7:] + env.limits["gripper"]["min"][1:]
    high = env.limits["joint_q"]["max"][:7] + env.limits["gripper"]["max"][:1]
    high += env.limits["joint_q"]["max"][7:] + env.limits["gripper"]["max"][1:]
    MotionFilter(low, high, config["motion"]["smoothing_alpha"], config["motion"]["max_joint_velocity_rad_s"])
    head_radians(config["motion"].get("head_init_degrees"))
    VisualServo(config["servo"])
    GraspMonitor(config["grasp"])
    ParcelDetector(config["vision"])
    PolicyTraceWindow(config.get('policy_trace', {}), ROOT)
    contract_config = config.get('handover_contract', {})
    cycle_config = config.get('visual_cycle', {})
    if cycle_config.get('enabled', False):
        if (not config['cartesian_alignment']['enabled']
                or config['cartesian_alignment'].get('strategy') != 'act_residual'
                or any(config['wrist_assist'].get(key, False) for key in
                       ('act_after_first_grasp', 'act_after_regrasp', 'handover_grip_guard',
                        'refresh_stale_handover_ensemble'))
                or contract_config.get('enabled', False)):
            raise ValueError('Visual completion requires residual grasping without ACT takeover/legacy guards')
        if cycle_config.get('max_parcels', 1) not in (1, 2, 3, 4):
            raise ValueError('visual_cycle.max_parcels must be 1 through 4')
        validate_layout(cycle_config)
        if not 0 < cycle_config.get('weighing_gain_scale', 1.) <= 1:
            raise ValueError('Visual weighing gain scale must be greater than zero and at most one')
    if contract_config.get('enabled', False):
        HandoverContract(contract_config)
        if (not config['cartesian_alignment']['enabled']
                or config['cartesian_alignment'].get('strategy') != 'act_residual'
                or not config['wrist_assist'].get('act_after_regrasp', False)
                or config['wrist_assist'].get('act_after_first_grasp', False)
                or config['wrist_assist'].get('handover_grip_guard', False)
                or config['wrist_assist'].get('refresh_stale_handover_ensemble', False)):
            raise ValueError('Handover contract requires regrasp takeover without legacy guard/refresh')
    if config["cartesian_alignment"]["enabled"] and config["servo"]["enabled"]:
        raise ValueError("Use Cartesian or pixel-Jacobian correction, not both")
    base.inference.eval_episodes = 1
    return config, base, expected_images


def preflight(config, base, images):
    import torch
    if base.inference.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Original ACT requests CUDA but CUDA is unavailable")
    print(json.dumps({
        "status": "preflight_ok", "model": base.inference.pretrained_path,
        "model_type": "act", "training": False, "image_shapes": images,
        "inference_hz": base.env.ros_rate, "max_steps": config["max_steps"],
        "direct_to_wbc": base.env.direct_to_wbc,
        "effective_head_init_sdk_radians": base.env.head_init,
        "visual_servo_enabled": config["servo"]["enabled"],
        "rgbd_grasp_alignment_enabled": config["cartesian_alignment"]["enabled"],
        "note": "No ROS connection, model execution or control commands in preflight",
    }, ensure_ascii=False, indent=2))


class SensorWatchdog:
    """Wall-clock receipt watchdog, independent of a stopped simulation clock."""
    def __init__(self, rospy, topic_map, timeout):
        import threading
        from sensor_msgs.msg import CompressedImage, JointState
        from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData, lejuClawState
        types = {"CompressedImage": CompressedImage, "JointState": JointState,
                 "sensorsData": sensorsData, "lejuClawState": lejuClawState}
        self.rospy = rospy
        self.timeout = timeout
        self.received = {}
        self.message_metadata = {}
        self.lock = threading.Lock()
        self.topics = sorted(topic_map)
        # rospy shares a topic transport within a process. AnyMsg here would
        # cause original typed ObsBuffer callbacks to receive undecoded bytes.
        self.subscribers = [rospy.Subscriber(topic, types[topic_map[topic]],
                            lambda message, topic=topic: self._receive(topic, message),
                            queue_size=1, buff_size=2**22) for topic in self.topics]

    def _receive(self, topic, message=None):
        with self.lock:
            self.received[topic] = time.monotonic()
            header = getattr(message, 'header', None)
            stamp = getattr(header, 'stamp', None)
            self.message_metadata[topic] = dict(
                receipt_monotonic_s=self.received[topic],
                source_stamp_s=stamp.to_sec() if stamp is not None else None,
                receipts=self.message_metadata.get(topic, {}).get('receipts', 0) + 1)

    def snapshot(self):
        # These are receipt-side metadata, not exact timestamps of the tensors
        # chosen by ObsBuffer. Do not imply frame alignment from these alone.
        with self.lock:
            return {topic: dict(value) for topic, value in self.message_metadata.items()}

    def ages(self):
        now = time.monotonic()
        with self.lock:
            return {topic: now - self.received[topic] if topic in self.received else None
                    for topic in self.topics}

    def ready(self):
        return all(age is not None and age <= self.timeout for age in self.ages().values())

    def check(self):
        ages = self.ages()
        stale = {topic: age for topic, age in ages.items() if age is None or age > self.timeout}
        if stale:
            raise RuntimeError(f"Sensor messages missing/stale; no next action sent: {stale}")
        return ages

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()


def run(args, config, base, images):
    # Reuse original preprocessing, ACT loading, bag replay and ROS environment.
    os.chdir(config["project_root"])
    import rosgraph
    import rospy
    import torch
    from lerobot_patches import custom_patches  # noqa: F401; same patch as original runner
    from kuavo_deploy.src.scripts.script import ArmMove, arm_controller
    from kuavo_deploy.utils.policy_loader import inject_task_prompt, load_native_policy_bundle

    if not rosgraph.is_master_online():
        raise RuntimeError("ROS master unavailable. Start official scene1 simulation first.")
    rospy.init_node("kuavo_deploy", anonymous=True)
    publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
    for topic, nodes in publishers:
        # Official template advertises arm_traj even though its task body is
        # empty; the IK/gripper bridge nodes also advertise standby interfaces.
        # They are present in the original ACT workflow, not competing policies.
        standby = ("/arms_ik_node", "/sim_leju_claw_interface")
        others = [node for node in nodes if node not in standby
                  and not (node == "/challenge_task_scene1" or node.startswith("/challenge_task_scene1_"))]
        if topic in ("/kuavo_arm_traj", "/gripper/command") and others:
            raise RuntimeError(f"Another node already publishes {topic}: {others}. Stop the other policy first.")
    topic_map = {entry["topic"]: entry["msg_type"] for entry in base.env.obs_key_map.values() if "topic" in entry}
    watchdog = SensorWatchdog(rospy, topic_map, config["sensor_timeout_s"])
    arm = None
    rgbd_scene = None
    output = ROOT / "outputs" / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    log = None
    steps = 0
    reason = "not_started"
    error = None
    policy_trace = None
    trace_window = PolicyTraceWindow(config.get('policy_trace', {}), output)
    contract = (HandoverContract(config['handover_contract'])
                if config.get('handover_contract', {}).get('enabled', False) else None)
    policy_epoch = 0
    epoch_inference_index = 0
    contract_stop = None
    visual_cycle = None
    visual_pickup = None
    pickup_template = None
    first_placement_gain = PlacementGain(config.get('visual_cycle', {}).get('first_weighing_lower_gain_scale', 1.))
    first_placement_target = PlacementTarget(config.get('visual_cycle', {}).get('first_weighing_roof_offset_m', 0.),
        config.get('visual_cycle', {}).get('first_weighing_roof_axis'))
    first_grasp_plan = None
    weighing_pad = None
    try:
        deadline = time.monotonic() + config["startup_timeout_s"]
        while not watchdog.ready():
            if rospy.is_shutdown() or time.monotonic() > deadline:
                raise RuntimeError(f"Sensor startup timed out: {watchdog.ages()}")
            time.sleep(0.05)
        policy, preprocess, postprocess, _ = load_native_policy_bundle(
            pretrained_path=base.inference.pretrained_path,
            device=torch.device(base.inference.device), strict=True)
        if policy.config.type != "act":
            raise RuntimeError("Loaded policy is not ACT")
        def reset_policy():
            nonlocal policy_epoch, epoch_inference_index
            policy.reset()
            policy_epoch += 1
            epoch_inference_index = 0
        if (config['wrist_assist'].get('trace_act_handover', False)
                or config['wrist_assist'].get('refresh_stale_handover_ensemble', False)
                or trace_window.enabled):
            policy_trace = ACTChunkTrace(policy)
        watchdog.check()
        arm = ArmMove(base)
        env = arm.env
        if args.mode == "optimized" and config["cartesian_alignment"].get("strategy") == "act_residual":
            env._set_direct_to_wbc(0)
        if args.task == "go_run":
            # A freshly reset scene starts with internal arm control. Switch
            # before replaying the preparation bag, not only afterward in reset.
            env._enter_external_control_mode()
            arm.go()
            if arm_controller.should_exit() or rospy.is_shutdown():
                reason = "stopped_during_go"
                return
        # The original reset opens both grippers. A held-parcel recovery must
        # initialize interpolation/observations without dropping its payload.
        observation, _ = reset_for_manipulation(env,
            keep_head=any((args.resume_from, args.regrasp_from, args.pickup_from, args.act_handover_from, args.visual_cycle_from, args.visual_next_from, args.visual_weigh_from)),
            keep_grippers=any((args.resume_from, args.act_handover_from, args.visual_cycle_from, args.visual_next_from, args.visual_weigh_from)))
        observation_received = watchdog.snapshot()
        if args.mode == "optimized" and env.direct_to_wbc:
            env.low_pass_filter = ActionPassthrough()
        initial = observation["observation.state"].detach().cpu().numpy().reshape(-1)
        handover_command = None
        if args.act_handover_from:
            handover_command = validate_act_handover_resume(load_run_history(args.act_handover_from),initial)
        if args.task == "go_run" and np.linalg.norm(initial[[*range(7), *range(8, 15)]]) < 0.15:
            raise RuntimeError("Preparation commands had no joint response. Start the simulation controller before ACT.")
        watchdog.check()
        motion = MotionFilter(env.action_space.low, env.action_space.high,
                              config["motion"]["smoothing_alpha"],
                              config["motion"]["max_joint_velocity_rad_s"])
        motion.reset(observation["observation.state"].detach().cpu().numpy().reshape(-1))
        detector = ParcelDetector(config["vision"])
        wrist_detectors = {side: ParcelDetector({**config["vision"], "roi": [0, 0, 1, 1], "table_segmentation": False})
                           for side in ("left", "right")}
        grasp_monitor = GraspMonitor(config["grasp"])
        servo_detector = ParcelDetector({**config["vision"], "roi": [0, 0, 1, 1], "table_segmentation": False})
        servo = VisualServo(config["servo"])
        aligner = None
        head_tracker = None
        if args.mode == "optimized" and config["cartesian_alignment"]["enabled"]:
            rgbd_scene = RgbdScene(rospy, config["cartesian_alignment"])
            kinematics = UrdfArmKinematics(rospy)
            for side,start in enumerate((0,8)):
                motion.low[start:start+7] = np.maximum(motion.low[start:start+7],kinematics.limits[side][:,0])
                motion.high[start:start+7] = np.minimum(motion.high[start:start+7],kinematics.limits[side][:,1])
            motion.reset(initial)
            if config['cartesian_alignment'].get('strategy') == 'act_residual':
                aligner = ResidualAssist(config['visual_residual'], kinematics,
                                         env.action_space.low, env.action_space.high)
                if config['wrist_assist']['enabled']:
                    aligner.wrist_projector = WristProjector(
                        rospy, rgbd_scene.buffer, kinematics, config['wrist_assist'])
                    aligner.wrist_config = config['wrist_assist']
            else:
                aligner = CartesianGraspAligner(config['cartesian_alignment'], kinematics,
                                                env.action_space.low, env.action_space.high)
            if any((args.resume_from, args.regrasp_from, args.pickup_from)):
                deadline = time.monotonic() + config["startup_timeout_s"]
                while True:
                    try:
                        snapshot = rgbd_scene.read()
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.05)
                records = load_run_history(args.resume_from or args.regrasp_from or args.pickup_from)
                if args.pickup_from is not None:
                    aligner.restore_pickup(records, initial, snapshot)
                elif args.regrasp_from is not None:
                    aligner.restore_regrasp(records, initial, snapshot,
                        load_last_arm_command(args.arm_state_from) if args.arm_state_from else None,
                        wrist_roll_degrees=args.wrist_roll_degrees)
                else:
                    aligner.restore_weighing(records, initial, snapshot)
                motion.reset(load_last_arm_command(args.arm_state_from) if args.arm_state_from else records[-1]['sent_action'])
            if config['head_camera']['enabled']:
                head_tracker = HeadLookController(config['head_camera'],
                    rospy.get_param(config['cartesian_alignment']['description_param']))
        reset_policy()
        if config.get('visual_cycle', {}).get('enabled', False):
            visual_cycle = VisualCycle(config['visual_cycle'], kinematics,
                                       env.action_space.low, env.action_space.high, initial)
            if args.visual_cycle_from:
                saved = json.loads((Path(args.visual_cycle_from)/'visual-cycle-checkpoint.json').read_text())
                if not saved.get('ros_run_id') or saved['ros_run_id'] != rospy.get_param('/run_id', None):
                    raise RuntimeError('Visual resume source is from a different ROS scene session')
                visual_cycle.restore(saved['controller'], initial, saved['measured_state'])
                pickup_template = saved.get('pickup_template')
                if saved.get('weighing_pad') is not None:
                    weighing_pad = np.asarray(saved['weighing_pad'], float)
                motion.reset(visual_cycle.command)
        period = 1.0 / env.ros_rate
        last_send = time.monotonic()
        previous_sent = initial.copy()
        if visual_cycle is not None and visual_cycle.active:
            previous_sent = visual_cycle.command.copy()
        if args.visual_next_from or args.visual_weigh_from:
            restore_context = restore_weighing_context if args.visual_weigh_from else restore_next_context
            pickup_template, weighing_pad, previous_sent, lift_recovery = restore_context(
                args.visual_weigh_from or args.visual_next_from, visual_cycle, aligner, initial, rospy.get_param('/run_id', None))
            motion.reset(previous_sent)
            aligner.observe_command(previous_sent)
            visual_pickup = VisualPickup(visual_cycle, pickup_template, weighing_pad, initial, previous_sent, lift_recovery)
        act_handover = args.act_handover_from is not None
        pure_act_takeover = False
        handover_grip_guard = HandoverGripGuard(config['wrist_assist'].get('handover_contact_frames',3))
        handover_phase_refresh = HandoverPhaseRefresh()
        if handover_command is not None:
            previous_sent = handover_command.copy()
            motion.reset(handover_command)
            if contract is not None:
                contract.begin(initial, previous_sent, 1, time.monotonic())
            trace_window.trigger(1, 'act_handover_resume')
        paused_after_grasp = False
        paused_after_weighing = False
        log = (output / "steps.jsonl").open("w", buffering=1)
        (output / "run.json").write_text(json.dumps({
            "mode": args.mode, "task": args.task, "config": config,
            "stop_after_parcels": args.stop_after_parcels,
            "original_model": base.inference.pretrained_path,
            "base_config": config["base_config"],
            "resume_from": str(Path(args.resume_from).resolve()) if args.resume_from else None,
            "regrasp_from": str(Path(args.regrasp_from).resolve()) if args.regrasp_from else None,
            "pickup_from": str(Path(args.pickup_from).resolve()) if args.pickup_from else None,
            "arm_state_from": str(Path(args.arm_state_from).resolve()) if args.arm_state_from else None,
            "act_handover_from": str(Path(args.act_handover_from).resolve()) if args.act_handover_from else None,
            "visual_next_from": args.visual_next_from,
            "visual_weigh_from": args.visual_weigh_from,
            "visual_cycle_from": str(Path(args.visual_cycle_from).resolve()) if args.visual_cycle_from else None,
            "task_success": None,
            "policy_runtime": {key: getattr(policy.config, key, None) for key in
                               ('chunk_size', 'n_obs_steps', 'n_action_steps', 'temporal_ensemble_coeff')},
        }, ensure_ascii=False, indent=2))
        for step in range(config["max_steps"]):
            while arm_controller.is_paused() and not arm_controller.should_exit() and not rospy.is_shutdown():
                time.sleep(0.05)
                # Refresh observation on resume, not a pre-pause image.
                observation = env.get_obs()
                observation_received = watchdog.snapshot()
            if arm_controller.should_exit() or rospy.is_shutdown():
                reason = "user_stop"
                break
            started = time.monotonic()
            ages = watchdog.check()
            trace_payload = None
            selected_normalized = None
            act_before_refresh = None
            if contract is not None:
                contract.check_deadline(steps + 1, started)
            if (not act_handover and isinstance(aligner,ResidualAssist)
                    and aligner.first_grasp_complete and aligner.pending is None
                    and config['wrist_assist'].get('act_after_first_grasp',False)):
                # This branch is permanent for the remainder of the episode.
                # Drop the custom controllers themselves, so later ACT release,
                # regrasp and handover events cannot invoke them again.
                act_handover = pure_act_takeover = True
                reset_policy()
                trace_window.trigger(steps + 1, 'act_after_first_grasp')
                aligner = None
                head_tracker = None
                if rgbd_scene is not None:
                    rgbd_scene.close()
                    rgbd_scene = None
                print(f'ACT_DIRECT_AFTER_FIRST_GRASP step={steps+1}',flush=True)
            for key, shape in images.items():
                if list(observation[key].shape) != [1] + shape:
                    raise ValueError(f"Runtime image shape mismatch: {key}: {observation[key].shape}")
            raw_image = rgb_image(observation[config["vision"]["image_key"]])
            detections = detector.detect(raw_image)
            wrist_images = {
                "left": rgb_image(observation["observation.images.wrist_cam_l"]),
                "right": rgb_image(observation["observation.images.wrist_cam_r"]),
            }
            wrist_detections = {side: wrist_detectors[side].detect(image)
                                for side, image in wrist_images.items()}
            if isinstance(aligner, ResidualAssist):
                aligner.observe_wrist(wrist_detections,
                    {side: [image.shape[1], image.shape[0]] for side, image in wrist_images.items()},
                    wrist_images)
            rgbd_snapshot = None
            rgbd_error = None
            if aligner is not None:
                try:
                    rgbd_snapshot = read_manipulation_rgbd(
                        rgbd_scene, aligner, visual_cycle, visual_pickup, config)
                except Exception as exc:
                    rgbd_error = str(exc)
            measured = observation['observation.state'].detach().cpu().numpy().reshape(-1)
            if (pickup_template is None and isinstance(aligner, ResidualAssist)
                    and aligner.first_grasp_complete and not aligner.regrasp_complete
                    and aligner.geometry[1] is not None):
                if config.get('visual_cycle', {}).get('visual_after_first_parcel') and first_grasp_plan is None:
                    raise RuntimeError('No requested first grasp target was captured for visual continuation')
                pickup_template = grasp_template(aligner.geometry[1], first_grasp_plan)
            if weighing_pad is None and isinstance(aligner, ResidualAssist) and aligner.weighing_target is not None:
                weighing_pad = aligner.weighing_target.copy()
            if visual_cycle is not None:
                if step % 10 == 0 or visual_cycle.active:
                    visual_cycle.observe_scene(rgbd_snapshot)
                if not visual_cycle.active and aligner.regrasp_complete and aligner.pending is None:
                    visual_cycle.begin(aligner, measured, previous_sent)
                    print(f'VISUAL_CYCLE_AFTER_REGRASP step={steps+1} parcel={visual_cycle.completed+1}', flush=True)
            if (not act_handover and isinstance(aligner,ResidualAssist)
                    and aligner.regrasp_complete and aligner.pending is None
                    and config['wrist_assist'].get('act_after_regrasp',False)):
                act_handover = True
                reset_policy()  # infer a fresh chunk after the confirmed weighing regrasp
                trace_window.trigger(steps + 1, 'act_after_regrasp')
                if contract is not None:
                    contract.begin(measured, previous_sent, steps + 1, started)
                print(f'ACT_DIRECT_AFTER_REGRASP step={steps+1}',flush=True)
            head_info = {'state': 'disabled'}
            head_returning = False
            if head_tracker is not None:
                goal = (visual_pickup.camera_target(measured, aligner) if visual_pickup is not None and visual_pickup.active
                        else visual_cycle.camera_target(measured) if visual_cycle is not None and visual_cycle.active
                        else None if act_handover else aligner.camera_target(measured))
                command, head_info = head_tracker.update(goal, rgbd_scene.head_angles(), rgbd_snapshot, period)
                env.robot.control_head(float(command[0]), float(command[1]))
                head_returning = goal is None and not head_info['act_view_ready']
            cycle_active = visual_cycle is not None and visual_cycle.active
            pickup_active = visual_pickup is not None and visual_pickup.active
            frozen = pickup_active or cycle_active or (aligner is not None and ((aligner.pending is not None and not act_handover) or head_returning))
            prediction_info = None
            if trace_window.wants_payload():
                # Copy BEFORE preprocessing: a processor may replace or modify
                # tensors. The archive must contain the original policy input.
                trace_payload = {key: value.detach().cpu().numpy().copy()
                                 for key, value in observation.items() if key.startswith('observation.')}
            if frozen:
                raw_action = (previous_sent.copy() if head_returning or cycle_active or pickup_active else aligner.held_action.copy())
            else:
                model_input = preprocess(inject_task_prompt(dict(observation), base.inference.task_prompt))
                with torch.inference_mode():
                    if policy_trace is not None:
                        policy_trace.begin_inference()
                    normalized_action = policy.select_action(model_input)
                    epoch_inference_index += 1
                    if trace_window.wants_payload():
                        selected_normalized = normalized_action.detach().clone()
                    action = postprocess(normalized_action)
                    if policy_trace is not None and (act_handover or trace_window.enabled):
                        prediction_info = policy_trace.describe(postprocess)
                        if (act_handover and not pure_act_takeover and prediction_info is not None
                                and config['wrist_assist'].get('refresh_stale_handover_ensemble', False)):
                            fresh = prediction_info['fresh_chunk'][0]['action']
                            refresh, refresh_info = handover_phase_refresh.update(
                                action.detach().cpu().numpy().reshape(-1), fresh, measured)
                            prediction_info['phase_refresh'] = refresh_info
                            if refresh:
                                # Same current observation and same predicted chunk;
                                # discard only the conflicting historical ensemble.
                                act_before_refresh = action.detach().cpu().numpy().reshape(-1).tolist()
                                reset_policy()
                                normalized_action = policy.temporal_ensembler.update(policy_trace.chunk.clone())
                                epoch_inference_index = 1
                                if trace_window.wants_payload():
                                    selected_normalized = normalized_action.detach().clone()
                                action = postprocess(normalized_action)
                                print(f'ACT_HANDOVER_PHASE_REFRESH step={steps+1}',flush=True)
                raw_action = action_vector(action.detach().cpu().numpy().reshape(-1))
            if trace_payload is not None and not frozen:
                if policy_trace.chunk is not None:
                    trace_payload['normalized_chunk'] = policy_trace.chunk.cpu().numpy().copy()
                trace_payload['selected_normalized'] = selected_normalized.cpu().numpy().copy()
            inferred = time.monotonic()
            if config["servo"]["enabled"]:
                servo_image = rgb_image(observation[config["servo"]["image_key"]])
                servo_detections = servo_detector.detect(servo_image)
                residual, servo_info = servo.apply(raw_action, servo_detections,
                                                   [servo_image.shape[1], servo_image.shape[0]])
            else:
                residual, servo_info = raw_action, {"servo": "disabled"}
            cartesian_info = ({'state':'act_direct_after_first_grasp','strategy':'act',
                               'custom_controllers_disabled':True} if pure_act_takeover
                              else {"state": "disabled"})
            if aligner is not None:
                if cycle_active:
                    residual, cartesian_info = visual_cycle.update(measured, rgbd_snapshot, aligner)
                elif pickup_active:
                    residual, cartesian_info = visual_pickup.update(measured, rgbd_snapshot, aligner, head_info)
                elif head_returning:
                    residual = previous_sent.copy()
                    cartesian_info = {'state': 'returning_policy_view'}
                elif act_handover:
                    # ACT controls both arms; only prevent a premature giver
                    # release until the receiver has stable contact feedback.
                    residual = raw_action.copy()
                    cartesian_info = {'state':'act_direct','strategy':'act'}
                    if config['wrist_assist'].get('handover_grip_guard',False):
                        residual,grip_info = handover_grip_guard.apply(raw_action,measured)
                        cartesian_info['handover_grip_guard'] = grip_info
                else:
                    if isinstance(aligner, ResidualAssist):
                        first_placement_gain.before_step(aligner)
                        first_placement_target.before_step(aligner)
                    residual, cartesian_info = aligner.apply(raw_action, measured, rgbd_snapshot)
                    if isinstance(aligner, ResidualAssist):
                        first_placement_gain.after_step(aligner)
                        if first_placement_target.trace(aligner) is not None:
                            cartesian_info['placement_target_compensation'] = first_placement_target.trace(aligner)
                if (pickup_template is None and isinstance(aligner, ResidualAssist)
                        and aligner.pending is not None and aligner.pending['kind'] == 'grasp'
                        and 'support_z_base_m' not in aligner.pending['target']
                        and cartesian_info.get('grasp_target_base_m') is not None):
                    first_grasp_plan = grasp_plan(aligner.pending['target'], aligner.pending['plane'],
                                                 cartesian_info['grasp_target_base_m'])
                cartesian_info["sensor_error"] = rgbd_error
                cartesian_info["candidate_count"] = len(rgbd_snapshot["candidates"]) if rgbd_snapshot else 0
            dt = min(period, max(1e-4, time.monotonic() - last_send))
            contract_info = None
            if contract is not None and contract.active and not frozen:
                feedback = observation_received.get('/gripper/state', {})
                stamp = feedback.get('source_stamp_s')
                token = ('stamp', stamp) if stamp else ('receipt', feedback.get('receipts'))
                try:
                    residual, contract_info = contract.apply(residual, measured, previous_sent,
                        step=steps + 1, now=time.monotonic(), dt=dt,
                        feedback_token=token if feedback else None)
                    residual = action_vector(residual)
                except HandoverContractStop as exc:
                    # Save the rejected proposal and its exact inputs; no command
                    # is sent and no environment reset is performed here.
                    if trace_payload is not None:
                        trace_payload.update(act_action=raw_action, proposed_action=np.asarray(residual))
                        trace_window.trigger(steps + 1, 'contract_stop')
                        trace_window.capture(steps + 1, trace_payload, dict(
                            step=steps + 1, policy_epoch=policy_epoch,
                            epoch_inference_index=epoch_inference_index,
                            policy_paused=False, action_sent=False,
                            observation_receipts=observation_received, contract=exc.details))
                    raise
            motion.max_velocity = (config["cartesian_alignment"]["approach_joint_velocity_rad_s"]
                                   if aligner is not None and aligner.pending is not None and not act_handover
                                   else config["motion"]["max_joint_velocity_rad_s"])
            filtered, motion_info = motion.apply(residual, dt)
            # Original env.step clips against the same action_space.
            sent = filtered if args.mode == "optimized" else np.clip(raw_action, env.action_space.low, env.action_space.high)
            # Observe mode compares a hypothetical filter against measured state.
            # Its internal previous target must follow the actual sent action.
            if args.mode != "optimized":
                motion.previous = np.clip(raw_action, env.action_space.low, env.action_space.high)
            send_ages = watchdog.check()  # model inference may have outlasted sensor freshness
            if arm_controller.should_exit() or rospy.is_shutdown():
                reason = "user_stop_before_send"
                break
            if arm_controller.is_paused():
                # Do not send an action inferred before a pause request.
                observation = env.get_obs()
                observation_received = watchdog.snapshot()
                continue
            last_send = time.monotonic()
            next_observation, reward, terminated, truncated, info = env.step(sent)
            next_observation_received = watchdog.snapshot()
            previous_sent = sent.copy()
            if cycle_active:
                visual_cycle.command = sent.copy()
            if pickup_active:
                visual_pickup.motion.command = sent.copy()
            if aligner is not None:
                aligner.observe_command(sent)
            grasp_info = grasp_monitor.update(
                sent, next_observation["observation.state"].detach().cpu().numpy().reshape(-1),
                wrist_detections,
                {side: [image.shape[1], image.shape[0]] for side, image in wrist_images.items()})
            steps += 1
            ended = time.monotonic()
            if visual_cycle is not None and (visual_cycle.active or visual_cycle.done):
                checkpoint_state = dict(schema=1, ros_run_id=rospy.get_param('/run_id', None),
                    step=steps, controller=visual_cycle.state_dict(),
                    pickup_template=pickup_template, weighing_pad=weighing_pad.tolist() if weighing_pad is not None else None,
                    measured_state=next_observation['observation.state'].detach().cpu().numpy().reshape(-1).tolist())
                checkpoint_path = output/'visual-cycle-checkpoint.json'
                temporary = output/'visual-cycle-checkpoint.tmp'
                temporary.write_text(json.dumps(checkpoint_state))
                temporary.replace(checkpoint_path)
            if pickup_active:
                write_pickup_checkpoint(output, steps, rospy.get_param('/run_id', None),
                    visual_pickup, visual_cycle, aligner,
                    next_observation['observation.state'].detach().cpu().numpy().reshape(-1))
            record = {
                "step": steps, "mode": args.mode,
                "inference_s": inferred - started, "step_s": ended - started,
                "sensor_age_s": ages, "detections": detections,
                "sensor_age_before_send_s": send_ages,
                "observation_receipts": observation_received,
                "send_monotonic_s": last_send,
                "act_action": raw_action.tolist(), "proposed_action": residual.tolist(),
                "filtered_action": filtered.tolist(), "sent_action": sent.tolist(),
                "motion": motion_info, "servo": servo_info,
                "head_camera": head_info,
                "wrist_detections": wrist_detections, "grasp": grasp_info,
                "cartesian_alignment": cartesian_info, "policy_paused_for_alignment": frozen,
                "act_prediction": prediction_info,
                "act_before_refresh": act_before_refresh,
                "policy_epoch": policy_epoch,
                "epoch_inference_index": epoch_inference_index,
                "handover_contract": contract_info,
                "measured_state": next_observation["observation.state"].detach().cpu().numpy().reshape(-1).tolist(),
                "observation_state": measured.tolist(),
                "rgbd_stamp": rgbd_snapshot.get('stamp') if rgbd_snapshot else None,
                "rgbd_candidates": rgbd_snapshot["candidates"] if rgbd_snapshot else [],
                "rgbd_spatial_components": bool(rgbd_snapshot and rgbd_snapshot.get('candidate_spatial_components')),
                "rgbd_surfaces": rgbd_snapshot.get("surfaces", []) if rgbd_snapshot else [],
                "rgbd_table_z_base_m": rgbd_snapshot.get("table_z_base_m") if rgbd_snapshot else None,
                "terminated": bool(terminated), "truncated": bool(truncated),
            }
            if steps % config["log_every_steps"] == 0:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            grasp_check = (cartesian_info.get('state') == 'grasp_contact_confirmed')
            checkpoint = cartesian_info.get('state') in (
                'weighing_release', 'short_retreat_complete', 'placement_stable',
                'pregrasp_aligned', 'regrasp_pose_verified')
            checkpoint = checkpoint or bool(cartesian_info.get('transition_to'))
            if trace_payload is not None:
                trace_payload.update(act_action=raw_action, proposed_action=residual,
                                     filtered_action=filtered, sent_action=sent)
                trace_window.capture(steps, trace_payload, dict(
                    step=steps, policy_epoch=policy_epoch, epoch_inference_index=epoch_inference_index,
                    policy_paused=bool(frozen), action_sent=True,
                    observation_receipts=observation_received,
                    send_monotonic_s=last_send, sensor_age_before_send_s=send_ages,
                    cartesian_state=cartesian_info.get('state'), head_camera=head_info))
            if prediction_info is not None and (steps % 20 == 0 or steps == 1):
                np.savez_compressed(output / f'policy-observation-{steps:04d}.npz',
                    **{key: value.detach().cpu().numpy() for key, value in observation.items()
                       if key.startswith('observation.')})
            if steps == 1 or steps % config["snapshot_every_steps"] == 0 or grasp_check or checkpoint:
                cv2.imwrite(str(output / f"vision-{steps:04d}.jpg"), detector.annotate(raw_image, detections))
                for side in ("left", "right"):
                    cv2.imwrite(str(output / f"wrist-{side}-{steps:04d}.jpg"),
                                wrist_detectors[side].annotate(wrist_images[side], wrist_detections[side]))
                if (checkpoint or config.get('save_periodic_rgbd', False)) and rgbd_snapshot is not None:
                    sensor_payload = dict(rgb=rgbd_snapshot['rgb'], depth_m=rgbd_snapshot['depth_m'],
                        intrinsic=rgbd_snapshot['intrinsic'], camera_to_base=rgbd_snapshot['camera_to_base'],
                        observation_state=measured, wrist_right=wrist_images['right'],
                        wrist_left=wrist_images['left'])
                    # Diagnostics only: the estimate after processing this
                    # observation, derived from RGB-D/FK rather than scene truth.
                    geometry = getattr(aligner, 'geometry', [None, None])[1]
                    if config.get('save_periodic_rgbd', False) and geometry is not None:
                        sensor_payload.update(held_center_tool_m=geometry['center'],
                            held_rotation_tool_xyzw=geometry['rotation'].as_quat(),
                            held_extent_m=geometry['extent'])
                    np.savez_compressed(output / f'sensors-{steps:04d}.npz', **sensor_payload)
                print(f"step={steps} candidates={[d['label'] for d in detections]} "
                      f"limited={motion_info['velocity_limited']} mode={args.mode}")
            observation = next_observation
            observation_received = next_observation_received
            if visual_cycle is not None and visual_cycle.done:
                if visual_cycle.completed >= min(args.stop_after_parcels or 4, config['visual_cycle'].get('max_parcels', 1)):
                    reason = 'visual_cycle_target_reached'
                    break
                # A new parcel gets a new local state machine. Do not env.reset:
                # the hands have already opened and withdrawn under feedback.
                projector = aligner.wrist_projector
                aligner = ResidualAssist(config['visual_residual'], kinematics,
                                         env.action_space.low, env.action_space.high)
                aligner.wrist_projector, aligner.wrist_config = projector, config['wrist_assist']
                aligner.observe_command(sent)
                visual_cycle.done = False
                visual_cycle.stage = 'idle'
                if config['visual_cycle'].get('visual_after_first_parcel', False):
                    if pickup_template is None or weighing_pad is None:
                        raise RuntimeError('Missing observed first grasp or weighing pad for visual continuation')
                    visual_pickup = VisualPickup(visual_cycle, pickup_template, weighing_pad,
                        next_observation['observation.state'].detach().cpu().numpy().reshape(-1), sent)
                    print(f'NEXT_PARCEL_VISUAL step={steps+1} completed={visual_cycle.completed}', flush=True)
                else:
                    reset_policy()
                    trace_window.trigger(steps+1, 'next_parcel_act')
                    print(f'NEXT_PARCEL_ACT step={steps+1} completed={visual_cycle.completed}', flush=True)
            if args.pause_after_grasp and grasp_check and not paused_after_grasp:
                paused_after_grasp = True
                arm_controller.pause()
                print(f'PAUSED_AFTER_GRASP step={steps} output={output}', flush=True)
            if (args.stop_after_regrasp and aligner is not None
                    and cartesian_info.get('state') == 'verify_regrasp'
                    and cartesian_info.get('grasp_verified') and aligner.pending is None):
                reason = 'weighing_regrasp_verified'
                break
            weighing_check = (aligner is not None
                    and cartesian_info.get('state') == 'release_dwell'
                    and cartesian_info.get('arm') == 1 and aligner.pending is None
                    and aligner.weighing_target is not None)
            if args.pause_after_weighing and weighing_check and not paused_after_weighing:
                paused_after_weighing = True
                arm_controller.pause()
                print(f'PAUSED_AFTER_WEIGHING step={steps} output={output}', flush=True)
            if args.stop_after_weighing and weighing_check:
                reason = 'weighing_release_completed'
                break
            if terminated or truncated:
                reason = "environment_terminated" if terminated else "environment_truncated"
                break
            # The original environment already sleeps; only fill remaining time.
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        else:
            reason = "max_steps_reached"
        print(f"Stopped: {reason}; steps={steps}; task success requires scene/rule evaluation. Logs: {output}")
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
        print("Stopped by keyboard; no further commands will be sent.")
    except Exception as exc:
        reason = 'handover_contract_stop' if isinstance(exc, HandoverContractStop) else 'error'
        if isinstance(exc, HandoverContractStop):
            contract_stop = exc.details
        error = str(exc)
        if 'measured' in locals():
            failed = dict(step=steps+1, error=error, observation_state=measured.tolist(),
                          rgbd_error=rgbd_error,
                          rgbd_candidates=rgbd_snapshot['candidates'] if rgbd_snapshot else [],
                          rgbd_spatial_components=bool(rgbd_snapshot and rgbd_snapshot.get('candidate_spatial_components')),
                          rgbd_stamp=rgbd_snapshot.get('stamp') if rgbd_snapshot else None,
                          pending_kind=aligner.pending.get('kind') if aligner and aligner.pending else None)
            (output / 'failed_observation.json').write_text(json.dumps(failed,indent=2))
            for side, rgb in wrist_images.items():
                cv2.imwrite(str(output / f'failed-wrist-{side}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            if rgbd_snapshot is not None:
                np.savez_compressed(output / 'failed-sensors.npz',rgb=rgbd_snapshot['rgb'],
                    depth_m=rgbd_snapshot['depth_m'], intrinsic=rgbd_snapshot['intrinsic'],
                    camera_to_base=rgbd_snapshot['camera_to_base'], observation_state=measured)
        raise
    finally:
        if policy_trace is not None:
            policy_trace.close()
        if log is not None:
            log.close()
        watchdog.close()
        if rgbd_scene is not None:
            rgbd_scene.close()
        if arm is not None:
            arm.env.close()
        trace_manifest_error = None
        try:
            trace_window.close()
        except OSError as exc:
            trace_manifest_error = str(exc)
            print(f'Trace manifest could not be saved: {exc}', flush=True)
        (output / "summary.json").write_text(json.dumps({
            "steps": steps, "reason": reason, "task_success": None,
            "error": error,
            "handover_contract_stop": contract_stop,
            "last_sensor_receipts": watchdog.snapshot(),
            "trace_manifest_error": trace_manifest_error,
            "visual_confirmed_parcels": visual_cycle.completed if visual_cycle is not None else None,
            "note": "Original KuavoSimEnv returns reward=0 and does not judge scene1 success.",
        }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--task", choices=("run", "go_run"), default="go_run")
    parser.add_argument("--mode", choices=("baseline", "observe", "optimized"), default="optimized")
    parser.add_argument("--preflight", action="store_true", help="Read configuration/model metadata only; send no commands")
    parser.add_argument("--max-steps", type=int, help="Override configured step cap for a short simulation test")
    parser.add_argument("--resume-from", help="Continue weighing a held parcel using this previous run directory")
    parser.add_argument("--act-handover-from", help="Continue a confirmed right-hand regrasp with original ACT arms/grippers")
    parser.add_argument("--visual-weigh-from", help="Resume observed right-hand payload into visual preparation for weighing in the same scene")
    parser.add_argument("--visual-next-from", help="Continue next parcel visually in the same empty-handed scene")
    parser.add_argument("--visual-cycle-from", help="Continue this same live scene from a visual completion checkpoint, preserving grippers")
    parser.add_argument("--stop-after-parcels", type=int, choices=(1, 2, 3, 4), help="Stop after this many confirmed parcels including prior scene completions")
    parser.add_argument("--stop-after-weighing", action="store_true", help="Stop after the first weighing release for visual inspection")
    parser.add_argument("--pause-after-weighing", action="store_true", help="Pause once after weighing release, retaining ACT state for inspection and regrasp")
    parser.add_argument("--pause-after-grasp", action="store_true", help="Pause once after confirmed grasp, retaining ACT state for inspection")
    parser.add_argument("--regrasp-from", help="Pick a parcel still on the weighing pad after this completed run")
    parser.add_argument("--stop-after-regrasp", action="store_true", help="Stop holding after verifying the weighing regrasp")
    parser.add_argument("--pickup-from", help="Retry an interrupted open pickup using fresh RGB-D geometry")
    parser.add_argument("--arm-state-from", help="Latest empty-arm tracking log after retreating from the weighing release")
    parser.add_argument("--wrist-roll-degrees", type=float, choices=(0., 180.), default=0., help="Rotate the gripper about its approach axis for regrasp")
    args = parser.parse_args()
    # Original environment setup changes cwd. Resolve recovery paths before
    # importing it, and reject a missing checkpoint before initializing ROS control.
    for name in ('resume_from', 'regrasp_from', 'pickup_from', 'arm_state_from',
                 'act_handover_from', 'visual_cycle_from', 'visual_next_from', 'visual_weigh_from'):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    if args.visual_weigh_from and not (Path(args.visual_weigh_from)/'visual-pickup-checkpoint.json').is_file():
        parser.error('--visual-weigh-from checkpoint does not exist')
    if args.visual_next_from and not (Path(args.visual_next_from)/'run.json').is_file():
        parser.error('--visual-next-from checkpoint does not exist')
    if args.visual_cycle_from and not (Path(args.visual_cycle_from)/'visual-cycle-checkpoint.json').is_file():
        parser.error('--visual-cycle-from checkpoint does not exist')
    if args.arm_state_from and not args.regrasp_from:
        parser.error("--arm-state-from requires --regrasp-from")
    if sum(bool(v) for v in (args.resume_from, args.regrasp_from, args.pickup_from, args.act_handover_from, args.visual_cycle_from, args.visual_next_from, args.visual_weigh_from)) > 1:
        parser.error("Recovery source options are mutually exclusive")
    if args.act_handover_from and (args.task != 'run' or args.mode != 'optimized'):
        parser.error('--act-handover-from requires --task run --mode optimized')
    if args.pickup_from and (args.task != "run" or args.mode != "optimized"):
        parser.error("--pickup-from requires --task run --mode optimized")
    if args.resume_from is not None and (args.task != 'run' or args.mode != 'optimized'):
        parser.error('--resume-from requires --task run --mode optimized')
    if args.stop_after_weighing and args.mode != 'optimized':
        parser.error('--stop-after-weighing requires --mode optimized')
    if args.regrasp_from is not None:
        if args.task != 'run' or args.mode != 'optimized' or args.resume_from is not None:
            parser.error('--regrasp-from requires --task run --mode optimized and excludes --resume-from')
        summary = json.loads((Path(args.regrasp_from) / 'summary.json').read_text())
        released = (summary['reason'] == 'weighing_release_completed' or
                    any(r['cartesian_alignment'].get('state') == 'weighing_release'
                        for r in load_run_history(args.regrasp_from)))
        if not released:
            parser.error('--regrasp-from source must have completed weighing release')
    if args.stop_after_regrasp and args.regrasp_from is None:
        parser.error('--stop-after-regrasp requires --regrasp-from')
    config, base, images = load_config(args.config)
    if config.get('handover_contract', {}).get('enabled', False) and args.mode != 'optimized':
        parser.error('Handover contract experiments require --mode optimized')
    if config.get('visual_cycle', {}).get('enabled', False) and args.mode != 'optimized':
        parser.error('Visual completion requires --mode optimized')
    if (args.visual_next_from or args.visual_weigh_from) and (args.task != 'run' or args.mode != 'optimized' or not config.get('visual_cycle', {}).get('visual_after_first_parcel', False)):
        parser.error('--visual-next-from requires --task run --mode optimized and visual_after_first_parcel')
    if args.visual_cycle_from and (args.task != 'run' or not config.get('visual_cycle', {}).get('enabled', False)):
        parser.error('--visual-cycle-from requires --task run and visual_cycle.enabled')
    if (config['cartesian_alignment'].get('strategy') == 'act_residual'
            and any((args.resume_from, args.pickup_from))):
        parser.error('ACT residual recovery currently supports --regrasp-from only')
    # SDK accepts radians in every mode. Preserve the original intended 20°
    # camera pose for pure ACT as well as assisted runs.
    if config["motion"].get("head_init_degrees") is not None:
        base.env.head_init = head_radians(config["motion"]["head_init_degrees"])
    if args.mode == "optimized":
        base.env.direct_to_wbc = config["motion"].get("direct_to_wbc", base.env.direct_to_wbc)
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("--max-steps must be positive")
        config["max_steps"] = args.max_steps
    preflight(config, base, images)
    if not args.preflight:
        import fcntl
        with (ROOT / ".run.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another scene1 extension run is active in this container")
            run(args, config, base, images)


if __name__ == "__main__":
    main()
