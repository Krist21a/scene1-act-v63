import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from calibrate_servo import fit
from scene1.motion import GRIPPERS, JOINTS, MotionFilter, head_radians
from scene1.grasp import GraspMonitor
from scene1.servo import VisualServo
from scene1.vision import ParcelDetector, rgb_image
from run_scene1 import SensorWatchdog

ROOT = Path(__file__).resolve().parents[1]


class MotionTests(unittest.TestCase):
    def test_physical_stop_roundoff_is_clipped_without_widening_commands(self):
        low = np.r_[[-1.]*7,0.,[-1.]*7,0.]
        high = np.ones(16)
        motion = MotionFilter(low,high)
        measured = np.zeros(16)
        for roundoff in (1.2e-5, 2.1e-4):
            measured[4] = -1.-roundoff
            motion.reset(measured)
            result,_ = motion.apply(measured,.1)
            self.assertEqual(result[4],-1.)
        measured[4] = -1.001
        with self.assertRaises(ValueError):
            motion.reset(measured)

    def setUp(self):
        self.low = np.full(16, -3.14)
        self.high = np.full(16, 3.14)
        self.low[GRIPPERS], self.high[GRIPPERS] = 0, 1
        self.filter = MotionFilter(self.low, self.high, alpha=0.8, max_velocity=1.5)
        self.filter.reset(np.zeros(16))

    def test_first_step_limited_from_measurement_and_grippers_unchanged(self):
        desired = np.ones(16)
        desired[GRIPPERS] = [0.9, 0.2]
        actual, info = self.filter.apply(desired, 0.1)
        self.assertLessEqual(np.max(np.abs(actual[JOINTS])), 0.15 + 1e-12)
        np.testing.assert_array_equal(actual[GRIPPERS], desired[GRIPPERS])
        self.assertTrue(info["velocity_limited"])

    def test_direction_preserved_and_converges_without_overshoot(self):
        target = np.linspace(-1, 1, 16)
        target[GRIPPERS] = [1, 0]
        actual, _ = self.filter.apply(target, 0.1)
        ratio = actual[JOINTS] / target[JOINTS]
        np.testing.assert_allclose(ratio, ratio[0])
        for _ in range(60):
            previous = actual.copy()
            actual, _ = self.filter.apply(target, 0.1)
            self.assertLessEqual(np.max(np.abs(actual[JOINTS] - previous[JOINTS])), 0.15 + 1e-12)
            self.assertTrue(np.all(np.abs(target[JOINTS] - actual[JOINTS]) <= np.abs(target[JOINTS] - previous[JOINTS]) + 1e-12))
        np.testing.assert_allclose(actual, target, atol=1e-6)

    def test_nan_shape_and_invalid_interval_rejected_without_state_change(self):
        previous = self.filter.previous.copy()
        for target, dt in ((np.full(16, np.nan), 0.1), (np.zeros(14), 0.1), (np.zeros(16), 0)):
            with self.assertRaises(ValueError):
                self.filter.apply(target, dt)
            np.testing.assert_array_equal(self.filter.previous, previous)

    def test_out_of_bounds_bounded(self):
        target = np.full(16, 999)
        actual, info = self.filter.apply(target, 0.1)
        self.assertTrue(info["bounds_clipped"])
        self.assertTrue(np.all(actual <= self.high))
        self.assertTrue(np.all(actual >= self.low))

    def test_explicit_passthrough_parameters(self):
        motion = MotionFilter(self.low, self.high, alpha=1, max_velocity=100)
        motion.reset(np.zeros(16))
        target = np.ones(16) * 0.1
        actual, _ = motion.apply(target, 0.1)
        np.testing.assert_array_equal(actual, target)

    def test_head_degree_conversion_and_sdk_limits(self):
        np.testing.assert_allclose(head_radians([0, 20]), [0, np.pi / 9])
        self.assertIsNone(head_radians(None))
        with self.assertRaises(ValueError):
            head_radians([0, 26])


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "config.yaml").read_text())["vision"]
        self.config["table_segmentation"] = False
        self.detector = ParcelDetector(self.config)
        hsv = np.zeros((480, 848, 3), np.uint8)
        cv2.rectangle(hsv, (200, 150), (260, 190), (28, 180, 220), -1)
        self.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def test_rgb_tensor_layout_and_normalization(self):
        tensor = np.moveaxis(self.image / 255.0, -1, 0)[None]
        np.testing.assert_allclose(rgb_image(tensor), self.image, atol=1)
        with self.assertRaises(ValueError):
            rgb_image(tensor - 2)

    def test_detection_stability_missing_target_and_centroid(self):
        for i in range(3):
            detections = self.detector.detect(self.image)
            yellow = next(d for d in detections if d["label"] == "yellow")
            self.assertEqual(yellow["stable"], i == 2)
            np.testing.assert_allclose(yellow["uv"], [230, 170], atol=1)
        self.assertEqual(self.detector.detect(np.zeros_like(self.image)), [])
        yellow = next(d for d in self.detector.detect(self.image) if d["label"] == "yellow")
        self.assertFalse(yellow["stable"])

    def test_roi_removes_background_candidate(self):
        self.config["roi"] = [0.5, 0, 1, 1]
        self.assertEqual(self.detector.detect(self.image), [])


class ServoTests(unittest.TestCase):
    def test_disabled_is_exact_identity(self):
        config = yaml.safe_load((ROOT / "config.yaml").read_text())["servo"]
        target = np.arange(16) * 0.05
        actual, _ = VisualServo(config).apply(target, [], [848, 480])
        np.testing.assert_array_equal(actual, target)

    def test_calibration_sign_and_rank_checks(self):
        rng = np.random.default_rng(2)
        q = rng.normal(0, 0.01, (30, 7))
        jacobian = rng.normal(0, 150, (2, 7))
        uv = q @ jacobian.T
        data = {"arm": "left", "target": "yellow", "image_key": "observation.images.wrist_cam_l",
                "image_wh": [848, 480], "samples": [
                    {"delta_q": a.tolist(), "delta_uv": b.tolist()} for a, b in zip(q, uv)]}
        result = fit(data)
        np.testing.assert_allclose(np.asarray(result["image_jacobian"]), jacobian, atol=1e-10)
        error = np.array([3, -2])
        np.testing.assert_allclose(jacobian @ np.asarray(result["pixel_to_joint"]) @ error, error)
        with self.assertRaises(ValueError):
            fit({**data, "samples": [{"delta_q": [0] * 7, "delta_uv": [2, 3]}] * 12})

    def test_servo_cap_ambiguity_resolution_and_offset_hold(self):
        config = yaml.safe_load((ROOT / "config.yaml").read_text())["servo"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            matrix = np.ones((7, 2)) * 0.01
            path.write_text(json.dumps({"pixel_to_joint": matrix.tolist(), "arm": "left", "target": "yellow",
                "image_key": config["image_key"], "image_wh": [848, 480]}))
            config.update(enabled=True, desired_uv=[100, 100], calibration_file=str(path))
            servo = VisualServo(config)
            action = np.zeros(16)
            target = {"label": "yellow", "uv": [90, 90], "stable": True, "candidate_count": 1}
            actual, _ = servo.apply(action, [target], [848, 480])
            np.testing.assert_allclose(actual[:7], 0.03)
            np.testing.assert_array_equal(actual[7:], action[7:])
            held, _ = servo.apply(action, [], [848, 480])
            np.testing.assert_array_equal(actual, held)
            ambiguous, _ = servo.apply(action, [{**target, "candidate_count": 2}], [848, 480])
            np.testing.assert_array_equal(actual, ambiguous)
            with self.assertRaises(ValueError):
                servo.apply(action, [target], [640, 480])


class GraspTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "config.yaml").read_text())["grasp"]
        self.command = np.zeros(16)
        self.measured = np.zeros(16)
        self.wh = {"left": [848, 480], "right": [848, 480]}
        self.empty = {"left": [], "right": []}

    def test_uncalibrated_window_never_claims_grasp_or_empty(self):
        monitor = GraspMonitor(self.config)
        self.command[7] = 1
        for _ in range(20):
            info = monitor.update(self.command, self.measured, self.empty, self.wh)
        self.assertEqual(info["left"]["state"], "closed_unverified_roi_not_calibrated")
        self.assertIsNone(info["left"]["confirmed_grasp"])
        self.assertEqual(info["left"]["feedback_error"], 1)

    def test_visible_candidate_then_loss_is_only_a_suspicion(self):
        self.config["left_roi"] = [0.2, 0.2, 0.8, 0.8]
        monitor = GraspMonitor(self.config)
        objects = {"left": [{"label": "yellow", "uv": [424, 240], "stable": True, "candidate_count": 1}], "right": []}
        monitor.update(self.command, self.measured, objects, self.wh)
        self.command[7] = self.measured[7] = 1
        for _ in range(12):
            info = monitor.update(self.command, self.measured, objects, self.wh)
        self.assertEqual(info["left"]["state"], "candidate_visible_in_gripper_roi")
        for _ in range(5):
            info = monitor.update(self.command, self.measured, self.empty, self.wh)
        self.assertEqual(info["left"]["state"], "suspected_drop_or_occlusion")
        self.assertIsNone(info["left"]["confirmed_grasp"])
        self.command[7] = 0
        info = monitor.update(self.command, self.measured, self.empty, self.wh)
        self.assertEqual(info["left"]["state"], "open")
        self.assertIsNone(info["left"]["target"])


class WatchdogTests(unittest.TestCase):
    def test_typed_subscriptions_freshness_and_cleanup(self):
        image_type, joint_type, sensor_type, claw_type = (object() for _ in range(4))
        modules = {"sensor_msgs.msg": SimpleNamespace(CompressedImage=image_type, JointState=joint_type),
                   "kuavo_humanoid_sdk.msg.kuavo_msgs.msg": SimpleNamespace(sensorsData=sensor_type, lejuClawState=claw_type)}
        subscribers = []
        class Subscriber:
            def __init__(self, topic, msg_type, callback, **kwargs):
                self.topic, self.msg_type, self.callback = topic, msg_type, callback
                self.closed = False
                subscribers.append(self)
            def unregister(self):
                self.closed = True
        with patch.dict("sys.modules", modules), patch("run_scene1.time.monotonic", return_value=10):
            watchdog = SensorWatchdog(SimpleNamespace(Subscriber=Subscriber),
                                     {"/camera": "CompressedImage", "/joints": "JointState"}, 2)
            self.assertIs(subscribers[0].msg_type, image_type)
            self.assertIs(subscribers[1].msg_type, joint_type)
            self.assertFalse(watchdog.ready())
            with self.assertRaises(RuntimeError):
                watchdog.check()
            for subscriber in subscribers:
                subscriber.callback(None)
            self.assertTrue(watchdog.ready())
            watchdog.check()
        with patch("run_scene1.time.monotonic", return_value=13):
            with self.assertRaises(RuntimeError):
                watchdog.check()
        watchdog.close()
        self.assertTrue(all(s.closed for s in subscribers))


if __name__ == "__main__":
    unittest.main()
