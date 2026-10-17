"""Optional bounded residual using a measured local image Jacobian."""
import json
from pathlib import Path
import numpy as np
from .motion import action_vector


class VisualServo:
    def __init__(self, config):
        self.config = config
        self.matrix = None
        self.offset = np.zeros(7)
        if not config["enabled"]:
            return
        if config["arm"] not in ("left", "right"):
            raise ValueError("servo arm must be left or right")
        desired = np.asarray(config["desired_uv"], dtype=float)
        if desired.shape != (2,) or not np.isfinite(desired).all():
            raise ValueError("Calibrate and set desired_uv before enabling servo")
        calibration = json.loads(Path(config["calibration_file"]).read_text())
        for key in ("image_key", "target", "arm"):
            if calibration[key] != config[key]:
                raise ValueError(f"Calibration {key} does not match servo configuration")
        self.calibration = calibration
        self.matrix = np.asarray(calibration["pixel_to_joint"], dtype=float)
        if self.matrix.shape != (7, 2) or not np.isfinite(self.matrix).all():
            raise ValueError("Calibration pixel_to_joint must be a finite 7x2 matrix")
        for key in ("gain", "max_joint_offset_rad", "max_error_px", "deadband_px"):
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"servo {key} must be positive")
        lo, hi = config["gripper_window"]
        if not 0 <= lo < hi <= 1:
            raise ValueError("servo gripper_window must lie in [0,1]")

    def apply(self, act_action, detections, image_wh):
        action = action_vector(act_action)
        if self.matrix is None:
            return action, {"servo": "disabled"}
        if list(image_wh) != self.calibration["image_wh"]:
            raise ValueError("Servo image resolution differs from calibration")
        start, gripper = (0, 7) if self.config["arm"] == "left" else (8, 15)
        target = next((d for d in detections if d["label"] == self.config["target"]), None)
        lo, hi = self.config["gripper_window"]
        # Keep the last bounded offset during occlusion/closure, avoiding a
        # discontinuous return to the uncorrected trajectory while carrying.
        state = "offset_held"
        if lo <= action[gripper] <= hi and target and target["stable"] and target["candidate_count"] == 1 and not target.get("clipped", False):
            error = np.asarray(self.config["desired_uv"]) - target["uv"]
            norm = float(np.linalg.norm(error))
            if norm <= self.config["max_error_px"]:
                if norm > self.config["deadband_px"]:
                    self.offset += self.config["gain"] * self.matrix @ error
                    limit = self.config["max_joint_offset_rad"]
                    self.offset = np.clip(self.offset, -limit, limit)
                state = "correcting" if norm > self.config["deadband_px"] else "aligned"
            else:
                state = "error_outside_calibration_range"
        action[start:start + 7] += self.offset
        return action, {"servo": state, "offset_rad": self.offset.tolist()}
