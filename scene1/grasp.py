"""Passive grasp diagnostics. Never override ACT or claim physical success."""
from .motion import action_vector


class GraspMonitor:
    def __init__(self, config):
        self.config = config
        if not 0 <= config["open_threshold"] < config["closed_threshold"] <= 1:
            raise ValueError("Grasp thresholds must satisfy 0<=open<closed<=1")
        for key in ("missing_frames", "feedback_grace_frames"):
            if not isinstance(config[key], int) or config[key] < 1:
                raise ValueError(f"grasp {key} must be a positive integer")
        for side in ("left", "right"):
            roi = config[f"{side}_roi"]
            if roi is not None:
                x0, y0, x1, y1 = roi
                if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
                    raise ValueError(f"Invalid {side} grasp ROI")
        self.arms = {arm: {"target": None, "closed_frames": 0, "missing": 0,
                           "seen_while_closed": False} for arm in ("left", "right")}

    def update(self, command, measured, detections, image_wh):
        command = action_vector(command, "sent command")
        measured = action_vector(measured, "measured state")
        result = {}
        for arm, grip in (("left", 7), ("right", 15)):
            memory = self.arms[arm]
            roi = self.config[f"{arm}_roi"]
            objects = detections[arm]
            relevant = []
            if roi is not None:
                x0, y0, x1, y1 = roi
                width, height = image_wh[arm]
                relevant = [d for d in objects if d["stable"] and d["candidate_count"] == 1 and not d.get("clipped", False)
                            and x0 <= d["uv"][0] / width <= x1
                            and y0 <= d["uv"][1] / height <= y1]
            closed = command[grip] >= self.config["closed_threshold"]
            if not closed:
                memory.update(target=None, closed_frames=0, missing=0, seen_while_closed=False)
                if len(relevant) == 1:
                    memory["target"] = relevant[0]["label"]
                state = "open" if command[grip] <= self.config["open_threshold"] else "closing_or_opening"
            else:
                memory["closed_frames"] += 1
                if roi is None:
                    state = "closed_unverified_roi_not_calibrated"
                else:
                    if memory["target"] is None and len(relevant) == 1:
                        memory["target"] = relevant[0]["label"]
                    visible = memory["target"] is not None and any(
                        d["label"] == memory["target"] for d in relevant)
                    if visible:
                        memory["seen_while_closed"] = True
                        memory["missing"] = 0
                    else:
                        memory["missing"] += 1
                    if memory["closed_frames"] <= self.config["feedback_grace_frames"]:
                        state = "waiting_for_gripper_feedback"
                    elif memory["missing"] >= self.config["missing_frames"]:
                        state = "suspected_drop_or_occlusion" if memory["seen_while_closed"] else "suspected_empty_or_occlusion"
                    elif visible:
                        state = "candidate_visible_in_gripper_roi"
                    else:
                        state = "closed_visual_uncertain"
            result[arm] = {"state": state, "target": memory["target"],
                           "command_gripper": float(command[grip]),
                           "measured_gripper": float(measured[grip]),
                           "feedback_error": float(abs(command[grip] - measured[grip])),
                           "visible_labels": [d["label"] for d in objects],
                           "roi_calibrated": roi is not None,
                           "confirmed_grasp": None}
        return result
