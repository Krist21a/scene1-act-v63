"""Filter decoded ACT actions, never normalized network tensors.

Layout: left joints [0:7], left gripper [7], right joints [8:15],
right gripper [15]. Joints are radians; grippers are normalized [0,1].
"""
import numpy as np

JOINTS = np.array(list(range(7)) + list(range(8, 15)))
GRIPPERS = np.array([7, 15])


def reset_for_manipulation(env, keep_head=False, keep_grippers=False):
    """Initialize observations/interpolation while preserving an active gaze/grasp."""
    head, grippers = env._reset_head, env._reset_eef
    try:
        if keep_head:
            env._reset_head = lambda: None
        if keep_grippers:
            env._reset_eef = lambda: None
        return env.reset()
    finally:
        env._reset_head, env._reset_eef = head, grippers


class ActionPassthrough:
    """Outer MotionFilter and 100 Hz interpolation already smooth commands."""
    def reset(self):
        pass

    def update(self, action):
        return action_vector(action)


def head_radians(degrees):
    if degrees is None:
        return None
    angles = np.asarray(degrees, dtype=np.float64)
    if angles.shape != (2,) or not np.isfinite(angles).all():
        raise ValueError("head_init_degrees must be finite [yaw,pitch] or null")
    if abs(angles[0]) > 80 or abs(angles[1]) > 25:
        raise ValueError("Head angles exceed SDK limits: yaw +/-80, pitch +/-25 degrees")
    return np.deg2rad(angles).tolist()


def action_vector(value, name="action"):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be 16 finite numbers, got {value.shape}")
    return value.copy()


class MotionFilter:
    def __init__(self, low, high, alpha=0.8, max_velocity=1.5):
        self.low = action_vector(low, "lower limits")
        self.high = action_vector(high, "upper limits")
        if np.any(self.low >= self.high):
            raise ValueError("All lower limits must be below upper limits")
        if not 0 < alpha <= 1 or not np.isfinite(max_velocity) or max_velocity <= 0:
            raise ValueError("alpha must be in (0,1]; max_velocity must be positive")
        self.alpha = float(alpha)
        self.max_velocity = float(max_velocity)
        self.previous = None

    def reset(self, measured_state):
        self.previous = action_vector(measured_state, "measured state")
        # Simulation feedback can cross a hard stop by numerical roundoff
        # (observed 2.1e-4 rad under payload). Commands still use exact limits.
        if np.any(self.previous[JOINTS] < self.low[JOINTS]-5e-4) or np.any(
            self.previous[JOINTS] > self.high[JOINTS]+5e-4
        ):
            raise ValueError("Measured joints are outside configured bounds")
        self.previous[JOINTS] = np.clip(self.previous[JOINTS],self.low[JOINTS],self.high[JOINTS])

    def apply(self, proposed, dt):
        if self.previous is None:
            raise RuntimeError("reset() with measured state before filtering")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        proposed = action_vector(proposed)
        bounded = np.clip(proposed, self.low, self.high)
        delta = self.alpha * (bounded[JOINTS] - self.previous[JOINTS])
        # One scale for the whole joint vector preserves its direction.
        peak = float(np.max(np.abs(delta)))
        scale = min(1.0, self.max_velocity * dt / peak) if peak else 1.0
        result = bounded.copy()
        result[JOINTS] = self.previous[JOINTS] + scale * delta
        # Do not low-pass or delay the ACT gripper sequence.
        self.previous = result.copy()
        return result, {
            "velocity_limited": scale < 1.0,
            "velocity_scale": scale,
            "bounds_clipped": bool(np.any(proposed != bounded)),
            "max_joint_change_rad": float(np.max(np.abs(scale * delta))),
        }
