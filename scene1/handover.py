"""ACT handover checks: gripper contact and conflicting temporal predictions."""
from .motion import action_vector


class HandoverPhaseRefresh:
    """Refresh one stale ensemble when current ACT predictions agree on holding.

    This does not infer a task phase from elapsed time or skip future actions.
    It only discards conflicting historical forecasts once per takeover.
    """
    def __init__(self, stable_frames=3):
        self.stable_frames = stable_frames
        self.conflict_frames = 0
        self.previous_right = None
        self.previous_arm = None
        self.refreshed = False

    def update(self, ensembled, fresh, measured):
        ensembled, fresh, measured = map(action_vector, (ensembled, fresh, measured))
        right = float(measured[15])
        held = (.20 < right < .85 and self.previous_right is not None
                and abs(right-self.previous_right) <= .015)
        self.previous_right = right
        arm_motion = (None if self.previous_arm is None else
                      float(max(abs(measured[8:15]-self.previous_arm))))
        self.previous_arm = measured[8:15].copy()
        arm_settled = arm_motion is not None and arm_motion <= .006
        # Once the receiving hand is engaging, let the original ensemble finish
        # the handover. Its legitimately opening giver must not be reset here.
        before_receiver = max(measured[7], ensembled[7], fresh[7]) <= .2
        conflict = (held and arm_settled and before_receiver and ensembled[15] <= .85
                    and fresh[15] >= .95 and fresh[15]-ensembled[15] >= .2)
        self.conflict_frames = self.conflict_frames+1 if conflict and not self.refreshed else 0
        refresh = self.conflict_frames >= self.stable_frames and not self.refreshed
        if refresh:
            self.refreshed = True
        return refresh, dict(refreshed=bool(refresh), already_refreshed=self.refreshed,
            conflict_frames=self.conflict_frames, held_feedback_stable=bool(held),
            arm_settled=bool(arm_settled), max_right_joint_movement_rad=arm_motion,
            ensembled_right=float(ensembled[15]), fresh_right=float(fresh[15]))


class HandoverGripGuard:
    def __init__(self, stable_frames=3):
        self.stable_frames = stable_frames
        self.contact_frames = 0
        self.previous_left = None
        self.previous_left_command = 0.
        self.release_started = False
        self.complete = False

    def apply(self, proposed, measured):
        result = action_vector(proposed)
        measured = action_vector(measured)
        if self.complete:
            return result, dict(state='complete',right_open_blocked=False)
        left = float(measured[7])
        stationary = self.previous_left is not None and abs(left-self.previous_left) <= .015
        left_contact = (.20 < left < .85 and stationary
                        and result[7] >= .9 and self.previous_left_command >= .9)
        self.previous_left = left
        self.previous_left_command = float(result[7])
        self.contact_frames = self.contact_frames+1 if left_contact else 0
        ready = self.contact_frames >= self.stable_frames
        if self.release_started and ready and measured[15] <= .15:
            self.complete = True
        blocked = not ready and result[15] < .98
        if not ready:
            result[15] = max(.98,result[15])
            self.release_started = False
        elif result[15] <= .25:
            self.release_started = True
        return result, dict(state='complete' if self.complete else 'receiver_ready' if ready else 'holding_right',
                            right_open_blocked=bool(blocked),left_contact_frames=self.contact_frames,
                            left_feedback=left,right_feedback=float(measured[15]),
                            requested_right_gripper=float(proposed[15]))
