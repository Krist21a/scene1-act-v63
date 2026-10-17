"""Bounded, opt-in regrasp-to-ACT experiment; no ROS or model dependencies.

This is an execution contract, not a parcel-success detector. Finger feedback
cannot independently prove lifting or transfer. On an unsupported action the
caller must stop BEFORE env.step, retaining the last command (no reset/open).
"""
import math


JOINTS = tuple(range(7)) + tuple(range(8, 15))


class HandoverContractStop(RuntimeError):
    def __init__(self, reason, details):
        super().__init__(reason)
        self.details = dict(details, stop_reason=reason, next_action_sent=False)


def vector(value):
    result = [float(x) for x in value]
    if len(result) != 16 or not all(math.isfinite(x) for x in result):
        raise ValueError('Expected 16 finite joint/gripper values')
    return result


class HandoverContract:
    def __init__(self, config):
        self.max_steps = config.get('max_steps', 300)
        self.max_seconds = config.get('max_seconds', 45.)
        self.contact_frames = config.get('contact_frames', 3)
        self.transition_steps = config.get('transition_steps', 0)
        self.transition_velocity = config.get('transition_velocity_rad_s', .3)
        self.hold_threshold = config.get('hold_threshold', .9)
        for name in ('max_steps', 'contact_frames', 'transition_steps'):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int) or val < (0 if name == 'transition_steps' else 1):
                raise ValueError(f'Invalid {name}')
        for name in ('max_seconds', 'transition_velocity', 'hold_threshold'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'Invalid {name}')
        if not .85 <= self.hold_threshold <= 1:
            raise ValueError('hold_threshold must be in [.85, 1]')
        self.active = self.complete = False
        self.releasing = False
        self.stable = self.lost = self.transfer_stable = self.action_steps = 0
        self.last_feedback_token = self.previous_left = None

    def begin(self, measured, previous_sent, step, now):
        measured, previous_sent = vector(measured), vector(previous_sent)
        if self.active or self.complete:
            raise ValueError('Handover contract can only begin once per episode')
        if not (.2 < measured[15] < .85 and previous_sent[15] >= .9 and measured[7] <= .2):
            raise HandoverContractStop('handover_entry_contact_invalid', {'step': step})
        self.active = True
        self.start_step, self.start_time = step, now

    def check_deadline(self, step, now):
        if self.active and (step - self.start_step >= self.max_steps or now - self.start_time >= self.max_seconds):
            raise HandoverContractStop('handover_deadline_without_transfer',
                                      {'step': step, 'elapsed_s': now - self.start_time})

    def apply(self, proposed, measured, previous_sent, *, step, now, dt, feedback_token):
        proposed, measured, previous_sent = map(vector, (proposed, measured, previous_sent))
        if not self.active:
            return proposed, {'state': 'feedback_transfer_complete' if self.complete else 'inactive'}
        self.check_deadline(step, now)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive')
        if feedback_token is None:
            raise HandoverContractStop('handover_feedback_metadata_missing', {'step': step})
        fresh_feedback = feedback_token != self.last_feedback_token
        left_holding = .2 < measured[7] < .85 and previous_sent[7] >= .9 and proposed[7] >= .9
        if fresh_feedback:
            stationary = self.previous_left is not None and abs(measured[7] - self.previous_left) <= .015
            self.stable = self.stable + 1 if left_holding and stationary else 0
            self.previous_left = measured[7]
            self.last_feedback_token = feedback_token
            self.lost = self.lost + 1 if not (.2 < measured[15] < .85) and not self.releasing else 0
            self.transfer_stable = self.transfer_stable + 1 if (
                self.releasing and left_holding and stationary and measured[15] <= .15) else 0
        # A changed command invalidates contact immediately, even on a duplicate sample.
        ready = left_holding and self.stable >= self.contact_frames
        info = dict(state='receiving' if ready else 'transporting', step=step,
                    fresh_feedback=fresh_feedback, receiver_contact_frames=self.stable,
                    requested_right=proposed[15], right_feedback=measured[15],
                    left_feedback=measured[7], elapsed_s=now - self.start_time)
        if self.releasing and not ready:
            raise HandoverContractStop('receiver_contact_lost_during_release', info)
        if self.lost >= self.contact_frames:
            raise HandoverContractStop('giver_contact_lost_before_transfer', info)
        if proposed[15] < self.hold_threshold and not ready:
            raise HandoverContractStop('unsupported_giver_open_before_receiver', info)
        if ready and proposed[15] <= .25:
            self.releasing = True
        if self.transfer_stable >= self.contact_frames:
            self.complete, self.active = True, False
            info['state'] = 'feedback_transfer_complete'
            return proposed, info
        self.action_steps += 1
        # Preserve vector direction and leave fingers to ACT. The normal motion
        # filter still enforces physical limits and the original velocity cap.
        if self.action_steps <= self.transition_steps:
            delta = [proposed[j] - previous_sent[j] for j in JOINTS]
            peak = max(abs(x) for x in delta)
            scale = min(1., self.transition_velocity * dt / peak) if peak else 1.
            for j, change in zip(JOINTS, delta):
                proposed[j] = previous_sent[j] + scale * change
            info.update(state='bounded_transition', transition_scale=scale)
        return proposed, info
