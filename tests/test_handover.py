import unittest
import numpy as np

from scene1.handover import HandoverGripGuard, HandoverPhaseRefresh
from scene1.motion import JOINTS


class HandoverPhaseRefreshTests(unittest.TestCase):
    def test_refresh_waits_for_arm_to_settle_even_with_blocked_fingers(self):
        gate = HandoverPhaseRefresh()
        old, fresh, measured = [np.zeros(16) for _ in range(3)]
        old[15], fresh[15], measured[15] = .2,.99,.51
        for _ in range(8):
            measured[8] += .019
            self.assertFalse(gate.update(old,fresh,measured)[0])
        events = [gate.update(old,fresh,measured)[0] for _ in range(3)]
        self.assertEqual(events,[False,False,True])

    def test_persistent_conflict_refreshes_once_after_stable_holding(self):
        gate = HandoverPhaseRefresh()
        old, fresh, measured = [np.zeros(16) for _ in range(3)]
        old[15], fresh[15], measured[15] = .6, .99, .51
        events = [gate.update(old, fresh, measured)[0] for _ in range(10)]
        self.assertEqual(events, [False, False, False, True]+[False]*6)

    def test_fresh_opening_agreement_and_empty_gripper_cannot_refresh(self):
        for grip, latest in ((.51,.1),(.98,.99),(.01,.99)):
            gate = HandoverPhaseRefresh()
            old, fresh, measured = [np.zeros(16) for _ in range(3)]
            old[15], fresh[15], measured[15] = .1, latest, grip
            self.assertFalse(any(gate.update(old,fresh,measured)[0] for _ in range(6)))

    def test_receiver_engagement_and_transient_conflict_do_not_refresh(self):
        for slot in ('measured','fresh','old'):
            gate = HandoverPhaseRefresh()
            values = {key: np.zeros(16) for key in ('measured','fresh','old')}
            values['measured'][15], values['fresh'][15], values['old'][15] = .51,.99,.6
            values[slot][7] = .3
            self.assertFalse(any(gate.update(values['old'],values['fresh'],values['measured'])[0]
                                 for _ in range(6)))
        gate = HandoverPhaseRefresh()
        old, fresh, measured = [np.zeros(16) for _ in range(3)]
        old[15], fresh[15], measured[15] = .6,.99,.51
        for i in range(10):
            fresh[15] = .99 if i % 3 else .7
            self.assertFalse(gate.update(old,fresh,measured)[0])


class HandoverGripTests(unittest.TestCase):
    def test_premature_open_is_blocked_without_changing_arm_or_left_commands(self):
        guard = HandoverGripGuard()
        raw = np.linspace(-.6,.6,16); raw[7] = .1; raw[15] = 0.
        measured = np.zeros(16); measured[15] = .51
        result,info = guard.apply(raw,measured)
        np.testing.assert_array_equal(result[JOINTS],raw[JOINTS])
        self.assertEqual(result[7],raw[7])
        self.assertEqual(result[15],.98)
        self.assertTrue(info['right_open_blocked'])

    def test_empty_closed_receiver_and_closing_transient_do_not_release_giver(self):
        guard = HandoverGripGuard()
        raw = np.zeros(16); raw[7] = 1.
        measured = np.zeros(16); measured[15] = .51
        for left in (.3,.5,.7,.8,.97,.97,.97,.97):
            measured[7] = left
            result,_ = guard.apply(raw,measured)
            self.assertEqual(result[15],.98)

    def test_stable_receiver_allows_release_then_bin_open_passes_through(self):
        guard = HandoverGripGuard()
        raw = np.zeros(16); raw[7] = 1.
        measured = np.zeros(16); measured[7] = .47; measured[15] = .51
        for _ in range(3):
            result,_ = guard.apply(raw,measured)
            self.assertEqual(result[15],.98)
        result,info = guard.apply(raw,measured)
        self.assertEqual(result[15],0.)
        self.assertEqual(info['state'],'receiver_ready')
        measured[15] = .05
        result,info = guard.apply(raw,measured)
        self.assertTrue(guard.complete)
        raw[7] = 0.
        result,_ = guard.apply(raw,measured)
        np.testing.assert_array_equal(result,raw)

    def test_receiver_releasing_before_transfer_keeps_giver_closed(self):
        guard = HandoverGripGuard()
        raw = np.zeros(16); raw[7] = 1.
        measured = np.zeros(16); measured[7] = .47; measured[15] = .51
        for _ in range(4):
            guard.apply(raw,measured)
        raw[7] = 0.
        result,info = guard.apply(raw,measured)
        self.assertEqual(result[15],.98)
        self.assertEqual(info['left_contact_frames'],0)
