"""Offline execution-contract and recording tests; no robot or model required."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scene1.handover_contract import HandoverContract, HandoverContractStop
from scene1.policy_trace import ACTChunkTrace
from scene1.trace_window import PolicyTraceWindow


class TraceWindowTests(unittest.TestCase):
    def make(self, **config):
        written = []
        recorder = PolicyTraceWindow(dict(enabled=True, pre_steps=2, post_steps=3,
                                         max_frames=5, **config), '.',
                                    lambda step, payload, meta: written.append((step, payload, meta)))
        return recorder, written

    def test_pre_and_post_are_contiguous_including_paused_frames(self):
        recorder, written = self.make()
        for step in range(1, 5):
            recorder.capture(step, {'step': step}, {'paused': True})
        recorder.trigger(5, 'takeover')
        for step in range(5, 9):
            recorder.capture(step, {'step': step}, {'paused': False})
        self.assertEqual([r[0] for r in written], [3, 4, 5, 6, 7])
        self.assertTrue(written[0][2]['paused'])
        self.assertFalse(recorder.wants_payload())

    def test_overlapping_events_never_duplicate_or_exceed_budget(self):
        recorder, written = self.make()
        recorder.trigger(1, 'start')
        for step in range(1, 10):
            recorder.trigger(step, 'repeated')
            recorder.capture(step, {}, {})
        self.assertEqual([r[0] for r in written], list(range(1, 6)))

    def test_disabled_recorder_does_not_write_or_request_payload(self):
        recorder = PolicyTraceWindow({}, '.', lambda *args: self.fail('Unexpected write'))
        recorder.trigger(1, 'takeover')
        recorder.capture(1, {}, {})
        self.assertFalse(recorder.wants_payload())
        self.assertEqual(recorder.events, [])

    def test_manifest_distinguishes_budget_exhaustion_from_complete_window(self):
        with tempfile.TemporaryDirectory() as output:
            recorder = PolicyTraceWindow(dict(enabled=True, pre_steps=0, post_steps=2, max_frames=2),
                                         output, lambda *args: None)
            recorder.trigger(1, 'takeover')
            recorder.capture(1, {}, {})
            recorder.close()
            meta = json.loads((Path(output) / 'policy-trace.json').read_text())
            self.assertFalse(meta['budget_exhausted'])
            self.assertEqual(meta['saved_steps'], [1])

    def test_invalid_budget_is_rejected_before_robot_creation(self):
        with self.assertRaises(ValueError):
            PolicyTraceWindow(dict(enabled=True, pre_steps=8, post_steps=200, max_frames=10), '.')

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy unavailable in this host session')
    def test_npz_preserves_full_float_tensors_and_replay_metadata(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as output:
            recorder = PolicyTraceWindow(dict(enabled=True, pre_steps=0, post_steps=1, max_frames=1), output)
            chunk = np.arange(1600, dtype=np.float32).reshape(1, 100, 16)
            original = np.zeros((1, 3, 4, 5), dtype=np.float32)
            payload = {'normalized_chunk': chunk, 'observation.images.head_cam_h': original}
            meta = dict(policy_epoch=2, epoch_inference_index=1, policy_paused=False)
            recorder.trigger(10, 'takeover')
            recorder.capture(10, payload, meta)
            with np.load(Path(output) / 'policy-trace/frame-00010.npz', allow_pickle=False) as data:
                np.testing.assert_array_equal(data['normalized_chunk'], chunk)
                np.testing.assert_array_equal(data['observation.images.head_cam_h'], original)
                self.assertEqual(json.loads(str(data['metadata_json'])), meta)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(copy.deepcopy(self.values))


class TraceHookTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable in this host session')
    def test_torch_hook_and_inplace_decoder_leave_prediction_unchanged(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self, value):
                return (value + 1, None)
        class Policy:
            model = Model()
        policy = Policy()
        value = torch.arange(1600.).reshape(1, 100, 16)
        expected = policy.model(value)[0].clone()
        trace = ACTChunkTrace(policy)
        actual = policy.model(value)[0]
        trace.describe(lambda x: x.mul_(2))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(trace.chunk, expected, rtol=0, atol=0)
        trace.close()

    def test_hook_returns_none_and_does_not_alias_model_output(self):
        class Model:
            def register_forward_hook(self, hook):
                self.hook = hook
                return self

            def remove(self):
                self.hook = None
        class Policy:
            model = Model()
        trace = ACTChunkTrace(Policy())
        output = FakeTensor([[1., 2.]])
        self.assertIsNone(trace._capture(None, None, (output,)))
        trace.chunk.values[0][0] = 9
        self.assertEqual(output.values, [[1., 2.]])
        self.assertEqual(trace.generation, 1)
        trace.begin_inference()
        self.assertIsNone(trace.chunk)
        trace.close()
        self.assertIsNone(Policy.model.hook)


class HandoverContractTests(unittest.TestCase):
    def setUp(self):
        self.gate = HandoverContract({})
        self.measured = [0.] * 16
        self.measured[15] = .5
        self.sent = [0.] * 16
        self.sent[15] = .98
        self.proposed = self.sent.copy()
        self.gate.begin(self.measured, self.sent, 10, 0.)
        self.step = 10

    def tick(self, token=None):
        self.step += 1
        return self.gate.apply(self.proposed, self.measured, self.sent, step=self.step,
                               now=(self.step-10)*.1, dt=.1,
                               feedback_token=self.step if token is None else token)

    def test_premature_release_stops_instead_of_substituting_closed_command(self):
        self.proposed[15] = .5
        with self.assertRaises(HandoverContractStop) as stopped:
            self.tick()
        self.assertEqual(str(stopped.exception), 'unsupported_giver_open_before_receiver')
        self.assertFalse(stopped.exception.details['next_action_sent'])
        self.assertEqual(self.proposed[15], .5)

    def test_fully_closed_empty_receiver_never_allows_release(self):
        self.measured[7], self.proposed[7], self.sent[7] = .98, 1., 1.
        for _ in range(6):
            self.tick()
        self.proposed[15] = 0.
        with self.assertRaises(HandoverContractStop):
            self.tick()

    def test_contact_requires_previous_sent_command(self):
        self.measured[7], self.proposed[7] = .5, 1.
        for _ in range(8):
            self.tick()
        self.assertEqual(self.gate.stable, 0)

    def test_duplicate_feedback_cannot_complete_contact(self):
        self.measured[7], self.proposed[7], self.sent[7] = .5, 1., 1.
        for _ in range(8):
            self.tick(token='same_message')
        self.assertEqual(self.gate.stable, 0)
        self.proposed[15] = 0.
        with self.assertRaises(HandoverContractStop):
            self.tick(token='same_message')

    def receiver_ready(self):
        self.measured[7], self.proposed[7], self.sent[7] = .5, 1., 1.
        for _ in range(4):
            self.tick()

    def test_verified_feedback_transfer_exits_then_bin_release_passes(self):
        self.receiver_ready()
        self.proposed[15] = 0.
        result, _ = self.tick()
        self.assertEqual(result[15], 0.)
        self.measured[15] = .05
        for _ in range(3):
            self.tick()
        self.assertTrue(self.gate.complete)
        self.proposed[7] = 0.
        result, info = self.tick()
        self.assertEqual(result, self.proposed)
        self.assertEqual(info['state'], 'feedback_transfer_complete')

    def test_receiver_loss_during_release_stops(self):
        self.receiver_ready()
        self.proposed[15] = 0.
        self.tick()
        self.proposed[7] = 0.
        with self.assertRaisesRegex(HandoverContractStop, 'receiver_contact_lost'):
            self.tick()

    def test_missing_feedback_metadata_stops(self):
        with self.assertRaisesRegex(HandoverContractStop, 'metadata_missing'):
            self.gate.apply(self.proposed, self.measured, self.sent,
                            step=11, now=.1, dt=.1, feedback_token=None)

    def test_giver_contact_loss_stops_even_with_closed_prediction(self):
        self.measured[15] = .99
        self.tick()
        self.tick()
        with self.assertRaisesRegex(HandoverContractStop, 'giver_contact_lost'):
            self.tick()

    def test_deadline_also_covers_head_view_wait_without_inference(self):
        with self.assertRaisesRegex(HandoverContractStop, 'deadline'):
            self.gate.check_deadline(310, 29.)
        with self.assertRaisesRegex(HandoverContractStop, 'deadline'):
            self.gate.check_deadline(12, 46.)

    def test_transition_caps_vector_direction_without_changing_grippers(self):
        self.gate = HandoverContract(dict(transition_steps=1, transition_velocity_rad_s=.3))
        self.gate.begin(self.measured, self.sent, 10, 0.)
        self.proposed[0], self.proposed[8] = .6, -.3
        result, _ = self.tick()
        self.assertAlmostEqual(result[0], .03)
        self.assertAlmostEqual(result[8], -.015)
        self.assertEqual(result[15], .98)
        result, _ = self.tick()
        self.assertEqual(result, self.proposed)

    def test_no_transition_is_exact_passthrough_for_supported_actions(self):
        self.proposed[0] = .6
        result, _ = self.tick()
        self.assertEqual(result, self.proposed)
        self.assertIsNot(result, self.proposed)

    def test_entry_requires_held_giver_and_empty_receiver(self):
        for slot, val in ((15, .99), (15, .01), (7, .5)):
            measured = self.measured.copy()
            measured[slot] = val
            with self.assertRaises(HandoverContractStop):
                HandoverContract({}).begin(measured, self.sent, 1, 0.)


if __name__ == '__main__':
    unittest.main()
