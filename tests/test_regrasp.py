import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from scene1.cartesian import CartesianGraspAligner
from scene1.motion import reset_for_manipulation
from test_manipulation import LinearArm

ROOT = Path(__file__).resolve().parents[1]


class GazeResetTests(unittest.TestCase):
    def test_weighing_gaze_survives_regrasp_and_retreat_initialization(self):
        calls = []
        env = SimpleNamespace(_reset_head=lambda: calls.append('head'),
                              _reset_eef=lambda: calls.append('grippers'))
        def reset():
            env._reset_head(); env._reset_eef()
            return 'observation', {}
        env.reset = reset
        self.assertEqual(reset_for_manipulation(env, keep_head=True)[0], 'observation')
        self.assertEqual(calls, ['grippers'])
        env._reset_head()
        self.assertEqual(calls[-1], 'head')

    def test_failed_reset_restores_callbacks_and_preserves_held_gripper(self):
        calls = []
        head = lambda: calls.append('head')
        grip = lambda: calls.append('grip')
        env = SimpleNamespace(_reset_head=head, _reset_eef=grip)
        def reset():
            env._reset_head(); env._reset_eef()
            raise RuntimeError('lost sensor')
        env.reset = reset
        with self.assertRaisesRegex(RuntimeError, 'lost sensor'):
            reset_for_manipulation(env, keep_head=True, keep_grippers=True)
        self.assertEqual(calls, [])
        self.assertIs(env._reset_head, head)
        self.assertIs(env._reset_eef, grip)


class RegraspTests(unittest.TestCase):
    def setUp(self):
        c = yaml.safe_load((ROOT / 'config.yaml').read_text())['cartesian_alignment']
        self.a = CartesianGraspAligner(c, LinearArm())
        self.measured = np.zeros(16)
        self.measured[8:11] = [.4, -.3, .16]
        source = self.measured.copy()
        source[10] = .02
        source[15] = .5
        target = {'position_base_m': [.42, -.3, .02], 'extent_xy_m': [.12, .04]}
        self.rows = [dict(cartesian_alignment=dict(state='close_dwell', arm=1, target=target),
                          measured_state=source.tolist(), sent_action=source.tolist(),
                          act_action=source.tolist(), rgbd_table_z_base_m=-.08),
                     dict(cartesian_alignment=dict(state='release_dwell'),
                          measured_state=self.measured.tolist(), sent_action=self.measured.tolist())]
        self.snapshot = dict(candidates=[target], table_z_base_m=-.08,
                             surfaces=[{'position_base_m': [.42, -.3, -.08]}])

    def test_regrasp_waits_for_alignment_then_verifies_contact_and_lift(self):
        self.a.restore_regrasp(self.rows, self.measured, self.snapshot)
        self.assertIsNotNone(self.a.camera_target(self.measured))
        seen = []
        for _ in range(200):
            result, info = self.a.apply(self.measured, self.measured, self.snapshot)
            seen.append(info['state'])
            if info['state'] in ('lift_before_regrasp', 'align_above_weighing_parcel', 'descend_before_regrasp'):
                self.assertEqual(result[15], 0)
            self.measured = result.copy()
            self.measured[15] = .5 if result[15] > .75 else 0
            self.a.observe_command(result)
            if self.a.pending is None:
                break
        self.assertTrue(info['grasp_verified'])
        self.assertIn('regrasp_trial_lift', seen)
        self.assertTrue(self.a.closed[1])
        self.assertEqual(result[15], .98)

    def test_empty_closed_gripper_never_counts_as_regrasp(self):
        self.a.restore_regrasp(self.rows, self.measured, self.snapshot)
        with self.assertRaisesRegex(RuntimeError, 'not confirmed'):
            for _ in range(200):
                result, _ = self.a.apply(self.measured, self.measured, self.snapshot)
                self.measured = result.copy()

    def test_ambiguous_pad_parcels_block_regrasp(self):
        self.snapshot['candidates'].append({'position_base_m': [.43, -.29, .02]})
        with self.assertRaisesRegex(RuntimeError, 'found 2'):
            self.a.restore_regrasp(self.rows, self.measured, self.snapshot)


class PlacementHeadingTests(unittest.TestCase):
    def test_diagonal_parcel_is_turned_to_pad_axes(self):
        from scipy.spatial.transform import Rotation
        c = yaml.safe_load((ROOT / 'config.yaml').read_text())['cartesian_alignment']
        aligner = CartesianGraspAligner(c, LinearArm())
        aligner.held_major_axis_in_tool[1] = np.array([1., 0., 0.])
        rotation = Rotation.from_euler('z', -43, degrees=True)
        surface = dict(footprint_xy_m=[[.3, -.6], [.5, -.6], [.5, -.4], [.3, -.4]])
        yaws = aligner._placement_yaws(rotation, surface)
        self.assertAlmostEqual(np.rad2deg(yaws[0]), 43, places=4)
        for yaw in yaws:
            angle = -43 + np.rad2deg(yaw)
            self.assertAlmostEqual((angle + 45) % 90 - 45, 0, places=4)
        self.assertFalse(any(abs(yaw) < 1e-5 for yaw in yaws))

    def test_tracking_and_geometry_error_budget_is_inside_pad(self):
        from scipy.spatial.transform import Rotation
        c = yaml.safe_load((ROOT / 'config.yaml').read_text())['cartesian_alignment']
        aligner = CartesianGraspAligner(c, LinearArm())
        aligner.held_extent[1] = [.11, .04]
        aligner.held_place_orientation[1] = np.array([0, 0, 0, 1])
        surface = dict(footprint_xy_m=[[.3, -.6], [.5, -.6], [.5, -.4], [.3, -.4]])
        self.assertFalse(aligner._parcel_fits([.37, -.5, 0], Rotation.identity(), surface))
        self.assertTrue(aligner._parcel_fits([.4, -.5, 0], Rotation.identity(), surface))


class SurfaceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(position_base_m=[.4, -.5, -.08], extent_xy_m=[.2, .2],
                        footprint_xy_m=[[.3, -.6], [.5, -.6], [.5, -.4], [.3, -.4]])
        self.fragment = dict(position_base_m=[.35, -.5, -.08], partial=True,
                             footprint_xy_m=[[.31, -.52], [.36, -.52], [.36, -.48], [.31, -.48]])

    def test_visible_fragment_keeps_previously_observed_full_boundary(self):
        snap = dict(table_z_base_m=-.079, surfaces=[self.fragment])
        result = CartesianGraspAligner.restore_surface_view(snap, [dict(rgbd_surfaces=[self.old])])
        self.assertEqual(result['surfaces'][0]['footprint_xy_m'], self.old['footprint_xy_m'])
        self.assertTrue(result['surfaces'][0]['boundary_from_visual_history'])
        self.assertEqual(result['surfaces'][0]['position_base_m'][2], -.079)

    def test_unrelated_fragment_cannot_create_pad(self):
        self.fragment['footprint_xy_m'] = [[.7, -.52], [.75, -.52], [.75, -.48], [.7, -.48]]
        snap = dict(table_z_base_m=-.08, surfaces=[self.fragment])
        result = CartesianGraspAligner.restore_surface_view(snap, [dict(rgbd_surfaces=[self.old])])
        self.assertEqual(result['surfaces'], [])

    def test_missing_current_surface_does_not_reuse_history_blindly(self):
        result = CartesianGraspAligner.restore_surface_view(dict(table_z_base_m=-.08, surfaces=[]),
                                                           [dict(rgbd_surfaces=[self.old])])
        self.assertEqual(result['surfaces'], [])


if __name__ == '__main__':
    unittest.main()
