import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from scene1.cartesian import CartesianGraspAligner
from scene1.kinematics import UrdfArmKinematics

ROOT = Path(__file__).resolve().parents[1]


class LinearArm:
    def fk(self, action):
        action = np.asarray(action)
        def pose(q):
            return SimpleNamespace(pos_xyz=q[:3].tolist(), quat_xyzw=Rotation.from_rotvec(q[3:6]).as_quat().tolist())
        return SimpleNamespace(left_pose=pose(action[:7]), right_pose=pose(action[8:15]))

    def ik(self, poses, seed, orientation_tolerance=.10):
        result = seed.copy()
        for start, pose in ((0, poses.left_pose), (8, poses.right_pose)):
            result[start:start+3] = pose.pos_xyz
            result[start+3:start+6] = Rotation.from_quat(pose.quat_xyzw).as_rotvec()
        return np.r_[result[:7], result[8:15]]


class ManipulationTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / 'config.yaml').read_text())['cartesian_alignment']
        self.config.update(skip_initial_grasps=0, alignment_timeout_steps=20, target_stable_frames=1)
        self.aligner = CartesianGraspAligner(self.config, LinearArm())
        self.raw = np.zeros(16)
        self.raw[8:11] = [0.4, -0.3, 0.0]
        self.measured = self.raw.copy()
        self.snapshot = {'candidates': [{'position_base_m': [0.44, -0.3, 0.02]}]}
        self.raw[15] = .7

    def test_lift_then_translate_then_descend_and_close_dwell(self):
        states = []
        for _ in range(100):
            result, info = self.aligner.apply(self.raw, self.measured, self.snapshot)
            states.append(info['state'])
            if info['state'] != 'close_dwell':
                self.assertEqual(result[15], 0)
            self.measured = result.copy()
            if self.aligner.pending is None:
                break
        order = list(dict.fromkeys(states))
        self.assertEqual(order, ['lift_before_alignment', 'translate_above_parcel', 'prepare_grasp_orientation', 'orient_before_descent', 'descend_before_close', 'close_dwell'])
        self.assertGreaterEqual(states.count('close_dwell'), self.config['close_dwell_steps'])
        self.assertTrue(self.aligner.closed[1])
        np.testing.assert_allclose(result[8:11], [.44, -.3, .02])

    def test_contact_stall_never_closes(self):
        for _ in range(19):
            result, _ = self.aligner.apply(self.raw, self.measured, self.snapshot)
            self.assertEqual(result[15], 0)
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            self.aligner.apply(self.raw, self.measured, self.snapshot)

    def test_release_requires_measured_arrival_then_open_dwell(self):
        self.aligner.closed[1] = True
        self.raw[15] = 0
        self.measured[8] -= .04
        result, info = self.aligner.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'settle_before_release')
        self.assertEqual(result[15], .98)
        for _ in range(15):
            self.measured = result.copy()
            result, info = self.aligner.apply(self.raw, self.measured, self.snapshot)
            if self.aligner.pending is None:
                break
        self.assertEqual(info['state'], 'release_dwell')
        self.assertEqual(result[15], 0)
        self.assertFalse(self.aligner.closed[1])

    def test_grasp_correction_fades_after_lift(self):
        self.aligner.closed[1] = True
        self.aligner.ready_to_close[1] = False
        self.aligner.grasp_z[1] = 0
        self.aligner.offset[8] = .04
        self.raw[10] = .08
        result, _ = self.aligner.apply(self.raw, self.measured, self.snapshot)
        self.assertAlmostEqual(result[8] - self.raw[8], .032)
        self.assertEqual(result[15], .98)

    def test_ambiguous_target_stops(self):
        self.snapshot['candidates'].append({'position_base_m': [.445, -.3, .02]})
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            self.aligner.apply(self.raw, self.measured, self.snapshot)

    def test_wrong_orientation_does_not_advance(self):
        self.aligner.config['skip_initial_grasps'] = 1
        self.measured[11] = .3
        result, info = self.aligner.apply(self.raw, self.measured, self.snapshot)
        self.assertGreater(info['orientation_error_rad'], .29)
        self.assertEqual(info['stable_steps'], 0)
        self.assertEqual(result[15], 0)

    def test_unreachable_joint_limits_stop_before_command(self):
        high = np.full(16, 3.)
        high[10] = .03
        aligner = CartesianGraspAligner(self.config, LinearArm(), np.full(16, -3.), high)
        with self.assertRaisesRegex(RuntimeError, 'joint limits'):
            aligner.apply(self.raw, self.measured, self.snapshot)

    def test_handover_does_not_require_a_table_candidate(self):
        self.raw[7] = .98
        self.aligner.previous_grip[0] = .98
        self.aligner.ready_to_close[0] = False
        self.aligner.closed[0] = True
        result, info = self.aligner.apply(self.raw, self.measured, None)
        self.assertEqual(info['grasp_kind'], 'handover')
        self.assertEqual(result[15], 0)
        self.assertEqual(result[7], .98)

    def test_merged_gripper_does_not_replace_occluded_parcel(self):
        self.aligner._remember_targets(self.snapshot)
        merged = {'position_base_m': [.42, -.3, .02], 'extent_xy_m': [.17, .125]}
        self.aligner._remember_targets({'candidates': [merged]})
        remembered = self.aligner._targets()
        self.assertEqual(len(remembered), 1)
        np.testing.assert_allclose(remembered[0]['position_base_m'], [.44, -.3, .02])
        self.assertEqual(remembered[0]['occluded_steps'], 1)
        for _ in range(self.config['target_memory_steps']):
            self.aligner._remember_targets({'candidates': []})
        self.assertEqual(self.aligner._targets(), [])

    def test_weigh_release_is_lowered_relative_to_observed_table(self):
        self.aligner.closed[1] = True
        self.raw[15] = 0
        self.raw[10] = .18
        self.snapshot['table_z_base_m'] = -.08
        self.snapshot['surfaces'] = [{'position_base_m': [.4, -.3, -.08]}]
        result, info = self.aligner.apply(self.raw, self.measured, self.snapshot)
        phases = self.aligner.pending['phases']
        self.assertEqual(info['state'], 'lift_before_placement')
        self.assertAlmostEqual(phases[-1]['action'][10], 0)
        self.assertTrue(all(p['action'][15] == .98 for p in phases[:-1]))

    def test_tracking_compensation_is_bounded_and_does_not_accumulate(self):
        for _ in range(19):
            result, info = self.aligner.apply(self.raw, self.measured, self.snapshot)
        bias = np.asarray(info['tracking_compensation_rad'])
        self.assertLessEqual(np.max(np.abs(bias)), self.config['max_tracking_compensation_rad'])
        self.assertEqual(result[15], 0)

    def test_loaded_escape_starts_from_measured_pose_before_rotation(self):
        self.aligner.closed[1] = True
        self.raw[15] = 0
        self.raw[8:14] = [.42, -.32, .18, 0, .3, 0]
        self.measured[8:14] = [.4, -.3, .04, 0, -.2, 0]
        self.snapshot.update(table_z_base_m=-.08,
                             surfaces=[{'position_base_m': [.42, -.3, -.08]}])
        self.aligner.apply(self.raw, self.measured, self.snapshot)
        phases = self.aligner.pending['phases']
        lifts = [p['action'] for p in phases if p['name'] == 'lift_before_placement']
        for action in lifts:
            np.testing.assert_allclose(action[8:10], self.measured[8:10], atol=1e-9)
            np.testing.assert_allclose(action[11:14], self.measured[11:14], atol=1e-9)
        centered = [p['action'] for p in phases if p['name'] == 'center_above_weighing'][-1]
        np.testing.assert_allclose(centered[11:14], self.raw[11:14], atol=1e-9)

    def test_held_recovery_compensates_static_load_without_dropping(self):
        measured = self.measured.copy()
        measured[10] = .04
        measured[15] = .5
        sent = measured.copy()
        sent[13] -= .2
        sent[15] = .98
        target = {'position_base_m': [.42, -.3, .04]}
        record = dict(cartesian_alignment=dict(state='close_dwell', arm=1, target=target),
                      measured_state=measured.tolist(), sent_action=sent.tolist(),
                      act_action=self.raw.tolist())
        snapshot = dict(candidates=[], table_z_base_m=-.08,
                        surfaces=[{'position_base_m': [.42, -.3, -.08]}])
        self.aligner.restore_weighing([record], measured, snapshot)
        disturbance = self.aligner.loaded_feedforward.copy()
        for _ in range(160):
            result, info = self.aligner.apply(self.raw, measured, snapshot)
            if info['state'] != 'release_dwell':
                self.assertEqual(result[15], .98)
            measured = result - disturbance
            if self.aligner.pending is None:
                break
        self.assertIsNone(self.aligner.pending)
        self.assertEqual(info['state'], 'release_dwell')
        self.assertFalse(self.aligner.closed[1])
        np.testing.assert_allclose(self.aligner.loaded_feedforward, 0)

    def test_placement_calibrates_held_load_before_planning(self):
        self.aligner.closed[1] = True
        self.raw[15] = 0
        self.measured[10] = .15
        sent = self.measured.copy()
        sent[13] -= .2
        sent[15] = .98
        self.aligner.observe_command(sent)
        self.snapshot.update(table_z_base_m=-.08,
                             surfaces=[{'position_base_m': [.42, -.3, -.08]}])
        for i in range(10):
            result, info = self.aligner.apply(self.raw, self.measured, self.snapshot)
            self.assertEqual(info['state'], 'stabilize_loaded_hold')
            np.testing.assert_allclose(result, sent)
        self.assertEqual(self.aligner.pending['phases'][0]['name'], 'lift_before_placement')
        self.assertAlmostEqual(self.aligner.loaded_feedforward[13], -.2)
        self.assertEqual(result[15], .98)

    def test_rotated_pad_checks_whole_parcel_footprint(self):
        self.aligner.held_extent[1] = [.18, .04]
        self.aligner.held_place_orientation[1] = np.array([0, 0, 0, 1])
        surface = {'footprint_xy_m': [[0, -.16], [.16, 0], [0, .16], [-.16, 0]]}
        self.assertTrue(self.aligner._parcel_fits([0, 0, 0], Rotation.identity(), surface))
        self.assertFalse(self.aligner._parcel_fits([.05, 0, 0], Rotation.identity(), surface))

    def test_load_estimate_never_accumulates_at_table_contact(self):
        self.aligner.closed[1] = True
        self.aligner.weighing_target = np.array([.4, -.3, -.08])
        goal = self.measured.copy()
        goal[15] = .98
        self.measured[13] = .3
        self.aligner.observe_command(goal)
        self.aligner._begin(goal, 1, [dict(name='lower_before_release', action=goal)])
        for _ in range(25):
            result, info = self.aligner.apply(goal, self.measured, self.snapshot)
            self.assertEqual(result[15], .98)
            self.assertNotIn('load_estimate_updated', info)
        np.testing.assert_allclose(self.aligner.loaded_feedforward, 0)

    def test_free_descent_can_refresh_load_only_with_observed_bottom_clearance(self):
        self.aligner.closed[1] = True
        self.aligner.weighing_target = np.array([.4, -.3, -.08])
        self.aligner.grasp_tool_height[1] = .10
        goal = self.measured.copy(); goal[10] = .12; goal[15] = .98
        self.measured = goal.copy(); self.measured[13] = .3
        self.aligner.observe_command(goal)
        self.aligner._begin(goal, 1, [dict(name='lower_before_release', action=goal)])
        updated = False
        for _ in range(30):
            _, info = self.aligner.apply(goal, self.measured, self.snapshot)
            updated |= info.get('load_estimate_updated', False)
        self.assertTrue(updated)

    def test_known_bottom_near_table_blocks_descent_load_learning(self):
        self.aligner.closed[1] = True
        self.aligner.weighing_target = np.array([.4, -.3, -.08])
        self.aligner.grasp_tool_height[1] = .10
        goal = self.measured.copy(); goal[10] = .01; goal[15] = .98
        self.measured = goal.copy(); self.measured[13] = .3
        self.aligner.observe_command(goal)
        self.aligner._begin(goal, 1, [dict(name='lower_before_release', action=goal)])
        for _ in range(30):
            _, info = self.aligner.apply(goal, self.measured, self.snapshot)
            self.assertNotIn('load_estimate_updated', info)
        np.testing.assert_allclose(self.aligner.loaded_feedforward, 0)

    def test_calibrated_empty_hand_uses_empty_feedback_gain(self):
        goal = self.measured.copy(); goal[15] = 0
        self.measured[13] = .15
        self.aligner.loaded_feedforward[13] = .3
        self.aligner.tracking_bias[13] = .3
        self.aligner._begin(goal, 1, [dict(name='settle_before_close', action=goal)])
        self.aligner.pending['steps'] = 4
        result, _ = self.aligner.apply(goal, self.measured, self.snapshot)
        self.assertAlmostEqual(result[13], .27)

    def test_held_hand_retains_loaded_feedback_gain(self):
        goal = self.measured.copy(); goal[15] = .98
        self.measured[13] = .15
        self.aligner.closed[1] = True
        self.aligner.loaded_feedforward[13] = .3
        self.aligner.tracking_bias[13] = .3
        self.aligner._begin(goal, 1, [dict(name='settle_before_close', action=goal)])
        self.aligner.pending['steps'] = 4
        result, _ = self.aligner.apply(goal, self.measured, self.snapshot)
        self.assertAlmostEqual(result[13], .288)

    def test_horizontal_grasp_targets_front_edge_instead_of_box_center(self):
        pose = SimpleNamespace(quat_xyzw=Rotation.from_euler('y', -np.pi / 2).as_quat())
        item = {'position_base_m': [.4, -.3, .02], 'extent_xy_m': [.12, .04]}
        point = self.aligner._grasp_point(item, pose)
        np.testing.assert_allclose(point, [.36, -.3, .02], atol=1e-9)

    def test_partial_closing_is_withheld_until_alignment(self):
        self.raw[15] = .4
        result, info = self.aligner.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'act')
        self.assertEqual(result[15], 0)

    def test_unseen_weighing_pad_keeps_gripper_closed(self):
        self.aligner.closed[1] = True
        self.raw[15] = 0
        self.snapshot['table_z_base_m'] = -.08
        with self.assertRaisesRegex(RuntimeError, 'pad is not visible'):
            self.aligner.apply(self.raw, self.measured, self.snapshot)

    def test_parcel_center_offset_is_accounted_for_on_the_weighing_pad(self):
        self.aligner.closed[1] = True
        self.aligner.held_center_in_tool[1] = np.array([.04, 0, 0])
        self.raw[15] = 0
        self.raw[10] = .18
        self.snapshot.update(table_z_base_m=-.08, surfaces=[{'position_base_m': [.42, -.3, -.08]}])
        self.aligner.apply(self.raw, self.measured, self.snapshot)
        np.testing.assert_allclose(self.aligner.pending['phases'][-1]['action'][8:11], [.38, -.3, 0], atol=1e-9)


class KinematicsTests(unittest.TestCase):
    def test_urdf_dimensions_and_tool_are_used_instead_of_service_model(self):
        xml = '<robot name="fixture">'
        for side in ('l', 'r'):
            for i in range(1, 8):
                parent = 'base_link' if i == 1 else f'zarm_{side}{i-1}_link'
                xml += f'<joint name="zarm_{side}{i}_joint" type="revolute"><parent link="{parent}"/><child link="zarm_{side}{i}_link"/><origin xyz="0 0 -.1"/><axis xyz="0 1 0"/><limit lower="-2" upper="2"/></joint>'
        model = UrdfArmKinematics(description=xml + '</robot>')
        action = np.zeros(16)
        poses = model.fk(action)
        np.testing.assert_allclose(poses.right_pose.pos_xyz, [0, 0, -.8488])
        solved = model.ik(copy.deepcopy(poses), action)
        np.testing.assert_allclose(solved, 0, atol=1e-8)
        q = action.copy()
        q[8] = .1
        altered = model.fk(q)
        recovered = model.ik(altered, action)
        assembled = np.r_[recovered[:7], 0, recovered[7:], 0]
        np.testing.assert_allclose(model.fk(assembled).right_pose.pos_xyz, altered.right_pose.pos_xyz, atol=.005)


if __name__ == '__main__':
    unittest.main()
