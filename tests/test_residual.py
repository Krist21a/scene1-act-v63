from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from scene1.residual import ResidualAssist


class LocalArm:
    def matrix(self, q, side):
        matrix = np.eye(4)
        matrix[:3, 3] = q[:3]
        matrix[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return matrix

    def fk(self, action):
        def pose(q):
            return SimpleNamespace(pos_xyz=np.asarray(q[:3]), quat_xyzw=Rotation.from_rotvec(q[3:6]).as_quat())
        return SimpleNamespace(left_pose=pose(action[:7]), right_pose=pose(action[8:15]))


class ResidualTests(unittest.TestCase):
    def setUp(self):
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config.yaml').read_text())
        self.a = ResidualAssist(config['visual_residual'], LocalArm(),
                                np.r_[[-4]*7, 0, [-4]*7, 0], np.r_[[4]*7, 1, [4]*7, 1])
        self.measured = np.zeros(16); self.measured[8:11] = [.3, -.2, 0]
        self.raw = self.measured.copy()
        self.snapshot = dict(table_z_base_m=-.08, surfaces=[], candidates=[
            dict(position_base_m=[.31, -.2, -.03], extent_xy_m=[.10, .04],
                 oriented_extent_m=[.10, .04], major_axis_xy=[1., 0.], top_z_base_m=.02)])

    def test_original_act_command_passes_through_without_a_grip_event(self):
        self.raw[8] += .03
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        np.testing.assert_array_equal(result, self.raw)
        self.assertEqual(info['strategy'], 'act_residual')

    def test_regrasp_cannot_close_when_fingers_lose_level_even_with_wrist_match(self):
        target = self.snapshot['candidates'][0].copy()
        target['grasp_goal_xy'] = self.measured[8:10].tolist()
        self.a.wrist_config = dict(aligned_error_px=22, regrasp_finger_tilt_degrees=5.)
        self.a._wrist_measure = lambda *args, **kwargs: dict(
            correction_base_xy=np.zeros(2), centering_error_px=0.)
        self.a._begin('grasp',1,self.raw,target=target,plane=-.08,
                      grasp_z=float(self.measured[10]),wrist_verified=True,
                      grasp_rotation=Rotation.identity())
        self.measured[12] = np.deg2rad(25.)
        for _ in range(3):
            result, info = self.a._pending_action(self.measured,self.snapshot)
            self.assertEqual(result[15],0.)
            self.assertEqual(info['stable_steps'],0)
            self.assertGreater(info['finger_line_tilt_degrees'],20.)
        self.assertNotEqual(self.a.pending.get('stage'),'close_confirm')

    def test_regrasp_descent_waits_for_lateral_alignment(self):
        self.measured[10] = .09
        self.raw = self.measured.copy()
        self.a.wrist_config = dict(aligned_error_px=22)
        self.a._wrist_measure = lambda *args, **kwargs: None
        self.a._parcel_displacement = lambda *args: None  # head occlusion must not block the stand-off descent
        self.a._begin('pregrasp',1,self.raw,target=self.snapshot['candidates'][0],plane=-.08,
                      grasp_rotation=Rotation.identity(),regrasp_stage='descend',
                      grasp_position=np.array([.32,-.2,-.03]),stand_off_position=np.array([.32,-.2,-.03]),
                      view_position=self.measured[8:11].copy())
        result,_ = self.a._regrasp_action(
            self.measured,self.snapshot,self.a.kinematics.fk(self.measured),{})
        self.assertAlmostEqual(result[10],self.measured[10])
        self.assertGreater(result[8],self.measured[8])
        self.a.pending['stand_off_position'][0] = self.measured[8]
        self.a.offset[:] = 0
        result,_ = self.a._regrasp_action(self.measured,self.snapshot,self.a.kinematics.fk(self.measured),{})
        self.assertLess(result[10],self.measured[10])

    def test_regrasp_cannot_close_without_current_wrist_edges(self):
        target = self.snapshot['candidates'][0].copy()
        target['grasp_goal_xy'] = self.measured[8:10].tolist()
        self.a.wrist_config = dict(aligned_error_px=22)
        self.a._wrist_measure = lambda *args, **kwargs: None
        self.a._begin('grasp',1,self.raw,target=target,plane=-.08,
                      grasp_z=float(self.measured[10]),wrist_verified=True,
                      grasp_rotation=Rotation.identity())
        for _ in range(3):
            result,info = self.a._pending_action(self.measured,self.snapshot)
            self.assertEqual(result[15],0.)
            self.assertEqual(info['stable_steps'],0)

    def _insert_fixture(self, goal, wrist):
        rotation = Rotation.from_matrix([[0,0,-1],[1,0,0],[0,-1,0]])
        self.measured[11:14] = rotation.as_rotvec()
        self.raw = self.measured.copy()
        self.a.wrist_config = dict(aligned_error_px=22)
        self.a.weighing_target = np.array([.31,-.2,-.08])
        self.a.wrist_projector = object()
        self.snapshot['candidates'][0]['support_z_base_m'] = -.08
        self.a._wrist_measure = lambda *args, **kwargs: wrist
        self.a._begin('pregrasp',1,self.raw,target=self.snapshot['candidates'][0],plane=-.08,
                      grasp_rotation=rotation,regrasp_stage='insert',
                      grasp_position=np.array(goal),view_position=self.measured[8:11].copy())

    def test_insertion_restores_height_even_with_large_forward_error(self):
        self._insert_fixture([.335,-.2,-.025],dict(correction_base_xy=[0,0],centering_error_px=0))
        result, _ = self.a._pending_action(self.measured,self.snapshot)
        self.assertLess(result[10],self.measured[10])
        self.assertAlmostEqual(result[8],self.measured[8])  # recover height before advancing

    def test_wrist_centering_replaces_conflicting_head_lateral_goal(self):
        self._insert_fixture([.32,-.17,0],dict(correction_base_xy=[0,-.006],centering_error_px=-15))
        result, _ = self.a._pending_action(self.measured,self.snapshot)
        self.assertLess(result[9],self.measured[9])

    def test_settled_wrist_pose_closes_without_chasing_old_lateral_goal(self):
        self._insert_fixture([.3,-.17,0],dict(correction_base_xy=[0,0],centering_error_px=0))
        for _ in range(3):
            result, info = self.a._pending_action(self.measured,self.snapshot)
        self.assertEqual(info['state'],'regrasp_pose_verified')
        self.assertEqual(self.a.pending['stage'],'close_confirm')
        self.assertEqual(result[15],.98)
        np.testing.assert_array_equal(result[8:15],self.raw[8:15])
        np.testing.assert_array_equal(self.a.pending['hold_position'],self.measured[8:11])

    def test_missing_wrist_keeps_open_and_recovers_height_without_advancing(self):
        self._insert_fixture([.335,-.2,-.025],None)
        for _ in range(3):
            result, _ = self.a._pending_action(self.measured,self.snapshot)
            self.assertEqual(result[15],0)
            self.assertAlmostEqual(result[8],self.measured[8])
            self.assertLess(result[10],self.measured[10])

    def test_open_hand_pauses_act_descent_to_align_above_box(self):
        self.measured[8:11] = [.28, -.24, .095]
        self.raw = self.measured.copy(); self.raw[10] = .045
        self.a.observe_command(self.measured.copy())
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'small_pregrasp')
        self.assertEqual(result[15], 0.)
        self.assertGreater(result[10], self.raw[10] + .03)
        self.assertLessEqual(np.max(np.abs(result[8:15] - self.measured[8:15])),
                             self.a.config['max_joint_step_rad'] + 1e-8)

    def test_pregrasp_stops_if_box_moves_before_finger_contact(self):
        self.measured[8:11] = [.28, -.24, .095]
        self.raw = self.measured.copy()
        self.a.observe_command(self.measured.copy())
        self.a.apply(self.raw, self.measured, self.snapshot)
        self.snapshot['candidates'][0]['position_base_m'][0] += .012
        with self.assertRaisesRegex(RuntimeError, 'Parcel moved before grasp alignment'):
            self.a.apply(self.raw, self.measured, self.snapshot)

    def test_initial_pregrasp_holds_aligned_command_before_settling(self):
        self.measured[8:11] = [.305,-.2,.095]
        self.raw = self.measured.copy()
        target = self.snapshot['candidates'][0].copy()
        target['grasp_goal_xy'] = [.31,-.2]
        self.a.approach[1] = dict(target=target)
        self.a._begin('pregrasp',1,self.raw,target=target,plane=-.08,
                      last_position=np.array([.3,-.2,.095]))
        result,info = self.a._pending_action(self.measured,self.snapshot)
        self.assertTrue(info['pose_hold'])
        self.assertEqual(info['stable_steps'],0)  # 5 mm of measured movement
        np.testing.assert_array_equal(result,self.raw)
        for _ in range(3):
            result,info = self.a._pending_action(self.measured,self.snapshot)
            np.testing.assert_array_equal(result,self.raw)
        self.assertEqual(info['state'],'pregrasp_aligned')
        self.assertIsNone(self.a.pending)

    def test_pregrasp_ignores_shared_base_drift_but_detects_target_motion(self):
        target = dict(self.snapshot['candidates'][0])
        target['position_base_m'] = list(target['position_base_m'])
        for x in (.52, .72):
            reference = dict(target); reference['position_base_m'] = [x, -.2, -.03]
            self.snapshot['candidates'].append(reference)
        self.a._parcel_displacement(self.snapshot, target)
        for candidate in self.snapshot['candidates']:
            candidate['position_base_m'][0] += .012
        self.assertLess(self.a._parcel_displacement(self.snapshot, target), .002)
        self.snapshot['candidates'][0]['position_base_m'][0] += .012
        self.assertGreater(self.a._parcel_displacement(self.snapshot, target), .01)

    def test_shortened_box_from_hand_occlusion_does_not_look_like_motion(self):
        self.measured[8:11] = [.28, -.24, .095]
        self.raw = self.measured.copy()
        self.a.observe_command(self.measured.copy())
        self.a.apply(self.raw, self.measured, self.snapshot)
        visible = self.snapshot['candidates'][0]
        visible['position_base_m'][0] += .012
        visible['oriented_extent_m'][0] = .08
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'small_pregrasp')
        self.assertIsNone(info['parcel_displacement_m'])
        self.assertEqual(result[15], 0.)

    def test_fifteen_percent_occlusion_does_not_establish_centroid_motion(self):
        target = self.snapshot['candidates'][0].copy()
        target['position_base_m'] = list(target['position_base_m'])
        target['oriented_extent_m'] = list(target['oriented_extent_m'])
        self.snapshot['candidates'][0]['position_base_m'][0] += .011
        self.snapshot['candidates'][0]['oriented_extent_m'][0] *= .85
        self.assertIsNone(self.a._parcel_displacement(self.snapshot,target))

    def test_pregrasp_alignment_returns_control_to_act_once(self):
        self.measured[8:11] = [.28, -.24, .095]
        self.raw = self.measured.copy(); self.raw[10] = .045
        for x in (.52, .72):
            reference = dict(self.snapshot['candidates'][0])
            reference['position_base_m'] = [x, -.2, -.03]
            self.snapshot['candidates'].append(reference)
        self.a.observe_command(self.measured.copy())
        for _ in range(30):
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.measured = result.copy()
            self.a.observe_command(result)
            if info['state'] == 'pregrasp_aligned':
                break
        self.assertEqual(info['state'], 'pregrasp_aligned')
        for candidate in self.snapshot['candidates']:
            candidate['position_base_m'][0] += .012
        self.measured[9] -= .02
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'open_hand_visual_tracking')
        self.assertIsNone(self.a.pending)

    def test_pregrasp_settles_with_delayed_joint_response(self):
        self.measured[8:11] = [.28, -.24, .095]
        self.raw = self.measured.copy(); self.raw[10] = .045
        self.a.observe_command(self.measured.copy())
        for _ in range(80):
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.measured += .35 * (result - self.measured)
            self.a.observe_command(result)
            if info['state'] == 'pregrasp_aligned':
                break
        self.assertEqual(info['state'], 'pregrasp_aligned')
        self.assertLessEqual(info['xy_error_m'], .01)
        self.assertLessEqual(info['lateral_movement_m'], .002)

    def test_act_close_height_is_only_trimmed_near_proxy_fingertips(self):
        self.measured[10] = .02
        self.raw = self.measured.copy(); self.raw[15] = .7
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'small_grasp')
        self.assertAlmostEqual(info['grasp_target_base_m'][2], .005)
        self.assertEqual(result[15], 0.)

    def test_grasp_uses_only_small_joint_residual_and_keeps_other_arm(self):
        self.raw[15] = .7
        for _ in range(25):
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.assertLessEqual(np.max(np.abs(result[8:15] - self.raw[8:15])), self.a.config['max_joint_offset_rad'] + 1e-8)
            np.testing.assert_array_equal(result[:8], self.raw[:8])
            np.testing.assert_allclose(result[11:15], self.raw[11:15], atol=1e-8)
            self.measured = result.copy()
            if result[15] >= .9:
                self.measured[15] = .50
            self.a.observe_command(result)
            if self.a.pending is None:
                break
        self.assertEqual(result[15], .98)
        self.assertIsNone(self.a.closing_geometry[1])
        self.assertTrue(self.a.closed[1])

    def test_close_without_finger_contact_is_not_a_grasp(self):
        self.raw[15] = .9
        with self.assertRaisesRegex(RuntimeError, 'finger contact'):
            for _ in range(30):
                result, _ = self.a.apply(self.raw, self.measured, self.snapshot)
                self.measured = result.copy(); self.measured[15] = .99
                self.a.observe_command(result)
        self.assertFalse(self.a.closed[1])

    def test_top_face_bias_does_not_move_estimated_box_center_up(self):
        pose = self.a.kinematics.fk(self.measured).right_pose
        target = dict(self.snapshot['candidates'][0]); target['position_base_m'][2] = .01
        geometry = self.a._geometry(target, pose, -.08)
        self.assertAlmostEqual(geometry['center'][2], -.03)
        self.assertAlmostEqual(geometry['extent'][2], .1)

    def test_verified_regrasp_point_stays_fixed_when_wrist_turns(self):
        target = dict(self.snapshot['candidates'][0], grasp_goal_xy=[.27, -.23])
        pose = self.a.kinematics.fk(self.measured).right_pose
        first = self.a._grasp_goal(target, pose, -.08)
        turned = SimpleNamespace(pos_xyz=pose.pos_xyz,
            quat_xyzw=Rotation.from_euler('z', 60, degrees=True).as_quat())
        second = self.a._grasp_goal(target, turned, -.08)
        np.testing.assert_allclose(first[:2], [.27, -.23])
        np.testing.assert_allclose(second[:2], first[:2])

    def weighing_fixture(self, yaw=0., position=None):
        self.measured[8:11] = [.4, -.5, 0] if position is None else position
        self.raw = self.measured.copy(); self.raw[15] = 0
        self.a.closed[1] = True
        self.a.release_ready[1] = True
        self.a.geometry[1] = dict(center=np.array([0,0,-.03]), rotation=Rotation.from_euler('z', yaw, degrees=True), extent=np.array([.1,.04,.1]))
        held = self.raw.copy(); held[15] = .95; self.a.observe_command(held)
        surface = dict(position_base_m=[.4,-.5,-.08], extent_xy_m=[.2,.2], footprint_xy_m=[[.3,-.6],[.5,-.6],[.5,-.4],[.3,-.4]])
        matrix = np.eye(4); matrix[:3,3] = [0,0,.5]
        direction = np.array([.4,-.5,-.02]) - matrix[:3,3]
        matrix[:3,2] = direction / np.linalg.norm(direction)
        self.snapshot = dict(table_z_base_m=-.08, surfaces=[surface], candidates=[], camera_to_base=matrix)
        self.a._observe_box = lambda snapshot, poses: dict(points=80, bottom_base_m=-.08,
            orientation_confirmed=True, heading_error_degrees=0., tilt_degrees=0.)

    def aim_camera(self):
        if self.a.pending and self.a.pending['kind'] == 'weighing':
            matrix = self.snapshot['camera_to_base']
            direction = self.a.camera_target(self.measured)-matrix[:3,3]
            matrix[:3,2] = direction/np.linalg.norm(direction)

    def test_progressive_close_triggers_visual_check_after_partial_closure(self):
        self.raw[15] = .12
        self.a.apply(self.raw, self.measured, self.snapshot)
        self.measured[15] = .32
        self.raw[15] = .28
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'act')
        self.assertEqual(result[15], 0)
        self.assertIsNone(self.a.pending)
        self.assertTrue(self.a.ready_close[1])
        self.raw[15] = .62
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'small_grasp')
        self.assertEqual(result[15], 0)

    def test_act_descent_continues_with_open_fingers_before_final_grasp_check(self):
        self.raw[15] = .30
        self.raw[10] = -.015
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        np.testing.assert_allclose(result[8:15],self.raw[8:15])
        self.assertEqual(result[15],0.)
        self.assertIsNone(self.a.pending)
        self.raw[15] = .65
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'],'small_grasp')
        self.assertAlmostEqual(info['z_error_m'], -.015)
        self.assertAlmostEqual(info['grasp_target_base_m'][2], -.015)
        self.assertEqual(info['stable_steps'],0)
        self.assertEqual(result[15],0.)
        self.assertLess(result[10],self.raw[10])

    def test_moving_fingers_are_not_contact_but_stationary_blocked_fingers_are(self):
        pose = self.a.kinematics.fk(self.measured).right_pose
        self.a.closing_geometry[1] = self.a._geometry(self.snapshot['candidates'][0], pose, -.08)
        self.a.ready_close[1] = False
        self.raw[15] = .95; self.a.observe_command(self.raw)
        for finger in (.3,.4,.5,.6,.7,.8):
            self.measured[15] = finger
            self.a.apply(self.raw, self.measured, self.snapshot)
            self.assertFalse(self.a.closed[1])
        for _ in range(3):
            self.measured[15] = .5
            self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertTrue(self.a.closed[1])

    def test_residual_is_capped_at_ten_degrees_even_with_stalled_visual_error(self):
        self.a.config['max_joint_offset_rad']=np.deg2rad(10)
        for _ in range(80):
            result = self.a._correct(self.raw, self.measured, 1, [.04,0,0])
        self.assertAlmostEqual(result[8]-self.raw[8], np.deg2rad(10), places=8)
        np.testing.assert_allclose(result[11:15], self.raw[11:15])

    def test_weighing_can_reach_thirty_degree_cap_without_changing_grasp_cap(self):
        for _ in range(70):
            ordinary = self.a._correct(self.raw, self.measured, 1, [.04,0,0])
        self.assertAlmostEqual(ordinary[8]-self.raw[8], np.deg2rad(15), places=8)
        for _ in range(70):
            weighing = self.a._correct(self.raw, self.measured, 1, [.04,0,0],
                offset_cap_rad=self.a.config['max_placement_joint_offset_rad'])
        self.assertAlmostEqual(weighing[8]-self.raw[8], np.deg2rad(30), places=8)

    def test_no_visible_box_blocks_release_even_with_predicted_good_pose(self):
        self.weighing_fixture()
        self.a._observe_box = lambda snapshot, poses: None
        with self.assertRaisesRegex(RuntimeError, 'small correction limits'):
            for _ in range(self.a.config['max_release_steps']+15):
                self.aim_camera()
                result, _ = self.a.apply(self.raw, self.measured, self.snapshot)
                self.a.observe_command(result)
                self.assertGreater(result[15], .9)
        self.assertTrue(self.a.closed[1])

    def test_tilted_box_is_levelled_above_pad_before_descent(self):
        self.weighing_fixture()
        rotation = Rotation.from_euler('y', 12, degrees=True)
        self.a.geometry[1]['rotation'] = rotation
        self.a._observe_box = lambda snapshot, poses: dict(
            points=80, bottom_base_m=-.02, orientation_confirmed=True,
            heading_error_degrees=0., tilt_degrees=12.,
            roof_normal_base=rotation.apply([0,0,1]).tolist())
        for _ in range(10):
            self.aim_camera()
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.a.observe_command(result)
            if info.get('placement_phase'):
                break
        self.assertEqual(info['placement_phase'], 'level')
        self.assertAlmostEqual(result[10], self.raw[10], places=8)
        self.assertGreater(result[15], .9)

    def test_leveling_draws_a_drifting_box_back_to_safe_clearance(self):
        self.weighing_fixture()
        rotation = Rotation.from_euler('y', 12, degrees=True)
        self.a.geometry[1]['rotation'] = rotation
        self.a._observe_box = lambda snapshot, poses: dict(
            points=80, bottom_base_m=.05, orientation_confirmed=True,
            heading_error_degrees=0., tilt_degrees=12.,
            roof_normal_base=rotation.apply([0,0,1]).tolist())
        for _ in range(10):
            self.aim_camera()
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.a.observe_command(result)
            if info.get('placement_phase'):
                break
        self.assertEqual(info['placement_phase'], 'level')
        self.assertLess(result[10], self.raw[10])
        self.assertGreater(result[15], .9)

    def test_eight_degree_edge_contact_can_release_after_visual_settlement(self):
        self.weighing_fixture()
        rotation = Rotation.from_euler('y', 8, degrees=True)
        self.a.geometry[1]['rotation'] = rotation
        self.a._observe_box = lambda snapshot, poses: dict(
            points=80, bottom_base_m=-.08, orientation_confirmed=True,
            heading_error_degrees=0., tilt_degrees=8.,
            roof_normal_base=rotation.apply([0,0,1]).tolist())
        for _ in range(25):
            self.aim_camera()
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.a.observe_command(result)
            if info['state'] == 'weighing_release':
                break
        self.assertEqual(info['state'], 'weighing_release')
        self.assertEqual(result[15], 0.)

    def test_small_tilt_still_requires_low_edge_contact_before_release(self):
        self.weighing_fixture()
        gap = [.003]
        self.a._observe_box = lambda snapshot,poses: dict(points=80,bottom_base_m=-.08+gap[0],
            orientation_confirmed=True,heading_error_degrees=0.,tilt_degrees=8.)
        for _ in range(20):
            self.aim_camera()
            result,info = self.a.apply(self.raw,self.measured,self.snapshot)
            self.a.observe_command(result)
            self.assertGreater(result[15],.9)
        gap[0] = 0.
        for _ in range(10):
            self.aim_camera()
            result,info = self.a.apply(self.raw,self.measured,self.snapshot)
            self.a.observe_command(result)
            if info['state'] == 'weighing_release':
                break
        self.assertEqual(info['state'],'weighing_release')

    def test_weighing_resumes_leveling_if_tilt_drifts_above_release_limit(self):
        self.weighing_fixture()
        rotation = Rotation.from_euler('y', 16, degrees=True)
        self.a.geometry[1]['rotation'] = rotation
        self.a._observe_box = lambda snapshot, poses: dict(
            points=80, bottom_base_m=-.075, orientation_confirmed=True,
            heading_error_degrees=0., tilt_degrees=16.,
            roof_normal_base=rotation.apply([0, 0, 1]).tolist())
        for _ in range(10):
            self.aim_camera()
            self.a.apply(self.raw, self.measured, self.snapshot)
            if self.a.pending and self.a.pending['stage'] == 'correct':
                break
        self.a.pending['level_ready'] = True
        captured = {}
        def capture(base, measured, side, xyz, angular, **kwargs):
            captured['xyz'] = xyz
            captured['angular'] = angular
            return base.copy()
        self.a._correct = capture
        self.aim_camera()
        _, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['placement_phase'], 'lower')
        self.assertAlmostEqual(captured['xyz'][2], 0.)
        self.assertGreater(np.linalg.norm(captured['angular']), 0.)

    def test_aligned_weighing_descent_uses_doubled_joint_step(self):
        self.weighing_fixture()
        self.a._observe_box = lambda snapshot, poses: dict(
            points=80, bottom_base_m=-.04, orientation_confirmed=True,
            heading_error_degrees=0., tilt_degrees=8.)
        for _ in range(10):
            self.aim_camera()
            self.a.apply(self.raw, self.measured, self.snapshot)
            if self.a.pending and self.a.pending['stage'] == 'correct':
                break
        self.a.pending['level_ready'] = True
        captured = {}
        def capture(base, measured, side, xyz, angular, **kwargs):
            captured['xyz'] = xyz
            captured['step_scale'] = kwargs['step_scale']
            captured['translation_limit_m'] = kwargs['translation_limit_m']
            return base.copy()
        self.a._correct = capture
        self.aim_camera()
        self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertLess(captured['xyz'][2], -.02)
        self.assertEqual(captured['step_scale'], 1.)
        self.assertEqual(captured['translation_limit_m'], .04)

    def test_weigh_release_opens_then_uses_short_retreat_before_regrasp(self):
        self.weighing_fixture()
        for _ in range(20):
            self.aim_camera()
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.a.observe_command(result)
            np.testing.assert_allclose(result[8:15], self.raw[8:15], atol=1e-9)
            if info['state'] == 'weighing_release':
                break
        self.assertEqual(info['state'], 'weighing_release')
        self.assertEqual(result[15], 0)
        self.assertFalse(self.a.closed[1])
        self.assertEqual(self.a.pending['kind'], 'short_retreat')
        self.raw[8] -= .02
        for _ in range(12):
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            np.testing.assert_allclose(result[8:15], self.measured[8:15])
            self.assertEqual(result[15], 0.)
        self.snapshot['candidates'] = [dict(position_base_m=[.4,-.5,-.03],
            top_z_base_m=.02, extent_xy_m=[.1,.04], oriented_extent_m=[.1,.04])]
        for _ in range(100):
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.measured = result.copy(); self.a.observe_command(result)
            if info['state'] == 'short_retreat_complete':
                break
        self.assertEqual(info['state'], 'short_retreat_complete')
        self.assertEqual(self.a.pending['kind'], 'settle')
        self.assertLess(np.linalg.norm(self.measured[8:10] - [.4,-.5]), .05)

    def test_release_keeps_original_act_reference_without_command_jump(self):
        self.weighing_fixture()
        reference = self.raw.copy()
        self.a.weighing_target = np.array([.4,-.5,-.08])
        self.a._begin('weighing',1,self.raw,stage='release',
                      reference_base=reference.copy(),requested_open=0.,
                      surface=self.snapshot['surfaces'][0])
        self.a.pending['base'][8] += .3
        command, info = self.a.apply(self.raw,self.measured,self.snapshot)
        self.assertEqual(info['state'],'weighing_release')
        np.testing.assert_allclose(self.a.pending['base'][8:15],reference[8:15])
        self.assertAlmostEqual(self.a.offset[8],.3)
        following, _ = self.a.apply(self.raw,self.measured,self.snapshot)
        np.testing.assert_allclose(following,command)

    def test_pad_view_is_kept_through_release_and_regrasp(self):
        self.a.weighing_target = np.array([.4,-.5,-.08])
        for kind in ('weighing','short_retreat','settle','pregrasp','grasp'):
            self.a._begin(kind,1,self.raw,stage='release',target=dict(support_z_base_m=-.08))
            np.testing.assert_allclose(self.a.camera_target(self.measured),[.4,-.5,-.02])
        self.a.pending = None
        self.assertIsNone(self.a.camera_target(self.measured))

    def test_lost_pad_view_prevents_release(self):
        self.weighing_fixture()
        self.a.weighing_target = np.array([.4,-.5,-.08])
        self.a._begin('weighing',1,self.raw,stage='release',reference_base=self.raw.copy(),
                      requested_open=0.,surface=self.snapshot['surfaces'][0])
        self.a.pending['base'][15] = .98
        self.snapshot['camera_to_base'][:3,2] = [0,0,1]
        result,info = self.a._pending_action(self.measured,self.snapshot)
        self.assertEqual(info['state'],'waiting_release_view')
        self.assertEqual(result[15],.98)

    def test_thirty_degree_regrasp_cap_does_not_widen_initial_grasp(self):
        self.a.wrist_config['max_regrasp_joint_offset_rad'] = np.deg2rad(30)
        for _ in range(100):
            self.a._correct(self.raw,self.measured,1,[.04,0,0])
        self.assertAlmostEqual(self.a.offset[8],np.deg2rad(15),places=8)
        self.a.weighing_target = np.array([.4,-.5,-.08])
        for _ in range(100):
            self.a._correct(self.raw,self.measured,1,[.04,0,0])
        self.assertAlmostEqual(self.a.offset[8],np.deg2rad(30),places=8)

    def test_inherited_placement_offset_can_only_shrink_outside_normal_cap(self):
        self.a.offset[8] = .4
        self.a.unwinding_placement = True
        self.a._correct(self.raw,self.measured,1,[.04,0,0])
        self.assertLessEqual(self.a.offset[8],.4)
        before = self.a.offset[8]
        self.a._correct(self.raw,self.measured,1,[-.04,0,0])
        self.assertLess(self.a.offset[8],before)

    def test_diagonal_box_cannot_be_released_by_a_large_wrist_change(self):
        self.weighing_fixture(yaw=43)
        with self.assertRaisesRegex(RuntimeError, 'small correction limits'):
            for _ in range(self.a.config['max_release_steps']+15):
                self.aim_camera()
                result, _ = self.a.apply(self.raw, self.measured, self.snapshot)
                self.assertGreater(result[15], .9)
                self.assertLessEqual(np.max(abs(result[8:15]-self.raw[8:15])),
                                     self.a.config['max_placement_joint_offset_rad']+1e-8)
                self.a.observe_command(result)
        self.assertTrue(self.a.closed[1])

    def test_weighing_holds_clearance_while_correcting_diagonal_heading(self):
        self.weighing_fixture(yaw=22)
        for _ in range(10):
            self.aim_camera()
            self.a.apply(self.raw, self.measured, self.snapshot)
            if self.a.pending and self.a.pending['stage'] == 'correct':
                break
        captured = {}
        def capture(base, measured, side, xyz, angular, **kwargs):
            captured['xyz'] = xyz
            captured['angular'] = angular
            return base.copy()
        self.a._correct = capture
        self.aim_camera()
        _, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertGreater(info['heading_error_degrees'], 12.)
        self.assertGreater(captured['xyz'][2], 0.)
        self.assertLess(captured['angular'][2], 0.)

    def test_pad_error_outside_small_range_never_opens(self):
        self.weighing_fixture(position=[.49,-.5,0])
        with self.assertRaisesRegex(RuntimeError, 'small correction range'):
            for _ in range(10):
                self.aim_camera()
                result, _ = self.a.apply(self.raw, self.measured, self.snapshot)
                self.a.observe_command(result)
        self.assertTrue(self.a.closed[1])

    def test_unconfirmed_receiver_keeps_giver_closed(self):
        self.weighing_fixture()
        self.raw[7] = .95; self.measured[7] = .99
        with self.assertRaisesRegex(RuntimeError, 'Receiving gripper'):
            for _ in range(12):
                result, _ = self.a.apply(self.raw, self.measured, self.snapshot)
                self.assertGreaterEqual(result[15], .9)
                self.a.observe_command(result)
        self.assertTrue(self.a.closed[1])

    def test_regrasp_resume_keeps_previous_act_residual_bound(self):
        self.weighing_fixture()
        self.snapshot['candidates'] = [dict(position_base_m=[.4,-.5,-.03],
                                            top_z_base_m=.02)]
        act = self.raw.copy(); sent = act.copy(); sent[8] += .10
        records = [dict(cartesian_alignment=dict(state='small_weighing',
                     target_base_m=[.4,-.5,-.08])),
                   dict(cartesian_alignment=dict(state='weighing_release')),
                   dict(cartesian_alignment=dict(state='act'), sent_action=sent,
                        act_action=act)]
        self.a.restore_regrasp(records, self.measured, self.snapshot)
        self.assertEqual(self.a.pending['kind'], 'pregrasp')
        self.assertAlmostEqual(self.a.offset[8], .10)
        self.assertAlmostEqual(self.a.pending['base'][8], act[8])
        self.assertIsNotNone(self.a.weighing_target)
        self.assertFalse(self.a.approach[1]['aligned'])

    def test_verified_wrist_limits_next_open_act_joint_jump(self):
        class SeenWrist:
            def measure(self, *args, **kwargs):
                return dict(centering_error_px=0., correction_base_xy=[0., 0.])
        target = self.snapshot['candidates'][0]
        self.a.approach[1] = dict(target=target, plane=-.08,
                                  aligned=True, wrist_verified=True)
        self.a.weighing_target = np.array([.4, -.5, -.08])
        self.a.wrist_projector = SeenWrist()
        self.a.wrist_config = dict(max_open_joint_delta_rad=.025,
                                   max_regrasp_center_error_px=45)
        self.a.observe_wrist({'right': []}, {'right': [848, 480]})
        self.a.observe_command(self.measured.copy())
        self.raw[8] += .20
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'open_hand_visual_tracking')
        self.assertLessEqual(np.max(abs(result[8:15]-self.measured[8:15])), .025+1e-8)
        self.assertLessEqual(np.max(abs(result[8:15]-self.raw[8:15])),
                             self.a.config['max_joint_offset_rad']+1e-8)

    def test_left_wrist_confirms_stationary_handover_before_left_close(self):
        class ConfirmedWrist:
            def measure(self, side, target, measured, detections, image_wh):
                return dict(centering_error_px=2., correction_base_xy=[0.,0.])
        self.weighing_fixture()
        self.a.wrist_projector = ConfirmedWrist()
        self.a.wrist_config = dict(aligned_error_px=22, stable_frames=3,
                                   max_handover_align_steps=10)
        self.a.observe_wrist({'left': []}, {'left': [848,480]})
        self.raw[7] = .65; self.raw[15] = .98
        self.measured[7] = 0.
        for _ in range(7):
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.a.observe_command(result)
            self.assertEqual(result[7],0.)
            self.assertGreater(result[15],.9)
            if info['state'] == 'handover_camera_aligned':
                break
        self.assertEqual(info['state'],'handover_camera_aligned')
        self.assertTrue(self.a.handover_camera_ready)
        self.assertTrue(self.a.closed[1])

    def test_gradual_release_is_checked_before_fingers_lose_contact(self):
        self.weighing_fixture()
        self.measured[15] = .50
        self.raw[15] = .64
        result, info = self.a.apply(self.raw, self.measured, self.snapshot)
        self.assertEqual(info['state'], 'small_weighing')
        self.assertGreaterEqual(result[15], .9)
        self.assertEqual(self.a.pending['requested_open'], 0.)

    def test_original_opening_ramp_does_not_reclose_after_checked_release(self):
        self.weighing_fixture()
        self.raw[15] = .64; self.measured[15] = .50
        for _ in range(20):
            self.aim_camera()
            result, _ = self.a.apply(self.raw, self.measured, self.snapshot)
            self.a.observe_command(result)
            if self.a.pending is None:
                break
        self.assertEqual(result[15], 0.)
        for grip in (.55, .40, .29, .20):
            self.raw[15] = grip
            result, _ = self.a.apply(self.raw, self.measured, self.snapshot)
            self.assertEqual(result[15], 0.)
        self.assertFalse(self.a.opening[1])

    def test_tilted_bottom_receives_small_leveling_correction_and_cannot_release(self):
        self.weighing_fixture()
        self.a.geometry[1]['rotation'] = Rotation.from_euler('x', 13, degrees=True)
        self.a._observe_box = lambda snapshot, poses: dict(points=80, bottom_base_m=-.08,
            orientation_confirmed=True, heading_error_degrees=0., tilt_degrees=13.,
            roof_normal_base=Rotation.from_euler('x',13,degrees=True).apply([0,0,1]).tolist())
        with self.assertRaisesRegex(RuntimeError, 'small correction limits'):
            for _ in range(self.a.config['max_release_steps']+15):
                self.aim_camera()
                result, info = self.a.apply(self.raw, self.measured, self.snapshot)
                self.a.observe_command(result)
                self.assertGreaterEqual(result[15], .9)
                self.assertLessEqual(np.max(np.abs(result[8:15]-self.raw[8:15])), self.a.config['max_placement_joint_offset_rad']+1e-8)
        self.assertLess(self.a.offset[11], 0.)
        self.assertGreater(info['bottom_height_spread_m'], .006)

    def test_confirmed_grasp_does_not_release_during_original_closing_ramp(self):
        self.a.closed[1] = True
        self.a.ready_close[1] = False
        self.measured[15] = .5
        for grip in (.35, .60, .82, .95):
            self.raw[15] = grip
            result, info = self.a.apply(self.raw, self.measured, self.snapshot)
            self.assertEqual(info['state'], 'act')
            self.assertGreaterEqual(result[15], .98)
        self.assertTrue(self.a.release_ready[1])

    def roof_snapshot(self, tilt_degrees=0):
        """Ray-plane synthetic depth of a brown box top, no robot imagery."""
        self.weighing_fixture()
        rotation = Rotation.from_euler('y', tilt_degrees, degrees=True)
        self.a.geometry[1]['rotation'] = rotation
        center = np.array([.4, -.5, -.03])
        matrix = np.eye(4); matrix[:3,:3] = np.diag([1., -1., -1.])
        matrix[:3,3] = [.4, -.5, 1.02]
        intrinsic = np.array([[1000.,0.,100.],[0.,1000.,60.],[0.,0.,1.]])
        v,u = np.mgrid[:120,:200]
        rays = np.stack(((u-100)/1000., -(v-60)/1000., -np.ones_like(u)), axis=-1)
        normal = rotation.apply([0,0,1])
        roof_center = center + rotation.apply([0,0,.05])
        distance = float(normal @ (roof_center-matrix[:3,3])) / (rays @ normal)
        points = rays*distance[...,None] + matrix[:3,3]
        local = rotation.inv().apply((points-center).reshape(-1,3)).reshape(points.shape)
        mask = (abs(local[...,0]) <= .05) & (abs(local[...,1]) <= .02)
        depth = np.where(mask,distance,0.)
        rgb = np.zeros((120,200,3),np.uint8); rgb[mask] = [160,110,60]
        self.snapshot.update(depth_m=depth, rgb=rgb, intrinsic=intrinsic, camera_to_base=matrix)
        # weighing_fixture mocks this method only for controller tests.
        del self.a._observe_box
        return self.a.kinematics.fk(self.measured)

    def test_rgbd_roof_recovers_level_box_bottom_and_normal(self):
        poses = self.roof_snapshot()
        seen = self.a._observe_box(self.snapshot, poses)
        self.assertTrue(seen['orientation_confirmed'])
        self.assertAlmostEqual(seen['tilt_degrees'], 0., places=5)
        self.assertAlmostEqual(seen['bottom_base_m'], -.08, places=5)
        np.testing.assert_allclose(abs(np.array(seen['roof_normal_base'])),[0,0,1],atol=1e-8)

    def test_rgbd_roof_detects_tilt_and_rejects_incomplete_top(self):
        poses = self.roof_snapshot(10)
        seen = self.a._observe_box(self.snapshot, poses)
        self.assertTrue(seen['orientation_confirmed'])
        self.assertAlmostEqual(seen['tilt_degrees'], 10., places=4)
        self.snapshot['depth_m'][:,90:] = 0
        self.assertIsNone(self.a._observe_box(self.snapshot, poses))

    def test_roof_fit_separates_top_from_side_face_and_raised_label(self):
        x,y=np.meshgrid(np.linspace(-.05,.05,20),np.linspace(-.02,.02,10))
        top=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.1)))
        x,z=np.meshgrid(np.linspace(-.05,.05,25),np.linspace(0,.1,30))
        side=np.column_stack((x.ravel(),np.full(x.size,-.02),z.ravel()))
        label=top[80:110].copy(); label[:,2]+=.008
        normal=self.a._roof_normal(np.vstack((top,side,label)))
        np.testing.assert_allclose(normal,[0,0,1],atol=1e-8)
        self.assertIsNone(self.a._roof_normal(side))

    def test_saturated_joint_shares_correction_with_remaining_joint(self):
        class CoupledArm(LocalArm):
            def matrix(self,q,side):
                matrix=super().matrix(q,side)
                matrix[:3,3]=[q[0]+q[1],q[2],q[6]]
                return matrix
        self.a.kinematics=CoupledArm()
        self.a.offset[8]=self.a.config['max_joint_offset_rad']
        result=self.a._correct(self.raw,self.measured,1,[.02,0,0])
        self.assertAlmostEqual(result[8]-self.raw[8],self.a.config['max_joint_offset_rad'],places=7)
        self.assertGreater(result[9]-self.raw[9],.003)
        self.assertLessEqual(result[9]-self.raw[9],self.a.config['max_joint_step_rad'])

    def test_visual_leveling_keeps_act_descent_and_does_not_freeze_policy(self):
        self.weighing_fixture()
        self.measured[10]=.10
        self.raw=self.measured.copy(); self.raw[10]=.08; self.raw[15]=.98
        rotation=Rotation.from_euler('x',12,degrees=True)
        self.a.geometry[1]['rotation']=rotation
        self.a._observe_box=lambda snapshot,poses: dict(points=80,bottom_base_m=.01,
            orientation_confirmed=True,tilt_degrees=12.,roof_normal_base=rotation.apply([0,0,1]).tolist())
        result,info=self.a.apply(self.raw,self.measured,self.snapshot)
        self.assertEqual(info['state'],'act_transport_leveling')
        self.assertIsNone(self.a.pending)
        np.testing.assert_allclose(result[:8],self.raw[:8])
        np.testing.assert_allclose(result[8:11],self.raw[8:11],atol=1e-8)
        self.assertLess(result[11],self.raw[11])
        self.assertLessEqual(np.max(abs(result[8:15]-self.raw[8:15])),self.a.config['max_joint_offset_rad'])

    def test_physical_joint_limit_takes_priority_over_larger_residual_cap(self):
        limits=np.array([[-4.,4.]]*7)
        limits[0,1]=self.raw[8]+.005
        self.a.kinematics.limits=[limits.copy(),limits.copy()]
        for _ in range(50):
            result=self.a._correct(self.raw,self.measured,1,[.02,0,0])
            self.assertLessEqual(result[8],limits[0,1]+1e-10)
        self.assertAlmostEqual(result[8],limits[0,1],places=8)

    def test_close_settlement_uses_visual_goal_instead_of_early_low_pose(self):
        self.measured[8:11]=[.31,-.2,-.039]
        self.raw=self.measured.copy(); self.raw[15]=.98
        for _ in range(2):
            result,_=self.a.apply(self.raw,self.measured,self.snapshot)
            self.a.observe_command(result)
        np.testing.assert_allclose(self.a.pending['hold_position'],[.31,-.2,-.03])
        self.measured[8:11]=[.31,-.2,-.03]; self.measured[15]=.5
        for _ in range(6):
            result,info=self.a.apply(self.raw,self.measured,self.snapshot)
            self.a.observe_command(result)
            if self.a.pending is None:
                break
        self.assertEqual(info['state'],'grasp_contact_confirmed')
        self.assertAlmostEqual(info['holding_error_m'],0.)

    def test_head_scan_retains_preexisting_small_arm_correction(self):
        self.weighing_fixture()
        self.a.offset[8]=.01
        result,info=self.a.apply(self.raw,self.measured,self.snapshot)
        self.assertEqual(info['head_scan_stage'],'scan_left')
        self.assertAlmostEqual(result[8]-self.raw[8],.01)

    def test_unexpected_roof_orientation_is_not_used_to_steer_the_arm(self):
        poses=self.roof_snapshot(10)
        self.a.geometry[1]['rotation']=Rotation.from_euler('y',-20,degrees=True)
        seen=self.a._observe_box(self.snapshot,poses)
        self.assertIsNotNone(seen)
        self.assertFalse(seen['orientation_confirmed'])
        self.assertGreater(seen['orientation_difference_degrees'],20)

    def test_roof_fit_over_forty_five_degrees_cannot_bypass_consistency_check(self):
        poses = self.roof_snapshot(10)
        original = self.a.geometry[1]['rotation'].as_matrix().copy()
        self.a._roof_normal = lambda group: Rotation.from_euler('y',51,degrees=True).apply([0,0,1])
        seen = self.a._observe_box(self.snapshot,poses)
        self.assertGreater(seen['tilt_degrees'],45.)
        self.assertGreater(seen['orientation_difference_degrees'],20.)
        self.assertFalse(seen['orientation_confirmed'])
        np.testing.assert_allclose(self.a.geometry[1]['rotation'].as_matrix(),original)

    def test_leveling_allows_small_spin_when_a_joint_cannot_cancel_yaw(self):
        class CoupledRotation(LocalArm):
            def matrix(self,q,side):
                matrix=np.eye(4)
                matrix[:3,3]=q[2:5]
                matrix[:3,:3]=Rotation.from_rotvec([q[0],q[5],q[0]+q[1]]).as_matrix()
                return matrix
        self.a.kinematics=CoupledRotation()
        raw=np.zeros(16); measured=raw.copy()
        self.a.offset[9]=-self.a.config['max_joint_offset_rad']
        initial=self.a.offset.copy()
        full=self.a._correct(raw,measured,1,[0,0,0],[.05,0,0])
        self.a.offset=initial.copy()
        leveled=self.a._correct(raw,measured,1,[0,0,0],[.05,0,0],level_axis=[0,0,1])
        self.assertGreater(leveled[8],full[8]*1.5)
        self.assertLessEqual(leveled[8],self.a.config['max_joint_step_rad'])
        self.assertLessEqual(np.max(abs(leveled[8:15])),self.a.config['max_joint_offset_rad'])

    def test_partial_or_ambiguous_components_do_not_pull_the_held_box_center(self):
        def candidate(x,size):
            return dict(position_base_m=[x,-.5,.02],extent_xy_m=size,oriented_extent_m=size)
        for candidates in ([candidate(.423,[.05,.04])],
                           [candidate(.415,[.1,.04]),candidate(.425,[.1,.04])]):
            with self.subTest(candidates=candidates):
                poses=self.roof_snapshot()
                self.snapshot['candidates']=candidates
                seen=self.a._observe_box(self.snapshot,poses)
                self.assertEqual(seen['position_source'],'held_box_roi')
                self.assertAlmostEqual(self.a.geometry[1]['center'][0],0.,places=6)

    def begin_contact(self):
        self.measured[8:11]=[.31,-.2,-.03]
        self.raw=self.measured.copy(); self.raw[15]=.98
        for _ in range(2):
            result,_=self.a.apply(self.raw,self.measured,self.snapshot)
            self.a.observe_command(result)

    def test_settled_contact_registers_object_pose_after_the_wrist_changes(self):
        self.begin_contact()
        self.measured[13]=np.deg2rad(10); self.measured[15]=.5
        for _ in range(6):
            result,info=self.a.apply(self.raw,self.measured,self.snapshot)
            self.a.observe_command(result)
            if self.a.pending is None:
                break
        self.assertEqual(info['state'],'grasp_contact_confirmed')
        center,rotation,_=self.a._world_geometry(1,self.a.kinematics.fk(self.measured))
        np.testing.assert_allclose(center,[.31,-.2,-.03],atol=1e-8)
        np.testing.assert_allclose(rotation.as_matrix(),np.eye(3),atol=1e-8)

    def test_contact_can_settle_inside_box_without_chasing_unloaded_tcp(self):
        self.begin_contact()
        self.assertFalse(self.a.first_grasp_complete)
        held=self.a.pending['holding_action'].copy()
        self.measured[10]+=.018; self.measured[15]=.5
        for _ in range(6):
            result,info=self.a.apply(self.raw,self.measured,self.snapshot)
            self.a.observe_command(result)
            np.testing.assert_allclose(result,held)
            if self.a.pending is None:
                break
        self.assertEqual(info['state'],'grasp_contact_confirmed')
        self.assertTrue(info['contact_envelope_ok'])
        self.assertTrue(self.a.first_grasp_complete)
        self.assertFalse(self.a.regrasp_complete)

    def test_regrasp_contact_does_not_set_first_grasp_takeover_event(self):
        self.snapshot['candidates'][0]['support_z_base_m'] = -.08
        self.begin_contact()
        self.measured[15] = .5
        for _ in range(6):
            result,info = self.a.apply(self.raw,self.measured,self.snapshot)
            self.a.observe_command(result)
            if self.a.pending is None:
                break
        self.assertEqual(info['state'],'grasp_contact_confirmed')
        self.assertTrue(self.a.regrasp_complete)
        self.assertFalse(self.a.first_grasp_complete)

    def test_stationary_fingers_outside_box_envelope_do_not_confirm_grasp(self):
        self.begin_contact()
        self.measured[10]+=.07; self.measured[15]=.5
        with self.assertRaisesRegex(RuntimeError,'did not settle'):
            for _ in range(45):
                result,info=self.a.apply(self.raw,self.measured,self.snapshot)
                self.a.observe_command(result)
                self.assertFalse(info['contact_envelope_ok'])


if __name__ == '__main__':
    unittest.main()
