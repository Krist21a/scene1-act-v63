import copy
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from scene1.weighing_pose import alternate_weighing_pose, support_margin, supported_candidate, regrasp_posture_reserve
from scene1.kinematics import UrdfArmKinematics
from scene1.visual_pickup import update_pad_reference
from test_visual_cycle import Arm


class WeighingPoseTests(unittest.TestCase):
    def setUp(self):
        self.surface = dict(position_base_m=[.4,-.5,0.], extent_xy_m=[.2,.2],
            footprint_xy_m=[[.3,-.6],[.5,-.6],[.5,-.4],[.3,-.4]], partial=False)

    def test_smaller_occluded_boundary_cannot_move_existing_reference(self):
        smaller = copy.deepcopy(self.surface)
        smaller['extent_xy_m'] = [.19,.15]
        smaller['position_base_m'] = [.41,-.52,0.]
        smaller['footprint_xy_m'] = [[.31,-.6],[.5,-.6],[.5,-.45],[.31,-.45]]
        before = copy.deepcopy(self.surface)
        result = update_pad_reference(self.surface,[smaller],[.4,-.5,0.])
        self.assertIs(result,self.surface)
        self.assertEqual(result,before)

    def test_reference_can_gain_visible_area_but_not_adopt_partial_surface(self):
        reference = copy.deepcopy(self.surface)
        reference['footprint_xy_m'] = [[.31,-.6],[.49,-.6],[.49,-.44],[.31,-.44]]
        self.assertGreater(support_margin([.4,-.5,.1],Rotation.identity(),[.11,.04,.1],
            self.surface['footprint_xy_m']),.015)
        full = update_pad_reference(reference,[self.surface],[.4,-.5,0.])
        self.assertEqual(full,self.surface)
        self.assertIsNot(full,self.surface)
        self.surface['partial'] = True
        self.assertIs(update_pad_reference(reference,[self.surface],[.4,-.5,0.]),reference)

    def fixture(self):
        arm = Arm(); arm.limits = [np.tile([-3.,3.],(7,1)) for _ in range(2)]
        arm.limits[1][0,1] = .385
        measured = np.zeros(16); measured[8:11] = [.38,-.5,.1]
        geometry = dict(center=np.zeros(3),extent=np.array([.11,.04,.1]),rotation=Rotation.identity())
        goal = np.eye(4); goal[:3,3] = [.4,-.5,.1]
        return arm,measured,geometry,goal

    def test_bounded_supported_alternative_is_cartesian_not_joint_command(self):
        arm,measured,geometry,goal = self.fixture()
        result = alternate_weighing_pose(arm,measured,measured,geometry,goal,self.surface)
        self.assertIsNotNone(result)
        self.assertLess(result['center'][0],.4)
        self.assertLessEqual(np.linalg.norm(result['center'][:2]-goal[:2,3]),.037)
        self.assertGreaterEqual(result['support_margin_m'],.025)
        self.assertLessEqual(result['predicted_error'],result['baseline_error']*.8)
        self.assertNotIn('joints',result)

    def test_unreachable_goal_cannot_move_parcel_beyond_observed_support(self):
        arm,measured,geometry,goal = self.fixture()
        narrow = dict(self.surface,footprint_xy_m=[[.33,-.54],[.47,-.54],[.47,-.46],[.33,-.46]])
        self.assertIsNone(alternate_weighing_pose(arm,measured,measured,geometry,goal,narrow))

    def test_failure_replay_prefers_more_support_with_small_prediction_cost(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/weighing_search_support_v42.json').read_text())
        candidates = fixture['candidates']; before = copy.deepcopy(candidates)
        minimum = min(candidates, key=lambda c: c['predicted_error'])
        chosen = supported_candidate(candidates)
        self.assertEqual(chosen['shift_xy_m'], [0., .03])
        self.assertGreater(chosen['support_margin_m']-minimum['support_margin_m'], .009)
        self.assertLessEqual(chosen['predicted_error'], 1.1*minimum['predicted_error'])
        self.assertLess(chosen['predicted_error'], .8*chosen['baseline_error'])
        self.assertLessEqual(chosen['predicted_position_m'], .01)
        self.assertEqual(candidates, before)

    def test_support_ranking_cannot_trade_away_large_reachability_cost(self):
        reachable = dict(predicted_error=.02, support_margin_m=.03)
        costly = dict(predicted_error=.023, support_margin_m=.06)
        self.assertIs(supported_candidate([costly, reachable]), reachable)
        self.assertIsNone(supported_candidate([]))

    def test_boundary_noise_does_not_select_a_worse_equal_support_pose(self):
        reachable = dict(predicted_error=.02, support_margin_m=.040)
        noisy = dict(predicted_error=.021, support_margin_m=.041)
        self.assertIs(supported_candidate([noisy, reachable]), reachable)

    def test_actual_preparation_feedback_accepts_reserve_only_with_pose_margin(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/weighing_posture_reserve_v46.json').read_text())
        arm = UrdfArmKinematics(description=fixture['public_urdf'])
        for case in fixture['cases']:
            measured, command, goal = map(np.asarray, (case['measured'], case['command'], case['goal']))
            before = [value.copy() for value in (measured, command, goal)]
            plan = regrasp_posture_reserve(arm, measured, command, goal)
            self.assertEqual(plan is not None, case['expected_accepted'])
            for value, original in zip((measured, command, goal), before):
                np.testing.assert_array_equal(value, original)
            if plan is not None:
                self.assertGreaterEqual(plan['wrist_reserve_degrees'], 15.)
                self.assertLessEqual(plan['predicted_position_m'], .008)
                self.assertLessEqual(plan['predicted_angle_degrees'], 9.3)
                for joints in (plan['reference'], plan['reference']+measured[8:15]-command[8:15]):
                    self.assertTrue(np.all(joints >= arm.limits[1][:, 0]-1e-9))
                    self.assertTrue(np.all(joints <= arm.limits[1][:, 1]+1e-9))
            goal[0, 3] += 1.
            self.assertIsNone(regrasp_posture_reserve(arm, measured, command, goal))

    def reserve_refinement_case(self, index):
        fixtures = Path(__file__).parent/'fixtures'
        robot = json.loads((fixtures/'weighing_posture_reserve_v46.json').read_text())
        case = json.loads((fixtures/'weighing_reserve_refinement_v49.json').read_text())['cases'][index]
        return UrdfArmKinematics(description=robot['public_urdf']), case

    def test_actual_rejected_preparation_can_preserve_minimum_reserve_without_task_tradeoff(self):
        arm, case = self.reserve_refinement_case(0)
        measured, command, goal = map(np.asarray, (case['measured'], case['command'], case['goal']))
        originals = [v.copy() for v in (measured, command, goal)]
        plan = regrasp_posture_reserve(arm, measured, command, goal)
        self.assertIsNotNone(plan)
        self.assertTrue(plan['same_goal_refined'])
        q = plan['reference']; actual = q+measured[8:15]-command[8:15]
        pose = arm.matrix(actual, 1)
        angle = (Rotation.from_matrix(goal[:3,:3])*Rotation.from_matrix(pose[:3,:3]).inv()).magnitude()
        self.assertLessEqual(np.linalg.norm(pose[:3,3]-goal[:3,3]), .008)
        self.assertLessEqual(np.rad2deg(angle), 9.3)
        self.assertGreaterEqual(np.rad2deg(arm.limits[1][4,1]-q[4]), 15.)
        for values in (q, actual):
            self.assertTrue(np.all(values >= arm.limits[1][:,0]-1e-9))
            self.assertTrue(np.all(values <= arm.limits[1][:,1]+1e-9))
        for value, original in zip((measured, command, goal), originals):
            np.testing.assert_array_equal(value, original)

    def test_already_accepted_actual_third_parcel_plan_is_preserved(self):
        arm, case = self.reserve_refinement_case(1)
        plan = regrasp_posture_reserve(arm, *map(np.asarray,
            (case['measured'], case['command'], case['goal'])))
        self.assertIsNotNone(plan)
        self.assertFalse(plan['same_goal_refined'])
        for name in ('predicted_position_m','predicted_angle_degrees','wrist_reserve_degrees'):
            self.assertAlmostEqual(plan[name], case['recorded_plan'][name], places=8)

    def test_actual_infeasible_reserve_stays_rejected_after_same_goal_refinement(self):
        arm, case = self.reserve_refinement_case(2)
        self.assertIsNone(regrasp_posture_reserve(arm, *map(np.asarray,
            (case['measured'], case['command'], case['goal']))))

    def test_saved_fourth_pose_pitch_counterfactual_keeps_original_reserve_gates(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/pickup_pitch_reserve_v60.json').read_text())
        arm = UrdfArmKinematics(description=fixture['public_urdf'])
        measured, command, goal = map(np.asarray,
            (fixture['measured'], fixture['command'], fixture['goal']))
        self.assertIsNone(regrasp_posture_reserve(arm, measured, command, goal))
        adjusted = goal.copy()
        adjusted[:3,:3] = goal[:3,:3] @ Rotation.from_euler('x', 2.5, degrees=True).as_matrix()
        plan = regrasp_posture_reserve(arm, measured, command, adjusted)
        self.assertIsNotNone(plan)
        self.assertFalse(plan['same_goal_refined'])
        self.assertGreater(plan['wrist_reserve_degrees'], 19.)
        self.assertLess(plan['predicted_angle_degrees'], 9.3)
        self.assertLess(plan['predicted_position_m'], .008)
        np.testing.assert_array_equal(adjusted[:3,3], goal[:3,3])
        # The model must still reject the actual old goal; it never changes a gate.
        self.assertIsNone(regrasp_posture_reserve(arm, measured, command, goal))
