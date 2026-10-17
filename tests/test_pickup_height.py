"""Real v61 grasp targets: ACT-local height must not bias later visual grasps."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scene1.kinematics import UrdfArmKinematics
from scene1.residual import ResidualAssist
from scene1.rgbd import metric_candidates
from scene1.visual_cycle import VisualCycle
from scene1.visual_pickup import VisualPickup


class PickupHeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).parent/'fixtures'
        cls.fixture = json.loads((cls.directory/'pickup_height_v61.json').read_text())
        cls.k = UrdfArmKinematics(description=cls.fixture['public_urdf'])

    def setup_case(self, case, enabled):
        frame = case['frame']
        measured, command = np.array(frame['measured']), np.array(frame['previous_command'])
        config = dict(case['visual_config'])
        if enabled is not None:
            config['pickup_center_grasp_height'] = enabled
        cycle = VisualCycle(config, self.k, np.full(16,-10.), np.full(16,10.), command)
        cycle.completed = int(case['tag'].split('_')[-1])-1
        pickup = VisualPickup(cycle, copy.deepcopy(case['template']), [0.,0.,0.], measured, command)
        pickup.target = copy.deepcopy(frame['target']); pickup.plane = frame['plane']
        assist = ResidualAssist(case['config'], self.k, np.full(16,-10.), np.full(16,10.))
        assist.wrist_config = case['wrist_config']
        return pickup, assist, measured

    def test_real_targets_use_observed_height_and_preserve_xy_pitch_and_insertion(self):
        for case in self.fixture['cases']:
            old, old_assist, measured = self.setup_case(case, None)
            new, assist, _ = self.setup_case(case, True)
            old_goal, goal = old._target_goal(measured, old_assist), new._target_goal(measured, assist)
            frame = case['frame']
            np.testing.assert_allclose(old.approach_goal[:3,3]+[0.,0.,.005],
                                       np.array(frame['goal'])[:3,3], atol=1e-12)
            np.testing.assert_allclose(goal[:2,3], old_goal[:2,3], atol=1e-12)
            np.testing.assert_array_equal(goal[:3,:3], old_goal[:3,:3])
            np.testing.assert_allclose(new.approach_goal[:2,3], old.approach_goal[:2,3], atol=1e-12)
            expected = (frame['plane']+frame['target']['top_z_base_m'])/2
            self.assertAlmostEqual(goal[2,3], expected, places=12)
            self.assertAlmostEqual(new.approach_goal[2,3], expected, places=12)
            self.assertEqual(new.target['extra_insertion_m'], .015)
            self.assertEqual(new.template, case['template'])
            if case['template']['box_to_tool_m'][2] > 0:
                self.assertGreater(old_goal[2,3]-goal[2,3], .0053)
                self.assertLess(old_goal[2,3]-goal[2,3], .0055)
            else:
                np.testing.assert_array_equal(goal, old_goal)
                np.testing.assert_array_equal(new.approach_goal, old.approach_goal)

    def test_explicitly_disabled_keeps_original_requested_height(self):
        for case in self.fixture['cases']:
            old, old_assist, measured = self.setup_case(case, None)
            disabled, assist, _ = self.setup_case(case, False)
            np.testing.assert_array_equal(old._target_goal(measured, old_assist),
                                          disabled._target_goal(measured, assist))
            self.assertNotIn('grasp_height_source', disabled.target)

    def test_real_raw_depth_confirms_visual_midpoint_without_annotations(self):
        with np.load(self.directory/'pickup_height_v61.npz') as raw:
            for case in self.fixture['cases']:
                frame = case['frame']; tag = case['tag']
                args = [raw[tag+'_'+key] for key in ('rgb','depth_m','intrinsic','camera_to_base')]
                candidates, plane = metric_candidates(*args, case['rgbd_config'])
                target = min(candidates, key=lambda c: np.linalg.norm(
                    np.asarray(c['position_base_m'])[:2]-frame['target']['position_base_m'][:2]))
                self.assertAlmostEqual(plane, frame['plane'], places=12)
                self.assertLess(abs((target['top_z_base_m']+plane)/2-
                    (frame['target']['top_z_base_m']+frame['plane'])/2), .0005)

    def test_center_height_cannot_bypass_original_preclose_orientation_guard(self):
        fixture = json.loads((self.directory/'initial_grasp_tilt_v48.json').read_text())
        measured, command = np.array(fixture['measured']), np.array(fixture['previous_command'])
        k = UrdfArmKinematics(description=fixture['public_urdf'])
        cycle = VisualCycle(dict(pickup_level_grasp=True, pickup_center_grasp_height=True,
            pickup_extra_insertion_m=.015, pickup_pitch_offset_degrees=5.), k,
            np.full(16,-10.), np.full(16,10.), command)
        pickup = VisualPickup(cycle, copy.deepcopy(fixture['template']), [0.,0.,0.], measured, command)
        pickup.target = copy.deepcopy(fixture['target']); pickup.plane = fixture['plane']
        assist = ResidualAssist(fixture['config'], k, np.full(16,-10.), np.full(16,10.))
        assist.wrist_config = fixture['wrist_config']
        goal = pickup._target_goal(measured, assist)
        assist._begin('grasp', 1, command, target=pickup.target, plane=pickup.plane,
            grasp_z=goal[2,3], wrist_verified=False, grasp_rotation=Rotation.from_matrix(goal[:3,:3]))
        assist.pending['stable'] = 1
        action, info = assist._pending_action(measured, None)
        self.assertEqual(action[15], 0.)
        self.assertGreater(info['finger_line_tilt_degrees'], 5.)
        self.assertNotEqual(assist.pending.get('stage'), 'close_confirm')
