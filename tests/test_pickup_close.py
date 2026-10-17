"""Replay actual pre-close commands, retaining the original grasp state machine."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from scene1.clearance import open_fingers_clear
from scene1.kinematics import UrdfArmKinematics
from scene1.residual import ResidualAssist
from scene1.visual_cycle import VisualCycle
from scene1.visual_pickup import VisualPickup


class PickupCloseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent/'fixtures/pickup_close_v62.json').read_text())
        cls.k = UrdfArmKinematics(description=cls.fixture['public_urdf'])

    def assist(self, case):
        assist = ResidualAssist(case['config'].copy(), self.k, np.full(16,-10.), np.full(16,10.))
        assist.wrist_config = case['wrist_config']
        assist._begin('grasp', 1, np.array(case['base']), target=copy.deepcopy(case['target']),
            plane=case['plane'], grasp_z=case['goal'][2], wrist_verified=False,
            grasp_rotation=Rotation.from_matrix(case['rotation']))
        return assist

    def pickup(self, case, tolerance=None, stage='local_grasp'):
        config = {} if tolerance is None else dict(pickup_grasp_z_tolerance_m=tolerance)
        command = np.array(case['base'])
        motion = VisualCycle(config, self.k, np.full(16,-10.), np.full(16,10.), command)
        pickup = VisualPickup(motion, {}, [0.,0.,0.], command, command)
        pickup._stage(stage)
        return pickup

    def test_default_exactly_reproduces_nine_real_preclose_sequences(self):
        for case in self.fixture['cases']:
            assist = self.assist(case)
            for f in case['frames']:
                action, _ = assist._pending_action(np.array(f['measured']), None)
                np.testing.assert_array_equal(action, f['sent'])
            self.assertEqual(assist.pending['stage'], 'close_confirm')

    def test_stricter_height_uses_original_solver_budget_and_clear_fingers(self):
        for case in self.fixture['cases']:
            assist = self.assist(case); assist.config['grasp_z_tolerance_m'] = .005
            command = np.array(case['base']); original_limit = case['config']['max_grasp_steps']
            closed = False
            for i in range(original_limit):
                f = case['frames'][min(i,len(case['frames'])-1)]
                measured = command+np.array(f['measured'])-f['previous_command']
                before = command.copy()
                command, info = assist._pending_action(measured, None)
                delta = command-before
                self.assertLessEqual(max(abs(delta[8:15])), case['config']['max_joint_step_rad']+1e-10)
                for scale in (0.,.5,1.):
                    self.assertTrue(open_fingers_clear(self.k.matrix(measured[8:15]+scale*delta[8:15],1),
                                                      case['target'], case['plane']))
                if i+1 == len(case['frames']) and abs(info['z_error_m']) > .005:
                    self.assertEqual(command[15], 0.)
                if command[15] > .9:
                    self.assertLessEqual(abs(info['z_error_m']), .005)
                    self.assertLessEqual(info['finger_line_tilt_degrees'], 5.)
                    pose = self.k.matrix(measured[8:15],1)
                    angle = (Rotation.from_matrix(case['rotation'])*Rotation.from_matrix(pose[:3,:3]).inv()).magnitude()
                    self.assertLessEqual(angle, np.deg2rad(8.))
                    closed = True
                    break
            self.assertTrue(closed)
            self.assertLessEqual(i+1, original_limit)
            self.assertEqual(assist.pending['steps'], i+1)

    def test_only_initial_grasp_scopes_tolerance_and_always_restores_config(self):
        case = self.fixture['cases'][-1]
        for stage, tolerance, fail in [('local_grasp',.005,False),('local_grasp',.005,True),
                                       ('local_grasp',None,False),('local_weighing',.005,False)]:
            assist = self.assist(case); original = assist.config
            pickup = self.pickup(case,tolerance,stage)
            expected = .005 if stage=='local_grasp' and tolerance is not None else original['grasp_z_tolerance_m']
            def apply(*args):
                self.assertEqual(assist.config['grasp_z_tolerance_m'],expected)
                self.assertEqual(assist.config['max_grasp_steps'],original['max_grasp_steps'])
                self.assertEqual(assist.config['max_joint_offset_rad'],original['max_joint_offset_rad'])
                if fail:
                    raise RuntimeError('original local grasp failure')
                return np.array(case['base']),dict(state='small_grasp')
            with patch.object(assist,'apply',side_effect=apply), patch.object(assist,'_remember'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'original local grasp failure'):
                        pickup.update(np.array(case['base']),dict(stamp=1.),assist)
                else:
                    pickup.update(np.array(case['base']),dict(stamp=1.),assist)
            self.assertIs(assist.config,original)

    def test_invalid_or_weaker_height_tolerance_is_rejected(self):
        case = self.fixture['cases'][-1]
        for value in (0.,-.001,.01001,float('nan'),float('inf')):
            assist = self.assist(case); original = assist.config
            pickup = self.pickup(case,value)
            with patch.object(assist,'_remember'), patch.object(assist,'apply') as apply:
                with self.assertRaisesRegex(ValueError,'must tighten'):
                    pickup.update(np.array(case['base']),dict(stamp=1.),assist)
                apply.assert_not_called()
            self.assertIs(assist.config,original)
