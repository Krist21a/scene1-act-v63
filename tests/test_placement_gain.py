import unittest
from types import SimpleNamespace
import numpy as np
from scene1.placement_gain import PlacementGain


class PlacementGainTests(unittest.TestCase):
    def setUp(self):
        self.original = dict(gain=.25,max_joint_step_rad=.012,max_bottom_height_spread_m=.023)
        self.assist = SimpleNamespace(config=self.original,pending=dict(kind='weighing'),offset=np.arange(16.)/100)

    def test_initial_leveling_is_unchanged_then_damping_latches_without_reset(self):
        schedule=PlacementGain(.4);schedule.before_step(self.assist)
        self.assertIs(self.assist.config,self.original)
        self.assist.pending['level_ready']=True;before=self.assist.offset.copy()
        schedule.before_step(self.assist)
        self.assertAlmostEqual(self.assist.config['gain'],.1)
        self.assertEqual(self.assist.config['max_bottom_height_spread_m'],.023)
        np.testing.assert_array_equal(self.assist.offset,before)
        self.assist.pending['level_ready']=False;schedule.before_step(self.assist)
        self.assertAlmostEqual(self.assist.config['gain'],.1)

    def test_short_retreat_restores_exact_original_configuration(self):
        schedule=PlacementGain(.4);self.assist.pending['level_ready']=True
        schedule.before_step(self.assist);self.assist.pending=dict(kind='short_retreat')
        schedule.after_step(self.assist)
        self.assertIs(self.assist.config,self.original)
        self.assist.pending=dict(kind='grasp');schedule.before_step(self.assist)
        self.assertIs(self.assist.config,self.original)

    def test_default_profile_remains_unchanged_and_invalid_gain_is_rejected(self):
        schedule=PlacementGain();self.assist.pending['level_ready']=True
        schedule.before_step(self.assist);self.assertIs(self.assist.config,self.original)
        for scale in (0.,1.1):
            with self.assertRaises(ValueError):PlacementGain(scale)
