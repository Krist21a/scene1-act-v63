import copy
import unittest
from types import SimpleNamespace
import numpy as np
from scene1.placement_target import PlacementTarget


class PlacementTargetTests(unittest.TestCase):
    def assist(self):
        return SimpleNamespace(pending=dict(kind='weighing',stage='correct',level_ready=True,
            steps=100,last_box_step=100,stable=2,last_box_observation=dict(
                roof_normal_base=[.2,-.05,.9785],orientation_confirmed=True,tilt_degrees=12.)),
            weighing_target=np.array([.43,-.56,-.08]),geometry=[None,dict(extent=[.12,.04,.10])],
            config=dict(level_tolerance_degrees=15.,placement_margin_m=.025),offset=np.zeros(16))

    def test_compensation_is_bounded_latched_and_keeps_support_and_joint_state(self):
        assist=self.assist();assist.pending['surface']=dict(position_base_m=[.43,-.56,-.08])
        surface=copy.deepcopy(assist.pending['surface']);config=assist.config
        schedule=PlacementTarget(.008);original=assist.weighing_target.copy()
        schedule.before_step(assist);target=assist.weighing_target.copy()
        self.assertAlmostEqual(np.linalg.norm(target-original),.008)
        self.assertGreater(target[0],original[0]);self.assertLess(target[1],original[1])
        self.assertEqual(target[2],original[2]);self.assertEqual(assist.pending['stable'],0)
        assist.pending['last_box_observation']['roof_normal_base']=[-.2,0.,.98]
        schedule.before_step(assist);np.testing.assert_array_equal(assist.weighing_target,target)
        self.assertEqual(assist.pending['surface'],surface);self.assertIs(assist.config,config)
        np.testing.assert_array_equal(assist.offset,np.zeros(16))

    def test_default_and_unconfirmed_or_stale_observation_leave_target_untouched(self):
        cases=[('default',None),('level',False),('orientation',False),('stale',80),('stage','release')]
        for kind,value in cases:
            assist=self.assist();original=assist.weighing_target.copy()
            if kind=='level':assist.pending['level_ready']=value
            if kind=='orientation':assist.pending['last_box_observation']['orientation_confirmed']=value
            if kind=='stale':assist.pending['last_box_step']=value
            if kind=='stage':assist.pending['stage']=value
            schedule=PlacementTarget(0. if kind=='default' else .008);schedule.before_step(assist)
            np.testing.assert_array_equal(assist.weighing_target,original)
            self.assertIsNone(schedule.trace(assist))

    def test_other_stages_never_receive_the_target_adjustment(self):
        for kind in ('grasp','short_retreat','settle','pregrasp'):
            assist=self.assist();assist.pending['kind']=kind;original=assist.weighing_target.copy()
            PlacementTarget(.008).before_step(assist)
            np.testing.assert_array_equal(assist.weighing_target,original)
        for value in (-.001,.009,float('nan')):
            with self.assertRaises(ValueError):PlacementTarget(value)

    def test_axis_constraint_never_changes_lateral_goal_even_with_oblique_roof(self):
        assist=self.assist();original=assist.weighing_target.copy()
        schedule=PlacementTarget(.008,axis=0);schedule.before_step(assist)
        np.testing.assert_allclose(assist.weighing_target-original,[.008,0.,0.])
        self.assertEqual(schedule.trace(assist)['axis'],0)
        with self.assertRaises(ValueError):PlacementTarget(.008,axis=2)
