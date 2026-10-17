import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from scene1.kinematics import UrdfArmKinematics
from scene1.visual_cycle import VisualCycle
from test_visual_cycle import Arm


class WeighingPoseRatioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=Path(__file__).parent/'fixtures'
        cls.fixture=json.loads((directory/'weighing_pose_ratio_v54.json').read_text())
        cls.raw=np.load(directory/'weighing_pose_ratio_v54.npz')
        cls.k=UrdfArmKinematics(description=cls.fixture['public_urdf'])

    def cycle(self,command):
        c=VisualCycle({},self.k,np.full(16,-10.),np.full(16,10.),command)
        c.command=np.array(command);c._stage('prepare_weighing');c.servo_precision[1]=True
        return c

    def test_default_reproduces_actual_failure_commands(self):
        for case in self.fixture['cases']:
            c=self.cycle(case['command']);c.goal=[None,np.array(case['goal'])]
            c.servo_previous[1]=self.k.matrix(np.array(case['previous_measured'])[8:15],1)
            c._arrived(np.array(case['measured']),position_limit=.008,angle_limit=np.deg2rad(10))
            np.testing.assert_allclose(c.command,case['old_sent_command'],rtol=0.,atol=1e-10)

    def test_common_scaling_preserves_pose_direction_and_both_original_caps(self):
        for xyz,angle in ((.006,.3),(.1,.02),(.001,.01)):
            m=np.zeros(16);c=VisualCycle({},Arm(),np.full(16,-3.),np.full(16,3.),m)
            c.command=m.copy();c._stage('prepare_weighing');c.servo_precision[1]=True
            goal=np.eye(4);goal[0,3]=xyz;goal[:3,:3]=Rotation.from_rotvec([0,angle,0]).as_matrix()
            with patch('scene1.visual_cycle.lsq_linear',side_effect=lambda matrix,target,**kw:SimpleNamespace(x=np.zeros(matrix.shape[1]))) as solve:
                c._servo(m,1,goal,coupled_pose_request=True)
            requested=solve.call_args.args[1][:6]/.06
            self.assertLessEqual(np.linalg.norm(requested[:3]),.012+1e-12)
            self.assertLessEqual(np.linalg.norm(requested[3:])/.12,.045+1e-12)
            self.assertAlmostEqual(requested[0]/xyz,requested[4]/(.12*angle))

    def test_real_saved_geometry_reaches_original_checks_in_remaining_budget(self):
        # This regression is a frozen-feedback solver model, not a physical
        # or image sequence success test. The actual raw geometry is retained.
        for case in self.fixture['cases']:
            prefix=f"frame_{case['step']}_"
            center=self.raw[prefix+'held_center_tool_m'];extent=self.raw[prefix+'held_extent_m']
            relative=Rotation.from_quat(self.raw[prefix+'held_rotation_tool_xyzw'])
            command=np.array(case['command']);measured=np.array(case['measured']);bias=measured[8:15]-command[8:15]
            goal=np.array(case['goal']);surface_z=case['surface']['position_base_m'][2]
            def bottom(q):
                pose=self.k.matrix(q,1);rot=Rotation.from_matrix(pose[:3,:3])
                return pose[2,3]+rot.apply(center)[2]-abs((rot*relative).as_matrix()[2])@extent/2
            optical=case['seen']['bottom_base_m']-bottom(measured[8:15])
            for enabled in (False,True):
                with self.subTest(step=case['step'],coupled=enabled):
                    c=self.cycle(command);stable=0;ready=None
                    for count in range(case['remaining']):
                        actual=c.command.copy();actual[8:15]+=bias
                        pose=self.k.matrix(actual[8:15],1);gap=bottom(actual[8:15])+optical-surface_z
                        target=goal.copy();target[2,3]=pose[2,3]+np.clip(.03-gap,-.02,.02)
                        before=c.command.copy();pos,angle,_=c._servo(actual,1,target,coupled_pose_request=enabled)
                        tilt=np.arccos(np.clip((Rotation.from_matrix(pose[:3,:3])*relative).as_matrix()[2,2],-1,1))
                        self.assertLessEqual(np.max(abs(c.command-before)),.006+1e-10)
                        self.assertTrue(np.all(c.command>=c.low-1e-12));self.assertTrue(np.all(c.command<=c.high+1e-12))
                        np.testing.assert_array_equal(c.command[[7,15]],command[[7,15]])
                        good=pos<.008 and angle<np.deg2rad(10) and tilt<=np.deg2rad(10) and .022<=gap<=.038
                        stable=stable+1 if good else 0
                        if stable>=3:ready=count+1;break
                    self.assertEqual(ready is not None,enabled)
                    np.testing.assert_array_equal(goal,np.array(case['goal']))

    def test_bounded_request_cannot_hide_original_large_orientation_error(self):
        m=np.zeros(16);c=VisualCycle({},Arm(),np.full(16,-3.),np.full(16,3.),m)
        c.command=m.copy();c._stage('prepare_weighing');goal=np.eye(4)
        goal[:3,:3]=Rotation.from_euler('y',11,degrees=True).as_matrix();c.goal=[None,goal]
        for _ in range(4):
            arrived,errors=c._arrived(m,position_limit=.008,angle_limit=np.deg2rad(10),coupled_pose_request=True)
            self.assertFalse(arrived);self.assertAlmostEqual(np.rad2deg(errors[0]['angle_rad']),11.)
