import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from scene1.kinematics import UrdfArmKinematics
from scene1.visual_cycle import VisualCycle
from scene1.visual_pickup import VisualPickup
from scene1.rgbd import metric_candidates
from scene1.residual import ResidualAssist
from scene1.clearance import open_fingers_clear
from test_visual_cycle import Arm


class InitialDescentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=Path(__file__).parent/'fixtures'
        cls.fixture=json.loads((cls.directory/'initial_descent_drift_v52.json').read_text())
        cls.k=UrdfArmKinematics(description=cls.fixture['public_urdf'])

    def cycle(self,frame,arm=None):
        c=VisualCycle(self.fixture['visual_config'],arm or self.k,np.full(16,-10.),np.full(16,10.),frame['previous_command'])
        c.command=np.array(frame['previous_command']);c._stage('descend');c.goal=[None,np.array(frame['goal'])]
        return c

    def test_real_old_commands_reproduce_when_descent_alignment_is_not_requested(self):
        for frame in self.fixture['precontact_sequence']:
            c=self.cycle(frame);c.servo_precision[1]=frame['servo_precision']
            if frame['previous_measured'] is not None:
                c.servo_previous[1]=self.k.matrix(np.array(frame['previous_measured'])[8:15],1)
            c._arrived(np.array(frame['measured']))
            np.testing.assert_allclose(c.command,frame['old_sent_command'],atol=1e-10,rtol=0.)

    def test_real_precontact_step_prioritizes_lateral_alignment_and_keeps_limits(self):
        frame=next(f for f in self.fixture['frames'] if f['step']==9360)
        m=np.array(frame['measured']);pose=self.k.matrix(m[8:15],1);goal=np.array(frame['goal']);corrections=[]
        for enabled in (False,True):
            c=self.cycle(frame);before=c.command.copy();c._arrived(m,descent_alignment=enabled)
            delta=c.command-before
            self.assertLessEqual(np.max(abs(delta)),.012+1e-10)
            self.assertEqual(c.command[15],0.)
            self.assertTrue(np.all(c.command>=c.low));self.assertTrue(np.all(c.command<=c.high))
            predicted=self.k.matrix(m[8:15]+delta[8:15],1)
            self.assertTrue(open_fingers_clear(predicted,frame['target'],frame['plane']))
            corrections.append(predicted[:3,3]-pose[:3,3])
        direction=(goal[:2,3]-pose[:2,3]);direction/=np.linalg.norm(direction)
        self.assertGreater(corrections[1][:2]@direction,corrections[0][:2]@direction)
        self.assertLess(abs(corrections[1][2]),abs(corrections[0][2]))

    def test_intermediate_height_and_bad_orientation_cannot_complete_descent(self):
        m=np.zeros(16);goal=np.eye(4);goal[2,3]=-.14
        for rotation in (Rotation.identity(),Rotation.from_euler('y',9,degrees=True)):
            m[11:14]=rotation.as_rotvec()
            frame=dict(previous_command=m.tolist(),goal=goal.tolist());c=self.cycle(frame,Arm())
            for _ in range(4):
                arrived,errors=c._arrived(m,descent_alignment=True)
                self.assertFalse(arrived);self.assertAlmostEqual(errors[0]['position_m'],.14)
            if rotation.magnitude()>.01:
                np.testing.assert_allclose(c.command[8:11],m[8:11],atol=1e-12)

    def test_real_overlap_holds_open_and_cannot_start_grasp_even_if_arrival_claimed(self):
        frame=next(f for f in self.fixture['frames'] if f['step']==9376);m=np.array(frame['measured']);c=self.cycle(frame)
        pickup=VisualPickup(c,{},[.425,-.56,-.084],m,c.command)
        pickup.target=copy.deepcopy(frame['target']);pickup.plane=frame['plane'];pickup.grasp_goal=np.array(frame['goal'])
        pickup.motion.goal=[None,pickup.grasp_goal.copy()];pickup._stage('descend')
        assist=SimpleNamespace(_remember=lambda _:None,_parcel_displacement=lambda *_:0.)
        before=pickup.motion.command.copy()
        with patch.object(pickup.motion,'_arrived',return_value=(True,[])):
            for stamp in range(1,20):
                action,info=pickup.update(m,dict(stamp=stamp),assist)
                np.testing.assert_array_equal(action,before);self.assertEqual(pickup.stage,'descend')
                self.assertFalse(info['open_fingers_clear']);self.assertEqual(pickup.motion.stable,0)
            with self.assertRaisesRegex(RuntimeError,'Open finger sweep intersects'):
                pickup.update(m,dict(stamp=20),assist)
        self.assertEqual(pickup.motion.stage_steps,20)

    def test_raw_failed_frame_retains_original_real_motion_guard(self):
        raw=np.load(self.directory/'initial_descent_drift_v52.npz');tops=[]
        for prefix in ('before_','failed_'):
            candidates,plane=metric_candidates(raw[prefix+'rgb'],raw[prefix+'depth_m'],raw[prefix+'intrinsic'],raw[prefix+'camera_to_base'],self.fixture['rgbd_config'])
            target=self.fixture['failed_target'];candidate=min(candidates,key=lambda x:np.linalg.norm(np.asarray(x['position_base_m'])[:2]-target['position_base_m'][:2]))
            tops.append(candidate['top_z_base_m'])
            if prefix=='failed_':
                assist=ResidualAssist(self.fixture['rgbd_config'],self.k,np.full(16,-10.),np.full(16,10.))
                self.assertGreater(assist._parcel_displacement(dict(candidates=candidates),target),.01)
        self.assertGreater(tops[1]-tops[0],.012)
