import unittest
import numpy as np
from scene1.head import HeadLookController


class HeadTests(unittest.TestCase):
    def setUp(self):
        config = dict(home_degrees=[0,20], limits_degrees=[70,25], speed_degrees_s=[35,20],
                      home_tolerance_degrees=2.6, stable_frames=3)
        xml = '<robot><joint name="head_yaw" type="revolute"><parent link="base_link"/><child link="zhead_1_link"/><origin xyz="0 0 .6"/></joint></robot>'
        self.head = HeadLookController(config, xml)
        a = np.sqrt(.5)
        self.camera = np.eye(4)
        self.camera[:3, :3] = [[0,-a,a],[-1,0,0],[0,-a,-a]]
        self.camera[:3,3] = [.121,0,.035]
        self.snapshot = dict(head_angles_rad=np.deg2rad([0,20]),
                             camera_to_base=self.head.head_matrix(np.deg2rad([0,20])) @ self.camera)

    def test_left_right_targets_are_centered_in_camera(self):
        for point in ([.44,-.55,-.08], [.5,.35,.05]):
            goal = self.head.desired(point, self.snapshot)
            self.assertEqual(np.sign(goal[0]), np.sign(point[1]))
            camera = self.head.head_matrix(goal) @ self.camera
            ray = camera[:3,:3].T @ (np.asarray(point) - camera[:3,3])
            self.assertGreater(ray[2],0)
            np.testing.assert_allclose(ray[:2], 0, atol=1e-5)

    def test_head_motion_is_limited(self):
        command, info = self.head.update([.44,-.55,-.08], np.deg2rad([0,20]), self.snapshot, .1)
        self.assertLessEqual(abs(np.rad2deg(command)[0]),3.5+1e-9)
        self.assertLessEqual(abs(np.rad2deg(command)[1])-20,2+1e-9)
        self.assertFalse(info['act_view_ready'])

    def test_act_view_requires_measured_recovery_and_stability(self):
        _, info = self.head.update(None,np.deg2rad([-30,5]),self.snapshot,.1)
        self.assertFalse(info['act_view_ready'])
        for i in range(3):
            _, info = self.head.update(None,np.deg2rad([0,21]),self.snapshot,.1)
            self.assertEqual(info['act_view_ready'],i==2)


if __name__=='__main__':unittest.main()
