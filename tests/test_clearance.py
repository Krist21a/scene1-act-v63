import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scene1.clearance import boxes_overlap, open_fingers_clear


class FingerClearanceTests(unittest.TestCase):
    def setUp(self):
        self.target = dict(position_base_m=[0.,0.,.04], top_z_base_m=.08,
                           major_axis_xy=[1.,0.], extent_xy_m=[.12,.04])
        self.pose = np.eye(4)
        # Fingers close across the short side; approach along the long side.
        self.pose[:3,:3] = np.array([[0,0,-1],[1,0,0],[0,-1,0]])
        self.pose[:3,3] = [-.04,0.,.04]

    def test_centered_open_fingers_straddle_parcel(self):
        self.assertTrue(open_fingers_clear(self.pose,self.target,0.))

    def test_lateral_offset_puts_one_finger_into_parcel(self):
        self.pose[1,3] = .025
        self.assertFalse(open_fingers_clear(self.pose,self.target,0.))

    def test_finger_roll_can_cause_collision_at_same_tcp(self):
        self.pose[:3,:3] = Rotation.from_euler('x',60,degrees=True).as_matrix()@self.pose[:3,:3]
        self.assertFalse(open_fingers_clear(self.pose,self.target,0.))

    def test_rotated_boxes_with_overlapping_axis_aligned_bounds_can_be_separate(self):
        rotation = Rotation.from_euler('z',45,degrees=True).as_matrix()
        half = np.array([1.,.05,.05])
        self.assertFalse(boxes_overlap(np.zeros(3),rotation,half,
                         np.array([-.2,.2,0.]),rotation,half))


if __name__ == '__main__':
    unittest.main()
