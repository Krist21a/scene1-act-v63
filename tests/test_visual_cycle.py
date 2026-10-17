import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from scene1.visual_cycle import VisualCycle, bin_rim, inside_bin, largest_surface, track_bin_rim


class Arm:
    limits = [np.tile([-3., 3.], (7, 1)) for _ in range(2)]

    def matrix(self, q, side):
        matrix = np.eye(4)
        matrix[:3, 3] = q[:3]
        matrix[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return matrix


class VisualCycleTests(unittest.TestCase):
    def setUp(self):
        self.measured = np.zeros(16); self.measured[8:11] = [.4, -.4, 0.]
        self.measured[15] = .5
        self.sent = self.measured.copy(); self.sent[15] = .98
        low = np.r_[[-3.]*7, 0., [-3.]*7, 0.]
        high = np.r_[[3.]*7, 1., [3.]*7, 1.]
        self.cycle = VisualCycle({}, Arm(), low, high, self.measured)
        self.cycle.rim = dict(center=np.array([.65, .25]), rim_z=.22, plane_z=-.08,
            corners=np.array([[.45,.05],[.85,.05],[.85,.45],[.45,.45]], np.float32), stamp=0.)
        geometry = dict(center=np.zeros(3), rotation=Rotation.identity(), extent=np.array([.11,.04,.1]))
        self.assist = SimpleNamespace(regrasp_complete=True, pending=None, geometry=[None, geometry],
                                      weighing_target=np.array([.4, -.4, -.08]))
        self.cycle.begin(self.assist, self.measured, self.sent)

    def test_rgbd_rim_moves_with_observation_and_pad_is_rejected(self):
        rgb = np.zeros((480, 848, 3), np.uint8)
        rgb[80:400,120:700] = [0,130,0]
        depth = np.zeros((480,848)); depth[80:400,120:700] = .3
        depth[90:390,130:690] = .05
        matrix = np.eye(4); matrix[:2,3] = [.3,.1]
        snap = dict(rgb=rgb, depth_m=depth, intrinsic=np.diag([400.,400.,1.]),
                    camera_to_base=matrix, table_z_base_m=0., stamp=1.)
        rim = bin_rim(snap)
        self.assertIsNotNone(rim)
        center = rim['center'].copy()
        snap['camera_to_base'][0,3] += .05
        np.testing.assert_allclose(bin_rim(snap)['center']-center, [.05,0], atol=.001)
        snap['table_z_base_m'] = .3
        self.assertIsNone(bin_rim(snap))

    def test_full_payload_footprint_must_clear_rim(self):
        self.assertTrue(inside_bin([[.6,.2,.4]], self.cycle.rim))
        self.assertFalse(inside_bin([[.6,.2,.4],[.46,.2,.4]], self.cycle.rim))

    def test_bin_waypoint_allows_small_pose_error_but_never_outside_footprint(self):
        for x, opened in ((.6, True), (.46, False)):
            with self.subTest(x=x):
                self.cycle._stage('bin_transport'); self.cycle.side = 0
                self.measured[:3] = [x, .2, .4]; self.measured[7] = .5
                self.cycle.command[7] = .98
                self.cycle.goal = [self.cycle.k.matrix(self.measured[:7], 0), None]
                self.cycle.goal[0][0, 3] += .02
                with patch.object(self.cycle, '_bin_points', return_value=np.empty((0, 3))):
                    for stamp in [.1, .2, .3, .4]:
                        action, _ = self.cycle.update(self.measured, dict(stamp=stamp), self.assist)
                self.assertEqual(action[7] == 0., opened)

    def test_visible_surface_evidence_rejects_scattered_colored_noise(self):
        x,y=np.mgrid[0:10,0:10]
        patch_points=np.column_stack((x.ravel()*.002,y.ravel()*.002,np.zeros(100)))
        noise=np.column_stack((np.arange(90)*.03+.3,np.ones(90),np.zeros(90)))
        np.testing.assert_array_equal(np.sort(largest_surface(np.vstack((patch_points,noise))),axis=0),
                                      np.sort(patch_points,axis=0))
        self.assertEqual(len(largest_surface(noise)),1)

    def test_previous_parcel_revealed_by_retreat_cannot_confirm_next_drop(self):
        x, y = np.mgrid[0:10, 0:10]
        surface = np.column_stack((x.ravel()*.002, y.ravel()*.002, np.zeros(100)))
        old = surface + [.65, .145, .08]
        new = surface + [.65, .215, .08]
        self.cycle.config = dict(max_parcels=4,
            bin_lateral_offsets_m=[-.105, -.035, .035, .105])
        for points, expected in ((old, 1), (np.vstack((old, new)), 2)):
            with self.subTest(expected=expected):
                self.cycle.completed = 1
                self.cycle._stage('verify_bin')
                # The old parcel was completely occluded before opening.
                self.cycle.bin_before = np.empty((0, 3))
                with patch.object(self.cycle, '_bin_points', return_value=points):
                    for stamp in (1., 1.25, 1.5, 1.75, 2.):
                        self.cycle.update(self.measured, dict(stamp=stamp), self.assist)
                self.assertEqual(self.cycle.completed, expected)

    def test_partial_rim_tracks_translation_without_shrinking_container(self):
        t=np.linspace(0,1,100)
        points=np.concatenate((np.column_stack((np.full(100,.85),.05+.4*t,np.full(100,.22))),
                               np.column_stack((.45+.4*t,np.full(100,.45),np.full(100,.22)))))
        points[:,:2]+=[.012,-.009]
        with patch('scene1.visual_cycle.cloud',return_value=(points,np.tile([60,150,150],(200,1)))):
            rim=track_bin_rim(dict(stamp=2.),self.cycle.rim)
        np.testing.assert_allclose(rim['center'],self.cycle.rim['center']+[.012,-.009],atol=.002)
        np.testing.assert_allclose(np.ptp(rim['corners'],axis=0),[.4,.4],atol=1e-6)
        with patch('scene1.visual_cycle.cloud',return_value=(points[:80],np.tile([60,150,150],(80,1)))):
            self.assertIsNone(track_bin_rim(dict(stamp=2.),self.cycle.rim))

    def test_known_rim_rejects_higher_green_hand_distractor(self):
        edge = np.linspace(0,1,100)
        points = np.concatenate([np.column_stack((.45+.4*edge,np.full(100,y),np.full(100,.22)))
                                 for y in (.05,.45)] +
                                [np.column_stack((np.full(100,x),.05+.4*edge,np.full(100,.22)))
                                 for x in (.45,.85)] +
                                [np.column_stack((.3+.1*edge,-.1+.1*edge,np.full(100,.34)))])
        colors = np.tile([60,150,150],(len(points),1))
        with patch('scene1.visual_cycle.cloud',return_value=(points,colors)):
            snapshot = dict(table_z_base_m=-.08,stamp=10.)
            self.assertIsNone(bin_rim(snapshot))
            rim = bin_rim(snapshot,self.cycle.rim)
            np.testing.assert_allclose(rim['center'],[.65,.25],atol=1e-6)
            self.assertAlmostEqual(rim['rim_z'],.22)

    def test_handover_approaches_are_on_opposite_ends_for_both_axis_signs(self):
        self.cycle.geometry['center'] = np.array([.003,-.003,-.044])
        original = Rotation.from_euler('xyz',[-4.24,-78.82,95.46],degrees=True)
        for flip in (0,180):
            self.cycle.geometry['rotation'] = original*Rotation.from_euler('z',flip,degrees=True)
            goal = self.cycle._handover_goal()
            self.measured[8:11] = goal[:3,3]
            self.measured[11:14] = Rotation.from_matrix(goal[:3,:3]).as_rotvec()
            center,_,_=self.cycle.world_box(self.measured)
            receiver = self.cycle._receiver_goal(self.measured,0.)
            self.assertLess(goal[1,3],center[1]-.015)
            self.assertGreater(receiver[1,3],center[1]+.015)
            self.assertLess(np.dot(goal[:3,3]-center,receiver[:3,3]-center),0.)

    def test_feedback_servo_respects_joint_step_and_keeps_giver_closed(self):
        before = self.cycle.command.copy()
        goal = self.cycle.goal[1].copy(); goal[0,3] += .2
        self.cycle._servo(self.measured,1,goal)
        self.assertLessEqual(max(abs(self.cycle.command-before)), .018+1e-9)
        self.assertEqual(self.cycle.command[15],.98)

    def test_empty_receiver_never_releases_giver(self):
        self.cycle._stage('close_left'); self.measured[7] = .99
        for stamp in range(10):
            action,_ = self.cycle.update(self.measured,dict(stamp=stamp),self.assist)
            self.assertEqual(action[15],.98)

    def test_receiver_insertion_waits_for_edges_and_measured_progress(self):
        self.cycle._stage('left_insert')
        self.cycle.goal[0] = self.cycle.k.matrix(self.measured[:7], 0)
        self.cycle.goal[1] = self.cycle.k.matrix(self.measured[8:15], 1)
        self.cycle.receiver_nominal = np.zeros(3)
        self.cycle.receiver_offset = 0.
        self.cycle.receiver_insert_start = np.zeros(3)
        self.cycle.receiver_insert_m = 0.
        aligned = dict(centering_error_px=0., correction_base_m=[0., 0., 0.])
        with patch.object(self.cycle, '_wrist_support', return_value=None):
            for stamp in range(5):
                action, _ = self.cycle.update(self.measured, dict(stamp=stamp), self.assist)
                self.assertEqual(self.cycle.receiver_insert_m, 0.)
                self.assertEqual(action[7], 0.)
        with patch.object(self.cycle, '_wrist_support', return_value=aligned):
            for stamp in range(5, 25):
                action, _ = self.cycle.update(self.measured, dict(stamp=stamp), self.assist)
                self.assertLessEqual(self.cycle.receiver_insert_m, .006)
                self.assertEqual(action[7], 0.)
                self.assertEqual(action[15], .98)

    def test_stable_receiver_contact_precedes_right_release(self):
        self.cycle._stage('close_left'); self.cycle.command[7] = .98
        self.measured[7] = .48
        for stamp in range(3):
            action,_ = self.cycle.update(self.measured,dict(stamp=stamp),self.assist)
            self.assertEqual(action[15],.98)
        action,_ = self.cycle.update(self.measured,dict(stamp=4),self.assist)
        self.assertEqual(action[15],0.)
        self.assertEqual(self.cycle.side,0)
        self.assertEqual(self.cycle.stage,'release_right')

    def test_receiver_position_is_held_while_giver_withdraws(self):
        self.cycle._stage('close_left'); self.cycle.command[7] = .98
        self.measured[7] = .48
        for stamp in range(4):
            self.cycle.update(self.measured,dict(stamp=stamp),self.assist)
        goal = self.cycle.goal[0].copy()
        self.measured[15] = .01
        for stamp in range(4,7):
            self.cycle.update(self.measured,dict(stamp=stamp),self.assist)
        self.assertEqual(self.cycle.stage,'right_retreat')
        np.testing.assert_array_equal(self.cycle.goal[0],goal)
        self.measured[2] -= .02
        before = self.cycle.command.copy()
        self.cycle._arrived(self.measured)
        self.assertGreater(self.cycle.command[2],before[2])

    def test_duplicate_camera_evidence_cannot_complete_contact(self):
        self.cycle._stage('close_left'); self.measured[7] = .48
        for _ in range(8):
            action,_ = self.cycle.update(self.measured,dict(stamp=1),self.assist)
            self.assertEqual(action[15],.98)

    def test_lost_payload_stops_without_opening_the_giver(self):
        self.cycle._stage('close_left'); self.measured[15] = .99
        for i in range(3):
            self.cycle.update(self.measured,dict(stamp=i),self.assist)
        with self.assertRaisesRegex(RuntimeError,'contact lost'):
            self.cycle.update(self.measured,dict(stamp=4),self.assist)
        self.assertEqual(self.cycle.command[15],.98)

    def test_fresh_timestamps_cannot_mask_frozen_robot_feedback(self):
        self.cycle._stage('transport')
        goal = self.cycle.goal[1].copy(); goal[0, 3] += .3
        with patch.object(self.cycle, '_handover_goal', return_value=goal):
            with self.assertRaisesRegex(RuntimeError, 'feedback values are frozen'):
                for stamp in range(40):
                    self.cycle.update(self.measured, dict(stamp=stamp), self.assist)
        self.assertEqual(self.cycle.command[15], .98)

    def test_empty_hand_retreat_requires_clearance_but_not_grasp_orientation(self):
        for height, expected in ((.25,'left_retreat'),(.5,'verify_bin')):
            with self.subTest(height=height):
                self.cycle.side = 0
                self.cycle._stage('left_retreat')
                self.measured[2] = height
                goal = self.cycle.k.matrix(self.measured[:7],0)
                goal[:3,:3] = Rotation.from_euler('z',30,degrees=True).as_matrix()
                self.cycle.goal = [goal,None]
                self.cycle.command[[7,15]] = 0.
                for stamp in range(4):
                    self.cycle.update(self.measured,dict(stamp=stamp),self.assist)
                self.assertEqual(self.cycle.stage,expected)

    def test_checkpoint_preserves_commands_geometry_and_current_controller_config(self):
        saved = self.cycle.state_dict()
        self.cycle.command[:] = 0.
        self.cycle.config = {'servo_gain': .1}
        self.cycle.restore(saved,self.measured,self.measured)
        np.testing.assert_array_equal(self.cycle.command,self.sent)
        self.assertEqual(self.cycle.config['servo_gain'],.1)
        self.assertEqual(self.cycle.rim['corners'].dtype,np.float32)
        self.assertEqual(self.cycle.stage,'lift')

    def test_missing_rgbd_holds_arms_and_grips_then_stops(self):
        before = self.cycle.command.copy()
        for _ in range(9):
            action, info = self.cycle.update(self.measured,None,self.assist)
            np.testing.assert_array_equal(action,before)
            self.assertEqual(info['state'],'visual_cycle_sensor_hold')
        with self.assertRaisesRegex(RuntimeError,'10 consecutive'):
            self.cycle.update(self.measured,None,self.assist)
        np.testing.assert_array_equal(self.cycle.command,before)

    def test_checkpoint_rejects_different_scene_pose_or_missing_payload(self):
        saved = self.cycle.state_dict()
        moved = self.measured.copy(); moved[8] += .2
        with self.assertRaisesRegex(RuntimeError,'physical arm state'):
            self.cycle.restore(saved,moved,self.measured)
        moved = self.measured.copy(); moved[15] = .99
        with self.assertRaisesRegex(RuntimeError,'payload contact'):
            self.cycle.restore(saved,moved,self.measured)

    def test_checkpoint_accepts_settling_toward_last_command(self):
        self.cycle.command[14] = .3
        saved = self.cycle.state_dict()
        settled = self.measured.copy(); settled[14] = .23
        self.cycle.restore(saved,settled,self.measured)
        self.assertEqual(self.cycle.command[14],.3)


if __name__ == '__main__':
    unittest.main()
