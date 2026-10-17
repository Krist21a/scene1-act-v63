import copy
import json
from pathlib import Path
import unittest
import numpy as np
from scene1.table_reference import table_frame, table_motion, move_target, table_boundary
from scene1.visual_pickup import table_hull
from scene1.residual import ResidualAssist


class TableReferenceTests(unittest.TestCase):
    def setUp(self):
        self.hull=np.array([[.2,-.9],[1.,-.9],[1.,.5],[.2,.5]],np.float32)
        self.reference=table_frame(self.hull,-.08)
        self.target=dict(position_base_m=[.48,-.31,-.02],top_z_base_m=.02,
            major_axis_xy=[1.,0.],oriented_extent_m=[.11,.04],extent_xy_m=[.11,.04],
            grasp_goal_xy=[.45,-.31])

    def test_common_motion_is_removed_but_real_parcel_motion_still_trips_guard(self):
        moved=table_frame(self.hull+[.012,-.004],-.079)
        motion=table_motion(self.reference,moved)
        target=move_target(self.target,motion['transform'])
        np.testing.assert_allclose(np.asarray(target['grasp_goal_xy'])-[.45,-.31],[.012,-.004],atol=1e-6)
        candidate=copy.deepcopy(target)
        self.assertLess(ResidualAssist._parcel_displacement(dict(candidates=[candidate]),target),1e-6)
        candidate['position_base_m'][0]+=.016
        self.assertGreater(ResidualAssist._parcel_displacement(dict(candidates=[candidate]),target),.01)
        self.assertEqual(self.target['position_base_m'],[.48,-.31,-.02])

    def test_small_rotation_moves_goal_once_and_incomplete_or_large_changes_reject(self):
        angle=np.deg2rad(.6);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        new_hull=(self.hull-self.reference['center'][:2])@rotation.T+self.reference['center'][:2]+[.006,0.]
        result=table_motion(self.reference,table_frame(new_hull,-.08))
        self.assertAlmostEqual(result['angle_degrees'],.6,places=4)
        self.assertIsNone(table_motion(self.reference,table_frame(self.hull+[.04,0.],-.08)))
        resized=self.hull.copy();resized[:,0]*=1.05
        self.assertIsNone(table_motion(self.reference,table_frame(resized,-.08)))
        self.assertIsNone(table_frame(self.hull[:3],-.08))
        self.assertIsNone(table_motion(self.reference,None))

    def test_recorded_third_approach_separates_table_motion_from_parcel_motion(self):
        events=json.loads((Path(__file__).parent/'fixtures/table_motion_v32.json').read_text())['events']
        first,last=events[0],events[-1]
        reference=table_frame(first['hull_xy_m'],first['plane_z'])
        current=table_frame(last['hull_xy_m'],last['plane_z'])
        result=table_motion(reference,current)
        self.assertIsNotNone(result)
        for observation in last['comparisons']:
            before=np.r_[observation['origin_xy_m'],0.,1.]
            predicted=(result['transform']@before)[:2]
            distance=np.linalg.norm(predicted-observation['observed_xy_m'])
            self.assertLess(distance,.007)
        self.assertGreater(last['comparisons'][0]['raw_shift_m'],.01)

    def test_raw_fourth_approach_keeps_actual_edge_discarded_by_hull(self):
        with np.load(Path(__file__).parent/'fixtures/table_hull_failure_rgbd.npz') as data:
            snapshots = [{key: data[tag+'_'+key] for key in
                ('rgb', 'depth_m', 'intrinsic', 'camera_to_base', 'table_z_base_m')}
                for tag in ('reference', 'failure')]
        frames = []
        for snapshot in snapshots:
            hull = table_hull(snapshot)
            frame = table_frame(hull, snapshot['table_z_base_m'], table_boundary(snapshot))
            self.assertIsNotNone(frame)
            self.assertGreater(min(frame['support']), .18)
            frames.append(frame)
        self.assertIsNone(table_frame(table_hull(snapshots[1]), snapshots[1]['table_z_base_m']))
        motion = table_motion(*frames)
        self.assertIsNotNone(motion)
        self.assertLess(np.linalg.norm(motion['center_shift_m']), .005)

    def depth_fixture(self):
        snapshot = dict(rgb=np.zeros((100, 100, 3), np.uint8), depth_m=np.full((100, 100), 2.05),
            intrinsic=np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]),
            camera_to_base=np.diag([1., -1., -1., 1.]), table_z_base_m=0.)
        snapshot['camera_to_base'][2, 3] = 2.
        snapshot['rgb'][20:80, 20:80] = [0, 130, 0]
        snapshot['depth_m'][20:80, 20:80] = 2.
        return snapshot

    def test_missing_or_image_clipped_table_edge_still_rejects(self):
        for clipped in (False, True):
            with self.subTest(clipped=clipped):
                snapshot = self.depth_fixture()
                hull = table_hull(snapshot)
                self.assertIsNotNone(table_frame(hull, 0., table_boundary(snapshot)))
                if clipped:
                    snapshot['rgb'][:20, 20:80] = [0, 130, 0]
                    snapshot['depth_m'][:20, 20:80] = 2.
                else:
                    # A hand covers the whole far edge; its raised boundary
                    # must not substitute for a visible table-to-floor edge.
                    snapshot['rgb'][:30] = [90, 90, 90]
                    snapshot['depth_m'][:30] = 1.9
                self.assertIsNone(table_frame(table_hull(snapshot), 0., table_boundary(snapshot)))

    def test_high_hand_inside_table_is_not_an_exterior_boundary(self):
        snapshot = self.depth_fixture()
        snapshot['rgb'][40:60, 40:60] = [90, 90, 90]
        snapshot['depth_m'][40:60, 40:60] = 1.9
        boundary = table_boundary(snapshot)
        self.assertFalse(np.any(np.all(abs(boundary) < .3, axis=1)))
        self.assertIsNotNone(table_frame(table_hull(snapshot), 0., boundary))
