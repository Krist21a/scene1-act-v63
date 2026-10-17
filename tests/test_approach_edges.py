import json
from pathlib import Path
import unittest

import numpy as np

from scene1.approach_edges import roof_edges, roof_edge_motion, supported_edge


class ApproachEdgeTests(unittest.TestCase):
    def synthetic(self, shift=(0., 0.), occlusion=False, width=.04):
        v, u = np.indices((160, 240))
        x, y = (u-120)*.9/700., -(v-80)*.9/700.
        roof = (abs(x-shift[0]) < .055) & (abs(y-shift[1]) < width/2)
        depth = np.ones(x.shape); depth[roof] = .9
        rgb = np.zeros((*x.shape, 3), np.uint8); rgb[:] = [0, 150, 0]
        rgb[roof] = [180, 170, 145]
        if occlusion:
            finger = roof & (x < -.025)
            depth[finger] = .82; rgb[finger] = 20
        matrix = np.diag([1., -1., -1., 1.]); matrix[2, 3] = 1.
        snapshot = dict(rgb=rgb, depth_m=depth, intrinsic=np.array(
            [[700., 0., 120.], [0., 700., 80.], [0., 0., 1.]]), camera_to_base=matrix)
        target = dict(position_base_m=[0., 0., .1], major_axis_xy=[1., 0.],
                      oriented_extent_m=[.11, .04], extent_xy_m=[.11, .04], top_z_base_m=.1)
        return snapshot, target

    def test_saved_partial_roofs_have_three_stable_physical_edges(self):
        for run in ['20261015-173011', '20261015-183318']:
            with self.subTest(run=run), np.load(Path(__file__).parent/'fixtures'/f'approach_edges_{run}.npz') as data:
                observations = []
                for name in ['reference', 'current']:
                    snapshot = {k: data[name+'_'+k] for k in ['rgb', 'depth_m', 'intrinsic', 'camera_to_base']}
                    observations.append(roof_edges(snapshot, json.loads(str(data[name+'_target']))))
                self.assertEqual(len(observations[0]), 4)
                motion = roof_edge_motion(*observations)
                self.assertIsNotNone(motion)
                self.assertEqual(len(motion['matched_edges']), 3)
                self.assertLess(motion['displacement_m'], .002)

    def test_real_depth_translation_beyond_original_limit_remains_detectable(self):
        snapshot, target = self.synthetic()
        reference = roof_edges(snapshot, target)
        self.assertEqual(len(reference), 4)
        for shift in [(.014, 0.), (-.014, 0.)]:
            with self.subTest(shift=shift):
                snapshot, _ = self.synthetic(shift=shift)
                motion = roof_edge_motion(reference, roof_edges(snapshot, target))
                self.assertIsNotNone(motion)
                self.assertGreater(motion['displacement_m'], .01)
                np.testing.assert_allclose(motion['shift_local_m'], shift, atol=.002)

    def test_large_lateral_motion_losing_edge_support_is_unknown_not_zero(self):
        snapshot, target = self.synthetic()
        reference = roof_edges(snapshot, target)
        for shift in [(0., .012), (0., -.012)]:
            snapshot, _ = self.synthetic(shift=shift)
            self.assertIsNone(roof_edge_motion(reference, roof_edges(snapshot, target)))

    def test_higher_finger_is_not_an_exterior_parcel_edge(self):
        snapshot, target = self.synthetic()
        reference = roof_edges(snapshot, target)
        occluded, _ = self.synthetic(occlusion=True)
        current = roof_edges(occluded, target)
        self.assertNotIn('0:-1', current)
        motion = roof_edge_motion(reference, current)
        self.assertIsNotNone(motion)
        self.assertLess(motion['displacement_m'], .001)

    def test_missing_edges_or_changed_dimensions_never_report_zero_motion(self):
        snapshot, target = self.synthetic()
        reference = roof_edges(snapshot, target)
        partial = {key: value for key, value in reference.items() if key.startswith('0:')}
        self.assertIsNone(roof_edge_motion(reference, partial))
        self.assertIsNone(roof_edge_motion(reference, None))
        snapshot, _ = self.synthetic(width=.052)
        self.assertIsNone(roof_edge_motion(reference, roof_edges(snapshot, target)))
        snapshot['depth_m'][:] = 0.
        self.assertIsNone(roof_edge_motion(reference, roof_edges(snapshot, target)))

    def test_real_corner_boundary_does_not_hide_the_supported_far_edge(self):
        with np.load(Path(__file__).parent/'fixtures/approach_edges_20261015-220604.npz') as data:
            edges = []
            for name in ('reference', 'current'):
                snapshot = {k: data[name+'_'+k] for k in ('rgb', 'depth_m', 'intrinsic', 'camera_to_base')}
                edges.append(roof_edges(snapshot, json.loads(str(data[name+'_target']))))
            edge = edges[1]['0:-1']
            self.assertEqual(edge['discarded_boundary_points'], 6)
            self.assertGreaterEqual(edge['points'], 20)
            self.assertGreater(edge['span_m'], .026)
            self.assertLess(edge['normal_spread_m'], .009)
            motion = roof_edge_motion(*edges)
            self.assertEqual(len(motion['matched_edges']), 3)
            self.assertLess(motion['displacement_m'], .003)
            # A real displacement must not be erased by grouping a corner.
            snapshot['camera_to_base'] = snapshot['camera_to_base'].copy()
            snapshot['camera_to_base'][0, 3] += .014
            moved = roof_edge_motion(edges[0], roof_edges(snapshot, json.loads(str(data['current_target']))))
            self.assertTrue(moved is None or moved['displacement_m'] > .01)

    def test_disconnected_ambiguous_boundaries_cannot_supply_an_edge(self):
        tangent = np.linspace(-.02, .02, 12)
        samples = np.vstack((np.column_stack((np.full(12, -.06), tangent)),
                             np.column_stack((np.full(12, -.04), tangent))))
        self.assertIsNone(supported_edge(samples, 0, .04))

    def test_majority_without_required_span_cannot_replace_a_real_edge(self):
        short = np.column_stack((np.full(20, -.06), np.linspace(-.003, .003, 20)))
        isolated = np.column_stack((np.linspace(-.04, -.02, 6), np.linspace(-.02, .02, 6)))
        self.assertIsNone(supported_edge(np.vstack((short, isolated)), 0, .04))

    def v59_frames(self):
        with np.load(Path(__file__).parent/'fixtures/approach_edges_v59.npz') as data:
            return {name: ({k: data[name+'_'+k] for k in
                           ('rgb', 'depth_m', 'intrinsic', 'camera_to_base')},
                          json.loads(str(data[name+'_target'])))
                    for name in ('reference', 'before', 'current')}

    def test_real_end_portion_recovers_three_edges_without_changing_good_frames(self):
        frames = self.v59_frames()
        reference = roof_edges(*frames['reference'])
        before = roof_edges(*frames['before'])
        current = roof_edges(*frames['current'])
        self.assertEqual(len(reference), 4)
        self.assertEqual(len(before), 3)
        self.assertFalse(any(e.get('extended_tangent_support') for e in before.values()))
        self.assertAlmostEqual(roof_edge_motion(reference, before)['displacement_m'],
                               .000616004531750318)
        self.assertEqual(len(current), 3)
        edge = current['1:-1']
        self.assertTrue(edge['extended_tangent_support'])
        self.assertEqual(edge['points'], 11)
        self.assertGreater(edge['span_m'], .038)
        self.assertLess(edge['normal_spread_m'], .003)
        motion = roof_edge_motion(reference, current)
        self.assertEqual(len(motion['matched_edges']), 3)
        self.assertLess(motion['displacement_m'], .002)

    def test_real_failure_with_physical_translation_never_claims_stationary_edges(self):
        frames = self.v59_frames()
        reference = roof_edges(*frames['reference'])
        for axis in (0, 1):
            for shift in (-.014, .014):
                with self.subTest(axis=axis, shift=shift):
                    snapshot, target = frames['current']
                    snapshot = dict(snapshot)
                    snapshot['camera_to_base'] = snapshot['camera_to_base'].copy()
                    snapshot['camera_to_base'][axis, 3] += shift
                    motion = roof_edge_motion(reference, roof_edges(snapshot, target))
                    self.assertTrue(motion is None or motion['displacement_m'] > .01)

    def test_end_support_cannot_be_inferred_when_depth_points_are_absent(self):
        frames = self.v59_frames()
        reference = roof_edges(*frames['reference'])
        snapshot, target = frames['current']
        d = snapshot['depth_m'].copy()
        v, u = np.indices(d.shape)
        intrinsic = snapshot['intrinsic']
        camera = np.stack(((u-intrinsic[0, 2])*d/intrinsic[0, 0],
                           (v-intrinsic[1, 2])*d/intrinsic[1, 1], d), axis=-1)
        matrix = snapshot['camera_to_base']
        points = camera @ matrix[:3, :3].T+matrix[:3, 3]
        major = np.array(target['major_axis_xy']); major /= np.linalg.norm(major)
        local = (points[:, :, :2]-np.array(target['position_base_m'])[:2]) @ np.column_stack(
            (major, [-major[1], major[0]]))
        size = np.array(target['oriented_extent_m'])
        d[(local[:, :, 1] < 0) & (abs(local[:, :, 0]) >= .35*size[0])] = 0
        snapshot = dict(snapshot, depth_m=d)
        self.assertIsNone(roof_edge_motion(reference, roof_edges(snapshot, target)))
