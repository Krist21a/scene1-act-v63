import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scene1.bin_layout import plan_bin_drop, validate_layout, slot_axis, slot_points
from scene1.visual_cycle import track_bin_rim


class BinLayoutTests(unittest.TestCase):
    def setUp(self):
        def decode(value):
            if isinstance(value, dict):
                if 'array' in value:
                    return np.asarray(value['array'], dtype=value['dtype'])
                if 'rotation_xyzw' in value:
                    return Rotation.from_quat(value['rotation_xyzw'])
                return {k: decode(v) for k, v in value.items()}
            return value
        self.decode = decode
        self.data = decode(json.loads((Path(__file__).parent/'fixtures/bin_layout_geometry.json').read_text()))

    def test_default_matches_independently_completed_first_parcel_waypoint(self):
        goal, info = plan_bin_drop(self.data['rim'], self.data['geometry'], {}, 0)
        np.testing.assert_allclose(goal[:3, 3], self.data['legacy_goal_tcp'], atol=.004)
        self.assertGreater(info['planned_margin_m'], .025)

    def test_four_distinct_observed_bin_slots_have_full_footprint_clearance(self):
        config = dict(max_parcels=4, bin_tool_rpy_degrees=[80., 5., -90.],
                      bin_lateral_offsets_m=[-.105, -.035, .035, .105])
        centers = []
        for i in range(4):
            goal, info = plan_bin_drop(self.data['rim'], self.data['geometry'], config, i)
            self.assertGreaterEqual(info['planned_margin_m'], .025)
            centers.append(info['target_center_base_m'])
        np.testing.assert_allclose(np.linalg.norm(np.diff(centers, axis=0), axis=1), .07, atol=1e-6)
        with self.assertRaisesRegex(RuntimeError, 'does not fit'):
            plan_bin_drop(self.data['rim'], self.data['geometry'], dict(bin_lateral_offsets_m=[.3]), 0)

    def test_multiple_parcels_require_separate_finite_slots(self):
        for config in [dict(max_parcels=4), dict(bin_tool_rpy_degrees=[float('nan'), 0, 0]),
                       dict(bin_depth_offsets_m=[]), dict(bin_depth_offsets_m=[float('nan')]),
                       dict(max_parcels=2, bin_lateral_offsets_m=[0., 0.])]:
            with self.assertRaises(ValueError):
                validate_layout(config)

    def test_slot_evidence_excludes_neighbor_and_shared_boundary_in_rotated_bin(self):
        config = dict(max_parcels=4, bin_lateral_offsets_m=[-.105, -.035, .035, .105])
        for angle in (0., 27., -21.):
            rim = dict(self.data['rim'])
            rotation = Rotation.from_euler('z', angle, degrees=True).as_matrix()[:2, :2]
            rim['corners'] = (rim['corners']@rotation.T).astype(np.float32)
            rim['center'] = rim['center']@rotation.T
            origin = rim['center']+[.025, 0.]
            axis = slot_axis(rim)
            points = np.column_stack((origin+np.array([-.105, -.035, -.070])[:, None]*axis,
                                      np.zeros(3)))
            np.testing.assert_allclose(slot_points(points, rim, config, 1), points[1:2])
            np.testing.assert_array_equal(slot_points(points, rim, {}, 0), points)

    def test_fourth_failure_raw_rgbd_nearer_goal_preserves_rim_and_slot_checks(self):
        snapshot = dict(np.load(Path(__file__).parent/'fixtures/bin_fourth_failure_rgbd.npz'))
        state = self.decode(json.loads(str(snapshot['state_json'])))
        rim = track_bin_rim(snapshot, state['rim'])
        self.assertIsNotNone(rim)
        config = dict(max_parcels=4, bin_tool_rpy_degrees=[80., 5., -90.],
                      bin_lateral_offsets_m=[-.105, -.035, .035, .105])
        changed = dict(config, bin_depth_offsets_m=[0., 0., 0., -.025])
        goal, info = plan_bin_drop(rim, state['geometry'], changed, 3)
        original, _ = plan_bin_drop(rim, state['geometry'], config, 3)
        actual = snapshot['measured_tcp'][:3, 3]
        self.assertGreater(np.linalg.norm(original[:3, 3]-actual), .025)
        self.assertLess(np.linalg.norm(goal[:3, 3]-actual), .010)
        self.assertGreater(info['planned_margin_m'], .025)
        for i in range(3):
            np.testing.assert_array_equal(plan_bin_drop(rim, state['geometry'], changed, i)[0],
                                          plan_bin_drop(rim, state['geometry'], config, i)[0])
        with self.assertRaisesRegex(RuntimeError, 'does not fit'):
            plan_bin_drop(rim, state['geometry'], dict(changed, bin_depth_offsets_m=[0., 0., 0., -.3]), 3)

    def test_depth_change_preserves_lateral_slot_ownership_for_rotated_bins(self):
        config = dict(max_parcels=4, bin_tool_rpy_degrees=[80., 5., -90.],
                      bin_lateral_offsets_m=[-.105, -.035, .035, .105])
        for angle in (0., 27., -21.):
            case = dict(config, bin_tool_rpy_degrees=[80., 5., -90.+angle])
            changed = dict(case, bin_depth_offsets_m=[0., 0., 0., -.025])
            rim = dict(self.data['rim'])
            rotation = Rotation.from_euler('z', angle, degrees=True).as_matrix()[:2, :2]
            rim['corners'] = (rim['corners']@rotation.T).astype(np.float32)
            rim['center'] = rim['center']@rotation.T
            old, _ = plan_bin_drop(rim, self.data['geometry'], case, 3)
            new, info = plan_bin_drop(rim, self.data['geometry'], changed, 3)
            shift = new[:2, 3]-old[:2, 3]
            self.assertAlmostEqual(np.linalg.norm(shift), .025)
            self.assertAlmostEqual(shift@slot_axis(rim), 0.)
            point = np.asarray([info['target_center_base_m']])
            np.testing.assert_array_equal(slot_points(point, rim, changed, 3), point)
            for index in range(3):
                self.assertEqual(len(slot_points(point, rim, changed, index)), 0)
