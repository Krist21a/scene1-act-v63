import json
from pathlib import Path
import unittest

import numpy as np

from scene1.rgbd import metric_candidates, spatial_groups
from scene1.residual import ResidualAssist
from scene1.settlement import resting_observation


class SamplingTests(unittest.TestCase):
    def test_dense_settlement_recovers_real_partially_occluded_parcel(self):
        fixture=Path(__file__).parent/'fixtures/settlement_partial_rgbd.npz'
        with np.load(fixture,allow_pickle=False) as data:
            config=json.loads(str(data['config_json']))
            args=[data[key] for key in ('rgb','depth_m','intrinsic','camera_to_base')]+[config]
            original,plane=metric_candidates(*args)
            dense,dense_plane=metric_candidates(*args,stride=2)
        near=lambda candidates:[c for c in candidates if np.linalg.norm(
            np.asarray(c['position_base_m'])[:2]-[.428,-.567]) < .12]
        self.assertEqual(near(original),[])
        recovered=near(dense)
        self.assertEqual(len(recovered),1)
        self.assertGreaterEqual(recovered[0]['points'],100)
        np.testing.assert_allclose(recovered[0]['oriented_extent_m'],[.117,.041],atol=.005)
        self.assertLess(abs(dense_plane-plane),.003)

    def merge_fixture(self, tag='merged'):
        with np.load(Path(__file__).parent/'fixtures/settlement_robot_merge_rgbd.npz') as data:
            snapshot = {key: data[tag+'_'+key] for key in ('rgb','depth_m','intrinsic','camera_to_base')}
            config = json.loads(str(data['config_json']))
            surface = json.loads(str(data['surface_json']))
            extent = data['extent'].copy()
        snapshot['stamp'] = 1.
        return snapshot, config, surface, extent

    def test_real_failure_uses_separate_3d_component_and_original_support_checks(self):
        snapshot, config, surface, extent = self.merge_fixture()
        args = [snapshot[key] for key in ('rgb','depth_m','intrinsic','camera_to_base')]+[config]
        original_depth = snapshot['depth_m'].copy()
        snapshot['candidates'], snapshot['table_z_base_m'] = metric_candidates(*args, stride=2)
        self.assertIsNone(resting_observation(snapshot, surface, extent, ResidualAssist._roof_normal))
        snapshot['candidates'], snapshot['table_z_base_m'] = metric_candidates(
            *args, stride=2, spatial_components=True)
        observed = resting_observation(snapshot, surface, extent, ResidualAssist._roof_normal)
        self.assertIsNotNone(observed)
        self.assertTrue(observed['support_height_ok'])
        self.assertGreater(observed['footprint_margin_m'], .015)
        self.assertLess(observed['tilt_degrees'], 4.)
        np.testing.assert_allclose(observed['target']['oriented_extent_m'], [.1146, .0433], atol=.001)
        np.testing.assert_array_equal(snapshot['depth_m'], original_depth)
        # A repaired component is still rejected as unsupported by the
        # untouched settlement monitor when the support really is too small.
        from scene1.settlement import SettlementMonitor
        observed['footprint_margin_m'] = .009
        self.assertEqual(SettlementMonitor().update(observed)['reason'], 'unsupported_or_outside_pad')

    def test_existing_clear_roof_retains_exact_candidate_geometry(self):
        snapshot, config, surface, extent = self.merge_fixture('clear')
        args = [snapshot[key] for key in ('rgb','depth_m','intrinsic','camera_to_base')]+[config]
        observed = []
        for enabled in (False, True):
            snapshot['candidates'], snapshot['table_z_base_m'] = metric_candidates(
                *args, stride=2, spatial_components=enabled)
            observed.append(resting_observation(snapshot, surface, extent, ResidualAssist._roof_normal))
        self.assertIsNotNone(observed[0])
        self.assertEqual(observed[0], observed[1])

    def test_spatial_connectivity_preserves_real_contact_and_separates_depth_gap(self):
        x, y = np.mgrid[0:.111:.01, 0:.041:.01]
        roof = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        raised = roof+np.array([0.,0.,.03])
        groups = spatial_groups(np.vstack((roof, raised)))
        self.assertEqual(sorted(map(len, groups)), [len(roof), len(raised)])
        # Same XY is not enough to merge different depths. Conversely, a
        # physically connected observed bridge must not be cut to force a fit.
        bridge = np.array([[0.,0.,.01],[0.,0.,.02]])
        self.assertEqual(len(spatial_groups(np.vstack((roof, raised, bridge)))), 1)

    def test_missing_raw_parcel_is_not_reconstructed_from_expected_size(self):
        snapshot, config, surface, extent = self.merge_fixture()
        depth = snapshot['depth_m'].copy()
        args = [snapshot['rgb'],depth,snapshot['intrinsic'],snapshot['camera_to_base'],config]
        snapshot['candidates'], snapshot['table_z_base_m'] = metric_candidates(
            *args, stride=2, spatial_components=True)
        observed = resting_observation(snapshot, surface, extent, ResidualAssist._roof_normal)
        x, y, w, h = observed['target']['bbox']
        depth[max(0,y-2):y+h+2, max(0,x-2):x+w+2] = 0.
        snapshot['depth_m'] = depth
        snapshot['candidates'], snapshot['table_z_base_m'] = metric_candidates(
            *args, stride=2, spatial_components=True)
        self.assertIsNone(resting_observation(snapshot, surface, extent, ResidualAssist._roof_normal))
        with self.assertRaisesRegex(ValueError, 'dense 2-pixel'):
            metric_candidates(*args, stride=4, spatial_components=True)
