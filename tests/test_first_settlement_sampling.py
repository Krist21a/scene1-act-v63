import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from run_scene1 import read_manipulation_rgbd
from scene1.rgbd import metric_candidates
from scene1.residual import ResidualAssist
from scene1.settlement import resting_observation, SettlementMonitor


class FirstSettlementSamplingTests(unittest.TestCase):
    def fixture(self, tag):
        with np.load(Path(__file__).parent/'fixtures/settlement_first_merge_v57.npz') as data:
            snapshot = {key: data[tag+'_'+key].copy()
                        for key in ('rgb', 'depth_m', 'intrinsic', 'camera_to_base')}
            config = dict(cartesian_alignment=json.loads(str(data['config_json'])),
                          visual_cycle=dict(settlement_spatial_components=True))
            pending = dict(kind='settle', surface=json.loads(str(data['surface_json'])),
                           extent=data['extent'].copy())
        snapshot['candidates'], snapshot['table_z_base_m'] = metric_candidates(
            snapshot['rgb'], snapshot['depth_m'], snapshot['intrinsic'],
            snapshot['camera_to_base'], config['cartesian_alignment'], stride=2)
        snapshot.update(stamp=1., candidate_spatial_components=False)
        aligner = SimpleNamespace(pending=pending, _roof_normal=ResidualAssist._roof_normal)
        return snapshot, config, aligner

    def observe(self, snapshot, aligner):
        p = aligner.pending
        return resting_observation(snapshot, p['surface'], p['extent'], aligner._roof_normal)

    def test_first_parcel_missing_geometry_uses_real_separated_points(self):
        for tag in ('merged', 'failed'):
            snapshot, config, aligner = self.fixture(tag)
            self.assertIsNone(self.observe(snapshot, aligner))
            source_candidates = copy.deepcopy(snapshot['candidates'])
            scene = SimpleNamespace(read=Mock(return_value=snapshot))
            recovered = read_manipulation_rgbd(scene, aligner, object(), None, config)
            scene.read.assert_called_once_with(candidate_stride=2, spatial_components=False)
            self.assertTrue(recovered['candidate_spatial_components'])
            observed = self.observe(recovered, aligner)
            self.assertIsNotNone(observed)
            np.testing.assert_allclose(observed['target']['oriented_extent_m'], [.1124, .0428], atol=.001)
            self.assertTrue(observed['support_height_ok'])
            self.assertGreater(observed['footprint_margin_m'], .015)
            self.assertLess(observed['tilt_degrees'], 3.)
            self.assertEqual(snapshot['candidates'], source_candidates)
            np.testing.assert_array_equal(recovered['depth_m'], snapshot['depth_m'])
            self.assertFalse(SettlementMonitor().update(observed)['confirmed'])

    def test_existing_valid_first_parcel_observation_is_returned_unchanged(self):
        snapshot, config, aligner = self.fixture('valid')
        old = self.observe(snapshot, aligner)
        self.assertIsNotNone(old)
        recovered = read_manipulation_rgbd(SimpleNamespace(read=lambda **kw: snapshot),
            aligner, object(), None, config)
        self.assertIs(recovered, snapshot)
        self.assertEqual(self.observe(recovered, aligner), old)

    def test_existing_unsupported_observation_does_not_trigger_geometry_fallback(self):
        snapshot, config, aligner = self.fixture('valid')
        aligner.pending['extent'][2] += .04
        observed = self.observe(snapshot, aligner)
        self.assertFalse(observed['support_height_ok'])
        recovered = read_manipulation_rgbd(SimpleNamespace(read=lambda **kw: snapshot),
            aligner, object(), None, config)
        self.assertIs(recovered, snapshot)
        self.assertEqual(SettlementMonitor().update(self.observe(recovered, aligner))['reason'],
                         'unsupported_or_outside_pad')

    def test_sampling_scope_keeps_initial_grasp_retreat_and_later_parcel_behavior(self):
        for kind, cycle, pickup, enabled, stride, spatial in (
            ('grasp', object(), None, True, 4, False),
            ('short_retreat', object(), None, True, 2, False),
            ('settle', None, None, True, 4, False),
            ('settle', object(), None, False, 2, False),
            ('settle', object(), SimpleNamespace(active=True), True, 2, True)):
            snapshot, config, aligner = self.fixture('merged')
            aligner.pending['kind'] = kind
            config['visual_cycle']['settlement_spatial_components'] = enabled
            scene = SimpleNamespace(read=Mock(return_value=snapshot))
            self.assertIs(read_manipulation_rgbd(scene, aligner, cycle, pickup, config), snapshot)
            scene.read.assert_called_once_with(candidate_stride=stride, spatial_components=spatial)
        scene = SimpleNamespace(read=Mock(return_value=None))
        self.assertIsNone(read_manipulation_rgbd(scene, object(), None, None, {}))
        scene.read.assert_called_once_with(candidate_stride=4, spatial_components=False)

    def test_missing_parcel_pixels_cannot_be_reconstructed(self):
        snapshot, config, aligner = self.fixture('failed')
        snapshot['depth_m'][214:258, 408:446] = 0.
        recovered = read_manipulation_rgbd(SimpleNamespace(read=lambda **kw: snapshot),
            aligner, object(), None, config)
        self.assertIsNone(self.observe(recovered, aligner))
        self.assertEqual(SettlementMonitor().update(None)['reason'], 'missing_geometry')
