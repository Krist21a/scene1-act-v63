import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scene1.placement_height import PlacementHeight
from scene1.rgbd import metric_candidates
from scene1.visual_cycle import VisualCycle
from scene1.visual_pickup import VisualPickup
from test_visual_cycle import Arm


class PlacementHeightTests(unittest.TestCase):
    def assist(self):
        return SimpleNamespace(pending=dict(kind='weighing', stage='correct', steps=87,
            surface=dict(position_base_m=[.443, -.553, -.086],
                footprint_xy_m=[[.34,-.65],[.54,-.65],[.54,-.45],[.34,-.45]])),
            weighing_target=np.array([.443, -.523, -.086]))

    def sample(self, tracker, assist, stamp, height=-.083, **kwargs):
        return tracker.update(assist, dict(stamp=stamp, table_z_base_m=height),
                              view_ready=kwargs.get('view_ready', True),
                              fresh=kwargs.get('fresh', True))

    def test_fresh_stable_height_changes_only_z_and_keeps_original_phase_clock(self):
        assist = self.assist(); original = copy.deepcopy(assist.pending)
        tracker = PlacementHeight()
        self.assertIsNone(self.sample(tracker, assist, 1., -.0831))
        self.assertIsNone(self.sample(tracker, assist, 1.1, -.0829))
        result = self.sample(tracker, assist, 1.2, -.083)
        self.assertAlmostEqual(result['observed_z_m'], -.083)
        self.assertEqual(assist.pending['steps'], 87)
        self.assertEqual(assist.pending['surface']['position_base_m'][:2],
                         original['surface']['position_base_m'][:2])
        self.assertEqual(assist.pending['surface']['footprint_xy_m'],
                         original['surface']['footprint_xy_m'])
        np.testing.assert_array_equal(assist.weighing_target, [.443, -.523, -.083])
        self.assertEqual(original['surface']['position_base_m'][2], -.086)

    def test_old_frames_head_motion_and_gaps_cannot_supply_continuous_evidence(self):
        for condition in ('duplicate', 'old', 'head', 'stale', 'gap'):
            assist = self.assist(); tracker = PlacementHeight()
            self.sample(tracker, assist, 1.)
            self.sample(tracker, assist, 1.1)
            stamp = {'duplicate':1.1, 'old':1., 'gap':2.}.get(condition, 1.2)
            result = self.sample(tracker, assist, stamp,
                view_ready=condition != 'head', fresh=condition != 'stale')
            self.assertIsNone(result)
            self.assertEqual(assist.pending['surface']['position_base_m'][2], -.086)

    def test_nonfinite_outlying_and_unstable_heights_keep_cached_reference(self):
        for values in ((-.083, -.083, float('nan')), (-.083, -.083, -.070),
                       (-.083, -.081, -.083), (-.083, -.083, float('inf'))):
            assist = self.assist(); tracker = PlacementHeight()
            for stamp, height in zip((1.,1.1,1.2), values):
                self.assertIsNone(self.sample(tracker, assist, stamp, height))
            self.assertEqual(assist.pending['surface']['position_base_m'][2], -.086)

    def test_release_retreat_and_regrasp_do_not_change_support_reference(self):
        for kind, stage in (('weighing','release'), ('short_retreat',None),
                            ('settle',None), ('pregrasp',None)):
            assist = self.assist(); tracker = PlacementHeight()
            assist.pending.update(kind=kind, stage=stage)
            original = copy.deepcopy(assist.pending)
            for stamp in (1.,1.1,1.2):
                self.assertIsNone(self.sample(tracker, assist, stamp))
            self.assertEqual(assist.pending, original)

    def test_new_weighing_phase_cannot_inherit_old_height_samples(self):
        assist = self.assist(); tracker = PlacementHeight()
        self.sample(tracker, assist, 1.); self.sample(tracker, assist, 1.1)
        assist.pending = copy.deepcopy(assist.pending)
        self.assertIsNone(self.sample(tracker, assist, 1.2))
        self.assertEqual(assist.pending['surface']['position_base_m'][2], -.086)

    def test_real_rgbd_places_failed_cached_support_below_current_plane(self):
        directory = Path(__file__).parent/'fixtures'
        fixture = json.loads((directory/'placement_height_v58.json').read_text())
        with np.load(directory/'placement_height_v58.npz') as raw:
            for frame in fixture['frames']:
                args = [raw[frame['tag']+'_'+key] for key in
                        ('rgb','depth_m','intrinsic','camera_to_base')]
                _, plane = metric_candidates(*args, fixture['config'])
                shift = plane-frame['surface']['position_base_m'][2]
                if frame['tag'].startswith('failed'):
                    self.assertGreater(shift, .003)
                    self.assertLess(shift, .0035)
                    self.assertGreater(frame['original_gap_m'], .002)
                    self.assertLess(frame['original_gap_m']-shift, .002)
                else:
                    self.assertLess(shift, .002)
                    self.assertGreater(frame['original_gap_m']-shift, .002)

    def test_only_enabled_local_weighing_refreshes_before_original_assist_call(self):
        for stage, enabled in (('local_grasp',True), ('local_weighing',False),
                               ('local_weighing',True)):
            measured = np.zeros(16)
            motion = VisualCycle(dict(weighing_live_support_height=enabled), Arm(),
                                 np.full(16,-3.), np.full(16,3.), measured)
            pickup = VisualPickup(motion, {}, [.443,-.553,-.086], measured, measured)
            pickup._stage(stage)
            assist = self.assist(); pickup.surface = assist.pending['surface']
            assist._remember = lambda snapshot: None
            assist.held_action = measured
            assist.first_grasp_complete = True
            assist.regrasp_complete = False
            self.sample(pickup.placement_height, assist, 1.)
            self.sample(pickup.placement_height, assist, 1.1)
            expected = -.083 if stage == 'local_weighing' and enabled else -.086
            def apply(*args):
                self.assertAlmostEqual(assist.pending['surface']['position_base_m'][2], expected)
                return measured.copy(), dict(state='original_assist')
            assist.apply = apply
            with patch.object(pickup, '_pad_reference_view_ready', return_value=True):
                action, info = pickup.update(measured,
                    dict(stamp=1.2, table_z_base_m=-.083), assist)
            np.testing.assert_array_equal(action, measured)
            self.assertEqual('weighing_support_height' in info,
                             stage == 'local_weighing' and enabled)
            self.assertEqual(assist.pending['steps'], 87)
