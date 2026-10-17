from pathlib import Path
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scene1.receiver import receiver_section, NearFaceProjector
from scene1.wrist import WristProjector


class ReceiverTests(unittest.TestCase):
    def test_raw_first_parcel_edges_just_below_closing_line_are_measured(self):
        # Unannotated exception frame 2058 from 20261015-182108. The right
        # background is dark above the closing line and clear just below it.
        with np.load(Path(__file__).parent/'fixtures/receiver_below_closing_rgb.npz') as data:
            section = receiver_section(data['rgb'], data['projected_hull'], data['closing_uv'])
            self.assertIsNotNone(section)
            self.assertAlmostEqual(section['center_uv'][0], 426.5, delta=1.)
            self.assertAlmostEqual(section['width_px'], 141., delta=2.)
            self.assertGreater(section['center_uv'][1], data['closing_uv'][1])
            self.assertLess(section['center_uv'][1]-data['closing_uv'][1], 5.)
            self.assertEqual(section['edge_rows'], 5)

    def test_downward_section_search_is_bounded(self):
        with np.load(Path(__file__).parent/'fixtures/receiver_below_closing_rgb.npz') as data:
            closing = data['closing_uv'].copy(); closing[1] = 350.
            self.assertIsNone(receiver_section(data['rgb'], data['projected_hull'], closing))

    def test_downward_search_does_not_accept_a_finger_boundary(self):
        with np.load(Path(__file__).parent/'fixtures/receiver_below_closing_rgb.npz') as data:
            for columns in (slice(345, 365), slice(485, 505)):
                rgb = data['rgb'].copy(); rgb[375:425, columns] = 0
                self.assertIsNone(receiver_section(rgb, data['projected_hull'], data['closing_uv']))

    def regrasp_failure(self):
        with np.load(Path(__file__).parent/'fixtures/receiver_regrasp_failure_rgb.npz') as data:
            projector = WristProjector.__new__(WristProjector)
            tool = data['tool_matrix'].copy()
            projector.kinematics = SimpleNamespace(matrix=lambda q, side: tool,
                                                   tool_xyz=data['tool_xyz'].copy())
            projector.calibration = [{k: np.asarray(v) for k, v in item.items()}
                                    for item in json.loads(str(data['calibration']))]
            projector.config = json.loads(str(data['projector_config']))
            return projector, json.loads(str(data['target'])), data['measured'].copy(), data['rgb'].copy()

    def test_regrasp_raw_failure_uses_near_face_without_dropping_roof_checks(self):
        projector, target, q, rgb = self.regrasp_failure()
        original = projector.measure_outline(1, target, q, rgb)
        refined = NearFaceProjector(projector).measure_outline(1, target, q, rgb)
        self.assertAlmostEqual(original['observed_uv'][0], 405., delta=2.)
        self.assertAlmostEqual(refined['observed_uv'][0], 374.5, delta=2.)
        self.assertAlmostEqual(refined['visible_width_px'], 147., delta=3.)
        self.assertEqual(refined['near_face_section']['edge_rows'], 5)
        self.assertEqual(refined['edge_support'], original['edge_support'])
        self.assertEqual(refined['roof_polygon_px'], original['roof_polygon_px'])
        self.assertLessEqual(np.linalg.norm(refined['correction_base_xy']),
                             projector.config['max_correction_m']+1e-12)
        with patch.object(projector, 'measure_outline', return_value=None):
            self.assertIsNone(NearFaceProjector(projector).measure_outline(1, target, q, rgb))

    def test_regrasp_near_face_rejects_either_occluded_side(self):
        projector, target, q, rgb = self.regrasp_failure()
        for columns in (slice(280, 320), slice(435, 470)):
            occluded = rgb.copy(); occluded[195:425, columns] = 0
            self.assertIsNone(NearFaceProjector(projector).measure_outline(1, target, q, occluded))

    def test_regrasp_requires_observed_support_and_preserves_left_method(self):
        projector, target, q, rgb = self.regrasp_failure()
        target.pop('support_z_base_m')
        self.assertIsNone(NearFaceProjector(projector).measure_outline(1, target, q, rgb))
        with patch.object(projector, 'measure_outline', return_value={'original': True}) as method:
            self.assertEqual(NearFaceProjector(projector).measure_outline(0, target, q, rgb),
                             {'original': True})
            method.assert_called_once()

    def test_third_parcel_raw_failure_retains_chromatic_exterior_edges(self):
        # Unannotated exception frame 9262 from the independent v35 run;
        # neither the annotated 9260 JPEG nor a fabricated 9261 frame.
        with np.load(Path(__file__).parent/'fixtures/receiver_third_failure_rgb.npz') as data:
            evidence = receiver_section(data['rgb'], data['projected_hull'], data['closing_uv'])
            self.assertIsNotNone(evidence)
            self.assertAlmostEqual(evidence['center_uv'][0], 424., delta=2.)
            self.assertAlmostEqual(evidence['width_px'], 130., delta=3.)
            self.assertAlmostEqual(evidence['center_uv'][1], 380., delta=2.)
            self.assertEqual(evidence['edge_rows'], 5)

    def test_third_parcel_still_requires_both_unoccluded_sides(self):
        with np.load(Path(__file__).parent/'fixtures/receiver_third_failure_rgb.npz') as data:
            for side, columns in [('left', slice(345, 375)), ('right', slice(475, 505))]:
                with self.subTest(side=side):
                    rgb = data['rgb'].copy()
                    rgb[220:425, columns] = 0
                    self.assertIsNone(receiver_section(rgb, data['projected_hull'], data['closing_uv']))

    def test_third_parcel_color_edges_do_not_replace_projection_association(self):
        with np.load(Path(__file__).parent/'fixtures/receiver_third_failure_rgb.npz') as data:
            self.assertIsNone(receiver_section(data['rgb'],
                np.array([[20, 250], [80, 250], [80, 350], [20, 350]]), data['closing_uv']))

    def test_saved_preinsertion_frame_detects_real_lateral_offset(self):
        with np.load(Path(__file__).parent/'fixtures/receiver_offset_rgb.npz') as data:
            evidence = receiver_section(data['rgb'], data['projected_hull'], data['closing_uv'])
            self.assertIsNotNone(evidence)
            self.assertLess(evidence['centering_error_px'], -20.)
            self.assertGreater(evidence['centering_error_px'], -30.)
            self.assertEqual(evidence['edge_rows'], 5)
            self.assertIsNone(receiver_section(np.zeros_like(data['rgb']),
                                               data['projected_hull'], data['closing_uv']))

    def test_component_outside_predicted_parcel_cannot_center_receiver(self):
        with np.load(Path(__file__).parent/'fixtures/receiver_offset_rgb.npz') as data:
            self.assertIsNone(receiver_section(data['rgb'],
                np.array([[20, 250], [80, 250], [80, 350], [20, 350]]), data['closing_uv']))

    def test_second_parcel_shadow_uses_same_material_and_actual_edges(self):
        with np.load(Path(__file__).parent/'fixtures/receiver_shadow_rgb.npz') as data:
            rgb, hull, closing = data['rgb'], data['projected_hull'], data['closing_uv']
            evidence = receiver_section(rgb, hull, closing)
            self.assertIsNotNone(evidence)
            self.assertAlmostEqual(evidence['center_uv'][0], 413., delta=2.)
            self.assertAlmostEqual(evidence['width_px'], 96., delta=3.)
            self.assertLess(evidence['centering_error_px'], -10.)
            self.assertEqual(evidence['edge_rows'], 5)
            # A finger hiding one side must still prevent an insertion decision.
            occluded = rgb.copy()
            occluded[220:425, 440:480] = 0
            self.assertIsNone(receiver_section(occluded, hull, closing))
            # Without a bright material seed, a dark colored region is insufficient.
            unseeded = rgb.copy()
            unseeded[220:290, 350:470] = 0
            self.assertIsNone(receiver_section(unseeded, hull, closing))
