"""Unannotated v55 failure: background in the association margin is not material."""
from pathlib import Path
import unittest

import cv2
import numpy as np

from scene1.receiver import receiver_section


class ReceiverSeedTests(unittest.TestCase):
    def frame(self, step=8873):
        with np.load(Path(__file__).parent/'fixtures/receiver_seed_background_v55.npz') as data:
            return tuple(data[f'{key}_{step}'].copy() for key in ('rgb', 'hull', 'closing'))

    def test_raw_failed_frames_recover_current_bilateral_edges(self):
        for step in (8440, 8740, 8860, 8873):
            with self.subTest(step=step):
                section = receiver_section(*self.frame(step))
                self.assertIsNotNone(section)
                self.assertEqual(section['edge_rows'], 5)
                self.assertAlmostEqual(section['center_uv'][0], 421., delta=2.)
                self.assertAlmostEqual(section['center_uv'][1], 380., delta=2.)
                self.assertAlmostEqual(section['width_px'], 99., delta=2.)

    def test_last_valid_frame_retains_its_real_section(self):
        section = receiver_section(*self.frame(8420))
        self.assertEqual(section['center_uv'], [418., 380.])
        self.assertEqual(section['width_px'], 96.)
        self.assertEqual(section['edge_rows'], 5)

    def test_seed_keeps_projection_margin_for_real_parcel_brightness(self):
        # Restricting seeds to the exact hull incorrectly loses this real frame.
        section = receiver_section(*self.frame(8380))
        self.assertEqual(section['center_uv'], [422., 380.])
        self.assertEqual(section['width_px'], 90.)

    def test_either_occluded_edge_still_prevents_a_section(self):
        rgb, hull, closing = self.frame()
        for columns in (slice(350, 395), slice(450, 505)):
            with self.subTest(columns=columns):
                obscured = rgb.copy()
                obscured[180:425, columns] = 0
                self.assertIsNone(receiver_section(obscured, hull, closing))

    def test_bright_seed_is_required_on_each_current_frame(self):
        rgb, hull, closing = self.frame()
        self.assertIsNotNone(receiver_section(rgb, hull, closing))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        rgb[hsv[:, :, 2] >= 190] = 0
        self.assertIsNone(receiver_section(rgb, hull, closing))
        self.assertIsNone(receiver_section(np.zeros_like(rgb), hull, closing))

    def test_material_cannot_replace_projection_association(self):
        rgb, _, closing = self.frame()
        wrong = np.array([[20, 250], [80, 250], [80, 350], [20, 350]])
        self.assertIsNone(receiver_section(rgb, wrong, closing))


if __name__ == '__main__':
    unittest.main()
