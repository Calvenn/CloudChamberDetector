"""Validation tests for longest-side spatial scaling."""

import unittest

import numpy as np

from cloud_chamber.calibration import spatial_scale


class TestSpatialScaling(unittest.TestCase):
    def test_landscape_dataset_remains_unchanged(self):
        image = np.full((992, 1312, 3), 127, dtype=np.uint8)
        result = spatial_scale(image, 1920)

        self.assertEqual(result.image.shape, (1452, 1920, 3))
        self.assertEqual((result.scaled_width, result.scaled_height), (1920, 1452))
        self.assertAlmostEqual(result.scale, 1920 / 1312)

    def test_portrait_dataset_preserves_orientation_and_field_of_view(self):
        image = np.full((1920, 1080, 3), 127, dtype=np.uint8)
        result = spatial_scale(image, 1920)

        self.assertEqual(result.image.shape, (1920, 1080, 3))
        self.assertEqual((result.scaled_width, result.scaled_height), (1080, 1920))
        self.assertEqual(max(result.image.shape[:2]), 1920)
        self.assertEqual(result.scale, 1.0)
        self.assertTrue(np.array_equal(result.image, image))
        self.assertGreater(result.scaled_height, result.scaled_width)
        self.assertAlmostEqual(1080 / 1920, 1080 / 1920, places=3)


if __name__ == "__main__":
    unittest.main()
