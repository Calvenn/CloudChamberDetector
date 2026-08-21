"""Integration checks for ROI-first enhancement and detection."""

import unittest

import numpy as np

from app import _process_pipeline_image
from cloud_chamber.config import load_config


class TestRoiFirstPipeline(unittest.TestCase):
    def test_enhancement_receives_scaled_roi_not_complete_source(self):
        image = np.zeros((992, 1312, 3), dtype=np.uint8)
        image[500:900, 850:1250] = 180
        original = image.copy()
        result = _process_pipeline_image(
            image,
            load_config(),
            {
                "roi_coordinates": {
                    "x": 850,
                    "y": 500,
                    "width": 400,
                    "height": 400,
                }
            },
        )

        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(result["selected_roi_image"].shape, (400, 400, 3))
        self.assertEqual(result["input_image"].shape, (640, 640, 3))
        self.assertEqual(result["enhancement"].grey.shape, (640, 640))
        self.assertEqual(result["segmentation"].binary_mask.shape, (640, 640))
        self.assertEqual(result["spatial_scaling"]["roi_x"], 850)
        self.assertEqual(result["spatial_scaling"]["roi_y"], 500)
        self.assertAlmostEqual(result["spatial_scaling"]["scale_x"], 1.6)
        self.assertAlmostEqual(result["spatial_scaling"]["scale_y"], 1.6)

    def test_applied_processing_resolution_can_be_changed(self):
        image = np.zeros((900, 1200, 3), dtype=np.uint8)
        result = _process_pipeline_image(
            image,
            load_config(),
            {
                "roi_coordinates": {
                    "x": 750,
                    "y": 300,
                    "width": 400,
                    "height": 600,
                },
                "processing_size": (320, 480),
            },
        )
        self.assertEqual(result["input_image"].shape, (480, 320, 3))
        self.assertEqual(result["spatial_scaling"]["roi_x"], 750)
        self.assertEqual(result["spatial_scaling"]["roi_y"], 300)
        self.assertLessEqual(
            result["spatial_scaling"]["roi_x"]
            + result["spatial_scaling"]["roi_width"],
            image.shape[1],
        )
        self.assertLessEqual(
            result["spatial_scaling"]["roi_y"]
            + result["spatial_scaling"]["roi_height"],
            image.shape[0],
        )

    def test_rectification_occurs_between_roi_extraction_and_scaling(self):
        image = np.zeros((800, 1000, 3), dtype=np.uint8)
        image[200:600, 300:700] = 120
        original = image.copy()
        result = _process_pipeline_image(
            image,
            load_config(),
            {
                "roi_coordinates": {
                    "x": 300,
                    "y": 200,
                    "width": 400,
                    "height": 400,
                },
                "processing_size": (640, 640),
                "rectification_points": [
                    (20, 10),
                    (380, 20),
                    (370, 390),
                    (10, 380),
                ],
            },
        )
        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(result["selected_roi_image"].shape, (400, 400, 3))
        self.assertEqual(result["rectified_roi_image"].shape, (400, 400, 3))
        self.assertEqual(result["input_image"].shape, (640, 640, 3))
        self.assertTrue(result["rectified"])
        self.assertTrue(result["rectification"]["applied"])
        self.assertEqual(
            np.asarray(
                result["rectification"]["source_to_rectified_homography"]
            ).shape,
            (3, 3),
        )


if __name__ == "__main__":
    unittest.main()
