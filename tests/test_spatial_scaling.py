"""Validation tests for dynamic ROI selection and spatial scaling."""

import unittest

import numpy as np

from cloud_chamber.calibration import (
    RoiCoordinates,
    constrain_roi,
    default_roi_coordinates,
    extract_roi,
    map_processed_point_to_original,
    rectify_image_with_transform,
    select_and_scale_roi,
    spatial_scale_roi,
)
from cloud_chamber.segmentation import scale_pixel_parameters


class TestRoiSpatialScaling(unittest.TestCase):
    def test_landscape_uses_square_roi_without_changing_source(self):
        image = np.arange(992 * 1312 * 3, dtype=np.uint8).reshape(992, 1312, 3)
        original = image.copy()
        result = select_and_scale_roi(image, target_size=(640, 640))

        self.assertEqual(result.roi_image.shape, (992, 992, 3))
        self.assertEqual(result.image.shape, (640, 640, 3))
        self.assertEqual(result.coordinates, RoiCoordinates(160, 0, 992, 992))
        self.assertTrue(np.array_equal(image, original))

    def test_portrait_uses_square_roi_and_preserves_source(self):
        image = np.full((1920, 1080, 3), 127, dtype=np.uint8)
        original = image.copy()
        result = select_and_scale_roi(image, target_size=(640, 640))

        self.assertEqual(result.coordinates, RoiCoordinates(0, 420, 1080, 1080))
        self.assertEqual(result.image.shape, (640, 640, 3))
        self.assertTrue(np.array_equal(image, original))

    def test_roi_moves_to_boundary_without_exceeding_image(self):
        image = np.zeros((600, 900, 3), dtype=np.uint8)
        constrained = constrain_roi(
            image,
            {"x": 850, "y": 550, "width": 400, "height": 400},
            (640, 640),
        )
        self.assertEqual(constrained, RoiCoordinates(500, 200, 400, 400))

    def test_cropper_left_top_coordinate_names_are_supported(self):
        image = np.zeros((800, 1000, 3), dtype=np.uint8)
        constrained = constrain_roi(
            image,
            {"left": 125, "top": 75, "width": 500, "height": 500},
            (640, 640),
        )
        self.assertEqual(constrained, RoiCoordinates(125, 75, 500, 500))

    def test_roi_resize_retains_configured_aspect_ratio(self):
        image = np.zeros((1000, 1600, 3), dtype=np.uint8)
        constrained = constrain_roi(
            image,
            {"x": 10, "y": 20, "width": 900, "height": 500},
            (640, 640),
        )
        self.assertEqual((constrained.width, constrained.height), (500, 500))

    def test_different_roi_sizes_reach_same_processing_size(self):
        for side in (400, 800, 1000):
            image = np.zeros((1200, 1200, 3), dtype=np.uint8)
            result = select_and_scale_roi(
                image,
                RoiCoordinates(0, 0, side, side),
                (640, 640),
            )
            self.assertEqual(result.image.shape, (640, 640, 3))
            self.assertAlmostEqual(result.scale_x, result.scale_y)

    def test_upscale_and_downscale_do_not_stretch(self):
        small = np.zeros((400, 400, 3), dtype=np.uint8)
        large = np.zeros((800, 800, 3), dtype=np.uint8)
        for roi in (small, large):
            scaled, scale_x, scale_y = spatial_scale_roi(roi, (640, 640))
            self.assertEqual(scaled.shape, (640, 640, 3))
            self.assertEqual(scale_x, scale_y)

    def test_equal_size_returns_copy_without_resampling(self):
        roi = np.arange(640 * 640 * 3, dtype=np.uint8).reshape(640, 640, 3)
        scaled, scale_x, scale_y = spatial_scale_roi(roi, (640, 640))
        self.assertTrue(np.array_equal(scaled, roi))
        self.assertFalse(np.shares_memory(scaled, roi))
        self.assertEqual((scale_x, scale_y), (1.0, 1.0))

    def test_metadata_records_resolution_scale_and_operation(self):
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        result = select_and_scale_roi(
            image, RoiCoordinates(0, 0, 500, 500), (800, 800)
        )
        metadata = result.metadata()
        self.assertEqual(metadata["processing_resolution"], "800 × 800")
        self.assertEqual(metadata["scale_factor"], 1.6)
        self.assertEqual(metadata["scaling_operation"], "upscale")

    def test_non_matching_aspect_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            spatial_scale_roi(np.zeros((400, 800, 3), dtype=np.uint8), (640, 640))

    def test_extract_roi_returns_only_requested_pixels(self):
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        image[300:500, 300:500] = 255
        selected, coordinates = extract_roi(
            image, RoiCoordinates(300, 300, 200, 200), (640, 640)
        )
        self.assertEqual(coordinates, RoiCoordinates(300, 300, 200, 200))
        self.assertTrue(np.all(selected == 255))

    def test_processed_coordinate_maps_back_to_original(self):
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        result = select_and_scale_roi(
            image, RoiCoordinates(100, 200, 800, 800), (640, 640)
        )
        original = map_processed_point_to_original((320, 160), result)
        self.assertAlmostEqual(original[0], 500.0)
        self.assertAlmostEqual(original[1], 400.0)

    def test_default_roi_supports_other_resolution(self):
        image = np.zeros((777, 1234, 3), dtype=np.uint8)
        roi = default_roi_coordinates(image, (640, 640))
        self.assertEqual((roi.width, roi.height), (777, 777))
        self.assertGreaterEqual(roi.x, 0)
        self.assertGreaterEqual(roi.y, 0)

    def test_pixel_parameters_keep_relative_meaning_at_640(self):
        scaled = scale_pixel_parameters(
            {
                "top_hat_kernel": 61,
                "closing_kernel": 5,
                "opening_kernel": 3,
                "minimum_object_area": 343,
                "minimum_major_axis": 51,
            },
            reference_size=1920,
            processing_size=(640, 640),
        )
        self.assertEqual(scaled["top_hat_kernel"], 21)
        self.assertEqual(scaled["closing_kernel"], 3)
        self.assertEqual(scaled["minimum_object_area"], 38)
        self.assertEqual(scaled["minimum_major_axis"], 17)

    def test_rectification_can_preserve_roi_dimensions(self):
        roi = np.zeros((400, 400, 3), dtype=np.uint8)
        roi[100:300, 100:300] = 255
        points = [(20, 10), (380, 20), (370, 390), (10, 380)]
        rectified, transform = rectify_image_with_transform(
            roi, points, output_size=(400, 400)
        )
        self.assertEqual(rectified.shape, (400, 400, 3))
        self.assertEqual(transform.shape, (3, 3))
        self.assertTrue(np.isfinite(transform).all())


if __name__ == "__main__":
    unittest.main()
