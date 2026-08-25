"""Unit and integration tests for automatic overlapping tiling."""

import unittest

import numpy as np

from app import _process_tiled_pipeline_image
from cloud_chamber.config import load_config
from cloud_chamber.tiling import (
    axis_positions,
    coverage_map,
    generate_overlapping_tiles,
    merge_tile_masks,
    spatial_scale_tile,
)


class TestAutomaticTiling(unittest.TestCase):
    def test_final_axis_position_reaches_boundary(self):
        positions = axis_positions(1312, 800, 680)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1] + 800, 1312)

    def test_landscape_and_portrait_have_no_coverage_gaps(self):
        for height, width in ((992, 1312), (1920, 1080), (1080, 1920)):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            tiles = generate_overlapping_tiles(image, 800, (640, 640), 0.15)
            self.assertGreaterEqual(int(coverage_map(image.shape, tiles).min()), 1)

    def test_small_image_is_padded_only_in_tile_copy(self):
        image = np.full((300, 500, 3), 77, dtype=np.uint8)
        original = image.copy()
        tile = generate_overlapping_tiles(image, 800, (640, 640), 0.15)[0]
        self.assertEqual(tile.image.shape, (800, 800, 3))
        self.assertEqual(tile.metadata.padding_right, 300)
        self.assertEqual(tile.metadata.padding_bottom, 500)
        self.assertTrue(np.array_equal(image, original))

    def test_scaling_uses_one_fixed_resolution(self):
        scaled, scale_x, scale_y = spatial_scale_tile(
            np.zeros((800, 800, 3), dtype=np.uint8), (640, 640)
        )
        self.assertEqual(scaled.shape, (640, 640, 3))
        self.assertEqual((scale_x, scale_y), (0.8, 0.8))

    def test_overlap_masks_merge_in_image_coordinates(self):
        image = np.zeros((800, 1200, 3), dtype=np.uint8)
        tiles = generate_overlapping_tiles(image, 800, (640, 640), 0.25)
        masks = []
        for _ in tiles:
            mask = np.zeros((640, 640), dtype=np.uint8)
            mask[300:340, 300:340] = 255
            masks.append(mask)
        merged = merge_tile_masks(image.shape, masks, tiles)
        self.assertEqual(merged.shape, image.shape[:2])
        self.assertGreater(int(np.count_nonzero(merged)), 0)

    def test_overlap_consensus_rejects_one_tile_only_detection(self):
        image = np.zeros((800, 1200, 3), dtype=np.uint8)
        tiles = generate_overlapping_tiles(image, 800, (640, 640), 0.25)
        masks = [np.zeros((640, 640), dtype=np.uint8) for _ in tiles]
        # Source x=500 lies in the two-tile overlap. Mark it in only the first
        # processed view; union retains it but consensus must reject it.
        first = tiles[0].metadata
        px = int(round((500 - first.x_start) * first.scale_x))
        py = int(round((400 - first.y_start) * first.scale_y))
        masks[0][py - 2:py + 3, px - 2:px + 3] = 255
        union = merge_tile_masks(image.shape, masks, tiles)
        consensus = merge_tile_masks(
            image.shape, masks, tiles, minimum_overlap_agreement=0.75
        )
        self.assertGreater(int(np.count_nonzero(union)), 0)
        self.assertEqual(int(np.count_nonzero(consensus)), 0)

    def test_pipeline_preserves_source_and_covers_full_image(self):
        image = np.zeros((992, 1312, 3), dtype=np.uint8)
        original = image.copy()
        result = _process_tiled_pipeline_image(image, load_config())
        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(result["segmentation"].binary_mask.shape, image.shape[:2])
        self.assertGreater(result["spatial_scaling"]["tile_count"], 1)
        self.assertGreaterEqual(int(result["coverage_map"].min()), 1)


if __name__ == "__main__":
    unittest.main()
