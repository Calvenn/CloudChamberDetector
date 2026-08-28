"""Automatic overlapping tiling and coordinate reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TileMetadata:
    """Geometry required to reconstruct one processed tile."""

    tile_id: int
    original_image_width: int
    original_image_height: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    original_tile_width: int
    original_tile_height: int
    padded_tile_width: int
    padded_tile_height: int
    processing_width: int
    processing_height: int
    scale_x: float
    scale_y: float
    overlap_ratio: float
    padding_right: int
    padding_bottom: int

    def to_dict(self) -> dict[str, int | float | dict[str, int]]:
        return {
            "tile_id": self.tile_id,
            "original_image_width": self.original_image_width,
            "original_image_height": self.original_image_height,
            "x_start": self.x_start,
            "y_start": self.y_start,
            "x_end": self.x_end,
            "y_end": self.y_end,
            "original_tile_width": self.original_tile_width,
            "original_tile_height": self.original_tile_height,
            "processing_width": self.processing_width,
            "processing_height": self.processing_height,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "overlap_ratio": self.overlap_ratio,
            "padding": {"right": self.padding_right, "bottom": self.padding_bottom},
        }


@dataclass(frozen=True)
class ImageTile:
    """Processing-sized tile paired with coordinates for full-image merging."""

    image: np.ndarray
    metadata: TileMetadata


def axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Return starts that cover an axis completely, including its final pixel."""
    if min(length, tile_size, stride) < 1:
        raise ValueError("Image length, tile size and stride must be positive")
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if positions[-1] != final_start:
        positions.append(final_start)
    return positions


def generate_overlapping_tiles(
    image: np.ndarray,
    tile_size: int,
    processing_size: tuple[int, int],
    overlap_ratio: float,
) -> list[ImageTile]:
    """Extract padded square copies that cover every original-image pixel."""
    if image is None or image.size == 0:
        raise ValueError("Tiling input image is empty")
    if tile_size < 2:
        raise ValueError("Tile size must exceed one pixel")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("Overlap ratio must be in the range [0, 1)")
    target_width, target_height = (int(value) for value in processing_size)
    if target_width < 2 or target_height < 2 or target_width != target_height:
        raise ValueError("Processing resolution must be a square larger than one pixel")

    overlap_pixels = int(round(tile_size * overlap_ratio))
    stride = tile_size - overlap_pixels
    if stride < 1:
        raise ValueError("Overlap leaves no positive tile stride")
    height, width = image.shape[:2]
    x_positions = axis_positions(width, tile_size, stride)
    y_positions = axis_positions(height, tile_size, stride)
    tiles: list[ImageTile] = []
    tile_id = 1
    for y_start in y_positions:
        for x_start in x_positions:
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)
            valid = image[y_start:y_end, x_start:x_end].copy()
            valid_height, valid_width = valid.shape[:2]
            padding_right = tile_size - valid_width
            padding_bottom = tile_size - valid_height
            if padding_right or padding_bottom:
                valid = cv2.copyMakeBorder(
                    valid,
                    0,
                    padding_bottom,
                    0,
                    padding_right,
                    cv2.BORDER_REPLICATE,
                )
            scale_x = target_width / float(tile_size)
            scale_y = target_height / float(tile_size)
            metadata = TileMetadata(
                tile_id=tile_id,
                original_image_width=width,
                original_image_height=height,
                x_start=x_start,
                y_start=y_start,
                x_end=x_end,
                y_end=y_end,
                original_tile_width=valid_width,
                original_tile_height=valid_height,
                padded_tile_width=tile_size,
                padded_tile_height=tile_size,
                processing_width=target_width,
                processing_height=target_height,
                scale_x=scale_x,
                scale_y=scale_y,
                overlap_ratio=float(overlap_ratio),
                padding_right=padding_right,
                padding_bottom=padding_bottom,
            )
            tiles.append(ImageTile(valid, metadata))
            tile_id += 1
    return tiles


def spatial_scale_tile(
    tile: np.ndarray, target_size: tuple[int, int]
) -> tuple[np.ndarray, float, float]:
    """Resize a square processing copy without changing its source image."""
    if tile is None or tile.size == 0:
        raise ValueError("Spatial-scaling tile is empty")
    height, width = tile.shape[:2]
    target_width, target_height = (int(value) for value in target_size)
    if width != height or target_width != target_height:
        raise ValueError("Tile and processing resolution must both be square")
    scale_x = target_width / float(width)
    scale_y = target_height / float(height)
    if (width, height) == (target_width, target_height):
        return tile.copy(), 1.0, 1.0
    interpolation = cv2.INTER_AREA if target_width < width else cv2.INTER_LINEAR
    return (
        cv2.resize(tile, (target_width, target_height), interpolation=interpolation),
        scale_x,
        scale_y,
    )


def coverage_map(
    image_shape: tuple[int, ...], tiles: list[ImageTile]
) -> np.ndarray:
    """Count how many generated tiles cover each valid source pixel."""
    height, width = image_shape[:2]
    coverage = np.zeros((height, width), dtype=np.uint16)
    for tile in tiles:
        item = tile.metadata
        coverage[item.y_start:item.y_end, item.x_start:item.x_end] += 1
    return coverage


def map_processed_mask_to_image(
    processed_mask: np.ndarray, metadata: TileMetadata
) -> np.ndarray:
    """Return a valid, unpadded tile mask at original tile resolution."""
    if processed_mask.shape[:2] != (
        metadata.processing_height,
        metadata.processing_width,
    ):
        raise ValueError("Processed mask does not match tile processing resolution")
    valid_processed_width = max(
        1, int(round(metadata.original_tile_width * metadata.scale_x))
    )
    valid_processed_height = max(
        1, int(round(metadata.original_tile_height * metadata.scale_y))
    )
    valid = processed_mask[:valid_processed_height, :valid_processed_width]
    return cv2.resize(
        valid,
        (metadata.original_tile_width, metadata.original_tile_height),
        interpolation=cv2.INTER_NEAREST,
    )


def merge_tile_masks(
    image_shape: tuple[int, ...],
    masks: list[np.ndarray],
    tiles: list[ImageTile],
    minimum_overlap_agreement: float = 0.0,
) -> np.ndarray:
    """Merge masks in image space, optionally requiring overlap consensus.

    Pixels covered by only one tile remain eligible. In overlap regions a
    positive agreement ratio rejects detections that appear in only one view.
    """
    if len(masks) != len(tiles):
        raise ValueError("Every tile must have exactly one processed mask")
    if not 0.0 <= minimum_overlap_agreement <= 1.0:
        raise ValueError("minimum_overlap_agreement must be in [0, 1]")
    votes = np.zeros(image_shape[:2], dtype=np.uint16)
    coverage = np.zeros(image_shape[:2], dtype=np.uint16)
    for processed_mask, tile in zip(masks, tiles, strict=True):
        item = tile.metadata
        mapped = map_processed_mask_to_image(processed_mask, item)
        region = np.s_[item.y_start:item.y_end, item.x_start:item.x_end]
        votes[region] += (mapped > 0).astype(np.uint16)
        coverage[region] += 1
    if minimum_overlap_agreement <= 0:
        accepted = votes > 0
    else:
        required = np.where(
            coverage <= 1,
            1,
            np.ceil(coverage.astype(np.float32) * minimum_overlap_agreement),
        ).astype(np.uint16)
        accepted = votes >= required
    return np.where(accepted, 255, 0).astype(np.uint8)


def draw_tile_boundaries(image: np.ndarray, tiles: list[ImageTile]) -> np.ndarray:
    """Draw a debug copy; never modify the supplied source image."""
    preview = image.copy()
    for tile in tiles:
        item = tile.metadata
        cv2.rectangle(
            preview,
            (item.x_start, item.y_start),
            (item.x_end - 1, item.y_end - 1),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            str(item.tile_id),
            (item.x_start + 6, item.y_start + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return preview
