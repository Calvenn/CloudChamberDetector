"""Presentation helpers for contour-feature tables."""

from __future__ import annotations

from cloud_chamber.feature_extraction.contour_features import TrackFeatures


def feature_row(
    feature: TrackFeatures,
    centimetres_per_pixel: float | None = None,
) -> dict[str, int | float]:
    """Return the complete, consistently formatted row used by model pages."""
    row: dict[str, int | float] = {
        "Track": feature.track_id,
        "Area (px²)": round(feature.area_pixels, 3),
        "Perimeter (px)": round(feature.perimeter_pixels, 3),
        "Length (px)": round(feature.major_axis_pixels, 3),
        "Width (px)": round(feature.mean_width_pixels, 3),
        "Aspect ratio": round(feature.aspect_ratio, 3),
        "Solidity": round(feature.solidity, 3),
        "Rectangularity": round(feature.rectangularity, 3),
        "Thickness (px)": round(feature.thickness_pixels, 3),
        "Orientation (°)": round(feature.orientation_degrees, 3),
        "Mean intensity": round(feature.mean_intensity, 3),
    }
    if centimetres_per_pixel is None:
        return row

    scale = float(centimetres_per_pixel)
    row.update(
        {
            "Area (cm²)": round(feature.area_pixels * scale**2, 6),
            "Perimeter (cm)": round(feature.perimeter_pixels * scale, 6),
            "Length (cm)": round(feature.major_axis_pixels * scale, 6),
            "Width (cm)": round(feature.mean_width_pixels * scale, 6),
            "Thickness (cm)": round(feature.thickness_pixels * scale, 6),
        }
    )
    return row


def basic_feature_row(feature: TrackFeatures) -> dict[str, str | float]:
    """Return the most interpretable measurements for the primary table."""
    return {
        "Track": f"T{feature.track_id}",
        "Area (px²)": round(feature.area_pixels, 2),
        "Length (px)": round(feature.major_axis_pixels, 2),
        "Mean width (px)": round(feature.mean_width_pixels, 2),
        "Aspect ratio": round(feature.aspect_ratio, 2),
        "Orientation (°)": round(feature.orientation_degrees, 2),
        "Mean intensity": round(feature.mean_intensity, 2),
    }


def advanced_feature_row(feature: TrackFeatures) -> dict[str, str | float]:
    """Return secondary shape, intensity, and orientation descriptors."""
    return {
        "Track": f"T{feature.track_id}",
        "Perimeter (px)": round(feature.perimeter_pixels, 2),
        "Thickness (px)": round(feature.thickness_pixels, 2),
        "Solidity": round(feature.solidity, 3),
        "Rectangularity": round(feature.rectangularity, 3),
        "Intensity SD": round(feature.intensity_stddev, 2),
        "Circularity": round(feature.circularity, 3),
        "Convexity": round(feature.convexity, 3),
        "Perimeter / length": round(feature.perimeter_to_major_axis, 3),
        "Orientation sine (2 × angle)": round(feature.orientation_sin_2x, 3),
        "Orientation cosine (2 × angle)": round(feature.orientation_cos_2x, 3),
    }
