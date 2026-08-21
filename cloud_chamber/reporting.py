"""Human-readable, traceable reporting for particle classifications.

Classification confidence comes from the trained model. Contour quality is a
separate, explicitly heuristic review aid based on segmentation measurements;
it must never be interpreted as a probability that the particle is correct.
"""

from __future__ import annotations

import csv
import io
import json
import textwrap
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cloud_chamber.features import TrackFeatures
from cloud_chamber.ml.member_models.mlp import DISPLAY_NAMES


def assess_contour_quality(
    track: TrackFeatures,
    enhanced_image: np.ndarray,
    binary_mask: np.ndarray,
    segmentation_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Return an explainable heuristic quality assessment for one contour."""
    image_height, image_width = binary_mask.shape
    x, y, width, height = track.bounding_box
    padding = max(5, int(round(min(image_height, image_width) * 0.01)))
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    local_image = enhanced_image[top:bottom, left:right]
    local_mask = binary_mask[top:bottom, left:right] > 0
    foreground_values = local_image[local_mask]
    background_values = local_image[~local_mask]
    foreground_mean = float(np.mean(foreground_values)) if foreground_values.size else 0.0
    background_median = (
        float(np.median(background_values)) if background_values.size else foreground_mean
    )
    local_contrast = foreground_mean - background_median

    roi = segmentation_parameters.get("roi_pixels", {})
    roi_left = int(roi.get("left", 0))
    roi_top = int(roi.get("top", 0))
    roi_right = int(roi.get("right", image_width))
    roi_bottom = int(roi.get("bottom", image_height))
    boundary_margin = max(3, int(round(min(image_height, image_width) * 0.01)))
    near_boundary = (
        x <= roi_left + boundary_margin
        or y <= roi_top + boundary_margin
        or x + width >= roi_right - boundary_margin
        or y + height >= roi_bottom - boundary_margin
    )

    general_minimum = float(segmentation_parameters.get("minimum_area", 0))
    accepted_as_small_track = track.area_pixels < general_minimum
    very_thin = track.mean_width_pixels <= 5.0 and track.aspect_ratio >= 3.0

    score = 100
    warnings: list[str] = []
    evidence: list[str] = []
    if local_contrast < 5:
        score -= 35
        warnings.append("Very low local contrast; segmentation may be unstable.")
    elif local_contrast < 12:
        score -= 20
        warnings.append("Low local contrast; inspect the contour mask.")
    else:
        evidence.append(f"Local contrast is {local_contrast:.1f} intensity levels.")
    if near_boundary:
        score -= 20
        warnings.append("Touches the analysis boundary and may be incomplete.")
    if track.solidity < 0.35:
        score -= 15
        warnings.append("Irregular/fragment-like contour shape.")
    elif track.solidity >= 0.65:
        evidence.append(f"Contour solidity is {track.solidity:.2f}.")
    if accepted_as_small_track:
        score -= 5
        evidence.append("Accepted through the specialised thin-track rule.")
    else:
        evidence.append("Passed the normal contour size rule.")
    if very_thin:
        evidence.append(
            f"Thin elongated structure: width {track.mean_width_pixels:.1f}px, "
            f"aspect ratio {track.aspect_ratio:.1f}."
        )
    else:
        evidence.append(
            f"Length {track.major_axis_pixels:.1f}px and width "
            f"{track.mean_width_pixels:.1f}px."
        )
    score = int(np.clip(score, 0, 100))
    grade = "High" if score >= 75 else "Moderate" if score >= 50 else "Low"
    return {
        "track_id": track.track_id,
        "score": score,
        "grade": grade,
        "local_contrast": round(local_contrast, 3),
        "near_boundary": near_boundary,
        "very_thin": very_thin,
        "accepted_as_small_track": accepted_as_small_track,
        "warnings": warnings,
        "evidence": evidence,
        "method": "explainable contour-quality heuristic; not model confidence",
    }


def assess_all_contours(
    features: Iterable[TrackFeatures],
    enhanced_image: np.ndarray,
    binary_mask: np.ndarray,
    segmentation_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Assess contours in the same track order used by the classifier."""
    return [
        assess_contour_quality(
            track, enhanced_image, binary_mask, segmentation_parameters
        )
        for track in features
    ]


def reporting_status(
    confidence: float, confidence_threshold: float, quality_score: int
) -> str:
    """Combine two independent signals into a plain-language review status."""
    confident = confidence >= confidence_threshold
    good_contour = quality_score >= 50
    if confident and quality_score >= 75:
        return "Reliable candidate"
    if confident and not good_contour:
        return "Review segmentation"
    if not confident and good_contour:
        return "Ambiguous particle class"
    if not confident and not good_contour:
        return "Manual review required"
    return "Usable with caution"


def build_summary(
    predictions: list[dict[str, Any]],
    quality_assessments: list[dict[str, Any]],
    confidence_threshold: float,
    processing_time_ms: float,
) -> dict[str, Any]:
    """Build an image-level summary from particle and quality results."""
    class_counts = Counter(item["predicted_class"] for item in predictions)
    confident = sum(item["confidence"] >= confidence_threshold for item in predictions)
    quality_scores = [item["score"] for item in quality_assessments]
    mean_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
    overall_grade = (
        "High" if mean_quality >= 75 else "Moderate" if mean_quality >= 50 else "Low"
    )
    dominant = class_counts.most_common(1)[0][0] if class_counts else None
    return {
        "detected_contours": len(predictions),
        "confident_classifications": confident,
        "uncertain_classifications": len(predictions) - confident,
        "class_counts": {
            DISPLAY_NAMES.get(name, name): class_counts.get(name, 0)
            for name in ("alpha", "electron_positron", "proton", "v_track")
        },
        "dominant_prediction": DISPLAY_NAMES.get(dominant, dominant or "None"),
        "mean_contour_quality": round(mean_quality, 2),
        "overall_contour_quality": overall_grade,
        "processing_time_ms": round(float(processing_time_ms), 3),
    }


def encode_json_report(report: dict[str, Any]) -> bytes:
    """Encode a complete reproducibility record as readable UTF-8 JSON."""
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")


def encode_rows_csv(rows: list[dict[str, Any]]) -> bytes:
    """Encode flat report rows as spreadsheet-compatible CSV."""
    if not rows:
        return b""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def make_traceability_metadata(
    input_name: str,
    source_description: str,
    roi_profile: str,
    confidence_threshold: float,
    model_path: str,
    model_classes: Iterable[str],
    segmentation_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Capture enough context to reproduce and audit one report."""
    selected_parameter_names = (
        "otsu_value",
        "threshold_offset",
        "applied_threshold",
        "hysteresis_low_threshold_offset",
        "hysteresis_low_threshold_value",
        "hysteresis_seed_pixels",
        "top_hat_kernel",
        "closing_kernel",
        "directional_closing_length",
        "alignment_merge_gap",
        "minimum_area",
        "minimum_major_axis",
        "minimum_thin_area",
        "minimum_thin_major_axis",
        "minimum_thin_aspect_ratio",
    )
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_name": input_name,
        "source": source_description,
        "roi_profile": roi_profile,
        "model": model_path,
        "model_classes": [str(value) for value in model_classes],
        "class_mapping": {
            name: DISPLAY_NAMES.get(name, name) for name in model_classes
        },
        "confidence_threshold": confidence_threshold,
        "segmentation_parameters": {
            name: segmentation_parameters.get(name)
            for name in selected_parameter_names
            if name in segmentation_parameters
        },
    }


def encode_pdf_report(
    overlay_bgr: np.ndarray,
    report: dict[str, Any],
) -> bytes:
    """Create a dependency-light, multipage PDF summary using Pillow."""
    page_size = (1240, 1754)
    margin = 60
    font = ImageFont.load_default()
    pages: list[Image.Image] = []
    first_page = Image.new("RGB", page_size, "white")
    draw = ImageDraw.Draw(first_page)
    draw.text((margin, margin), "Cloud Chamber Particle Report", fill="black", font=font)
    metadata = report["metadata"]
    summary = report["summary"]
    lines = [
        f"Input: {metadata['input_name']}",
        f"Source: {metadata['source']}",
        f"Generated: {metadata['generated_at']}",
        f"Model: {metadata['model']}",
        f"Detected contours: {summary['detected_contours']}",
        f"Confident / uncertain: {summary['confident_classifications']} / "
        f"{summary['uncertain_classifications']}",
        f"Contour quality: {summary['overall_contour_quality']} "
        f"({summary['mean_contour_quality']:.1f}/100)",
        f"Class counts: {summary['class_counts']}",
    ]
    calibration = metadata.get("calibration", {})
    if calibration.get("enabled"):
        lines.extend(
            [
                "Spatial scale: "
                f"{calibration['centimetres_per_pixel']:.6f} cm/pixel",
                "Perspective rectification: "
                f"{'applied' if calibration.get('perspective_rectified') else 'not applied'}",
            ]
        )
    y = margin + 30
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += 22
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    overlay = Image.fromarray(overlay_rgb)
    maximum_width = page_size[0] - 2 * margin
    maximum_height = page_size[1] - y - margin
    overlay.thumbnail((maximum_width, maximum_height), Image.Resampling.LANCZOS)
    first_page.paste(overlay, (margin, y + 15))
    pages.append(first_page)

    tracks = report.get("tracks", [])
    for start in range(0, len(tracks), 10):
        page = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(page)
        draw.text((margin, margin), "Particle Evidence", fill="black", font=font)
        y = margin + 35
        for item in tracks[start : start + 10]:
            prediction = item["prediction"]
            quality = item["quality"]
            features = item.get("features", {})
            draw.text(
                (margin, y),
                f"T{item['track_id']}  {prediction['particle_type']}  "
                f"confidence {prediction['confidence']:.1%}  "
                f"quality {quality['grade']} ({quality['score']}/100)",
                fill="black",
                font=font,
            )
            y += 20
            draw.text((margin + 20, y), f"Status: {item['reporting_status']}", fill="black", font=font)
            y += 18
            if "Length (cm)" in features:
                draw.text(
                    (margin + 20, y),
                    f"Length: {features['Length (cm)']:.4f} cm  "
                    f"Width: {features['Width (cm)']:.4f} cm",
                    fill="black",
                    font=font,
                )
                y += 18
            warning = quality["warnings"][0] if quality["warnings"] else "No major quality warning."
            draw.text((margin + 20, y), warning[:150], fill="black", font=font)
            y += 32
        pages.append(page)

    field_guide = report.get("field_guide", [])
    if field_guide:
        page = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(page)
        draw.text((margin, margin), "How to Read This Report", fill="black", font=font)
        y = margin + 38
        for item in field_guide:
            heading = str(item.get("field", ""))
            description = str(item.get("description", ""))
            draw.text((margin, y), heading, fill="black", font=font)
            y += 20
            for line in textwrap.wrap(description, width=135) or [""]:
                draw.text((margin + 20, y), line, fill="black", font=font)
                y += 17
            y += 12
        pages.append(page)

    output = io.BytesIO()
    pages[0].save(
        output,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )
    return output.getvalue()
