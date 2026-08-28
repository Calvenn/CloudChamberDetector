"""JSON, CSV, and PDF encoders for reproducible classification reports."""

from __future__ import annotations

import csv
import io
import json
import textwrap
from collections import Counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    if metadata.get("model_name") == "Extremely Randomized Trees":
        lines[4:4] = [
            f"Model name: {metadata['model_name']}",
            f"Model version: {metadata.get('model_version', 'not recorded')}",
            f"Configuration: {metadata.get('model_configuration', 'not recorded')}",
            f"Random seed: {metadata.get('random_seed', 'not recorded')}",
            f"Confidence threshold: {metadata.get('confidence_threshold', 0):.0%}",
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


def _report_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a readable report font with portable fallbacks."""
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def encode_batch_pdf_report(
    entries: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> bytes:
    """Create a readable batch/video report with one annotated input per page."""
    page_size = (1240, 1754)
    margin = 70
    title_font = _report_font(34, bold=True)
    heading_font = _report_font(25, bold=True)
    body_font = _report_font(18)
    bold_font = _report_font(18, bold=True)
    small_font = _report_font(15)
    pages: list[Image.Image] = []
    class_names = ("Alpha", "Electron/Positron", "Proton", "V-track")
    aggregate_counts = Counter(
        prediction["particle_type"]
        for entry in entries
        for prediction in entry.get("predictions", [])
    )
    total_tracks = sum(aggregate_counts.values())
    threshold = float(metadata.get("confidence_threshold", 0.60))
    uncertain_tracks = sum(
        float(prediction["confidence"]) < threshold
        for entry in entries
        for prediction in entry.get("predictions", [])
    )

    cover = Image.new("RGB", page_size, "white")
    draw = ImageDraw.Draw(cover)
    draw.text(
        (margin, margin),
        "Extremely Randomized Trees Research Report",
        fill="#152238",
        font=title_font,
    )
    draw.text(
        (margin, margin + 52),
        "Batch and video-frame classification summary",
        fill="#4B5563",
        font=heading_font,
    )
    y = margin + 125
    cover_lines = [
        ("Generated", metadata.get("generated_at", "Not recorded")),
        ("Input type", metadata.get("input_type", "Batch images")),
        ("Inputs analysed", str(len(entries))),
        ("Detected tracks", str(total_tracks)),
        ("Accepted tracks", str(total_tracks - uncertain_tracks)),
        ("Tracks requiring review", str(uncertain_tracks)),
        ("Confidence threshold", f"{threshold:.0%}"),
        ("Model version", str(metadata.get("model_version", "Not recorded"))),
        ("Model configuration", str(metadata.get("model_configuration", "Not recorded"))),
    ]
    for label, value in cover_lines:
        draw.text((margin, y), f"{label}:", fill="#111827", font=bold_font)
        for line_index, line in enumerate(textwrap.wrap(str(value), width=85) or [""]):
            draw.text(
                (margin + 250, y + line_index * 24),
                line,
                fill="#111827",
                font=body_font,
            )
        y += max(34, 24 * len(textwrap.wrap(str(value), width=85) or [""]))
    y += 25
    draw.text((margin, y), "Aggregate particle counts", fill="#152238", font=heading_font)
    y += 42
    for class_name in class_names:
        draw.text(
            (margin + 25, y),
            f"{class_name}: {aggregate_counts.get(class_name, 0)}",
            fill="#111827",
            font=body_font,
        )
        y += 31
    draw.text(
        (margin, page_size[1] - margin - 25),
        f"Page 1 of {len(entries) + 1}",
        fill="#6B7280",
        font=small_font,
    )
    pages.append(cover)

    for page_index, entry in enumerate(entries, start=2):
        page = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(page)
        draw.text(
            (margin, margin),
            f"Input {page_index - 1}: {entry['name']}",
            fill="#152238",
            font=heading_font,
        )
        y = margin + 42
        for line in textwrap.wrap(str(entry.get("source", "")), width=105) or [""]:
            draw.text((margin, y), line, fill="#4B5563", font=small_font)
            y += 21
        predictions = entry.get("predictions", [])
        counts = Counter(item["particle_type"] for item in predictions)
        uncertain = sum(float(item["confidence"]) < threshold for item in predictions)
        summary_text = (
            f"Tracks: {len(predictions)}   Accepted: {len(predictions) - uncertain}   "
            f"Review: {uncertain}   Processing: {float(entry.get('processing_time_ms', 0)):.1f} ms"
        )
        draw.text((margin, y + 8), summary_text, fill="#111827", font=bold_font)
        y += 43
        counts_text = "   ".join(
            f"{name}: {counts.get(name, 0)}" for name in class_names
        )
        draw.text((margin, y), counts_text, fill="#111827", font=small_font)
        y += 35

        overlay_rgb = cv2.cvtColor(entry["overlay"], cv2.COLOR_BGR2RGB)
        overlay = Image.fromarray(overlay_rgb)
        maximum_width = page_size[0] - 2 * margin
        maximum_height = page_size[1] - y - margin - 55
        overlay.thumbnail((maximum_width, maximum_height), Image.Resampling.LANCZOS)
        image_x = (page_size[0] - overlay.width) // 2
        page.paste(overlay, (image_x, y))
        draw.rectangle(
            (image_x, y, image_x + overlay.width, y + overlay.height),
            outline="#CBD5E1",
            width=2,
        )
        draw.text(
            (margin, page_size[1] - margin - 25),
            f"Page {page_index} of {len(entries) + 1}",
            fill="#6B7280",
            font=small_font,
        )
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
