"""SVM Streamlit Page & Scientific Analysis Dashboard.

Provides single-image, batch-image, and video-sequence track classification analysis,
track inspection with feature deviation profiles, model performance metrics, and PDF/CSV/JSON/PNG exports.
"""

from pathlib import Path
from time import perf_counter
from datetime import datetime
from dataclasses import asdict
import json
import io
import re
from collections import Counter

import numpy as np
import cv2
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

import cloud_chamber.ml.member_models.svm as svm_module
from cloud_chamber.config import load_config as _load_svm_config
try:
    from .context import PageContext
except Exception:
    import cloud_chamber.ui.model_pages.context as _context_mod
    PageContext = _context_mod.PageContext

FEATURE_COLUMNS = svm_module.FEATURE_COLUMNS
CLASS_COLOURS = svm_module.CLASS_COLOURS
DISPLAY_NAMES = svm_module.DISPLAY_NAMES

# RGB hex colors corresponding to particle classes & status
RGB_COLORS = {
    "alpha": "#FFA500",             # Bright Orange
    "electron_positron": "#0078FF", # Clear Blue
    "proton": "#00C800",            # Vivid Green
    "v_track": "#FF00FF",           # Bright Magenta
    "uncertain": "#FFFF00"          # Caution Yellow
}


def render(context: PageContext) -> None:
    """Render the comprehensive SVM Track Classification Dashboard."""
    # 1. Custom CSS Styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-card h4 {
        margin: 0;
        color: #888888;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        margin: 8px 0 0 0;
        font-size: 22px;
        font-weight: bold;
    }
    .status-badge {
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .info-callout {
        background-color: rgba(0, 120, 255, 0.05);
        border-left: 4px solid #0078ff;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 13px;
    }
    .warning-callout {
        background-color: rgba(255, 193, 7, 0.08);
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔬 SVM Track Classifier & Scientific Analysis")

    # Initialize session states
    st.session_state.setdefault("svm_batch_results", {})
    st.session_state.setdefault("svm_session_info", None)
    st.session_state.setdefault("svm_predictions", None)

    # Load Trained Model and Metadata
    model_path = Path("models/svm_classifier.joblib")
    model_metadata = _get_model_metadata(model_path)
    training_report_path = Path("models/svm_training_report.json")
    training_report = _load_training_report(training_report_path)

    # Check for Shared Pipeline results
    result = st.session_state.get("pipeline_result")
    if result is None:
        st.info("💡 Process an image or video frame on the **Shared Processing Pipeline** page first to begin.")
        return

    if not model_path.exists():
        st.error("⚠️ SVM Classifier model file (`models/svm_classifier.joblib`) was not found. Run `python scripts/train_svm.py` first.")
        return

    model_bundle = svm_module.load_model(model_path)

    # ==============================================================================
    # 1. INPUT & PROCESSING SUMMARY & CONTROLS
    # ==============================================================================
    st.markdown("---")
    st.header("1️⃣ Input & Processing Summary")

    batch_samples = st.session_state.get("input_batch", [])
    has_batch = len(batch_samples) > 1
    is_video = has_batch and any("_frame_" in sample["name"] for sample in batch_samples)

    if is_video:
        detected_input_type = "Video"
    elif has_batch:
        detected_input_type = "Batch Images"
    else:
        detected_input_type = "Single Image"

    col_preview, col_checklist = st.columns([3, 2])

    with col_preview:
        st.subheader("🖼️ Input Frame / Overlay Preview")
        contours = result["segmentation"].contours
        active_image = result["input_image"].copy()
        cv2.drawContours(active_image, contours, -1, (255, 255, 0), 2)
        st.image(context.bgr_to_rgb(active_image), caption=f"Active Frame Contour Boundaries ({detected_input_type})", use_container_width=True)

        frame_name = st.session_state.get("input_name", "N/A")
        frame_desc = st.session_state.get("source_description", "N/A")
        img_h, img_w = result["input_image"].shape[:2]

        details_html = f"""
        <div style="background-color: rgba(128,128,128,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.15); margin-top: 8px; font-size: 13px;">
            <b>Input Type:</b> <span style="color:#0078ff; font-weight:bold;">{detected_input_type}</span><br>
            <b>Active Frame Name:</b> {frame_name}<br>
            <b>Image Dimensions:</b> {img_w} × {img_h} px<br>
            <b>Detected Tracks:</b> {len(contours)}<br>
            <b>Source Description:</b> {frame_desc}
        """
        frame_num, timestamp = _parse_video_metadata(frame_name, frame_desc)
        if frame_num is not None:
            details_html += f"<br><b>Frame Number:</b> {frame_num}"
        if timestamp is not None:
            details_html += f"<br><b>Video Timestamp:</b> {timestamp:.2f} s"
        details_html += "</div>"
        st.markdown(details_html, unsafe_allow_html=True)

    with col_checklist:
        st.subheader("📋 Pre-Classification Checklist")
        img_uploaded = st.session_state.get("input_image") is not None
        st.markdown(_render_checklist_item("Image/Video Uploaded", "success" if img_uploaded else "danger"), unsafe_allow_html=True)
        st.markdown(_render_checklist_item("Preprocessing Complete", "success" if result else "danger"), unsafe_allow_html=True)
        st.markdown(_render_checklist_item(f"Tracks Detected ({len(contours)})", "success" if len(contours) > 0 else "warning"), unsafe_allow_html=True)
        has_features = len(result.get("features", [])) > 0
        st.markdown(_render_checklist_item("Features Extracted", "success" if has_features else "danger"), unsafe_allow_html=True)
        st.markdown(_render_checklist_item("SVM Model Loaded", "success" if model_metadata["status"] == "Loaded" else "danger"), unsafe_allow_html=True)

    # Classification Scope & Threshold Controls
    st.markdown("---")
    st.subheader("⚙️ Classification Scope & Threshold Controls")
    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        confidence_threshold = st.slider(
            "Reporting Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.50,
            step=0.05,
            help="Predictions below this probability threshold will be marked as 'Uncertain' in reports."
        )

        scope_options = ["Active Frame Only"]
        if has_batch:
            scope_options.append("Entire Batch/Video")

        scope_choice = st.radio("Classification Scope", scope_options, horizontal=True)

    with ctrl_col2:
        st.markdown("**Visualization Toggles**")
        show_probs = st.checkbox("Show probability distribution", value=True)
        show_features_contrib = st.checkbox("Show feature deviation profile (Z-scores)", value=True)

        temporal_analysis = False
        if is_video and scope_choice == "Entire Batch/Video":
            st.markdown("**Video Trajectory Toggles**")
            temporal_analysis = st.toggle("Enable Temporal Trajectory Tracking", value=True)

    # Run Button
    prereqs_met = img_uploaded and result is not None and model_metadata["status"] == "Loaded"
    if st.button("▶ Start SVM Classification", type="primary", disabled=not prereqs_met, use_container_width=True):
        samples_to_process = batch_samples if (scope_choice == "Entire Batch/Video" and has_batch) else [
            {
                "image": st.session_state["input_image"],
                "name": st.session_state["input_name"],
                "description": st.session_state["source_description"]
            }
        ]

        batch_results = {}
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        start_time = perf_counter()
        total_tracks = 0

        for index, sample in enumerate(samples_to_process):
            _config = st.session_state.get("config") or _load_svm_config()
            sample_result = context.process_pipeline_image(
                sample["image"],
                config=_config,
                calibration_settings=st.session_state.get("calibration_settings")
            )

            # Use official SVM predict_tracks function from svm_module
            sample_preds = svm_module.predict_tracks(model_bundle, sample_result["features"])

            batch_results[sample["name"]] = {
                "image": sample["image"],
                "analysis_image": sample_result["input_image"],
                "description": sample["description"],
                "result": sample_result,
                "predictions": sample_preds,
            }

            total_tracks += len(sample_preds)
            elapsed = perf_counter() - start_time
            processed = index + 1
            progress_bar.progress(processed / len(samples_to_process))
            status_text.markdown(f"⏳ **Classifying sample {processed} of {len(samples_to_process)}...** (`{total_tracks}` total tracks)")

        progress_bar.empty()
        status_text.empty()

        st.session_state["svm_batch_results"] = batch_results
        active_frame_name = st.session_state["input_name"]
        if active_frame_name in batch_results:
            st.session_state["svm_predictions"] = batch_results[active_frame_name]["predictions"]
        else:
            first_key = list(batch_results.keys())[0]
            st.session_state["svm_predictions"] = batch_results[first_key]["predictions"]

        elapsed_total = perf_counter() - start_time
        st.session_state["svm_session_info"] = {
            "input_type": detected_input_type,
            "samples_processed": len(batch_results),
            "processing_time_s": elapsed_total,
            "total_tracks": total_tracks,
            "kernel": model_metadata["kernel"],
            "C": model_metadata["C"],
            "gamma": model_metadata["gamma"],
        }
        st.success(f"Successfully classified {total_tracks} track(s) across {len(batch_results)} input sample(s)!")

    # Check available results
    batch_results = {k: v for k, v in st.session_state.get("svm_batch_results", {}).items() if k is not None}
    if not batch_results:
        return

    # Sync selected frame
    batch_keys = list(batch_results.keys())
    if len(batch_keys) > 1:
        st.markdown("---")
        st.subheader("🖼️ Select Input Frame / Image for Detailed Inspection")
        current_input_name = st.session_state.get("input_name", batch_keys[0])
        default_index = batch_keys.index(current_input_name) if current_input_name in batch_keys else 0
        selected_idx = st.selectbox(
            "Active Sample:",
            range(len(batch_keys)),
            index=default_index,
            format_func=lambda i: f"[{i+1}/{len(batch_keys)}] {batch_keys[i]}"
        )
        selected_name = batch_keys[selected_idx]
    else:
        selected_name = batch_keys[0]

    selected_entry = batch_results[selected_name]
    st.session_state["input_image"] = selected_entry["image"]
    st.session_state["input_name"] = selected_name
    st.session_state["source_description"] = selected_entry["description"]
    st.session_state["pipeline_result"] = selected_entry["result"]
    st.session_state["svm_predictions"] = selected_entry["predictions"]

    result = selected_entry["result"]
    predictions = selected_entry["predictions"]
    frame_name = selected_name
    frame_desc = selected_entry["description"]

    # Calculate aggregate & active stats
    all_tracks_flat = []
    for b_name, b_entry in batch_results.items():
        f_num, f_ts = _parse_video_metadata(b_name, b_entry["description"])
        for trk, prd in zip(b_entry["result"]["features"], b_entry["predictions"]):
            all_tracks_flat.append({
                "image_name": b_name,
                "frame_num": f_num,
                "timestamp": f_ts,
                "track": trk,
                "prediction": prd
            })

    # ==============================================================================
    # 2. CLASSIFICATION SUMMARY & KPI CARDS
    # ==============================================================================
    # ==============================================================================
    # 2. CLASSIFICATION SUMMARY
    # ==============================================================================
    st.markdown("---")
    st.header("2️⃣ Classification Summary")

    if detected_input_type == "Single Image":
        total_tracks = len(predictions)
        accepted_cnt = sum(1 for p in predictions if p["confidence"] >= confidence_threshold)
        uncertain_cnt = total_tracks - accepted_cnt
        conf_list = [p["confidence"] for p in predictions]
        avg_conf = np.mean(conf_list) if conf_list else 0.0
        class_counts = Counter(p["predicted_class"] for p in predictions)
        dominant_cls = class_counts.most_common(1)[0][0] if class_counts else "None"

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Tracks", total_tracks)
        k2.metric("Accepted", accepted_cnt)
        k3.metric("Uncertain", uncertain_cnt)
        k4.metric("Avg Confidence", f"{avg_conf:.1%}")
        k5.metric("Dominant Class", DISPLAY_NAMES.get(dominant_cls, dominant_cls))

        st.markdown(f"""
        <div class="info-callout">
            <b>Analysis Interpretation:</b><br>
            A total of <b>{total_tracks}</b> particle track(s) were detected in image <code>{frame_name}</code>. 
            <b>{accepted_cnt}</b> classification(s) met the reporting confidence threshold (≥ {confidence_threshold:.0%}), while 
            <b>{uncertain_cnt}</b> were flagged as uncertain. 
            <b>{DISPLAY_NAMES.get(dominant_cls, dominant_cls)}</b> was the dominant predicted particle class.
        </div>
        """, unsafe_allow_html=True)

    elif detected_input_type == "Batch Images":
        num_images = len(batch_results)
        total_tracks = len(all_tracks_flat)
        accepted_cnt = sum(1 for item in all_tracks_flat if item["prediction"]["confidence"] >= confidence_threshold)
        uncertain_cnt = total_tracks - accepted_cnt
        conf_list = [item["prediction"]["confidence"] for item in all_tracks_flat]
        avg_conf = np.mean(conf_list) if conf_list else 0.0
        class_counts = Counter(item["prediction"]["predicted_class"] for item in all_tracks_flat)
        dominant_cls = class_counts.most_common(1)[0][0] if class_counts else "None"
        total_proc_time_s = st.session_state.get("svm_session_info", {}).get("processing_time_s", 0.0)
        avg_proc_ms = (total_proc_time_s * 1000.0) / max(1, num_images)

        b1, b2, b3 = st.columns(3)
        b1.metric("Images Processed", num_images)
        b2.metric("Total Tracks", total_tracks)
        b3.metric("Dominant Class", DISPLAY_NAMES.get(dominant_cls, dominant_cls))

        b4, b5, b6 = st.columns(3)
        b4.metric("Accepted Tracks", accepted_cnt)
        b5.metric("Uncertain Tracks", uncertain_cnt)
        b6.metric("Avg Confidence", f"{avg_conf:.1%}")

        dom_pct = (class_counts.get(dominant_cls, 0) / max(1, total_tracks)) * 100.0
        st.markdown(f"""
        <div class="info-callout">
            <b>Batch Interpretation:</b><br>
            Across <b>{num_images}</b> batch images, a total of <b>{total_tracks}</b> tracks were detected and classified. 
            <b>{accepted_cnt}</b> ({accepted_cnt/max(1,total_tracks):.1%}) met the reporting confidence threshold (≥ {confidence_threshold:.0%}), 
            and <b>{uncertain_cnt}</b> were flagged as uncertain. 
            <b>{DISPLAY_NAMES.get(dominant_cls, dominant_cls)}</b> represented <b>{dom_pct:.1f}%</b> of all predictions across the batch. 
            Total execution time was {total_proc_time_s:.2f} s ({avg_proc_ms:.1f} ms/image).
        </div>
        """, unsafe_allow_html=True)

    else: # Video
        num_frames = len(batch_results)
        total_tracks = len(all_tracks_flat)
        accepted_cnt = sum(1 for item in all_tracks_flat if item["prediction"]["confidence"] >= confidence_threshold)
        uncertain_cnt = total_tracks - accepted_cnt
        conf_list = [item["prediction"]["confidence"] for item in all_tracks_flat]
        avg_conf = np.mean(conf_list) if conf_list else 0.0
        class_counts = Counter(item["prediction"]["predicted_class"] for item in all_tracks_flat)
        dominant_cls = class_counts.most_common(1)[0][0] if class_counts else "None"

        max_ts = max([item["timestamp"] for item in all_tracks_flat if item["timestamp"] is not None] or [0.0])

        v1, v2, v3 = st.columns(3)
        v1.metric("Video Frames", num_frames)
        v2.metric("Total Tracks", total_tracks)
        v3.metric("Dominant Class", DISPLAY_NAMES.get(dominant_cls, dominant_cls))

        v4, v5, v6 = st.columns(3)
        v4.metric("Accepted Tracks", accepted_cnt)
        v5.metric("Uncertain Tracks", uncertain_cnt)
        v6.metric("Avg Confidence", f"{avg_conf:.1%}")

        dom_pct = (class_counts.get(dominant_cls, 0) / max(1, total_tracks)) * 100.0
        st.markdown(f"""
        <div class="info-callout">
            <b>Video Sequence Interpretation:</b><br>
            Across <b>{num_frames}</b> video sequence frames (spanning ~{max_ts:.1f} seconds), <b>{total_tracks}</b> track detections were recorded. 
            <b>{accepted_cnt}</b> classifications met the reporting threshold, while <b>{uncertain_cnt}</b> were flagged as uncertain. 
            <b>{DISPLAY_NAMES.get(dominant_cls, dominant_cls)}</b> was the dominant classification (<b>{dom_pct:.1f}%</b> of total detections).
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # 3. CLASSIFICATION MAP / VISUAL RESULTS
    # ==============================================================================
    st.markdown("---")
    st.header("3️⃣ Classification Map / Visual Results")

    overlay_img, _ = svm_module.build_visual_report(
        result["input_image"],
        result["features"],
        predictions,
        confidence_threshold
    )

    st.image(context.bgr_to_rgb(overlay_img), caption=f"Annotated Classification Overlay for {frame_name}", use_container_width=True)

    # Color Legend Bar
    legend_cols = st.columns(5)
    legend_cols[0].markdown(f'<div style="background-color:{RGB_COLORS["alpha"]}; padding:6px; text-align:center; border-radius:4px; font-weight:bold; color:black;">Alpha</div>', unsafe_allow_html=True)
    legend_cols[1].markdown(f'<div style="background-color:{RGB_COLORS["electron_positron"]}; padding:6px; text-align:center; border-radius:4px; font-weight:bold; color:white;">Electron/Positron</div>', unsafe_allow_html=True)
    legend_cols[2].markdown(f'<div style="background-color:{RGB_COLORS["proton"]}; padding:6px; text-align:center; border-radius:4px; font-weight:bold; color:white;">Proton</div>', unsafe_allow_html=True)
    legend_cols[3].markdown(f'<div style="background-color:{RGB_COLORS["v_track"]}; padding:6px; text-align:center; border-radius:4px; font-weight:bold; color:white;">V-track</div>', unsafe_allow_html=True)
    legend_cols[4].markdown(f'<div style="background-color:{RGB_COLORS["uncertain"]}; padding:6px; text-align:center; border-radius:4px; font-weight:bold; color:black;">Uncertain (&lt; {confidence_threshold:.0%})</div>', unsafe_allow_html=True)

    # ==============================================================================
    # 4. PARTICLE / CLASS DISTRIBUTION
    # ==============================================================================
    st.markdown("---")
    st.header("4️⃣ Particle / Class Distribution")

    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        # Distribution in current active frame
        active_counts = Counter(p["predicted_class"] for p in predictions)
        active_df = pd.DataFrame([
            {
                "Particle Class": DISPLAY_NAMES.get(cls, cls),
                "Count": cnt,
                "Percentage": (cnt / max(1, len(predictions))) * 100.0
            }
            for cls, cnt in active_counts.items()
        ])

        if not active_df.empty:
            fig_active_bar = px.bar(
                active_df,
                x="Particle Class",
                y="Count",
                color="Particle Class",
                text=active_df["Percentage"].apply(lambda p: f"{p:.1f}%"),
                color_discrete_map={
                    "Alpha": RGB_COLORS["alpha"],
                    "Electron/Positron": RGB_COLORS["electron_positron"],
                    "Proton": RGB_COLORS["proton"],
                    "V-track": RGB_COLORS["v_track"]
                },
                title=f"Class Distribution in Active Frame ({frame_name})"
            )
            fig_active_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_active_bar, use_container_width=True)

    with dist_col2:
        # Aggregate distribution across batch/video if applicable, or pie chart for single image
        if has_batch:
            agg_counts = Counter(item["prediction"]["predicted_class"] for item in all_tracks_flat)
            agg_df = pd.DataFrame([
                {
                    "Particle Class": DISPLAY_NAMES.get(cls, cls),
                    "Count": cnt,
                    "Percentage": (cnt / max(1, len(all_tracks_flat))) * 100.0
                }
                for cls, cnt in agg_counts.items()
            ])
            fig_agg_pie = px.pie(
                agg_df,
                names="Particle Class",
                values="Count",
                color="Particle Class",
                color_discrete_map={
                    "Alpha": RGB_COLORS["alpha"],
                    "Electron/Positron": RGB_COLORS["electron_positron"],
                    "Proton": RGB_COLORS["proton"],
                    "V-track": RGB_COLORS["v_track"]
                },
                title=f"Overall Aggregate Class Distribution ({detected_input_type})"
            )
            st.plotly_chart(fig_agg_pie, use_container_width=True)
        else:
            if not active_df.empty:
                fig_active_pie = px.pie(
                    active_df,
                    names="Particle Class",
                    values="Count",
                    color="Particle Class",
                    color_discrete_map={
                        "Alpha": RGB_COLORS["alpha"],
                        "Electron/Positron": RGB_COLORS["electron_positron"],
                        "Proton": RGB_COLORS["proton"],
                        "V-track": RGB_COLORS["v_track"]
                    },
                    title="Particle Class Breakdown"
                )
                st.plotly_chart(fig_active_pie, use_container_width=True)

    # ==============================================================================
    # 5. CONFIDENCE ANALYSIS
    # ==============================================================================
    st.markdown("---")
    st.header("5️⃣ Confidence Analysis")

    target_preds = [item["prediction"] for item in all_tracks_flat] if has_batch else predictions
    conf_values = [p["confidence"] for p in target_preds]

    if conf_values:
        avg_c = np.mean(conf_values)
        med_c = np.median(conf_values)
        min_c = np.min(conf_values)
        max_c = np.max(conf_values)
        above_thresh = sum(1 for c in conf_values if c >= confidence_threshold)
        below_thresh = len(conf_values) - above_thresh

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Avg Confidence", f"{avg_c:.1%}")
        m2.metric("Median Confidence", f"{med_c:.1%}")
        m3.metric("Min Confidence", f"{min_c:.1%}")
        m4.metric("Max Confidence", f"{max_c:.1%}")
        m5.metric("≥ Threshold", f"{above_thresh} ({above_thresh/len(conf_values):.1%})")
        m6.metric("< Threshold", f"{below_thresh} ({below_thresh/len(conf_values):.1%})")

        conf_col1, conf_col2 = st.columns(2)
        with conf_col1:
            fig_conf_hist = px.histogram(
                pd.DataFrame({"Confidence Score": conf_values}),
                x="Confidence Score",
                nbins=15,
                title=f"Confidence Distribution ({'Batch/Video' if has_batch else 'Active Image'})"
            )
            fig_conf_hist.add_vline(
                x=confidence_threshold,
                line_width=3,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({confidence_threshold:.0%})"
            )
            st.plotly_chart(fig_conf_hist, use_container_width=True)

        with conf_col2:
            # Per-track confidence & prediction margin bar chart
            margin_data = []
            for p in predictions[:20]: # Show up to 20 tracks
                probs = p["probabilities"]
                sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                first_cls, first_prob = sorted_p[0]
                second_cls, second_prob = sorted_p[1] if len(sorted_p) > 1 else ("N/A", 0.0)
                margin = first_prob - second_prob
                margin_data.append({
                    "Track ID": f"T{p['track_id']}",
                    "Top Confidence": first_prob,
                    "Margin": margin,
                    "Second Best": DISPLAY_NAMES.get(second_cls, second_cls)
                })
            m_df = pd.DataFrame(margin_data)
            if not m_df.empty:
                fig_margin = px.bar(
                    m_df,
                    x="Track ID",
                    y=["Top Confidence", "Margin"],
                    barmode="group",
                    title="Active Frame Track Confidence & Prediction Margin"
                )
                st.plotly_chart(fig_margin, use_container_width=True)

    st.markdown("""
    <div class="warning-callout">
        <b>📌 Technical Clarification on Prediction Rules vs Reporting Threshold:</b><br>
        • <b>SVM Model Prediction Rule:</b> The trained Support Vector Machine assigns multi-class probabilities to candidate tracks. A dedicated physics rule enforces that a track is only classified as <code>Proton</code> if $P(\\text{{Proton}}) > 0.75$; otherwise, the second-most probable particle class is assigned.<br>
        • <b>User Reporting Threshold:</b> The selected confidence slider (currently <b>{:.0%}</b>) acts purely as an aesthetic & reporting filter. Predictions below this threshold are marked <code>Uncertain</code> in reports, but the underlying SVM probabilities and particle assignments remain unchanged.
    </div>
    """.format(confidence_threshold), unsafe_allow_html=True)

    # ==============================================================================
    # 6. TRACK-LEVEL RESULTS TABLE
    # ==============================================================================
    st.markdown("---")
    st.header("6️⃣ Track-Level Results Table")

    # Table Filters
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        status_filter = st.radio("Status Filter:", ["All", "Accepted", "Uncertain"], horizontal=True)
    with f_col2:
        class_filter = st.selectbox("Particle Class Filter:", ["All", "Alpha", "Electron/Positron", "Proton", "V-track"])

    table_rows = []
    source_items = all_tracks_flat if has_batch else [
        {"image_name": frame_name, "frame_num": _parse_video_metadata(frame_name, frame_desc)[0], "timestamp": _parse_video_metadata(frame_name, frame_desc)[1], "track": trk, "prediction": prd}
        for trk, prd in zip(result["features"], predictions)
    ]

    for item in source_items:
        img_name = item["image_name"]
        trk = item["track"]
        pred = item["prediction"]
        conf = pred["confidence"]
        status = "Accepted" if conf >= confidence_threshold else "Uncertain"
        pred_cls_display = pred["particle_type"]

        probs = pred["probabilities"]
        sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        second_cls = DISPLAY_NAMES.get(sorted_p[1][0], sorted_p[1][0]) if len(sorted_p) > 1 else "N/A"
        second_prob = sorted_p[1][1] if len(sorted_p) > 1 else 0.0
        margin = conf - second_prob

        # Apply filters
        if status_filter != "All" and status != status_filter:
            continue
        if class_filter != "All" and pred_cls_display != class_filter:
            continue

        prob_str = ", ".join([f"{DISPLAY_NAMES.get(c, c)}: {p:.1%}" for c, p in sorted_p])

        row_dict = {}
        if has_batch:
            row_dict["Image / Frame"] = img_name
            if item["frame_num"] is not None:
                row_dict["Frame #"] = item["frame_num"]
            if item["timestamp"] is not None:
                row_dict["Time (s)"] = f"{item['timestamp']:.2f}"

        row_dict.update({
            "Track ID": f"T{trk.track_id}",
            "Prediction": pred_cls_display,
            "Confidence": f"{conf:.1%}",
            "Status": status,
            "Second-Best Class": second_cls,
            "Prediction Margin": f"{margin:.1%}",
            "Probabilities": prob_str
        })
        table_rows.append(row_dict)

    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No tracks match the selected table filters.")

    # ==============================================================================
    # 7. TRACK INSPECTOR
    # ==============================================================================
    st.markdown("---")
    st.header("7️⃣ Track Inspector")

    raw_track_ids = [t.track_id for t in result["features"]]
    if raw_track_ids:
        selected_tid = st.selectbox("Select a Track to inspect in detail:", [f"Track T{tid}" for tid in raw_track_ids])
        tid_num = int(selected_tid.replace("Track T", ""))
        t_idx = raw_track_ids.index(tid_num)

        selected_feat = result["features"][t_idx]
        selected_pred = predictions[t_idx]
        conf_val = selected_pred["confidence"]
        status_str = "Accepted" if conf_val >= confidence_threshold else "Uncertain"

        insp_col1, insp_col2 = st.columns(2)

        with insp_col1:
            st.markdown(f"#### Track **T{tid_num}** Details")
            st.markdown(f"""
            • **Predicted Class:** `{selected_pred['particle_type']}`  
            • **Confidence Score:** `{conf_val:.1%}`  
            • **Status:** `{status_str}`  
            """)

            # Probability Distribution Chart
            prob_df = pd.DataFrame([
                {"Particle Class": DISPLAY_NAMES.get(c, c), "Probability": p}
                for c, p in selected_pred["probabilities"].items()
            ]).sort_values("Probability", ascending=True)

            fig_insp_prob = px.bar(
                prob_df,
                x="Probability",
                y="Particle Class",
                orientation="h",
                color="Particle Class",
                color_discrete_map={
                    "Alpha": RGB_COLORS["alpha"],
                    "Electron/Positron": RGB_COLORS["electron_positron"],
                    "Proton": RGB_COLORS["proton"],
                    "V-track": RGB_COLORS["v_track"]
                },
                title=f"T{tid_num} Class Probabilities"
            )
            fig_insp_prob.update_layout(showlegend=False)
            st.plotly_chart(fig_insp_prob, use_container_width=True)

            # Cropped Track Outline
            st.markdown("**Cropped Contour Image**")
            x, y, w, h = selected_feat.bounding_box
            pad = 20
            h_img, w_img = result["input_image"].shape[:2]
            y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
            x1, x2 = max(0, x - pad), min(w_img, x + w + pad)
            cropped_img = result["input_image"][y1:y2, x1:x2].copy()

            contours_active = result["segmentation"].contours
            if t_idx < len(contours_active):
                shifted_c = contours_active[t_idx] - np.array([x1, y1])
                cv2.drawContours(cropped_img, [shifted_c], -1, (255, 255, 0), 2)

            st.image(context.bgr_to_rgb(cropped_img), caption=f"Track T{tid_num} Bounding Box Crop ({w}×{h} px)", width=240)

        with insp_col2:
            st.markdown("#### Feature Values & Feature Deviation Profile")

            # Table of extracted feature values
            feat_dict = asdict(selected_feat)
            area = float(feat_dict.get("area_pixels", 0.0))
            perimeter = float(feat_dict.get("perimeter_pixels", 0.0))
            major_axis = float(feat_dict.get("major_axis_pixels", 0.0))
            mean_intensity = float(feat_dict.get("mean_intensity", 0.0))
            feat_dict["line_density"] = (mean_intensity * area) / major_axis if major_axis > 0 else 0.0
            feat_dict["tortuosity"] = perimeter / (2.0 * major_axis) if major_axis > 0 else 1.0

            feat_table = pd.DataFrame([
                {"Feature": col, "Extracted Value": f"{feat_dict[col]:.4f}"}
                for col in FEATURE_COLUMNS
            ])
            st.dataframe(feat_table, use_container_width=True, hide_index=True)

            # Feature Deviation Profile (Z-scores)
            if "scaler" in model_bundle["model"].named_steps:
                scaler = model_bundle["model"].named_steps["scaler"]
                center = getattr(scaler, "center_", getattr(scaler, "mean_", np.zeros(len(FEATURE_COLUMNS))))
                scale = getattr(scaler, "scale_", np.ones(len(FEATURE_COLUMNS)))

                raw_vec = np.array([feat_dict[col] for col in FEATURE_COLUMNS])
                z_scores = (raw_vec - center) / scale

                z_df = pd.DataFrame({
                    "Feature": FEATURE_COLUMNS,
                    "Z-score": z_scores
                }).sort_values("Z-score", ascending=True)

                fig_z = px.bar(
                    z_df,
                    x="Z-score",
                    y="Feature",
                    orientation="h",
                    title=f"Feature Deviation Profile (T{tid_num})"
                )
                st.plotly_chart(fig_z, use_container_width=True)
                st.caption("Positive and negative Z-scores indicate how far the feature value lies above or below the training distribution centre.")

    # ==============================================================================
    # 8. OVERALL FEATURE ANALYSIS
    # ==============================================================================
    st.markdown("---")
    st.header("8️⃣ Overall Feature Analysis")

    if result["features"]:
        feat_matrix = svm_module.features_to_matrix(result["features"])
        df_feats = pd.DataFrame(feat_matrix, columns=FEATURE_COLUMNS)

        feat_stats = df_feats.describe().T[["mean", "std", "min", "50%", "max"]]
        feat_stats.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]

        feat_stats_col, z_all_col = st.columns(2)
        with feat_stats_col:
            st.markdown("**Feature Statistics Across Frame Tracks**")
            st.dataframe(feat_stats.style.format("{:.3f}"), use_container_width=True)

        with z_all_col:
            if "scaler" in model_bundle["model"].named_steps:
                scaler = model_bundle["model"].named_steps["scaler"]
                center = getattr(scaler, "center_", getattr(scaler, "mean_", np.zeros(len(FEATURE_COLUMNS))))
                scale = getattr(scaler, "scale_", np.ones(len(FEATURE_COLUMNS)))

                z_matrix = (feat_matrix - center) / scale
                avg_z = np.mean(z_matrix, axis=0)

                z_avg_df = pd.DataFrame({
                    "Feature": FEATURE_COLUMNS,
                    "Mean Z-score": avg_z
                }).sort_values("Mean Z-score", ascending=True)

                fig_z_avg = px.bar(
                    z_avg_df,
                    x="Mean Z-score",
                    y="Feature",
                    orientation="h",
                    title="Overall Mean Feature Z-Score Profile"
                )
                st.plotly_chart(fig_z_avg, use_container_width=True)

    # ==============================================================================
    # 9. BATCH / VIDEO ANALYTICS SECTION (when applicable)
    # ==============================================================================
    if has_batch:
        st.markdown("---")
        if is_video:
            st.header("9️⃣ Video Temporal Analytics & Trajectory Journey")

            sorted_frames = sorted(list(batch_results.keys()))

            # Video Timeline Evolution Chart
            evol_rows = []
            for f_name in sorted_frames:
                f_entry = batch_results[f_name]
                f_counts = Counter(p["predicted_class"] for p in f_entry["predictions"])
                f_num, f_ts = _parse_video_metadata(f_name, f_entry["description"])
                row = {
                    "Frame": f_num if f_num is not None else f_name,
                    "Timestamp (s)": f_ts if f_ts is not None else 0.0,
                    "Track Count": len(f_entry["predictions"])
                }
                for cls in ["alpha", "electron_positron", "proton", "v_track"]:
                    row[DISPLAY_NAMES[cls]] = f_counts.get(cls, 0)
                evol_rows.append(row)

            evol_df = pd.DataFrame(evol_rows)
            fig_video_line = px.line(
                evol_df,
                x="Timestamp (s)" if "Timestamp (s)" in evol_df.columns else "Frame",
                y=["Alpha", "Electron/Positron", "Proton", "V-track"],
                color_discrete_map={
                    "Alpha": RGB_COLORS["alpha"],
                    "Electron/Positron": RGB_COLORS["electron_positron"],
                    "Proton": RGB_COLORS["proton"],
                    "V-track": RGB_COLORS["v_track"]
                },
                title="Particle Classification Evolution Over Video Sequence"
            )
            st.plotly_chart(fig_video_line, use_container_width=True)

            # Trajectory Association Overlay
            if temporal_analysis:
                st.subheader("📍 Multi-Frame Centroid Trajectory Association")
                global_tracks = _associate_tracks(batch_results, sorted_frames)
                multi_frame_gts = [gt for gt in global_tracks if len(gt["history"]) > 1]

                if multi_frame_gts:
                    gt_ids = [gt["global_id"] for gt in multi_frame_gts]
                    sel_gt_id = st.selectbox("Select Trajectory Track to trace:", gt_ids, format_func=lambda x: f"Global Track GT-{x}")
                    selected_gt = next(gt for gt in global_tracks if gt["global_id"] == sel_gt_id)

                    journey_overlay = _draw_track_journey(result["input_image"], selected_gt, frame_name, sorted_frames)
                    st.image(context.bgr_to_rgb(journey_overlay), caption=f"Trajectory History of GT-{sel_gt_id}", use_container_width=True)

                    gt_history_rows = []
                    for occ in selected_gt["history"]:
                        f_n, _ = _parse_video_metadata(occ["frame"], "")
                        gt_history_rows.append({
                            "Frame Name": occ["frame"],
                            "Frame #": f_n if f_n is not None else "N/A",
                            "Track ID": f"T{occ['track_id']}",
                            "Predicted Class": DISPLAY_NAMES.get(occ["class"], occ["class"]),
                            "Confidence": f"{occ['confidence']:.1%}"
                        })
                    st.dataframe(pd.DataFrame(gt_history_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No multi-frame trajectories associated across current video sampling interval.")
        else:
            st.header("9️⃣ Batch Image Overview & Summary Table")

            # Image-Level Summary Table
            img_table_rows = []
            for b_name, b_entry in batch_results.items():
                b_preds = b_entry["predictions"]
                b_cnt = len(b_preds)
                b_acc = sum(1 for p in b_preds if p["confidence"] >= confidence_threshold)
                b_unc = b_cnt - b_acc
                b_conf_list = [p["confidence"] for p in b_preds]
                b_avg_conf = np.mean(b_conf_list) if b_conf_list else 0.0
                b_class_cnt = Counter(p["predicted_class"] for p in b_preds)
                b_dom = b_class_cnt.most_common(1)[0][0] if b_class_cnt else "None"
                b_time_ms = float(b_entry["result"]["segmentation"].processing_time_ms)

                img_table_rows.append({
                    "Image Name": b_name,
                    "Tracks Detected": b_cnt,
                    "Dominant Class": DISPLAY_NAMES.get(b_dom, b_dom),
                    "Avg Confidence": f"{b_avg_conf:.1%}",
                    "Accepted": b_acc,
                    "Uncertain": b_unc,
                    "Processing Time (ms)": f"{b_time_ms:.1f}",
                    "Status": "Complete"
                })

            st.dataframe(pd.DataFrame(img_table_rows), use_container_width=True, hide_index=True)

    # ==============================================================================
    # 10. MODEL PERFORMANCE SECTION
    # ==============================================================================
    st.markdown("---")
    st.header("🔟 Model Performance Evaluation")

    st.markdown("""
    <div class="info-callout">
        <b>NOTE:</b> The metrics below represent <b>Model Evaluation Performance</b> measured on the holdout validation/test datasets during model training. They are not metrics calculated from your uploaded image.
    </div>
    """, unsafe_allow_html=True)

    if training_report:
        seg_test = training_report.get("segmented_final_test", {})
        if seg_test:
            acc = seg_test.get("accuracy", 0.0)
            bal_acc = seg_test.get("balanced_accuracy", 0.0)
            macro_f1 = seg_test.get("macro_f1", 0.0)
            weighted_f1 = seg_test.get("weighted_f1", 0.0)

            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Final Test Accuracy", f"{acc:.1%}")
            pm2.metric("Balanced Accuracy", f"{bal_acc:.1%}")
            pm3.metric("Macro F1-Score", f"{macro_f1:.1%}")
            pm4.metric("Weighted F1-Score", f"{weighted_f1:.1%}")

            perf_col1, perf_col2 = st.columns(2)

            with perf_col1:
                st.markdown("#### Per-Class Metrics (Final Test Set)")
                clf_rep = seg_test.get("classification_report", {})
                per_class_rows = []
                for cls in ["alpha", "electron_positron", "proton", "v_track"]:
                    if cls in clf_rep:
                        stats = clf_rep[cls]
                        per_class_rows.append({
                            "Class": DISPLAY_NAMES.get(cls, cls),
                            "Precision": f"{stats.get('precision', 0.0):.1%}",
                            "Recall": f"{stats.get('recall', 0.0):.1%}",
                            "F1-Score": f"{stats.get('f1-score', 0.0):.1%}",
                            "Support": int(stats.get("support", 0))
                        })
                st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)

            with perf_col2:
                st.markdown("#### Confusion Matrix (Final Test Set)")
                cm = seg_test.get("confusion_matrix", [])
                if cm:
                    class_labels = [DISPLAY_NAMES.get(c, c) for c in seg_test.get("class_names", ["alpha", "electron_positron", "proton", "v_track"])]
                    fig_cm = px.imshow(
                        cm,
                        x=class_labels,
                        y=class_labels,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted Class", y="True Ground Truth Class", color="Count"),
                        title="Final Test Confusion Matrix"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Training report metadata is not available.")

    # ==============================================================================
    # 11. MODEL / METHODOLOGY INFORMATION
    # ==============================================================================
    st.markdown("---")
    with st.expander("ℹ️ 11. Model & Methodology Information", expanded=False):
        st.markdown(f"""
        ### SVM Architecture & Preprocessing Pipeline
        • **Model Pipeline:** `scikit-learn Pipeline` (`RobustScaler` + `SVC`)  
        • **Kernel Type:** `{model_metadata['kernel']}`  
        • **Regularization Parameter (C):** `{model_metadata['C']}`  
        • **Kernel Coefficient (gamma):** `{model_metadata['gamma']}`  
        • **Class Weighting:** `class_weight='balanced'`  
        • **Number of Features:** `{len(FEATURE_COLUMNS)}` numerical contour features  
        • **Feature Names:** `{', '.join(FEATURE_COLUMNS)}`  
        • **Spatial Scale Calibration:** `{f"{result.get('centimetres_per_pixel'):.6f} cm/px" if result.get('centimetres_per_pixel') else "Pixels (Uncalibrated)"}`  
        • **ROI Profile:** `{result.get('roi_profile', 'Full Frame')}`  
        • **Rectification Status:** `{result.get('rectified', False)}`  
        • **Reporting Threshold:** `{confidence_threshold:.2f}`  
        • **Proton Rule Enforcement:** Only assigned if $P(\\text{{Proton}}) > 0.75$  
        """)

    # ==============================================================================
    # 12. REPORT & EXPORT CENTRE
    # ==============================================================================
    st.markdown("---")
    st.header("12. 📤 Report Export Centre")

    # Generate metadata dictionary for exports
    export_metadata = {
        "input_type": detected_input_type,
        "input_name": frame_name,
        "source_description": frame_desc,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_threshold": confidence_threshold,
        "roi_profile": result.get("roi_profile", "N/A"),
        "centimetres_per_pixel": result.get("centimetres_per_pixel"),
        "rectified": result.get("rectified", False),
        "total_tracks": len(all_tracks_flat) if has_batch else len(predictions)
    }

    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)

    # 1. Download PDF Report
    with ex_col1:
        pdf_bytes = _generate_pdf_report(
            detected_input_type,
            batch_results,
            selected_name,
            confidence_threshold,
            export_metadata,
            model_metadata,
            training_report
        )
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{Path(frame_name).stem}_svm_analysis_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    # 2. Download CSV Data
    with ex_col2:
        csv_bytes = _generate_csv_data(detected_input_type, batch_results, confidence_threshold, context)
        st.download_button(
            "📊 Download CSV Data",
            data=csv_bytes,
            file_name=f"{Path(frame_name).stem}_svm_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 3. Download JSON Data
    with ex_col3:
        json_bytes = _generate_json_data(detected_input_type, batch_results, confidence_threshold, export_metadata, model_metadata)
        st.download_button(
            "📈 Download JSON Data",
            data=json_bytes,
            file_name=f"{Path(frame_name).stem}_svm_predictions.json",
            mime="application/json",
            use_container_width=True
        )

    # 4. Download PNG Overlay Image
    with ex_col4:
        png_bytes = svm_module.encode_report_png(overlay_img)
        st.download_button(
            "🖼️ Download PNG Image",
            data=png_bytes,
            file_name=f"{Path(frame_name).stem}_svm_overlay.png",
            mime="image/png",
            use_container_width=True
        )


# ==============================================================================
# Helper Functions & Report Generators
# ==============================================================================

def _get_model_metadata(model_path: Path) -> dict:
    metadata = {
        "status": "Not Loaded",
        "kernel": "linear",
        "C": 1.0,
        "gamma": "scale",
        "train_date": "N/A"
    }
    if model_path.exists():
        metadata["status"] = "Loaded"
        try:
            mtime = model_path.stat().st_mtime
            metadata["train_date"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            bundle = joblib.load(model_path)
            params = bundle.get("selected_parameters", {})
            metadata["kernel"] = params.get("kernel", "linear")
            metadata["C"] = params.get("C", 1.0)
            svc = bundle["model"].named_steps["svm"]
            metadata["gamma"] = getattr(svc, "gamma", "scale")
        except Exception:
            pass
    return metadata


def _load_training_report(report_path: Path) -> dict | None:
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _parse_video_metadata(name: str, description: str) -> tuple[int | None, float | None]:
    frame_match = re.search(r"frame\s*(\d+)", description, re.IGNORECASE) or re.search(r"_frame_(\d+)", name, re.IGNORECASE)
    time_match = re.search(r"([\d\.]+)\s*seconds", description, re.IGNORECASE)
    frame_num = int(frame_match.group(1)) if frame_match else None
    timestamp = float(time_match.group(1)) if time_match else None
    return frame_num, timestamp


def _render_checklist_item(title: str, status: str) -> str:
    if status == "success":
        icon, color, bg = "✅", "#28a745", "rgba(40, 167, 69, 0.05)"
    elif status == "warning":
        icon, color, bg = "⏳", "#ffc107", "rgba(255, 193, 7, 0.05)"
    else:
        icon, color, bg = "❌", "#dc3545", "rgba(220, 53, 69, 0.05)"
    return f"""
    <div style="display: flex; align-items: center; padding: 8px 12px; border-radius: 6px; background-color: {bg}; border: 1px solid {color}44; margin-bottom: 8px; font-size: 13px;">
        <span style="font-size: 15px; margin-right: 8px;">{icon}</span>
        <span style="font-weight: 600; color: {color};">{title}</span>
    </div>
    """


def _associate_tracks(batch_results: dict, sorted_frames: list) -> list[dict]:
    global_tracks = []
    next_global_id = 1
    for f_name in sorted_frames:
        entry = batch_results[f_name]
        features = entry["result"]["features"]
        predictions = entry["predictions"]
        for track, pred in zip(features, predictions):
            x, y, w, h = track.bounding_box
            centroid = (x + w/2, y + h/2)
            best_gt = None
            min_dist = 60.0
            for gt in global_tracks:
                last_occ = gt["history"][-1]
                last_f_idx = sorted_frames.index(last_occ["frame"])
                curr_f_idx = sorted_frames.index(f_name)
                if 0 < curr_f_idx - last_f_idx <= 3:
                    dist = np.hypot(centroid[0] - last_occ["centroid"][0], centroid[1] - last_occ["centroid"][1])
                    if dist < min_dist:
                        min_dist = dist
                        best_gt = gt
            occ_data = {
                "frame": f_name,
                "track_id": track.track_id,
                "centroid": centroid,
                "class": pred["predicted_class"],
                "confidence": pred["confidence"],
                "bounding_box": track.bounding_box
            }
            if best_gt is not None:
                best_gt["history"].append(occ_data)
            else:
                global_tracks.append({"global_id": next_global_id, "history": [occ_data]})
                next_global_id += 1
    return global_tracks


def _draw_track_journey(image: np.ndarray, global_track: dict, active_frame_name: str, sorted_frames: list) -> np.ndarray:
    overlay = image.copy()
    history = global_track["history"]
    active_idx = -1
    for i, occ in enumerate(history):
        if occ["frame"] == active_frame_name:
            active_idx = i
            break
    for i in range(len(history)):
        cx, cy = int(history[i]["centroid"][0]), int(history[i]["centroid"][1])
        if history[i]["frame"] == active_frame_name:
            color = (0, 0, 255) # Red for active
            cv2.circle(overlay, (cx, cy), 6, color, -1)
            x, y, w, h = history[i]["bounding_box"]
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
            cv2.putText(overlay, f"GT-{global_track['global_id']}", (x, max(y-5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            color = (0, 255, 0) if (i < active_idx or active_idx == -1) else (0, 255, 255)
            cv2.circle(overlay, (cx, cy), 4, color, -1)
        if i > 0:
            prev_cx, prev_cy = int(history[i-1]["centroid"][0]), int(history[i-1]["centroid"][1])
            line_color = (0, 255, 0) if (i <= active_idx or active_idx == -1) else (0, 255, 255)
            cv2.line(overlay, (prev_cx, prev_cy), (cx, cy), line_color, 2)
    return overlay


def _generate_csv_data(input_type: str, batch_results: dict, threshold: float, context: PageContext) -> bytes:
    csv_rows = []
    for b_name, b_entry in batch_results.items():
        f_num, f_ts = _parse_video_metadata(b_name, b_entry["description"])
        cent_per_px = b_entry["result"].get("centimetres_per_pixel")
        for track, pred in zip(b_entry["result"]["features"], b_entry["predictions"]):
            feat_row = context.feature_row(track, cent_per_px)
            feat_row.pop("Track", None)

            probs = pred["probabilities"]
            sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            second_cls = sorted_p[1][0] if len(sorted_p) > 1 else "N/A"
            second_prob = sorted_p[1][1] if len(sorted_p) > 1 else 0.0
            margin = pred["confidence"] - second_prob

            row = {
                "Input Name": b_name,
                "Frame Number": f_num if f_num is not None else "",
                "Timestamp (s)": f_ts if f_ts is not None else "",
                "Track ID": track.track_id,
                "Predicted Particle": pred["particle_type"],
                "Confidence": pred["confidence"],
                "Status": "Accepted" if pred["confidence"] >= threshold else "Uncertain",
                "Second-Best Class": DISPLAY_NAMES.get(second_cls, second_cls),
                "Prediction Margin": margin,
                "Inference Time (ms)": pred["inference_time_ms"],
                **{f"P({DISPLAY_NAMES.get(n, n)})": p for n, p in probs.items()},
                **feat_row
            }
            csv_rows.append(row)
    return pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8-sig")


def _generate_json_data(input_type: str, batch_results: dict, threshold: float, metadata: dict, model_metadata: dict) -> bytes:
    json_data = {
        "metadata": metadata,
        "model_parameters": model_metadata,
        "batch_summary": {
            "samples_processed": len(batch_results),
            "total_tracks": sum(len(e["predictions"]) for e in batch_results.values())
        },
        "samples": []
    }
    for b_name, b_entry in batch_results.items():
        f_num, f_ts = _parse_video_metadata(b_name, b_entry["description"])
        json_data["samples"].append({
            "name": b_name,
            "frame_number": f_num,
            "timestamp": f_ts,
            "track_count": len(b_entry["predictions"]),
            "predictions": b_entry["predictions"]
        })
    return json.dumps(json_data, indent=2).encode("utf-8")


def _generate_pdf_report(
    input_type: str,
    batch_results: dict,
    active_name: str,
    threshold: float,
    metadata: dict,
    model_metadata: dict,
    training_report: dict | None
) -> bytes:
    """Generate professional, multi-page input-aware PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20, leading=24, textColor=colors.HexColor('#1a365d'), spaceAfter=10)
    h2_style = ParagraphStyle('Heading2Style', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=colors.HexColor('#2b6cb0'), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('BodyStyle', parent=styles['BodyText'], fontName='Helvetica', fontSize=7, leading=11, textColor=colors.HexColor('#2d3748'))
    bold_style = ParagraphStyle('BoldStyle', parent=body_style, fontName='Helvetica-Bold')

    # Title Banner
    story.append(Paragraph(f"SVM Cloud Chamber Classification Report ({input_type})", title_style))
    story.append(Spacer(1, 4))

    # Metadata Table
    scale_val = f"{metadata.get('centimetres_per_pixel'):.6f} cm/px" if metadata.get('centimetres_per_pixel') else "Pixels Only"
    meta_data = [
        [Paragraph("<b>Input Name:</b>", body_style), Paragraph(metadata.get('input_name', 'N/A'), body_style), Paragraph("<b>Generated At:</b>", body_style), Paragraph(metadata.get('generated_at', 'N/A'), body_style)],
        [Paragraph("<b>Input Type:</b>", body_style), Paragraph(input_type, body_style), Paragraph("<b>Reporting Threshold:</b>", body_style), Paragraph(f"{threshold:.0%}", body_style)],
        [Paragraph("<b>Model Architecture:</b>", body_style), Paragraph(f"SVC ({model_metadata.get('kernel')} kernel, C={model_metadata.get('C')})", body_style), Paragraph("<b>Spatial Scale:</b>", body_style), Paragraph(scale_val, body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[1.4*inch, 2.1*inch, 1.4*inch, 2.1*inch])
    meta_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f7fafc')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Executive Summary & KPIs
    all_tracks = []
    for b_entry in batch_results.values():
        for trk, prd in zip(b_entry["result"]["features"], b_entry["predictions"]):
            all_tracks.append(prd)

    total_cnt = len(all_tracks)
    accepted_cnt = sum(1 for p in all_tracks if p["confidence"] >= threshold)
    uncertain_cnt = total_cnt - accepted_cnt
    conf_list = [p["confidence"] for p in all_tracks]
    avg_conf = np.mean(conf_list) if conf_list else 0.0
    class_cnts = Counter(p["predicted_class"] for p in all_tracks)
    dom_cls = class_cnts.most_common(1)[0][0] if class_cnts else "None"

    story.append(Paragraph("1. Executive Classification Summary", h2_style))
    sum_data = [
        [Paragraph("<b>Total Inputs Processed:</b>", body_style), Paragraph(str(len(batch_results)), body_style), Paragraph("<b>Total Tracks Detected:</b>", body_style), Paragraph(str(total_cnt), body_style)],
        [Paragraph("<b>Accepted Classifications:</b>", body_style), Paragraph(f"{accepted_cnt} ({accepted_cnt/max(1,total_cnt):.1%})", body_style), Paragraph("<b>Uncertain Classifications:</b>", body_style), Paragraph(f"{uncertain_cnt} ({uncertain_cnt/max(1,total_cnt):.1%})", body_style)],
        [Paragraph("<b>Average Confidence:</b>", body_style), Paragraph(f"{avg_conf:.1%}", body_style), Paragraph("<b>Dominant Particle Class:</b>", body_style), Paragraph(DISPLAY_NAMES.get(dom_cls, dom_cls), body_style)]
    ]
    sum_table = Table(sum_data, colWidths=[1.6*inch, 1.9*inch, 1.6*inch, 1.9*inch])
    sum_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 10))

    # Annotated Image Visual Overlay
    story.append(Paragraph(f"2. Visual Classification Overlay ({active_name})", h2_style))
    active_entry = batch_results[active_name]
    overlay_img, _ = svm_module.build_visual_report(active_entry["result"]["input_image"], active_entry["result"]["features"], active_entry["predictions"], threshold)
    success, encoded = cv2.imencode(".png", overlay_img)
    if success:
        img_io = io.BytesIO(encoded.tobytes())
        rl_img = RLImage(img_io, width=6.5*inch, height=4.2*inch)
        story.append(rl_img)
    story.append(PageBreak())

    # Page 2: Class Distribution & Track Table
    story.append(Paragraph("3. Particle Class Distribution & Confidence Analysis", h2_style))

    # Draw Matplotlib bar chart for PDF
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    labels = [DISPLAY_NAMES.get(k, k) for k in class_cnts.keys()]
    values = list(class_cnts.values())
    bar_colors = [RGB_COLORS.get(k, "#888888") for k in class_cnts.keys()]
    ax.bar(labels, values, color=bar_colors, edgecolor='grey')
    ax.set_ylabel("Count")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    chart_io = io.BytesIO()
    plt.savefig(chart_io, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    chart_io.seek(0)
    story.append(RLImage(chart_io, width=6.2*inch, height=2.1*inch))
    story.append(Spacer(1, 10))

    # Track Details Table
    story.append(Paragraph("4. Track Detections & Class Probabilities", h2_style))
    tbl_headers = [Paragraph("<b>Track ID</b>", bold_style), Paragraph("<b>Prediction</b>", bold_style), Paragraph("<b>Confidence</b>", bold_style), Paragraph("<b>Status</b>", bold_style), Paragraph("<b>Margin</b>", bold_style), Paragraph("<b>Probabilities</b>", bold_style)]
    tbl_rows = [tbl_headers]

    for p in active_entry["predictions"][:25]: # limit to 25 rows per page section
        probs = p["probabilities"]
        sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        second_p = sorted_p[1][1] if len(sorted_p) > 1 else 0.0
        margin = p["confidence"] - second_p
        prob_s = ", ".join([f"{c[:3].capitalize()}:{v:.0%}" for c, v in sorted_p])

        tbl_rows.append([
            Paragraph(f"T{p['track_id']}", body_style),
            Paragraph(p["particle_type"], body_style),
            Paragraph(f"{p['confidence']:.1%}", body_style),
            Paragraph("Accepted" if p["confidence"] >= threshold else "Uncertain", body_style),
            Paragraph(f"{margin:.1%}", body_style),
            Paragraph(prob_s, body_style)
        ])

    track_table = Table(tbl_rows, colWidths=[0.7*inch, 1.3*inch, 0.8*inch, 0.8*inch, 0.7*inch, 2.7*inch], repeatRows=1)
    track_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#edf2f7')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(track_table)
    story.append(PageBreak())

    # Page 3: Model Evaluation Performance & Methodology
    story.append(Paragraph("5. SVM Model Evaluation & Performance Metrics", h2_style))
    if training_report and "segmented_final_test" in training_report:
        seg_test = training_report["segmented_final_test"]
        perf_data = [
            [Paragraph("<b>Test Accuracy:</b>", body_style), Paragraph(f"{seg_test.get('accuracy', 0):.1%}", body_style), Paragraph("<b>Balanced Accuracy:</b>", body_style), Paragraph(f"{seg_test.get('balanced_accuracy', 0):.1%}", body_style)],
            [Paragraph("<b>Macro F1-Score:</b>", body_style), Paragraph(f"{seg_test.get('macro_f1', 0):.1%}", body_style), Paragraph("<b>Weighted F1-Score:</b>", body_style), Paragraph(f"{seg_test.get('weighted_f1', 0):.1%}", body_style)]
        ]
        perf_table = Table(perf_data, colWidths=[1.6*inch, 1.9*inch, 1.6*inch, 1.9*inch])
        perf_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f7fafc')),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(perf_table)
        story.append(Spacer(1, 10))

    # Methodology Section
    story.append(Paragraph("6. Methodology & Preprocessing Specifications", h2_style))
    method_text = f"""
    <b>Model Pipeline:</b> scikit-learn Pipeline with RobustScaler and SVC.<br/>
    <b>Trained Kernel:</b> {model_metadata.get('kernel')} (C={model_metadata.get('C')}, gamma={model_metadata.get('gamma')}).<br/>
    <b>Feature Set:</b> 11 numerical features (area, perimeter, major axis, mean width, aspect ratio, solidity, rectangularity, thickness, mean intensity, line density, tortuosity).<br/>
    <b>Proton Prediction Rule:</b> Enforces P(Proton) &gt; 0.75 for proton assignment.<br/>
    <b>Reporting Threshold:</b> {threshold:.0%} (visual distinction between Accepted and Uncertain status).
    """
    story.append(Paragraph(method_text, body_style))

    doc.build(story)
    return buffer.getvalue()
