"""SVM Streamlit page."""
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

import cloud_chamber.ml.member_models.svm as svm_module
from cloud_chamber.config import load_config as _load_svm_config
from .context import PageContext

FEATURE_COLUMNS = svm_module.FEATURE_COLUMNS
CLASS_COLOURS = svm_module.CLASS_COLOURS
DISPLAY_NAMES = svm_module.DISPLAY_NAMES

# RGB colors corresponding to BGR CLASS_COLOURS
RGB_COLORS = {
    "alpha": "#FFA500",            # BGR: (0, 165, 255) -> RGB: (255, 165, 0)
    "electron_positron": "#0078FF", # BGR: (255, 120, 0) -> RGB: (0, 120, 255)
    "proton": "#00C800",           # BGR: (0, 200, 0) -> RGB: (0, 200, 0)
    "v_track": "#FF00FF"           # Bright magenta improves visibility on dark chamber images.
}


def render(context: PageContext) -> None:
    # Inject Custom Styling for Premium Look
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
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        margin: 8px 0 0 0;
        font-size: 24px;
        font-weight: bold;
    }
    .tag-chip {
        background-color: rgba(0, 120, 255, 0.1);
        color: #0078ff;
        padding: 4px 10px;
        border-radius: 12px;
        margin: 4px;
        display: inline-block;
        font-size: 11px;
        font-family: monospace;
        border: 1px solid rgba(0, 120, 255, 0.3);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .tag-chip:hover {
        background-color: rgba(0, 120, 255, 0.2);
        border-color: #0078ff;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
    }
    .status-badge-loaded {
        background-color: rgba(40, 167, 69, 0.2);
        color: #28a745;
        border: 1px solid #28a745;
    }
    .status-badge-notloaded {
        background-color: rgba(220, 53, 69, 0.2);
        color: #dc3545;
        border: 1px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("SVM Track Classifier")

    # Initialize state
    st.session_state.setdefault("svm_batch_results", {})
    st.session_state.setdefault("svm_session_info", None)
    st.session_state.setdefault("svm_predictions", None)

    # 1. Load Model and Parse Metadata
    model_path = Path("models/svm_classifier.joblib")
    model_metadata = _get_model_metadata(model_path)

    # Abort if no pipeline results are loaded
    result = st.session_state.get("pipeline_result")
    if result is None:
        st.info("💡 Process an image or video frame on the **Shared Processing Pipeline** page first to begin.")
        return

    # Load Model Bundle
    if not model_path.exists():
        return
    model_bundle = svm_module.load_model(model_path)

    # 2. Input Preview & Checklist Layout (2 columns)
    st.markdown("---")
    col_preview, col_checklist = st.columns([3, 2])
    
    with col_preview:
        st.subheader("🖼️ Input Frame Preview")
        # Draw contour boundaries
        contours = result["segmentation"].contours
        active_image = result["input_image"].copy()
        
        # Color contours in cyan
        cv2.drawContours(active_image, contours, -1, (255, 255, 0), 2)
        st.image(context.bgr_to_rgb(active_image), width=300, caption="Selected frame with active contour boundaries", use_container_width=True)
        
        # Frame Details
        frame_name = st.session_state.get("input_name", "N/A")
        frame_desc = st.session_state.get("source_description", "N/A")
        num_tracks = len(contours)
        
        details_html = f"""
        <div style="background-color: rgba(128,128,128,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.15); margin-top: 10px; font-size: 13px;">
            <b>Active Frame Name:</b> {frame_name}<br>
            <b>Tracks Detected:</b> {num_tracks}<br>
            <b>Source:</b> {frame_desc}
        """
        frame_num, timestamp = _parse_video_metadata(frame_name, frame_desc)
        if frame_num is not None:
            details_html += f"<br><b>Frame Number:</b> {frame_num}"
        if timestamp is not None:
            details_html += f"<br><b>Video Timestamp:</b> {timestamp:.2f} seconds"
        details_html += "</div>"
        st.markdown(details_html, unsafe_allow_html=True)

    with col_checklist:
        st.subheader("📋 Pre-Classification Checklist")
        
        img_uploaded = st.session_state.get("input_image") is not None
        st.markdown(_render_checklist_item("Image/Video Uploaded", "success" if img_uploaded else "danger"), unsafe_allow_html=True)
        
        st.markdown(_render_checklist_item("Preprocessing Complete", "success" if result else "danger"), unsafe_allow_html=True)
        
        has_contours = len(contours) > 0
        st.markdown(_render_checklist_item("Contours Detected", "success" if has_contours else "warning"), unsafe_allow_html=True)
        
        has_features = len(result.get("features", [])) > 0
        st.markdown(_render_checklist_item("Features Extracted", "success" if has_features else "danger"), unsafe_allow_html=True)
        
        st.markdown(_render_checklist_item("Model Loaded Successfully", "success" if model_metadata["status"] == "Loaded" else "danger"), unsafe_allow_html=True)

    # 3. Classification Controls
    st.markdown("---")
    st.subheader("⚙️ Classification Scope & Controls")
    
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Predictions below this probability threshold will be marked as 'Uncertain' in reports."
        )
        
        # Scope selection
        has_batch = len(st.session_state.get("input_batch", [])) > 1
        is_video = any("_frame_" in sample["name"] for sample in st.session_state.get("input_batch", []))
        
        scope_options = ["Active Frame Only"]
        if has_batch:
            scope_options.append("Entire Batch/Video")
            
        scope_choice = st.radio("Classification Scope", scope_options, horizontal=True)

    with ctrl_col2:
        st.markdown("**Visualization & Output Toggles**")
        show_probs = st.checkbox("Show probability distribution for each track", value=True)
        show_features_contrib = st.checkbox("Show feature importance/contributions", value=True)
        export_csv_toggle = st.checkbox("Export detailed features to CSV", value=True)
        
        temporal_analysis = False
        frame_rate = 10
        if is_video and scope_choice == "Entire Batch/Video":
            st.markdown("**Video Temporal Settings**")
            temporal_analysis = st.toggle("Enable Temporal Trajectory Association", value=True)
            frame_rate = st.selectbox("Video Frame Sampling Rate", [1, 5, 10, 30], index=2, format_func=lambda x: f"{x} fps")

    # Start SVM Button
    if scope_choice == "Entire Batch/Video" and has_batch:
        prereqs_met = model_metadata["status"] == "Loaded" and has_batch
    else:
        prereqs_met = img_uploaded and result is not None and has_features and model_metadata["status"] == "Loaded"
    
    st.write("")
    if st.button("▶ Start SVM Classification", type="primary", disabled=not prereqs_met, use_container_width=True):
        samples = st.session_state.get("input_batch") if (scope_choice == "Entire Batch/Video" and has_batch) else [
            {
                "image": st.session_state["input_image"],
                "name": st.session_state["input_name"],
                "description": st.session_state["source_description"]
            }
        ]
        
        # Batch Predict Loop
        batch_results = {}
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        cancel_placeholder = st.empty()
        
        # cancellation reset
        st.session_state["cancel_processing"] = False
        
        start_time = perf_counter()
        total_tracks = 0
        
        for index, sample in enumerate(samples):
            # Check cancel click (via session state)
            if st.session_state.get("cancel_processing", False):
                break
                
            _config = st.session_state.get("config") or _load_svm_config()
            sample_result = context.process_pipeline_image(
                sample["image"],
                config=_config,
                calibration_settings=st.session_state.get("calibration_settings")
            )
            
            sample_predictions = _predict_tracks_svm_internal(model_bundle, sample_result["features"])
            
            batch_results[sample["name"]] = {
                "image": sample["image"],
                "analysis_image": sample_result["input_image"],
                "description": sample["description"],
                "result": sample_result,
                "predictions": sample_predictions,
            }
            
            total_tracks += len(sample_predictions)
            elapsed = perf_counter() - start_time
            avg_track_ms = (elapsed * 1000.0) / max(total_tracks, 1)
            
            # Estimate time remaining
            processed = index + 1
            est_remaining = (elapsed / processed) * (len(samples) - processed)
            
            progress_bar.progress(processed / len(samples))
            status_text.markdown(f"""
            ⏳ **Classifying Frame {processed} of {len(samples)}...**  
            Tracks processed: `{total_tracks}` | Avg speed: `{avg_track_ms:.1f} ms/track` | Est. Remaining: `{est_remaining:.1f}s`
            """)
            
            # Draw Cancel Button
            if cancel_placeholder.button("🔴 Cancel Batch Classification", key=f"cancel_btn_{index}"):
                st.session_state["cancel_processing"] = True
                st.rerun()
                
        # Clean progress bars
        progress_bar.empty()
        status_text.empty()
        cancel_placeholder.empty()
        
        if st.session_state.get("cancel_processing", False):
            st.warning("Classification was cancelled before completion.")
            st.session_state["cancel_processing"] = False
        else:
            # Save batch results
            st.session_state["svm_batch_results"] = batch_results
            
            # Sync active predictions
            active_frame_name = st.session_state["input_name"]
            if active_frame_name in batch_results:
                st.session_state["svm_predictions"] = batch_results[active_frame_name]["predictions"]
            else:
                st.session_state["svm_predictions"] = batch_results[list(batch_results.keys())[0]]["predictions"]
            
            import uuid as _uuid
            session_id = f"SVC-{datetime.now().strftime('%Y%m%d-%H%M')}"
            elapsed_total = perf_counter() - start_time
            st.session_state["svm_session_info"] = {
                "session_id": session_id,
                "input_type": "Video" if is_video else "Image",
                "frame_count": len(batch_results),
                "processing_time_s": elapsed_total,
                "total_tracks": total_tracks,
                "kernel": model_metadata["kernel"],
                "C": model_metadata["C"],
                "gamma": model_metadata["gamma"],
            }
            st.success(f"Successfully classified {total_tracks} tracks across {len(batch_results)} frame(s)!")

    # Check if we have results available
    batch_results = {k: v for k, v in st.session_state.get("svm_batch_results", {}).items() if k is not None}
    predictions = st.session_state.get("svm_predictions")

    if not batch_results:
        return

    # --- Batch overview expanders (one per frame, like MLP page) ---
    st.markdown("---")
    st.header("📊 Batch Classification Overview")

    for index, (name, entry) in enumerate(batch_results.items(), start=1):
        entry_preds = entry["predictions"]
        entry_counts = Counter(p["predicted_class"] for p in entry_preds)
        dominant = DISPLAY_NAMES.get(entry_counts.most_common(1)[0][0], "None") if entry_counts else "None"
        with st.expander(
            f"Frame {index}: {name} — {len(entry_preds)} tracks — dominant: {dominant}",
            expanded=False,
        ):
            if entry_preds:
                entry_overlay, _ = svm_module.build_visual_report(
                    entry["analysis_image"],
                    entry["result"]["features"],
                    entry_preds,
                    confidence_threshold,
                )
                st.image(context.bgr_to_rgb(entry_overlay), use_container_width=True)
            else:
                st.warning("No tracks detected in this frame.")

    # --- Select one frame for detailed report ---
    batch_keys = list(batch_results.keys())
    selected_name = st.selectbox(
        "Choose a frame for detailed analysis",
        range(len(batch_keys)),
        index=batch_keys.index(st.session_state.get("input_name")) if st.session_state.get("input_name") in batch_keys else 0,
        format_func=lambda i: f"Frame {i+1}: {batch_keys[i]}",
    )
    selected_name = batch_keys[selected_name]
    selected_entry = batch_results[selected_name]

    # Sync session state to selected frame
    st.session_state["input_image"] = selected_entry["image"]
    st.session_state["input_name"] = selected_name
    st.session_state["source_description"] = selected_entry["description"]
    st.session_state["pipeline_result"] = selected_entry["result"]
    st.session_state["svm_predictions"] = selected_entry["predictions"]

    result = selected_entry["result"]
    predictions = selected_entry["predictions"]
    frame_name = selected_name
    frame_desc = selected_entry["description"]

    if not predictions:
        st.warning("No tracks detected in the selected frame.")
        return
    st.markdown("---")
    st.header("📊 Classification Results & Analytics")
    
    # Calculate stats
    total_tracks_analyzed = len(predictions)
    conf_threshold = confidence_threshold
    
    confidences = [p["confidence"] for p in predictions]
    avg_conf = np.mean(confidences) if confidences else 0.0
    
    high_conf = sum(1 for c in confidences if c >= 0.80)
    med_conf = sum(1 for c in confidences if 0.50 <= c < 0.80)
    low_conf = sum(1 for c in confidences if c < 0.50)
    
    class_counts = Counter(p["predicted_class"] for p in predictions)
    dominant_class = class_counts.most_common(1)[0][0] if class_counts else "None"
    
    # 5. Summary Statistics Card
    with st.container(border=True):
        st.subheader("📈 Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tracks Analyzed", total_tracks_analyzed)
        c2.metric("Average Confidence", f"{avg_conf:.1%}")
        c3.metric("Dominant Prediction", DISPLAY_NAMES.get(dominant_class, dominant_class))
        
        # Confidence breakdown text
        c4.markdown(f"""
        **Confidence Groups:**
        - High (≥80%): `{high_conf}`
        - Medium (50%-80%): `{med_conf}`
        - Low (<50%): `{low_conf}`
        """)
        
    # Class Distribution & Confidence Visualizations
    st.subheader("📉 Visual Analytics Dashboard")
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        # Plotly bar chart
        bar_data = pd.DataFrame([
            {"Particle Class": DISPLAY_NAMES.get(cls, cls), "Count": count}
            for cls, count in class_counts.items()
        ])
        
        if not bar_data.empty:
            fig_bar = px.bar(
                bar_data,
                x="Particle Class",
                y="Count",
                color="Particle Class",
                color_discrete_map={
                    "Alpha": RGB_COLORS["alpha"],
                    "Electron/Positron": RGB_COLORS["electron_positron"],
                    "Proton": RGB_COLORS["proton"],
                    "V-track": RGB_COLORS["v_track"]
                },
                title="Class Counts distribution"
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No track classes to plot.")
            
    with vis_col2:
        # Confidence Histogram
        if confidences:
            fig_hist = px.histogram(
                pd.DataFrame({"Confidence": confidences}),
                x="Confidence",
                nbins=15,
                title="Confidence Scores Distribution"
            )
            fig_hist.add_vline(x=conf_threshold, line_width=3, line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig_hist, use_container_width=True)

    # 6. Video-Specific Timeline & Analysis (if batch is video)
    if is_video and batch_results:
        st.markdown("---")
        st.header("🎞️ Video Timeline & Journey Analysis")
        
        sorted_frames = sorted(list(batch_results.keys()))
        
        # Timeline scrubbing slider
        active_idx = sorted_frames.index(frame_name) if frame_name in sorted_frames else 0
        selected_idx = st.slider("Frame Scrubbing Timeline", 0, len(sorted_frames) - 1, active_idx)
        selected_frame_name = sorted_frames[selected_idx]
        
        # If scrubbing changed, update input and rerun
        if selected_frame_name != frame_name:
            st.session_state["input_image"] = batch_results[selected_frame_name]["image"]
            st.session_state["input_name"] = selected_frame_name
            st.session_state["source_description"] = batch_results[selected_frame_name]["description"]
            st.session_state["pipeline_result"] = batch_results[selected_frame_name]["result"]
            st.session_state["svm_predictions"] = batch_results[selected_frame_name]["predictions"]
            st.rerun()
            
        # Evolution charts over time
        evol_data = []
        for f_name in sorted_frames:
            f_entry = batch_results[f_name]
            f_counts = Counter(p["predicted_class"] for p in f_entry["predictions"])
            f_num, f_ts = _parse_video_metadata(f_name, f_entry["description"])
            
            row = {"Frame": f_num if f_num is not None else f_name, "Timestamp": f_ts if f_ts is not None else 0.0}
            for cls in ["alpha", "electron_positron", "proton", "v_track"]:
                row[DISPLAY_NAMES[cls]] = f_counts.get(cls, 0)
            evol_data.append(row)
            
        evol_df = pd.DataFrame(evol_data)
        
        # Line chart
        fig_line = px.line(
            evol_df,
            x="Timestamp" if "Timestamp" in evol_df.columns else "Frame",
            y=["Alpha", "Electron/Positron", "Proton", "V-track"],
            color_discrete_map={
                "Alpha": RGB_COLORS["alpha"],
                "Electron/Positron": RGB_COLORS["electron_positron"],
                "Proton": RGB_COLORS["proton"],
                "V-track": RGB_COLORS["v_track"]
            },
            title="Temporal Particle Count Evolution"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Track Journey Overlay
        if temporal_analysis:
            global_tracks = _associate_tracks(batch_results, sorted_frames)
            multi_frame_tracks = [gt for gt in global_tracks if len(gt["history"]) > 1]
            
            if multi_frame_tracks:
                st.markdown("##### Multi-Frame Track Centroid Trajectory Overlay")
                gt_selectbox_ids = [gt["global_id"] for gt in multi_frame_tracks]
                sel_gt_id = st.selectbox("Select a Track to Trace across Frames", gt_selectbox_ids)
                
                selected_gt = next(gt for gt in global_tracks if gt["global_id"] == sel_gt_id)
                journey_img = _draw_track_journey(result["input_image"], selected_gt, frame_name, sorted_frames)
                st.image(context.bgr_to_rgb(journey_img), caption=f"Centroid Trajectory History of Global Track {sel_gt_id}", use_container_width=True)
                
                # Show evolution history details
                evol_details = []
                for occ in selected_gt["history"]:
                    f_num, _ = _parse_video_metadata(occ["frame"], "")
                    evol_details.append({
                        "Frame": f"Frame {f_num}" if f_num is not None else occ["frame"],
                        "Class Prediction": DISPLAY_NAMES.get(occ["class"], occ["class"]),
                        "Confidence": f"{occ['confidence']:.1%}"
                    })
                st.dataframe(pd.DataFrame(evol_details), use_container_width=True, hide_index=True)
            else:
                st.info("No multi-frame tracks associated. Try extracting video frames closer together.")

    # 7. Track Results Table
    st.markdown("---")
    st.subheader("📋 Track Detections & Predictions")

    table_data = []
    for track, pred in zip(result["features"], predictions):
        prob_str = ", ".join([
            f"{DISPLAY_NAMES.get(cls, cls)}: {prob:.1%}"
            for cls, prob in sorted(pred["probabilities"].items(), key=lambda x: x[1], reverse=True)
        ])
        table_data.append({
            "ID": f"T{track.track_id}",
            "Predicted Particle": DISPLAY_NAMES.get(pred["predicted_class"], pred["predicted_class"]),
            "Confidence": f"{pred['confidence']:.1%}",
            "Status": "Accepted" if pred["confidence"] >= conf_threshold else "Uncertain",
            "Class Probabilities": prob_str,
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    
    # 8. Individual Track Details (Inspector)
    st.write("")
    st.subheader("🔍 Individual Track Inspector")
    
    track_ids = [track.track_id for track in result["features"]]
    selected_track_id = st.selectbox("Choose a Track ID to deep-dive", track_ids)
    
    selected_idx = track_ids.index(selected_track_id)
    selected_track_features = result["features"][selected_idx]
    selected_pred = predictions[selected_idx]
    
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        # Probability chart
        prob_df = pd.DataFrame([
            {"Class": DISPLAY_NAMES.get(cls, cls), "Probability": prob}
            for cls, prob in selected_pred["probabilities"].items()
        ]).sort_values("Probability", ascending=True)
        
        fig_prob = px.bar(
            prob_df,
            x="Probability",
            y="Class",
            orientation="h",
            color="Class",
            color_discrete_map={
                "Alpha": RGB_COLORS["alpha"],
                "Electron/Positron": RGB_COLORS["electron_positron"],
                "Proton": RGB_COLORS["proton"],
                "V-track": RGB_COLORS["v_track"]
            },
            title=f"T{selected_track_id} Class Probabilities"
        )
        fig_prob.update_layout(showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Trajectory Crop Visual
        st.markdown("**Trajectory Crop Visual**")
        x, y, w, h = selected_track_features.bounding_box
        pad = 20
        h_img, w_img = result["input_image"].shape[:2]
        y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
        x1, x2 = max(0, x - pad), min(w_img, x + w + pad)

        cropped_img = result["input_image"][y1:y2, x1:x2].copy()

        current_contours = result["segmentation"].contours
        if selected_idx < len(current_contours):
            selected_contour = current_contours[selected_idx]
            shifted_contour = selected_contour - np.array([x1, y1])
            cv2.drawContours(cropped_img, [shifted_contour], -1, (255, 255, 0), 2)

        st.image(context.bgr_to_rgb(cropped_img), caption=f"Cropped track outline (T{selected_track_id})", width=260)
        
    with detail_col2:
        # Z-Scores Feature Profile Explanation
        if show_features_contrib and "scaler" in model_bundle["model"].named_steps:
            scaler = model_bundle["model"].named_steps["scaler"]
            feature_dict = asdict(selected_track_features)
            feature_values = [feature_dict[col] for col in FEATURE_COLUMNS]
            z_scores = (np.array(feature_values) - scaler.mean_) / scaler.scale_
            
            z_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Z-score": z_scores
            }).sort_values("Z-score", ascending=True)
            
            fig_z = px.bar(
                z_df,
                x="Z-score",
                y="Feature",
                orientation="h",
                title=f"Feature Z-Score Profile (deviation from training mean)"
            )
            st.plotly_chart(fig_z, use_container_width=True)
            st.caption("Positive/Negative Z-scores indicate how many standard deviations the feature value lies above or below the training average.")

    # 9. Report Generation & Export Section
    st.markdown("---")
    st.header("📤 Report Export Center")
    
    # Re-draw clean overlay for export
    overlay_export, report_rows_export = svm_module.build_visual_report(
        result["input_image"],
        result["features"],
        predictions,
        conf_threshold
    )
    
    # Complete Metadata for JSON/PDF
    metadata = {
        "input_name": frame_name,
        "source_description": frame_desc,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_threshold": conf_threshold,
        "roi_profile": result.get("roi_profile", "N/A"),
        "centimetres_per_pixel": result.get("centimetres_per_pixel"),
        "rectified": result.get("rectified", False)
    }
    
    summary_stats = {
        "detected_contours": total_tracks_analyzed,
        "confident_classifications": high_conf + med_conf,
        "uncertain_classifications": low_conf,
        "dominant_prediction": DISPLAY_NAMES.get(dominant_class, dominant_class),
        "processing_time_ms": float(result["segmentation"].processing_time_ms)
    }
    
    # Layout exports buttons
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    
    # 1. PDF export
    with ex_col1:
        pdf_bytes = _generate_pdf_report(overlay_export, predictions, summary_stats, metadata)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{Path(frame_name).stem}_svm_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
    # 2. CSV export
    with ex_col2:
        # Create CSV rows combining features and predictions
        detailed_csv_rows = []
        for track, pred in zip(result["features"], predictions):
            feat_row = context.feature_row(track, result["centimetres_per_pixel"])
            feat_row.pop("Track", None) # remove duplicate ID
            
            row = {
                "Track ID": track.track_id,
                "Predicted Particle": pred["particle_type"],
                "Confidence": pred["confidence"],
                "Status": "Accepted" if pred["confidence"] >= conf_threshold else "Uncertain",
                "Inference Time (ms)": pred["inference_time_ms"],
                **{f"P({DISPLAY_NAMES.get(name, name)})": prob for name, prob in pred["probabilities"].items()},
                **feat_row
            }
            detailed_csv_rows.append(row)
            
        csv_bytes = pd.DataFrame(detailed_csv_rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📊 Download CSV Data",
            data=csv_bytes,
            file_name=f"{Path(frame_name).stem}_svm_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # 3. JSON export
    with ex_col3:
        json_export_data = {
            "metadata": metadata,
            "summary": summary_stats,
            "predictions": predictions
        }
        json_bytes = json.dumps(json_export_data, indent=2).encode("utf-8")
        st.download_button(
            "📈 Download JSON Data",
            data=json_bytes,
            file_name=f"{Path(frame_name).stem}_svm_predictions.json",
            mime="application/json",
            use_container_width=True
        )
        
    # 4. PNG Overlay export
    with ex_col4:
        png_bytes = svm_module.encode_report_png(overlay_export)
        st.download_button(
            "🖼️ Download PNG Image",
            data=png_bytes,
            file_name=f"{Path(frame_name).stem}_svm_overlay.png",
            mime="image/png",
            use_container_width=True
        )

    # Preview report in UI
    st.markdown("##### 📄 PDF Report Preview")
    with st.container(height=350, border=True):
        st.markdown(f"""
        ### SVM Particle Classification Summary Report
        **Source Frame:** `{metadata['input_name']}`  
        **Generated At:** `{metadata['generated_at']}`  
        
        #### Executive Summary
        * Total Tracks Analyzed: `{summary_stats['detected_contours']}`
        * Confident Classifications: `{high_conf}` (confidence ≥ 80%)
        * Uncertain Classifications: `{low_conf}` (confidence < 50%)
        * Dominant Particle Class: **{summary_stats['dominant_prediction']}**
        
        #### Active Feature Extraction Contract
        * Dimensions: aspect ratio, solidity, rectangularity, thickness
        * Scale calibration: {f"{metadata['centimetres_per_pixel']:.6f} cm/px" if metadata['centimetres_per_pixel'] else "Pixels (Uncalibrated)"}
        """)


# ==============================================================================
# Helper Functions
# ==============================================================================

def _get_model_metadata(model_path: Path) -> dict:
    metadata = {
        "status": "Not Loaded",
        "kernel": "N/A",
        "C": "N/A",
        "gamma": "N/A",
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "class_counts": {},
        "train_date": "N/A"
    }
    
    if model_path.exists():
        metadata["status"] = "Loaded"
        try:
            mtime = model_path.stat().st_mtime
            metadata["train_date"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            bundle = joblib.load(model_path)
            selected_params = bundle.get("selected_parameters", {})
            metadata["kernel"] = selected_params.get("kernel", "linear")
            metadata["C"] = selected_params.get("C", 1.0)
            
            # Extract validation metrics
            val_metrics = bundle.get("validation_metrics", {})
            metadata["accuracy"] = val_metrics.get("accuracy")
            metadata["balanced_accuracy"] = val_metrics.get("balanced_accuracy")
            metadata["macro_f1"] = val_metrics.get("macro_f1")
            
            svc = bundle["model"].named_steps["svm"]
            metadata["gamma"] = getattr(svc, "gamma", "scale")
            
            # Load class counts from report if available
            report_path = model_path.with_name("svm_training_report.json")
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                    metadata["class_counts"] = report.get("class_counts", {}).get("validation", {})
        except Exception:
            pass
            
    return metadata


def _parse_video_metadata(name: str, description: str) -> tuple[int | None, float | None]:
    # Search for frame number and timestamp in description or name
    frame_match = re.search(r"frame\s*(\d+)", description, re.IGNORECASE)
    if not frame_match:
        frame_match = re.search(r"_frame_(\d+)", name, re.IGNORECASE)
        
    time_match = re.search(r"([\d\.]+)\s*seconds", description, re.IGNORECASE)
    
    frame_num = int(frame_match.group(1)) if frame_match else None
    timestamp = float(time_match.group(1)) if time_match else None
    
    return frame_num, timestamp


def _render_checklist_item(title: str, status: str) -> str:
    if status == "success":
        icon = "✅"
        color = "#28a745"
        bg = "rgba(40, 167, 69, 0.05)"
    elif status == "warning":
        icon = "⏳"
        color = "#ffc107"
        bg = "rgba(255, 193, 7, 0.05)"
    elif status == "danger":
        icon = "❌"
        color = "#dc3545"
        bg = "rgba(220, 53, 69, 0.05)"
    else:
        icon = "⬜"
        color = "#6c757d"
        bg = "rgba(108, 117, 125, 0.05)"
        
    return f"""
    <div style="display: flex; align-items: center; padding: 10px 14px; border-radius: 8px; background-color: {bg}; border: 1px solid {color}55; margin-bottom: 10px; font-family: sans-serif; font-size: 13px;">
        <span style="font-size: 16px; margin-right: 10px; vertical-align: middle;">{icon}</span>
        <span style="font-weight: 600; color: {color}; vertical-align: middle;">{title}</span>
    </div>
    """


def _predict_tracks_svm_internal(model_bundle: dict, features: list) -> list[dict]:
    matrix = svm_module.features_to_matrix(features)
    if matrix.shape[0] == 0:
        return []
        
    model = model_bundle["model"]
    started = perf_counter()
    
    probabilities = model.predict_proba(matrix)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    elapsed_per_track = (perf_counter() - started) * 1000.0 / len(matrix)
    
    return [
        {
            "track_id": track.track_id,
            "predicted_class": str(label),
            "particle_type": DISPLAY_NAMES.get(str(label), str(label)),
            "confidence": float(np.max(probability)),
            "inference_time_ms": elapsed_per_track,
            "probabilities": {
                str(class_name): float(class_probability)
                for class_name, class_probability in zip(
                    model.classes_, probability, strict=True
                )
            },
        }
        for track, label, probability in zip(
            features, predictions, probabilities, strict=True
        )
    ]


def _get_class_badge_html(class_name: str) -> str:
    key = class_name.lower().replace("/", "_").replace("-", "_")
    color = RGB_COLORS.get(key, "#888888")
    bg = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)"
    display = DISPLAY_NAMES.get(key, class_name)
    return f'<span class="status-badge" style="background-color: {bg}; color: {color}; border: 1px solid {color};">{display}</span>'


def _get_status_badge_html(status: str) -> str:
    if status == "Accepted":
        return '<span class="status-badge" style="background-color: rgba(40, 167, 69, 0.15); color: #28a745; border: 1px solid #28a745;">Accepted</span>'
    return '<span class="status-badge" style="background-color: rgba(255, 193, 7, 0.15); color: #ffc107; border: 1px solid #ffc107;">Uncertain</span>'


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
            
            # Find matching active track in previous frames
            best_gt = None
            min_dist = 60.0 # Pixel distance threshold
            
            for gt in global_tracks:
                last_occ = gt["history"][-1]
                last_frame_idx = sorted_frames.index(last_occ["frame"])
                curr_frame_idx = sorted_frames.index(f_name)
                
                # Look back up to 3 frames
                if 0 < curr_frame_idx - last_frame_idx <= 3:
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
                global_tracks.append({
                    "global_id": next_global_id,
                    "history": [occ_data]
                })
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
            
    # Draw trajectory line and points
    for i in range(len(history)):
        cx, cy = int(history[i]["centroid"][0]), int(history[i]["centroid"][1])
        
        # Color: active frame is Red, past is Green, future is Yellow
        if history[i]["frame"] == active_frame_name:
            color = (0, 0, 255) # BGR Red
            cv2.circle(overlay, (cx, cy), 6, color, -1)
            # bounding box
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


def _generate_pdf_report(overlay_bgr: np.ndarray, predictions: list, summary: dict, metadata: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=15
    )
    heading2_style = ParagraphStyle(
        'Heading2Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=12,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2d3748')
    )
    bold_body_style = ParagraphStyle(
        'BoldBodyStyle',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph("SVM Cloud Chamber Classification Report", title_style))
    story.append(Spacer(1, 10))
    
    # Metadata Table
    scale_val = f"{metadata.get('centimetres_per_pixel'):.6f} cm/px" if metadata.get('centimetres_per_pixel') else "Pixels Only"
    meta_data = [
        [Paragraph("<b>File Name:</b>", body_style), Paragraph(metadata.get('input_name', 'N/A'), body_style),
         Paragraph("<b>Generated At:</b>", body_style), Paragraph(metadata.get('generated_at', 'N/A'), body_style)],
        [Paragraph("<b>Model Name:</b>", body_style), Paragraph("Support Vector Machine (SVC)", body_style),
         Paragraph("<b>Confidence Threshold:</b>", body_style), Paragraph(f"{metadata.get('confidence_threshold', 0.5):.2f}", body_style)],
        [Paragraph("<b>ROI Profile:</b>", body_style), Paragraph(metadata.get('roi_profile', 'N/A'), body_style),
         Paragraph("<b>Spatial scale:</b>", body_style), Paragraph(scale_val, body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
    meta_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f7fafc')),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # Summary Statistics
    story.append(Paragraph("Summary Statistics", heading2_style))
    summary_data = [
        [Paragraph("<b>Total Tracks Analyzed:</b>", body_style), Paragraph(str(summary.get('detected_contours', 0)), body_style)],
        [Paragraph("<b>Confident Tracks:</b>", body_style), Paragraph(str(summary.get('confident_classifications', 0)), body_style)],
        [Paragraph("<b>Uncertain Tracks:</b>", body_style), Paragraph(str(summary.get('uncertain_classifications', 0)), body_style)],
        [Paragraph("<b>Dominant Class:</b>", body_style), Paragraph(summary.get('dominant_prediction', 'N/A'), body_style)],
        [Paragraph("<b>Processing Time:</b>", body_style), Paragraph(f"{summary.get('processing_time_ms', 0):.1f} ms", body_style)]
    ]
    summary_table = Table(summary_data, colWidths=[2.5*inch, 4*inch])
    summary_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # Annotated Image Overlay
    story.append(Paragraph("Annotated Frame Overlay", heading2_style))
    success, encoded_img = cv2.imencode(".png", overlay_bgr)
    if success:
        h_img, w_img = overlay_bgr.shape[:2]
        aspect = w_img / h_img
        max_width = 6.5 * inch
        max_height = 5.5 * inch  # safe limit within letter page frame
        width = max_width
        height = width / aspect
        if height > max_height:
            height = max_height
            width = height * aspect
        img_io = io.BytesIO(encoded_img.tobytes())
        rl_img = RLImage(img_io, width=width, height=height)
        story.append(rl_img)
    story.append(PageBreak())
    
    # Class counts chart (Matplotlib static bar chart inside PDF)
    class_counts = Counter(p["predicted_class"] for p in predictions)
    if class_counts:
        story.append(Paragraph("Predicted Class Distribution Chart", heading2_style))
        fig, ax = plt.subplots(figsize=(6.5, 2.5))
        labels = [DISPLAY_NAMES.get(cls, cls) for cls in class_counts.keys()]
        values = list(class_counts.values())
        colors_list = [RGB_COLORS.get(cls, "#888888") for cls in class_counts.keys()]
        
        ax.bar(labels, values, color=colors_list, edgecolor='grey')
        ax.set_ylabel("Count")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        chart_io = io.BytesIO()
        plt.savefig(chart_io, format="png", bbox_inches="tight", dpi=150)
        chart_io.seek(0)
        rl_chart = RLImage(chart_io, width=5.5*inch, height=2.1*inch)
        story.append(rl_chart)
        plt.close(fig)
        story.append(Spacer(1, 15))
        
    # Track Predictions Table
    story.append(Paragraph("Track Details & Class Probabilities", heading2_style))
    table_headers = [
        Paragraph("<b>Track ID</b>", bold_body_style),
        Paragraph("<b>Predicted Particle</b>", bold_body_style),
        Paragraph("<b>Confidence</b>", bold_body_style),
        Paragraph("<b>Status</b>", bold_body_style)
    ]
    
    class_names = sorted(list(predictions[0]["probabilities"].keys())) if predictions else []
    for cls in class_names:
        table_headers.append(Paragraph(f"<b>P({DISPLAY_NAMES.get(cls, cls)})</b>", bold_body_style))
        
    table_rows = [table_headers]
    for pred in predictions:
        row = [
            Paragraph(f"T{pred['track_id']}", body_style),
            Paragraph(pred["particle_type"], body_style),
            Paragraph(f"{pred['confidence']:.1%}", body_style),
            Paragraph("Accepted" if pred["confidence"] >= metadata.get("confidence_threshold", 0.5) else "Uncertain", body_style)
        ]
        for cls in class_names:
            prob = pred["probabilities"].get(cls, 0.0)
            row.append(Paragraph(f"{prob:.1%}", body_style))
        table_rows.append(row)
        
    base_cols = [0.8*inch, 1.5*inch, 0.9*inch, 0.9*inch]
    num_classes = len(class_names)
    class_col_width = (2.4 / max(1, num_classes)) * inch
    col_widths = base_cols + [class_col_width] * num_classes
    
    pred_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
    pred_table.setStyle(TableStyle([
        ('HEADER', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#edf2f7')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(pred_table)
    
    doc.build(story)
    return buffer.getvalue()
