"""Decision Tree Streamlit page with gallery explorer and batch processing."""
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from collections import Counter
from cloud_chamber.ml.member_models.decision_tree import (
    build_visual_report as decision_tree_build_visual_report,
    encode_report_png as decision_tree_encode_report_png,
    load_model as decision_tree_load_model,
    predict_tracks as decision_tree_predict_tracks,
    DISPLAY_NAMES,
)
from cloud_chamber.reporting import (
    assess_all_contours,
    build_summary,
    reporting_status,
)
from .context import PageContext


def _extract_particle_thumbnail(image: np.ndarray, bounding_box: tuple, size: int = 100) -> np.ndarray:
    """Extract a thumbnail crop of the particle region."""
    x, y, width, height = bounding_box
    x1, y1 = max(0, x - 5), max(0, y - 5)
    x2, y2 = min(image.shape[1], x + width + 5), min(image.shape[0], y + height + 5)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    height_crop, width_crop = crop.shape[:2]
    scale = size / max(width_crop, height_crop)
    new_size = (int(width_crop * scale), int(height_crop * scale))
    resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_LINEAR)
    thumbnail = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - resized.shape[0]) // 2
    x_offset = (size - resized.shape[1]) // 2
    thumbnail[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
    return thumbnail


def _create_gallery_item(track, prediction, quality, image, analysis_image, confidence_threshold):
    """Create a gallery item with thumbnail and metadata."""
    is_confident = prediction["confidence"] >= confidence_threshold
    quality_grade = quality["grade"]
    
    thumbnail = _extract_particle_thumbnail(analysis_image, track.bounding_box)
    
    return {
        "track_id": track.track_id,
        "image_name": st.session_state.get("input_name", "Unknown"),
        "predicted_class": prediction["predicted_class"],
        "particle_type": DISPLAY_NAMES.get(prediction["predicted_class"], prediction["predicted_class"]),
        "confidence": prediction["confidence"],
        "is_confident": is_confident,
        "quality_grade": quality_grade,
        "quality_score": quality["score"],
        "thumbnail": thumbnail,
        "track": track,
        "prediction": prediction,
        "quality": quality,
        "image": image,
        "analysis_image": analysis_image,
    }


def _build_gallery_montage(gallery_items, cols=5, thumb_size=100, gap=10):
    """Build a single PNG montage from particle gallery thumbnails."""
    if not gallery_items:
        return np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)

    rows = int(np.ceil(len(gallery_items) / cols))
    canvas_w = cols * thumb_size + (cols - 1) * gap
    canvas_h = rows * thumb_size + (rows - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for idx, item in enumerate(gallery_items):
        row = idx // cols
        col = idx % cols
        y = row * (thumb_size + gap)
        x = col * (thumb_size + gap)
        canvas[y:y + thumb_size, x:x + thumb_size] = item["thumbnail"]

    return canvas


def render(context: PageContext) -> None:
    """Render the Decision Tree classification page with gallery explorer."""
    st.title("Decision Tree Classifier - Explorer")
    st.subheader("Browse and analyze particle detections")
    
    result = st.session_state.get("pipeline_result")
    if result is None:
        st.warning(
            "Process an image or video frame on the Shared Processing "
            "Pipeline page first."
        )
        return
    
    model_path = Path("models/decision_tree_classifier.joblib")
    if not model_path.exists():
        st.warning(
            "Train the model first: `python scripts/train_decision_tree.py`"
        )
        return

    # Load model once for efficiency
    model_bundle = decision_tree_load_model(model_path)
    
    # Get batch of inputs (images and/or video frames)
    samples = st.session_state.get("input_batch") or [
        {
            "image": st.session_state["input_image"],
            "name": st.session_state["input_name"],
            "description": st.session_state["source_description"],
        }
    ]
    
    # Confidence threshold must be chosen before classification
    st.divider()
    st.subheader("Confidence Threshold Setting")
    st.write("Select the confidence threshold before running the batch classification.")
    confidence_threshold = st.slider(
        "Classification Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("decision_tree_confidence_threshold", 0.60),
        step=0.05,
        key="decision_tree_confidence_threshold_top",
        help="Predictions below this value are treated as uncertain for the report and gallery status.",
    )
    st.session_state["decision_tree_confidence_threshold"] = confidence_threshold
    confidence_threshold = st.session_state["decision_tree_confidence_threshold"]
    
    # Batch processing button
    col1, col2 = st.columns([3, 1])
    
    with col2:
        button_label = (
            f"Classify all {len(samples)}"
            if len(samples) > 1
            else "Classify"
        )
        
        if st.button(button_label, type="primary"):
            batch_results = {}
            progress = st.progress(0.0, text="Processing batch...")
            
            config = st.session_state.get("config")
            if not config:
                try:
                    from cloud_chamber.config import load_config
                    config = load_config()
                except:
                    config = {}
            
            for index, sample in enumerate(samples):
                sample_result = context.process_pipeline_image(
                    sample["image"],
                    config,
                    st.session_state.get("calibration_settings"),
                )
                
                sample_predictions = decision_tree_predict_tracks(
                    model_bundle, sample_result["features"]
                )
                
                sample_quality = assess_all_contours(
                    sample_result["features"],
                    sample_result["enhancement"].enhanced,
                    sample_result["segmentation"].binary_mask,
                    sample_result["segmentation"].parameters,
                )
                
                batch_results[sample["name"]] = {
                    "image": sample["image"],
                    "analysis_image": sample_result["input_image"],
                    "description": sample["description"],
                    "result": sample_result,
                    "predictions": sample_predictions,
                    "quality": sample_quality,
                }
                
                progress.progress(
                    (index + 1) / len(samples),
                    text=f"Processed {index + 1} of {len(samples)} inputs",
                )
            
            progress.empty()
            st.session_state["decision_tree_batch_results"] = batch_results
    
    # Display batch results if available
    batch_results = st.session_state.get("decision_tree_batch_results", {})
    if not batch_results:
        st.info("No batch results. Process images first using the Classify button.")
        return
    
    # ===== SECTION 1: SEARCH & FILTER =====
    st.divider()
    st.subheader("1. Search & Filter Images")
    
    filter_cols = st.columns(3)
    
    # Search by image name
    search_query = filter_cols[0].text_input(
        "🔍 Search by image name",
        placeholder="Enter image name...",
        key="dt_search_image"
    )
    
    # Get all unique particle types across all images
    all_particle_types = set()
    for entry in batch_results.values():
        for pred in entry["predictions"]:
            all_particle_types.add(pred["predicted_class"])
    
    particle_types = ["All types"]
    particle_types.extend(sorted(all_particle_types))
    
    selected_particle_type = filter_cols[1].selectbox(
        "Filter by particle type in image",
        particle_types,
        key="dt_filter_type_image"
    )

    detection_status_options = ["All images", "Images with particles", "Images without particles"]
    selected_detection_status = filter_cols[2].selectbox(
        "Filter by detection status in image",
        detection_status_options,
        key="dt_filter_status_image"
    )
    
    # Count particles by type and detection status for display
    particle_type_counts = {}
    images_with_particles = 0
    images_without_particles = 0
    for entry in batch_results.values():
        predictions = entry["predictions"]
        if len(predictions) > 0:
            images_with_particles += 1
            for pred in predictions:
                pred_class = pred["predicted_class"]
                particle_type_counts[pred_class] = particle_type_counts.get(pred_class, 0) + 1
        else:
            images_without_particles += 1
    
    # Display counts next to filters
    count_col1, count_col2 = st.columns(2)
    with count_col1:
        st.markdown(f"**Particle Types detected:** ({len(particle_types)-1} types found)")
        for ptype in sorted(all_particle_types):
            count = particle_type_counts.get(ptype, 0)
            st.write(f"  • {DISPLAY_NAMES.get(ptype, ptype)}: {count} images")
    
    with count_col2:
        st.markdown("**Detection Status:**")
        st.write(f"  • Images with particles: {images_with_particles}")
        st.write(f"  • Images without particles: {images_without_particles}")
    
    # ===== APPLY IMAGE FILTERS =====
    filtered_images = []
    for image_name, entry in batch_results.items():
        # Search filter
        if search_query.lower() not in image_name.lower():
            continue
        
        predictions = entry["predictions"]
        particle_classes = set(p["predicted_class"] for p in predictions)
        has_particles = len(predictions) > 0
        
        # Particle type filter
        if selected_particle_type != "All types":
            if selected_particle_type not in particle_classes:
                continue

        if selected_detection_status == "Images with particles" and not has_particles:
            continue
        if selected_detection_status == "Images without particles" and has_particles:
            continue
        
        filtered_images.append({
            "name": image_name,
            "entry": entry,
            "particle_count": len(predictions),
            "particle_classes": particle_classes
        })
    
    # ===== SECTION 2: CHOOSE IMAGE TO EXPLORE =====
    st.divider()
    st.subheader("2. Choose an Image to Explore")
    
    if not filtered_images:
        st.warning("No images match the selected filters.")
        return
    
    st.write(f"**Found {len(filtered_images)} matching image(s) out of {len(batch_results)} total**")
    
    # Create image options with details
    image_options = []
    for img in filtered_images:
        # Count particles by type
        class_counts = Counter(p["predicted_class"] for p in img["entry"]["predictions"])
        particle_details = ", ".join(
            f"{DISPLAY_NAMES.get(k, k)}: {v}"
            for k, v in sorted(class_counts.items())
        ) if class_counts else "No particles"
        
        option_text = f"{img['name']} ({img['particle_count']} particles) - {particle_details}"
        image_options.append(option_text)
    
    selected_image_idx = st.selectbox(
        "Select image to analyze:",
        range(len(filtered_images)),
        format_func=lambda i: image_options[i],
        key="dt_image_selector"
    )
    
    selected_image = filtered_images[selected_image_idx]
    selected_entry = selected_image["entry"]
    st.session_state["input_image"] = selected_entry["image"]
    st.session_state["input_name"] = selected_image["name"]
    st.session_state["source_description"] = selected_entry["description"]
    st.session_state["pipeline_result"] = selected_entry["result"]
    st.session_state["decision_tree_predictions"] = selected_entry["predictions"]
    st.session_state["decision_tree_quality"] = selected_entry["quality"]
    result = selected_entry["result"]
    predictions = selected_entry["predictions"]
    quality_assessments = selected_entry["quality"]
    
    # ===== SECTION 3: DISPLAY FULL SCANNED IMAGE =====
    st.divider()
    st.subheader("3. Scanned Image with Detections")
    
    overlay, _ = decision_tree_build_visual_report(
        result["input_image"],
        result["features"],
        predictions,
        confidence_threshold,
    )
    
    st.image(context.bgr_to_rgb(overlay))
    st.caption(
        "Decision Tree particle predictions. Annotation colours identify the "
        "predicted particle type; confidence still controls the report status."
    )
    
    # ===== SECTION 4: IMAGE-LEVEL DETAILS =====
    st.divider()
    st.subheader("4. Image Analysis Details")
    
    processing_time_ms = float(result["segmentation"].processing_time_ms) + sum(
        float(item["inference_time_ms"]) for item in predictions
    )
    summary = build_summary(
        predictions=predictions,
        quality_assessments=quality_assessments,
        confidence_threshold=confidence_threshold,
        processing_time_ms=processing_time_ms,
    )
    
    # Top metrics (wrapped to prevent overflow)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Particles", summary["detected_contours"])
    metric_cols[1].metric("Confident", summary["confident_classifications"])
    metric_cols[2].metric("Uncertain", summary["uncertain_classifications"])
    
    metric_cols = st.columns(3)
    metric_cols[0].metric("Dominant Type", summary["dominant_prediction"])
    metric_cols[1].metric("Quality Grade", summary["overall_contour_quality"])
    metric_cols[2].metric("Avg Quality", f"{summary['mean_contour_quality']:.0f}/100")
    
    metric_cols = st.columns(2)
    metric_cols[0].metric("Processing Time", f"{processing_time_ms:.1f}ms")
    
    # Class distribution - graphical only
    st.markdown("**Particle Type Distribution**")
    class_counts = summary["class_counts"]
    if class_counts:
        chart_data = {
            DISPLAY_NAMES.get(k, k): v
            for k, v in class_counts.items()
        }
        st.bar_chart(chart_data)
        class_data = [
            {"Type": DISPLAY_NAMES.get(name, name), "Count": count}
            for name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        st.dataframe(class_data, use_container_width=True, hide_index=True)
    
    # ===== SECTION 5: PARTICLE GALLERY =====
    st.divider()
    st.subheader("5. Particle Gallery")
    
    # Create gallery items
    gallery_items = []
    for track, prediction, quality in zip(result["features"], predictions, quality_assessments):
        item = _create_gallery_item(
            track, prediction, quality, 
            st.session_state["input_image"],
            result["input_image"],
            confidence_threshold
        )
        gallery_items.append(item)
    
    # Particle-level filters
    particle_type_filter = st.selectbox(
        "Filter by particle type",
        ["All types"] + list(set(item["particle_type"] for item in gallery_items)),
        key="dt_particle_type"
    )
    
    # Apply particle filters
    filtered_particles = []
    for item in gallery_items:
        if particle_type_filter != "All types" and item["particle_type"] != particle_type_filter:
            continue
        filtered_particles.append(item)
    
    # Pagination
    items_per_page = 20
    total_pages = max(1, (len(filtered_particles) + items_per_page - 1) // items_per_page)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if total_pages > 1:
            current_page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.get("dt_current_page", 1),
                key="dt_page_selector"
            )
            st.session_state["dt_current_page"] = current_page
        else:
            current_page = 1
    
    with col2:
        st.write(f"**Showing {len(filtered_particles)} of {len(gallery_items)} particles** | Page {current_page} of {total_pages}")
    
    if len(filtered_particles) == 0:
        st.warning("No particles match the selected filters.")
        st.stop()
    
    # Get items for current page
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_particles))
    page_items = filtered_particles[start_idx:end_idx]
    
    # Display gallery grid
    st.markdown("### Particle Thumbnails (5 per row)")
    for row_idx in range(0, len(page_items), 5):
        cols = st.columns(5)
        for col_idx, item in enumerate(page_items[row_idx:row_idx+5]):
            with cols[col_idx]:
                thumbnail_rgb = cv2.cvtColor(item["thumbnail"], cv2.COLOR_BGR2RGB)
                
                if st.button(
                    f"🔍",
                    key=f"gallery_item_{item['track_id']}_{row_idx}_{col_idx}",
                    use_container_width=True,
                ):
                    st.session_state["selected_particle"] = item
                
                st.image(thumbnail_rgb, use_container_width=True)
                
                confidence_pct = f"{item['confidence']:.0%}"
                color = "🟢" if item["is_confident"] else "🔴"
                st.caption(f"**T{item['track_id']}**\n{item['particle_type']}\n{color} {confidence_pct}")
    
    # ===== SECTION 6: PARTICLE DETAILS =====
    st.divider()
    st.subheader("6. Particle Details")
    
    selected_particle = st.session_state.get("selected_particle")
    if selected_particle is None:
        st.info("👆 Click on a particle thumbnail above to view detailed information.")
    else:
        col_preview, col_details = st.columns([1, 2])
        
        with col_preview:
            st.markdown("### Particle Preview")
            thumbnail_rgb = cv2.cvtColor(selected_particle["thumbnail"], cv2.COLOR_BGR2RGB)
            st.image(thumbnail_rgb, use_container_width=True)
            st.caption(f"Track ID: {selected_particle['track_id']}\nImage: {selected_particle['image_name']}")
        
        with col_details:
            st.markdown("### Detection Attributes")
            
            # Key metrics
            metrics_cols = st.columns(3)
            metrics_cols[0].metric(
                "Classification",
                selected_particle["particle_type"],
            )
            if selected_particle["is_confident"]:
                confidence_delta = "Above threshold"
                confidence_color = "normal"
            else:
                confidence_delta = "Below threshold"
                confidence_color = "inverse"

            metrics_cols[1].metric(
                "Confidence",
                f"{selected_particle['confidence']:.1%}",
                delta=confidence_delta,
                delta_color=confidence_color,
            )
            metrics_cols[2].metric(
                "Contour Quality",
                selected_particle["quality_grade"],
                delta=f"{selected_particle['quality_score']}/100"
            )
            
            # Quality details
            st.markdown("#### Contour Assessment")
            quality_data = selected_particle["quality"]
            quality_cols = st.columns(2)
            
            with quality_cols[0]:
                st.write(f"**Local Contrast:** {quality_data['local_contrast']:.2f}")
                st.write(f"**Solidity:** {selected_particle['track'].solidity:.3f}")
                st.write(f"**Aspect Ratio:** {selected_particle['track'].aspect_ratio:.2f}")
            
            with quality_cols[1]:
                st.write(f"**Near Boundary:** {'Yes' if quality_data['near_boundary'] else 'No'}")
                st.write(f"**Very Thin:** {'Yes' if quality_data['very_thin'] else 'No'}")
                st.write(f"**Area (px²):** {selected_particle['track'].area_pixels:.1f}")
            
            # Decision status
            decision = reporting_status(
                confidence=float(selected_particle["prediction"]["confidence"]),
                confidence_threshold=confidence_threshold,
                quality_score=int(selected_particle["quality"]["score"]),
            )
            
            st.markdown(f"#### Decision Status")
            st.info(f"**{decision}**")
            
            # Track features
            st.markdown("#### Track Features")
            features_cols = st.columns(2)
            
            with features_cols[0]:
                st.write(f"**Perimeter (px):** {selected_particle['track'].perimeter_pixels:.2f}")
                st.write(f"**Major Axis (px):** {selected_particle['track'].major_axis_pixels:.2f}")
                st.write(f"**Mean Width (px):** {selected_particle['track'].mean_width_pixels:.2f}")
                st.write(f"**Thickness (px):** {selected_particle['track'].thickness_pixels:.2f}")
            
            with features_cols[1]:
                st.write(f"**Orientation (°):** {selected_particle['track'].orientation_degrees:.2f}")
                st.write(f"**Mean Intensity:** {selected_particle['track'].mean_intensity:.2f}")
                st.write(f"**Rectangularity:** {selected_particle['track'].rectangularity:.3f}")
            
            # Probability distribution
            st.markdown("#### Classification Probabilities")
            probs = selected_particle["prediction"]["probabilities"]
            prob_data = {DISPLAY_NAMES.get(k, k): v for k, v in probs.items()}
            st.dataframe(
                [{"Class": k, "Probability": f"{v:.1%}"} for k, v in sorted(prob_data.items(), key=lambda x: x[1], reverse=True)],
                use_container_width=True,
                hide_index=True
            )
            
            # Quality evidence/warnings
            if not selected_particle["is_confident"]:
                st.caption("Confidence is below threshold; quality evidence is hidden for this particle.")
            else:
                if quality_data.get("warnings"):
                    st.warning(f"⚠️ **Quality Warnings:**\n" + "\n".join(f"• {w}" for w in quality_data["warnings"]))
                
                if quality_data.get("evidence"):
                    st.success(f"✓ **Quality Evidence:**\n" + "\n".join(f"• {e}" for e in quality_data["evidence"]))
    
    # ===== SECTION 7: CONCLUSION TABLE =====
    st.divider()
    st.subheader("7. Classification Summary - All Images in Batch")
    
    summary_rows = []
    summary_rows_dict = []
    for image_name, entry in batch_results.items():
        for track, prediction, quality in zip(entry["result"]["features"], entry["predictions"], entry["quality"]):
            decision = reporting_status(
                confidence=float(prediction["confidence"]),
                confidence_threshold=confidence_threshold,
                quality_score=int(quality["score"]),
            )
            
            particle_type = DISPLAY_NAMES.get(prediction["predicted_class"], prediction["predicted_class"])
            confidence_val = float(prediction['confidence'])
            
            summary_rows.append({
                "Image": image_name,
                "Track ID": track.track_id,
                "Particle Type": particle_type,
                "Confidence": f"{confidence_val:.1%}",
                "Quality Grade": quality["grade"],
                "Quality Score": f"{quality['score']}/100",
                "Area (px²)": f"{track.area_pixels:.1f}",
                "Aspect Ratio": f"{track.aspect_ratio:.2f}",
                "Decision": decision,
                "Local Contrast": f"{quality['local_contrast']:.2f}",
            })
            
            summary_rows_dict.append({
                "Image": image_name,
                "Track ID": track.track_id,
                "Particle Type": particle_type,
                "Confidence": confidence_val,
                "Quality Score": int(quality["score"]),
                "Decision": decision,
            })
    
    st.caption(f"Showing results aggregated across all {len(batch_results)} processed image(s) in the batch.")
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)
    
    # Visualizations for conclusion
    st.subheader("Summary Visualizations")
    
    viz_cols = st.columns(2)
    
    with viz_cols[0]:
        st.markdown("**Decision Distribution**")
        decision_counts = Counter(row["Decision"] for row in summary_rows_dict)
        st.bar_chart(decision_counts)
    
    with viz_cols[1]:
        st.markdown("**Particle Type Distribution (All)**")
        type_counts = Counter(row["Particle Type"] for row in summary_rows_dict)
        st.bar_chart(type_counts)

    st.markdown("**Particle Type Distribution (Reliable Candidates Only)**")
    reliable_type_counts = Counter(
        row["Particle Type"]
        for row in summary_rows_dict
        if float(row["Confidence"]) >= confidence_threshold
    )
    st.bar_chart(reliable_type_counts)
    # Export options
    st.divider()
    st.subheader("Export Results")
    
    export_cols = st.columns(1)
    safe_name = Path(st.session_state["input_name"]).stem

    png_choice = st.selectbox(
        "PNG export target",
        ["Current full image", "Selected image particle gallery"],
        key="dt_png_export_target",
    )

    if png_choice == "Current full image":
        png_data = decision_tree_encode_report_png(overlay)
        png_name = f"{safe_name}_decision_tree_full_image.png"
    else:
        gallery_export = _build_gallery_montage(gallery_items)
        png_data = decision_tree_encode_report_png(gallery_export)
        png_name = f"{safe_name}_decision_tree_particle_gallery.png"

    export_cols[0].download_button(
        "📥 PNG Image",
        data=png_data,
        file_name=png_name,
        mime="image/png",
    )
