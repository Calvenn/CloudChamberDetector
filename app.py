"""Streamlit composition root for the cloud-chamber application.

The entry point owns only application wiring. Image-processing behaviour lives
in ``cloud_chamber.ui.shared_pipeline``, while every classifier owns its page.
"""

from __future__ import annotations

import streamlit as st

from cloud_chamber.core.config import load_config
from cloud_chamber.ui import shared_pipeline
from cloud_chamber.ui.comparison_page import render as render_comparison_page
from cloud_chamber.ui.model_pages import (
    cnn_page,
    decision_tree_page,
    extra_trees_page,
    mlp_page,
    svm_page,
)
from cloud_chamber.ui.model_pages.context import PageContext
from cloud_chamber.ui.navigation import (
    COMPARISON_PAGE,
    MLP_PAGE,
    PAGES,
    SHARED_PIPELINE_PAGE,
    render_selected_page,
)


def main() -> None:
    """Configure Streamlit and render the page selected by the user."""
    st.set_page_config(
        page_title="Cloud Chamber Particle Classification",
        layout="wide",
    )
    config = load_config()
    shared_pipeline.initialise_state()

    st.sidebar.title("Cloud Chamber")
    page = st.sidebar.radio("Navigate", PAGES)
    shared_pipeline.input_status()

    render_selected_page(page, config, _build_page_handlers())


def _build_page_handlers() -> dict:
    """Connect navigation labels to their independent page renderers."""
    context = PageContext(
        bgr_to_rgb=shared_pipeline.bgr_to_rgb,
        process_pipeline_image=shared_pipeline.process_pipeline_image,
        feature_row=shared_pipeline.feature_row,
    )
    return {
        SHARED_PIPELINE_PAGE: shared_pipeline.render,
        "Convolutional Neural Network": lambda config: cnn_page.render(config, context),
        "Support Vector Machine": lambda _config: svm_page.render(context),
        "Decision Tree": lambda _config: decision_tree_page.render(context),
        "Multilayer Perceptron": lambda config: mlp_page.render(config, context),
        "Extremely Randomized Trees": lambda config: extra_trees_page.render(
            config, context
        ),
        COMPARISON_PAGE: lambda _config: render_comparison_page(),
    }


if __name__ == "__main__":
    main()
