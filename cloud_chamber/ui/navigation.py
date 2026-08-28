"""Central page registry and routing for the Streamlit application.

Model members only need to expose a page-rendering function and connect it in
``build_page_handlers`` in ``app.py``. Keeping navigation here avoids mixing
sidebar routing decisions with the individual page implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


MLP_PAGE = "Multilayer Perceptron"

MODEL_PAGES = {
    "Convolutional Neural Network": "Convolutional Neural Network",
    "Support Vector Machine": "Support Vector Machine",
    "Decision Tree": "Decision Tree",
    "Multilayer Perceptron": "Multilayer Perceptron (MLP Ensemble)",
    "Extremely Randomized Trees": "Extremely Randomized Trees",
}

SHARED_PIPELINE_PAGE = "Shared Processing Pipeline"
COMPARISON_PAGE = "Final Model Comparison"
PAGES = [SHARED_PIPELINE_PAGE, *MODEL_PAGES, COMPARISON_PAGE]

PageRenderer = Callable[[dict[str, Any]], None]


def render_selected_page(
    page: str,
    config: dict[str, Any],
    handlers: Mapping[str, PageRenderer],
) -> None:
    """Render the selected page through its registered application handler."""
    try:
        handler = handlers[page]
    except KeyError as exc:
        raise ValueError(f"No UI handler is connected for page: {page}") from exc
    handler(config)

