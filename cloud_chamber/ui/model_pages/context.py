"""Callbacks from the shared application pipeline."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PageContext:
    """Shared processing callbacks supplied to each classifier dashboard."""

    bgr_to_rgb: Callable[[Any], Any]
    process_pipeline_image: Callable[..., dict]
    feature_row: Callable[..., dict]
