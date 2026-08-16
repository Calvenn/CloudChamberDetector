"""Callbacks from the shared application pipeline."""
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any
@dataclass(frozen=True)
class PageContext:
    bgr_to_rgb: Callable[[Any], Any]
    process_pipeline_image: Callable[..., dict]
    feature_row: Callable[..., dict]
