"""Classical image-processing stages used by every classifier.

Modules remain separate because acquisition, enhancement, segmentation, and
orchestration have different responsibilities and can be tested independently.
"""

__all__ = [
    "acquisition",
    "calibration",
    "enhancement",
    "pipeline",
    "segmentation",
    "tiling",
    "video_processing",
]
