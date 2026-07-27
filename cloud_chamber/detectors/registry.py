"""Central detector registry used by the CLI and future UI."""

from __future__ import annotations

from collections.abc import Callable

from cloud_chamber.detectors.base import Detector


DetectorFactory = Callable[[], Detector]
_DETECTORS: dict[str, DetectorFactory] = {}


def register_detector(name: str) -> Callable[[DetectorFactory], DetectorFactory]:
    normalised_name = name.strip().lower()
    if not normalised_name:
        raise ValueError("Detector registration name cannot be empty")

    def decorator(factory: DetectorFactory) -> DetectorFactory:
        if normalised_name in _DETECTORS:
            raise ValueError(f"Detector already registered: {normalised_name}")
        _DETECTORS[normalised_name] = factory
        return factory

    return decorator


def create_detector(name: str) -> Detector:
    normalised_name = name.strip().lower()
    try:
        return _DETECTORS[normalised_name]()
    except KeyError as error:
        available = ", ".join(list_detectors())
        raise KeyError(
            f"Unknown detector '{name}'. Available detectors: {available}"
        ) from error


def list_detectors() -> list[str]:
    return sorted(_DETECTORS)

