import numpy as np
from types import SimpleNamespace

import cloud_chamber.ml.member_models.decision_tree as decision_tree
from cloud_chamber.ml.member_models.decision_tree import (
    FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    augment_feature_matrix,
)


def test_decision_tree_feature_augmentation_is_finite_and_deterministic():
    raw = np.array(
        [
            [100.0, 40.0, 20.0, 5.0, 45.0, 4.0, 0.8, 0.7, 5.0, 200.0],
            [0.0, 0.0, 0.0, 0.0, -90.0, 0.0, 0.0, 0.0, 0.0, 10.0],
        ]
    )

    augmented = augment_feature_matrix(raw)

    assert augmented.shape == (2, len(MODEL_FEATURE_COLUMNS))
    np.testing.assert_array_equal(augmented[:, : len(FEATURE_COLUMNS)], raw)
    assert np.isclose(augmented[0, len(FEATURE_COLUMNS)], np.pi / 4)
    assert np.isclose(augmented[0, len(FEATURE_COLUMNS) + 1], 2.0)
    assert np.isfinite(augmented).all()


def test_decision_tree_visual_report_hides_predictions_below_threshold(monkeypatch):
    calls = []

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 0

        @staticmethod
        def rectangle(*args):
            calls.append("rectangle")

        @staticmethod
        def putText(*args):
            calls.append("text")

    monkeypatch.setattr(decision_tree, "cv2", FakeCV2)
    features = [
        SimpleNamespace(track_id=1, bounding_box=(1, 1, 4, 4)),
        SimpleNamespace(track_id=2, bounding_box=(6, 6, 4, 4)),
    ]
    predictions = [
        {
            "predicted_class": "alpha",
            "particle_type": "Alpha",
            "confidence": 0.59,
            "inference_time_ms": 1.0,
            "probabilities": {"alpha": 0.59},
        },
        {
            "predicted_class": "proton",
            "particle_type": "Proton",
            "confidence": 0.60,
            "inference_time_ms": 1.0,
            "probabilities": {"proton": 0.60},
        },
    ]

    _, rows = decision_tree.build_visual_report(
        np.zeros((12, 12, 3), dtype=np.uint8),
        features,
        predictions,
        confidence_threshold=0.60,
    )

    assert calls == ["rectangle", "text"]
    assert [row["Track"] for row in rows] == [2]
    assert rows[0]["Status"] == "Accepted"
