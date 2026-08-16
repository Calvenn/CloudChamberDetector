import unittest
from pathlib import Path

from cloud_chamber.ml.member_models.cnn import (
    DEFAULT_DATASET_ROOT,
    FeatureTrackCNN,
    build_class_mapping,
)
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS


class TestCNNSmoke(unittest.TestCase):
    def test_default_dataset_root_exists(self):
        root = Path(DEFAULT_DATASET_ROOT)
        self.assertTrue(root.exists())
        self.assertTrue((root / "development").exists())

    def test_cnn_uses_shared_feature_contract(self):
        class_map = build_class_mapping()
        self.assertIn("alpha", class_map)
        self.assertIn("electron_positron", class_map)
        self.assertEqual(len(FEATURE_COLUMNS), 10)
        model = FeatureTrackCNN(input_dim=len(FEATURE_COLUMNS), num_classes=len(class_map))
        self.assertEqual(model.input_dim, len(FEATURE_COLUMNS))
        self.assertEqual(model.output_dim, len(class_map))


if __name__ == "__main__":
    unittest.main()
