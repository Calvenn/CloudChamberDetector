import unittest

from cloud_chamber.ml.member_models.extra_trees import summarise_predictions


class ExtraTreesSummaryTests(unittest.TestCase):
    def test_missing_classes_are_reported_as_zero(self):
        summary = summarise_predictions(
            [
                {
                    "predicted_class": "alpha",
                    "particle_type": "Alpha",
                    "confidence": 0.8,
                }
            ]
        )
        self.assertEqual(summary["Alpha"], 1)
        self.assertEqual(summary["Electron/Positron"], 0)
        self.assertEqual(summary["Proton"], 0)
        self.assertEqual(summary["V-track"], 0)
        self.assertEqual(summary["Uncertain"], 0)

    def test_empty_predictions_keep_stable_schema(self):
        summary = summarise_predictions([])
        self.assertEqual(summary["Total"], 0)
        self.assertEqual(summary["Electron/Positron"], 0)


if __name__ == "__main__":
    unittest.main()
