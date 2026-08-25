import unittest
from pathlib import Path

import app
from cloud_chamber.ui.navigation import MODEL_PAGES


class UiContractTests(unittest.TestCase):
    def test_every_navigation_page_has_a_handler(self):
        handlers = app._build_page_handlers()
        self.assertTrue(set(MODEL_PAGES).issubset(handlers))

    def test_extra_trees_report_path_is_consistent(self):
        report_path = Path("models/extra_trees_hybrid_training_report.json")
        self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
