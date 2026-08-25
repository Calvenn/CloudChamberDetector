import unittest
from pathlib import Path
from unittest.mock import patch

import app
from cloud_chamber.ui.navigation import MODEL_PAGES
from cloud_chamber.ui.model_pages.context import PageContext


class UiContractTests(unittest.TestCase):
    def test_every_navigation_page_has_a_handler(self):
        handlers = app._build_page_handlers()
        self.assertTrue(set(MODEL_PAGES).issubset(handlers))

    def test_extra_trees_report_path_is_consistent(self):
        report_path = Path("models/extra_trees_hybrid_training_report.json")
        self.assertTrue(report_path.is_file())

    def test_extra_trees_handler_passes_config_and_context(self):
        config = {"project": {"random_seed": 42}}
        with patch.object(app.extra_trees_page, "render") as render:
            app._build_page_handlers()["Extremely Randomized Trees"](config)

        render.assert_called_once()
        received_config, received_context = render.call_args.args
        self.assertIs(received_config, config)
        self.assertIsInstance(received_context, PageContext)

    def test_extra_trees_page_has_no_unbound_feature_signature(self):
        source = Path(
            "cloud_chamber/ui/model_pages/extra_trees_page.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("feature_signature", source)


if __name__ == "__main__":
    unittest.main()
