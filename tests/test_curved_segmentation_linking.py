import unittest

import cv2
import numpy as np

from cloud_chamber.segmentation import _link_curved_contours


class CurvedSegmentationLinkingTests(unittest.TestCase):
    def _fragments(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.ellipse(mask, (35, 50), (20, 12), 0, 170, 260, 255, 5)
        cv2.ellipse(mask, (67, 50), (20, 12), 0, 280, 350, 255, 5)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return mask, contours

    def test_links_curved_fragments_with_bright_path_evidence(self):
        mask, contours = self._fragments()
        evidence = cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8))
        linked, link_count = _link_curved_contours(
            mask,
            contours,
            evidence,
            maximum_gap=45,
            maximum_tangent_angle=55,
            minimum_aspect_ratio=1.1,
            evidence_threshold=1,
            minimum_evidence_fraction=0.1,
        )
        component_count = cv2.connectedComponents(
            (linked > 0).astype(np.uint8)
        )[0] - 1
        self.assertEqual(link_count, 1)
        self.assertEqual(component_count, 1)

    def test_rejects_connector_without_bright_path_evidence(self):
        mask, contours = self._fragments()
        linked, link_count = _link_curved_contours(
            mask,
            contours,
            np.zeros_like(mask),
            maximum_gap=45,
            maximum_tangent_angle=55,
            minimum_aspect_ratio=1.1,
            evidence_threshold=1,
            minimum_evidence_fraction=0.1,
        )
        component_count = cv2.connectedComponents(
            (linked > 0).astype(np.uint8)
        )[0] - 1
        self.assertEqual(link_count, 0)
        self.assertEqual(component_count, 2)


if __name__ == "__main__":
    unittest.main()
