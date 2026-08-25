import unittest

import cv2
import numpy as np

from cloud_chamber.video_processing import temporal_track_composite


class TemporalTrackCompositeTests(unittest.TestCase):
    def test_transient_bright_track_is_boosted(self):
        frames = [np.full((48, 64, 3), 30, dtype=np.uint8) for _ in range(5)]
        cv2.line(frames[1], (8, 24), (54, 24), (90, 90, 90), 1)
        composite, evidence = temporal_track_composite(
            frames, reference_index=2, evidence_gain=2.0
        )
        self.assertGreater(int(evidence[24, 30]), 0)
        self.assertGreater(int(composite[24, 30, 0]), int(frames[2][24, 30, 0]))
        self.assertEqual(composite.shape, frames[2].shape)

    def test_static_background_is_not_temporal_evidence(self):
        frame = np.full((32, 40, 3), 50, dtype=np.uint8)
        cv2.circle(frame, (10, 10), 3, (120, 120, 120), -1)
        _, evidence = temporal_track_composite([frame.copy() for _ in range(5)])
        self.assertEqual(int(evidence.max()), 0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            temporal_track_composite([])
        with self.assertRaises(ValueError):
            temporal_track_composite(
                [np.zeros((10, 10, 3), dtype=np.uint8)], evidence_gain=-1
            )


if __name__ == "__main__":
    unittest.main()
