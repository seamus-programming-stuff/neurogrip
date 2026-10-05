import unittest

import cv2
import numpy as np

from vision_tracking import RegionTracker


def textured_scene(dx=0, dy=0):
    image = np.full((240, 320, 3), 25, np.uint8)
    rng = np.random.RandomState(7)
    texture = rng.randint(80, 245, (80, 100, 3)).astype(np.uint8)
    image[70 + dy:150 + dy, 90 + dx:190 + dx] = texture
    return image


class TrackerTests(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(7)

    def test_translation_tracks_mask_and_box(self):
        tracker = RegionTracker()
        first = tracker.initialize(textured_scene(), (82, 62, 116, 96))
        self.assertTrue(first.valid, first.reason)
        self.assertEqual(first.mask.dtype, np.uint8)
        for delta in (3, 6, 9):
            result = tracker.update(textured_scene(delta, delta // 3))
            self.assertTrue(result.valid, result.reason)
            self.assertGreater(result.quality, 0.45)
            self.assertAlmostEqual(result.bbox[0] - first.bbox[0], delta, delta=2)
            self.assertAlmostEqual(result.bbox[1] - first.bbox[1], delta // 3, delta=2)
            self.assertGreater(np.count_nonzero(result.mask), 6500)

    def test_lost_object_never_returns_last_box(self):
        tracker = RegionTracker()
        self.assertTrue(tracker.initialize(textured_scene(), (82, 62, 116, 96)).valid)
        result = tracker.update(np.full((240, 320, 3), 25, np.uint8))
        self.assertFalse(result.valid)
        self.assertIsNone(result.bbox)
        self.assertIsNone(result.mask)
        # Returning the object does not silently re-arm a failed tracker.
        again = tracker.update(textured_scene())
        self.assertFalse(again.valid)
        self.assertIsNone(again.bbox)
        self.assertTrue(tracker.initialize(textured_scene(), (82, 62, 116, 96)).valid)

    def test_plain_foreground_is_rejected(self):
        image = np.full((240, 320, 3), 25, np.uint8)
        image[70:150, 90:190] = 180
        result = RegionTracker().initialize(image, (82, 62, 116, 96))
        self.assertFalse(result.valid)
        self.assertIsNone(result.mask)
        self.assertIn(result.reason, ("insufficient_foreground_texture", "features_too_concentrated"))

    def test_size_change_is_explicit_loss(self):
        tracker = RegionTracker()
        self.assertTrue(tracker.initialize(textured_scene(), (82, 62, 116, 96)).valid)
        result = tracker.update(cv2.resize(textured_scene(), (160, 120)))
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "image_size_changed")


if __name__ == "__main__":
    unittest.main()
