import unittest
import numpy as np

from proximity import StrikeConfig, ProximityEngine, distances_to_path, points_from_depth
from protocol import Direction, Mode


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.ones((80, 120), np.float32)
        self.finger = np.zeros_like(self.depth, bool)
        self.target = np.zeros_like(self.depth, bool)
        self.finger[35:45, 35:45] = True
        self.target[35:45, 55:65] = True
        self.depth[self.finger] = .5
        self.depth[self.target] = .55
        self.k = np.array([[100., 0, 60], [0, 100, 40], [0, 0, 1]])
        self.config = StrikeConfig(
            gap_error_mm=1, measurement_validated=True,
            capture_timing_validated=True, source_delay_bound_ms=0,
            camera_to_finger=[[1, 0, 0, 100], [0, 1, 0, 0], [0, 0, 1, -500], [0, 0, 0, 1]],
            strokes={"FORWARDS": [[0, 0, 0], [100, 0, 50]]},
            stroke_geometry_validated=True, tip_radius_mm=10,
            max_target_speed_mm_s=100,
        )
        self.engine = ProximityEngine(self.config)

    def evaluate(self, **extra):
        options = dict(metric=True, armed=True, now=1, mode=Mode.LIVE)
        options.update(extra)
        return self.engine.evaluate(self.depth, self.finger, self.target, self.k, **options)

    def settle(self, **extra):
        result = None
        for i in range(5):
            result = self.evaluate(**dict(extra, now=float(i + 1)))
        return result

    def test_unproject_not_just_depth_difference(self):
        mask = np.zeros((80, 120), bool)
        mask[40, 80] = True
        point = points_from_depth(np.ones((80, 120)), mask, self.k)[0]
        np.testing.assert_allclose(point, [200, 0, 1000])
        result = self.evaluate()
        self.assertGreater(result.gap_mm, abs(result.target_depth_mm - result.finger_depth_mm))

    def test_reachable_debounce_and_immediate_disarm(self):
        self.assertFalse(self.evaluate().strike_permit)
        result = self.settle()
        self.assertTrue(result.strike_permit, result.block_reasons)
        self.assertEqual(result.direction, Direction.FORWARDS)
        self.assertFalse(self.evaluate(now=7, armed=False).strike_permit)

    def test_simulation_never_physical_permit(self):
        result = self.settle(mode=Mode.SIMULATION)
        self.assertTrue(result.candidate_permit)
        self.assertFalse(result.strike_permit)
        self.assertIn("simulation", result.block_reasons)

    def test_farther_target_same_image_position_blocked(self):
        self.depth[self.target] = .9
        result = self.settle()
        self.assertFalse(result.strike_permit)
        self.assertIn("unreachable", result.block_reasons)

    def test_unknown_scale_no_metric_gap(self):
        result = self.evaluate(metric=False)
        self.assertIsNone(result.gap_mm)
        self.assertFalse(result.strike_permit)

    def test_invalid_depth_or_lost_tracking_revokes_immediately(self):
        self.assertTrue(self.settle().strike_permit)
        self.depth[self.target] = np.nan
        result = self.evaluate(now=8)
        self.assertFalse(result.strike_permit)
        self.assertIn("unknown_depth", result.block_reasons)
        self.assertFalse(self.evaluate(now=9, target_valid=False).strike_permit)

    def test_stale_measurement_revokes(self):
        self.assertTrue(self.settle().strike_permit)
        self.assertFalse(self.evaluate(now=8, capture_age_ms=500).strike_permit)

    def test_mask_overlap_cannot_be_interpreted_as_contact(self):
        self.target |= self.finger
        self.assertIn("occluded", self.settle().block_reasons)

    def test_obstacle_on_observed_path_blocks(self):
        self.depth[38:43, 49:53] = .525
        self.assertFalse(self.settle().strike_permit)

    def test_missing_timing_or_geometry_validation_blocks(self):
        self.config.capture_timing_validated = False
        self.assertIn("calibration_unvalidated", self.settle().block_reasons)
        self.config.capture_timing_validated = True
        self.config.stroke_geometry_validated = False
        self.assertIn("no_stroke_calibration", self.settle().block_reasons)

    def test_bad_transform_rejected(self):
        with self.assertRaises(ValueError):
            StrikeConfig(camera_to_finger=np.zeros((4, 4))).checked()

    def test_path_geometry(self):
        result = distances_to_path(np.array([[5, 3, 0], [20, 0, 0]]), np.array([[0, 0, 0], [10, 0, 0]]))
        np.testing.assert_allclose(result, [3, 10])


if __name__ == "__main__":
    unittest.main()
