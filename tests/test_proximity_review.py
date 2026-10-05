"""Regression cases from independent review of visible-tip reachability."""
import unittest

import numpy as np

from proximity import ProximityEngine, StrikeConfig, distances_to_path, points_from_depth
from protocol import Direction, Mode


class ProximityReviewTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.ones((80, 120), np.float32)
        self.finger = np.zeros(self.depth.shape, bool)
        self.target = np.zeros(self.depth.shape, bool)
        self.finger[35:45, 35:45] = True
        self.target[35:45, 55:65] = True
        self.depth[self.finger] = 0.5
        self.depth[self.target] = 0.55
        self.k = np.array([[100., 0., 60.], [0., 100., 40.], [0., 0., 1.]])
        self.config = StrikeConfig(
            gap_error_mm=1., measurement_validated=True,
            capture_timing_validated=True, source_delay_bound_ms=0.,
            camera_to_finger=[[1, 0, 0, 100], [0, 1, 0, 0],
                              [0, 0, 1, -500], [0, 0, 0, 1]],
            strokes={"FORWARDS": [[0., 0., 0.], [100., 0., 50.]]},
            stroke_geometry_validated=True, tip_radius_mm=10.,
            max_target_speed_mm_s=100.,
        )
        self.engine = ProximityEngine(self.config)

    def settle(self):
        for now in range(1, 6):
            result = self.engine.evaluate(
                self.depth, self.finger, self.target, self.k,
                metric=True, armed=True, now=float(now), mode=Mode.LIVE,
            )
        return result

    def test_unknown_depth_inside_tip_corridor_blocks_permission(self):
        # Centreline projects to y=40. This unknown patch is beside it but
        # within the projected physical tip radius at the middle of the path.
        self.depth[41:44, 47:52] = np.nan
        result = self.settle()
        self.assertFalse(result.strike_permit,
                         "Unknown visible swept volume must not be discarded as empty space")

    def test_near_path_cannot_expand_physical_reach(self):
        # All target points remain >14 mm from this path. A 10 mm tip cannot
        # contact them; proximity to the path must not grant physical reach.
        path = np.array([[0., 0., 0.], [100., 40., 50.]])
        self.config.strokes = {Direction.FORWARDS: path}
        target_points = points_from_depth(self.depth, self.target, self.k)
        target_points += [100., 0., -500.]
        self.assertGreater(np.min(distances_to_path(target_points, path)),
                           self.config.tip_radius_mm)
        result = self.settle()
        self.assertFalse(result.strike_permit,
                         "A stroke geometrically missing the target cannot have a permit")

    def test_outside_surface_hiding_side_of_corridor_blocks(self):
        # A closer foreground surface hides the off-axis part of the sweep.
        # It is too far away in XYZ to be an intersecting obstacle itself.
        self.depth[41:44, 47:52] = .3
        self.assertFalse(self.settle().strike_permit)

    def test_moving_finger_blocks_even_when_target_is_stationary(self):
        self.assertTrue(self.settle().strike_permit)
        self.depth[self.finger] = 0.51
        result = self.engine.evaluate(
            self.depth, self.finger, self.target, self.k,
            metric=True, armed=True, now=5.001, mode=Mode.LIVE,
        )
        self.assertGreater(result.finger_speed_mm_s, self.config.max_finger_speed_mm_s)
        self.assertEqual(result.target_speed_mm_s, 0.)
        self.assertFalse(result.strike_permit)
        self.assertIn("low_quality", result.block_reasons)

    def test_one_finger_pixel_near_start_does_not_match_tip_pose(self):
        self.depth[self.finger] = 1.
        self.finger[:] = False
        self.finger[35:45, 20:40] = True
        self.depth[self.finger] = 0.5
        points = points_from_depth(self.depth, self.finger, self.k)
        points += [100., 0., -500.]
        self.assertLess(np.min(np.linalg.norm(points, axis=1)),
                        self.config.start_tolerance_mm)
        self.assertGreater(np.linalg.norm(np.median(points, axis=0)),
                           self.config.start_tolerance_mm)
        self.assertFalse(self.settle().strike_permit)


if __name__ == "__main__":
    unittest.main()
