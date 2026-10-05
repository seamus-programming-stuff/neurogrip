"""Deterministic synthetic geometry for the UI/demo, never a learned prediction."""
import math
import cv2
import numpy as np
from depth_backend import DepthResult
from proximity import StrikeConfig


K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)


def demo_config():
    return StrikeConfig(
        min_region_pixels=30, tip_radius_mm=8, obstacle_margin_mm=2,
        gap_error_mm=1, measurement_validated=True, capture_timing_validated=True,
        source_delay_bound_ms=0, stroke_geometry_validated=True,
        max_target_speed_mm_s=1000, max_finger_speed_mm_s=1000,
        camera_to_finger=[[1, 0, 0, 40], [0, 1, 0, 0], [0, 0, 1, -600], [0, 0, 0, 1]],
        strokes={"FORWARDS": [[0, 0, 0], [105, 0, 50]],
                 "UPRIGHT": [[0, 0, 0], [0, -90, 50]],
                 "BACKWARDS": [[0, 0, 0], [-105, 0, 50]]},
    ).checked()


def scene(seconds):
    bgr = np.full((480, 640, 3), (30, 35, 44), np.uint8)
    for x in range(0, 640, 40):
        cv2.line(bgr, (x, 0), (x, 479), (43, 48, 57), 1)
    for y in range(0, 480, 40):
        cv2.line(bgr, (0, y), (639, y), (43, 48, 57), 1)
    finger, target = np.zeros((480, 640), np.uint8), np.zeros((480, 640), np.uint8)
    finger[226:254, 270:290] = 255
    target[220:260, 365:395] = 255
    rng = np.random.default_rng(4)
    bgr[finger > 0] = rng.integers(90, 230, (np.count_nonzero(finger), 3), dtype=np.uint8)
    bgr[target > 0] = rng.integers(50, 210, (np.count_nonzero(target), 3), dtype=np.uint8)
    depth = np.full((480, 640), 1.4, np.float32)
    depth[finger > 0] = .6
    depth[target > 0] = .65 + .12 * (1 + math.sin(seconds * .7)) / 2
    cv2.putText(bgr, "SYNTHETIC DEMO - NO ACTUATOR PERMISSION", (14, 35), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 1)
    return bgr, DepthResult(depth, np.ones_like(depth, bool), "synthetic ground truth", True, depth), finger, target
