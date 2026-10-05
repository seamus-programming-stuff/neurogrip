"""Fit/validate a millimetre stereo rig from paired checkerboard observations.

Board points use measured millimetres. Left/right point order must correspond
to the SAME stationary board corners in each pair. Collect varied positions,
distances and tilts. Quality thresholds reject poor fits; they do not prove
timing synchronization, fixed camera mounts, or final distance accuracy.

OpenCV primary reference for calibrateCamera, stereoCalibrate FIX_INTRINSIC,
stereoRectify, and undistortPoints:
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
"""
from __future__ import annotations

import cv2
import numpy as np

from stereo_core import StereoCalibration, _image_size


MIN_PAIRED_VIEWS = 12
MAX_RMS_PX = 1.5
MAX_EPIPOLAR_MEDIAN_PX = .75
MAX_EPIPOLAR_P95_PX = 1.5


def _points(value, dimensions, name):
    points = np.asarray(value, dtype=np.float32)
    if points.ndim == 3 and points.shape[1:] == (1, dimensions):
        points = points[:, 0]
    if points.ndim != 2 or points.shape[1] != dimensions or len(points) < 6 or not np.isfinite(points).all():
        raise ValueError(f"{name} requires at least 6 finite {dimensions}D points")
    return np.ascontiguousarray(points)


def calibrate_stereo(object_points_mm, left_points, right_points, image_size,
                     left_source, right_source, metadata=None):
    size = _image_size(image_size)
    if not (len(object_points_mm) == len(left_points) == len(right_points)) or len(object_points_mm) < MIN_PAIRED_VIEWS:
        raise ValueError(f"Collect at least {MIN_PAIRED_VIEWS} matched board views for both cameras")
    objects, first, second = [], [], []
    for index, (world, left, right) in enumerate(zip(object_points_mm, left_points, right_points)):
        world = _points(world, 3, f"object_points_mm[{index}]")
        left = _points(left, 2, f"left_points[{index}]")
        right = _points(right, 2, f"right_points[{index}]")
        if not len(world) == len(left) == len(right):
            raise ValueError("Board corner counts must match within every pair")
        if np.linalg.matrix_rank(world - world.mean(axis=0), tol=1e-4) < 2:
            raise ValueError("Calibration board coordinates are collinear or degenerate")
        for pixels in (left, right):
            if np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= size[0]) or np.any(pixels[:, 1] < 0) or np.any(pixels[:, 1] >= size[1]):
                raise ValueError("Detected board corners must lie within calibrated image bounds")
        objects.append(world)
        first.append(left.reshape(-1, 1, 2))
        second.append(right.reshape(-1, 1, 2))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    try:
        rms1, K1, D1, _, _ = cv2.calibrateCamera(objects, first, size, None, None, criteria=criteria)
        rms2, K2, D2, _, _ = cv2.calibrateCamera(objects, second, size, None, None, criteria=criteria)
        stereo_rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
            objects, first, second, K1, D1, K2, D2, size,
            flags=cv2.CALIB_FIX_INTRINSIC, criteria=criteria)
        if not np.isfinite([rms1, rms2, stereo_rms]).all() or max(rms1, rms2, stereo_rms) > MAX_RMS_PX:
            raise ValueError(f"Calibration reprojection RMS is too high: left {rms1:.2f}px/right {rms2:.2f}px/stereo {stereo_rms:.2f}px")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if abs(P2[1, 3]) > 1e-5:
            raise ValueError("Vertical stereo geometry is unsupported; rotate both streams consistently and recalibrate")
        if P2[0, 3] >= 0:
            raise ValueError("Camera order must produce positive left-minus-right disparity; swap left/right and recalibrate")
        epipolar_errors = []
        for left, right in zip(first, second):
            a = cv2.undistortPoints(left, K1, D1, R=R1, P=P1).reshape(-1, 2)
            b = cv2.undistortPoints(right, K2, D2, R=R2, P=P2).reshape(-1, 2)
            epipolar_errors.extend(np.abs(a[:, 1] - b[:, 1]).tolist())
        median, p95 = np.percentile(epipolar_errors, [50, 95])
        if not np.isfinite([median, p95]).all() or median > MAX_EPIPOLAR_MEDIAN_PX or p95 > MAX_EPIPOLAR_P95_PX:
            raise ValueError(f"Rectified epipolar error is too high: median {median:.2f}px/p95 {p95:.2f}px")
    except cv2.error as exc:
        raise ValueError(f"OpenCV could not calibrate this board dataset: {exc}") from exc
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    quality = dict(metadata or {})
    quality.update(left_rms_px=float(rms1), right_rms_px=float(rms2), stereo_rms_px=float(stereo_rms),
                   epipolar_median_px=float(median), epipolar_p95_px=float(p95),
                   paired_views=len(objects), baseline_mm=float(np.linalg.norm(T)),
                   board_units="mm", coordinate_frame="rectified_left_camera", positive_disparity=True,
                   timing_synchronized=False)
    return StereoCalibration(size, left_source, right_source, K1, D1, K2, D2, R, T,
                             R1, R2, P1, P2, Q, quality, tuple(roi1), tuple(roi2))
