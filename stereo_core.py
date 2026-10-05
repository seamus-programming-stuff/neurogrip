"""Calibrated horizontal stereo in the rectified LEFT camera coordinate frame.

No learned model or OpenCV contrib module is required. Camera order is fixed:
left pixels minus corresponding right pixels must have positive disparity.
Calibration translations and Q reconstruction use millimetres; public depth Z
uses metres. Images are never silently resized or cropped after calibration.

Primary API references:
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
Right-search convention matches OpenCV's createRightMatcher implementation:
https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/src/disparity_filters.cpp
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import cv2
import numpy as np


def _image_size(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("image_size must contain [width,height]")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 32 for v in value):
        raise ValueError("image_size needs integer dimensions of at least 32 pixels")
    return tuple(int(v) for v in value)


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix with shape {shape}")
    return result.copy()


def _rotation(value, name):
    result = _array(value, (3, 3), name)
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(result), 1, atol=1e-5):
        raise ValueError(f"{name} must be a proper rigid rotation")
    return result


def _intrinsic(value, name):
    result = _array(value, (3, 3), name)
    if result[0, 0] <= 0 or result[1, 1] <= 0 or not np.allclose(result[2], [0, 0, 1], atol=1e-7):
        raise ValueError(f"{name} needs positive focal lengths and bottom row [0,0,1]")
    if abs(result[0, 1]) > 1e-7 or abs(result[1, 0]) > 1e-7:
        raise ValueError("Nonzero camera skew is unsupported")
    return result


def _distortion(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim not in (1, 2) or (result.ndim == 2 and 1 not in result.shape):
        raise ValueError(f"{name} must be a distortion vector")
    result = result.reshape(-1)
    if result.size not in (4, 5, 8, 12, 14) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires 4,5,8,12 or 14 finite coefficients")
    return result.copy()


def _roi(value, image_size, name):
    if value is None:
        return (0, 0, *image_size)
    if not isinstance(value, (tuple, list)) or len(value) != 4 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in value
    ):
        raise ValueError(f"{name} must be integer [x,y,width,height]")
    x, y, width, height = map(int, value)
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image_size[0] or y + height > image_size[1]:
        raise ValueError(f"{name} is empty or outside calibrated image bounds")
    return x, y, width, height


def _image(value, size, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim not in (2, 3):
        raise ValueError(f"{name} must be uint8 grayscale or BGR")
    if value.ndim == 3 and value.shape[2] != 3:
        raise ValueError(f"{name} must have exactly 3 BGR channels")
    if (value.shape[1], value.shape[0]) != size:
        raise ValueError(f"{name} resolution differs from calibrated image_size {size}; recalibrate this capture mode")
    return value


@dataclass
class StereoCalibration:
    image_size: tuple[int, int]
    left_source: str
    right_source: str
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T_mm: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    metadata: dict = field(default_factory=dict)
    roi1: tuple[int, int, int, int] | None = None
    roi2: tuple[int, int, int, int] | None = None
    _maps: object = field(default=None, init=False, repr=False)
    _map_masks: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.image_size = _image_size(self.image_size)
        for name in ("left_source", "right_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty source identifier")
        if self.left_source == self.right_source:
            raise ValueError("Left and right sources must identify different cameras")
        self.K1, self.K2 = _intrinsic(self.K1, "K1"), _intrinsic(self.K2, "K2")
        self.D1, self.D2 = _distortion(self.D1, "D1"), _distortion(self.D2, "D2")
        self.R, self.R1, self.R2 = (_rotation(getattr(self, name), name) for name in ("R", "R1", "R2"))
        translation = np.asarray(self.T_mm, dtype=np.float64)
        if translation.shape not in ((3,), (3, 1), (1, 3)) or not np.isfinite(translation).all():
            raise ValueError("T_mm must contain 3 finite translation coordinates in millimetres")
        self.T_mm = translation.reshape(3).copy()
        if not 10 < self.baseline_mm <= 2000:
            raise ValueError("Stereo baseline must be greater than 10 mm and no more than 2000 mm")
        self.P1, self.P2 = _array(self.P1, (3, 4), "P1"), _array(self.P2, (3, 4), "P2")
        self.Q = _array(self.Q, (4, 4), "Q")
        rectified_translation = self.R2 @ self.T_mm
        if abs(self.P2[1, 3]) > 1e-5 or abs(rectified_translation[1]) > self.baseline_mm * 1e-4:
            raise ValueError("Vertical stereo geometry is unsupported; rotate both streams consistently and recalibrate")
        if rectified_translation[0] >= -1e-5 or self.P2[0, 3] >= 0:
            raise ValueError("Camera order must produce positive left-minus-right disparity; swap left/right and recalibrate")
        if abs(rectified_translation[2]) > self.baseline_mm * 1e-4 or not np.allclose(self.R2 @ self.R, self.R1, atol=1e-5):
            raise ValueError("Rectification rotations are inconsistent with stereo R/T")
        rectified_intrinsic = _intrinsic(self.P1[:, :3], "P1 intrinsic")
        if not np.isclose(rectified_intrinsic[0, 0], rectified_intrinsic[1, 1], rtol=1e-5):
            raise ValueError("Rectified stereo requires equal horizontal/vertical focal lengths")
        if not np.allclose(self.P1[:, 3], 0, atol=1e-5) or not np.allclose(self.P2[:, :3], rectified_intrinsic, rtol=1e-5, atol=1e-5):
            raise ValueError("P1/P2 must use matching zero-disparity rectified intrinsics")
        expected_projection = np.array([rectified_intrinsic[0, 0] * rectified_translation[0], 0, 0])
        if not np.allclose(self.P2[:, 3], expected_projection, rtol=1e-5, atol=1e-4):
            raise ValueError("P2 translation is inconsistent with T_mm baseline")
        focal, cx, cy = rectified_intrinsic[0, 0], rectified_intrinsic[0, 2], rectified_intrinsic[1, 2]
        expected_q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, focal],
                               [0, 0, -1 / rectified_translation[0], 0]], dtype=np.float64)
        if not np.allclose(self.Q, expected_q, rtol=1e-5, atol=1e-6):
            raise ValueError("Q is inconsistent with rectified P1/P2 and millimetre baseline")
        self.roi1 = _roi(self.roi1, self.image_size, "roi1")
        self.roi2 = _roi(self.roi2, self.image_size, "roi2")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        try:
            self.metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain finite JSON values") from exc
        self._validate_metadata()

    def _validate_metadata(self):
        metadata = self.metadata
        for name in ("simulation", "timing_synchronized", "positive_disparity"):
            if name in metadata and not isinstance(metadata[name], bool):
                raise ValueError(f"metadata {name} must be boolean")
        if metadata.get("positive_disparity") is False:
            raise ValueError("metadata positive_disparity contradicts camera order")
        if "baseline_mm" in metadata:
            value = metadata["baseline_mm"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isclose(value, self.baseline_mm, rtol=1e-5, atol=1e-4):
                raise ValueError("metadata baseline_mm is inconsistent with T_mm")
        if "coordinate_frame" in metadata and metadata["coordinate_frame"] != "rectified_left_camera":
            raise ValueError("metadata coordinate_frame must be rectified_left_camera")
        if "board_units" in metadata and metadata["board_units"] != "mm":
            raise ValueError("metadata board_units must be mm")
        limits = {"left_rms_px": 1.5, "right_rms_px": 1.5, "stereo_rms_px": 1.5,
                  "epipolar_median_px": .75, "epipolar_p95_px": 1.5}
        for name, maximum in limits.items():
            if name in metadata:
                value = metadata[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= maximum:
                    raise ValueError(f"metadata {name} is invalid or exceeds calibration quality limit")
        if "epipolar_p95_px" in metadata and "epipolar_median_px" in metadata and metadata["epipolar_p95_px"] < metadata["epipolar_median_px"]:
            raise ValueError("metadata epipolar quantiles are inconsistent")
        if "paired_views" in metadata and (type(metadata["paired_views"]) is not int or metadata["paired_views"] < 12):
            raise ValueError("metadata paired_views requires at least 12 observations")

    @property
    def K_rect(self):
        return self.P1[:, :3].copy()

    @property
    def baseline_mm(self):
        return float(np.linalg.norm(self.T_mm))

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict) or type(values.get("schema_version")) is not int or values["schema_version"] != 1:
            raise ValueError("Expected stereo calibration schema_version 1")
        fields = ("image_size", "left_source", "right_source", "K1", "D1", "K2", "D2", "R", "T_mm", "R1", "R2", "P1", "P2", "Q")
        missing = set(fields) - values.keys()
        if missing:
            raise ValueError(f"Stereo calibration is missing {sorted(missing)}")
        if set(values) - set(fields) - {"schema_version", "metadata", "roi1", "roi2"}:
            raise ValueError("Unknown stereo calibration fields")
        return cls(**{name: values[name] for name in fields}, metadata=values.get("metadata", {}),
                   roi1=values.get("roi1"), roi2=values.get("roi2"))

    def save(self, path):
        values = {"schema_version": 1, "image_size": list(self.image_size), "left_source": self.left_source,
                  "right_source": self.right_source, "metadata": self.metadata, "roi1": list(self.roi1), "roi2": list(self.roi2)}
        for name in ("K1", "D1", "K2", "D2", "R", "T_mm", "R1", "R2", "P1", "P2", "Q"):
            values[name] = getattr(self, name).tolist()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _ensure_maps(self):
        if self._maps is not None:
            return
        first = cv2.initUndistortRectifyMap(self.K1, self.D1, self.R1, self.P1, self.image_size, cv2.CV_32FC1)
        second = cv2.initUndistortRectifyMap(self.K2, self.D2, self.R2, self.P2, self.image_size, cv2.CV_32FC1)
        width, height = self.image_size
        masks = []
        for (map_x, map_y), roi in zip((first, second), (self.roi1, self.roi2)):
            valid = np.isfinite(map_x) & np.isfinite(map_y) & (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
            region = np.zeros((height, width), bool)
            x, y, w, h = roi
            region[y:y+h, x:x+w] = True
            masks.append(valid & region)
        self._maps, self._map_masks = (first, second), tuple(masks)

    def rectify(self, left, right):
        _image(left, self.image_size, "left")
        _image(right, self.image_size, "right")
        if left.ndim != right.ndim:
            raise ValueError("Left and right images must have matching channel formats")
        self._ensure_maps()
        return (cv2.remap(left, *self._maps[0], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
                cv2.remap(right, *self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))


@dataclass(frozen=True)
class StereoResult:
    rectified_left: np.ndarray
    rectified_right: np.ndarray
    depth_m: np.ndarray
    disparity_px: np.ndarray
    valid_mask: np.ndarray
    visualization_depth: np.ndarray
    metric: bool = True
    backend: str = "OpenCV SGBM stereo"
    quality_valid_fraction: float = 0.0


class StereoDepth:
    """Metric rectified-left Z. Match validity is not an accuracy certificate."""

    def __init__(self, calibration, num_disparities=128, block_size=5, min_depth_mm=100, max_depth_mm=2000):
        if not isinstance(calibration, StereoCalibration):
            raise TypeError("calibration must be StereoCalibration")
        if isinstance(num_disparities, bool) or not isinstance(num_disparities, int) or num_disparities <= 0 or num_disparities % 16:
            raise ValueError("num_disparities must be positive and divisible by 16")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 3 or block_size > 21 or block_size % 2 == 0:
            raise ValueError("block_size must be odd and between 3 and 21")
        if num_disparities + block_size >= calibration.image_size[0]:
            raise ValueError("Disparity search is too wide for calibrated image width")
        if not np.isfinite([min_depth_mm, max_depth_mm]).all() or not 0 < min_depth_mm < max_depth_mm:
            raise ValueError("Depth limits must be finite, positive, increasing millimetre values")
        self.calibration = calibration
        self.num_disparities, self.block_size = num_disparities, block_size
        self.min_depth_mm, self.max_depth_mm = float(min_depth_mm), float(max_depth_mm)
        self.lr_tolerance_px = 1.0
        options = dict(numDisparities=num_disparities, blockSize=block_size,
                       P1=8 * block_size**2, P2=32 * block_size**2,
                       disp12MaxDiff=1000000, preFilterCap=31, uniquenessRatio=10,
                       speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self._left_matcher = cv2.StereoSGBM_create(minDisparity=0, **options)
        self._right_min = -num_disparities + 1
        self._right_matcher = cv2.StereoSGBM_create(minDisparity=self._right_min, **options)
        self._support_masks = None

    def compute(self, left, right):
        rectified_left, rectified_right = self.calibration.rectify(left, right)
        gray_left = rectified_left if rectified_left.ndim == 2 else cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = rectified_right if rectified_right.ndim == 2 else cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
        # OpenCV fixed-point disparities have4 fractional bits, including the
        # right matcher's negative-disparity search and invalid sentinel.
        disparity = self._left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        reverse = self._right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0
        height, width = disparity.shape
        yy, xx = np.indices(disparity.shape)
        right_x = np.rint(xx - disparity).astype(np.int32)
        in_bounds = (right_x >= 0) & (right_x < width)
        lookup_x = np.clip(right_x, 0, width - 1)
        corresponding_reverse = reverse[yy, lookup_x]
        if self._support_masks is None:
            # The whole matching block must be inside each valid remap region,
            # not only its centre. This excludes distorted/cropped border fill.
            kernel = np.ones((self.block_size, self.block_size), np.uint8)
            self._support_masks = tuple(cv2.erode(mask.astype(np.uint8), kernel,
                                                  borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
                                        for mask in self.calibration._map_masks)
        left_valid, right_valid = self._support_masks
        valid = ((disparity > 0) & (disparity < self.num_disparities) & in_bounds & left_valid
                 & right_valid[yy, lookup_x] & (corresponding_reverse >= self._right_min)
                 & (corresponding_reverse < 0) & (np.abs(disparity + corresponding_reverse) <= self.lr_tolerance_px))
        # This excludes incomplete block/search support as well as invalid
        # remap pixels. Checking both maps at correspondence addresses occlusions.
        roi = cv2.getValidDisparityROI(self.calibration.roi1, self.calibration.roi2, 0,
                                      self.num_disparities, self.block_size)
        x, y, roi_w, roi_h = roi
        roi_mask = np.zeros_like(valid)
        roi_mask[y:y+roi_h, x:x+roi_w] = True
        valid &= roi_mask
        xyz_mm = cv2.reprojectImageTo3D(disparity, self.calibration.Q, handleMissingValues=False)
        depth_mm = xyz_mm[:, :, 2]
        valid &= np.isfinite(xyz_mm).all(axis=2) & (depth_mm >= self.min_depth_mm) & (depth_mm <= self.max_depth_mm)
        depth_m = np.where(valid, depth_mm / 1000.0, np.nan).astype(np.float32)
        disparity_px = np.where(valid, disparity, np.nan).astype(np.float32)
        return StereoResult(rectified_left, rectified_right, depth_m, disparity_px, valid, depth_m.copy(),
                            quality_valid_fraction=float(valid.mean()))


def check_epipolar_alignment(rectified_left, rectified_right, *, min_matches=20):
    """Detect demonstrated gross vertical disagreement; never certify geometry.

    Insufficient features returns aligned=None. Horizontal camera movement or
    a changed baseline can escape this test, so a passing result cannot replace
    recalibration after a camera moves. Run on static background when possible.
    """
    if rectified_left.shape != rectified_right.shape:
        raise ValueError("Alignment images must have matching shapes")
    _image(rectified_left, (rectified_left.shape[1], rectified_left.shape[0]), "left")
    _image(rectified_right, (rectified_right.shape[1], rectified_right.shape[0]), "right")
    gray1 = rectified_left if rectified_left.ndim == 2 else cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray2 = rectified_right if rectified_right.ndim == 2 else cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=1400)
    points1, descriptor1 = detector.detectAndCompute(gray1, None)
    points2, descriptor2 = detector.detectAndCompute(gray2, None)
    result = {"aligned": None, "status": "insufficient_features", "matches": 0,
              "median_vertical_error_px": None, "p90_vertical_error_px": None}
    if descriptor1 is None or descriptor2 is None:
        return result
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptor1, descriptor2, k=2)
    accepted = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 64 and pair[0].distance < .75 * pair[1].distance]
    # Prevent many left features reusing the same right feature.
    unique = {match.trainIdx: match for match in sorted(accepted, key=lambda match: match.distance, reverse=True)}
    result["matches"] = len(unique)
    if len(unique) < min_matches:
        return result
    vertical = np.array([abs(points1[match.queryIdx].pt[1] - points2[match.trainIdx].pt[1]) for match in unique.values()])
    median, p90 = np.percentile(vertical, [50, 90])
    aligned = bool(median <= 2.0 and p90 <= 4.0)
    result.update(aligned=aligned, status="aligned" if aligned else "misaligned",
                  median_vertical_error_px=float(median), p90_vertical_error_px=float(p90))
    return result
