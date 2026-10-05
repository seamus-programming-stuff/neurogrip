"""Conservative, explicitly taught foreground tracking (OpenCV + NumPy only)."""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackResult:
    bbox: Optional[Tuple[int, int, int, int]]
    mask: Optional[np.ndarray]
    valid: bool
    quality: float
    reason: str


class RegionTracker:
    """Teach a box containing the object and a small amount of background.

    The learned foreground mask is transported by KLT/RANSAC and checked against
    the learned appearance. A failed tracker stays invalid until re-taught.
    Boxes use (x, y, width, height); valid masks contain 0/255 uint8 pixels.
    """

    def __init__(self, min_points=10, min_ncc=0.45, max_fb_error=1.5):
        self.min_points = max(6, int(min_points))
        self.min_ncc = float(min_ncc)
        self.max_fb_error = float(max_fb_error)
        self._active = False
        self._reason = "not_initialized"

    def _invalid(self, reason):
        self._active = False
        self._reason = reason
        return TrackResult(None, None, False, 0.0, reason)

    @staticmethod
    def _gray(bgr):
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected a nonempty uint8 BGR image")
        if bgr.dtype != np.uint8 or min(bgr.shape[:2]) < 16:
            raise ValueError("Expected a nonempty uint8 BGR image")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _features(self, gray, mask):
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=180, qualityLevel=0.015, minDistance=5,
            mask=mask, blockSize=5, useHarrisDetector=False)

    @staticmethod
    def _bbox(mask):
        pts = cv2.findNonZero(mask)
        return None if pts is None else tuple(int(v) for v in cv2.boundingRect(pts))

    def initialize(self, bgr, bbox):
        self._active = False
        try:
            gray = self._gray(bgr)
            x, y, w, h = [int(v) for v in bbox]
        except (ValueError, TypeError):
            return self._invalid("invalid_image_or_bbox")
        height, width = gray.shape
        if w < 16 or h < 16 or x < 0 or y < 0 or x + w > width or y + h > height:
            return self._invalid("bbox_out_of_bounds_or_too_small")
        # GrabCut needs definite background beyond the taught rectangle.
        if x == 0 and y == 0 and w == width and h == height:
            return self._invalid("teach_box_requires_background_margin")
        gc_mask = np.zeros(gray.shape, np.uint8)
        try:
            cv2.grabCut(bgr, gc_mask, (x, y, w, h), np.zeros((1, 65)),
                        np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return self._invalid("foreground_segmentation_failed")
        foreground = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
        if count < 2:
            return self._invalid("foreground_segmentation_empty")
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
        area = int(np.count_nonzero(mask))
        if area < max(100, int(w * h * 0.04)):
            return self._invalid("foreground_too_small")
        if np.count_nonzero(cv2.erode(mask, np.ones((5, 5), np.uint8))) < 64:
            return self._invalid("foreground_too_thin")
        interior = cv2.erode(mask, np.ones((3, 3), np.uint8))
        points = self._features(gray, interior)
        if points is None or len(points) < self.min_points:
            return self._invalid("insufficient_foreground_texture")
        support = cv2.contourArea(cv2.convexHull(points))
        if support < 0.025 * area:
            return self._invalid("features_too_concentrated")
        self._reference = gray.copy()
        self._reference_mask = mask.copy()
        self._mask = mask
        self._previous = gray
        self._points = points
        self._transform = np.eye(3, dtype=np.float64)
        self._reference_area = area
        self._active = True
        self._reason = "ok"
        quality = min(1.0, len(points) / float(2 * self.min_points))
        return TrackResult(self._bbox(mask), mask.copy(), True, quality, "ok")

    def update(self, bgr):
        if not self._active:
            return TrackResult(None, None, False, 0.0, self._reason)
        try:
            gray = self._gray(bgr)
        except ValueError:
            return self._invalid("invalid_image")
        if gray.shape != self._previous.shape:
            return self._invalid("image_size_changed")
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        try:
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                self._previous, gray, self._points, None, **lk)
            if next_points is None or status is None or error is None:
                return self._invalid("optical_flow_failed")
            back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._previous, next_points, None, **lk)
            if back_points is None or back_status is None:
                return self._invalid("reverse_flow_failed")
            fb = np.linalg.norm(back_points - self._points, axis=2).ravel()
            good = ((status.ravel() == 1) & (back_status.ravel() == 1)
                    & np.isfinite(fb) & (fb <= self.max_fb_error)
                    & (error.ravel() < 25.0))
            old = self._points.reshape(-1, 2)[good]
            new = next_points.reshape(-1, 2)[good]
            if len(old) < self.min_points:
                return self._invalid("too_few_consistent_features")
            affine, inliers = cv2.estimateAffinePartial2D(
                old, new, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                maxIters=2000, confidence=0.99, refineIters=10)
        except cv2.error:
            return self._invalid("optical_flow_or_fit_failed")
        if affine is None or inliers is None or not np.isfinite(affine).all():
            return self._invalid("similarity_fit_failed")
        inlier = inliers.ravel().astype(bool)
        ratio = float(np.count_nonzero(inlier)) / len(old)
        if np.count_nonzero(inlier) < self.min_points or ratio < 0.6:
            return self._invalid("inconsistent_object_motion")
        scale = float(np.hypot(affine[0, 0], affine[1, 0]))
        angle = abs(float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))))
        if not 0.8 <= scale <= 1.25 or angle > 45.0:
            return self._invalid("implausible_motion")
        step = np.eye(3, dtype=np.float64)
        step[:2] = affine
        transform = step.dot(self._transform)
        total_scale = float(np.hypot(transform[0, 0], transform[1, 0]))
        if not 0.35 <= total_scale <= 3.0:
            return self._invalid("object_scale_out_of_range")
        height, width = gray.shape
        mask = cv2.warpAffine(self._reference_mask, transform[:2], (width, height),
                              flags=cv2.INTER_NEAREST, borderValue=0)
        visible_area = int(np.count_nonzero(mask))
        expected_area = self._reference_area * total_scale * total_scale
        if visible_area < 0.75 * expected_area:
            return self._invalid("object_leaving_image")
        if cv2.contourArea(cv2.convexHull(new[inlier])) < 0.025 * visible_area:
            return self._invalid("features_too_concentrated")
        # Bring current pixels back to their learned positions. This rejects
        # a moving box that has drifted onto unrelated texture or lost its object.
        aligned = cv2.warpAffine(gray, transform[:2], (width, height),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        test_mask = cv2.erode(self._reference_mask, np.ones((5, 5), np.uint8)) > 0
        reference_pixels = self._reference[test_mask].astype(np.float32)
        current_pixels = aligned[test_mask].astype(np.float32)
        reference_pixels -= reference_pixels.mean()
        current_pixels -= current_pixels.mean()
        denominator = float(np.linalg.norm(reference_pixels) * np.linalg.norm(current_pixels))
        ncc = (float(np.dot(reference_pixels, current_pixels)) / denominator
               if denominator > 1e-6 else 0.0)
        if not np.isfinite(ncc) or ncc < self.min_ncc:
            return self._invalid("learned_appearance_lost")
        points = self._features(gray, cv2.erode(mask, np.ones((3, 3), np.uint8)))
        if points is None or len(points) < self.min_points:
            return self._invalid("insufficient_foreground_texture")
        self._previous = gray
        self._points = points
        self._mask = mask
        self._transform = transform
        quality = float(np.clip(min(ratio, ncc, len(points) / float(2 * self.min_points)), 0, 1))
        return TrackResult(self._bbox(mask), mask.copy(), True, quality, "ok")
