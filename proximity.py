"""Visible-surface geometry and conservative, debounced strike recommendations.

The monocular network estimates depth. It does not prove collision-free motion.
No actuator I/O is performed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np

from protocol import Direction, Mode


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} and finite")
    return value


@dataclass
class StrikeConfig:
    valid_for_ms: int = 500
    min_quality: float = 0.6
    min_valid_fraction: float = 0.75
    min_region_pixels: int = 30
    enable_hits: int = 3
    max_target_speed_mm_s: float = 50.0
    max_finger_speed_mm_s: float = 50.0
    min_depth_mm: float = 100.0
    max_depth_mm: float = 2000.0
    depth_scale: float = 1.0
    gap_error_mm: float | None = None
    measurement_validated: bool = False
    capture_timing_validated: bool = False
    source_delay_bound_ms: float | None = None
    camera_to_finger: np.ndarray | None = None
    strokes: dict[Direction, np.ndarray] = field(default_factory=dict)
    tip_radius_mm: float = 5.0
    start_tolerance_mm: float = 15.0
    obstacle_margin_mm: float = 10.0
    stroke_geometry_validated: bool = False

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("config must be a JSON object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**values).checked()

    def checked(self):
        for name in ("valid_for_ms", "min_region_pixels", "enable_hits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.valid_for_ms > 65535:
            raise ValueError("valid_for_ms exceeds wire representation")
        for name in ("min_quality", "min_valid_fraction"):
            value = _positive(getattr(self, name), name)
            if value > 1:
                raise ValueError(f"{name} must be at most 1")
        for name in ("max_target_speed_mm_s", "max_finger_speed_mm_s", "min_depth_mm", "max_depth_mm", "depth_scale",
                     "tip_radius_mm", "start_tolerance_mm", "obstacle_margin_mm"):
            setattr(self, name, _positive(getattr(self, name), name))
        if self.max_depth_mm <= self.min_depth_mm:
            raise ValueError("max_depth_mm must exceed min_depth_mm")
        for name in ("measurement_validated", "stroke_geometry_validated", "capture_timing_validated"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.gap_error_mm is not None:
            self.gap_error_mm = _positive(self.gap_error_mm, "gap_error_mm", zero=True)
        if self.source_delay_bound_ms is not None:
            self.source_delay_bound_ms = _positive(self.source_delay_bound_ms, "source_delay_bound_ms", zero=True)
        if self.camera_to_finger is not None:
            transform = np.asarray(self.camera_to_finger, dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("camera_to_finger must be a finite 4x4 rigid transform (translation in mm)")
            rotation = transform[:3, :3]
            if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
                raise ValueError("camera_to_finger must be a rigid rotation/translation")
            self.camera_to_finger = transform
        checked = {}
        for name, path in self.strokes.items():
            try:
                direction = name if isinstance(name, Direction) else Direction[name.upper()]
            except (KeyError, AttributeError):
                raise ValueError("stroke keys must be BACKWARDS, UPRIGHT, FORWARDS")
            points = np.asarray(path, dtype=np.float64)
            if direction == Direction.UNKNOWN or points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError("each stroke needs at least two finite XYZ tip positions in mm")
            if np.max(np.abs(points)) > 10000:
                raise ValueError("stroke coordinates exceed bench geometry range")
            checked[direction] = points
        self.strokes = checked
        return self


@dataclass
class Measurement:
    gap_mm: float | None = None
    finger_depth_mm: float | None = None
    target_depth_mm: float | None = None
    direction: Direction = Direction.UNKNOWN
    reachable: bool = False
    candidate_permit: bool = False
    strike_permit: bool = False
    quality: float = 0.0
    scale_valid: bool = False
    block_reasons: tuple[str, ...] = ()
    target_speed_mm_s: float | None = None
    finger_speed_mm_s: float | None = None
    path_distance_mm: float | None = None


def validate_intrinsics(intrinsics):
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("K must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("K must have positive focal lengths and bottom row [0,0,1]")
    if not np.isclose(matrix[0, 1], 0) or not np.isclose(matrix[1, 0], 0):
        raise ValueError("nonzero camera skew is unsupported")
    return matrix


def points_from_depth(depth_m, mask, intrinsics, max_points=400):
    """Unproject finite positive optical-axis depths into camera XYZ in mm."""
    depth = np.asarray(depth_m)
    mask = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth and mask must have matching HxW shape")
    matrix = validate_intrinsics(intrinsics)
    y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if len(x) > max_points:
        indices = np.linspace(0, len(x) - 1, max_points).astype(int)
        y, x = y[indices], x[indices]
    z = depth[y, x].astype(np.float64) * 1000.0
    return np.column_stack(((x - matrix[0, 2]) * z / matrix[0, 0],
                            (y - matrix[1, 2]) * z / matrix[1, 1], z))


def _transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def distances_to_path(points, path):
    """Euclidean distances to the polyline, in the same coordinate system."""
    best = np.full(len(points), np.inf)
    for start, end in zip(path[:-1], path[1:]):
        vector = end - start
        squared_length = float(vector @ vector)
        if squared_length < 1e-12:
            distance = np.linalg.norm(points - start, axis=1)
        else:
            proportion = np.clip((points - start) @ vector / squared_length, 0, 1)
            distance = np.linalg.norm(points - (start + proportion[:, None] * vector), axis=1)
        best = np.minimum(best, distance)
    return best


class ProximityEngine:
    def __init__(self, config):
        self.config = config.checked()
        self._hits = 0
        self._direction = Direction.UNKNOWN
        self._previous_target = None
        self._previous_finger = None
        self._previous_time = None

    def reset(self):
        self._hits = 0
        self._direction = Direction.UNKNOWN
        self._previous_target = self._previous_finger = self._previous_time = None

    def _finish(self, result, reasons, mode):
        reasons = list(dict.fromkeys(reasons))
        eligible = not reasons and result.direction != Direction.UNKNOWN
        if eligible and result.direction == self._direction:
            self._hits += 1
        elif eligible:
            self._hits, self._direction = 1, result.direction
        else:
            self._hits, self._direction = 0, Direction.UNKNOWN
        result.candidate_permit = bool(eligible and self._hits >= self.config.enable_hits)
        result.strike_permit = result.candidate_permit and mode == Mode.LIVE
        if mode == Mode.SIMULATION:
            reasons.append("simulation")
        result.block_reasons = tuple(reasons)
        return result

    def evaluate(self, depth, finger_mask, target_mask, intrinsics, *, metric,
                 finger_valid=True, target_valid=True, tracking_quality=1.0,
                 capture_age_ms=0, now=0.0, armed=False, mode=Mode.LIVE):
        config = self.config
        result, reasons = Measurement(), []
        if capture_age_ms >= config.valid_for_ms:
            reasons.append("stale_frame")
        if not armed:
            reasons.append("controller_unarmed")
        if not finger_valid:
            reasons.append("finger_missing")
        if not target_valid:
            reasons.append("target_missing")
        if not metric or intrinsics is None:
            reasons.append("uncalibrated")
            return self._finish(result, reasons, mode)
        try:
            intrinsics = validate_intrinsics(intrinsics)
            if not math.isfinite(tracking_quality) or not 0 <= tracking_quality <= 1:
                raise ValueError("invalid tracking quality")
            depth = np.asarray(depth, dtype=np.float32) * config.depth_scale
            finger_mask = np.asarray(finger_mask, dtype=bool)
            target_mask = np.asarray(target_mask, dtype=bool)
            if depth.ndim != 2 or finger_mask.shape != depth.shape or target_mask.shape != depth.shape:
                raise ValueError("invalid shape")
            if np.any(finger_mask & target_mask):
                reasons.append("occluded")
            valid = np.isfinite(depth) & (depth > 0)
            qualities = []
            for region in (finger_mask, target_mask):
                count = int(np.count_nonzero(region))
                valid_count = int(np.count_nonzero(region & valid))
                fraction = valid_count / max(1, count)
                qualities.append(fraction)
                if valid_count < config.min_region_pixels or fraction < config.min_valid_fraction:
                    reasons.append("unknown_depth")
            result.quality = float(min(*qualities, max(0.0, min(1.0, tracking_quality))))
            if result.quality < config.min_quality:
                reasons.append("low_quality")
            if any(reason in reasons for reason in ("unknown_depth", "finger_missing", "target_missing", "occluded")):
                return self._finish(result, reasons, mode)
            # Trim depth outliers inside each foreground mask; never fill missing depth.
            masks = []
            for region in (finger_mask, target_mask):
                samples = depth[region & valid]
                low, high = np.percentile(samples, [5, 95])
                masks.append(region & valid & (depth >= low) & (depth <= high))
            finger_points = points_from_depth(depth, masks[0], intrinsics)
            target_points = points_from_depth(depth, masks[1], intrinsics)
            result.finger_depth_mm = float(np.median(finger_points[:, 2]))
            result.target_depth_mm = float(np.median(target_points[:, 2]))
            if any(value < config.min_depth_mm or value > config.max_depth_mm
                   for value in (result.finger_depth_mm, result.target_depth_mm)):
                reasons.append("depth_out_of_range")
                return self._finish(result, reasons, mode)
            distances = np.linalg.norm(finger_points[:, None, :] - target_points[None, :, :], axis=2)
            # 5th percentile of per-finger nearest distances is less sensitive to one bad pixel.
            result.gap_mm = float(np.percentile(np.min(distances, axis=1), 5))
            if not math.isfinite(result.gap_mm) or result.gap_mm > 1_000_000:
                result.gap_mm = None
                reasons.append("invalid_geometry")
                return self._finish(result, reasons, mode)
            result.scale_valid = bool(config.measurement_validated and config.gap_error_mm is not None)
            target_center = np.median(target_points, axis=0)
            finger_center = np.median(finger_points, axis=0)
            if self._previous_time is not None and now > self._previous_time:
                result.target_speed_mm_s = float(np.linalg.norm(target_center - self._previous_target) / (now - self._previous_time))
                result.finger_speed_mm_s = float(np.linalg.norm(finger_center - self._previous_finger) / (now - self._previous_time))
                if result.target_speed_mm_s > config.max_target_speed_mm_s:
                    reasons.append("low_quality")
                if result.finger_speed_mm_s > config.max_finger_speed_mm_s:
                    reasons.append("low_quality")
            else:
                # First frame cannot establish motion; require another independent frame.
                reasons.append("low_quality")
            self._previous_target, self._previous_finger, self._previous_time = target_center, finger_center, now
            if not config.measurement_validated or config.gap_error_mm is None:
                reasons.append("calibration_unvalidated")
            if not config.capture_timing_validated or config.source_delay_bound_ms is None:
                reasons.append("calibration_unvalidated")
            if config.camera_to_finger is None or not config.strokes or not config.stroke_geometry_validated:
                reasons.append("no_stroke_calibration")
                return self._finish(result, reasons, mode)
            finger_local = _transform(finger_points, config.camera_to_finger)
            target_local = _transform(target_points, config.camera_to_finger)
            # Visible obstacles only: hidden surfaces and other finger links require controller interlocks.
            other_mask = valid & ~(finger_mask | target_mask)
            obstacles = _transform(points_from_depth(depth, other_mask, intrinsics, max_points=depth.size), config.camera_to_finger)
            candidates = []
            error = config.gap_error_mm or 0.0
            for direction, path in config.strokes.items():
                start_distance = float(np.linalg.norm(np.median(finger_local, axis=0) - path[0]))
                if start_distance + error > config.start_tolerance_mm:
                    continue
                target_distance = float(np.percentile(distances_to_path(target_local, path), 5))
                if target_distance + error > config.tip_radius_mm:
                    continue
                # Sampled tip path must project into the image and have observed depth.
                camera_path = _transform(path, np.linalg.inv(config.camera_to_finger))
                visible = True
                for start, end in zip(camera_path[:-1], camera_path[1:]):
                    # At most one projected pixel between samples, capped at 5mm.
                    step_mm = min(5.0, max(.1, min(start[2], end[2]) / max(intrinsics[0, 0], intrinsics[1, 1])))
                    count = max(2, int(np.ceil(np.linalg.norm(end - start) / step_mm)) + 1)
                    samples = np.linspace(start, end, count)
                    if (samples[:, 2] <= 0).any():
                        visible = False
                        break
                    u = np.rint(intrinsics[0, 0] * samples[:, 0] / samples[:, 2] + intrinsics[0, 2]).astype(int)
                    v = np.rint(intrinsics[1, 1] * samples[:, 1] / samples[:, 2] + intrinsics[1, 2]).astype(int)
                    if (u < 0).any() or (u >= depth.shape[1]).any() or (v < 0).any() or (v >= depth.shape[0]).any():
                        visible = False
                        break
                    if not valid[v, u].all():
                        visible = False
                        break
                    # Depth surface in front of path hides that section; do not authorize unseen travel.
                    if (depth[v, u] * 1000 + config.tip_radius_mm + error < samples[:, 2]).any():
                        visible = False
                        break
                    # A depth hole beside the centreline can hide an obstacle.
                    # Require observed coverage over the projected swept-tip disk.
                    radius = config.tip_radius_mm + config.obstacle_margin_mm + error
                    for point, center_u, center_v in zip(samples, u, v):
                        radius_u = max(1, int(np.ceil(intrinsics[0, 0] * radius / point[2])))
                        radius_v = max(1, int(np.ceil(intrinsics[1, 1] * radius / point[2])))
                        if center_u - radius_u < 0 or center_u + radius_u >= depth.shape[1] or center_v - radius_v < 0 or center_v + radius_v >= depth.shape[0]:
                            visible = False
                            break
                        yy, xx = np.ogrid[-radius_v:radius_v + 1, -radius_u:radius_u + 1]
                        disk = (xx / radius_u) ** 2 + (yy / radius_v) ** 2 <= 1
                        patch = valid[center_v - radius_v:center_v + radius_v + 1, center_u - radius_u:center_u + radius_u + 1]
                        if not patch[disk].all():
                            visible = False
                            break
                        observed_depth = depth[center_v - radius_v:center_v + radius_v + 1, center_u - radius_u:center_u + radius_u + 1]
                        own_region = (finger_mask | target_mask)[center_v - radius_v:center_v + radius_v + 1, center_u - radius_u:center_u + radius_u + 1]
                        # An outside surface in front of the swept disk hides
                        # the volume behind it, even when it does not intersect
                        # the tip centreline in reconstructed XYZ.
                        if np.any(disk & ~own_region & (observed_depth * 1000 + config.tip_radius_mm + error < point[2])):
                            visible = False
                            break
                    if not visible:
                        break
                if not visible:
                    continue
                if len(obstacles) and np.min(distances_to_path(obstacles, path)) <= config.tip_radius_mm + config.obstacle_margin_mm + error:
                    continue
                candidates.append((target_distance, direction))
            if candidates:
                result.path_distance_mm, result.direction = min(candidates, key=lambda item: item[0])
                result.reachable = True
            else:
                reasons.append("unreachable")
            return self._finish(result, reasons, mode)
        except (ValueError, TypeError, IndexError, np.linalg.LinAlgError):
            reasons.append("invalid_geometry")
            return self._finish(result, reasons, mode)
