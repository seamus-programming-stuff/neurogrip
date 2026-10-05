"""Textured, geometrically shifted stereo test scene; never a physical permit."""
import time
from types import SimpleNamespace
import cv2
import numpy as np

from camera_source import FramePacket
from stereo_core import StereoCalibration


def demo_calibration():
    width, height, focal, baseline = 640, 480, 400.0, 60.0
    matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64)
    p1 = np.column_stack((matrix, np.zeros(3)))
    p2 = p1.copy()
    p2[0, 3] = -focal * baseline
    q = np.array([[1, 0, 0, -width / 2], [0, 1, 0, -height / 2],
                  [0, 0, 0, focal], [0, 0, 1 / baseline, 0]], np.float64)
    return StereoCalibration(image_size=(width, height), left_source="synthetic:left", right_source="synthetic:right",
        K1=matrix, D1=np.zeros(5), K2=matrix.copy(), D2=np.zeros(5), R=np.eye(3),
        T_mm=np.array([-baseline, 0, 0]), R1=np.eye(3), R2=np.eye(3), P1=p1, P2=p2, Q=q,
        roi1=(0, 0, width, height), roi2=(0, 0, width, height), metadata={"simulation": True})


class SyntheticStereoSource:
    def __init__(self):
        rng = np.random.default_rng(912)
        self.background = rng.integers(30, 135, (480, 640, 3), np.uint8)
        self.near = rng.integers(170, 255, (150, 130, 3), np.uint8)
        self.middle = rng.integers(100, 220, (140, 110, 3), np.uint8)
        self.started = time.monotonic()
        self.last = 0.0
        self.sequence = 0
        self.closed = False

    def start(self):
        return self

    def images(self, phase=0.0):
        # Z = f * baseline / disparity: 1500mm, 750mm, 500mm.
        left = self.background.copy()
        right = np.roll(self.background, -16, axis=1)
        x_middle = 410 + int(18 * np.sin(phase))
        left[150:290, x_middle:x_middle + 110] = self.middle
        right[150:290, x_middle - 32:x_middle + 110 - 32] = self.middle
        left[220:370, 240:370] = self.near
        right[220:370, 240 - 48:370 - 48] = self.near
        return left, right

    def read(self):
        now = time.monotonic()
        if self.closed or now - self.last < 1 / 15:
            return None
        self.last = now
        self.sequence += 1
        left, right = self.images((now - self.started) * .4)
        return SimpleNamespace(left=FramePacket(left, self.sequence, now), right=FramePacket(right, self.sequence, now),
            sequence=self.sequence, read_time=now, pair_skew_ms=0.0, source_epoch=0,
            timestamp_kind="synthetic_ground_truth", synchronization_verified=False)

    def status(self):
        return {"state": "closed" if self.closed else "simulation", "source_epoch": 0,
                "synchronization_verified": False, "sequence": self.sequence}

    def close(self):
        self.closed = True
        return True
