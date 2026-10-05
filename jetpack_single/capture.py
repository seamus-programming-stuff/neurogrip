"""Direct NEON sensor capture: 1080p V4L2 first, then upright 540x960."""
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import threading
import time

import cv2


@dataclass(frozen=True)
class Frame:
    image: object
    sequence: int
    captured_ns: int


def device_owners(device):
    if os.name != 'posix':
        return ''
    try:
        result = subprocess.run(['fuser', str(device)], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=3, check=False)
        return result.stdout.decode('ascii', errors='replace').strip()
    except FileNotFoundError:
        return ''


class NeonCapture:
    def __init__(self, device='/dev/video0', image=None):
        self.device = device
        self.image_path = image
        self.simulation = image is not None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._error = None
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError('capture already started')
        owners = '' if self.simulation else device_owners(self.device)
        if owners:
            raise RuntimeError('%s is already owned by PID(s) %s. Stop that camera service first.'
                               % (self.device, owners))
        self._thread = threading.Thread(target=self._loop, name='neon-capture', daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        camera = None
        try:
            still = None
            if self.simulation:
                still = cv2.imread(str(self.image_path))
                if still is None or still.shape[:2] != (960, 540):
                    raise RuntimeError('--image requires an upright 540x960 image; simulation mode only')
            else:
                camera = cv2.VideoCapture(str(self.device), cv2.CAP_V4L2)
                if not camera.isOpened():
                    raise RuntimeError('Cannot open %s with V4L2' % self.device)
                # The NEON sensor refuses its unsupported default 640x480 mode.
                # Both settings precede every read, including the first read.
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            sequence = 0
            while not self._stop.is_set():
                if still is None:
                    okay, raw = camera.read()
                    captured_ns = time.monotonic_ns()
                    if not okay or raw is None:
                        raise RuntimeError('Sensor read failed')
                    if raw.shape[:2] != (1080, 1920):
                        raise RuntimeError('Sensor returned %r; expected 1920x1080' % (raw.shape,))
                    upright = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
                    frame = cv2.resize(upright, (540, 960), interpolation=cv2.INTER_AREA)
                else:
                    captured_ns = time.monotonic_ns()
                    frame = still.copy()
                sequence += 1
                with self._lock:
                    self._frame = Frame(frame, sequence, captured_ns)
                if still is not None:
                    self._stop.wait(1 / 30.0)
        except Exception as exc:
            with self._lock:
                self._error = '%s: %s' % (type(exc).__name__, exc)
        finally:
            if camera is not None:
                camera.release()

    def read(self, after=-1):
        with self._lock:
            if self._error is not None:
                raise RuntimeError(self._error)
            return self._frame if self._frame is not None and self._frame.sequence > after else None

    def status(self):
        with self._lock:
            age = None if self._frame is None else (time.monotonic_ns() - self._frame.captured_ns) / 1e6
            return {'error': self._error, 'last_frame_age_ms': age,
                    'timestamp_kind': 'host_read_completion', 'simulation': self.simulation}

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
