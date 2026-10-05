"""Newest-frame capture. Read-completion times are never sensor timestamps."""
from dataclasses import dataclass
import threading
import time
from typing import Optional

import cv2
import numpy as np


@dataclass
class FramePacket:
    image: np.ndarray
    sequence: int
    read_time: float
    timestamp_kind: str = "host_read_completion"


class CameraSource:
    """Capture a webcam index, network URL, or video file in a daemon worker.

    read() returns the latest complete frame only while it is fresh; repeated
    calls may have the same sequence. Its time describes this host's completed
    OpenCV read, including unknown upstream buffering. It cannot prove exposure
    freshness or synchronization. Blocking capture never updates the timestamp.
    """

    def __init__(self, source, stale_after=0.5):
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        self.source = source
        self.stale_after = float(stale_after)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._packet = None  # type: Optional[FramePacket]
        self._sequence = 0
        self._state = "not_started"
        self._error = None

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="camera-source", daemon=True)
        self._thread.start()
        return self

    def _set_state(self, state, error=None, clear=False):
        with self._lock:
            self._state = state
            self._error = error
            if clear:
                self._packet = None

    def _open(self):
        is_network = isinstance(self.source, str) and "://" in self.source
        if is_network and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            # FFmpeg supports opening/read timeouts on supported OpenCV builds.
            # An unsupported build fails explicitly rather than silently falling
            # back to an unbounded network open.
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
        return cv2.VideoCapture(self.source)

    def _run(self):
        is_file = isinstance(self.source, str) and "://" not in self.source
        try:
            while not self._stop.is_set():
                self._set_state("opening")
                capture = None
                try:
                    capture = self._open()
                    if not capture.isOpened():
                        self._set_state("open_failed", "OpenCV could not open source", clear=True)
                        if is_file:
                            break
                        self._stop.wait(0.5)
                        continue
                    # Backend-specific best effort. Some backends ignore this.
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    fps = capture.get(cv2.CAP_PROP_FPS) if is_file else 0.0
                    file_interval = 1.0 / fps if is_file and 1.0 <= fps <= 240.0 else 1.0 / 30.0
                    while not self._stop.is_set():
                        ok, frame = capture.read()
                        completed = time.monotonic()
                        if self._stop.is_set():
                            break
                        if not ok or frame is None or frame.size == 0:
                            self._set_state("eof" if is_file else "read_failed",
                                            "No complete frame", clear=True)
                            break
                        owned_frame = frame.copy()
                        with self._lock:
                            self._sequence += 1
                            self._packet = FramePacket(owned_frame, self._sequence, completed)
                            self._state = "running"
                            self._error = None
                        if is_file:
                            # Replay a file at its declared frame rate instead of
                            # consuming its whole clip before the UI can poll.
                            remaining = file_interval - (time.monotonic() - completed)
                            self._stop.wait(max(0.0, remaining))
                except (cv2.error, TypeError, ValueError) as exc:
                    self._set_state("capture_error", str(exc), clear=True)
                finally:
                    if capture is not None:
                        capture.release()
                if is_file:
                    break
                self._stop.wait(0.2)
        finally:
            if self._stop.is_set():
                self._set_state("closed", clear=True)

    def read(self):
        if self._stop.is_set():
            return None
        with self._lock:
            packet = self._packet
        if packet is None or time.monotonic() - packet.read_time > self.stale_after:
            return None
        # Copy outside the lock. The stored frame is never modified after publish.
        image = packet.image.copy()
        if self._stop.is_set() or time.monotonic() - packet.read_time > self.stale_after:
            return None
        return FramePacket(image, packet.sequence, packet.read_time, packet.timestamp_kind)

    def status(self):
        with self._lock:
            packet = self._packet
            state, error = self._state, self._error
        age = None if packet is None else max(0.0, time.monotonic() - packet.read_time)
        return {"state": state, "error": error, "sequence": None if packet is None else packet.sequence,
                "frame_age_seconds": age, "fresh": age is not None and age <= self.stale_after,
                "timestamp_kind": "host_read_completion", "sensor_synchronized": False}

    def close(self, timeout=1.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))
        # A daemon may still be inside a backend read. Do not pretend it stopped
        # or race capture.release() against that read from a different thread.
        alive = self._thread is not None and self._thread.is_alive()
        self._set_state("closing_blocked" if alive else "closed", clear=True)
        return not alive

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
