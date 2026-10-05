"""Bounded pairing of two independent camera streams on the PC clock.

Arrival/read-completion skew is a transport diagnostic. It does not establish
simultaneous exposures, even when it is zero. CameraSource/FFmpeg may buffer
upstream frames and does not expose the NEON multipart timing headers.
"""
from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from camera_source import CameraSource, FramePacket


@dataclass
class StereoPair:
    left: FramePacket
    right: FramePacket
    sequence: int
    read_time: float
    pair_skew_ms: float
    timestamp_kind: str = "host_read_completion"
    synchronization_verified: bool = False
    source_epoch: int = 0


class StereoSource:
    """Collect and consume each received frame at most once.

    Histories hold at most eight frames per side. Prefer a recent pair within
    max_pair_skew_ms; use nearest arrival times to resolve equally recent pairs.
    A reconnect, backwards counter/time, or changed image geometry clears both
    histories and increments source_epoch. A caller must invalidate calibration
    and tracking when the epoch changes. read() never replays the last pair.
    """

    def __init__(self, left_source, right_source, stale_after=0.5,
                 max_pair_skew_ms=20, history_size=8):
        if not math.isfinite(stale_after) or stale_after <= 0:
            raise ValueError("stale_after must be finite and positive")
        if not math.isfinite(max_pair_skew_ms) or max_pair_skew_ms < 0:
            raise ValueError("max_pair_skew_ms must be finite and nonnegative")
        if isinstance(history_size, bool) or not isinstance(history_size, int) or not 1 <= history_size <= 8:
            raise ValueError("history_size must be between 1 and 8")
        self.stale_after = float(stale_after)
        self.max_pair_skew_ms = float(max_pair_skew_ms)
        self.history_size = history_size
        self.sources = {"left": self._make_source(left_source),
                        "right": self._make_source(right_source)}
        self._history = {side: deque(maxlen=history_size) for side in self.sources}
        self._last_sequence = {side: None for side in self.sources}
        self._last_time = {side: None for side in self.sources}
        self._shape = {side: None for side in self.sources}
        self._available = {side: None for side in self.sources}
        self._source_status = {side: {"state": "not_started"} for side in self.sources}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._pair_sequence = 0
        self._source_epoch = 0
        self._last_pair_time = None
        self._last_skew_ms = None
        self._reason = "not_started"
        self._dropped = {"overflow": 0, "stale": 0, "skew": 0, "invalid": 0}

    def _make_source(self, source):
        if all(callable(getattr(source, method, None)) for method in ("read", "start", "status", "close")):
            return source
        if isinstance(source, str) and source.isdecimal():
            source = int(source)
        return CameraSource(source, stale_after=self.stale_after)

    def start(self):
        if self._thread is not None or self._stop.is_set():
            return self
        for source in self.sources.values():
            source.start()
        self._thread = threading.Thread(target=self._collect, name="stereo-pairing", daemon=True)
        self._thread.start()
        return self

    def _reset_locked(self, reason):
        for history in self._history.values():
            history.clear()
        self._source_epoch += 1
        self._last_pair_time = None
        self._last_skew_ms = None
        self._reason = reason

    def _unavailable(self, side, reason):
        with self._lock:
            if self._available[side] is True:
                self._reset_locked(reason)
            self._available[side] = False
            self._last_sequence[side] = None
            self._last_time[side] = None
            self._reason = reason

    def _record(self, side, packet, now):
        """Validate/record one new host packet; also used by deterministic tests."""
        try:
            valid = (isinstance(packet.sequence, int) and not isinstance(packet.sequence, bool)
                     and packet.sequence >= 0 and math.isfinite(packet.read_time)
                     and packet.read_time <= now + 0.002
                     and isinstance(packet.image, np.ndarray) and packet.image.dtype == np.uint8
                     and packet.image.ndim == 3 and packet.image.shape[2] == 3
                     and min(packet.image.shape[:2]) >= 14
                     and packet.timestamp_kind in ("host_read_completion", "host_receive_completion"))
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            with self._lock:
                self._dropped["invalid"] += 1
            self._unavailable(side, side + "_invalid_packet")
            return
        if now - packet.read_time >= self.stale_after:
            self._unavailable(side, side + "_stale")
            return
        with self._lock:
            if self._stop.is_set():
                return
            previous_sequence = self._last_sequence[side]
            previous_time = self._last_time[side]
            if packet.sequence == previous_sequence:
                return
            if ((previous_sequence is not None and packet.sequence < previous_sequence)
                    or (previous_time is not None and packet.read_time < previous_time)):
                self._reset_locked(side + "_source_restarted")
            shape = tuple(packet.image.shape)
            if self._shape[side] is not None and shape != self._shape[side]:
                self._reset_locked(side + "_resolution_changed")
            self._shape[side] = shape
            self._last_sequence[side] = packet.sequence
            self._last_time[side] = packet.read_time
            self._available[side] = True
            if len(self._history[side]) == self.history_size:
                self._dropped["overflow"] += 1
            self._history[side].append(packet)
            self._reason = "waiting_for_pair"

    def _collect(self):
        while not self._stop.is_set():
            for side, source in self.sources.items():
                try:
                    status = source.status()
                    with self._lock:
                        self._source_status[side] = status
                        last_sequence = self._last_sequence[side]
                    if status.get("state") != "running" or status.get("fresh") is False:
                        self._unavailable(side, side + "_unavailable")
                        continue
                    # CameraSource.read() copies an image. Avoid doing so when
                    # status already proves the newest sequence was collected.
                    if status.get("sequence") is not None and status["sequence"] == last_sequence:
                        continue
                    packet = source.read()
                    if packet is None:
                        self._unavailable(side, side + "_stale")
                    else:
                        self._record(side, packet, time.monotonic())
                except Exception as exc:
                    with self._lock:
                        self._source_status[side] = {"state": "capture_error", "error": str(exc)}
                    self._unavailable(side, side + "_capture_error")
            self._stop.wait(0.002)

    def read(self):
        if self._stop.is_set():
            return None
        now = time.monotonic()
        with self._lock:
            if self._stop.is_set():
                return None
            for history in self._history.values():
                while history and now - history[0].read_time >= self.stale_after:
                    history.popleft()
                    self._dropped["stale"] += 1
            left, right = self._history["left"], self._history["right"]
            if not left or not right:
                self._reason = "waiting_for_fresh_pair"
                return None
            candidates = []
            matching_size = False
            for i, left_packet in enumerate(left):
                for j, right_packet in enumerate(right):
                    if left_packet.image.shape != right_packet.image.shape:
                        continue
                    matching_size = True
                    skew_ms = abs(left_packet.read_time - right_packet.read_time) * 1000
                    if skew_ms <= self.max_pair_skew_ms + 1e-7:
                        # Recent shared time minimizes lag; tie-break by skew.
                        candidates.append((-min(left_packet.read_time, right_packet.read_time),
                                           skew_ms, i, j))
            if not candidates:
                self._reason = "pair_skew_exceeded" if matching_size else "image_size_mismatch"
                while left and right:
                    if left[0].read_time < right[0].read_time - self.max_pair_skew_ms / 1000:
                        left.popleft()
                    elif right[0].read_time < left[0].read_time - self.max_pair_skew_ms / 1000:
                        right.popleft()
                    else:
                        break
                    self._dropped["skew"] += 1
                return None
            _, skew_ms, i, j = min(candidates)
            left_packet, right_packet = left[i], right[j]
            for _ in range(i + 1):
                left.popleft()
            for _ in range(j + 1):
                right.popleft()
            self._pair_sequence += 1
            pair_time = min(left_packet.read_time, right_packet.read_time)
            self._last_pair_time, self._last_skew_ms = pair_time, skew_ms
            self._reason = None
            return StereoPair(left_packet, right_packet, self._pair_sequence, pair_time,
                              skew_ms, source_epoch=self._source_epoch)

    def status(self):
        with self._lock:
            age = None if self._last_pair_time is None else max(0.0, time.monotonic() - self._last_pair_time)
            return {"state": "closed" if self._stop.is_set() else "running" if self._thread else "not_started",
                    "reason": self._reason, "sequence": self._pair_sequence,
                    "source_epoch": self._source_epoch, "pair_skew_ms": self._last_skew_ms,
                    "pair_age_seconds": age, "fresh": age is not None and age < self.stale_after,
                    "timestamp_kind": "host_read_completion", "synchronization_verified": False,
                    "max_pair_skew_ms": self.max_pair_skew_ms,
                    "queued_frames": {side: len(history) for side, history in self._history.items()},
                    "dropped": dict(self._dropped),
                    "left": dict(self._source_status["left"]), "right": dict(self._source_status["right"])}

    def close(self, timeout=1.0):
        self._stop.set()
        deadline = time.monotonic() + max(0.0, timeout)
        if self._thread:
            self._thread.join(max(0.0, deadline - time.monotonic()))
        stopped = self._thread is None or not self._thread.is_alive()
        for source in self.sources.values():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                closed = source.close(timeout=remaining / 2)
            except TypeError:
                closed = source.close()
            stopped = (closed is not False) and stopped
        with self._lock:
            for history in self._history.values():
                history.clear()
            self._reason = "closed" if stopped else "closing_blocked"
        return stopped

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
