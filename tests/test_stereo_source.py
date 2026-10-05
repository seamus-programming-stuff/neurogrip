import ast
import importlib.util
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
import urllib.request

import cv2
import numpy as np

from camera_source import FramePacket
from neon_raw import CaptureState, make_handler, monotonic_ns, ThreadingHTTPServer
from stereo_source import StereoSource


class FakeSource:
    def __init__(self, packet=None):
        self.packet = packet
        self.started = False
        self.closed = False
        self.state = "running"

    def start(self):
        self.started = True
        return self

    def read(self):
        return self.packet

    def status(self):
        return {"state": self.state, "sequence": None if self.packet is None else self.packet.sequence,
                "fresh": self.packet is not None, "timestamp_kind": "host_read_completion"}

    def close(self, timeout=1):
        self.closed = True
        return True


def packet(sequence, read_time, shape=(30, 40, 3)):
    return FramePacket(np.full(shape, sequence % 255, np.uint8), sequence, read_time)


class StereoSourceTests(unittest.TestCase):
    def make_source(self, **kwargs):
        source = StereoSource(FakeSource(), FakeSource(), **kwargs)
        self.addCleanup(source.close)
        return source

    def read_at(self, source, now=100):
        with patch("stereo_source.time.monotonic", return_value=now):
            return source.read()

    def test_pair_consumes_each_sequence_only_once(self):
        source = self.make_source()
        source._record("left", packet(1, 99.99), 100)
        source._record("right", packet(1, 99.991), 100)
        pair = self.read_at(source)
        self.assertEqual(pair.sequence, 1)
        self.assertEqual((pair.left.sequence, pair.right.sequence), (1, 1))
        self.assertAlmostEqual(pair.read_time, 99.99)
        self.assertAlmostEqual(pair.pair_skew_ms, 1.0)
        self.assertEqual(pair.timestamp_kind, "host_read_completion")
        self.assertFalse(pair.synchronization_verified)
        self.assertIsNone(self.read_at(source))
        source._record("left", packet(2, 99.995), 100)
        source._record("right", packet(1, 99.991), 100)
        self.assertIsNone(self.read_at(source), "Consumed right frame was reused")
        source._record("right", packet(2, 99.997), 100)
        self.assertEqual(self.read_at(source).sequence, 2)

    def test_nearby_history_frame_pairs_when_latest_arrival_cannot(self):
        source = self.make_source(max_pair_skew_ms=2)
        source._record("left", packet(1, 99.98), 100)
        source._record("left", packet(2, 99.999), 100)
        source._record("right", packet(1, 99.9805), 100)
        pair = self.read_at(source)
        self.assertEqual(pair.left.sequence, 1)
        self.assertAlmostEqual(pair.pair_skew_ms, .5)
        source._record("right", packet(2, 99.9991), 100)
        self.assertEqual(self.read_at(source).left.sequence, 2)

    def test_history_is_bounded_and_prefers_recent_eligible_pair(self):
        source = self.make_source()
        for sequence in range(1, 13):
            source._record("left", packet(sequence, 99.80 + sequence * .01), 100)
        self.assertEqual(len(source._history["left"]), 8)
        self.assertEqual(source._dropped["overflow"], 4)
        source._record("right", packet(1, 99.9201), 100)
        self.assertEqual(self.read_at(source).left.sequence, 12)
        self.assertEqual(len(source._history["left"]), 0)

    def test_skew_and_stale_pairs_are_rejected(self):
        source = self.make_source(stale_after=.1, max_pair_skew_ms=10)
        source._record("left", packet(1, 99.97), 100)
        source._record("right", packet(1, 99.999), 100)
        self.assertIsNone(self.read_at(source))
        self.assertEqual(source._reason, "pair_skew_exceeded")
        source._record("left", packet(2, 99.998), 100)
        self.assertIsNone(self.read_at(source, 100.2), "Old queued frames stayed fresh")
        self.assertGreater(source._dropped["stale"], 0)

    def test_resize_clears_history_and_mismatch_never_pairs(self):
        source = self.make_source()
        source._record("left", packet(1, 99.98), 100)
        source._record("right", packet(1, 99.98), 100)
        source._record("left", packet(2, 99.99, (30, 50, 3)), 100)
        self.assertEqual(source._source_epoch, 1)
        self.assertIsNone(self.read_at(source), "Pre-resize right frame was retained")
        source._record("right", packet(2, 99.991), 100)
        self.assertIsNone(self.read_at(source))
        self.assertEqual(source._reason, "image_size_mismatch")
        source._record("right", packet(3, 99.995, (30, 50, 3)), 100)
        source._record("left", packet(3, 99.996, (30, 50, 3)), 100)
        pair = self.read_at(source)
        self.assertEqual(pair.source_epoch, 2)
        self.assertEqual(pair.left.image.shape, pair.right.image.shape)

    def test_disconnect_and_counter_restart_clear_old_pair(self):
        source = self.make_source()
        source._record("left", packet(10, 99.98), 100)
        source._record("right", packet(10, 99.981), 100)
        source._unavailable("left", "left_disconnected")
        self.assertEqual(source._source_epoch, 1)
        self.assertIsNone(self.read_at(source))
        source._record("left", packet(1, 99.99), 100)
        self.assertIsNone(self.read_at(source), "Old right packet paired across reconnect")
        source._record("right", packet(11, 99.991), 100)
        self.assertEqual(self.read_at(source).source_epoch, 1)
        source._record("left", packet(0, 99.995), 100)
        self.assertEqual(source._source_epoch, 2)
        self.assertIsNone(self.read_at(source))

    def test_invalid_or_foreign_clock_packet_is_rejected(self):
        source = self.make_source()
        for value, kind in ((float("nan"), "host_read_completion"),
                            (101, "host_read_completion"), (99.99, "sensor_exposure")):
            invalid = packet(1, value)
            invalid.timestamp_kind = kind
            source._record("left", invalid, 100)
        self.assertEqual(source._dropped["invalid"], 3)
        self.assertEqual(len(source._history["left"]), 0)

    def test_context_start_close_and_worker_are_idempotent(self):
        now = time.monotonic()
        left, right = FakeSource(packet(1, now - .01)), FakeSource(packet(1, now - .009))
        source = StereoSource(left, right)
        with source.start() as same:
            self.assertIs(same, source)
            deadline = time.monotonic() + 1
            pair = None
            while pair is None and time.monotonic() < deadline:
                pair = source.read()
                time.sleep(.002)
            self.assertIsNotNone(pair)
            self.assertFalse(source.status()["synchronization_verified"])
            self.assertTrue(left.started and right.started)
        self.assertTrue(left.closed and right.closed)
        self.assertIsNone(source.read())
        self.assertTrue(source.close())


class NeonMetadataTests(unittest.TestCase):
    def test_capture_resize_and_mode_setting_order_without_hardware(self):
        state = CaptureState(True, 80, output_width=30)
        operations = []

        class FakeCapture:
            def set(self, key, value):
                operations.append(("set", key, value))
                return True
            def isOpened(self):
                return True
            def read(self):
                operations.append(("read",))
                state.stop.set()
                return True, np.zeros((60, 80, 3), np.uint8)
            def release(self):
                operations.append(("release",))

        with patch("neon_raw.cv2.VideoCapture", return_value=FakeCapture()):
            state.capture()
        self.assertEqual(operations[:2], [("set", cv2.CAP_PROP_FRAME_WIDTH, 1920),
                                          ("set", cv2.CAP_PROP_FRAME_HEIGHT, 1080)])
        self.assertEqual(operations[2], ("read",))
        metadata = state.metadata()
        self.assertEqual(metadata["capture_image_size"], [80, 60])
        self.assertEqual([metadata["width"], metadata["height"]], [30, 40])
        self.assertEqual(metadata["timestamp_kind"], "neon_host_read_completion")
        self.assertIsInstance(metadata["capture_monotonic_ns"], int)

    def test_raw_multipart_headers_match_state_for_same_jpeg(self):
        state = CaptureState(False, 80)
        ok, encoded = cv2.imencode(".jpg", np.zeros((30, 40, 3), np.uint8))
        self.assertTrue(ok)
        state.jpeg, state.sequence = encoded.tobytes(), 7
        state.capture_monotonic_ns = monotonic_ns()
        state.read_time = state.capture_monotonic_ns / 1000000000.0
        state.shape, state.error = (30, 40, 3), None
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:%d" % server.server_address[1]
        try:
            with urllib.request.urlopen(base + "/state", timeout=2) as response:
                metadata = json.load(response)
            with urllib.request.urlopen(base + "/raw", timeout=2) as response:
                self.assertEqual(response.readline(), b"--frame\r\n")
                headers = {}
                while True:
                    line = response.readline()
                    if line == b"\r\n":
                        break
                    key, value = line.decode("ascii").strip().split(":", 1)
                    headers[key] = value.strip()
                self.assertEqual(int(headers["X-Frame-Sequence"]), metadata["sequence"])
                self.assertEqual(int(headers["X-Capture-Monotonic-Ns"]), metadata["capture_monotonic_ns"])
                self.assertEqual(headers["X-Camera-Session"], metadata["camera_session"])
                self.assertEqual(response.read(int(headers["Content-Length"])), state.jpeg)
                self.assertFalse(metadata["sensor_synchronized"])
        finally:
            state.stop.set()
            with state.condition:
                state.condition.notify_all()
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_helper_parses_as_python36_and_has_threaded_server_fallback(self):
        helper = Path(__file__).resolve().parents[1] / "neon_raw.py"
        ast.parse(helper.read_text(), feature_version=(3, 6))
        import builtins
        real_import = builtins.__import__

        def without_modern_server(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "http.server" and "ThreadingHTTPServer" in fromlist:
                raise ImportError("Simulated Python 3.6")
            return real_import(name, globals, locals, fromlist, level)

        spec = importlib.util.spec_from_file_location("neon_raw_py36_test", helper)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=without_modern_server):
            spec.loader.exec_module(module)
        from socketserver import ThreadingMixIn
        self.assertTrue(issubclass(module.ThreadingHTTPServer, ThreadingMixIn))
        with patch.object(module.time, "monotonic_ns", None), \
                patch.object(module.time, "monotonic", return_value=123.5):
            self.assertEqual(module.monotonic_ns(), 123500000000)


if __name__ == "__main__":
    unittest.main()
