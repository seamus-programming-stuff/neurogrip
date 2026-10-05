"""Dual-feed application tests without physical cameras or bus resources."""
from collections import deque
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import cv2
import numpy as np

from camera_source import FramePacket
from dual_camera import DualApplication, ThreadingHTTPServer, make_handler
from dual_calibrate import probe_neon_stream
from dual_config import DualConfig
from protocol import Direction, Mode
from proximity import Measurement
from synthetic_scene import demo_config
from synthetic_stereo import demo_calibration


def permitted():
    return Measurement(gap_mm=20, quality=.9, direction=Direction.FORWARDS,
                       scale_valid=True, reachable=True, candidate_permit=True, strike_permit=True)


def make_pair(sequence=1, epoch=0, read_time=None):
    read_time = time.monotonic() if read_time is None else read_time
    image = np.zeros((480, 640, 3), np.uint8)
    return SimpleNamespace(left=FramePacket(image, sequence, read_time),
        right=FramePacket(image.copy(), sequence, read_time), sequence=sequence,
        read_time=read_time, pair_skew_ms=0.0, source_epoch=epoch,
        timestamp_kind="host_read_completion", synchronization_verified=False)


class FakeSource:
    def __init__(self, pairs=(), close_error=None):
        self.pairs = deque(pairs)
        self.closed = False
        self.close_error = close_error
        self.epoch = 0

    def start(self):
        return self

    def read(self):
        pair = self.pairs.popleft() if self.pairs else None
        if pair is not None:
            self.epoch = pair.source_epoch
        return pair

    def status(self):
        return {"state": "closed" if self.closed else "running", "source_epoch": self.epoch}

    def close(self):
        self.closed = True
        if self.close_error:
            raise OSError(self.close_error)


class FakeBackend:
    def __init__(self, block=False, error=None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.error = error
        if not block:
            self.release.set()

    def compute(self, left, right):
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("Fake backend timed out")
        if self.error:
            raise RuntimeError(self.error)
        return SimpleNamespace(rectified_left=left, rectified_right=right,
                               depth_m=np.full(left.shape[:2], .7, np.float32))


class FakePublisher:
    session_id = 123

    def __init__(self):
        self.closed = False
        self.packets = []

    def publish(self, packet):
        if self.closed:
            raise RuntimeError("publisher is closed")
        self.packets.append(packet)

    def close(self):
        self.closed = True


@contextmanager
def server_for(app):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


class DualApplicationReviewTests(unittest.TestCase):
    def make_app(self, source=None, backend=None, simulation=False, **settings):
        alignment_patch = patch("dual_camera.check_epipolar_alignment", return_value={"aligned": True})
        alignment_patch.start()
        self.addCleanup(alignment_patch.stop)
        config = DualConfig(recommendations_enabled=True, stereo_timing_validated=True,
                            exposure_skew_bound_ms=0, **settings)
        calibration = demo_calibration()
        calibration.metadata["simulation"] = simulation
        publisher = FakePublisher()
        with patch("dual_camera.StrikeConfig.load", return_value=demo_config()):
            app = DualApplication(config, calibration, source=source or FakeSource(),
                                  backend=backend or FakeBackend(), publisher=publisher)
        def cleanup():
            if not app.closed:
                app.close()
        self.addCleanup(cleanup)
        return app

    @staticmethod
    def seed(app, age=0):
        with app.lock:
            app.measurement = permitted()
            app.frame_time = time.monotonic() - age
            app.error = None
            app.depth = np.full((480, 640), .7, np.float32)
            ok, image = cv2.imencode(".jpg", np.full((480, 1280, 3), 120, np.uint8))
            assert ok
            app.jpeg = image.tobytes()

    def process(self, app):
        worker = threading.Thread(target=app._process, daemon=True)
        app.threads.append(worker)
        worker.start()
        return worker

    def wait_for(self, predicate, timeout=1):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(predicate(), "Application condition did not arrive")

    def test_packet_and_http_expire_without_processing_updates(self):
        app = self.make_app()
        self.seed(app)
        self.assertTrue(app.packet().strike_permit)
        self.seed(app, age=1)
        self.assertFalse(app.packet().strike_permit)
        jpeg, sequence = app.display_frame()
        self.assertEqual(sequence, -1)
        self.assertNotEqual(jpeg, app.jpeg)
        with server_for(app) as base:
            with urllib.request.urlopen(base + "/telemetry", timeout=2) as response:
                value = json.load(response)
            self.assertFalse(value["strike_permit"])
            self.assertIn("stale_frame", value["block_reasons"])
            with self.assertRaises(urllib.error.HTTPError) as failure:
                urllib.request.urlopen(base + "/depth.npz", timeout=2)
            self.assertEqual(failure.exception.code, 503)
            with urllib.request.urlopen(base, timeout=2) as response:
                html = response.read().decode()
            self.assertIn('src="/stream"', html)
            self.assertNotIn("<button", html)

    def test_epoch_change_revokes_previous_permission_before_compute(self):
        backend = FakeBackend(block=True)
        app = self.make_app(source=FakeSource([make_pair(epoch=1)]), backend=backend)
        app.source_epoch = 0
        self.seed(app)
        try:
            self.process(app)
            self.assertTrue(backend.started.wait(1))
            self.assertFalse(app.packet().strike_permit,
                             "Old permission remained live while a reconnected camera was processing")
        finally:
            backend.release.set()

    def test_missing_trackers_have_missing_region_reasons(self):
        app = self.make_app(source=FakeSource([make_pair()]))
        self.process(app)
        self.wait_for(lambda: app.frame_sequence == 1)
        self.assertFalse(app.packet().strike_permit)
        self.assertIn("finger_missing", app.measurement.block_reasons)
        self.assertIn("target_missing", app.measurement.block_reasons)

    def test_engine_includes_upstream_delay_in_sample_freshness(self):
        app = self.make_app(source=FakeSource([make_pair(read_time=time.monotonic() - .1)]))
        app.stroke_config.source_delay_bound_ms = 450

        class AgeCheckingEngine:
            def reset(self):
                pass
            def evaluate(self, *args, **kwargs):
                return (Measurement(block_reasons=("stale_frame",))
                        if kwargs["capture_age_ms"] >= 500 else permitted())

        app.engine = AgeCheckingEngine()
        self.process(app)
        self.wait_for(lambda: app.frame_sequence == 1)
        self.assertFalse(app.measurement.candidate_permit,
                         "A sample already older than the budget entered the permit debounce")

    def test_missing_pairs_reset_debounce_after_effective_expiry(self):
        app = self.make_app()
        self.seed(app, age=.1)
        app.stroke_config.source_delay_bound_ms = 450

        class CountingEngine:
            resets = 0
            def reset(self):
                self.resets += 1

        app.engine = CountingEngine()
        self.process(app)
        self.wait_for(lambda: app.engine.resets > 0)
        self.assertFalse(app.packet().strike_permit)

    def test_backend_failure_blocks_immediately(self):
        app = self.make_app(source=FakeSource([make_pair()]), backend=FakeBackend(error="test compute failure"))
        self.seed(app)
        self.process(app)
        self.wait_for(lambda: app.error == "test compute failure")
        self.assertFalse(app.packet().strike_permit)
        self.assertIsNone(app.depth)
        self.assertEqual(app.display_frame()[1], -1)

    def test_capture_exception_blocks_previous_permission(self):
        class FailingSource(FakeSource):
            def read(self):
                raise OSError("test capture failure")
        app = self.make_app(source=FailingSource())
        self.seed(app)
        self.process(app)
        self.wait_for(lambda: app.error is not None and "test capture failure" in app.error)
        self.assertFalse(app.packet().strike_permit)
        self.assertIsNone(app.frame_time)

    def test_source_reports_loss_revokes_even_if_processing_stalls(self):
        class SourceWithFreshness(FakeSource):
            fresh = True
            def status(self):
                result = super().status()
                result["fresh"] = self.fresh
                return result
        source = SourceWithFreshness()
        app = self.make_app(source=source)
        self.seed(app)
        self.assertTrue(app.packet().strike_permit)
        source.fresh = False
        self.assertFalse(app.packet().strike_permit)

    def test_state_snapshot_does_not_mix_old_permission_with_new_error(self):
        app = self.make_app()
        self.seed(app)
        ready, invalidated = threading.Event(), threading.Event()
        original_packet = app.packet
        def delayed_packet():
            value = original_packet()
            ready.set()
            invalidated.wait(.1)
            return value
        def invalidate():
            if ready.wait(1):
                with app.lock:
                    app.error = "camera geometry changed"
                    app.frame_time = None
                    app.measurement = Measurement(block_reasons=("invalid_geometry",))
                invalidated.set()
        worker = threading.Thread(target=invalidate, daemon=True)
        worker.start()
        try:
            with patch.object(app, "packet", side_effect=delayed_packet):
                state = app.state()
        finally:
            worker.join(1)
        self.assertFalse(state["telemetry"]["strike_permit"] and state["error"] is not None)

    def test_constructor_validation_failure_does_not_leak_created_publisher(self):
        created = []
        def create_publisher(**kwargs):
            publisher = FakePublisher()
            created.append(publisher)
            return publisher
        with patch("dual_camera.SharedMemoryPublisher", side_effect=create_publisher), \
                patch("dual_camera.StrikeConfig.load", side_effect=ValueError("invalid stroke config")):
            with self.assertRaises(ValueError):
                DualApplication(DualConfig(), demo_calibration(), source=FakeSource(), backend=FakeBackend())
        self.assertTrue(all(publisher.closed for publisher in created),
                        "A startup error retained the shared-memory writer reservation")

    def test_failed_taught_reference_does_not_leak_created_publisher(self):
        created = []
        calibration = demo_calibration()
        calibration.metadata["regions_reference"] = "missing_reference_for_unit_test.png"
        def create_publisher(**kwargs):
            publisher = FakePublisher()
            created.append(publisher)
            return publisher
        with patch("dual_camera.SharedMemoryPublisher", side_effect=create_publisher), \
                patch("dual_camera.StrikeConfig.load", return_value=demo_config()), \
                patch("dual_camera.cv2.imread", return_value=None):
            with self.assertRaises(ValueError):
                DualApplication(DualConfig(), calibration, source=FakeSource(), backend=FakeBackend())
        self.assertTrue(all(publisher.closed for publisher in created))

    def test_close_is_idempotent_and_publishes_revocation(self):
        app = self.make_app()
        self.seed(app)
        app.close()
        self.assertTrue(app.publisher.closed)
        self.assertFalse(app.publisher.packets[-1].strike_permit)
        app.close()
        self.assertFalse(app.packet().strike_permit)

    def test_source_shutdown_failure_still_releases_publisher(self):
        app = self.make_app(source=FakeSource(close_error="test source close failure"))
        try:
            app.close()
        except OSError:
            pass
        self.assertTrue(app.publisher.closed, "Capture shutdown error leaked bus resources")

    def test_saved_relative_paths_resolve_beside_dual_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "dual.json"
            DualConfig(calibration="calibration/stereo.json", stroke_config="stroke.json").save(config_path)
            loaded = DualConfig.load(config_path)
            self.assertEqual(Path(loaded.calibration), Path(directory) / "calibration" / "stereo.json")
            self.assertEqual(Path(loaded.stroke_config), Path(directory) / "stroke.json")

    def test_raw_preflight_rejects_overlay_and_stale_sources(self):
        for value in ({"overlays": True, "fresh": True}, {"overlays": False, "fresh": False},
                      {"overlays": False, "fresh": True, "error": "no sensor"}):
            with self.subTest(value=value), \
                    patch("dual_calibrate.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(value).encode())):
                with self.assertRaises(ValueError):
                    probe_neon_stream("http://127.0.0.1:8081/raw")
        value = {"overlays": False, "fresh": True, "error": None, "camera_session": "test"}
        with patch("dual_calibrate.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(value).encode())):
            self.assertEqual(probe_neon_stream("http://127.0.0.1:8081/raw"), value)


if __name__ == "__main__":
    unittest.main()
