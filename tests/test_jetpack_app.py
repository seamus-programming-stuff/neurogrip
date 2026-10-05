"""Fault/freshness integration tests without a model, sensor, or private image."""
import ast
import importlib.util
import io
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import numpy as np

from protocol import Direction, Mode
from proximity import Measurement, StrikeConfig
from network_bus import UDPReceiver
from jetpack_single.mapped_bus import MappedSubscriber


PACKAGE = Path(__file__).resolve().parents[1] / "jetpack_single"
sys.path.insert(0, str(PACKAGE))
try:
    SPEC = importlib.util.spec_from_file_location("neon_app_review", str(PACKAGE / "app.py"))
    app_module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(app_module)
finally:
    sys.path.remove(str(PACKAGE))


def wait_for(condition, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(.005)
    raise AssertionError("condition did not become true before deadline")


def permitted():
    return Measurement(gap_mm=12., direction=Direction.FORWARDS,
                       reachable=True, candidate_permit=True, strike_permit=True,
                       quality=.8, scale_valid=True)


class FakeSource:
    def __init__(self, simulation=False):
        self.simulation = simulation
        self.error = None
        self.closed = False
        self.sequence = 0
        self.frame = None
        self.emit()

    def emit(self):
        self.sequence += 1
        self.frame = SimpleNamespace(image=np.zeros((960, 540, 3), np.uint8),
                                     sequence=self.sequence, captured_ns=time.monotonic_ns())

    def start(self):
        # Real capture timestamps are generated after start, not while the
        # application is constructing its disk-backed bus/configuration.
        self.emit()
        return self

    def read(self, after=-1):
        if self.error:
            raise RuntimeError(self.error)
        return self.frame if self.frame.sequence > after else None

    def status(self):
        return {"error": self.error, "last_frame_age_ms":
                (time.monotonic_ns() - self.frame.captured_ns) / 1e6,
                "timestamp_kind": "host_read_completion", "simulation": self.simulation}

    def close(self):
        self.closed = True


class FakeRuntime:
    manifest = {"model_id": "fake-local-test-only"}
    backend = "fake"

    def __init__(self, metric=True):
        self.metric = metric
        self.calls = 0
        self.fail = False
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def infer(self, rgb):
        self.calls += 1
        if self.block:
            self.entered.set()
            self.release.wait(3)
        if self.fail:
            raise RuntimeError("inference fault injected")
        depth = np.ones(rgb.shape[:2], np.float32)
        if not self.metric:
            depth[:] = np.nan
        return SimpleNamespace(depth_m=depth, metric=self.metric)


class PermitEngine:
    """Isolate application revocation from geometry/model accuracy."""
    def __init__(self):
        self.kwargs = None
        self.depth_seen = None
        self.reset_calls = 0

    def evaluate(self, *args, **kwargs):
        self.kwargs = kwargs
        self.depth_seen = np.asarray(args[0]).copy()
        return permitted()

    def reset(self):
        self.reset_calls += 1


class JetPackApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.apps = []
        self.receivers = []
        self.servers = []

    def tearDown(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(2)
        for app in self.apps:
            app.runtime.release.set()
            app.close()
        for receiver in self.receivers:
            receiver.close()
        self.temp.cleanup()

    def application(self, simulation=False, metric=True, source_delay=0, valid_for=500, udp=False):
        config = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))
        strike = json.loads((PACKAGE / "strike_config.json").read_text(encoding="utf-8"))
        strike.update(valid_for_ms=valid_for, capture_timing_validated=True,
                      source_delay_bound_ms=source_delay)
        strike_path = self.root / ("strike-" + secrets.token_hex(3) + ".json")
        strike_path.write_text(json.dumps(strike), encoding="utf-8")
        config.update(strike_config=str(strike_path), camera_calibration=str(self.root / "missing-camera.json"),
                      teaching=str(self.root / "missing-teaching.json"),
                      shared_memory_name="ng_app_" + secrets.token_hex(6),
                      controller_armed=True, bus_rate_hz=100, display_max_age_ms=100)
        if udp:
            receiver = UDPReceiver("127.0.0.1", 0, transport_delay_bound_ms=0,
                                   max_age_ms=500)
            self.receivers.append(receiver)
            config["udp"] = dict(enabled=True, host="127.0.0.1",
                                 port=receiver.local_address[1], source_port=0)
        else:
            config["udp"]["enabled"] = False
        runtime, source = FakeRuntime(metric), FakeSource(simulation)
        app = app_module.Application(config, bus_directory=self.temp.name,
                                     runtime=runtime, source=source)
        self.apps.append(app)
        if udp:
            receiver.expected_peer = app.udp.local_address
            receiver.pair_session(app.publisher.session_id)
        return app

    def http(self, app):
        server = ThreadingHTTPServer(("127.0.0.1", 0), app_module.handler_for(app))
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return "http://127.0.0.1:%d" % server.server_address[1]

    def test_python38_grammar(self):
        for filename in ("app.py", "capture.py"):
            ast.parse((PACKAGE / filename).read_text(encoding="utf-8"), feature_version=(3, 8))

    def test_packet_age_includes_validated_sensor_delay_bound(self):
        config = StrikeConfig(valid_for_ms=100, capture_timing_validated=True,
                              source_delay_bound_ms=200)
        result = app_module.packet_for(permitted(), 42, 1, time.monotonic_ns(), config, Mode.LIVE)
        self.assertGreaterEqual(result.capture_age_ms, 200)
        self.assertFalse(result.strike_permit)
        self.assertIn("stale_frame", result.block_reasons)

    def test_packet_timestamp_is_source_monotonic_not_wall_clock(self):
        with patch.object(app_module.time, "time_ns", return_value=9000000000000), \
                patch.object(app_module.time, "monotonic_ns", return_value=3000000000):
            result = app_module.packet_for(permitted(), 42, 1, 3000000000,
                                           StrikeConfig(), Mode.LIVE)
        self.assertEqual(result.timestamp_ms, 3000)

    def test_future_capture_timestamp_cannot_create_a_fresh_permit(self):
        result = app_module.packet_for(permitted(), 42, 1,
                                       time.monotonic_ns() + 1000000000,
                                       StrikeConfig(), Mode.LIVE)
        self.assertFalse(result.strike_permit)

    def test_sensor_bound_is_also_used_for_engine_evaluation(self):
        app = self.application(source_delay=200)
        app.engine = engine = PermitEngine()
        app.start()
        wait_for(lambda: app.frames_processed == 1)
        self.assertGreaterEqual(engine.kwargs["capture_age_ms"], 200)

    def test_model_prediction_outside_observed_sensor_pixels_stays_unknown(self):
        app = self.application()
        app.engine = engine = PermitEngine()
        app.observed_pixels = np.ones((960, 540), bool)
        app.observed_pixels[:, :4] = False
        app.start()
        wait_for(lambda: app.frames_processed == 1)
        self.assertTrue(np.isnan(engine.depth_seen[:, :4]).all())
        self.assertTrue(np.isfinite(engine.depth_seen[:, 4:]).all())
        payload = app.depth_bytes()
        self.assertIsNotNone(payload)
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            self.assertTrue(np.isnan(arrays["depth_m"][:, :4]).all())
            self.assertTrue(np.isfinite(arrays["depth_m"][:, 4:]).all())

    def test_udp_source_age_is_recomputed_after_local_bus_delay(self):
        app = self.application(udp=True)
        app.engine = PermitEngine()
        original_publish = app.publisher.publish

        def delayed_publish(packet):
            original_publish(packet)
            # Simulate a finite local publication delay. The UDP packet must
            # include it rather than reuse an earlier capture-age snapshot.
            app.stop.wait(.04)

        app.publisher.publish = delayed_publish
        app.start()
        receiver = self.receivers[-1]
        snapshot = wait_for(lambda: receiver.receive(timeout=.05))
        self.assertGreaterEqual(snapshot.packet.capture_age_ms, 35)

    def test_actual_mmap_fault_heartbeat_revokes_previous_permission(self):
        app = self.application()
        app.engine = engine = PermitEngine()
        app.start()
        with MappedSubscriber(app.publisher.name, self.temp.name) as reader:
            wait_for(lambda: (reader.read(max_age_ms=500) or SimpleNamespace(actionable=False)).actionable)
            app.runtime.fail = True
            app.source.emit()
            wait_for(lambda: app.error)
            wait_for(lambda: app.last_packet is not None and
                     not app.last_packet.strike_permit and
                     "stale_frame" in app.last_packet.block_reasons)
            snapshot = reader.read(max_age_ms=500)
            self.assertTrue(snapshot is None or not snapshot.actionable)
            self.assertGreater(engine.reset_calls, 0)
            self.assertIsNone(app.depth_bytes())
            self.assertFalse(app.packet(999).strike_permit)

    def test_stalled_inference_cannot_refresh_old_permit(self):
        app = self.application(valid_for=200)
        app.engine = PermitEngine()
        app.start()
        wait_for(lambda: app.frames_processed == 1)
        self.assertTrue(app.packet(10).strike_permit)
        app.runtime.block = True
        app.source.emit()
        self.assertTrue(app.runtime.entered.wait(1))
        time.sleep(.25)
        self.assertFalse(app.packet(11).strike_permit)
        self.assertIn("stale_frame", app.packet(12).block_reasons)

    def test_inflight_inference_cannot_restore_permission_during_close(self):
        app = self.application()
        app.engine = PermitEngine()
        app.runtime.block = True
        app.start()
        self.assertTrue(app.runtime.entered.wait(1))
        closer = threading.Thread(target=app.close)
        closer.start()
        wait_for(lambda: app._closed)
        app.runtime.release.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        self.assertFalse(app.packet(20).strike_permit,
                         "Completing a model run after shutdown must not restore a permit")
        self.assertIsNone(app.depth_bytes())

    def test_source_fault_is_immediate_even_without_another_inference(self):
        app = self.application()
        app.engine = PermitEngine()
        app.start()
        wait_for(lambda: app.frames_processed == 1)
        self.assertTrue(app.packet(10).strike_permit)
        app.source.error = "sensor disconnected"
        self.assertFalse(app.packet(11).strike_permit)
        self.assertIsNone(app.depth_bytes())

    def test_simulation_cannot_permit_even_when_fake_engine_requests_it(self):
        app = self.application(simulation=True)
        app.engine = PermitEngine()
        app.start()
        wait_for(lambda: app.frames_processed == 1)
        result = app.packet(10)
        self.assertEqual(result.mode, Mode.SIMULATION)
        self.assertFalse(result.strike_permit)
        self.assertIn("simulation", result.block_reasons)

    def test_stale_depth_endpoint_returns503_and_current_depth_is_metric(self):
        app = self.application()
        app.start()
        wait_for(lambda: app.frames_processed == 1)
        base = self.http(app)
        app.config["display_max_age_ms"] = 1000
        with urllib.request.urlopen(base + "/depth.npz", timeout=2) as response:
            with np.load(io.BytesIO(response.read()), allow_pickle=False) as arrays:
                self.assertEqual(arrays["depth_m"].shape, (960, 540))
                self.assertTrue(arrays["metric"].item())
        with app.lock:
            app.captured_ns = time.monotonic_ns() - 2000000000
        with self.assertRaises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(base + "/depth.npz", timeout=2)
        self.assertEqual(failure.exception.code, 503)

    def test_relative_depth_is_never_labelled_metric(self):
        app = self.application(metric=False)
        app.start()
        wait_for(lambda: app.error)
        self.assertEqual(app.frames_processed, 0)
        self.assertIsNone(app.depth_bytes())
        self.assertFalse(app.packet(10).strike_permit)

    def test_sensor_resolution_set_before_first_read_and_device_released(self):
        capture_module = sys.modules[app_module.NeonCapture.__module__]
        source = app_module.NeonCapture("/dev/fake-video")

        class Camera:
            def __init__(self):
                self.properties = []
                self.released = False

            def isOpened(self):
                return True

            def set(self, key, value):
                self.properties.append((key, value))
                return True

            def read(self):
                self.assertions()
                source._stop.set()
                return True, np.zeros((1080, 1920, 3), np.uint8)

            def assertions(self):
                assert (capture_module.cv2.CAP_PROP_FRAME_WIDTH, 1920) in self.properties
                assert (capture_module.cv2.CAP_PROP_FRAME_HEIGHT, 1080) in self.properties

            def release(self):
                self.released = True

        camera = Camera()
        with patch.object(capture_module.cv2, "VideoCapture", return_value=camera):
            source._loop()
        self.assertTrue(camera.released)
        frame = source.read()
        self.assertEqual(frame.image.shape, (960, 540, 3))
        self.assertGreater(frame.captured_ns, 0)
        self.assertEqual(source.status()["timestamp_kind"], "host_read_completion")

    def test_wrong_sensor_resolution_faults_and_releases_device(self):
        capture_module = sys.modules[app_module.NeonCapture.__module__]
        source = app_module.NeonCapture("/dev/fake-video")
        camera = SimpleNamespace(isOpened=lambda: True, set=lambda *args: True,
                                 read=lambda: (True, np.zeros((480, 640, 3), np.uint8)),
                                 release=lambda: None)
        with patch.object(capture_module.cv2, "VideoCapture", return_value=camera), \
                patch.object(camera, "release") as released:
            source._loop()
        self.assertTrue(released.called)
        self.assertIsNotNone(source.status()["error"])
        with self.assertRaises(RuntimeError):
            source.read()

    def test_http_state_and_udp_are_live_diagnostics_then_revoke(self):
        app = self.application(udp=True)
        app.engine = PermitEngine()
        app.start()
        wait_for(lambda: app.frames_processed == 1)
        receiver = self.receivers[-1]
        wait_for(lambda: (receiver.receive(timeout=.02) or SimpleNamespace(actionable=False)).actionable)
        base = self.http(app)
        with urllib.request.urlopen(base + "/telemetry", timeout=2) as response:
            telemetry = json.loads(response.read())
        self.assertEqual(telemetry["mode"], "LIVE")
        app.source.error = "sensor disconnected"
        with urllib.request.urlopen(base + "/telemetry", timeout=2) as response:
            telemetry = json.loads(response.read())
        self.assertFalse(telemetry["strike_permit"])
        deadline = time.monotonic() + 1
        blocked = None
        while time.monotonic() < deadline:
            item = receiver.receive(timeout=.05)
            if item is not None and not item.packet.strike_permit:
                blocked = item
                break
        # Unknown capture age can make a fault packet immediately expired;
        # either explicit false telemetry or watchdog expiry is acceptable.
        self.assertTrue(blocked is not None or receiver.latest() is None)


if __name__ == "__main__":
    unittest.main()
