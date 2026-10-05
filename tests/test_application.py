"""Application/HTTP integration checks with no camera, model, or actuator I/O."""
from argparse import Namespace
from contextlib import contextmanager
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import numpy as np

from camera_source import FramePacket
from depth_backend import DepthResult
from neurogrip import Application, ThreadingHTTPServer, make_handler, encode_jpeg
from protocol import Direction, Mode, decode_can_fd, decode_json
from proximity import Measurement
from synthetic_scene import K, demo_config, scene
from vision_tracking import TrackResult


class FakeSource:
    def __init__(self, packet=None):
        self.packet = packet
        self.closed = False

    def start(self):
        return self

    def read(self):
        return None if self.closed else self.packet

    def status(self):
        return {"state": "closed" if self.closed else "running",
                "timestamp_kind": "host_read_completion", "sensor_synchronized": False}

    def close(self):
        self.closed = True
        return True


class FakeBackend:
    def __init__(self, result=None, error=None, block=False):
        self.result = result
        self.error = error
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()

    def infer(self, image, matrix):
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("Fake model timed out")
        if self.error:
            raise RuntimeError(self.error)
        return self.result


def permitted_measurement():
    return Measurement(gap_mm=20.0, direction=Direction.FORWARDS,
                       quality=0.9, scale_valid=True, reachable=True,
                       candidate_permit=True, strike_permit=True)


@contextmanager
def local_server(app):
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


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=2) as reply:
        return reply.status, reply.headers, reply.read()


def post(base, path, body):
    request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(request, timeout=2))


class ApplicationTests(unittest.TestCase):
    def make_app(self, *, source="0", image=None, backend=None, camera=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        args = Namespace(source=source, image=image, demo=False, config="unused.json",
                         camera=None, device="cpu", process_res=364, offline=True,
                         output=directory.name)
        camera = camera or FakeSource()
        backend = backend or FakeBackend()
        with patch("neurogrip.StrikeConfig.load", return_value=demo_config()), \
                patch("neurogrip.MetricDepthBackend", return_value=backend), \
                patch("neurogrip.CameraSource", return_value=camera):
            app = Application(args)
        app.matrix = K
        self.addCleanup(app.close)
        return app

    @staticmethod
    def seed_permit(app):
        with app._lock:
            app.armed = True
            app.frame_time = time.monotonic()
            app.measurement = permitted_measurement()

    def process_thread(self, app):
        worker = threading.Thread(target=app._process, daemon=True)
        worker.start()
        app._processing = worker
        return worker

    def wait_for(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(predicate(), "Application condition did not arrive")

    def test_http_freshness_revokes_without_processing(self):
        app = self.make_app()
        self.seed_permit(app)
        with local_server(app) as base:
            self.assertEqual(get(base, "/present")[2], b"1")
            self.assertTrue(decode_json(get(base, "/telemetry")[2]).strike_permit)
            # No processing or publisher is running: endpoints must age the
            # previous result themselves rather than depend on another update.
            with app._lock:
                app.frame_time = time.monotonic() - app.config.valid_for_ms / 1000 - .05
            self.assertEqual(get(base, "/present")[2], b"0")
            telemetry = decode_json(get(base, "/telemetry")[2])
            self.assertFalse(telemetry.strike_permit)
            self.assertEqual(telemetry.direction, Direction.UNKNOWN)
            self.assertIn("stale_frame", telemetry.block_reasons)
            state = json.loads(get(base, "/state")[2])
            self.assertFalse(state["measurement"]["strike_permit"])
            self.assertFalse(state["measurement"]["candidate_permit"])

    def test_source_delay_is_added_once_and_can_expire_fresh_read(self):
        app = self.make_app()
        self.seed_permit(app)
        app.config.source_delay_bound_ms = 125
        app.frame_time = 99.75
        with patch("neurogrip.time.monotonic", return_value=100.0):
            packet = app.packet()
        self.assertEqual(packet.capture_age_ms, 375)
        self.assertTrue(packet.strike_permit)
        app.config.source_delay_bound_ms = app.config.valid_for_ms
        app.frame_time = time.monotonic()
        self.assertFalse(app.packet().strike_permit)
        self.assertEqual(app.timestamp_kind, "host_read_completion")

    def test_still_preview_repeats_display_without_refreshing_capture(self):
        app = self.make_app()
        self.seed_permit(app)
        original_capture = app.frame_time
        jpeg = encode_jpeg(np.zeros((40, 80, 3), np.uint8))
        app.combined_jpeg = jpeg
        app.frame_sequence = 0
        part = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        with local_server(app) as base:
            with urllib.request.urlopen(base + "/stream", timeout=2) as response:
                data = response.read(len(part) * 2)
            self.assertEqual(data, part * 2)
            self.assertEqual(app.frame_time, original_capture)
            self.assertEqual(app.frame_sequence, 0)
            self.assertFalse(app.packet().strike_permit)

    def test_arm_reset_teach_immediately_revoke(self):
        app = self.make_app()
        for action, body in (("arm", {"armed": False}), ("arm", {"armed": True}),
                             ("reset", {}), ("teach", {"kind": "finger", "bbox": [5, 5, 20, 20]})):
            with self.subTest(action=action, body=body):
                self.seed_permit(app)
                app.frozen = np.zeros((50, 50, 3), np.uint8)
                generation = app._generation
                app.command(action, body)
                self.assertGreater(app._generation, generation)
                self.assertFalse(app.packet().strike_permit)
                self.assertFalse(app.measurement.candidate_permit)

    def test_still_and_video_replay_never_permit(self):
        for source, image in (("recording.avi", None), ("file:///C:/recording.avi", None),
                              ("0", "snapshot.jpg")):
            with self.subTest(source=source, image=image):
                app = self.make_app(source=source, image=image)
                self.seed_permit(app)
                packet = app.packet()
                self.assertEqual(packet.mode, Mode.SIMULATION)
                self.assertFalse(packet.strike_permit)
                self.assertIn("simulation", packet.block_reasons)

    def test_close_immediately_revokes_without_publisher(self):
        app = self.make_app()
        self.seed_permit(app)
        app.close()
        self.assertFalse(app.packet().strike_permit)
        self.assertFalse(app.armed)

    def test_state_does_not_mix_reset_snapshot_with_old_permit(self):
        app = self.make_app()
        self.seed_permit(app)
        packet_ready, reset_done = threading.Event(), threading.Event()
        real_packet = app.packet

        def delayed_packet():
            result = real_packet()
            packet_ready.set()
            # If state owns the lock, reset will wait until the snapshot is
            # complete. If it does not, reset can invalidate this old packet.
            reset_done.wait(.1)
            return result

        def reset_on_packet():
            if packet_ready.wait(1):
                app.command("reset", {})
                reset_done.set()

        resetter = threading.Thread(target=reset_on_packet, daemon=True)
        resetter.start()
        try:
            with patch.object(app, "packet", side_effect=delayed_packet):
                state = app.state()
        finally:
            resetter.join(1)
        self.assertFalse(state["telemetry"]["strike_permit"] and not state["armed"],
                         "State combined permission from before reset with armed=false after reset")

    def test_in_flight_inference_cannot_undo_reset(self):
        bgr, result, _, _ = scene(0)
        backend = FakeBackend(result, block=True)
        camera = FakeSource(FramePacket(bgr, 1, time.monotonic()))
        app = self.make_app(backend=backend, camera=camera)
        self.seed_permit(app)
        with patch.object(app.engine, "evaluate", return_value=permitted_measurement()):
            self.process_thread(app)
            self.assertTrue(backend.started.wait(1))
            app.command("reset", {})
            backend.release.set()
            time.sleep(.1)
            self.assertFalse(app.packet().strike_permit)
            self.assertEqual(app.frame_sequence, 0, "Pre-reset inference was committed")
        app.close()

    def test_model_error_immediately_blocks_previous_permit(self):
        bgr, _, _, _ = scene(0)
        backend = FakeBackend(error="test model failure")
        app = self.make_app(backend=backend, camera=FakeSource(FramePacket(bgr, 1, time.monotonic())))
        self.seed_permit(app)
        self.process_thread(app)
        self.wait_for(lambda: app.last_error is not None)
        self.assertFalse(app.packet().strike_permit)
        self.assertFalse(app.armed)
        self.assertEqual(app.last_error, "test model failure")

    def test_calibration_error_immediately_blocks_previous_permit(self):
        bgr, _, _, _ = scene(0)
        app = self.make_app(camera=FakeSource(FramePacket(bgr, 1, time.monotonic())))
        self.seed_permit(app)
        with patch.object(app.calibration, "prepare", side_effect=ValueError("calibration dimensions changed")):
            self.process_thread(app)
            self.wait_for(lambda: app.last_error is not None)
            self.assertFalse(app.packet().strike_permit)
            self.assertFalse(app.armed)
        app.close()

    def test_extreme_finite_depth_does_not_break_telemetry(self):
        bgr, _, finger, target = scene(0)
        extreme = np.full(bgr.shape[:2], 1e20, np.float32)
        depth = DepthResult(extreme, np.ones_like(extreme, bool), "fake extreme", True, extreme)
        app = self.make_app(backend=FakeBackend(depth), camera=FakeSource(FramePacket(bgr, 1, time.monotonic())))
        self.seed_permit(app)
        for kind, mask in (("finger", finger), ("target", target)):
            result = TrackResult((0, 0, 10, 10), mask, True, 1.0, "fake")
            app.trackers[kind] = type("FakeTracker", (), {"update": lambda self, image, value=result: value})()
        self.process_thread(app)
        self.wait_for(lambda: app.frame_sequence == 1)
        self.assertFalse(app.packet().strike_permit)
        json.dumps(app.state(), allow_nan=False)

    def test_nonfinite_depth_is_blocked_and_state_stays_finite(self):
        bgr, _, _, _ = scene(0)
        invalid = np.full(bgr.shape[:2], np.nan, np.float32)
        invalid[0, 0] = np.inf
        depth = DepthResult(invalid, np.zeros_like(invalid, bool), "fake invalid", True, invalid)
        app = self.make_app(backend=FakeBackend(depth), camera=FakeSource(FramePacket(bgr, 1, time.monotonic())))
        self.process_thread(app)
        self.wait_for(lambda: app.frame_sequence == 1)
        self.assertFalse(app.packet().strike_permit)
        payload = json.dumps(app.state(), allow_nan=False)
        self.assertNotIn("NaN", payload)
        self.assertNotIn("Infinity", payload)

    def test_http_endpoints_and_command_validation(self):
        app = self.make_app()
        with local_server(app) as base:
            status, headers, body = get(base, "/")
            self.assertEqual(status, 200)
            self.assertIn("text/html", headers["Content-Type"])
            self.assertGreater(len(body), 100)
            self.assertFalse(decode_can_fd(get(base, "/telemetry.bin")[2]).strike_permit)
            json.loads(get(base, "/state")[2], parse_constant=lambda value: self.fail(value))
            with self.assertRaises(urllib.error.HTTPError) as failure:
                get(base, "/snapshot.jpg")
            self.assertEqual(failure.exception.code, 503)
            with self.assertRaises(urllib.error.HTTPError) as failure:
                get(base, "/depth.npz")
            self.assertEqual(failure.exception.code, 503)
            self.assertTrue(post(base, "/arm", {"armed": False})["ok"])
            with self.assertRaises(urllib.error.HTTPError) as failure:
                post(base, "/arm", {"armed": 1})
            self.assertEqual(failure.exception.code, 400)
            with self.assertRaises(urllib.error.HTTPError) as failure:
                get(base, "/missing")
            self.assertEqual(failure.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
