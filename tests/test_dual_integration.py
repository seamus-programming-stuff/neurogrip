"""Actual stereo matching through the local feed and both bus transports."""
import json
from pathlib import Path
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
import uuid
from http.server import ThreadingHTTPServer

import numpy as np

import neon_raw
from dual_camera import DualApplication, make_handler
from dual_config import DualConfig
from network_bus import UDPReceiver, UDPPublisher
from shared_memory_bus import SharedMemorySubscriber
from synthetic_stereo import demo_calibration, SyntheticStereoSource


ROOT = Path(__file__).resolve().parents[1]


class StereoEndToEndTests(unittest.TestCase):
    def test_actual_matching_publishes_shared_memory_udp_and_feed(self):
        config = DualConfig(shared_memory_name="ng_test_" + uuid.uuid4().hex,
                            stroke_config=str(ROOT / "config.example.json"), num_disparities=80)
        receiver = UDPReceiver(port=0, max_age_ms=500, transport_delay_bound_ms=0)
        udp = UDPPublisher(*receiver.local_address)
        receiver.expected_peer = udp.local_address
        app = DualApplication(config, demo_calibration(), source=SyntheticStereoSource(), udp=udp)
        receiver.pair_session(app.publisher.session_id)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        reader = SharedMemorySubscriber(config.shared_memory_name)
        try:
            app.start()
            thread.start()
            deadline = time.monotonic() + 3
            network = shared = None
            while time.monotonic() < deadline:
                network = receiver.receive(.05)
                shared = reader.read(max_age_ms=500)
                if network and shared and app.frame_sequence > 1:
                    break
            self.assertIsNotNone(network)
            self.assertIsNotNone(shared)
            self.assertEqual(network.packet.session_id, shared.packet.session_id)
            self.assertFalse(network.actionable)
            self.assertFalse(shared.actionable)
            self.assertEqual(network.packet.mode.name, "SIMULATION")
            self.assertIsNone(app.error)
            with app.lock:
                depth = app.depth.copy()
            self.assertAlmostEqual(float(np.nanmedian(depth[240:340, 260:350])), .500, places=3)
            self.assertAlmostEqual(float(np.nanmedian(depth[170:260, 435:490])), .750, places=3)
            self.assertAlmostEqual(float(np.nanmedian(depth[30:100, 180:500])), 1.500, places=3)
            url = "http://127.0.0.1:%d" % server.server_address[1]
            with urllib.request.urlopen(url + "/snapshot.jpg", timeout=2) as response:
                self.assertEqual(response.headers.get("Content-Type"), "image/jpeg")
                self.assertTrue(response.read().startswith(b"\xff\xd8"))
            with urllib.request.urlopen(url + "/state", timeout=2) as response:
                state = json.load(response)
                self.assertGreater(state["frame_sequence"], 0)
                self.assertFalse(state["telemetry"]["strike_permit"])
        finally:
            app.close()
            server.shutdown()
            server.server_close()
            thread.join(2)
            reader.close()
            receiver.close()

    def test_camera_movement_invalidates_metric_gap_and_download(self):
        config = DualConfig(shared_memory_name="ng_test_" + uuid.uuid4().hex,
                            stroke_config=str(ROOT / "config.example.json"), num_disparities=80)
        app = DualApplication(config, demo_calibration(), source=SyntheticStereoSource())
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        try:
            from proximity import Measurement
            with app.lock:
                app.frame_time = time.monotonic()
                app.depth = np.full((480, 640), .5, np.float32)
                app.measurement = Measurement(gap_mm=10, scale_valid=True, quality=1)
                app.error = None
                app.camera_moved = True
            packet = app.packet()
            self.assertIsNone(packet.gap_mm)
            self.assertFalse(packet.scale_valid)
            self.assertEqual(packet.quality, 0)
            self.assertIn("camera_moved", packet.block_reasons)
            thread.start()
            with self.assertRaises(urllib.error.HTTPError) as failure:
                urllib.request.urlopen("http://127.0.0.1:%d/depth.npz" % server.server_address[1], timeout=2)
            self.assertEqual(failure.exception.code, 503)
        finally:
            app.close()
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_capture_failure_exits_for_systemd_restart(self):
        class FailedCapture:
            def __init__(self, *_args):
                self.stop = threading.Event()
                self.condition = threading.Condition()
            def capture(self):
                return
            def metadata(self):
                return {"error": "sensor not ready"}
        with patch("sys.argv", ["neon_raw.py", "--bind", "127.0.0.1", "--port", "0"]), \
             patch.object(neon_raw, "CaptureState", FailedCapture), \
             patch.object(neon_raw.signal, "signal"):
            self.assertEqual(neon_raw.main(), 1)


if __name__ == "__main__":
    unittest.main()
