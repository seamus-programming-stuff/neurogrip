"""Feed-only calibrated stereo display, shared memory and Ethernet telemetry."""
import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
import webbrowser

import cv2
import numpy as np

from dual_config import DualConfig, controller_address
from protocol import Direction, Mode, VisionTelemetry
from proximity import Measurement, ProximityEngine, StrikeConfig
from shared_memory_bus import SharedMemoryPublisher
from network_bus import UDPPublisher
from stereo_core import StereoCalibration, StereoDepth, check_epipolar_alignment
from stereo_source import StereoSource
from vision_tracking import RegionTracker


FEED_HTML = b'''<!doctype html><html><head><meta charset="utf-8"><title>Neurogrip stereo feed</title><style>html,body{margin:0;width:100%;height:100%;background:#000;overflow:hidden}img{width:100%;height:100%;object-fit:contain}</style></head><body><img src="/stream" alt="Stereo camera and depth feed"></body></html>'''


def encode_jpeg(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Feed JPEG encoding failed")
    return encoded.tobytes()


def render_feed(left, depth, minimum_mm, maximum_mm, *, simulation=False, measurement=None):
    valid = np.isfinite(depth) & (depth > 0)
    # Fixed metric colour scale across time; missing depth is black.
    scaled = np.clip((depth * 1000 - minimum_mm) / (maximum_mm - minimum_mm), 0, 1)
    levels = np.where(valid, (1 - np.nan_to_num(scaled)) * 255, 0).astype(np.uint8)
    colour = cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    overlay = left.copy()
    if simulation:
        cv2.putText(overlay, "SIMULATION", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
    if measurement is not None and measurement.gap_mm is not None:
        cv2.putText(overlay, "Gap %.1f mm" % measurement.gap_mm, (14, left.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    return np.hstack((overlay, colour))


class DualApplication:
    def __init__(self, config, calibration, *, source=None, backend=None, publisher=None, udp=None, mode=Mode.LIVE):
        self.config = config.checked()
        self.calibration = calibration
        self.mode = Mode.SIMULATION if calibration.metadata.get("simulation", False) else mode
        self.source = source or StereoSource(calibration.left_source, calibration.right_source,
            stale_after=config.stale_after_ms / 1000, max_pair_skew_ms=config.max_pair_skew_ms)
        self.backend = backend or StereoDepth(calibration, num_disparities=config.num_disparities,
            block_size=config.block_size, min_depth_mm=config.min_depth_mm, max_depth_mm=config.max_depth_mm)
        self.publisher = None
        self.udp = udp
        self.stroke_config = StrikeConfig.load(config.stroke_config)
        # The stereo range, not the old monocular scale correction, defines this geometry.
        self.stroke_config = replace(self.stroke_config, min_depth_mm=config.min_depth_mm,
            max_depth_mm=config.max_depth_mm, depth_scale=1.0,
            valid_for_ms=min(self.stroke_config.valid_for_ms, config.stale_after_ms)).checked()
        self.engine = ProximityEngine(self.stroke_config)
        self.trackers = {"finger": RegionTracker(), "target": RegionTracker()}
        self.track_status = {name: "not_taught" for name in self.trackers}
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.measurement = Measurement(block_reasons=("unknown_depth",))
        self.frame_time = None
        self.frame_sequence = 0
        self.feed_sequence = 0
        self.last_pair_skew_ms = None
        self.source_epoch = None
        self.depth = None
        self.jpeg = None
        self.error = "waiting_for_stereo_pair"
        self.network_error = None
        self.processing_ms = None
        self.alignment = None
        self.camera_moved = False
        self.sequence = 0
        self.threads = []
        self.closed = False
        self._resources_closed = False
        self._teach_reference()
        # Reserve the single-writer bus only after all fallible setup succeeds.
        self.publisher = publisher or SharedMemoryPublisher(name=config.shared_memory_name)

    def _teach_reference(self):
        reference_path = self.calibration.metadata.get("regions_reference")
        if not reference_path:
            return
        reference = cv2.imread(str(reference_path))
        if reference is None or tuple(reference.shape[1::-1]) != tuple(self.calibration.image_size):
            raise ValueError("Taught reference image is missing or does not match calibration")
        for name, tracker in self.trackers.items():
            bbox = self.calibration.metadata.get(name + "_bbox")
            if bbox is not None:
                result = tracker.initialize(reference, bbox)
                self.track_status[name] = result.reason

    def start(self):
        self.source.start()
        self.threads = [threading.Thread(target=self._process, daemon=True, name="stereo-processing"),
                        threading.Thread(target=self._publish, daemon=True, name="stereo-bus")]
        for thread in self.threads:
            thread.start()
        return self

    def _process(self):
        last_pair = -1
        while not self.stop.is_set():
            try:
                pair = self.source.read()
            except Exception as error:
                self.engine.reset()
                with self.lock:
                    self.error = "Stereo capture failed: " + str(error)
                    self.frame_time = None
                    self.measurement = Measurement(block_reasons=("stale_frame",))
                self.stop.wait(.05)
                continue
            if pair is None or pair.sequence == last_pair:
                with self.lock:
                    upstream = self.stroke_config.source_delay_bound_ms or 0
                    expired = self.frame_time is not None and (time.monotonic() - self.frame_time) * 1000 + upstream >= self.stroke_config.valid_for_ms
                    if expired:
                        self.frame_time = None
                        self.measurement = Measurement(block_reasons=("stale_frame",))
                if expired:
                    self.engine.reset()
                self.stop.wait(.005)
                continue
            last_pair = pair.sequence
            try:
                started = time.monotonic()
                epoch = getattr(pair, "source_epoch", 0)
                if self.source_epoch is not None and epoch != self.source_epoch:
                    self.engine.reset()
                    self.trackers = {name: RegionTracker() for name in self.trackers}
                    with self.lock:
                        self.frame_time = None
                        self.depth = None
                        self.measurement = Measurement(block_reasons=("stale_frame", "finger_missing", "target_missing"))
                        self.track_status = {name: "camera_reconnected_reteach_required" for name in self.trackers}
                with self.lock:
                    self.source_epoch = epoch
                result = self.backend.compute(pair.left.image, pair.right.image)
                if pair.sequence % 10 == 1 or self.alignment is None:
                    self.alignment = check_epipolar_alignment(result.rectified_left, result.rectified_right)
                    if self.alignment.get("aligned") is False:
                        self.camera_moved = True
                tracks = {name: tracker.update(result.rectified_left) for name, tracker in self.trackers.items()}
                masks = {name: np.zeros(result.depth_m.shape, bool) if not track.valid else track.mask > 0 for name, track in tracks.items()}
                age_ms = max(0, (time.monotonic() - pair.read_time) * 1000) + (self.stroke_config.source_delay_bound_ms or 0)
                measurement = self.engine.evaluate(result.depth_m, masks["finger"], masks["target"],
                    self.calibration.K_rect, metric=True, mode=self.mode,
                    now=pair.read_time, capture_age_ms=age_ms,
                    finger_valid=tracks["finger"].valid, target_valid=tracks["target"].valid,
                    armed=self.config.recommendations_enabled,
                    tracking_quality=min(track.quality for track in tracks.values()))
                extra = []
                if not self.config.stereo_timing_validated or self.config.exposure_skew_bound_ms is None:
                    extra.append("calibration_unvalidated")
                if pair.pair_skew_ms > self.config.max_pair_skew_ms:
                    extra.append("stale_frame")
                if self.camera_moved:
                    extra.append("camera_moved")
                if self.alignment.get("aligned") is None:
                    extra.append("low_quality")
                if extra:
                    self.engine.reset()
                    measurement = replace(measurement, strike_permit=False, candidate_permit=False,
                        block_reasons=tuple(dict.fromkeys(measurement.block_reasons + tuple(extra))))
                if self.camera_moved:
                    measurement = replace(measurement, gap_mm=None, quality=0.0, scale_valid=False)
                feed = render_feed(result.rectified_left, result.depth_m, self.config.min_depth_mm,
                                   self.config.max_depth_mm, simulation=self.mode == Mode.SIMULATION, measurement=measurement)
                for name, track in tracks.items():
                    if track.valid and track.bbox:
                        x, y, w, h = track.bbox
                        cv2.rectangle(feed, (x, y), (x + w, y + h), (0, 255, 0) if name == "target" else (0, 150, 255), 2)
                jpeg = encode_jpeg(feed)
                with self.lock:
                    if self.stop.is_set():
                        break
                    self.measurement = measurement
                    self.frame_time = pair.read_time
                    self.frame_sequence = pair.sequence
                    self.feed_sequence += 1
                    self.depth = result.depth_m.copy()
                    self.jpeg = jpeg
                    self.last_pair_skew_ms = pair.pair_skew_ms
                    self.processing_ms = (time.monotonic() - started) * 1000
                    self.track_status = {name: track.reason for name, track in tracks.items()}
                    self.error = None
            except Exception as error:
                self.engine.reset()
                with self.lock:
                    self.error = str(error)
                    self.frame_time = None
                    self.measurement = Measurement(block_reasons=("invalid_geometry",))
                    self.depth = None

    def packet(self):
        now = time.monotonic()
        with self.lock:
            measurement = self.measurement
            frame_time = self.frame_time
            error = self.error
            closed = self.closed or self.stop.is_set()
            sequence = self.sequence
            camera_moved = self.camera_moved
            epoch = self.source_epoch
        upstream = self.stroke_config.source_delay_bound_ms or 0
        age = 65535 if frame_time is None else min(65535, int(max(0, (now - frame_time) * 1000) + upstream))
        try:
            source_status = self.source.status()
            source_invalid = epoch is not None and source_status.get("source_epoch", epoch) != epoch
            source_invalid |= source_status.get("fresh") is False
            for side in ("left", "right"):
                if side in source_status:
                    source_invalid |= source_status[side].get("state") != "running" or source_status[side].get("fresh") is False
        except Exception:
            source_invalid = True
        stale = closed or camera_moved or source_invalid or error is not None or age >= self.stroke_config.valid_for_ms
        reasons = list(measurement.block_reasons)
        if stale:
            reasons.append("stale_frame")
        if frame_time is None:
            reasons.append("unknown_depth")
        if not self.config.stereo_timing_validated:
            reasons.append("calibration_unvalidated")
        if camera_moved:
            reasons.append("camera_moved")
        reasons = tuple(dict.fromkeys(reasons))
        return VisionTelemetry(session_id=self.publisher.session_id, sequence=sequence,
            timestamp_ms=int(now * 1000), capture_age_ms=age, valid_for_ms=self.stroke_config.valid_for_ms,
            mode=self.mode, direction=Direction.UNKNOWN if stale else measurement.direction,
            strike_permit=bool(measurement.strike_permit and not reasons and not stale),
            gap_mm=None if stale else measurement.gap_mm, quality=0.0 if stale else measurement.quality,
            scale_valid=bool(measurement.scale_valid and not stale), block_reasons=reasons)

    def _publish(self):
        while not self.stop.is_set():
            with self.lock:
                self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            try:
                packet = self.packet()
                self.publisher.publish(packet)
                if self.udp is not None:
                    self.udp.publish(packet)
                self.network_error = None
            except Exception as error:
                self.network_error = str(error)
            self.stop.wait(self.config.heartbeat_ms / 1000)

    def display_frame(self):
        with self.lock:
            jpeg, sequence, frame_time = self.jpeg, self.feed_sequence, self.frame_time
            error = self.error
            camera_moved = self.camera_moved
        if jpeg is not None and frame_time is not None and not camera_moved and not self.stop.is_set() and not error and time.monotonic() - frame_time < self.config.stale_after_ms / 1000:
            return jpeg, sequence
        width, height = self.calibration.image_size
        blank = np.zeros((height, width * 2, 3), np.uint8)
        message = "SIMULATION - " if self.mode == Mode.SIMULATION else ""
        message += "RECALIBRATE CAMERAS" if camera_moved else "NO FRESH STEREO PAIR"
        cv2.putText(blank, message, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 200, 255), 2)
        return encode_jpeg(blank), -1

    def state(self):
        with self.lock:
            packet = self.packet()
            result = {"telemetry": packet.to_dict(), "frame_sequence": self.frame_sequence,
                "pair_receive_skew_ms": self.last_pair_skew_ms, "processing_ms": self.processing_ms,
                "error": self.error, "tracks": dict(self.track_status), "network_error": self.network_error}
        result.update({"camera": self.source.status(), "backend": "OpenCV StereoSGBM",
            "image_size": list(self.calibration.image_size), "metric": True,
            "exposure_sync_verified": self.config.stereo_timing_validated,
            "shared_memory_name": self.config.shared_memory_name,
            "epipolar_alignment": self.alignment,
            "udp_source": None if self.udp is None else self.udp.local_address})
        return result

    def close(self):
        with self.lock:
            if self._resources_closed:
                return
            self._resources_closed = True
            self.closed = True
        self.stop.set()
        try:
            self.source.close()
        except Exception as error:
            self.error = "Capture shutdown failed: " + str(error)
        for thread in self.threads:
            thread.join(timeout=2)
        with self.lock:
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        try:
            self.publisher.publish(self.packet())
            if self.udp is not None:
                self.udp.publish(self.packet())
        except Exception as error:
            self.network_error = "Final blocked telemetry failed: " + str(error)
        finally:
            try:
                if self.udp is not None:
                    self.udp.close()
            finally:
                self.publisher.close()


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_bytes(self, data, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    return self.send_bytes(FEED_HTML, "text/html; charset=utf-8")
                if path == "/state":
                    return self.send_bytes(json.dumps(app.state(), allow_nan=False).encode(), "application/json")
                if path == "/snapshot.jpg":
                    return self.send_bytes(app.display_frame()[0], "image/jpeg")
                if path == "/telemetry":
                    return self.send_bytes(json.dumps(app.packet().to_dict(), allow_nan=False).encode(), "application/json")
                if path == "/depth.npz":
                    with app.lock:
                        depth = None if app.depth is None else app.depth.copy()
                        timestamp = app.frame_time
                        sequence = app.frame_sequence
                        moved = app.camera_moved
                    if moved or depth is None or timestamp is None or time.monotonic() - timestamp >= app.config.stale_after_ms / 1000:
                        return self.send_bytes(b"No fresh stereo depth", "text/plain", 503)
                    output = io.BytesIO()
                    np.savez_compressed(output, depth_m=depth, valid_mask=np.isfinite(depth),
                        K=app.calibration.K_rect, frame_sequence=sequence, mode=app.mode.name,
                        timestamp_kind="host_read_completion", read_time_monotonic=timestamp)
                    return self.send_bytes(output.getvalue(), "application/octet-stream")
                if path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last_sequence, last_sent = None, 0.0
                    while not app.stop.is_set():
                        jpeg, sequence = app.display_frame()
                        if sequence != last_sequence or time.monotonic() - last_sent >= .5:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                            last_sequence, last_sent = sequence, time.monotonic()
                        app.stop.wait(.025)
                    return
                return self.send_bytes(b"Not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def run(config_path="dual_config.json", *, open_browser=True, demo=False, port=None):
    config = DualConfig.load(config_path) if not demo else DualConfig(shared_memory_name="neurogrip_demo_v1", num_disparities=80)
    if port is not None:
        config.display_port = port
    config.checked()
    if demo:
        from synthetic_stereo import demo_calibration, SyntheticStereoSource
        calibration, source = demo_calibration(), SyntheticStereoSource()
    else:
        calibration, source = StereoCalibration.load(config.calibration), None
        if calibration.metadata.get("simulation"):
            raise ValueError("Synthetic calibration cannot be used for physical cameras")
        from dual_calibrate import probe_neon_stream
        current_states = [probe_neon_stream(address) for address in (calibration.left_source, calibration.right_source)]
        for state in current_states:
            if (state.get("width"), state.get("height")) != tuple(calibration.image_size):
                raise ValueError("Raw stream resolution changed; recalibrate")
        saved_states = calibration.metadata.get("camera_states_at_calibration", [])
        for old, new in zip(saved_states, current_states):
            if old.get("rotated_90_clockwise") != new.get("rotated_90_clockwise"):
                raise ValueError("Camera rotation changed; recalibrate")
    destination = controller_address(config.controller)
    udp = None if destination is None else UDPPublisher(*destination, source_port=config.controller_source_port)
    app = None
    server = None
    try:
        app = DualApplication(config, calibration, source=source, udp=udp, mode=Mode.SIMULATION if demo else Mode.LIVE)
        server = ThreadingHTTPServer((config.display_host, config.display_port), make_handler(app))
        server.daemon_threads = True
        app.start()
        url = "http://%s:%d/" % (config.display_host, config.display_port)
        print("Feed:", url, "|", app.mode.name, flush=True)
        print("Shared memory:", config.shared_memory_name, "| session:", app.publisher.session_id, flush=True)
        print("Ethernet:", "disabled (no controller address)" if udp is None else "%s -> %s" % (udp.local_address, destination), flush=True)
        print("Receive-time pairing does not prove synchronized exposure. No motor commands are sent.", flush=True)
        if open_browser:
            webbrowser.open(url)
        server.serve_forever(poll_interval=.2)
    finally:
        if app is not None:
            app.close()
        elif udp is not None:
            udp.close()
        if server is not None:
            server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dual_config.json")
    parser.add_argument("--demo", action="store_true", help="synthetic stereo feed; no physical permission")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        run(args.config, open_browser=not args.no_browser, demo=args.demo, port=args.port)
    except KeyboardInterrupt:
        print("Stopped.")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, "Stereo startup failed: %s\n" % error)


if __name__ == "__main__":
    main()
