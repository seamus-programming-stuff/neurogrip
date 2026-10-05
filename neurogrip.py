"""Local single-camera depth/proximity dashboard; no actuator/bus transmission."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np

from camera_source import CameraSource
from depth_backend import MetricDepthBackend, DepthResult
from protocol import Direction, Mode, VisionTelemetry, encode_json, encode_can_fd
from proximity import Measurement, ProximityEngine, StrikeConfig, validate_intrinsics
from vision_tracking import RegionTracker, TrackResult
from synthetic_scene import scene, demo_config, K as DEMO_K


ROOT = Path(__file__).resolve().parent


class CameraCalibration:
    def __init__(self, path=None):
        self.matrix, self.distortion, self.size = None, None, None
        self._maps = None
        if path:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
            self.matrix = validate_intrinsics(values["K"])
            self.size = tuple(values["image_size"])
            if len(self.size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 14 for v in self.size):
                raise ValueError("camera image_size must be [width,height] of actual processed frames")
            self.distortion = np.asarray(values.get("distortion", [0] * 5), dtype=np.float64)
            if self.distortion.ndim != 1 or self.distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
                raise ValueError("distortion must contain 4,5,8,12 or14 finite coefficients")

    def prepare(self, bgr):
        if self.matrix is None:
            return bgr
        size = (bgr.shape[1], bgr.shape[0])
        if size != self.size:
            raise ValueError(f"Camera calibration size {self.size} differs from input {size}; recalibrate this capture mode")
        if self._maps is None:
            self._maps = cv2.initUndistortRectifyMap(self.matrix, self.distortion, None, self.matrix, size, cv2.CV_32FC1)
        return cv2.remap(bgr, *self._maps, cv2.INTER_LINEAR)


def depth_colors(result, low_m=.1, high_m=2.0):
    values = result.visualization_depth
    finite = np.isfinite(values) & result.valid_mask
    if result.metric:
        low, high = low_m, high_m
        label = f"Estimated depth Z: {low:.2f} to {high:.2f} m"
    elif finite.any():
        low, high = np.percentile(values[finite], [2, 98])
        label = "Relative depth - no millimetre scale"
    else:
        low, high, label = 0, 1, "Depth unavailable"
    values_safe = np.where(finite, values, low)
    normalized = np.clip((values_safe - low) / max(1e-6, high - low), 0, 1)
    colored = cv2.applyColorMap(np.uint8(255 * (1 - normalized)), cv2.COLORMAP_TURBO)
    colored[~finite] = 0
    cv2.rectangle(colored, (0, 0), (colored.shape[1], 42), (20, 22, 28), -1)
    cv2.putText(colored, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return colored


def draw_view(bgr, depth, tracks, measurement, mode, min_depth, max_depth):
    overlay = bgr.copy()
    for name, track in tracks.items():
        color = (255, 180, 40) if name == "finger" else (80, 230, 100)
        if track.valid and track.mask is not None:
            contour, _ = cv2.findContours(track.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contour, -1, color, 2)
            if track.bbox:
                x, y, _, _ = track.bbox
                cv2.putText(overlay, "Fingertip" if name == "finger" else "Target", (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    cv2.rectangle(overlay, (0, overlay.shape[0] - 65), (overlay.shape[1], overlay.shape[0]), (20, 22, 28), -1)
    gap = "unknown" if measurement.gap_mm is None else f"{measurement.gap_mm:.1f} mm (estimated)"
    cv2.putText(overlay, f"Gap: {gap}", (12, overlay.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    text = f"{mode.name} | {measurement.direction.name} | Estimated reachable: {int(measurement.reachable)}"
    cv2.putText(overlay, text, (12, overlay.shape[0] - 13), cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 230, 100) if measurement.strike_permit else (0, 180, 255), 1)
    return np.concatenate((overlay, depth_colors(depth, min_depth / 1000, max_depth / 1000)), axis=1)


def encode_jpeg(image):
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return data.tobytes()


class Application:
    def __init__(self, args):
        self.args = args
        source = int(args.source) if args.source.isdecimal() else args.source
        is_replay = args.image is not None or (isinstance(source, str) and ("://" not in source or urlparse(source).scheme.lower() == "file"))
        self.mode = Mode.SIMULATION if args.demo or is_replay else Mode.LIVE
        self.config = demo_config() if args.demo else StrikeConfig.load(args.config)
        self.calibration = CameraCalibration(args.camera)
        self.matrix = DEMO_K if args.demo else self.calibration.matrix
        self.engine = ProximityEngine(self.config)
        self.backend = None if args.demo else MetricDepthBackend(device=args.device, process_res=args.process_res, local_files_only=args.offline)
        self.camera = None if args.demo or args.image else CameraSource(source, stale_after=self.config.valid_for_ms / 1000).start()
        self._lock = threading.RLock()
        self.stop = threading.Event()
        self.trackers = {"finger": RegionTracker(), "target": RegionTracker()}
        self.tracks = {key: TrackResult(None, None, False, 0, "not_initialized") for key in self.trackers}
        self.armed = bool(args.demo)
        self.session_id = secrets.randbits(32)
        self.sequence = 0
        self.frame_sequence = 0
        self.frame_time = None
        self.timestamp_kind = "synthetic" if args.demo else "host_read_completion"
        self.image = self.combined_jpeg = self.depth = self.frozen = None
        self.measurement = Measurement(block_reasons=("unknown_depth", "controller_unarmed"))
        self.last_error = None
        self.inference_ms = 0.0
        self.output_path = Path(args.output)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.pending_teach = []
        self._generation = 0
        self._generation_applied = 0
        self._publisher = None
        self._processing = None
        self._source_was_stale = False

    def packet(self):
        """Freshness is recalculated even if camera/model processing has stalled."""
        with self._lock:
            measurement = self.measurement
            upstream_age = self.config.source_delay_bound_ms or 0
            age = 65535 if self.frame_time is None else min(65535, int(max(0, (time.monotonic() - self.frame_time) * 1000 + upstream_age)))
            reasons = list(measurement.block_reasons)
            permit = bool(measurement.strike_permit and self.armed)
            direction = measurement.direction
            if age >= self.config.valid_for_ms:
                reasons.append("stale_frame")
                permit, direction = False, Direction.UNKNOWN
            if not self.armed:
                reasons.append("controller_unarmed")
                permit = False
            if self.mode == Mode.SIMULATION:
                reasons.append("simulation")
                permit = False
            reasons = tuple(dict.fromkeys(reasons))
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            return VisionTelemetry(self.session_id, self.sequence, age, self.config.valid_for_ms,
                                   self.mode, direction, permit, measurement.gap_mm,
                                   measurement.quality, measurement.scale_valid, reasons,
                                   timestamp_ms=int(time.monotonic() * 1000))

    def state(self):
        with self._lock:
            packet = self.packet()
            measurement = asdict(self.measurement)
            measurement["direction"] = self.measurement.direction.name
            # The independent freshness packet is authoritative for permission.
            measurement["strike_permit"] = packet.strike_permit
            if "stale_frame" in packet.block_reasons:
                measurement["candidate_permit"] = False
            tracks = {key: {"valid": value.valid, "quality": value.quality, "reason": value.reason,
                            "bbox": value.bbox} for key, value in self.tracks.items()}
            return {"telemetry": packet.to_dict(), "measurement": measurement, "tracks": tracks,
                    "armed": self.armed, "frame_sequence": self.frame_sequence,
                    "timestamp_kind": self.timestamp_kind, "inference_ms": self.inference_ms,
                    "metric": False if self.depth is None else self.depth.metric,
                    "image_size": None if self.image is None else [self.image.shape[1], self.image.shape[0]],
                    "backend": None if self.depth is None else self.depth.backend,
                    "error": self.last_error,
                    "camera": None if self.camera is None else self.camera.status()}

    def _publish(self):
        previous_permit = None
        with (self.output_path / "telemetry.jsonl").open("a", encoding="utf-8", buffering=1) as output:
            while not self.stop.is_set():
                packet = self.packet()
                output.write(encode_json(packet) + "\n")
                if packet.strike_permit != previous_permit:
                    print(f"/present={int(packet.strike_permit)}  {packet.direction.name}  {','.join(packet.block_reasons) or 'vision conditions met'}", flush=True)
                    previous_permit = packet.strike_permit
                self.stop.wait(.1)
            # A clean exit publishes explicit revoked permission.
            with self._lock:
                self.armed = False
            output.write(encode_json(self.packet()) + "\n")

    def start(self):
        self._publisher = threading.Thread(target=self._publish, daemon=True, name="telemetry")
        self._processing = threading.Thread(target=self._process, daemon=True, name="vision")
        self._publisher.start()
        self._processing.start()

    def _process(self):
        started, seen = time.monotonic(), -1
        while not self.stop.is_set():
            try:
                if self.args.demo:
                    bgr, depth, finger, target = scene(time.monotonic() - started)
                    captured, frame_sequence = time.monotonic(), seen + 1
                    synthetic_tracks = {"finger": TrackResult((270, 226, 20, 28), finger, True, 1, "synthetic"),
                                        "target": TrackResult((365, 220, 30, 40), target, True, 1, "synthetic")}
                elif self.args.image:
                    if seen >= 0:
                        with self._lock:
                            needs_update = bool(self.pending_teach or self._generation != self._generation_applied)
                        if not needs_update:
                            self.stop.wait(.05)
                            continue
                    bgr = cv2.imread(self.args.image)
                    if bgr is None:
                        raise ValueError("Could not read --image")
                    bgr = self.calibration.prepare(bgr)
                    captured, frame_sequence = time.monotonic(), seen + 1
                    depth, synthetic_tracks = None, None
                else:
                    packet = self.camera.read()
                    if packet is None or packet.sequence == seen:
                        with self._lock:
                            expired = self.frame_time is not None and (time.monotonic() - self.frame_time) * 1000 + (self.config.source_delay_bound_ms or 0) >= self.config.valid_for_ms
                            if expired and not self._source_was_stale:
                                self._generation += 1
                                self._source_was_stale = True
                        self.stop.wait(.01)
                        continue
                    self._source_was_stale = False
                    bgr = self.calibration.prepare(packet.image)
                    captured, frame_sequence = packet.read_time, packet.sequence
                    depth, synthetic_tracks = None, None
                with self._lock:
                    if self._generation_applied != self._generation:
                        self.engine.reset()
                        self._generation_applied = self._generation
                    while self.pending_teach:
                        kind, taught_frame, bbox = self.pending_teach.pop(0)
                        self.tracks[kind] = self.trackers[kind].initialize(taught_frame, bbox)
                        self.engine.reset()
                    if synthetic_tracks:
                        self.tracks = synthetic_tracks
                    else:
                        self.tracks = {kind: tracker.update(bgr) for kind, tracker in self.trackers.items()}
                    tracks = dict(self.tracks)
                    generation = self._generation
                    armed = self.armed
                inference_start = time.monotonic()
                if depth is None:
                    depth = self.backend.infer(bgr, self.matrix)
                inference_ms = (time.monotonic() - inference_start) * 1000
                masks = {key: value.mask if value.mask is not None else np.zeros(bgr.shape[:2], np.uint8) for key, value in tracks.items()}
                age = (time.monotonic() - captured) * 1000 + (self.config.source_delay_bound_ms or 0)
                measurement = self.engine.evaluate(depth.depth_m, masks["finger"], masks["target"], self.matrix,
                                                   metric=depth.metric, finger_valid=tracks["finger"].valid,
                                                   target_valid=tracks["target"].valid,
                                                   tracking_quality=min(value.quality for value in tracks.values()),
                                                   capture_age_ms=age, now=captured, armed=armed, mode=self.mode)
                image_view = draw_view(bgr, depth, tracks, measurement, self.mode, self.config.min_depth_mm, self.config.max_depth_mm)
                if image_view.shape[1] > 1600:
                    image_view = cv2.resize(image_view, (1600, int(image_view.shape[0] * 1600 / image_view.shape[1])))
                jpeg = encode_jpeg(image_view)
                with self._lock:
                    # Ignore results computed before a reset/arm/teach request.
                    if generation != self._generation:
                        seen = frame_sequence
                        continue
                    self.image, self.depth, self.combined_jpeg = bgr.copy(), depth, jpeg
                    self.measurement, self.frame_time, self.frame_sequence = measurement, captured, frame_sequence
                    self.inference_ms, self.last_error = inference_ms, None
                seen = frame_sequence
                if self.args.demo:
                    self.stop.wait(.04)
            except Exception as exc:
                with self._lock:
                    self.last_error = str(exc)
                    self.measurement = Measurement(block_reasons=("unknown_depth", "invalid_geometry"))
                    self.armed = False
                    self.engine.reset()
                print(f"Vision blocked: {exc}", flush=True)
                if self.args.image:
                    seen = 0
                self.stop.wait(1)

    def command(self, action, data):
        with self._lock:
            if action == "freeze":
                if self.image is None:
                    raise ValueError("No frame available yet")
                self.frozen = self.image.copy()
                return {"ok": True, "image_size": [self.frozen.shape[1], self.frozen.shape[0]]}
            if action == "teach":
                kind, bbox = data.get("kind"), data.get("bbox")
                if kind not in self.trackers or self.frozen is None or not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError("Freeze a frame and select a fingertip/target box first")
                if any(isinstance(value, bool) or not isinstance(value, int) for value in bbox):
                    raise ValueError("bbox must contain four integers")
                self.pending_teach.append((kind, self.frozen.copy(), bbox))
                self.armed = False
            elif action == "arm":
                if not isinstance(data.get("armed"), bool):
                    raise ValueError("armed must be a boolean")
                self.armed = data["armed"]
            elif action == "reset":
                self.trackers = {key: RegionTracker() for key in self.trackers}
                self.tracks = {key: TrackResult(None, None, False, 0, "not_initialized") for key in self.trackers}
                self.pending_teach.clear()
                self.armed = False
            else:
                raise ValueError("Unknown action")
            self._generation += 1
            self.measurement.strike_permit = False
            self.measurement.candidate_permit = False
            self.measurement.block_reasons = tuple(dict.fromkeys(self.measurement.block_reasons + ("controller_unarmed",)))
            return {"ok": True}

    def close(self):
        with self._lock:
            self.armed = False
            self.measurement.strike_permit = False
            self.measurement.candidate_permit = False
            self._generation += 1
        self.stop.set()
        if self.camera:
            self.camera.close()
        for thread in (self._publisher, self._processing):
            if thread:
                thread.join(timeout=2)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_data(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    return self.send_data((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path == "/state":
                    return self.send_data(json.dumps(app.state(), allow_nan=False).encode(), "application/json")
                if path == "/present":
                    return self.send_data(str(int(app.packet().strike_permit)).encode(), "text/plain")
                if path == "/telemetry":
                    return self.send_data(encode_json(app.packet()).encode(), "application/json")
                if path == "/telemetry.bin":
                    return self.send_data(encode_can_fd(app.packet()), "application/octet-stream")
                if path in ("/snapshot.jpg", "/frozen.jpg"):
                    with app._lock:
                        image = app.frozen if path == "/frozen.jpg" else app.image
                        image = None if image is None else image.copy()
                    if image is None:
                        return self.send_data(b"No frame available", "text/plain", 503)
                    return self.send_data(encode_jpeg(image), "image/jpeg")
                if path == "/depth.npz":
                    with app._lock:
                        result = app.depth
                        frame_sequence = app.frame_sequence
                        frame_time = app.frame_time
                    if result is None:
                        return self.send_data(b"No depth available", "text/plain", 503)
                    output = io.BytesIO()
                    np.savez_compressed(output, depth_m=result.depth_m, visualization_depth=result.visualization_depth,
                                        valid_mask=result.valid_mask, metric=np.array(result.metric),
                                        frame_sequence=np.array(frame_sequence), mode=np.array(app.mode.name),
                                        K=np.empty((0, 0)) if app.matrix is None else app.matrix,
                                        read_time_monotonic=np.array(frame_time), timestamp_kind=np.array(app.timestamp_kind))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Disposition", 'attachment; filename="depth.npz"')
                    self.send_header("Content-Length", str(len(output.getvalue())))
                    self.end_headers()
                    self.wfile.write(output.getvalue())
                    return
                if path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last_sequence = -1
                    last_sent = 0.0
                    while not app.stop.is_set():
                        with app._lock:
                            jpeg, sequence = app.combined_jpeg, app.frame_sequence
                        # Browsers may wait for the next multipart boundary
                        # before painting a still image. Repeat the DISPLAY
                        # JPEG periodically; never refresh its capture time.
                        if jpeg is not None and (sequence != last_sequence or time.monotonic() - last_sent >= .5):
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                            last_sequence = sequence
                            last_sent = time.monotonic()
                        app.stop.wait(.03)
                    return
                return self.send_data(b"Not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    raise ValueError("Request body too large")
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                result = app.command(urlparse(self.path).path.strip("/"), data)
                self.send_data(json.dumps(result).encode(), "application/json")
            except (ValueError, TypeError) as exc:
                self.send_data(json.dumps({"error": str(exc)}).encode(), "application/json", 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="webcam index, raw MJPEG URL or replay video")
    parser.add_argument("--image", help="view a still image (always SIMULATION)")
    parser.add_argument("--demo", action="store_true", help="synthetic UI/geometry demo, no model")
    parser.add_argument("--camera", help="intrinsic calibration JSON; missing means relative visualization only")
    parser.add_argument("--config", default=str(ROOT / "config.example.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--process-res", type=int, default=364)
    parser.add_argument("--offline", action="store_true", help="require cached model weights")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    args = parser.parse_args()
    if args.demo and args.image:
        parser.error("--demo and --image are mutually exclusive")
    app = Application(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    app.start()
    print(f"Dashboard: http://{args.host}:{args.port} | {app.mode.name} | no actuator transmission", flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()


if __name__ == "__main__":
    main()
