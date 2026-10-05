"""Offline, single-camera NEON depth feed and vision telemetry. Python 3.8."""
from __future__ import annotations
import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from capture import NeonCapture
from depth_runtime import DepthRuntime
from mapped_bus import MappedPublisher
from network_bus import UDPPublisher
from protocol import VisionTelemetry, Direction, Mode
from proximity import ProximityEngine, StrikeConfig, Measurement, validate_intrinsics
from vision_tracking import RegionTracker

ROOT = Path(__file__).resolve().parent


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    allowed = set(json.loads((ROOT / 'config.json').read_text(encoding='utf-8')))
    if set(config) != allowed:
        raise ValueError('Config must contain the documented fields: %s' % sorted(allowed))
    for key in ('model', 'manifest', 'camera_calibration', 'teaching', 'strike_config'):
        value = config[key]
        if not isinstance(value, str) or not value:
            raise ValueError('%s must be a nonempty path' % key)
        config[key] = str((path.parent / value).resolve())
    for key, low, high in [('threads', 1, 16), ('port', 1, 65535),
                           ('bus_rate_hz', 1, 100), ('display_max_age_ms', 100, 60000)]:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or not low <= config[key] <= high:
            raise ValueError('Invalid %s' % key)
    if not isinstance(config['controller_armed'], bool):
        raise ValueError('controller_armed must be boolean')
    if config['backend'] not in ('cpu', 'trt', 'auto'):
        raise ValueError('backend must be cpu, trt or auto')
    if not isinstance(config['udp'], dict) or set(config['udp']) != {'enabled', 'host', 'port', 'source_port'}:
        raise ValueError('udp must contain enabled, host, port, source_port')
    if not isinstance(config['udp']['enabled'], bool):
        raise ValueError('udp.enabled must be boolean')
    return config


def packet_for(measurement, session, sequence, captured_ns, config, mode, extra=()):
    delay = config.source_delay_bound_ms or 0.0
    now_ns = time.monotonic_ns()
    invalid_time = (isinstance(captured_ns, bool) or not isinstance(captured_ns, int)
                    or captured_ns <= 0 or captured_ns > now_ns)
    age = 65535 if invalid_time else min(65535, int(math.ceil((now_ns - captured_ns) / 1e6 + delay)))
    reasons = list(measurement.block_reasons) + list(extra)
    if invalid_time:
        reasons.append('unknown_depth')
    if age >= config.valid_for_ms:
        reasons.append('stale_frame')
    if mode == Mode.SIMULATION:
        reasons.append('simulation')
    reasons = tuple(dict.fromkeys(reasons))
    return VisionTelemetry(session_id=session, sequence=sequence & 0xffffffff,
        capture_age_ms=age, valid_for_ms=config.valid_for_ms, mode=mode,
        direction=measurement.direction, strike_permit=bool(measurement.strike_permit and not reasons),
        gap_mm=measurement.gap_mm, quality=measurement.quality, scale_valid=measurement.scale_valid,
        block_reasons=reasons, timestamp_ms=time.monotonic_ns() // 1000000)


def load_camera(path):
    if not Path(path).exists():
        return None, None
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('image_size') != [540, 960]:
        raise ValueError('Calibration must describe the upright 540x960 feed')
    k = validate_intrinsics(data['K'])
    distortion = np.asarray(data['distortion'], dtype=np.float64)
    if distortion.ndim != 1 or len(distortion) not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError('Invalid camera distortion')
    maps = cv2.initUndistortRectifyMap(k, distortion, None, k, (540, 960), cv2.CV_32FC1)
    return k, maps


class Application:
    def __init__(self, config, image=None, bus_directory=None, runtime=None, source=None):
        self.config = config
        self.strike = StrikeConfig.load(config['strike_config'])
        self.engine = ProximityEngine(self.strike)
        self.runtime = runtime or DepthRuntime(config['model'], config['manifest'],
                                               backend=config['backend'], threads=config['threads'])
        self.source = source or NeonCapture(config['device'], image=image)
        self.mode = Mode.SIMULATION if self.source.simulation else Mode.LIVE
        self.k, self.maps = load_camera(config['camera_calibration'])
        self.observed_pixels = None if self.maps is None else (
            (self.maps[0] >= 0) & (self.maps[0] <= 539) &
            (self.maps[1] >= 0) & (self.maps[1] <= 959))
        self.trackers = [RegionTracker(), RegionTracker()]
        self.taught = False
        self.teaching_path = Path(config['teaching'])
        self.publisher = MappedPublisher(config['shared_memory_name'], directory=bus_directory)
        self.udp = None
        try:
            if config['udp']['enabled']:
                udp = config['udp']
                self.udp = UDPPublisher(udp['host'], udp['port'], source_port=udp['source_port'])
        except Exception:
            self.publisher.close()
            raise
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.measurement = Measurement(block_reasons=('unknown_depth', 'controller_unarmed'))
        self.captured_ns = None
        self.depth = self.rgb = None
        self.error = None
        self.bus_error = None
        self.inference_ms = None
        self.frames_processed = 0
        self.last_packet = None
        self.threads = []
        self._closed = False

    def _teach(self):
        self.taught = True
        if not self.teaching_path.exists():
            return
        teaching = json.loads(self.teaching_path.read_text(encoding='utf-8'))
        if teaching.get('image_size') != [540, 960] or self.k is None:
            raise ValueError('Teaching requires the matching calibrated 540x960 image')
        reference = cv2.imread(str(self.teaching_path.parent / teaching['reference']))
        import hashlib
        if teaching.get('camera_sha256') != hashlib.sha256(Path(self.config['camera_calibration']).read_bytes()).hexdigest():
            raise ValueError('Teaching was made with a different camera calibration; run calibrate.sh again')
        if reference is None or reference.shape[:2] != (960, 540):
            raise ValueError('Teaching reference missing or wrong resolution')
        for tracker, name in zip(self.trackers, ('finger', 'target')):
            result = tracker.initialize(reference, teaching[name])
            if not result.valid:
                raise ValueError('%s teaching failed: %s' % (name, result.reason))

    def start(self):
        try:
            self.source.start()
            self._teach()
            for name, target in [('depth-inference', self._infer), ('vision-bus', self._bus)]:
                thread = threading.Thread(target=target, name=name, daemon=True)
                self.threads.append(thread)
                thread.start()
            return self
        except Exception:
            self.close()
            raise

    def _infer(self):
        last_sequence = -1
        try:
            while not self.stop.is_set():
                frame = self.source.read(last_sequence)
                if frame is None:
                    self.stop.wait(.01)
                    continue
                last_sequence = frame.sequence
                if (isinstance(frame.captured_ns, bool) or not isinstance(frame.captured_ns, int)
                        or not 0 < frame.captured_ns <= time.monotonic_ns()):
                    raise RuntimeError('Invalid source monotonic timestamp')
                rgb = frame.image
                if self.maps is not None:
                    rgb = cv2.remap(rgb, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
                started = time.monotonic_ns()
                result = self.runtime.infer(rgb)
                if self.stop.is_set():
                    return
                if not result.metric:
                    raise RuntimeError('This application requires metric depth output; relative model refused')
                depth = result.depth_m.copy()
                if self.observed_pixels is not None:
                    # Undistortion black borders are outside the sensor image.
                    # Learned depth there does not create observed clearance.
                    depth[~self.observed_pixels] = np.nan
                elapsed = (time.monotonic_ns() - started) / 1e6
                tracks = [tracker.update(rgb) for tracker in self.trackers]
                masks = [track.mask.astype(bool) if track.valid else np.zeros(rgb.shape[:2], bool) for track in tracks]
                measurement = self.engine.evaluate(depth, masks[0], masks[1], self.k,
                    metric=result.metric, finger_valid=tracks[0].valid, target_valid=tracks[1].valid,
                    tracking_quality=min(track.quality for track in tracks),
                    capture_age_ms=(time.monotonic_ns() - frame.captured_ns) / 1e6 + (self.strike.source_delay_bound_ms or 0.0),
                    now=frame.captured_ns / 1e9, armed=self.config['controller_armed'], mode=self.mode)
                for name, track in zip(('finger', 'target'), tracks):
                    if track.valid:
                        x, y, w, h = track.bbox
                        cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(rgb, name, (x, max(20, y-6)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 1)
                with self.lock:
                    if self.stop.is_set():
                        return
                    self.measurement = measurement
                    self.rgb = rgb.copy()
                    self.depth = depth
                    self.captured_ns = frame.captured_ns
                    self.inference_ms = elapsed
                    self.frames_processed += 1
        except Exception as exc:
            self.engine.reset()
            with self.lock:
                self.error = '%s: %s' % (type(exc).__name__, exc)
                self.measurement = Measurement(block_reasons=('unknown_depth', 'stale_frame'))
                self.depth = self.rgb = None
                self.captured_ns = None

    def packet(self, sequence):
        source_status = self.source.status()
        with self.lock:
            extra = ('stale_frame',) if self.stop.is_set() or self._closed or self.error or source_status['error'] else ()
            return packet_for(self.measurement, self.publisher.session_id, sequence,
                              self.captured_ns, self.strike, self.mode, extra)

    def _bus(self):
        sequence = 0
        try:
            while not self.stop.is_set():
                sequence += 1
                packet = self.packet(sequence)
                self.publisher.publish(packet)
                if self.udp is not None:
                    try:
                        # mmap locking/publication may have consumed time. Age
                        # the network packet at send time, including new faults.
                        packet = self.packet(sequence)
                        self.udp.publish(packet)
                        with self.lock:
                            self.bus_error = None
                    except OSError as exc:
                        with self.lock:
                            self.bus_error = str(exc)
                with self.lock:
                    self.last_packet = packet
                self.stop.wait(1 / self.config['bus_rate_hz'])
        except Exception as exc:
            with self.lock:
                self.bus_error = '%s: %s' % (type(exc).__name__, exc)
            self.stop.set()

    def state(self):
        packet = self.packet(self.last_packet.sequence if self.last_packet else 0)
        with self.lock:
            return {'model': self.runtime.manifest['model_id'], 'backend': self.runtime.backend,
                    'depth_units': 'metres', 'depth_is_estimate': True,
                    'inference_ms': self.inference_ms, 'frames_processed': self.frames_processed,
                    'camera_calibrated': self.k is not None, 'error': self.error,
                    'bus_error': self.bus_error, 'telemetry': packet.to_dict(),
                    'source': self.source.status(), 'gap_is_estimate': True}

    def image(self):
        source_error = self.source.status()['error']
        with self.lock:
            rgb = None if self.rgb is None else self.rgb.copy()
            depth = None if self.depth is None else self.depth.copy()
            age = None if self.captured_ns is None else (time.monotonic_ns() - self.captured_ns) / 1e6
            error = self.error or source_error
            inference_ms = self.inference_ms
            gap = self.measurement.gap_mm
        if rgb is None or depth is None or error or age > self.config['display_max_age_ms']:
            view = np.zeros((960, 1080, 3), np.uint8)
            label = error or ('Starting local depth model...' if age is None else 'Feed stale; waiting for new inference')
        else:
            valid = np.isfinite(depth) & (depth > 0)
            # Fixed 0-2m colour range; absolute colours remain comparable between frames.
            normalized = np.where(valid, np.clip(depth / 2.0, 0, 1), 0)
            color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            color[~valid] = 0
            view = np.hstack([rgb, color])
            label = 'Estimated depth 0-2m | age %.0fms | inference %.0fms' % (age, inference_ms)
            if age >= self.strike.valid_for_ms:
                label = 'DELAYED | ' + label
            if gap is not None:
                label += ' | est. gap %.1fmm' % gap
        cv2.rectangle(view, (0, 0), (view.shape[1], 62), (0, 0, 0), -1)
        if self.mode == Mode.SIMULATION:
            label = 'RECORDED IMAGE SIMULATION | ' + label
        cv2.putText(view, label[:115], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
        cv2.putText(view, 'Single camera | RGB / depth | estimate, not verified clearance',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (220,220,220), 1)
        return view

    def depth_bytes(self):
        with self.lock:
            if self.depth is None or self.error or self.source.status()['error']:
                return None
            depth, captured_ns = self.depth.copy(), self.captured_ns
        age = (time.monotonic_ns() - captured_ns) / 1e6 + (self.strike.source_delay_bound_ms or 0.0)
        if age >= self.config['display_max_age_ms']:
            return None
        buffer = io.BytesIO()
        # Includes capture age at serialization; consumers account for transport/receive time.
        np.savez_compressed(buffer, depth_m=depth, captured_monotonic_ns=np.int64(captured_ns),
                            capture_age_ms=np.float64(age),
                            stale=np.bool_(age >= self.strike.valid_for_ms),
                            estimated=np.bool_(True), metric=np.bool_(True))
        return buffer.getvalue()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.stop.set()
        with self.lock:
            self.measurement = Measurement(block_reasons=('unknown_depth', 'stale_frame'))
            self.depth = self.rgb = None
            self.captured_ns = None
        for thread in self.threads:
            if thread.name == 'vision-bus':
                thread.join(timeout=1)
        self.publisher.close()
        for thread in self.threads:
            if thread.name != 'vision-bus':
                thread.join(timeout=5)
        self.source.close()
        if self.udp is not None:
            self.udp.close()
        if not any(thread.is_alive() for thread in self.threads) and hasattr(self.runtime, 'close'):
            self.runtime.close()


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, kind, body):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == '/':
                    self.send(200, 'text/html; charset=utf-8', b'<!doctype html><html><head><title>NEON depth feed</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body{margin:0;background:#000;height:100%;overflow:hidden}img{display:block;width:100%;height:100%;object-fit:contain}</style></head><body><img src="/stream" alt="NEON RGB and estimated depth"></body></html>')
                elif path in ('/state', '/telemetry'):
                    state = app.state()
                    self.send(200, 'application/json', json.dumps(state if path == '/state' else state['telemetry'], allow_nan=False).encode())
                elif path == '/depth.npz':
                    body = app.depth_bytes()
                    self.send(503 if body is None else 200, 'application/octet-stream', body or b'Depth not ready')
                elif path == '/snapshot.jpg':
                    okay, jpeg = cv2.imencode('.jpg', app.image(), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.send(200, 'image/jpeg', jpeg.tobytes())
                elif path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    while not app.stop.is_set():
                        okay, jpeg = cv2.imencode('.jpg', app.image(), [cv2.IMWRITE_JPEG_QUALITY, 80])
                        body = jpeg.tobytes()
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body + b'\r\n')
                        self.wfile.flush()
                        app.stop.wait(.2)
                else:
                    self.send(404, 'text/plain', b'Not found')
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--image', help='540x960 recorded image for SIMULATION testing')
    parser.add_argument('--bus-directory', help='Local mmap directory; default /dev/shm on Linux')
    parser.add_argument('--self-test', action='store_true', help='Verify model inference with a blank input; no camera or bus')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        runtime = DepthRuntime(config['model'], config['manifest'], backend=config['backend'], threads=config['threads'])
        try:
            result = runtime.infer(np.zeros((960,540,3), np.uint8))
            print(json.dumps({'backend': runtime.backend, 'shape': list(result.depth_m.shape),
                              'finite_fraction': float(np.isfinite(result.depth_m).mean()),
                              'depth_units': 'metres', 'model_sha256_verified': True}))
        finally:
            if hasattr(runtime, 'close'):
                runtime.close()
        return 0
    app = Application(config, image=args.image, bus_directory=args.bus_directory)
    server = None
    try:
        server = ThreadingHTTPServer((config['bind'], config['port']), handler_for(app))
        server.daemon_threads = True
        server.timeout = .25
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: app.stop.set())
        app.start()
        print('Feed: http://<NEON-IP>:%d/ | local model | bus session %d' % (config['port'], app.publisher.session_id), flush=True)
        while not app.stop.is_set():
            server.handle_request()
            if app.error:
                print(app.error, flush=True)
                return 1
        return 1 if app.bus_error else 0
    finally:
        if server is not None:
            server.server_close()
        app.close()


if __name__ == '__main__':
    raise SystemExit(main())
