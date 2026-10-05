"""Standalone overlay-free NEON V4L2 MJPEG helper (Python 3.6+ compatible).

The known sensor must be opened at 1920x1080 BEFORE the first read. Stop other
processes using /dev/video0 first. No Jetson SDK or pip OpenCV is required.
Capture/exposure synchronization is NOT provided by this helper.
"""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
try:
    from http.server import ThreadingHTTPServer
except ImportError:  # JetPack 4.x can use Python 3.6.
    from socketserver import ThreadingMixIn
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
import json
import platform
import signal
import threading
import time
import uuid

import cv2

WIDTH = 1920
HEIGHT = 1080
PORT = 8081
JPEG_QUALITY = 85
STALE_SECONDS = 1.0
OUTPUT_WIDTH = 540


def monotonic_ns():
    clock = getattr(time, "monotonic_ns", None)
    return clock() if clock is not None else int(time.monotonic() * 1000000000)


class CaptureState:
    def __init__(self, rotate90cw, quality, output_width=OUTPUT_WIDTH):
        self.rotate90cw = rotate90cw
        self.quality = quality
        self.output_width = output_width
        self.camera_session = uuid.uuid4().hex
        self.capture_monotonic_ns = None
        self.capture_shape = None
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.jpeg = None
        self.sequence = 0
        self.read_time = None
        self.shape = None
        self.error = "waiting_for_camera"
        self.times = deque(maxlen=60)

    def capture(self):
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            if not cap.isOpened():
                raise RuntimeError("Cannot open /dev/video0; stop other camera processes")
            while not self.stop.is_set():
                ok, frame = cap.read()
                completed_ns = monotonic_ns()
                completed = completed_ns / 1000000000.0
                if not ok or frame is None:
                    raise RuntimeError("V4L2 returned no frame")
                capture_shape = frame.shape
                if self.rotate90cw:
                    if hasattr(cv2, "rotate"):
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    else:
                        import numpy as np
                        frame = np.ascontiguousarray(np.rot90(frame, 3))
                if self.output_width and frame.shape[1] != self.output_width:
                    height = max(1, int(round(frame.shape[0] * self.output_width / float(frame.shape[1]))))
                    interpolation = cv2.INTER_AREA if self.output_width < frame.shape[1] else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (self.output_width, height), interpolation=interpolation)
                encoded, jpeg = cv2.imencode(".jpg", frame,
                                            [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not encoded:
                    raise RuntimeError("JPEG encoding failed")
                payload = jpeg.tobytes()
                with self.condition:
                    self.jpeg = payload
                    self.sequence += 1
                    self.read_time = completed
                    self.capture_monotonic_ns = completed_ns
                    self.capture_shape = capture_shape
                    self.shape = frame.shape
                    self.error = None
                    self.times.append(completed)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.jpeg = None
                self.condition.notify_all()
        finally:
            if cap is not None:
                cap.release()

    def fresh_locked(self):
        return (self.jpeg is not None and self.read_time is not None
                and time.monotonic() - self.read_time <= STALE_SECONDS)

    def metadata(self):
        with self.condition:
            elapsed = self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0
            fps = (len(self.times) - 1) / elapsed if elapsed > 0 else 0.0
            age = None if self.read_time is None else time.monotonic() - self.read_time
            return {"sequence": self.sequence, "read_time_monotonic": self.read_time,
                    "capture_monotonic_ns": self.capture_monotonic_ns,
                    "camera_session": self.camera_session,
                    "timestamp_kind": "neon_host_read_completion",
                    "sensor_synchronized": False, "fresh": self.fresh_locked(),
                    "frame_age_seconds": age, "width": None if self.shape is None else self.shape[1],
                    "height": None if self.shape is None else self.shape[0],
                    "rotated_90_clockwise": self.rotate90cw, "fps": fps,
                    "capture_mode_requested": [WIDTH, HEIGHT],
                    "capture_image_size": None if self.capture_shape is None else [self.capture_shape[1], self.capture_shape[0]],
                    "output_width_requested": self.output_width,
                    "python_version": platform.python_version(), "opencv_version": cv2.__version__,
                    "overlays": False, "error": self.error}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_bytes(self, status, mime, payload):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.connection.settimeout(3.0)
            path = self.path.split("?", 1)[0]
            try:
                if path == "/state":
                    self.send_bytes(200, "application/json", json.dumps(state.metadata()).encode("utf-8"))
                elif path == "/snapshot.jpg":
                    with state.condition:
                        jpeg = state.jpeg if state.fresh_locked() else None
                    if jpeg is None:
                        self.send_bytes(503, "text/plain", b"No fresh frame\n")
                    else:
                        self.send_bytes(200, "image/jpeg", jpeg)
                elif path in ("/raw", "/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last_sequence = -1
                    while not state.stop.is_set():
                        with state.condition:
                            state.condition.wait_for(
                                lambda: state.stop.is_set() or state.sequence > max(0, last_sequence)
                                or (state.error is not None and state.error != "waiting_for_camera"),
                                timeout=1.0)
                            if state.stop.is_set():
                                break
                            if not state.fresh_locked():
                                if state.error == "waiting_for_camera":
                                    continue
                                break
                            if state.sequence == last_sequence:
                                continue
                            jpeg, sequence = state.jpeg, state.sequence
                            capture_ns = state.capture_monotonic_ns
                            if capture_ns is None:
                                capture_ns = int(state.read_time * 1000000000)
                            camera_session = state.camera_session
                        header = ("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n"
                                  "X-Frame-Sequence: %d\r\nX-Capture-Monotonic-Ns: %d\r\n"
                                  "X-Camera-Session: %s\r\n\r\n" % (len(jpeg), sequence, capture_ns, camera_session)).encode("ascii")
                        self.wfile.write(header + jpeg + b"\r\n")
                        self.wfile.flush()
                        last_sequence = sequence
                else:
                    self.send_bytes(404, "text/plain", b"Use /raw, /snapshot.jpg or /state\n")
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                return
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--no-rotate", action="store_true", help="Keep native camera orientation")
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--output-width", type=int, default=OUTPUT_WIDTH,
                        help="Encoded output width after rotation; 0 keeps native size")
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.output_width != 0 and not 14 <= args.output_width <= 4096:
        parser.error("--output-width must be 0 or between 14 and 4096")
    state = CaptureState(not args.no_rotate, args.jpeg_quality, args.output_width)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(state))
    server.daemon_threads = True
    capture_thread = threading.Thread(target=state.capture, name="neon-capture", daemon=True)
    http_thread = threading.Thread(target=server.serve_forever, name="neon-http", daemon=True)
    capture_thread.start()
    http_thread.start()
    signal.signal(signal.SIGTERM, lambda signum, frame: state.stop.set())
    print("Raw camera: http://<NEON-IP>:%d/raw (no sensor synchronization)" % args.port, flush=True)
    exit_code = 0
    try:
        while not state.stop.wait(0.5):
            if not capture_thread.is_alive():
                print("Capture stopped: %s" % state.metadata()["error"], flush=True)
                exit_code = 1
                break
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        with state.condition:
            state.condition.notify_all()
        server.shutdown()
        server.server_close()
        capture_thread.join(timeout=2.0)
        http_thread.join(timeout=2.0)
        if capture_thread.is_alive():
            print("V4L2 read is still blocked; process exit ends the daemon worker", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
