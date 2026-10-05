#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera capture prototype for the ADLINK NEON-201A-JNX (AR0234, 1920x1200, V4L2 / GStreamer).

What it does
  1. Reports the environment: Python, OpenCV (and whether it was built with GStreamer), L4T release,
     video devices, available GStreamer elements.
  2. Asks the camera which formats / sizes / frame rates it offers (v4l2-ctl --list-formats-ext).
  3. Builds candidate GStreamer pipelines for those formats, tests each one with gst-launch-1.0 first
     (so a bad pipeline can't hang Python), and opens the first that works with cv2.VideoCapture.
     Falls back to OpenCV's own V4L2 backend.
  4. Captures frames, measures FPS, converts BGR -> RGB and proves the conversion is right.
  5. Saves snapshots + a report you can bring back to the laptop.
  If the camera only offers raw Bayer data, it grabs one raw frame and saves debayered previews.

Run on the NEON with Python 3 (plain `python` is Python 2.7 on JetPack 4.x):
    python3 neon_camera_test.py                     # probe + capture 150 frames from /dev/video0
    python3 neon_camera_test.py --probe-only        # just report formats and pipeline tests
    python3 neon_camera_test.py --try-all           # test every candidate pipeline, not just the first
    python3 neon_camera_test.py --display           # live window with FPS (needs a monitor on the NEON)
    python3 neon_camera_test.py --pipeline "v4l2src device=/dev/video0 ! ... ! appsink"
On the laptop (webcam, no GStreamer):  python neon_camera_test.py --device 0

Needs only OpenCV + NumPy. On the NEON use JetPack's OpenCV - never `pip install opencv-python` there.
Written to run on Python 3.6 (JetPack 4.5) and 3.8 (JetPack 5.1.2).
"""
from __future__ import print_function

import sys

if sys.version_info[0] < 3:
    sys.exit("Run this with Python 3:  python3 neon_camera_test.py")

import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
import time

import cv2
import numpy as np

# V4L2 pixel formats (as printed by v4l2-ctl) -> GStreamer video/x-raw formats
FOURCC_TO_GST = {
    "YUYV": "YUY2", "UYVY": "UYVY", "YVYU": "YVYU", "VYUY": "VYUY", "NV12": "NV12", "NV16": "NV16",
    "YU12": "I420", "YV12": "YV12", "GREY": "GRAY8", "Y16 ": "GRAY16_LE", "RGB3": "RGB", "BGR3": "BGR",
    "XR24": "BGRx", "AR24": "BGRA",
}
BAYER8_TO_GST = {"BA81": "bggr", "GBRG": "gbrg", "GRBG": "grbg", "RGGB": "rggb"}   # 8-bit Bayer
BAYER_HIGH = {  # 10/12/16-bit Bayer: GStreamer's stock elements can't debayer these
    "RG10": "RGGB", "BG10": "BGGR", "GB10": "GBRG", "BA10": "GRBG",
    "RG12": "RGGB", "BG12": "BGGR", "GB12": "GBRG", "BA12": "GRBG",
    "RG16": "RGGB", "BYR2": "BGGR", "GB16": "GBRG", "GR16": "GRBG",
}
NVVIDCONV_INPUTS = ("YUY2", "UYVY", "NV12", "I420", "GRAY8")   # formats Jetson's hardware converter accepts
APPSINK = "appsink drop=true max-buffers=1 sync=false"          # always hand OpenCV the newest frame

LOG = []


def log(msg=""):
    print(msg)
    sys.stdout.flush()
    LOG.append(msg)


def run(cmd, timeout=15):
    """Run a command; return (returncode, combined output). returncode None = timed out / not found."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        out = e.output if isinstance(e.output, str) else ""
        return None, (out or "") + "\n[timed out after {} s]".format(timeout)
    except OSError as e:
        return None, str(e)


# ------------------------------------------------------------------------------------------------
# 1. Environment
# ------------------------------------------------------------------------------------------------
def opencv_has_gstreamer():
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GStreamer"):
            return "YES" in line, line.strip()
    return False, "GStreamer: (not listed)"


def report_environment(device):
    log("=" * 72)
    log("Environment")
    log("=" * 72)
    log("Python   : {} ({})".format(platform.python_version(), sys.executable))
    gst_ok, gst_line = opencv_has_gstreamer()
    log("OpenCV   : {} | {}".format(cv2.__version__, gst_line))
    log("NumPy    : {}".format(np.__version__))
    if os.path.exists("/etc/nv_tegra_release"):
        with open("/etc/nv_tegra_release") as f:
            log("L4T      : " + f.readline().strip())
    if os.path.exists("/proc/device-tree/model"):
        with open("/proc/device-tree/model", "rb") as f:
            log("Module   : " + f.read().decode("ascii", "replace").strip("\x00 \n"))
    devs = sorted(d for d in os.listdir("/dev") if d.startswith("video")) if os.path.isdir("/dev") else []
    log("Devices  : " + (", ".join("/dev/" + d for d in devs) or "none"))
    if shutil.which("gst-inspect-1.0"):
        have = [e for e in ("v4l2src", "videoconvert", "appsink", "nvvidconv", "nvarguscamerasrc", "bayer2rgb", "jpegdec")
                if run(["gst-inspect-1.0", e], timeout=20)[0] == 0]
        log("GStreamer: elements available: " + ", ".join(have))
    else:
        have = []
        log("GStreamer: gst-inspect-1.0 not found (install gstreamer1.0-tools to test pipelines)")
    if shutil.which("systemctl"):
        rc, _ = run(["systemctl", "is-active", "--quiet", "nvargus-daemon"])
        log("Argus    : nvargus-daemon is " + ("active" if rc == 0 else "inactive"))
    log("Target   : " + device)
    return gst_ok, set(have)


# ------------------------------------------------------------------------------------------------
# 2. What the camera offers
# ------------------------------------------------------------------------------------------------
def parse_v4l2_formats(text):
    """Parse `v4l2-ctl --list-formats-ext` (old and new layouts) into a list of dicts."""
    formats, cur, size = [], None, None
    for line in text.splitlines():
        m = re.search(r"Pixel Format\s*:\s*'(.{4})'", line) or re.match(r"\s*\[\d+\]:\s*'(.{4})'", line)
        if m:
            cur = {"fourcc": m.group(1), "sizes": {}}
            formats.append(cur)
            continue
        m = re.search(r"Size:\s*Discrete\s+(\d+)x(\d+)", line)
        if m and cur is not None:
            size = (int(m.group(1)), int(m.group(2)))
            cur["sizes"].setdefault(size, [])
            continue
        m = re.search(r"Size:\s*Stepwise\s+\d+x\d+\s*-\s*(\d+)x(\d+)", line)
        if m and cur is not None:
            size = (int(m.group(1)), int(m.group(2)))
            cur["sizes"].setdefault(size, [])
            continue
        m = re.search(r"\(([\d.]+)\s*fps\)", line)
        if m and cur is not None and size is not None and size in cur["sizes"]:
            cur["sizes"][size].append(float(m.group(1)))
    return formats


def pick_mode(fmt, want_w, want_h, want_fps):
    """Choose (w, h, fps) for a format: the requested size if offered, else the largest."""
    sizes = fmt["sizes"]
    if not sizes:
        return want_w or 1920, want_h or 1200, want_fps or 30
    if want_w and want_h and (want_w, want_h) in sizes:
        w, h = want_w, want_h
    else:
        w, h = max(sizes, key=lambda s: s[0] * s[1])
    rates = sizes[(w, h)]
    fps = want_fps or (int(round(max(rates))) if rates else 30)
    return w, h, fps


def probe_formats(device, out_dir):
    log("")
    log("=" * 72)
    log("Camera formats (v4l2-ctl --list-formats-ext)")
    log("=" * 72)
    if not shutil.which("v4l2-ctl"):
        log("v4l2-ctl not found (package v4l-utils) - will rely on automatic negotiation")
        return []
    rc, text = run(["v4l2-ctl", "-d", device, "--list-formats-ext"])
    with open(os.path.join(out_dir, "v4l2_formats.txt"), "w") as f:
        f.write(text)
    formats = parse_v4l2_formats(text)
    if not formats:
        log("No formats parsed. Raw output:\n" + text.strip())
    for fmt in formats:
        modes = ", ".join("{}x{}@{}".format(w, h, "/".join("{:g}".format(r) for r in sorted(set(rs))) or "?")
                          for (w, h), rs in sorted(fmt["sizes"].items()))
        kind = ("-> GStreamer " + FOURCC_TO_GST[fmt["fourcc"]] if fmt["fourcc"] in FOURCC_TO_GST
                else "-> 8-bit Bayer" if fmt["fourcc"] in BAYER8_TO_GST
                else "-> high-bit-depth Bayer (raw, needs ISP/debayer)" if fmt["fourcc"] in BAYER_HIGH
                else "-> MJPEG" if fmt["fourcc"] == "MJPG" else "-> unknown to this script")
        log("  '{}' {}  {}".format(fmt["fourcc"], kind, modes))
    return formats


# ------------------------------------------------------------------------------------------------
# 3. Candidate pipelines
# ------------------------------------------------------------------------------------------------
def candidate_sources(device, formats, have, args):
    """Each candidate: dict(name, src, rest) -> pipeline = src ! rest ! appsink."""
    cands = []
    for fmt in formats:
        cc = fmt["fourcc"]
        w, h, fps = pick_mode(fmt, args.width, args.height, args.fps)
        rate = ", framerate={}/1".format(fps)
        src = "v4l2src device={}".format(device)
        if cc in FOURCC_TO_GST:
            g = FOURCC_TO_GST[cc]
            caps = "video/x-raw, format={}, width={}, height={}{}".format(g, w, h, rate)
            cands.append({"name": "v4l2 {} {}x{}@{} + videoconvert (CPU)".format(g, w, h, fps), "src": src,
                          "rest": caps + " ! videoconvert ! video/x-raw, format=BGR"})
            if "nvvidconv" in have and g in NVVIDCONV_INPUTS:
                cands.append({"name": "v4l2 {} {}x{}@{} + nvvidconv (hardware)".format(g, w, h, fps), "src": src,
                              "rest": caps + " ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR"})
        elif cc in BAYER8_TO_GST and "bayer2rgb" in have:
            caps = "video/x-bayer, format={}, width={}, height={}{}".format(BAYER8_TO_GST[cc], w, h, rate)
            cands.append({"name": "v4l2 bayer {} {}x{}@{} + bayer2rgb".format(cc, w, h, fps), "src": src,
                          "rest": caps + " ! bayer2rgb ! videoconvert ! video/x-raw, format=BGR"})
        elif cc == "MJPG":
            caps = "image/jpeg, width={}, height={}{}".format(w, h, rate)
            cands.append({"name": "v4l2 MJPEG {}x{}@{} + jpegdec".format(w, h, fps), "src": src,
                          "rest": caps + " ! jpegdec ! videoconvert ! video/x-raw, format=BGR"})
    cands.append({"name": "v4l2 automatic negotiation + videoconvert", "src": "v4l2src device={}".format(device),
                  "rest": "videoconvert ! video/x-raw, format=BGR"})
    if "nvarguscamerasrc" in have:
        w, h, fps = args.width or 1920, args.height or 1200, args.fps or 30
        cands.append({"name": "Argus/ISP nvarguscamerasrc {}x{}@{}".format(w, h, fps), "src": "nvarguscamerasrc sensor-id=0",
                      "rest": "video/x-raw(memory:NVMM), width={}, height={}, framerate={}/1 ! nvvidconv ! "
                              "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR".format(w, h, fps)})
    return cands


def gst_pretest(cand, timeout):
    """Run the pipeline in gst-launch-1.0 with 60 buffers into fakesink. Returns (ok, detail)."""
    line = "{} num-buffers=60 ! {} ! fakesink sync=false".format(cand["src"], cand["rest"])
    rc, out = run(["gst-launch-1.0"] + line.split(), timeout=timeout)  # gst-launch re-joins argv itself
    tail = " | ".join(l.strip() for l in out.strip().splitlines()[-3:])
    return rc == 0, "rc={} {}".format(rc, tail)


def open_capture(args, formats, gst_ok, have, attempts):
    """Return (cv2.VideoCapture, description) for the first source that works, or (None, None)."""
    if args.device.isdigit():  # laptop / USB webcam by index, OpenCV picks the backend
        cap = cv2.VideoCapture(int(args.device))
        return (cap, "OpenCV default backend, index " + args.device) if cap.isOpened() else (None, None)

    if args.pipeline:
        cands = [{"name": "custom --pipeline", "full": args.pipeline}]
    else:
        cands = candidate_sources(args.device, formats, have, args)
    can_pretest = shutil.which("gst-launch-1.0") is not None

    log("")
    log("=" * 72)
    log("Pipeline tests" + ("" if can_pretest else " (gst-launch-1.0 missing: trying OpenCV directly)"))
    log("=" * 72)
    chosen = None
    for cand in cands:
        full = cand.get("full") or "{} ! {} ! {}".format(cand["src"], cand["rest"], APPSINK)
        entry = {"name": cand["name"], "pipeline": full}
        if can_pretest and "full" not in cand:
            ok, detail = gst_pretest(cand, args.timeout)
            entry["gst-launch"] = ("PASS " if ok else "FAIL ") + detail
            log("  [{}] {}".format("PASS" if ok else "FAIL", cand["name"]))
            if not ok:
                log("         " + detail[-160:])
                attempts.append(entry)
                continue
        if chosen is None:
            if not gst_ok:
                entry["opencv"] = "skipped: this OpenCV has no GStreamer support"
                log("         OpenCV can't use it: built without GStreamer")
            else:
                cap = cv2.VideoCapture(full, cv2.CAP_GSTREAMER)
                ok = cap.isOpened() and cap.read()[0]
                entry["opencv"] = "opened" if ok else "failed to open"
                log("         OpenCV: " + entry["opencv"])
                if ok:
                    chosen = (cap, "GStreamer: " + full)
                else:
                    cap.release()
        attempts.append(entry)
        if chosen and not args.try_all:
            break
    if chosen:
        return chosen

    # Last resort: OpenCV's own V4L2 backend (handles YUYV / MJPG / GREY by itself)
    m = re.search(r"(\d+)$", args.device)
    if m and hasattr(cv2, "CAP_V4L2"):
        cap = cv2.VideoCapture(int(m.group(1)), cv2.CAP_V4L2)
        if args.width and args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        ok = cap.isOpened() and cap.read()[0]
        attempts.append({"name": "OpenCV V4L2 backend", "pipeline": "cv2.VideoCapture({}, cv2.CAP_V4L2)".format(m.group(1)),
                         "opencv": "opened" if ok else "failed to open"})
        log("  [{}] OpenCV V4L2 backend".format("PASS" if ok else "FAIL"))
        if ok:
            return cap, "OpenCV V4L2 backend, " + args.device
        cap.release()
    return None, None


# ------------------------------------------------------------------------------------------------
# 4. Capture + BGR -> RGB
# ------------------------------------------------------------------------------------------------
def capture_test(cap, args, out_dir, source_desc):
    log("")
    log("=" * 72)
    log("Capture test: " + source_desc)
    log("=" * 72)
    for _ in range(10):  # warm-up: exposure settles, first buffers are often slow
        cap.read()
    read_ms, frame, t_start = [], None, time.perf_counter()
    show = args.display and (os.name == "nt" or os.environ.get("DISPLAY"))
    for i in range(args.frames):
        t0 = time.perf_counter()
        ok, img = cap.read()
        read_ms.append((time.perf_counter() - t0) * 1e3)
        if not ok or img is None:
            log("read() failed at frame {}".format(i))
            break
        frame = img
        if show:
            fps_now = (i + 1) / (time.perf_counter() - t_start)
            view = cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1])))
            cv2.putText(view, "{:.1f} FPS".format(fps_now), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("NEON camera test (q to stop)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    elapsed = time.perf_counter() - t_start
    if show:
        cv2.destroyAllWindows()
    if frame is None:
        log("No frames captured.")
        return False

    n = len(read_ms)
    log("Frames      : {} in {:.2f} s -> {:.1f} FPS".format(n, elapsed, n / elapsed))
    log("read() time : mean {:.1f} ms, max {:.1f} ms".format(float(np.mean(read_ms)), float(np.max(read_ms))))
    log("Frame       : shape {} dtype {} (height, width, channels)".format(frame.shape, frame.dtype))
    if frame.ndim == 3:
        b, g, r = [float(frame[:, :, c].mean()) for c in range(3)]
        log("Channel mean: B {:.1f}  G {:.1f}  R {:.1f}   (OpenCV order is B, G, R)".format(b, g, r))

    # --- BGR -> RGB -------------------------------------------------------------------------
    bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    swapped = (np.array_equal(rgb[:, :, 0], bgr[:, :, 2]) and np.array_equal(rgb[:, :, 1], bgr[:, :, 1])
               and np.array_equal(rgb[:, :, 2], bgr[:, :, 0]))
    view = bgr[:, :, ::-1]  # the NumPy way: same values, but a strided *view* of the same memory
    t0 = time.perf_counter()
    for _ in range(50):
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t_cvt = (time.perf_counter() - t0) / 50 * 1e3
    t0 = time.perf_counter()
    for _ in range(50):
        np.ascontiguousarray(bgr[:, :, ::-1])
    t_np = (time.perf_counter() - t0) / 50 * 1e3
    log("")
    log("BGR -> RGB  : cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)")
    log("  channels correctly swapped (R<->B, G unchanged): {}".format(swapped))
    log("  same values as NumPy bgr[:, :, ::-1]: {}".format(np.array_equal(rgb, view)))
    log("  C-contiguous?  cvtColor result: {}   NumPy [::-1] view: {}".format(
        rgb.flags["C_CONTIGUOUS"], view.flags["C_CONTIGUOUS"]))
    log("  time per frame: cvtColor {:.2f} ms | NumPy + ascontiguousarray {:.2f} ms".format(t_cvt, t_np))
    log("  Remember: MiDaS / Depth Anything want RGB; Ultralytics YOLO wants the original BGR array.")

    # --- snapshots ----------------------------------------------------------------------------
    cv2.imwrite(os.path.join(out_dir, "frame_bgr.jpg"), bgr)
    cv2.imwrite(os.path.join(out_dir, "frame_rgb_written_as_bgr.jpg"), rgb)
    h = 360
    a = cv2.resize(bgr, (int(h * bgr.shape[1] / bgr.shape[0]), h))
    c = cv2.resize(rgb, (a.shape[1], h))
    for img, text in ((a, "BGR array -> imwrite: correct"), (c, "RGB array -> imwrite: red/blue swapped")):
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(out_dir, "comparison.jpg"), np.hstack([a, c]))
    log("")
    log("Saved frame_bgr.jpg, frame_rgb_written_as_bgr.jpg, comparison.jpg")
    return True


# ------------------------------------------------------------------------------------------------
# 5. Raw Bayer fallback
# ------------------------------------------------------------------------------------------------
def raw_bayer_fallback(device, formats, out_dir):
    fmt = next((f for f in formats if f["fourcc"] in BAYER_HIGH or f["fourcc"] in BAYER8_TO_GST), None)
    if fmt is None or not shutil.which("v4l2-ctl"):
        return
    cc = fmt["fourcc"]
    w, h, _ = pick_mode(fmt, None, None, None)
    raw_path = os.path.join(out_dir, "raw_frame.bin")
    log("")
    log("=" * 72)
    log("Raw Bayer fallback: capturing one '{}' {}x{} frame with v4l2-ctl".format(cc, w, h))
    log("=" * 72)
    base = ["v4l2-ctl", "-d", device, "--set-fmt-video=width={},height={},pixelformat={}".format(w, h, cc.strip()),
            "--stream-mmap", "--stream-count=1", "--stream-skip=5", "--stream-to=" + raw_path]
    rc, out = run(base + ["--set-ctrl", "bypass_mode=0"], timeout=20)  # Jetson VI drivers usually need this
    if rc != 0:
        rc, out = run(base, timeout=20)
    if rc != 0 or not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        log("raw capture failed: " + out.strip()[-300:])
        return
    data = np.fromfile(raw_path, dtype=np.uint8)
    bpp = 1 if cc in BAYER8_TO_GST else 2
    stride = len(data) // h
    img = data[:h * stride].reshape(h, stride)
    img = img[:, :w * bpp] if bpp == 1 else img[:, :w * 2].copy().view("<u2")[:, :w]
    img8 = (img.astype(np.float32) * (255.0 / max(1, int(img.max())))).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "raw_gray.jpg"), img8)
    for code in ("BayerBG2BGR", "BayerGB2BGR", "BayerRG2BGR", "BayerGR2BGR"):
        cv2.imwrite(os.path.join(out_dir, "debayer_{}.jpg".format(code)), cv2.cvtColor(img8, getattr(cv2, "COLOR_" + code)))
    log("Saved raw_gray.jpg and debayer_*.jpg: the one with natural colours shows the Bayer order.")
    log("Raw Bayer needs colour processing (ISP) for good images: the next thing to try is the Argus path")
    log("(sudo systemctl start nvargus-daemon, then rerun) - send the report back first.")


# ------------------------------------------------------------------------------------------------
def pick_out_dir(requested):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    choices = [requested] if requested else [os.path.dirname(os.path.abspath(__file__)), os.path.expanduser("~")]
    for base in choices:
        path = os.path.join(base, "camera_test_" + stamp) if not requested else base
        try:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, ".write_test"), "w") as f:
                f.write("ok")
            os.remove(os.path.join(path, ".write_test"))
            return path
        except OSError:
            continue
    sys.exit("Can't write results anywhere - pass --out <writable folder>")


def main():
    ap = argparse.ArgumentParser(description="NEON-201A-JNX camera capture prototype")
    ap.add_argument("--device", default="/dev/video0", help="/dev/videoN on the NEON, or a webcam index like 0")
    ap.add_argument("--width", type=int, help="capture width (default: largest the camera offers)")
    ap.add_argument("--height", type=int, help="capture height")
    ap.add_argument("--fps", type=int, help="capture frame rate (default: highest offered at that size)")
    ap.add_argument("--frames", type=int, default=150, help="frames to capture for the FPS test")
    ap.add_argument("--pipeline", help="test this exact GStreamer pipeline (must end in appsink)")
    ap.add_argument("--try-all", action="store_true", help="test every candidate pipeline")
    ap.add_argument("--probe-only", action="store_true", help="report formats and pipeline tests, no capture")
    ap.add_argument("--display", action="store_true", help="show a live window (needs a monitor)")
    ap.add_argument("--timeout", type=int, default=15, help="seconds allowed per gst-launch test")
    ap.add_argument("--out", help="output folder (default: next to this script, else your home folder)")
    args = ap.parse_args()

    out_dir = pick_out_dir(args.out)
    gst_ok, have = report_environment(args.device)
    formats = [] if args.device.isdigit() else probe_formats(args.device, out_dir)
    attempts = []
    cap, desc = (None, None)
    if not (args.probe_only and args.device.isdigit()):
        cap, desc = open_capture(args, formats, gst_ok, have, attempts)

    success = False
    if cap is not None:
        log("")
        log("Working source: " + desc)
        if not args.probe_only:
            success = capture_test(cap, args, out_dir, desc)
        else:
            success = True
        cap.release()
    elif not args.device.isdigit():
        log("")
        log("No capture path worked.")
        raw_bayer_fallback(args.device, formats, out_dir)

    with open(os.path.join(out_dir, "pipeline_attempts.txt"), "w") as f:
        for a in attempts:
            f.write("\n".join("{}: {}".format(k, v) for k, v in a.items()) + "\n\n")
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    log("")
    log("Results in: " + out_dir + "  (report.txt, pipeline_attempts.txt, images)")
    log("RESULT: " + ("OK" if success else "NO WORKING CAPTURE - send report.txt back"))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
