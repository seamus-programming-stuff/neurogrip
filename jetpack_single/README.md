# NEON single-camera depth prototype — JetPack 5.1.2

This is an offline, single-camera application for the ADLINK NEON-2000-JNX20xA
with Jetson Xavier NX, JetPack **5.1.2 / L4T R35.4.1**, Ubuntu 20.04 and
system Python 3.8. It captures the built-in MIPI sensor, predicts a depth map
locally and serves a feed containing RGB beside colour depth. No dashboard or
buttons are present in the running feed. All model weights and ARM64 wheels
are included. Internet, PyTorch and a separate processing PC are not required.

Download the complete bundle from the [GitHub Release](https://github.com/seamus-programming-stuff/neurogrip/releases/tag/v0.1.0-neon-jp512).
The source checkout contains the application and tests; the release assets
include the model weights and offline wheels needed by `install.sh`.

## Copy and run

On the Windows PC, open **cmd** in the folder containing the downloaded archive:

```cmd
scp neurogrip-neon-single-jp512.tar.gz adlink@192.168.55.1:
ssh -t adlink@192.168.55.1
```

On the NEON, as `adlink`, without `sudo`:

```bash
tar -xzf neurogrip-neon-single-jp512.tar.gz
cd neurogrip-neon-single-jp512
bash install.sh
bash run.sh
```

Open **http://192.168.55.1:8092/** on the PC. Use the NEON's Ethernet IP instead
when connected through the switch. Ctrl+C stops capture and invalidates the
local bus. The ZIP contains the same application; use the tar.gz on Linux to
retain script permissions. `bash run.sh` works regardless of executable bits.

The NEON must already have JetPack 5.1.2. The installer checks its exact L4T,
architecture and Python version and preserves the system OpenCV/GStreamer
build. It creates only a project `.venv`, uses the included wheels and verifies
the bundle hashes. It does not upgrade JetPack or install system packages.

Only one program may own `/dev/video0`. If your previous camera viewer is
running, `run.sh` reports its PID. Identify that owner with
`fuser -v /dev/video0` and `ps -fp <PID>`, then stop its exact service or process
before starting this application. The package does not kill other programs.
For the earlier Neurogrip raw-camera service, the specific stop command is
`sudo systemctl stop neurogrip-camera.service` when that is the reported owner.

## Model and depth output

The included model is [Depth Anything V2 Metric Hypersim Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small), Apache 2.0,
with 24.8M parameters. This is the indoor metric checkpoint, not the relative
depth checkpoint. Its output estimates camera-axis depth in metres. The
export uses a fixed 448×252 portrait input (252px short side); the output is
upsampled to the 540×960 upright sensor image. Lower input resolution reduces
embedded compute but loses fine detail. The model predicts depth without
camera intrinsics. Converting two image regions into an XYZ gap needs
intrinsics and foreground tracking.

The colour map spans 0–2 metres: near pixels use the low end of Turbo and far
pixels the high end. Values above 2m saturate visually; the numeric map retains
the predicted metres up to the checkpoint's 20m range. Depth holes are black.
Frame age and inference time are printed on the feed. `DELAYED` means the
result exceeded the 500ms vision lease. The display may show delayed results
for up to 10 seconds, explicitly labelled, while bus permission expires.

Endpoints:

| Path | Output |
| --- | --- |
| `/` or `/stream` | Feed-only RGB/depth display or MJPEG |
| `/snapshot.jpg` | Current RGB/depth JPEG |
| `/depth.npz` | Float32 `depth_m[960,540]`, age, stale flag, estimated/metric flags |
| `/state` | Model, inference time, camera status and telemetry |
| `/telemetry` | Version 1 vision packet as JSON |

The NPZ uses host monotonic capture-read completion timestamps. Those clocks
are meaningful only on the same NEON; a remote reader adds receive and
transport time to `capture_age_ms`. The NPZ endpoint returns 503 after its
10-second display deadline or a capture/model error.

Default inference uses the bundled **ONNX Runtime CPU** wheel, four CPU
threads. It works independently of CUDA/TensorRT Python bindings. Optional
`backend: "auto"` or `"trt"` in `config.json` uses the NEON's system TensorRT
8.5 and CUDA 11.4; the engine is built on the NEON and cached locally. `auto`
falls back to CPU if engine construction or inference fails; `trt` reports
the error. Engine construction can take several minutes. No PC-built engine
is shipped because engines depend on the target GPU/software.

Run the model smoke check without taking the sensor:

```bash
.venv/bin/python app.py --self-test
.venv/bin/python check_neon.py --no-camera --model models/depth.onnx --infer-smoke
```

## Optional camera calibration and finger/target teaching

Depth recognition starts with `run.sh`; calibration is optional for the colour
depth feed. For a fingertip-to-object XYZ gap, stop the running feed and use a
rigid printed checkerboard with **9×6 inner corners** (10×7 squares). Measure
the actual square edge; do not assume printer scale. On a NEON desktop with a
monitor, or with a functioning SSH X-forwarded display:

```bash
bash calibrate.sh --square-mm 25
```

Replace `25` with the measured edge in millimetres. Save at least 15 stationary
board views with SPACE, varying distance, tilt and image position; C computes
intrinsics. Then remove the board, put the finger and target in view, press
SPACE to freeze the reference, and draw each foreground box with a small
background margin. ENTER confirms each box. Add visible texture to plain
surfaces if the tracker cannot acquire them. The script saves camera/teaching
files and starts the feed automatically.

Use `--no-run` to save without starting, `--no-teach` for intrinsics only, or
`--teach-only` to retake the finger/target reference using the existing
intrinsics. Tracker loss requires re-teaching; it does not silently acquire a
different object. Teaching is tied to the camera calibration's hash. Camera
calibration files describe the rotated/resized image, before undistortion;
tracking and inference both use its undistorted image and matching K matrix.

This script calibrates image geometry. It cannot make learned monocular depth
accurate to a millimetre or infer actuator stroke geometry. The `gap_mm` field
is an estimate of visible-surface separation. Validate it against known
distances or your independent RGB-D camera across the actual workspace before
using it for movement. Recalibration or re-teaching resets all physical
validation flags.

## Controller output

Local programs read `/dev/shm/neurogrip_vision_v1`, a 96-byte mapped-memory
envelope containing the versioned 48-byte telemetry packet. See **BUS.md** for
locking, layout, flags, CRC, freshness and a Python reader. This shared memory
is local to the NEON. An electrical controller on Ethernet receives UDP.

Set `udp.enabled` to `true` and `udp.host` to the controller's IP in
`config.json`; the default destination port is 55050 and source port 55051.
The application sends the same 48-byte packet at 20Hz, including while depth
is unavailable or stale. `bus_schema.json`, `include/neurogrip_bus.h` and
`ethernet_bus_receiver.py` provide the schema and receiver reference. The
legacy function name `encode_can_fd` describes a 48-byte encoding only; this
application transmits it using UDP, not CAN electrical signalling.

Direction values are **BACKWARDS=0, UPRIGHT=1, FORWARDS=2, UNKNOWN=255**.
`strike_permit` is a time-limited vision recommendation, not a strike command.
By default it is false. Without calibration/teaching, gap is null and direction
UNKNOWN while depth recognition still runs. To produce validated direction
recommendations, the engineer must supply `camera_to_finger`, measured XYZ
stroke paths and physical error/timing bounds in `strike_config.json`, then
enable the corresponding validation flags. The controller must also be armed
explicitly with `controller_armed: true` after validation. Hidden surfaces and
other mechanical links require controller-side interlocks. Numerical coverage
and tracking quality do not represent neural-network confidence.

The receiver must pair the current boot session, pin the source IP/port,
reject backward/replayed sequences and CRC errors, and apply the verified
network-delay bound in addition to capture age. Unknown, stale, simulation,
unvalidated or blocked packets must revoke movement eligibility. The shipped
receiver implements these checks but contains no motor driver.

## Optional startup at boot

After a successful foreground run and stopping any previous sensor owner:

```bash
.venv/bin/python install_service.py --start
journalctl -u neurogrip-depth.service -n 40 --no-pager
```

This explicitly installs and enables only `neurogrip-depth.service`. To stop:
`sudo systemctl stop neurogrip-depth.service`. Stop that service before
interactive calibration, then restart it afterwards if desired.

## Validation and package integrity

The ONNX graph is IR8/opset14, compatible with the bundled ORT1.16.3 runtime.
Its fixed positional embedding was computed using the unchanged upstream
code to remove a TensorRT8.5-unsupported cubic resize. Export numerical checks
compare it with the original PyTorch model; results and source/checkpoint
revisions are in `models/manifest.json`. Python3.8 compatibility, offline
bootstrap, model preprocessing, bus faults and application freshness are
covered by the supplied test report. Desktop model inference has been run.
**Actual NEON CPU speed, TensorRT engine construction, Linux sensor capture
and Linux shared-memory behavior still require the on-device checks above.**

`SHA256SUMS` checks the initial application, model and every wheel. After
editing settings or calibration flags, `bash install.sh --repair` verifies all
immutable files and preserves the two mutable configuration JSONs while
repairing this package's virtual environment. An unrelated or incomplete
`.venv` is refused; extract into a fresh directory instead. No remote downloads
occur in either installation mode. The shipped wheels contain their third-party
licenses; model license and attribution are in `models/LICENSE` and
`models/NOTICE.txt`.
