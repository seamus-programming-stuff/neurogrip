# Neurogrip: stereo and single-camera depth

## Standalone NEON single camera — JetPack 5.1.2

Download the complete offline prototype from the [JetPack 5.1.2 release](https://github.com/seamus-programming-stuff/neurogrip/releases/tag/v0.1.0-neon-jp512), then follow [the NEON setup guide](jetpack_single/README.md). It runs the included Depth Anything V2 Metric Small ONNX model directly on a JetPack 5.1.2 NEON, with offline ARM64/Python 3.8 dependencies, a feed-only RGB/depth display, local mapped memory and optional Ethernet UDP telemetry. Camera calibration and teaching are available through one script. The package preserves system OpenCV and includes an optional on-device TensorRT 8.5 backend. The release assets include model weights and wheels; cloning this source repository does not. See [rebuild instructions](docs/neon-release-build.md) to reproduce the downloadable bundle.

## Dual NEON cameras: current setup path

The dual-camera path adds calibrated stereo depth on the PC, a **feed-only display**, named shared memory, and Ethernet UDP output for the electrical controller. The raw camera helper supports Python3.6+ for JetPack4.x and the existing JetPack5 camera. See [the setup guide](docs/dual-camera-setup.md) and [bus handoff](docs/shared-memory-bus.md).

One-time camera setup from Windows cmd, replacing the addresses:

```bat
setup_neon.cmd LEFT_IP
setup_neon.cmd RIGHT_IP
```

Then one command enters addresses, calibrates with a measured checkerboard, teaches fingertip/target, and starts the feed:

```bat
.venv\Scripts\python.exe dual_calibrate.py
```

Subsequent startup is `run_dual.cmd`. The feed at `http://127.0.0.1:8090/` has no controls. `dual_camera.py --demo` runs a labelled synthetic stereo test feed. Physical testing awaits the second camera and its real address. A switch does not synchronize exposures; physical strike recommendations require measured stroke, gap-error and timing validation. Cameras retain their own power and must stay still after calibration.

## Existing single-camera prototype

A Windows PC prototype estimates optical-axis depth using **Depth Anything 3 Metric Large**, tracks a taught fingertip and target, reconstructs their visible surfaces in 3D, and reports their estimated gap. A local browser dashboard shows the RGB overlay and depth map. The PC also provides versioned shared-bus telemetry for backwards, upright and forwards recommendations.

This implementation runs a pretrained model locally; it does not train a new foundation model. Model selection and scaling are based on the [official DA3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3) and [Apache-2.0 metric model card](https://huggingface.co/depth-anything/DA3METRIC-LARGE). Source and checkpoint revisions are pinned in `depth_backend.py`. The official metric checkpoint produces canonical depth; the backend converts it with **the resized image's mean focal length / 300**. It uses the upstream network and preprocessing directly, avoiding unused 3D-export dependencies.

## Run the PC prototype

For a fresh checkout, create a project virtual environment and install the PC requirements. These commands are for **Windows cmd**, from this folder:

```bat
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe neurogrip.py --demo
```

Open [the dashboard](http://127.0.0.1:8088). The demo generates known synthetic depths and moves a target into/out of a sample stroke. It tests the UI and geometry, not learned depth accuracy; every bus packet is SIMULATION with `strike_permit=false`.

To run the actual model on a bundled public test image:

```bat
.venv\Scripts\python.exe neurogrip.py --image .vendor\depth-anything-3\assets\examples\SOH\000.png --device cuda --offline --port 8089
```

Open [the model preview](http://127.0.0.1:8089). Without camera intrinsics it shows a relative depth visualization and no millimetre distance. A still image or a recorded video is always SIMULATION. `Ctrl+C` stops an app started in your terminal. Ports can be changed with `--port`.

## Live camera

For a PC webcam, use `--source 0`. For the NEON, first run the included raw-stream helper on the camera. It uses only Python 3.8, system OpenCV and NumPy. **Do not install PC requirements or pip OpenCV on the NEON.** The known capture mode is 1920x1080, set before the first frame read; the helper rotates clockwise by default, matching the brief.

Windows cmd:

```bat
scp neon_raw.py adlink@192.168.55.1:
ssh -t adlink@192.168.55.1 python3 -u neon_raw.py
```

In another PC terminal:

```bat
.venv\Scripts\python.exe neurogrip.py --source http://192.168.55.1:8081/raw --device cuda
```

The address above comes from the existing USB-network setup. If the NEON is on your gigabit switch, replace it with its actual LAN IP. The sensor has only one owner: stop any earlier script that still holds `/dev/video0` before starting `neon_raw.py`. Use `ssh -t` so `Ctrl+C` reaches that process. The raw stream contains no overlays; an annotated `/stream` from another app would contaminate depth/tracking. NEON hardware capture and LAN throughput are **VERIFY ON HARDWARE**.

In the dashboard, choose **Select fingertip & target**, keep the scene still, drag a box with a small background margin, and teach each region. Select only the visible fingertip, not the whole hand. Tracker loss clears its mask and requires re-teaching; it does not keep an old box valid. The tracker requires enough foreground texture. A small matte textured patch helps; a plain, reflective, transparent, deforming or occluded region may be rejected.

Click **Enable recommendations** after teaching. This arms the PC's recommendation logic only, not the MCU. Missing calibration/validation still blocks permission. You can preview depth and estimated distance independently of physical permission.

## Establish millimetres

Print a checkerboard on a rigid flat backing. Measure a square's real size; do not assume printer scaling. `--board 9x6` means **9 by 6 inner corners**, i.e. 10 by 7 squares.

```bat
.venv\Scripts\python.exe camera_calibrate.py --source http://192.168.55.1:8081/raw --board 9x6 --square-mm 25 --output camera.json
```

Hold the board still for each capture. SPACE saves a view; collect at least 15 varied views across the image and working distances, then press C. Use your measured square size instead of 25 if different. The utility writes focal lengths, distortion and resolution. Restart live inference with:

```bat
.venv\Scripts\python.exe neurogrip.py --source http://192.168.55.1:8081/raw --camera camera.json --config config.json --device cuda
```

Create `config.json` first by copying `config.example.json` and filling the measured values described below. The application undistorts frames with the same intrinsic matrix before depth inference. The calibration resolution must match the incoming frames exactly. Keep rotation, crop, capture mode and focus fixed. The RGB-D camera in your brief can independently test distance errors.

Without a calibrated K, `depth_m` contains NaNs, `metric=false`, and the visualization uses arbitrary canonical units. We never silently invent focal length or interpret relative depth as millimetres. Even with K, monocular metric depth is a learned estimate: calibration does not turn it into a contact sensor.

## Calibrate reachable strokes

The default config contains **no invented finger reach or direction mapping**. Live permission stays off until all relevant measurements are supplied:

| Config field | Required measurement |
|---|---|
| `depth_scale` | Optional multiplicative model-bias correction measured across working depths; default 1 |
| `gap_error_mm` | Conservative observed 3D gap-error bound across the intended conditions, not a guessed model confidence |
| `measurement_validated` | Set true only after measured gap validation |
| `source_delay_bound_ms` | Conservative measured exposure-to-PC-read delay bound, including encoding/transport/buffering |
| `capture_timing_validated` | Set true only after validating that upstream-age bound |
| `camera_to_finger` | Rigid 4x4 camera-to-finger-base transform; translation in **mm** |
| `strokes` | Measured tip-position polylines, in the finger-base frame, for physical BACKWARDS/UPRIGHT/FORWARDS strokes |
| `tip_radius_mm` | Actual validated tip contact envelope radius, not a proximity tolerance expanding reach |
| `start_tolerance_mm` | Accepted visible-tip reference starting-position error |
| `obstacle_margin_mm` | Additional measured clearance around the tip's swept corridor |
| `stroke_geometry_validated` | Set true only after validating the stroke and controller's complete mechanical envelope |

Camera coordinates are X right, Y down, Z away from camera. The finger-base axes are whichever axes you actually calibrate; the rigid transform maps between them. No image axis, servo angle or motor polarity is assumed to mean "forwards". Each stroke is a list such as `[[x0,y0,z0],[x1,y1,z1],...]`, with at least two measured points. Its start must match the tracked visible fingertip reference. Its envelope must account for the real finger tip/body and any offset between the visible surface reference and the mechanical tip centre. A marker on the palm alone does not locate a bending fingertip.

Moving the camera relative to the finger invalidates that transform. **Disarm and revalidate manually**; this prototype does not automatically solve camera-to-finger motion. Lens focus/capture-mode changes require appropriate intrinsic recalibration. Learned depth may vary by frame and by object/material, so validating one distance is insufficient for the whole workspace.

The geometric gate uses XYZ distances between foreground surfaces, rather than only the difference in their camera Z depths. It requires a target within the calibrated tip contact envelope, a matching starting pose, sufficiently slow target and fingertip, observed depth across the projected swept-tip corridor, and clearance from visible outside surfaces. Three independent eligible frames enable a recommendation by default; a failed condition revokes it immediately. The output can describe only visible geometry; hidden surfaces and the full hand's mechanics require controller-side checks. A depth hole or overlap between the two selected regions blocks contact interpretation.

`valid_for_ms` is a total observation-age budget, including inference time and the configured upstream delay bound. The default 500 ms is for bench inspection; choose a measured budget compatible with movement speed. The MCU helper independently defaults to a stricter 100 ms cap. Arrival/read timestamps are not sensor timestamps. No shared bus has been selected, and this app transmits no actuator messages.

## Electrical handoff and output

Give the engineer [docs/electrical-interface.md](docs/electrical-interface.md), [bus_schema.json](bus_schema.json) and [include/neurogrip_bus.h](include/neurogrip_bus.h). Direction values are BACKWARDS=0, UPRIGHT=1, FORWARDS=2, UNKNOWN=255. The schema includes finite/null millimetre gap, mode, quality, sequence/session, age/lifetime, block reasons and vision permit. `quality` is a heuristic coverage/tracking score, not an accuracy probability. Unknown distance is null, never zero.

The transport-neutral JSON has an optional 48-byte CAN-FD representation with CRC. A true permission heartbeat is **not** a strike request. The MCU needs local position/current/travel feedback and a separate one-shot action; a repeating heartbeat cannot repeat a strike. The shared bus, electrical levels, identifiers, bitrate and hardware adapter must be agreed with the engineer.

| Local endpoint | Output |
|---|---|
| `/stream` | Camera overlay and colored depth side by side |
| `/snapshot.jpg` | Undistorted image used by the model |
| `/depth.npz` | Raw float32 depth metres, relative/metric visualization, validity mask and frame/mode metadata |
| `/state` | Measurements, track status, model timing and telemetry |
| `/telemetry` | Strict JSON bus-schema packet |
| `/telemetry.bin` | Optional 48-byte CAN-FD payload; no transmission |
| `/present` | `1` only for a fresh LIVE vision permit, otherwise `0` |

`outputs/telemetry.jsonl` records 10 Hz telemetry; terminal output prints every `/present` change. HTTP/heartbeat handlers recompute age even while inference is blocked. Replay and synthetic data never emit physical permission. Live unknown/unvalidated geometry stays blocked. Saved NPZ uses `allow_pickle=False` when reading, with NaNs for unavailable numeric depths; JSON uses null for unavailable distances and rejects NaNs.

## Recreate the local model environment

Only needed on another machine or if rebuilding `.venv`. Commands target a Windows PC with an NVIDIA driver supporting the selected CUDA runtime. The tested build here is Python 3.13.14 / Torch 2.11.0+cu128 / Torchvision 0.26.0+cu128. CPU can also run via `--device cpu`, with slower inference.

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.venv\Scripts\python.exe setup_model.py
```

First inference downloads public weights to `.cache/huggingface`; later `--offline` runs use cached weights. The upstream source's original license files remain in `.vendor`. This direct metric path does not need xformers, Open3D, Gaussian splatting or pycolmap. Installation/runtime was tested on this PC, not the Jetson Xavier NX.

## Verification and limits

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

To repeat the real-model runtime check and save a JPEG/NPZ preview:

```bat
.venv\Scripts\python.exe model_smoke.py --device cuda --offline
```

Tests cover geometry, missed contact, uncertain/occluded corridors, moving fingertip, lost tracking, stale/unknown messages, simulation blocking, CRC, restarts/replay and rollover. The real DA3 model was run on the RTX 4060 Ti using the bundled public test image. At 504 processing resolution, five warm calls had median **62.7 ms** and peak CUDA allocation **1448.8 MiB**; cached startup took **13.4 s**. Those are model/runtime measurements, not whole-pipeline camera latency or accuracy. No artificial K from the smoke test is used by this app. The C reference header has not been compiled here because a C compiler is unavailable.

Live NEON capture, real finger geometry, physical bus transmission, camera-to-output age and distance/strike accuracy remain to be measured. Near-contact accuracy on plain, shiny, clear or occluded objects cannot be guaranteed by this monocular estimator. Validate empty scenes, approach from different depths, actual contact, target/camera movement, lighting changes and difficult materials against ground truth before using its recommendations for real movement.
