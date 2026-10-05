"""One PC command: calibrate both NEONs, teach regions, then start the feed."""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlparse, urlunparse
import urllib.request

import cv2
import numpy as np

from dual_config import DualConfig, stream_address, controller_address
from stereo_source import StereoSource
from stereo_core import StereoCalibration
from stereo_calibrate_core import calibrate_stereo


def probe_neon_stream(source, timeout=4):
    parts = urlparse(source)
    state_url = urlunparse((parts.scheme, parts.netloc, "/state", "", "", ""))
    with urllib.request.urlopen(state_url, timeout=timeout) as response:
        state = json.load(response)
    if not isinstance(state, dict) or state.get("overlays") is not False:
        raise ValueError("%s is not an overlay-free NEON helper. Install neon_raw.py; use port 8081 /raw." % source)
    if state.get("error") or state.get("fresh") is not True:
        raise ValueError("%s has no fresh camera frame: %s" % (source, state.get("error", "capture unavailable")))
    return state


def board_signature(corners, size):
    points = np.asarray(corners).reshape(-1, 2)
    span = np.ptp(points, axis=0)
    center = points.mean(axis=0) / np.asarray(size)
    scale = float(np.sqrt(np.prod(span)) / np.sqrt(np.prod(size)))
    # Pose diversity includes perspective distortion as well as location/size.
    centered = points - points.mean(axis=0)
    normalized = centered / max(1.0, np.linalg.norm(centered))
    return center, scale, normalized


def distinct_view(corners, size, previous):
    current = board_signature(corners, size)
    for old in previous:
        if (np.linalg.norm(current[0] - old[0]) < .06
                and abs(current[1] - old[1]) < .045
                and np.linalg.norm(current[2] - old[2]) < .09):
            return False
    return True


def collect_views(camera, board, square_mm, minimum_views, output_dir, connect_timeout=15):
    world = np.zeros((board[0] * board[1], 3), np.float32)
    world[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square_mm
    objects, left_points, right_points, signatures = [], [], [], []
    image_size = None
    previous = None
    stable_since = None
    source_epoch = None
    last_pair_time = time.monotonic()
    print("Keep cameras still. Hold the board STILL in both views for each capture.")
    print("SPACE saves a stable varied view; C solves after %d views; Q cancels." % minimum_views)
    print("Cover the corners, use tilts and several working distances. Avoid upside-down board flips.")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            pair = camera.read()
            if pair is None:
                if time.monotonic() - last_pair_time > connect_timeout:
                    raise RuntimeError("No usable stereo pair: " + json.dumps(camera.status()))
                if cv2.waitKey(10) & 255 == ord("q"):
                    raise KeyboardInterrupt()
                time.sleep(.005)
                continue
            last_pair_time = time.monotonic()
            if source_epoch is not None and pair.source_epoch != source_epoch:
                raise RuntimeError("Camera reconnected or changed resolution during calibration; restart the complete collection")
            source_epoch = pair.source_epoch
            left, right = pair.left.image, pair.right.image
            size = (left.shape[1], left.shape[0])
            if image_size is not None and size != image_size:
                raise ValueError("Camera resolution changed; restart calibration")
            found_left, corners_left = cv2.findChessboardCornersSB(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), board)
            found_right, corners_right = cv2.findChessboardCornersSB(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), board)
            found = found_left and found_right
            if found and previous is not None:
                moved = max(float(np.max(np.linalg.norm(corners_left - previous[0], axis=2))),
                            float(np.max(np.linalg.norm(corners_right - previous[1], axis=2))))
                if moved > 1.5:
                    stable_since = None
            elif not found:
                stable_since = None
            if found:
                if stable_since is None:
                    stable_since = time.monotonic()
                previous = (corners_left.copy(), corners_right.copy())
            else:
                previous = None
            stable = found and time.monotonic() - stable_since >= .7
            previews = []
            for frame, detected, corners in ((left, found_left, corners_left), (right, found_right, corners_right)):
                preview = frame.copy()
                if detected:
                    cv2.drawChessboardCorners(preview, board, corners, True)
                previews.append(preview)
            preview = np.hstack(previews)
            text = "%d/%d views | %s | SPACE save / C solve / Q quit" % (len(objects), minimum_views, "BOARD STABLE" if stable else "hold board still in BOTH views")
            cv2.putText(preview, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            cv2.imshow("Dual camera calibration", preview)
            key = cv2.waitKey(10) & 255
            if key == ord("q"):
                raise KeyboardInterrupt()
            if key == 32:
                if not stable:
                    print("Not saved: both cameras must see a stationary board for 0.7 seconds.")
                elif not distinct_view(corners_left, size, signatures):
                    print("Not saved: move/tilt the board to add a different calibration view.")
                else:
                    image_size = size
                    objects.append(world.copy())
                    left_points.append(corners_left.astype(np.float32))
                    right_points.append(corners_right.astype(np.float32))
                    signatures.append(board_signature(corners_left, size))
                    number = len(objects)
                    cv2.imwrite(str(output_dir / ("left_%03d.png" % number)), left)
                    cv2.imwrite(str(output_dir / ("right_%03d.png" % number)), right)
                    print("Saved view %d; receive-time skew %.1f ms (not exposure skew)." % (number, pair.pair_skew_ms))
            if key == ord("c"):
                if len(objects) < minimum_views:
                    print("Collect at least %d varied paired views first." % minimum_views)
                    continue
                return objects, left_points, right_points, image_size
    finally:
        cv2.destroyAllWindows()


def teach_regions(camera, calibration, reference_path):
    print("Remove the board; put the finger and target in their starting positions.")
    print("SPACE freezes the rectified left feed for two teaching boxes. Q skips teaching.")
    frame = None
    last_frame_time = time.monotonic()
    try:
        while True:
            pair = camera.read()
            if pair is not None:
                frame, _ = calibration.rectify(pair.left.image, pair.right.image)
                last_frame_time = time.monotonic()
                cv2.imshow("Place finger and target", frame)
            if time.monotonic() - last_frame_time > 15:
                raise RuntimeError("Camera feed lost while teaching")
            key = cv2.waitKey(10) & 255
            if key == ord("q"):
                return {}
            if key == 32 and frame is not None and time.monotonic() - last_frame_time < .5:
                break
        from vision_tracking import RegionTracker
        regions = {}
        for name in ("finger", "target"):
            print("Draw a tight box around the %s with a small background margin; ENTER accepts, ESC skips." % name)
            box = tuple(int(value) for value in cv2.selectROI("Teach " + name, frame, showCrosshair=True, fromCenter=False))
            if min(box[2:]) < 16:
                print("Teaching skipped; the runtime will still show depth and publish blocked telemetry.")
                return {}
            tracker = RegionTracker()
            result = tracker.initialize(frame, box)
            if not result.valid:
                raise ValueError("%s teaching failed: %s. Use a matte textured surface and repeat --reuse-calibration." % (name, result.reason))
            regions[name + "_bbox"] = list(box)
        if not cv2.imwrite(str(reference_path), frame):
            raise OSError("Cannot save taught reference image")
        regions["regions_reference"] = str(reference_path.resolve())
        return regions
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", help="left NEON IPv4 address or /raw URL")
    parser.add_argument("--right", help="right NEON IPv4 address or /raw URL")
    parser.add_argument("--controller", help="electrical controller IPv4:UDP-port; 'none' for local bench only")
    parser.add_argument("--square-mm", type=float, help="measured checkerboard square width")
    parser.add_argument("--board", default="9x6", help="INNER corners, columns x rows")
    parser.add_argument("--views", type=int, default=18)
    parser.add_argument("--output", default="stereo.json")
    parser.add_argument("--config", default="dual_config.json")
    parser.add_argument("--reuse-calibration", action="store_true", help="only re-teach regions; cameras/lenses must be unchanged")
    parser.add_argument("--skip-regions", action="store_true", help="depth-only feed, no object/finger recommendation")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    try:
        previous = DualConfig.load(config_path) if config_path.exists() else DualConfig()
        if args.reuse_calibration:
            calibration = StereoCalibration.load(output_path)
            left = stream_address(args.left or calibration.left_source)
            right = stream_address(args.right or calibration.right_source)
            if (left, right) != (calibration.left_source, calibration.right_source):
                raise ValueError("Reused calibration must use the same camera URLs")
        else:
            left = stream_address(args.left or input("Left NEON Ethernet IPv4 address: "))
            right = stream_address(args.right or input("Right NEON Ethernet IPv4 address: "))
            calibration = None
        if urlparse(left).hostname == urlparse(right).hostname:
            raise ValueError("Two different physical cameras and IP addresses are required")
        requested_controller = args.controller
        if requested_controller is None:
            requested_controller = previous.controller or input("Electrical controller IPv4:UDP-port (or none for bench): ")
        controller = None if requested_controller.strip().lower() == "none" else requested_controller.strip()
        controller_address(controller)
        states = [probe_neon_stream(left), probe_neon_stream(right)]
        sessions = [state.get("camera_session") for state in states]
        if sessions[0] is not None and sessions[0] == sessions[1]:
            raise ValueError("Both URLs report the same camera session; check the wiring/addresses")
        with StereoSource(left, right).start() as camera:
            if calibration is None:
                columns, rows = map(int, args.board.lower().split("x"))
                square = args.square_mm if args.square_mm is not None else float(input("Measured square width in mm: "))
                if min(columns, rows) < 3 or not np.isfinite(square) or square <= 0 or args.views < 15:
                    raise ValueError("Use >=15 views, >=3 inner corners per axis and a positive measured square size")
                directory = config_path.parent / "outputs" / "stereo-calibration" / time.strftime("%Y%m%d-%H%M%S")
                points = collect_views(camera, (columns, rows), square, args.views, directory)
                print("Solving camera geometry...")
                calibration = calibrate_stereo(*points, left, right, metadata={
                    "board_inner_corners": [columns, rows], "square_mm": square,
                    "camera_states_at_calibration": states,
                    "stereo_timing_validated": False, "source_timestamp_kind": "host_read_completion",
                    "captured_pairs_directory": str(directory),
                })
            # Discard an old reference even when teaching is skipped.
            for key in ("finger_bbox", "target_bbox", "regions_reference"):
                calibration.metadata.pop(key, None)
            if not args.skip_regions:
                calibration.metadata.update(teach_regions(camera, calibration, output_path.with_name("stereo_reference.png")))
            calibration.save(output_path)
        # Geometry calibration cannot validate physical stroke or exposure timing.
        previous.calibration = str(output_path)
        previous.stroke_config = str(Path(previous.stroke_config).resolve())
        previous.controller = controller
        previous.recommendations_enabled = False
        previous.stereo_timing_validated = False
        previous.exposure_skew_bound_ms = None
        # Keep at least half the image width available to the disparity matcher.
        search_limit = min(1024, 16 * ((calibration.image_size[0] // 2) // 16))
        needed = 16 * int(np.ceil(calibration.K_rect[0, 0] * calibration.baseline_mm / previous.min_depth_mm / 16)) + 16
        previous.num_disparities = max(16, min(search_limit, needed))
        previous.save(config_path)
        print("Saved", output_path)
        print("Saved", config_path)
        nearest = calibration.K_rect[0, 0] * calibration.baseline_mm / (previous.num_disparities - 1)
        print("Disparity search: %d pixels; approximate nearest observable depth %.0f mm (overlap/texture also required)." % (previous.num_disparities, nearest))
        print("Depth is in metres/mm. Strike permission remains blocked pending measured stroke/timing validation.")
        if not args.no_run:
            from dual_camera import run
            run(config_path, open_browser=not args.no_browser)
    except KeyboardInterrupt:
        print("Stopped.")
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        parser.exit(1, "Calibration not completed: %s\n" % error)


if __name__ == "__main__":
    main()
