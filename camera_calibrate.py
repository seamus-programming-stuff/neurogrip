"""PC-only chessboard calibration. Commands and board conventions: README.md."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from camera_source import CameraSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0")
    parser.add_argument("--board", default="9x6", help="INNER corners, columns x rows")
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--output", default="camera.json")
    args = parser.parse_args()
    try:
        columns, rows = map(int, args.board.lower().split("x"))
        if min(columns, rows) < 3 or not np.isfinite(args.square_mm) or args.square_mm <= 0:
            raise ValueError()
    except ValueError:
        parser.error("use positive measured square size and board like 9x6 (inner corners)")
    source = int(args.source) if args.source.isdecimal() else args.source
    world = np.zeros((columns * rows, 3), np.float32)
    world[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * args.square_mm
    object_points, image_points = [], []
    last_sequence, image_size = -1, None
    print("Hold a rigid flat board still. SPACE saves a view; C calibrates >=15 views; Q exits.")
    with CameraSource(source, stale_after=2).start() as camera:
        try:
            while True:
                packet = camera.read()
                if packet is None:
                    if camera.status()["state"] == "eof":
                        break
                    time.sleep(.01)
                    if cv2.waitKey(1) & 255 == ord("q"):
                        break
                    continue
                frame = packet.image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                size = (gray.shape[1], gray.shape[0])
                if image_size is not None and size != image_size:
                    raise RuntimeError("Resolution changed; restart calibration with fixed capture settings")
                found, corners = cv2.findChessboardCornersSB(gray, (columns, rows))
                preview = frame.copy()
                if found:
                    cv2.drawChessboardCorners(preview, (columns, rows), corners, found)
                cv2.putText(preview, f"Views: {len(image_points)}   SPACE save / C calibrate / Q quit", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                cv2.imshow("Camera calibration", preview)
                key = cv2.waitKey(10) & 255
                if key == ord("q"):
                    break
                if key == 32 and found and packet.sequence != last_sequence:
                    image_points.append(corners.astype(np.float32))
                    object_points.append(world.copy())
                    image_size, last_sequence = size, packet.sequence
                    print(f"Saved view {len(image_points)}")
                if key == ord("c"):
                    if len(image_points) < 15:
                        print("Collect at least 15 varied views, covering corners, tilts and working distances.")
                        continue
                    rms, matrix, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
                    result = {"schema_version": 1, "image_size": list(image_size), "K": matrix.tolist(),
                              "distortion": distortion.ravel().tolist(), "rms_px": float(rms),
                              "board_inner_corners": [columns, rows], "square_mm": args.square_mm,
                              "views": len(image_points), "timestamp_kind": "host_read_completion"}
                    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    print(f"Saved {args.output}; reprojection RMS {rms:.3f}px. Check known distances; RMS is not a depth accuracy guarantee.")
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
