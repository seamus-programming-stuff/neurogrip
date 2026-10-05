"""Run the real cached model and save relative preview/runtime diagnostics."""
import argparse
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from depth_backend import MetricDepthBackend, DEFAULT_MODEL_REVISION, UPSTREAM_SOURCE_REVISION
from neurogrip import depth_colors


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(root / ".vendor/depth-anything-3/assets/examples/SOH/000.png"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default=str(root / "outputs/model-check"))
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        parser.error("Cannot read image; run setup_model.py or supply --image")
    backend = MetricDepthBackend(device=args.device, process_res=args.process_res, local_files_only=args.offline)
    started = time.perf_counter()
    result = backend.infer(image)
    cold = time.perf_counter() - started
    if result.metric or not np.isnan(result.depth_m).all() or result.visualization_depth.shape != image.shape[:2]:
        raise RuntimeError("Uncalibrated depth contract failed")
    elapsed = []
    for _ in range(5):
        started = time.perf_counter()
        result = backend.infer(image)
        elapsed.append(time.perf_counter() - started)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    preview = np.concatenate((image, depth_colors(result)), axis=1)
    if preview.shape[1] > 1600:
        preview = cv2.resize(preview, (1600, int(preview.shape[0] * 1600 / preview.shape[1])))
    cv2.imwrite(str(output / "depth-preview.jpg"), preview)
    np.savez_compressed(output / "depth.npz", depth_m=result.depth_m, visualization_depth=result.visualization_depth,
                        valid_mask=result.valid_mask, metric=np.array(False), mode=np.array("SIMULATION"))
    report = {"backend": result.backend, "source_revision": UPSTREAM_SOURCE_REVISION,
              "checkpoint_revision": DEFAULT_MODEL_REVISION, "process_res": args.process_res,
              "image_size": [image.shape[1], image.shape[0]], "first_call_seconds": cold,
              "warm_calls_seconds": elapsed, "warm_median_seconds": statistics.median(elapsed),
              "valid_coverage": float(np.mean(result.valid_mask)), "metric": False,
              "note": "Runtime smoke test. No camera calibration, distance accuracy or physical strike validation."}
    (output / "runtime.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {output / 'depth-preview.jpg'}", flush=True)


if __name__ == "__main__":
    main()
