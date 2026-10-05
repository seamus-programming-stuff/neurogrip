"""Local monocular depth using the official Depth Anything 3 metric network.

The metric checkpoint predicts *canonical* depth, not metres directly. The
official conversion is raw_depth * mean(fx, fy) / 300, using focal lengths at
the model's processed image resolution. Real camera intrinsics are required
for that conversion. Input images must already be undistorted, with K matching
the actual image resolution and crop.

The result is estimated optical-axis depth Z, not Euclidean range or a contact
sensor. ``valid_mask`` denotes numeric/non-sky coverage, never confidence.
No Torch or upstream imports happen until the first inference call.

Sources inspected at upstream commit 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4:
https://github.com/ByteDance-Seed/Depth-Anything-3#-faq
https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/api.py
https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/utils/io/input_processor.py
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


DEFAULT_MODEL_ID = "depth-anything/DA3METRIC-LARGE"
DEFAULT_MODEL_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
UPSTREAM_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
CANONICAL_FOCAL_PX = 300.0


@dataclass(frozen=True)
class DepthResult:
    """Arrays have the same HxW shape as the input image.

    depth_m: float32 estimated Z in metres; all NaN when ``metric`` is False.
    valid_mask: bool finite, positive, non-sky model coverage. This does not
        certify correct depth, calibration, or actuator reachability.
    visualization_depth: float32 metres if metric, otherwise canonical depth
        in arbitrary units; invalid pixels are NaN in either case.
    """

    depth_m: np.ndarray
    valid_mask: np.ndarray
    backend: str
    metric: bool
    visualization_depth: np.ndarray


def _validate_intrinsics(intrinsics: Any) -> np.ndarray | None:
    if intrinsics is None:
        return None
    matrix = np.asarray(intrinsics, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must be a finite 3x3 camera matrix at input resolution")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("intrinsics fx and fy must be positive focal lengths in pixels")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-6):
        raise ValueError("intrinsics must have homogeneous bottom row [0, 0, 1]")
    if not np.isclose(matrix[1, 0], 0, atol=1e-6):
        raise ValueError("intrinsics must be an upper triangular pinhole camera matrix")
    return matrix.copy()


class MetricDepthBackend:
    """Lazy local DA3 metric inference; first use may download public weights.

    Install a matching Torch/Torchvision build, requirements-model.txt, and
    clone the pinned upstream source into .vendor/depth-anything-3. An already
    installed depth_anything_3 package also works. The direct official network
    and preprocessing are used so unused 3D-export dependencies are unnecessary.

    ``model_id`` may also be a local checkpoint directory containing config.json
    and model.safetensors. Only the DA3 metric-large architecture is accepted;
    relative and nested checkpoints are deliberately not treated as metric.
    ``local_files_only=True`` disables checkpoint downloads for deployment.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        process_res: int = 504,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(process_res, int) or isinstance(process_res, bool) or process_res < 112:
            raise ValueError("process_res must be an integer of at least 112 pixels")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonempty Hugging Face ID or checkpoint directory")
        self.model_id = model_id
        self.device = device
        self.process_res = process_res
        self.revision = revision
        if self.revision is None and model_id.lower() == DEFAULT_MODEL_ID.lower():
            self.revision = DEFAULT_MODEL_REVISION
        self.local_files_only = local_files_only
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / ".cache" / "huggingface"
        self._torch: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._actual_device: Any = None

    @property
    def backend_name(self) -> str:
        return f"DA3METRIC-LARGE:{self.model_id}"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        source_path = Path(__file__).parent / ".vendor" / "depth-anything-3" / "src"
        if importlib.util.find_spec("depth_anything_3") is None and source_path.is_dir():
            sys.path.insert(0, str(source_path))
        try:
            import torch
            from depth_anything_3.cfg import create_object, load_config
            from depth_anything_3.registry import MODEL_REGISTRY
            from depth_anything_3.utils.io.input_processor import InputProcessor
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Metric model dependencies/source are missing. Follow the model setup in README.md "
                "(requirements-model.txt plus pinned upstream source and matching Torch/Torchvision)."
            ) from exc

        chosen_device = self.device
        if chosen_device == "auto":
            chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
        actual_device = torch.device(chosen_device)
        if actual_device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported devices are auto, cpu, cuda, or cuda:<index>")
        if actual_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this Torch build cannot access CUDA")

        local_model_dir = Path(self.model_id)
        if local_model_dir.is_dir():
            config_path = local_model_dir / "config.json"
            weights_path = local_model_dir / "model.safetensors"
        else:
            download_kwargs = {
                "repo_id": self.model_id,
                "revision": self.revision,
                "cache_dir": str(self.cache_dir),
                "local_files_only": self.local_files_only,
            }
            config_path = Path(hf_hub_download(filename="config.json", **download_kwargs))
        with config_path.open("r", encoding="utf-8") as handle:
            checkpoint_config = json.load(handle)
        if checkpoint_config.get("model_name", "").lower() != "da3metric-large":
            raise ValueError("This backend requires the DA3METRIC-LARGE checkpoint, not relative-depth weights")
        if not local_model_dir.is_dir():
            weights_path = Path(hf_hub_download(filename="model.safetensors", **download_kwargs))

        # Load the known upstream architecture, never executable configuration
        # from an arbitrary downloaded model repository.
        model = create_object(load_config(MODEL_REGISTRY["da3metric-large"]))
        state = load_file(str(weights_path), device="cpu")
        if state and all(key.startswith("model.") for key in state):
            state = {key[len("model.") :]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True, assign=True)
        del state
        model = model.to(actual_device).eval()
        self._torch = torch
        self._processor = InputProcessor()
        self._actual_device = actual_device
        self._model = model

    def infer(self, bgr_image: np.ndarray, intrinsics: Any = None) -> DepthResult:
        """Estimate depth from one uint8 BGR image; no guess for missing K.

        A missing intrinsic matrix still produces a relative visualization, but
        depth_m is NaN and metric=False so physical-distance control can reject
        it. Pass the calibrated, undistorted image's effective K for metres.
        """
        if not isinstance(bgr_image, np.ndarray) or bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError("bgr_image must be an HxWx3 NumPy image")
        if bgr_image.dtype != np.uint8 or min(bgr_image.shape[:2]) < 14:
            raise ValueError("bgr_image must be uint8 with each dimension at least 14 pixels")
        matrix = _validate_intrinsics(intrinsics)
        self._ensure_loaded()
        import cv2

        rgb = np.ascontiguousarray(bgr_image[:, :, ::-1])
        imgs_cpu, _, processed_intrinsics = self._processor(
            image=[rgb],
            intrinsics=None if matrix is None else matrix[None],
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
            num_workers=1,
            sequential=True,
        )
        if min(imgs_cpu.shape[-2:]) < 14:
            raise ValueError("Image aspect ratio is too extreme for the selected processing resolution")
        torch = self._torch
        imgs = imgs_cpu.to(self._actual_device)[None].float()
        if self._actual_device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            autocast = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            # CPU uses float32, avoiding CUDA-oriented upstream autocast choices.
            autocast = nullcontext()
        with torch.inference_mode(), autocast:
            output = self._model(imgs)
        native_depth = output["depth"][0, 0].float().cpu().numpy()
        if native_depth.ndim == 3 and native_depth.shape[-1] == 1:
            native_depth = native_depth[..., 0]
        if native_depth.shape != tuple(imgs_cpu.shape[-2:]):
            raise RuntimeError("Upstream depth output does not match processed image geometry")
        native_valid = np.isfinite(native_depth) & (native_depth > 0)
        if output.get("sky") is not None:
            sky = output["sky"][0, 0].float().cpu().numpy()
            if sky.shape != native_depth.shape:
                raise RuntimeError("Upstream sky output does not match depth geometry")
            # Matches official compute_sky_mask threshold. This is a class mask,
            # not a depth confidence score or error estimate.
            native_valid &= np.isfinite(sky) & (sky < 0.3)

        metric = matrix is not None
        visualization = native_depth.astype(np.float32, copy=True)
        if metric:
            # InputProcessor adjusts K through BOTH resize stages (including
            # patch-size rounding); using the original focal would mis-scale Z.
            processed_matrix = processed_intrinsics[0].numpy()
            focal_px = float((processed_matrix[0, 0] + processed_matrix[1, 1]) / 2)
            visualization *= focal_px / CANONICAL_FOCAL_PX
        visualization[~native_valid] = np.nan

        height, width = bgr_image.shape[:2]
        # Nearest-neighbour upsampling preserves foreground/background boundary
        # values and invalid pixels, avoiding invented depths across boundaries.
        visualization = cv2.resize(visualization, (width, height), interpolation=cv2.INTER_NEAREST)
        valid_mask = cv2.resize(native_valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        valid_mask &= np.isfinite(visualization) & (visualization > 0)
        visualization[~valid_mask] = np.nan
        depth_m = visualization.copy() if metric else np.full((height, width), np.nan, dtype=np.float32)
        return DepthResult(
            depth_m=depth_m,
            valid_mask=valid_mask,
            backend=self.backend_name,
            metric=metric,
            visualization_depth=visualization,
        )
