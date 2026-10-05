"""Offline Depth Anything V2 inference for JetPack 5.1.2 / Python 3.8.

The packaged Hypersim Small model predicts camera-axis depth in metres. These
are learned estimates: ``valid_mask`` describes numerical coverage, not model
confidence or a verified clearance for a moving actuator.

Preprocessing and output interpolation match the official metric implementation:
https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/metric_depth/depth_anything_v2/dpt.py
https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/metric_depth/depth_anything_v2/util/transform.py

No PyTorch or network access is used on the NEON. ONNX Runtime is imported lazily
so the camera, protocol and fake-session tests can run without model packages.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import threading
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import cv2
import numpy as np


class _CudaRuntime:
    """Small CUDA 11.4 C-API binding; no PyCUDA or cuda-python dependency.

    CUDA declarations and memcpy enum values are verified against:
    https://docs.nvidia.com/cuda/archive/11.4.4/cuda-runtime-api/group__CUDART__MEMORY.html
    https://docs.nvidia.com/cuda/archive/11.4.4/cuda-driver-api/group__CUDA__DEVICE.html
    """

    def __init__(self) -> None:
        import ctypes
        import ctypes.util
        if sys.platform != "linux":
            raise RuntimeError("The optional TensorRT backend targets JetPack Linux")
        self.ctypes = ctypes
        candidates = [ctypes.util.find_library("cudart"), "libcudart.so.11.0", "libcudart.so"]
        self.library = None
        for candidate in candidates:
            if candidate:
                try:
                    self.library = ctypes.CDLL(candidate)
                    break
                except OSError:
                    continue
        if self.library is None:
            raise RuntimeError("JetPack CUDA runtime library libcudart is unavailable")
        signatures = {
            "cudaMalloc": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
            "cudaFree": ([ctypes.c_void_p], ctypes.c_int),
            "cudaMemcpy": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
            "cudaGetDevice": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "cudaSetDevice": ([ctypes.c_int], ctypes.c_int),
            "cudaDriverGetVersion": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "cudaGetErrorString": ([ctypes.c_int], ctypes.c_char_p),
        }
        for name, (arguments, return_type) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, return_type
        device = ctypes.c_int()
        self._check(self.library.cudaGetDevice(ctypes.byref(device)), "cudaGetDevice")
        self.device = device.value
        self._check(self.library.cudaFree(ctypes.c_void_p(0)), "initializing the CUDA primary context")
        version = ctypes.c_int()
        self._check(self.library.cudaDriverGetVersion(ctypes.byref(version)), "cudaDriverGetVersion")
        self.driver_version = version.value
        self.device_uuid = self._device_uuid()

    def _device_uuid(self) -> str:
        ctypes = self.ctypes
        # A CUDA driver UUID prevents loading an engine cached for another GPU.
        driver = ctypes.CDLL("libcuda.so.1")

        class UUID(ctypes.Structure):
            _fields_ = [("bytes", ctypes.c_ubyte * 16)]

        for name, arguments in (
            ("cuInit", [ctypes.c_uint]),
            ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
            ("cuDeviceGetUuid", [ctypes.POINTER(UUID), ctypes.c_int]),
        ):
            function = getattr(driver, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        device, uuid = ctypes.c_int(), UUID()
        if driver.cuInit(0) != 0 or driver.cuDeviceGet(ctypes.byref(device), self.device) != 0:
            raise RuntimeError("CUDA driver cannot identify the selected GPU")
        if driver.cuDeviceGetUuid(ctypes.byref(uuid), device.value) != 0:
            raise RuntimeError("CUDA driver cannot query a GPU UUID for the engine cache")
        return bytes(uuid.bytes).hex()

    def _check(self, status: int, operation: str) -> None:
        if status:
            description = self.library.cudaGetErrorString(status)
            message = description.decode("utf-8", errors="replace") if description else str(status)
            raise RuntimeError("%s failed: %s" % (operation, message))

    def activate(self) -> None:
        self._check(self.library.cudaSetDevice(self.device), "cudaSetDevice")

    def malloc(self, size: int) -> int:
        pointer = self.ctypes.c_void_p()
        self._check(self.library.cudaMalloc(self.ctypes.byref(pointer), size), "cudaMalloc")
        if pointer.value is None:
            raise RuntimeError("cudaMalloc returned a null pointer")
        return pointer.value

    def free(self, pointer: int) -> None:
        self._check(self.library.cudaFree(self.ctypes.c_void_p(pointer)), "cudaFree")

    def host_to_device(self, pointer: int, array: np.ndarray) -> None:
        self._check(self.library.cudaMemcpy(
            self.ctypes.c_void_p(pointer), self.ctypes.c_void_p(array.ctypes.data), array.nbytes, 1,
        ), "cudaMemcpy host-to-device")

    def device_to_host(self, array: np.ndarray, pointer: int) -> None:
        self._check(self.library.cudaMemcpy(
            self.ctypes.c_void_p(array.ctypes.data), self.ctypes.c_void_p(pointer), array.nbytes, 2,
        ), "cudaMemcpy device-to-host")


class _TensorRTSession:
    """Fixed FP32 TensorRT 8.5/8.6 engine with an ORT-compatible session interface.

    Python APIs are verified against NVIDIA's release/8.5 sources:
    https://github.com/NVIDIA/TensorRT/blob/release/8.5/python/docstrings/infer/pyCoreDoc.h
    Engine construction and execution still require validation on the actual NEON.
    """

    def __init__(self, model_path: Path, manifest: dict, *, trt_module: Any = None, cuda: Any = None) -> None:
        self._lock = threading.Lock()
        self._pointers = []
        self.context = None
        self.engine = None
        self.runtime = None
        self.cuda = None
        if trt_module is None:
            try:
                import tensorrt as trt_module
            except ImportError as exc:
                raise RuntimeError("System TensorRT is unavailable; use the CPU backend") from exc
        trt = trt_module
        version = tuple(int(part) for part in trt.__version__.split(".")[:2])
        if version not in ((8, 5), (8, 6)):
            raise RuntimeError("This JetPack backend requires TensorRT 8.5 or 8.6")
        self.backend_name = "TensorRT %s FP32" % trt.__version__
        self.cuda = cuda if cuda is not None else _CudaRuntime()
        self.cuda.activate()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        model_path = Path(model_path)
        identity = {
            "model_sha256": manifest["sha256"].lower(), "tensorrt": trt.__version__,
            "cuda_driver": self.cuda.driver_version, "gpu_uuid": self.cuda.device_uuid,
            "architecture": platform.machine(), "precision": "fp32-no-tf32-v1",
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        cache = model_path.parent / ".engine_cache"
        plan_path, metadata_path = cache / (key + ".plan"), cache / (key + ".json")
        self.engine = self._load_cache(plan_path, metadata_path, identity)
        if self.engine is None:
            print("Building an optional TensorRT FP32 engine on this NEON; first use may take several minutes.",
                  file=sys.stderr, flush=True)
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse(model_path.read_bytes()):
                errors = "; ".join(str(parser.get_error(index)) for index in range(min(parser.num_errors, 8)))
                raise RuntimeError("TensorRT cannot parse this ONNX export: " + errors)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)
            # Avoid half-precision transformer overflow and TF32 rounding changes.
            config.clear_flag(trt.BuilderFlag.FP16)
            config.clear_flag(trt.BuilderFlag.TF32)
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("TensorRT engine construction failed; use the CPU backend")
            serialized = bytes(plan)
            self.engine = self.runtime.deserialize_cuda_engine(serialized)
            if self.engine is None:
                raise RuntimeError("TensorRT could not deserialize its newly built engine")
            self._save_cache(plan_path, metadata_path, identity, serialized)
        try:
            self._prepare_bindings(trt, manifest)
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _load_cache(self, plan_path: Path, metadata_path: Path, identity: dict) -> Any:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("identity") != identity:
                return None
            serialized = plan_path.read_bytes()
            if hashlib.sha256(serialized).hexdigest() != metadata.get("plan_sha256"):
                return None
            return self.runtime.deserialize_cuda_engine(serialized)
        except (OSError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _save_cache(plan_path: Path, metadata_path: Path, identity: dict, serialized: bytes) -> None:
        # A read-only installation can still use its in-memory engine.
        try:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {"identity": identity, "plan_sha256": hashlib.sha256(serialized).hexdigest()}
            for target, data in (
                (plan_path, serialized),
                (metadata_path, json.dumps(metadata, indent=2).encode("utf-8")),
            ):
                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(dir=str(target.parent), delete=False) as handle:
                        temporary_path = Path(handle.name)
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(str(temporary_path), str(target))
                finally:
                    if temporary_path is not None and temporary_path.exists():
                        temporary_path.unlink()
        except OSError:
            pass

    def _prepare_bindings(self, trt: Any, manifest: dict) -> None:
        engine = self.engine
        if engine.has_implicit_batch_dimension or engine.num_bindings != 2:
            raise RuntimeError("TensorRT engine must have one explicit-batch input and one output")
        input_index = engine.get_binding_index(manifest["input_name"])
        output_index = engine.get_binding_index(manifest["output_name"])
        if input_index < 0 or output_index < 0 or input_index == output_index:
            raise RuntimeError("TensorRT engine names disagree with the ONNX manifest")
        self.input_index, self.output_index = input_index, output_index
        if not engine.binding_is_input(input_index) or engine.binding_is_input(output_index):
            raise RuntimeError("TensorRT engine input/output order is inconsistent")
        self.input_shape = tuple(engine.get_binding_shape(input_index))
        self.output_shape = tuple(engine.get_binding_shape(output_index))
        height, width = manifest["input_shape"][2:]
        if self.input_shape != tuple(manifest["input_shape"]):
            raise RuntimeError("TensorRT input shape disagrees with the ONNX manifest")
        if self.output_shape not in ((1, height, width), (1, 1, height, width)):
            raise RuntimeError("TensorRT output shape disagrees with the ONNX manifest")
        for index in (input_index, output_index):
            # TensorRT 8.5's nptype() references removed np.bool on NumPy 1.24.
            # We support only FP32 bindings, so comparing its enum avoids that API.
            if engine.get_binding_dtype(index) != trt.float32:
                raise RuntimeError("TensorRT bindings must remain float32")
            if engine.is_shape_binding(index) or not engine.is_execution_binding(index):
                raise RuntimeError("TensorRT requires fixed execution tensor bindings")
            if engine.get_binding_format(index) != trt.TensorFormat.LINEAR:
                raise RuntimeError("TensorRT engine requires unsupported non-linear buffer layout")
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self.bindings = [0] * engine.num_bindings
        for index, shape in ((input_index, self.input_shape), (output_index, self.output_shape)):
            pointer = self.cuda.malloc(int(np.prod(shape)) * np.dtype(np.float32).itemsize)
            self._pointers.append(pointer)
            self.bindings[index] = pointer
        self.input_name, self.output_name = manifest["input_name"], manifest["output_name"]
        self._input_nodes = [SimpleNamespace(name=self.input_name, shape=list(self.input_shape), type="tensor(float)")]
        self._output_nodes = [SimpleNamespace(name=self.output_name, shape=list(self.output_shape), type="tensor(float)")]

    def get_inputs(self) -> list:
        return self._input_nodes

    def get_outputs(self) -> list:
        return self._output_nodes

    def run(self, names: list, feeds: dict) -> list:
        if names != [self.output_name] or set(feeds) != {self.input_name}:
            raise ValueError("TensorRT inference input/output names do not match the manifest")
        tensor = feeds[self.input_name]
        if tensor.shape != self.input_shape or tensor.dtype != np.float32 or not tensor.flags.c_contiguous:
            raise ValueError("TensorRT input must match the contiguous float32 export shape")
        with self._lock:
            if self.context is None:
                raise RuntimeError("TensorRT session is closed")
            self.cuda.activate()
            self.cuda.host_to_device(self.bindings[self.input_index], tensor)
            if not self.context.execute_v2(bindings=self.bindings):
                raise RuntimeError("TensorRT GPU inference failed")
            output = np.empty(self.output_shape, dtype=np.float32)
            self.cuda.device_to_host(output, self.bindings[self.output_index])
        return [output]

    def close(self) -> None:
        with self._lock:
            error = None
            if self.cuda is not None and self._pointers:
                try:
                    self.cuda.activate()
                except Exception as exc:
                    error = exc
                for pointer in self._pointers:
                    try:
                        self.cuda.free(pointer)
                    except Exception as exc:
                        error = error or exc
                self._pointers = []
            self.context, self.engine, self.runtime = None, None, None
            if error is not None:
                raise error

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass
class DepthResult:
    depth_m: np.ndarray
    valid_mask: np.ndarray
    visualization_depth: np.ndarray
    metric: bool
    backend: str


def official_resize_shape(width: int, height: int, input_size: int) -> Tuple[int, int]:
    """Return (width, height) using the official lower-bound / patch-14 rule."""
    if width <= 0 or height <= 0 or input_size <= 0:
        raise ValueError("Image dimensions and input_size must be positive")
    scale = max(float(input_size) / width, float(input_size) / height)

    def rounded(value: float) -> int:
        result = int(np.round(value / 14.0) * 14)
        if result < input_size:
            result = int(np.ceil(value / 14.0) * 14)
        return result

    return rounded(scale * width), rounded(scale * height)


def _resize_align_corners(
    values: np.ndarray, valid: np.ndarray, shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Torch bilinear align_corners=True semantics without importing Torch.

    Invalid source values cannot turn into apparently valid interpolated depth.
    A neighbour whose interpolation weight is zero does not affect validity.
    """
    out_height, out_width = shape
    height, width = values.shape
    if (height, width) == shape:
        result = values.copy()
        result[~valid] = np.nan
        return result, valid.copy()

    y = np.linspace(0.0, height - 1, out_height) if out_height > 1 else np.zeros(1)
    x = np.linspace(0.0, width - 1, out_width) if out_width > 1 else np.zeros(1)
    y0, x0 = y.astype(np.intp), x.astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)
    wy, wx = y - y0, x - x0
    safe = np.where(valid, values, 0.0)
    interpolated = np.zeros(shape, dtype=np.float64)
    supported = np.ones(shape, dtype=bool)
    for rows, y_weight in ((y0, 1.0 - wy), (y1, wy)):
        for columns, x_weight in ((x0, 1.0 - wx), (x1, wx)):
            weight = y_weight[:, None] * x_weight[None, :]
            samples = safe[rows[:, None], columns[None, :]]
            sample_valid = valid[rows[:, None], columns[None, :]]
            interpolated += weight * samples
            supported &= sample_valid | (weight == 0.0)
    interpolated = interpolated.astype(np.float32)
    supported &= np.isfinite(interpolated) & (interpolated > 0.0)
    interpolated[~supported] = np.nan
    return interpolated, supported


class DepthRuntime:
    """Load a checksummed, fixed-shape ONNX export and infer a BGR frame.

    ``backend='cpu'`` is the portable offline path. ``trt`` uses JetPack's system
    TensorRT; ``auto`` attempts TensorRT and falls back to CPU if it is unavailable
    or fails. The default never builds a GPU engine. A supplied ``session`` is
    useful for deterministic tests and must expose the ONNX Runtime session API.
    The manifest's units control whether distances are exposed; a relative
    inverse-depth model never supplies metres to the actuator controller.
    """

    def __init__(
        self,
        model_path: Any,
        manifest_path: Any,
        backend: str = "cpu",
        threads: int = 4,
        *,
        session: Optional[Any] = None,
    ) -> None:
        if backend not in ("cpu", "trt", "auto"):
            raise ValueError("Unsupported depth backend %r; use 'cpu', 'trt' or 'auto'" % backend)
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer from 1 to 64")
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self._validate_manifest()
        self._verify_model_hash()
        self.input_shape = tuple(self.manifest["input_shape"])
        self.input_size = self.manifest["input_size"]
        self.input_name = self.manifest["input_name"]
        self.output_name = self.manifest["output_name"]
        self.metric = self.manifest["units"] == "metres"
        self.max_depth_m = self.manifest.get("max_depth_m")
        self.requested_backend = backend
        self.threads = threads
        self.fallback_reason = None
        self.backend = "ONNX Runtime CPU"

        if session is None:
            if backend in ("trt", "auto"):
                try:
                    session = _TensorRTSession(self.model_path, self.manifest)
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    if backend == "trt":
                        raise RuntimeError("Requested TensorRT backend is unavailable: %s" % exc) from exc
                    self.fallback_reason = str(exc)
                    print("TensorRT unavailable; using offline CPU inference: " + str(exc),
                          file=sys.stderr, flush=True)
            if session is None:
                session = self._cpu_session()
        self.session = session
        self.backend = getattr(session, "backend_name", "ONNX Runtime CPU")
        self._validate_session()

    def _cpu_session(self) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is missing. Run the package's offline setup script "
                "with Python 3.8, then launch using that environment."
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(self.model_path), sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if not isinstance(manifest, dict):
            raise ValueError("Model manifest must be a JSON object")
        if manifest.get("preprocessing") != "official_lower_bound":
            raise ValueError("Manifest preprocessing must be 'official_lower_bound'")
        if manifest.get("units") not in ("metres", "relative_inverse_depth"):
            raise ValueError("Manifest units must be 'metres' or 'relative_inverse_depth'")
        for key in ("input_name", "output_name"):
            if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                raise ValueError("Manifest %s must be a nonempty string" % key)
        shape = manifest.get("input_shape")
        if not isinstance(shape, list) or len(shape) != 4:
            raise ValueError("Manifest input_shape must be [1, 3, height, width]")
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape):
            raise ValueError("Manifest input_shape must contain positive integers")
        if shape[:2] != [1, 3] or shape[2] % 14 or shape[3] % 14:
            raise ValueError("Manifest input_shape must be [1, 3, H, W] with patch-14 dimensions")
        size = manifest.get("input_size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("Manifest input_size must be a positive integer")
        if min(shape[2:]) < size:
            raise ValueError("Manifest shape is smaller than the preprocessing lower bound")
        if official_resize_shape(shape[3], shape[2], size) != (shape[3], shape[2]):
            raise ValueError("Manifest shape disagrees with the official resize rule")
        digest = manifest.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("Manifest sha256 must be a 64-digit hexadecimal digest")
        if "max_depth_m" in manifest:
            maximum = manifest["max_depth_m"]
            if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
                raise ValueError("Manifest max_depth_m must be a positive finite number")
            if not np.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("Manifest max_depth_m must be a positive finite number")
            if manifest["units"] != "metres":
                raise ValueError("Manifest max_depth_m is meaningful only for metre output")

    def _verify_model_hash(self) -> None:
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.manifest["sha256"].lower():
            raise ValueError("ONNX model checksum differs from manifest; unpack a complete package")

    def _validate_session(self) -> None:
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != self.input_name:
            raise ValueError("ONNX input names disagree with the model manifest")
        if tuple(inputs[0].shape) != self.input_shape or inputs[0].type != "tensor(float)":
            raise ValueError("ONNX input must match the manifest's fixed float32 NCHW shape")
        outputs = [node for node in self.session.get_outputs() if node.name == self.output_name]
        if len(outputs) != 1 or outputs[0].type != "tensor(float)":
            raise ValueError("ONNX output name/type disagree with the model manifest")
        output_shape = tuple(outputs[0].shape)
        height, width = self.input_shape[2:]
        if output_shape not in ((1, height, width), (1, 1, height, width)):
            raise ValueError("ONNX output must be a fixed depth map at the exported resolution")

    def prepare_input(self, bgr_image: np.ndarray) -> np.ndarray:
        """RGB / ImageNet / lower-bound resize, yielding contiguous float32 NCHW."""
        if not isinstance(bgr_image, np.ndarray) or bgr_image.dtype != np.uint8:
            raise ValueError("Depth input must be a uint8 BGR image")
        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3 or min(bgr_image.shape[:2]) < 2:
            raise ValueError("Depth input must have shape [height, width, 3]")
        height, width = bgr_image.shape[:2]
        resized_width, resized_height = official_resize_shape(width, height, self.input_size)
        expected_height, expected_width = self.input_shape[2:]
        if (resized_height, resized_width) != (expected_height, expected_width):
            raise ValueError(
                "Camera aspect/orientation does not match this ONNX export: %dx%d would "
                "resize to %dx%d; model needs %dx%d. Use the package's configured "
                "portrait rotation and camera size, or export a matching model."
                % (width, height, resized_width, resized_height, expected_width, expected_height)
            )
        # The official implementation converts uint8 to float64 before resizing,
        # then casts to float32 only after ImageNet normalization.
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) / 255.0
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
        rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)

    def infer(self, bgr_image: np.ndarray) -> DepthResult:
        tensor = self.prepare_input(bgr_image)
        try:
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
        except RuntimeError as exc:
            if self.requested_backend != "auto" or not self.backend.startswith("TensorRT"):
                raise
            self.fallback_reason = str(exc)
            print("TensorRT inference failed; switching to offline CPU: " + str(exc),
                  file=sys.stderr, flush=True)
            close = getattr(self.session, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
            self.session = self._cpu_session()
            self.backend = "ONNX Runtime CPU"
            self._validate_session()
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise RuntimeError("Depth backend returned an unexpected number of outputs")
        native = np.asarray(outputs[0])
        height, width = self.input_shape[2:]
        if native.shape == (1, height, width):
            native = native[0]
        elif native.shape == (1, 1, height, width):
            native = native[0, 0]
        else:
            raise RuntimeError("Depth backend returned a shape different from the ONNX manifest")
        if not np.issubdtype(native.dtype, np.floating):
            raise RuntimeError("Depth backend returned a non-floating depth map")
        native = native.astype(np.float32, copy=False)
        valid = np.isfinite(native) & (native > 0.0)
        if self.max_depth_m is not None:
            valid &= native <= float(self.max_depth_m) * (1.0 + 1e-5)
        visualization, valid = _resize_align_corners(native, valid, bgr_image.shape[:2])
        depth_m = visualization if self.metric else np.full(visualization.shape, np.nan, np.float32)
        return DepthResult(
            depth_m=depth_m,
            valid_mask=valid,
            visualization_depth=visualization,
            metric=self.metric,
            backend=self.backend,
        )

    def close(self) -> None:
        close = getattr(self.session, "close", None)
        if close is not None:
            close()
