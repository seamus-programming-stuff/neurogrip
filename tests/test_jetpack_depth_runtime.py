"""Offline runtime contract tests; no model download, CUDA or ONNX Runtime needed."""
import ast
import ctypes
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from jetpack_single.depth_runtime import DepthRuntime, _CudaRuntime, _TensorRTSession, official_resize_shape


class FakeSession:
    def __init__(self, shape=(1, 3, 28, 14), output=None):
        self.shape = shape
        self.output = np.full((1, shape[2], shape[3]), 2.0, np.float32) if output is None else output
        self.input_nodes = [SimpleNamespace(name="image", shape=list(shape), type="tensor(float)")]
        self.output_nodes = [SimpleNamespace(name="depth", shape=list(self.output.shape), type="tensor(float)")]
        self.calls = []

    def get_inputs(self):
        return self.input_nodes

    def get_outputs(self):
        return self.output_nodes

    def run(self, names, feeds):
        self.calls.append((names, feeds))
        return [self.output]


class FakeCuda:
    """Device buffers are byte arrays, so pointer order and copy sizes matter."""
    def __init__(self):
        self.driver_version = 11040
        self.device_uuid = "a" * 32
        self.buffers = {}
        self.freed = []
        self.allocated = []
        self.allocation_failure = None
        self.activations = 0

    def activate(self):
        self.activations += 1

    def malloc(self, size):
        if self.allocation_failure == len(self.allocated):
            raise RuntimeError("simulated CUDA allocation failure")
        pointer = 0x100000000 + 1024 * len(self.allocated)
        self.allocated.append(pointer)
        self.buffers[pointer] = np.empty(size, np.uint8)
        return pointer

    def free(self, pointer):
        self.freed.append(pointer)
        del self.buffers[pointer]

    def host_to_device(self, pointer, array):
        self.buffers[pointer][:] = np.frombuffer(array.tobytes(), dtype=np.uint8)

    def device_to_host(self, array, pointer):
        array[:] = self.buffers[pointer].view(np.float32).reshape(array.shape)


class FakeTensorRT:
    """Official 8.5 API surface with output binding deliberately before input."""
    __version__ = "8.5.2.2"
    float32 = "float32"
    NetworkDefinitionCreationFlag = SimpleNamespace(EXPLICIT_BATCH=0)
    MemoryPoolType = SimpleNamespace(WORKSPACE="workspace")
    BuilderFlag = SimpleNamespace(FP16="fp16", TF32="tf32")
    TensorFormat = SimpleNamespace(LINEAR="linear")

    class Logger:
        WARNING = 2

        def __init__(self, severity):
            self.severity = severity

    def __init__(self, cuda):
        self.cuda = cuda
        self.builds = 0
        self.parses = 0
        self.deserialize_calls = 0
        self.parse_success = True
        self.binding_dtype = "float32"
        self.workspace = None
        self.cleared_flags = []
        self.network_flags = None
        self.last_context = None

    def Runtime(self, logger):
        owner = self

        class Runtime:
            def deserialize_cuda_engine(self, serialized):
                owner.deserialize_calls += 1
                if serialized != b"a valid simulated TensorRT engine":
                    return None
                return owner.engine()

        return Runtime()

    def Builder(self, logger):
        owner = self

        class Config:
            def set_memory_pool_limit(self, pool, size):
                owner.workspace = (pool, size)

            def clear_flag(self, flag):
                owner.cleared_flags.append(flag)

        class Builder:
            def create_network(self, flags):
                owner.network_flags = flags
                return object()

            def create_builder_config(self):
                return Config()

            def build_serialized_network(self, network, config):
                owner.builds += 1
                return b"a valid simulated TensorRT engine"

        return Builder()

    def OnnxParser(self, network, logger):
        owner = self

        class Parser:
            num_errors = 1

            def parse(self, data):
                owner.parses += 1
                return owner.parse_success

            def get_error(self, index):
                return "simulated unsupported cubic Resize"

        return Parser()

    def engine(self):
        owner = self

        class Context:
            def execute_v2(self, bindings):
                # An independently defined network: sum RGB channels per pixel.
                owner.last_context = self
                self.bindings = list(bindings)
                tensor = owner.cuda.buffers[bindings[1]].view(np.float32).reshape(1, 3, 28, 14)
                depth = (5.0 + tensor.sum(axis=1) * 0.1).astype(np.float32)
                owner.cuda.buffers[bindings[0]][:] = np.frombuffer(depth.tobytes(), np.uint8)
                return True

        class Engine:
            has_implicit_batch_dimension = False
            num_bindings = 2

            def get_binding_index(self, name):
                return {"depth": 0, "image": 1}.get(name, -1)

            def binding_is_input(self, index):
                return index == 1

            def get_binding_shape(self, index):
                return (1, 3, 28, 14) if index == 1 else (1, 28, 14)

            def get_binding_dtype(self, index):
                return owner.binding_dtype

            def is_shape_binding(self, index):
                return False

            def is_execution_binding(self, index):
                return True

            def get_binding_format(self, index):
                return owner.TensorFormat.LINEAR

            def create_execution_context(self):
                return Context()

        return Engine()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "depth.onnx"
        self.model.write_bytes(b"fake model bytes used only with an injected fake session")
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {
            "input_name": "image", "output_name": "depth", "input_shape": [1, 3, 28, 14],
            "input_size": 14, "preprocessing": "official_lower_bound", "units": "metres",
            "sha256": hashlib.sha256(self.model.read_bytes()).hexdigest(),
            "model_id": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
            "revision": "test", "license": "Apache-2.0", "max_depth_m": 20.0,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def runtime(self, session=None, changes=None):
        manifest = dict(self.manifest)
        if changes:
            manifest.update(changes)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return DepthRuntime(self.model, self.manifest_path, session=session or FakeSession())

    def test_original_portrait_resize_and_lower_bound_rounding(self):
        self.assertEqual(official_resize_shape(540, 960, 252), (252, 448))
        self.assertEqual(official_resize_shape(960, 540, 252), (448, 252))
        # Lower bounds that are not patch multiples round upward when necessary.
        self.assertEqual(official_resize_shape(100, 100, 250), (252, 252))
        self.assertEqual(official_resize_shape(100, 100, 255), (266, 266))
        session = FakeSession((1, 3, 448, 252))
        runtime = self.runtime(session, {"input_shape": [1, 3, 448, 252], "input_size": 252})
        result = runtime.infer(np.zeros((960, 540, 3), np.uint8))
        self.assertEqual(result.depth_m.shape, (960, 540))
        self.assertTrue(result.metric)
        np.testing.assert_array_equal(result.depth_m, 2.0)

    def test_bgr_conversion_imagenet_channels_and_session_contract(self):
        session = FakeSession()
        runtime = self.runtime(session)
        image = np.empty((140, 70, 3), np.uint8)
        image[:] = (17, 101, 233)
        result = runtime.infer(image)
        names, feeds = session.calls[0]
        self.assertEqual(names, ["depth"])
        self.assertEqual(list(feeds), ["image"])
        tensor = feeds["image"]
        self.assertEqual(tensor.shape, (1, 3, 28, 14))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        expected_rgb = np.array([233, 101, 17]) / 255.0
        expected = (expected_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        np.testing.assert_allclose(tensor[0, :, 10, 6], expected, atol=1e-6)
        self.assertEqual(result.backend, "ONNX Runtime CPU")
        self.assertTrue(result.valid_mask.all())
        self.assertEqual(result.depth_m.dtype, np.float32)
        self.assertFalse(hasattr(result, "confidence"))

    def test_bilinear_align_corners_preserves_independent_affine_plane(self):
        y, x = np.indices((28, 14), dtype=np.float32)
        plane = 0.5 + 0.1 * y + 0.2 * x
        runtime = self.runtime(FakeSession(output=plane[None]))
        result = runtime.infer(np.zeros((84, 42, 3), np.uint8))
        out_y, out_x = np.indices((84, 42), dtype=np.float64)
        expected = 0.5 + 0.1 * out_y * 27.0 / 83.0 + 0.2 * out_x * 13.0 / 41.0
        np.testing.assert_allclose(result.depth_m, expected, atol=6e-7)
        self.assertAlmostEqual(float(result.depth_m[0, 0]), 0.5)
        self.assertAlmostEqual(float(result.depth_m[-1, -1]), 5.8, places=6)
        self.assertTrue(result.valid_mask.all())

    def test_negative_zero_nan_infinity_and_above_metric_cap_are_invalid(self):
        output = np.full((1, 28, 14), 2.0, np.float32)
        for column, bad in enumerate((-1.0, 0.0, np.nan, np.inf, 20.1)):
            output[0, 10, column] = bad
        runtime = self.runtime(FakeSession(output=output))
        result = runtime.infer(np.zeros((28, 14, 3), np.uint8))
        self.assertFalse(result.valid_mask[10, :5].any())
        self.assertTrue(np.isnan(result.depth_m[10, :5]).all())
        self.assertTrue(np.isnan(result.visualization_depth[10, :5]).all())
        self.assertTrue(result.valid_mask[11].all())
        self.assertEqual(int(result.valid_mask.sum()), 28 * 14 - 5)

    def test_invalid_interpolation_support_cannot_make_false_valid_depth(self):
        output = np.full((1, 28, 14), 2.0, np.float32)
        output[0, 0, 0] = np.nan
        result = self.runtime(FakeSession(output=output)).infer(np.zeros((56, 28, 3), np.uint8))
        self.assertFalse(result.valid_mask[0, 0])
        self.assertFalse(result.valid_mask[1, 1])
        self.assertTrue(result.valid_mask[-1, -1])
        self.assertTrue(result.valid_mask[0, -1])
        self.assertTrue(np.isnan(result.depth_m[~result.valid_mask]).all())
        np.testing.assert_array_equal(result.depth_m[result.valid_mask], 2.0)

    def test_relative_model_never_claims_metres(self):
        manifest = dict(self.manifest)
        manifest.pop("max_depth_m")
        manifest["units"] = "relative_inverse_depth"
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        runtime = DepthRuntime(self.model, self.manifest_path, session=FakeSession())
        result = runtime.infer(np.zeros((56, 28, 3), np.uint8))
        self.assertFalse(result.metric)
        self.assertTrue(np.isnan(result.depth_m).all())
        self.assertTrue(result.valid_mask.all())
        np.testing.assert_array_equal(result.visualization_depth, 2.0)

    def test_aspect_and_orientation_mismatch_fail_before_model_execution(self):
        session = FakeSession()
        runtime = self.runtime(session)
        for shape in ((70, 140, 3), (70, 70, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "aspect/orientation"):
                runtime.infer(np.zeros(shape, np.uint8))
        self.assertEqual(session.calls, [])
        for image in (np.zeros((28, 14, 3), np.float32), np.zeros((28, 14), np.uint8)):
            with self.assertRaises(ValueError):
                runtime.infer(image)

    def test_manifest_and_checkpoint_integrity(self):
        for changes in (
            {"preprocessing": "warp"}, {"units": "mm"}, {"input_name": ""},
            {"input_shape": [1, 3, 27, 14]}, {"input_shape": [2, 3, 28, 14]},
            {"input_shape": [True, 3, 28, 14]}, {"input_size": 30},
            {"input_shape": [1, 3, 56, 28]}, {"sha256": "invalid"},
            {"max_depth_m": -2}, {"max_depth_m": float("inf")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.runtime(changes=changes)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.model.write_bytes(b"corrupted model")
        with self.assertRaisesRegex(ValueError, "checksum"):
            DepthRuntime(self.model, self.manifest_path, session=FakeSession())

    def test_session_metadata_cannot_silently_disagree(self):
        for wrong in ("name", "shape", "type", "output"):
            session = FakeSession()
            if wrong == "name":
                session.input_nodes[0].name = "other"
            elif wrong == "shape":
                session.input_nodes[0].shape = [1, 3, "height", "width"]
            elif wrong == "type":
                session.input_nodes[0].type = "tensor(float16)"
            else:
                session.output_nodes[0].shape = [1, 14, 28]
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                self.runtime(session)
        session = FakeSession()
        runtime = self.runtime(session)
        session.output = np.zeros((1, 14, 28), np.float32)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            runtime.infer(np.zeros((28, 14, 3), np.uint8))

    def test_bchw_output_is_supported_without_axis_mixing(self):
        output = np.full((1, 1, 28, 14), 3.0, np.float32)
        runtime = self.runtime(FakeSession(output=output))
        result = runtime.infer(np.zeros((56, 28, 3), np.uint8))
        np.testing.assert_array_equal(result.depth_m, 3.0)

    def test_python38_syntax_and_missing_optional_inference_dependency(self):
        path = Path(__file__).resolve().parents[1] / "jetpack_single" / "depth_runtime.py"
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 8))
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with self.assertRaisesRegex(RuntimeError, "offline setup"):
                DepthRuntime(self.model, self.manifest_path)

    def test_tensor_rt_pointer_order_real_copy_contract_and_precision(self):
        cuda = FakeCuda()
        trt = FakeTensorRT(cuda)
        with patch("sys.stderr", new=io.StringIO()):
            session = _TensorRTSession(self.model, self.manifest, trt_module=trt, cuda=cuda)
        runtime = self.runtime(session)
        image = np.empty((56, 28, 3), np.uint8)
        image[:] = (17, 101, 233)
        tensor = runtime.prepare_input(image)
        independent_expected = 5.0 + 0.1 * tensor[0, :, 0, 0].sum()
        result = runtime.infer(image)
        np.testing.assert_allclose(result.depth_m, independent_expected, atol=1e-6)
        self.assertEqual(result.backend, "TensorRT 8.5.2.2 FP32")
        self.assertEqual(trt.last_context.bindings[0], session.bindings[session.output_index])
        self.assertEqual(trt.last_context.bindings[1], session.bindings[session.input_index])
        self.assertTrue(all(pointer >= 0x100000000 for pointer in session.bindings))
        self.assertEqual(trt.network_flags, 1)
        self.assertEqual(trt.workspace, ("workspace", 512 * 1024 * 1024))
        self.assertCountEqual(trt.cleared_flags, ["fp16", "tf32"])
        session.close()
        session.close()
        self.assertCountEqual(cuda.freed, cuda.allocated)
        self.assertFalse(cuda.buffers)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.run(["depth"], {"image": tensor})

    def test_tensor_rt_cache_identity_checksum_and_gpu_change(self):
        cuda = FakeCuda()
        trt = FakeTensorRT(cuda)
        with patch("sys.stderr", new=io.StringIO()):
            first = _TensorRTSession(self.model, self.manifest, trt_module=trt, cuda=cuda)
        first.close()
        self.assertEqual(trt.builds, 1)
        second = _TensorRTSession(self.model, self.manifest, trt_module=trt, cuda=cuda)
        second.close()
        self.assertEqual(trt.builds, 1)
        plan = next((self.root / ".engine_cache").glob("*.plan"))
        plan.write_bytes(b"damaged plan bytes")
        with patch("sys.stderr", new=io.StringIO()):
            rebuilt = _TensorRTSession(self.model, self.manifest, trt_module=trt, cuda=cuda)
        rebuilt.close()
        self.assertEqual(trt.builds, 2)
        cuda.device_uuid = "b" * 32
        with patch("sys.stderr", new=io.StringIO()):
            other_gpu = _TensorRTSession(self.model, self.manifest, trt_module=trt, cuda=cuda)
        other_gpu.close()
        self.assertEqual(trt.builds, 3)
        self.assertEqual(len(list((self.root / ".engine_cache").glob("*.plan"))), 2)

    def test_tensor_rt_parse_failure_precision_and_partial_allocation_cleanup(self):
        for failure in ("parse", "dtype", "allocation"):
            with self.subTest(failure=failure):
                cuda = FakeCuda()
                trt = FakeTensorRT(cuda)
                if failure == "parse":
                    trt.parse_success = False
                elif failure == "dtype":
                    trt.binding_dtype = "float16"
                else:
                    cuda.allocation_failure = 1
                # A different model digest gives each failure an independent cache.
                manifest = dict(self.manifest, sha256=hashlib.sha256(failure.encode()).hexdigest())
                with patch("sys.stderr", new=io.StringIO()), self.assertRaises(RuntimeError):
                    _TensorRTSession(self.model, manifest, trt_module=trt, cuda=cuda)
                self.assertCountEqual(cuda.freed, cuda.allocated)
                self.assertFalse(cuda.buffers)

    def test_auto_fallback_on_startup_and_after_gpu_execution_failure(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        cpu = FakeSession()
        with patch("jetpack_single.depth_runtime._TensorRTSession", side_effect=RuntimeError("missing CUDA")), \
             patch.object(DepthRuntime, "_cpu_session", return_value=cpu), \
             patch("sys.stderr", new=io.StringIO()):
            runtime = DepthRuntime(self.model, self.manifest_path, backend="auto")
        self.assertEqual(runtime.backend, "ONNX Runtime CPU")
        self.assertEqual(runtime.fallback_reason, "missing CUDA")
        gpu = FakeSession()
        gpu.backend_name = "TensorRT test FP32"
        gpu.run = lambda names, feeds: (_ for _ in ()).throw(RuntimeError("GPU execution failed"))
        gpu.close = lambda: (_ for _ in ()).throw(RuntimeError("GPU context lost during cleanup"))
        with patch.object(DepthRuntime, "_cpu_session", return_value=cpu), \
             patch("sys.stderr", new=io.StringIO()):
            runtime = DepthRuntime(self.model, self.manifest_path, backend="auto", session=gpu)
            result = runtime.infer(np.zeros((56, 28, 3), np.uint8))
        self.assertEqual(result.backend, "ONNX Runtime CPU")
        self.assertEqual(runtime.fallback_reason, "GPU execution failed")
        np.testing.assert_array_equal(result.depth_m, 2.0)

    def test_default_cpu_never_builds_gpu_and_requested_gpu_never_silently_falls_back(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with patch("jetpack_single.depth_runtime._TensorRTSession") as factory, \
             patch.object(DepthRuntime, "_cpu_session", return_value=FakeSession()):
            DepthRuntime(self.model, self.manifest_path)
            factory.assert_not_called()
        with patch("jetpack_single.depth_runtime._TensorRTSession", side_effect=RuntimeError("missing CUDA")), \
             patch.object(DepthRuntime, "_cpu_session") as cpu:
            with self.assertRaisesRegex(RuntimeError, "Requested TensorRT"):
                DepthRuntime(self.model, self.manifest_path, backend="trt")
            cpu.assert_not_called()

    def test_ctypes_cuda_transfer_preserves_64bit_pointers_and_error_reporting(self):
        allocations, copy_kinds = {}, []

        def pointer_value(pointer):
            return pointer.value if isinstance(pointer, ctypes.c_void_p) else int(pointer)

        def malloc(pointer_pointer, size):
            buffer = ctypes.create_string_buffer(size)
            address = ctypes.addressof(buffer)
            allocations[address] = buffer
            ctypes.cast(pointer_pointer, ctypes.POINTER(ctypes.c_void_p))[0] = address
            return 0

        def copy(destination, source, size, kind):
            copy_kinds.append(kind)
            ctypes.memmove(pointer_value(destination), pointer_value(source), size)
            return 0

        def free(pointer):
            del allocations[pointer_value(pointer)]
            return 0

        cuda = _CudaRuntime.__new__(_CudaRuntime)
        cuda.ctypes = ctypes
        cuda.library = SimpleNamespace(
            cudaMalloc=malloc, cudaMemcpy=copy, cudaFree=free,
            cudaGetErrorString=lambda status: b"simulated transfer failure",
        )
        original = np.arange(57, dtype=np.float32).reshape(3, 19)
        device = cuda.malloc(original.nbytes)
        restored = np.zeros_like(original)
        cuda.host_to_device(device, original)
        cuda.device_to_host(restored, device)
        np.testing.assert_array_equal(restored, original)
        self.assertEqual(copy_kinds, [1, 2])
        cuda.free(device)
        self.assertFalse(allocations)
        with self.assertRaisesRegex(RuntimeError, "simulated transfer failure"):
            cuda._check(2, "cudaMemcpy")


if __name__ == "__main__":
    unittest.main()
