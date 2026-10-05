"""Validate the pinned JetPack 5.1.2 runtime without modifying system Python."""
import argparse
import importlib
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import sys
import time


EXPECTED_PYTHON = (3, 8)
EXPECTED_L4T = "35.4.1"
EXPECTED_NUMPY = "1.24.4"
EXPECTED_ORT = "1.16.3"
REPAIR_MUTABLE_FILES = frozenset(("config.json", "strike_config.json"))


def verify_bundle(root, repair=False):
    """Verify every release entry, optionally preserving two user settings files."""
    root = Path(root).resolve()
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    seen, preserved = set(), []
    verified = 0
    for line in lines:
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if match is None:
            raise RuntimeError("Malformed SHA256SUMS entry")
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or PureWindowsPath(name).is_absolute() or ".." in relative.parts:
            raise RuntimeError("Hash manifest path must remain inside the bundle: " + name)
        name = relative.as_posix()
        if name in seen:
            raise RuntimeError("Duplicate SHA256SUMS entry: " + name)
        seen.add(name)
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise RuntimeError("Hash manifest path escaped the bundle: " + name)
        if not path.is_file():
            raise RuntimeError("Required bundle file is missing: " + name)
        if repair and name in REPAIR_MUTABLE_FILES:
            preserved.append(name)
            continue
        actual = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                actual.update(block)
        if actual.hexdigest() != digest.lower():
            raise RuntimeError("Bundle integrity failed: " + name + "; extract the original download again")
        verified += 1
    if not seen or not verified:
        raise RuntimeError("SHA256SUMS contains no immutable release files")
    return {"verified_files": verified, "preserved_mutable_files": sorted(preserved), "repair": repair}


def parse_l4t(text):
    match = re.search(r"\bR(\d+)\b.*?\bREVISION:\s*([\d.]+)", text, flags=re.S)
    return None if match is None else match.group(1) + "." + match.group(2)


def platform_diagnostics(python_version=None, machine=None, operating_system=None, l4t_text=None):
    version = tuple(sys.version_info[:3]) if python_version is None else tuple(python_version)
    machine = platform.machine() if machine is None else machine
    operating_system = platform.system() if operating_system is None else operating_system
    if l4t_text is None:
        try:
            l4t_text = Path("/etc/nv_tegra_release").read_text()
        except OSError:
            l4t_text = ""
    release = parse_l4t(l4t_text)
    result = {"python": ".".join(str(part) for part in version), "machine": machine,
              "system": operating_system, "l4t": release}
    problems = []
    if version[:2] != EXPECTED_PYTHON:
        problems.append("system Python 3.8 is required; found " + result["python"])
    if machine.lower() != "aarch64" or operating_system != "Linux":
        problems.append("Linux aarch64 NEON is required; found %s %s" % (operating_system, machine))
    if release != EXPECTED_L4T:
        problems.append("JetPack 5.1.2 / L4T R35.4.1 is required; found " + str(release or "unknown"))
    if problems:
        raise RuntimeError("; ".join(problems))
    return result


def validate_system_cv2(cv2_path, venv_prefix=None, expected_path=None):
    location = Path(cv2_path).resolve()
    if "site-packages" in location.parts:
        raise RuntimeError("OpenCV comes from pip site-packages, not the NEON system build: " + str(location))
    if venv_prefix is not None:
        try:
            location.relative_to(Path(venv_prefix).resolve())
        except ValueError:
            pass
        else:
            raise RuntimeError("OpenCV must remain outside .venv: " + str(location))
    if expected_path is not None and location != Path(expected_path).resolve():
        raise RuntimeError("OpenCV changed from the recorded system module: " + str(location))
    return str(location)


def system_opencv_record(path):
    cv2 = importlib.import_module("cv2")
    location = validate_system_cv2(cv2.__file__)
    result = {"cv2_path": location, "opencv_version": cv2.__version__}
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)
    return result


def runtime_diagnostics(expected_cv2=None, modules=None):
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        raise RuntimeError("Use .venv/bin/python after bash install.sh; do not install into system Python")
    modules = modules or {name: importlib.import_module(name) for name in ("cv2", "numpy", "onnxruntime")}
    cv2, numpy, ort = (modules[name] for name in ("cv2", "numpy", "onnxruntime"))
    expected = None
    if expected_cv2 is not None:
        expected = json.loads(Path(expected_cv2).read_text())["cv2_path"]
    location = validate_system_cv2(cv2.__file__, sys.prefix, expected)
    if numpy.__version__ != EXPECTED_NUMPY or ort.__version__ != EXPECTED_ORT:
        raise RuntimeError("Expected numpy %s and onnxruntime %s; found %s and %s" %
                           (EXPECTED_NUMPY, EXPECTED_ORT, numpy.__version__, ort.__version__))
    # Exercises the system cv2 / bundled NumPy boundary, catching ABI trouble.
    small = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if gray.shape != (8, 8):
        raise RuntimeError("System OpenCV / NumPy ABI smoke check failed")
    providers = ort.get_available_providers()
    if "CPUExecutionProvider" not in providers:
        raise RuntimeError("The bundled ONNX Runtime CPU provider is unavailable")
    return {"opencv": cv2.__version__, "cv2_path": location,
            "numpy": numpy.__version__, "onnxruntime": ort.__version__,
            "onnx_providers": providers, "runtime": "ONNX Runtime CPU"}


def model_diagnostics(path, infer_smoke=False, ort=None):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("Bundled model is missing: " + str(path))
    ort = ort or importlib.import_module("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    result = {"model": str(path), "model_inputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in inputs],
              "model_outputs": [item.name for item in session.get_outputs()],
              "model_provider": session.get_providers()}
    if infer_smoke:
        import numpy as np
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or any(type(size) is not int or size < 1 for size in inputs[0].shape):
            raise RuntimeError("Smoke inference expects one fixed-shape float32 image input")
        sample = np.zeros(inputs[0].shape, dtype=np.float32)
        started = time.monotonic()
        outputs = session.run(None, {inputs[0].name: sample})
        elapsed = (time.monotonic() - started) * 1000
        if not outputs or not np.isfinite(outputs[0]).all():
            raise RuntimeError("Smoke inference returned nonfinite or missing depth")
        result["smoke_inference_ms"] = elapsed
        result["smoke_output_shape"] = list(outputs[0].shape)
    return result


def probe_camera(cv2=None):
    cv2 = cv2 or importlib.import_module("cv2")
    capture = cv2.VideoCapture(0, cv2.CAP_V4L)
    try:
        # Required by the verified NEON: both settings precede EVERY first read.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not capture.isOpened():
            raise RuntimeError("Cannot open /dev/video0; check permissions and its current capture owner")
        started = time.monotonic()
        ok, image = capture.read()
        completed = time.monotonic()
        if not ok or image is None:
            raise RuntimeError("No V4L2 frame at 1920x1080; stop the existing capture owner and verify the sensor mode")
        if tuple(image.shape[:2]) != (1080, 1920):
            raise RuntimeError("Requested 1920x1080 but camera returned %dx%d; verify capture mode" % (image.shape[1], image.shape[0]))
        upright = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        output = cv2.resize(upright, (540, 960), interpolation=cv2.INTER_AREA)
        return {"capture_size": [1920, 1080], "output_size": [output.shape[1], output.shape[0]],
                "rotation": "90_clockwise", "first_read_ms": (completed - started) * 1000,
                "timestamp_kind": "host_read_completion", "exposure_timestamp": False}
    finally:
        capture.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-camera", action="store_true", help="Never open or read the sensor")
    parser.add_argument("--platform-only", action="store_true", help="Check OS/Python before installing runtime wheels")
    parser.add_argument("--record-system-cv2", help="Record the system module before creating .venv")
    parser.add_argument("--expected-cv2", help="Compare the runtime module with this recorded system JSON")
    parser.add_argument("--model", help="Load a local ONNX model to check its operators/provider")
    parser.add_argument("--infer-smoke", action="store_true", help="Run one zero-image model inference")
    parser.add_argument("--verify-bundle", metavar="DIRECTORY", help="Verify SHA256SUMS before platform or package imports")
    parser.add_argument("--repair", action="store_true", help="With --verify-bundle, preserve only config.json and strike_config.json")
    args = parser.parse_args(argv)
    if args.repair and not args.verify_bundle:
        parser.error("--repair requires --verify-bundle")
    try:
        if args.verify_bundle:
            print(json.dumps(verify_bundle(args.verify_bundle, args.repair), indent=2))
            return 0
        result = {"platform": platform_diagnostics()}
        if args.record_system_cv2:
            result["system_opencv"] = system_opencv_record(args.record_system_cv2)
        if not args.platform_only:
            result["runtime"] = runtime_diagnostics(args.expected_cv2)
            if args.model:
                result["model"] = model_diagnostics(args.model, args.infer_smoke)
            elif args.infer_smoke:
                raise RuntimeError("--infer-smoke requires --model")
            if not args.no_camera:
                result["camera"] = probe_camera()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print("NEON check failed: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
