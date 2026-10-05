"""Installer/diagnostic checks without the NEON, root rights, or network."""
import ast
import email
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "jetpack_single"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load("jetpack_check_neon_test", "check_neon.py")
service = load("jetpack_install_service_test", "install_service.py")


class JetPackCheckTests(unittest.TestCase):
    def test_repair_preserves_only_the_fixed_mutable_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {"app.py": b"print('runtime')\n", "config.json": b"{}\n", "strike_config.json": b"{}\n"}
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)
            (root / "SHA256SUMS").write_text("".join(hashlib.sha256(payload).hexdigest() + "  " + name + "\n"
                                                   for name, payload in payloads.items()))
            self.assertEqual(check.verify_bundle(root)["verified_files"], 3)
            (root / "config.json").write_text('{"camera": "calibrated"}')
            (root / "strike_config.json").write_text('{"validated": true}')
            with self.assertRaisesRegex(RuntimeError, "integrity failed"):
                check.verify_bundle(root)
            result = check.verify_bundle(root, repair=True)
            self.assertEqual(result["preserved_mutable_files"], ["config.json", "strike_config.json"])
            self.assertEqual(result["verified_files"], 1)
            (root / "app.py").write_text("corrupted runtime")
            with self.assertRaisesRegex(RuntimeError, "app.py"):
                check.verify_bundle(root, repair=True)

    def test_hash_manifest_rejects_escape_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = hashlib.sha256(b"x").hexdigest()
            (root / "app.py").write_bytes(b"x")
            for manifest in (digest + "  ../outside.py\n", digest + "  app.py\n" + digest + "  app.py\n"):
                (root / "SHA256SUMS").write_text(manifest)
                with self.assertRaises(RuntimeError):
                    check.verify_bundle(root)

    def test_platform_accepts_only_verified_release_and_architecture(self):
        release = "# R35 (release), REVISION: 4.1, GCID: 33958178, BOARD: t186ref"
        result = check.platform_diagnostics((3, 8, 10), "aarch64", "Linux", release)
        self.assertEqual(result["l4t"], "35.4.1")
        for version, machine, system, text in (
                ((3, 6, 9), "aarch64", "Linux", release),
                ((3, 8, 10), "x86_64", "Linux", release),
                ((3, 8, 10), "aarch64", "Windows", release),
                ((3, 8, 10), "aarch64", "Linux", "# R32 (release), REVISION: 7.1"),
                ((3, 8, 10), "aarch64", "Linux", "")):
            with self.subTest(version=version, machine=machine, system=system, text=text):
                with self.assertRaises(RuntimeError):
                    check.platform_diagnostics(version, machine, system, text)

    def test_pip_opencv_or_changed_module_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system = root / "usr/lib/python3/dist-packages/cv2.so"
            venv = root / ".venv"
            self.assertEqual(check.validate_system_cv2(system, venv, system), str(system.resolve()))
            for path in (venv / "lib/python3.8/site-packages/cv2/cv2.so",
                         root / "usr/local/lib/python3.8/site-packages/cv2/cv2.so"):
                with self.assertRaises(RuntimeError):
                    check.validate_system_cv2(path, venv)
            with self.assertRaises(RuntimeError):
                check.validate_system_cv2(system, venv, root / "different/cv2.so")

    def test_camera_sets_both_dimensions_before_any_read_and_releases(self):
        operations = []
        class FakeCapture:
            def set(self, key, value):
                operations.append(("set", key, value))
            def isOpened(self):
                return True
            def read(self):
                operations.append(("read",))
                return True, np.zeros((1080, 1920, 3), np.uint8)
            def release(self):
                operations.append(("release",))
        with patch.object(cv2, "VideoCapture", return_value=FakeCapture()):
            result = check.probe_camera(cv2)
        self.assertEqual(operations[:2], [("set", cv2.CAP_PROP_FRAME_WIDTH, 1920),
                                         ("set", cv2.CAP_PROP_FRAME_HEIGHT, 1080)])
        self.assertEqual(operations[2:], [("read",), ("release",)])
        self.assertEqual(result["output_size"], [540, 960])
        self.assertFalse(result["exposure_timestamp"])

    def test_wrong_camera_mode_is_explicit_and_still_releases(self):
        capture = SimpleNamespace(set=lambda *args: True, isOpened=lambda: True,
                                  read=lambda: (True, np.zeros((480, 640, 3), np.uint8)))
        released = []
        capture.release = lambda: released.append(True)
        with patch.object(cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "Requested 1920x1080"):
                check.probe_camera(cv2)
        self.assertEqual(released, [True])

    def test_no_camera_flag_never_opens_sensor(self):
        with patch.object(check, "platform_diagnostics", return_value={"l4t": "35.4.1"}), \
                patch.object(check, "runtime_diagnostics", return_value={"onnxruntime": "1.16.3"}), \
                patch.object(check, "probe_camera") as camera, patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(check.main(["--no-camera"]), 0)
        camera.assert_not_called()

    def test_onnx_load_failure_is_reported_without_traceback(self):
        output = io.StringIO()
        with patch.object(check, "platform_diagnostics", return_value={}), \
                patch.object(check, "runtime_diagnostics", return_value={}), \
                patch.object(check, "model_diagnostics", side_effect=Exception("unsupported ONNX operator")), \
                patch("sys.stderr", new=output):
            self.assertEqual(check.main(["--no-camera", "--model", "model.onnx"]), 1)
        self.assertIn("unsupported ONNX operator", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())

    def test_scripts_are_linux_lines_and_python38_compatible(self):
        for source in PACKAGE.glob("*.py"):
            ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source), feature_version=(3, 8))
        for name in ("install.sh", "run.sh", "calibrate.sh"):
            self.assertNotIn(b"\r", (PACKAGE / name).read_bytes())


class ServiceInstallTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root / ".venv/bin").mkdir(parents=True)
        for name in (".venv/bin/python", "app.py", "config.json"):
            (root / name).write_text("")
        return root

    def test_generated_unit_uses_exact_user_local_env_and_sigint(self):
        unit = service.service_unit(Path(tempfile.gettempdir()) / "camera folder%")
        self.assertIn("User=adlink\n", unit)
        self.assertIn("KillSignal=SIGINT\n", unit)
        self.assertIn(".venv", unit)
        self.assertIn("app.py", unit)
        self.assertIn("--config", unit)
        self.assertIn("%%", unit)
        with self.assertRaises(ValueError):
            service.quote_systemd("bad\nExecStart=oops")

    def test_default_install_does_not_start_or_stop_capture(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=1)
        with tempfile.TemporaryDirectory() as directory:
            service.install_service(self.fixture(directory), runner=runner)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][-1], "/etc/systemd/system/neurogrip-depth.service")
        self.assertEqual(calls[1], ["sudo", "systemctl", "daemon-reload"])
        self.assertFalse(any(word in command for command in calls for word in ("stop", "disable", "restart", "enable")))

    def test_start_refuses_other_camera_owner_without_mutation(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0 if "fuser" in command[0] else 1)
        with tempfile.TemporaryDirectory() as directory, patch.object(service.shutil, "which", return_value="fuser"):
            with self.assertRaisesRegex(RuntimeError, "already owned"):
                service.install_service(self.fixture(directory), start=True, runner=runner)
        self.assertFalse(any(command[0] == "sudo" for command in calls))

    def test_start_changes_only_the_named_unit(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(service.shutil, "which", return_value=None):
            service.install_service(self.fixture(directory), start=True, runner=runner)
        changes = [command for command in calls if "enable" in command or "restart" in command]
        self.assertEqual(changes, [["sudo", "systemctl", "enable", service.UNIT],
                                  ["sudo", "systemctl", "restart", service.UNIT]])
        self.assertFalse(any("stop" in command or "disable" in command for command in calls))


class OfflineBootstrapTests(unittest.TestCase):
    def test_offline_wheel_metadata_closes_python38_aarch64_dependencies(self):
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.tags import compatible_tags, cpython_tags
        from packaging.utils import canonicalize_name, parse_wheel_filename
        target = {"implementation_name": "cpython", "implementation_version": "3.8.10",
                  "os_name": "posix", "platform_machine": "aarch64", "platform_system": "Linux",
                  "platform_python_implementation": "CPython", "platform_release": "5.10.104-tegra",
                  "platform_version": "", "python_version": "3.8", "python_full_version": "3.8.10",
                  "sys_platform": "linux", "extra": ""}
        pins = {}
        for line in (PACKAGE / "requirements-offline.txt").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                requirement = Requirement(line)
                pins[canonicalize_name(requirement.name)] = next(
                    item.version for item in requirement.specifier if item.operator == "==")
        platforms = ["manylinux_2_17_aarch64", "manylinux2014_aarch64"]
        supported = set(cpython_tags((3, 8), ["cp38"], platforms))
        supported.update(compatible_tags((3, 8), "cp38", platforms))
        found = set()
        for wheel in sorted((PACKAGE / "wheels").glob("*.whl")):
            name, version, build, tags = parse_wheel_filename(wheel.name)
            with self.subTest(wheel=wheel.name):
                found.add(name)
                self.assertEqual(str(version), pins.get(name))
                self.assertTrue(tags & supported, "Wheel is incompatible with CPython 3.8 aarch64")
                with zipfile.ZipFile(wheel) as archive:
                    metadata_path = next(path for path in archive.namelist() if path.endswith(".dist-info/METADATA"))
                    metadata = email.message_from_bytes(archive.read(metadata_path))
                self.assertTrue(SpecifierSet(metadata.get("Requires-Python", "")).contains("3.8.10"))
                for raw_dependency in metadata.get_all("Requires-Dist", []):
                    dependency = Requirement(raw_dependency)
                    if dependency.marker and not dependency.marker.evaluate(target):
                        continue
                    dependency_name = canonicalize_name(dependency.name)
                    self.assertIn(dependency_name, pins, raw_dependency)
                    self.assertTrue(dependency.specifier.contains(pins[dependency_name]), raw_dependency)
        self.assertEqual(found, set(pins), "Every pinned dependency needs its offline wheel")

    def test_bundled_virtualenv_and_pip_bootstrap_without_ensurepip(self):
        versions = ["virtualenv-20.26.6", "distlib-0.3.9", "filelock-3.13.4", "platformdirs-4.3.6",
                    "typing_extensions-4.12.2", "importlib_metadata-8.5.0", "zipp-3.20.2"]
        wheels = []
        for version in versions:
            matches = list((PACKAGE / "wheels").glob(version + "-*.whl"))
            if len(matches) != 1:
                self.skipTest("Offline bootstrap wheels not present yet")
            wheels.append(matches[0])
        pip_wheel = PACKAGE / "wheels/pip-23.3.2-py3-none-any.whl"
        self.assertTrue(pip_wheel.is_file())
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path) for path in wheels),
                           PYTHONNOUSERSITE="1", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "venv"
            process = subprocess.run([sys.executable, "-m", "virtualenv", "--python", sys.executable,
                "--system-site-packages", "--no-seed", "--no-download", "--no-periodic-update",
                "--app-data", str(Path(directory) / "cache"), str(destination)],
                env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertIn("include-system-site-packages = true", (destination / "pyvenv.cfg").read_text())
            python = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            environment["PYTHONPATH"] = str(pip_wheel)
            process = subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--no-index",
                "--ignore-installed", "--no-deps", str(pip_wheel)], env=environment,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            environment.pop("PYTHONPATH")
            process = subprocess.run([str(python), "-m", "pip", "--version"], env=environment,
                capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertIn("pip 23.3.2", process.stdout)
            self.assertIn(str(destination).lower(), process.stdout.lower())


if __name__ == "__main__":
    unittest.main()
