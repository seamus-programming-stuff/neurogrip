"""Explicitly install only neurogrip-depth.service; --start opts into boot capture."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


UNIT = "neurogrip-depth.service"


def quote_systemd(value, command=False):
    value = str(value)
    if any(character in value for character in ("\n", "\r", "\0")):
        raise ValueError("Service paths cannot contain control characters")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def service_unit(root, video_group=True):
    root = Path(root).resolve()
    return ("[Unit]\nDescription=Neurogrip single-camera local depth feed\nAfter=network.target\n\n"
            "[Service]\nType=simple\nUser=adlink\n" +
            ("SupplementaryGroups=video\n" if video_group else "") +
            "WorkingDirectory=" + quote_systemd(root) + "\n"
            "ExecStart=" + quote_systemd(root / ".venv/bin/python", command=True) + " -u " +
            quote_systemd(root / "app.py", command=True) + " --config " +
            quote_systemd(root / "config.json", command=True) + "\n"
            "Environment=PYTHONNOUSERSITE=1\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n"
            "Environment=PYTHONUNBUFFERED=1\nRestart=on-failure\nRestartSec=3\n"
            "KillSignal=SIGINT\nTimeoutStopSec=15\nNoNewPrivileges=true\nPrivateTmp=true\n\n"
            "[Install]\nWantedBy=multi-user.target\n")


def ensure_login_user():
    if platform.system() != "Linux":
        raise RuntimeError("Run this helper on the NEON after bash install.sh")
    import pwd
    if os.geteuid() == 0 or pwd.getpwuid(os.getuid()).pw_name != "adlink":
        raise RuntimeError("Run as the adlink login user without sudo; only unit installation requests sudo")


def install_service(root, start=False, runner=None, video_group=True):
    root = Path(root).resolve()
    runner = runner or subprocess.run
    for name in (".venv/bin/python", "app.py", "config.json"):
        if not (root / name).is_file():
            raise RuntimeError("Run bash install.sh first; package file missing: " + name)
    if start:
        active = runner(["systemctl", "is-active", "--quiet", UNIT], check=False).returncode == 0
        fuser = shutil.which("fuser")
        if not active and fuser and runner([fuser, "-v", "/dev/video0"], check=False).returncode == 0:
            raise RuntimeError("/dev/video0 is already owned by another capture process. Stop its owner first; no other service was changed")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".service", delete=False, encoding="utf-8") as handle:
            handle.write(service_unit(root, video_group))
            temporary = handle.name
        runner(["sudo", "install", "-m", "0644", temporary, "/etc/systemd/system/" + UNIT], check=True)
        runner(["sudo", "systemctl", "daemon-reload"], check=True)
        if start:
            runner(["sudo", "systemctl", "enable", UNIT], check=True)
            runner(["sudo", "systemctl", "restart", UNIT], check=True)
            runner(["systemctl", "status", "--no-pager", UNIT], check=False)
    finally:
        if temporary:
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help="Enable boot startup and restart this exact service")
    args = parser.parse_args(argv)
    try:
        ensure_login_user()
        import grp
        try:
            grp.getgrnam("video")
            video_group = True
        except KeyError:
            video_group = False
        install_service(Path(__file__).resolve().parent, args.start, video_group=video_group)
        print("Installed " + UNIT + (" and enabled boot capture." if args.start else "; capture has not been started."))
        print("Logs: journalctl -u %s -n 40 --no-pager" % UNIT)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, "Service installation failed: %s\n" % error)


if __name__ == "__main__":
    raise SystemExit(main())
