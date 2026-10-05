"""Run on either NEON: install the raw capture helper as a systemd service.

Python3.6+ and system OpenCV only. No firmware, JetPack or pip changes. Existing
camera programs must be stopped by their owner before enabling this service.
"""
import argparse
import getpass
import grp
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import tempfile


UNIT = "neurogrip-camera.service"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-width", type=int, default=540)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--no-rotate", action="store_true")
    args = parser.parse_args()
    if not 160 <= args.output_width <= 1920 or not 1 <= args.port <= 65535:
        parser.error("Invalid output width or port")
    if os.geteuid() == 0:
        parser.error("Run as the camera login user, not root; sudo is requested only for the service installation")
    user = getpass.getuser()
    entry = pwd.getpwnam(user)
    helper = Path(__file__).resolve().with_name("neon_raw.py")
    if not helper.exists():
        parser.error("Place neon_raw.py beside neon_install.py")
    import cv2
    print("Python and system OpenCV available:", cv2.__version__)
    status = subprocess.call(["systemctl", "is-active", "--quiet", UNIT])
    if status != 0:
        fuser = shutil.which("fuser")
        if fuser and subprocess.call([fuser, "-v", "/dev/video0"]) == 0:
            parser.error("/dev/video0 is already in use. Stop that specific viewer/capture process first, then rerun.")
    destination = Path(entry.pw_dir) / ".local" / "share" / "neurogrip"
    destination.mkdir(parents=True, exist_ok=True)
    deployed = destination / "neon_raw.py"
    shutil.copy2(str(helper), str(deployed))
    # Quote systemd path arguments without introducing a shell.
    def quote(value):
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'
    try:
        grp.getgrnam("video")
        groups = "SupplementaryGroups=video\n"
    except KeyError:
        groups = ""
    unit = ("[Unit]\nDescription=Neurogrip overlay-free camera stream\nAfter=network.target\n"
            "StartLimitIntervalSec=0\n\n[Service]\nType=simple\n"
            "User=" + user + "\n" + groups +
            "WorkingDirectory=" + quote(destination) + "\n"
            "ExecStart=/usr/bin/python3 -u " + quote(deployed) + " --output-width " + str(args.output_width) +
            " --port " + str(args.port) + (" --no-rotate" if args.no_rotate else "") + "\n"
            "Restart=on-failure\nRestartSec=3\nKillSignal=SIGINT\nTimeoutStopSec=5\n"
            "NoNewPrivileges=true\nPrivateTmp=true\n\n[Install]\nWantedBy=multi-user.target\n")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".service", delete=False) as output:
            output.write(unit)
            temporary = output.name
        subprocess.check_call(["sudo", "install", "-m", "0644", temporary, "/etc/systemd/system/" + UNIT])
        subprocess.check_call(["sudo", "systemctl", "daemon-reload"])
        subprocess.check_call(["sudo", "systemctl", "enable", "--now", UNIT])
        subprocess.check_call(["sudo", "systemctl", "restart", UNIT])
        subprocess.check_call(["systemctl", "status", "--no-pager", UNIT])
    finally:
        if temporary:
            os.unlink(temporary)
    print("Boot capture configured. Check http://CAMERA_LAN_IP:%d/state for fresh=true, overlays=false." % args.port)
    print("If capture fails: journalctl -u %s -n 40 --no-pager" % UNIT)


if __name__ == "__main__":
    main()
