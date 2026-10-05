"""Fetch pinned official DA3 source inside this repository (Windows PC)."""
from pathlib import Path
import subprocess

from depth_backend import UPSTREAM_SOURCE_REVISION


def main():
    destination = Path(__file__).resolve().parent / ".vendor" / "depth-anything-3"
    if destination.exists():
        current = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if current != UPSTREAM_SOURCE_REVISION:
            raise SystemExit("Existing vendor source has a different revision; preserve/review it before switching.")
        print("Pinned official source is already present.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", UPSTREAM_SOURCE_REVISION], check=True)
    print("Pinned official source ready. First model inference downloads/caches the public weights.")


if __name__ == "__main__":
    main()
