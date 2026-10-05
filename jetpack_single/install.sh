#!/usr/bin/env bash
# Offline installation for the verified JetPack 5.1.2 / Python 3.8 NEON.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TASK_SYSTEM_PYTHON="/usr/bin/python3"
cd -- "$TASK_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
TASK_REPAIR=()
if [[ "$#" == 1 && "$1" == '--repair' ]]; then
  TASK_REPAIR=(--repair)
elif [[ "$#" != 0 ]]; then
  printf '%s\n' 'Usage: bash install.sh [--repair]' >&2
  exit 2
fi

if [[ "${EUID}" == 0 ]]; then
  printf '%s\n' 'Run install.sh as the adlink login user, without sudo. System Python is preserved.' >&2
  exit 1
fi
if [[ ! -x "$TASK_SYSTEM_PYTHON" || ! -f SHA256SUMS || ! -d wheels || ! -f requirements-offline.txt ]]; then
  printf '%s\n' 'Incomplete bundle: system python3, SHA256SUMS, wheels/ and requirements-offline.txt are required.' >&2
  exit 1
fi
printf '%s\n' 'Checking the complete offline bundle...'
"$TASK_SYSTEM_PYTHON" check_neon.py --verify-bundle "$TASK_ROOT" "${TASK_REPAIR[@]}"
"$TASK_SYSTEM_PYTHON" check_neon.py --platform-only --record-system-cv2 .system-cv2.json

if [[ -e .venv ]]; then
  if [[ ! -x .venv/bin/python || ! -f .venv/.neurogrip-managed ]]; then
    printf '%s\n' 'An incomplete or unrelated .venv already exists. Use a fresh extraction directory; it will not be deleted.' >&2
    exit 1
  fi
else
  # Zip-import only pure Python bootstrap wheels. No apt, ensurepip, get-pip
  # download, system pip installation or global Python changes are needed.
  TASK_BOOTSTRAP_PATH="$("$TASK_SYSTEM_PYTHON" - "$TASK_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]) / 'wheels'
packages = ['virtualenv-20.26.6', 'distlib-0.3.9', 'filelock-3.13.4',
            'platformdirs-4.3.6', 'typing_extensions-4.12.2',
            'importlib_metadata-8.5.0', 'zipp-3.20.2']
paths = []
for package in packages:
    matches = list(root.glob(package + '-*.whl'))
    if len(matches) != 1:
        raise SystemExit('Expected exactly one bundled bootstrap wheel: ' + package)
    paths.append(str(matches[0]))
print(':'.join(paths))
PY
)"
  PYTHONPATH="$TASK_BOOTSTRAP_PATH" "$TASK_SYSTEM_PYTHON" -m virtualenv \
    --python "$TASK_SYSTEM_PYTHON" --system-site-packages --no-seed \
    --no-download --no-periodic-update --app-data "$TASK_ROOT/.virtualenv-cache" "$TASK_ROOT/.venv"
  printf '%s\n' 'neurogrip-jetpack-5.1.2-v1' > .venv/.neurogrip-managed
fi

TASK_PYTHON="$TASK_ROOT/.venv/bin/python"
TASK_PIP_WHEEL="$TASK_ROOT/wheels/pip-23.3.2-py3-none-any.whl"
if [[ ! -f "$TASK_PIP_WHEEL" ]]; then
  printf '%s\n' 'The pinned offline pip wheel is missing.' >&2
  exit 1
fi
# --ignore-installed writes only into this virtual environment, even when the
# system exposes older packages. --no-deps relies on the complete pinned list.
PYTHONPATH="$TASK_PIP_WHEEL" "$TASK_PYTHON" -m pip --isolated install \
  --no-index --no-deps --ignore-installed "$TASK_PIP_WHEEL"
"$TASK_PYTHON" -m pip --isolated install --no-index --no-deps --ignore-installed \
  --only-binary=:all: --find-links "$TASK_ROOT/wheels" -r requirements-offline.txt
"$TASK_PYTHON" check_neon.py --no-camera --expected-cv2 .system-cv2.json --model models/depth.onnx
printf '%s\n' 'Offline installation complete.' 'Run: bash run.sh' \
  'Camera calibration and teaching: bash calibrate.sh' \
  'Optional boot service: .venv/bin/python install_service.py --start'
