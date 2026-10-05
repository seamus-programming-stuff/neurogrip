#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$TASK_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Run bash install.sh first.' >&2
  exit 1
fi
# The calibrator prompts for the measured board, teaches regions, and starts
# the feed afterwards. It requires the NEON's display or an X-forwarded display.
exec "$TASK_ROOT/.venv/bin/python" -u "$TASK_ROOT/calibrate.py" "$@"
