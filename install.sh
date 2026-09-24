#!/usr/bin/env bash
# Gabarit: installation on Linux/macOS (no Docker, no ROS needed).
#   ./install.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m gabarit check || true
echo
echo "Done. Start the web interface:  .venv/bin/python -m gabarit web --data <folder with bags>"
echo "or ./start_web.sh <folder with bags>"
