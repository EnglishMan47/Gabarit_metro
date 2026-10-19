#!/usr/bin/env bash
# Gabarit: installation on Linux/macOS (no Docker, no ROS needed).
#   ./install.sh
# Python 3.10+. Docker/RViz installation: docs/INSTALLATION.md.
set -eu
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 'Python 3.10+ is required; recreate .venv with a supported Python.')"
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m gabarit check
echo
echo "Done. Start the web interface:  .venv/bin/python -m gabarit web --data <folder with bags>"
echo "or ./start_web.sh <folder with bags>"
