#!/usr/bin/env bash
# Gabarit: web interface without Docker/ROS (Linux/macOS).
#   ./start_web.sh [folder with bags]
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./install.sh
exec .venv/bin/python -m gabarit web --data "${1:-data/extracted/for_hackathon}" --host 0.0.0.0 --port "${PORT:-8000}"
