#!/usr/bin/env bash
# Gabarit: web interface without Docker/ROS (Linux/macOS).
#   ./start_web.sh [folder with bags]
#   without a folder: data/extracted/for_hackathon and data/extracted/synthetic_organizers
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./install.sh
exec .venv/bin/python -m gabarit web ${1:+--data "$1"} --host 0.0.0.0 --port "${PORT:-8000}"
