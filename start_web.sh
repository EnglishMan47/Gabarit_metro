#!/usr/bin/env bash
# Gabarit: local web interface without Docker/ROS (Linux/macOS).
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python -m web.server --data "${1:-data/extracted/for_hackathon}" --host 0.0.0.0 --port 8000
