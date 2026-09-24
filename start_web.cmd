@echo off
rem Gabarit: local web interface without Docker/ROS (Windows).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment...
  py -3 -m venv .venv || python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
echo.
echo   Gabarit: open http://localhost:8000 in the browser (keep this window open).
echo.
".venv\Scripts\python.exe" -m web.server --data "data/extracted/for_hackathon" --port 8000 %*
pause
