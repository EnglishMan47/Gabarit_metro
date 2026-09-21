@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run scripts\setup_web.ps1 first to install dependencies.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8000 in your browser.
echo Keep this window open. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m webapp.server --data-root "data/extracted/for_hackathon" --host 127.0.0.1 --port 8000
pause
