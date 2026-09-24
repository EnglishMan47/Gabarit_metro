@echo off
rem Gabarit: web interface without Docker/ROS (Windows).
rem   start_web.cmd [folder with bags]
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" powershell -ExecutionPolicy Bypass -File install.ps1
set DATA=%~1
if "%DATA%"=="" set DATA=data\extracted\for_hackathon
echo.
echo   Gabarit: open http://localhost:8000 in the browser (keep this window open).
echo.
".venv\Scripts\python.exe" -m gabarit web --data "%DATA%"
pause
