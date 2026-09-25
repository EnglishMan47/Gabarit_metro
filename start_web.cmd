@echo off
rem Gabarit: web interface without Docker/ROS (Windows).
rem   start_web.cmd [folder with bags]
rem   without a folder: data\extracted\for_hackathon and data\extracted\synthetic_organizers
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" powershell -ExecutionPolicy Bypass -File install.ps1
set DATA_ARG=
if not "%~1"=="" set DATA_ARG=--data "%~1"
echo.
echo   Gabarit: open http://localhost:8000 in the browser (keep this window open).
echo.
".venv\Scripts\python.exe" -m gabarit web %DATA_ARG%
pause
