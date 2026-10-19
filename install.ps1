# Gabarit: installation on Windows (no Docker, no ROS needed).
#   powershell -ExecutionPolicy Bypass -File install.ps1
# Python 3.10+. Docker/RViz installation: docs/INSTALLATION.md.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if ($py) { py -3 -m venv .venv } else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw "Cannot create .venv. Install Python 3.10+ with venv support." }
}
& .venv\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 'Python 3.10+ is required; recreate .venv with a supported Python.')"
if ($LASTEXITCODE -ne 0) { throw "Unsupported Python in .venv." }
& .venv\Scripts\python.exe -m pip install --upgrade pip | Out-Null
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed. Check the network and Python installation." }
& .venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& .venv\Scripts\python.exe -m gabarit check
if ($LASTEXITCODE -ne 0) { throw "Environment check failed." }
Write-Host ""
Write-Host "Done. Start the web interface:  .venv\Scripts\python.exe -m gabarit web --data <folder with bags>"
Write-Host "or double-click start_web.cmd"
