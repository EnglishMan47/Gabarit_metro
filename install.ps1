# Gabarit: installation on Windows (no Docker, no ROS needed).
#   powershell -ExecutionPolicy Bypass -File install.ps1
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if ($py) { py -3 -m venv .venv } else { python -m venv .venv }
}
& .venv\Scripts\python.exe -m pip install --upgrade pip | Out-Null
& .venv\Scripts\python.exe -m pip install -r requirements.txt
& .venv\Scripts\python.exe -m gabarit check
Write-Host ""
Write-Host "Done. Start the web interface:  .venv\Scripts\python.exe -m gabarit web --data <folder with bags>"
Write-Host "or double-click start_web.cmd"
