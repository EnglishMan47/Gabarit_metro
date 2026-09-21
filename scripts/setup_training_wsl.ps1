$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$Distributions = @(& wsl.exe --list --quiet) -replace "`0", ""
if ($Distributions -notcontains "Ubuntu-24.04") {
    throw "Ubuntu-24.04 не установлена. Выполните от администратора: wsl --install -d Ubuntu-24.04"
}

& wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip"
if ($LASTEXITCODE -ne 0) { throw "Не удалось установить Python в Ubuntu." }

& wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "mkdir -p /root/.venvs && python3 -m venv /root/.venvs/metro-bev && /root/.venvs/metro-bev/bin/python -m pip install --upgrade pip && /root/.venvs/metro-bev/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128"
if ($LASTEXITCODE -ne 0) { throw "Не удалось установить CUDA-сборку PyTorch." }

& wsl.exe -d Ubuntu-24.04 -u root --cd $ProjectRoot -- /root/.venvs/metro-bev/bin/pip install -r requirements-training.txt
if ($LASTEXITCODE -ne 0) { throw "Не удалось установить зависимости обучения." }

& wsl.exe -d Ubuntu-24.04 -u root --cd $ProjectRoot -- /root/.venvs/metro-bev/bin/python scripts/verify_training_runtime.py
if ($LASTEXITCODE -ne 0) { throw "PyTorch не смог выполнить проверку на GPU." }

Write-Output "Training environment is ready. Run: .\scripts\train_bev_wsl.ps1"
