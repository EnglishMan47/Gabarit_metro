param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int]$Epochs = 20,
    [int]$BatchSize = 16,
    [string]$Output = "models/bev_segmentation.onnx"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
& wsl.exe -d Ubuntu-24.04 -u root --cd $ProjectRoot -- /root/.venvs/metro-bev/bin/python scripts/verify_training_runtime.py
if ($LASTEXITCODE -ne 0) {
    throw "GPU-проверка завершилась с ошибкой."
}
& wsl.exe -d Ubuntu-24.04 -u root --cd $ProjectRoot -- /root/.venvs/metro-bev/bin/python scripts/train_bev.py --device $Device --epochs $Epochs --batch-size $BatchSize --output $Output
if ($LASTEXITCODE -ne 0) {
    throw "Обучение завершилось с ошибкой."
}
