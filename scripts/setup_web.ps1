param(
    [string]$PythonCommand = "py",
    [string]$EnvironmentDirectory = ".venv",
    [string]$WheelDirectory = "",
    [ValidateSet("geometry", "cpu", "gpu")]
    [string]$Inference = "geometry"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPath = Join-Path $ProjectRoot $EnvironmentDirectory
$PythonPath = Join-Path $EnvironmentPath "Scripts\python.exe"

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $PythonPath)) {
    if ([IO.Path]::GetFileNameWithoutExtension($PythonCommand) -eq "py") {
        & $PythonCommand -3 -m venv $EnvironmentPath
    } else {
        & $PythonCommand -m venv $EnvironmentPath
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Python environment." }
}

& $PythonPath -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 or newer is required." }
$RequirementsFile = switch ($Inference) {
    "cpu" { "requirements-inference-cpu.txt" }
    "gpu" { "requirements-inference-gpu.txt" }
    default { "requirements.txt" }
}
if ($WheelDirectory) {
    & $PythonPath -m pip install --no-index --find-links $WheelDirectory -r $RequirementsFile
} else {
    & $PythonPath -m pip install -r $RequirementsFile
}
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Environment is not ready." }

Write-Output "Environment is ready ($Inference inference). Run: .\scripts\run_web.ps1"
