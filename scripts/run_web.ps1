param(
    [string]$DataRoot = "data/extracted/for_hackathon",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python environment not found: $PythonPath"
}

Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "Data directory not found: $DataRoot. Pass -DataRoot with the extracted dataset directory."
}
& $PythonPath -m webapp.server --data-root $DataRoot --host $HostAddress --port $Port
if ($LASTEXITCODE -ne 0) { throw "Web server stopped with exit code $LASTEXITCODE. Check the error above." }
