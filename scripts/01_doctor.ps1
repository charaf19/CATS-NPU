$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m cats_npu doctor --out results/doctor.json
if ($LASTEXITCODE -ne 0) { throw 'Doctor failed' }
Write-Host 'Saved results/doctor.json'
