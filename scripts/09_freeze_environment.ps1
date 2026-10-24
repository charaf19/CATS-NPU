# Capture environment after successful setup and before each confirmatory study.
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
New-Item -ItemType Directory -Path results -Force | Out-Null
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m pip freeze | Out-File -FilePath 'results\requirements-installed.txt' -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed' }
& $python -m cats_npu doctor --out results/doctor.json
if ($LASTEXITCODE -ne 0) { throw 'Doctor failed' }
Write-Host 'Captured the resolved dependencies and device properties under results/'
