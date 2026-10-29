$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& $python -m cats_npu smoke --method numpy_adaptive --n 1024 --k 10 --tile 128 --batch 8 --distribution tied
if ($LASTEXITCODE -ne 0) { throw 'Adaptive CPU smoke test failed' }
Write-Host 'CPU prototype passed. Next: .\scripts\04_smoke_npu.ps1'
