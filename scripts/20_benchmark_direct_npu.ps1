param(
    [string]$Config='configs/phase6_direct_npu.json',
    [string]$Out='results/phase6/direct_npu_sift1m'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\run_phase5_sift1m.py --config $Config --out $Out
if ($LASTEXITCODE -ne 0) { throw 'Strict direct NPU control failed.' }
