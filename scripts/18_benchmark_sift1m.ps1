param(
    [string]$Config='configs/phase5_sift1m.json',
    [string]$Out='results/phase5/sift1m_evaluation'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\run_phase5_sift1m.py --config $Config --out $Out
if ($LASTEXITCODE -ne 0) { throw 'Strict SIFT1M evaluation failed.' }
