param(
    [string]$ResultsPath='results/phase3/ablation_baseline'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\analyze_phase3_ablation.py $ResultsPath
if ($LASTEXITCODE -ne 0) { throw 'Phase 3 ablation analysis failed.' }
