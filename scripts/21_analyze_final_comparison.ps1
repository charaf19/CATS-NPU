param(
    [Parameter(Mandatory=$true)][string]$Phase5Path,
    [Parameter(Mandatory=$true)][string]$DirectPath
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\analyze_phase5_real.py $Phase5Path $DirectPath `
    --out results/phase6/final_sift1m_summary
if ($LASTEXITCODE -ne 0) { throw 'Final comparison analysis failed.' }
