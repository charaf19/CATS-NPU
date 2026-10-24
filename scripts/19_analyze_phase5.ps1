param(
    [Parameter(Mandatory=$true)][string]$ResultsPath
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\analyze_phase5_real.py $ResultsPath --out results/phase5/sift1m_summary
if ($LASTEXITCODE -ne 0) { throw 'SIFT1M analysis failed.' }
