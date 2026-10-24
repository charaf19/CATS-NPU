param(
    [Parameter(Mandatory=$true)][string]$ResultsPath
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\analyze_phase5_real.py $ResultsPath `
    --out results/phase5/beir_summary --title 'BEIR semantic scores'
if ($LASTEXITCODE -ne 0) { throw 'BEIR evaluation analysis failed.' }
