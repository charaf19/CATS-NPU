$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\prepare_beir_scores.py
if ($LASTEXITCODE -ne 0) { throw 'BEIR semantic-score preparation failed.' }
