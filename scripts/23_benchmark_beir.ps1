$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\run_phase5_beir.py
if ($LASTEXITCODE -ne 0) { throw 'Strict BEIR evaluation failed.' }
