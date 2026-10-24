$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\run_phase7_confirmatory.py
if ($LASTEXITCODE -ne 0) { throw 'Locked confirmatory experiment failed.' }
