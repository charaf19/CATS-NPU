param(
    [Parameter(Mandatory=$true)][string]$ResultsPath
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\analyze_phase7_confirmatory.py $ResultsPath
if ($LASTEXITCODE -ne 0) { throw 'Confirmatory analysis failed.' }
