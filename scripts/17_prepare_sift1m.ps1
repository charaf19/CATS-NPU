param(
    [switch]$Download
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$arguments = @('scripts\prepare_sift1m.py')
if ($Download) { $arguments += '--download' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'SIFT1M preparation failed.' }
