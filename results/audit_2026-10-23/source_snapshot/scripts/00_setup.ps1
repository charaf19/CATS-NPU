# PowerShell 7 or Windows PowerShell 5.1; Python 3.11 recommended.
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
Write-Host 'Creating CATS-NPU virtual environment using Python 3.11 ...'
& py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 not found. Install from python.org or change py -3.11 to your Python.' }
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $python -m pip install -e '.[all]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network / corporate proxy.' }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed' }
Write-Host 'Setup completed. Next: .\scripts\01_doctor.ps1'
