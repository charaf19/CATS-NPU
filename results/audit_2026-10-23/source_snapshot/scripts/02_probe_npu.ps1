param([ValidateSet('f16','f32')][string]$Dtype='f16')
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m cats_npu probe --device NPU --dtype $Dtype --out "results/probe_npu_$Dtype.json"
if ($LASTEXITCODE -ne 0) { throw 'NPU not available or probing failed; inspect doctor output' }
Write-Host 'Inspect per-operator compiled, inferred, and correct fields in probe JSON.'
Write-Host 'A partial probe is a valid scientific result, not permission to pretend it passed.'
