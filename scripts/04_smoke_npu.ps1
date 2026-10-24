param(
    [ValidateSet('native','bitonic')][string]$Kernel='native',
    [ValidateSet('PREFER_PLUGIN','PLUGIN','DRIVER')][string]$CompilerType='PLUGIN'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$method = "ov_npu_${Kernel}_hier"
& $python -m cats_npu smoke --method $method --n 64 --k 4 --tile 16 --batch 1 --dtype f16 --distribution normal --compiler-type $CompilerType
if ($LASTEXITCODE -ne 0) { throw 'Strict NPU hardware Top-K smoke test failed. Do not proceed with NPU performance claims.' }
Write-Host 'Strict NPU method passed for this particular shape and kernel. Check probe output before scaling.'
