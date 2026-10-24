param(
    [ValidateSet('f16','f32')][string]$Dtype='f16',
    [ValidateSet('PREFER_PLUGIN','PLUGIN','DRIVER')][string]$CompilerType='PLUGIN'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$compilerTag = $CompilerType.ToLowerInvariant()
$output = "results/probe_npu_${Dtype}_${compilerTag}.json"
& $python -m cats_npu probe --device NPU --dtype $Dtype --compiler-type $CompilerType --out $output
if ($LASTEXITCODE -ne 0) { throw 'NPU probing failed; inspect the per-operator result JSON' }
Write-Host "Saved $output"
Write-Host 'Basic NPU operator probe passed; inspect compiler_type_used and execution_devices in the JSON.'
Write-Host 'Adversarial numerical exactness remains a separate research gate.'
