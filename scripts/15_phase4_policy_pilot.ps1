param(
    [ValidateRange(1,1000)][int]$Repeats=10,
    [ValidateRange(1,100)][int]$Blocks=2,
    [ValidateRange(0,100)][int]$Warmup=2,
    [ValidateSet('PREFER_PLUGIN','PLUGIN','DRIVER')][string]$CompilerType='PLUGIN'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$methods = @(
    'cpu_exact',
    'cpp_exact',
    'ov_npu_native_batched_hier',
    'ov_npu_native_batched_adaptive',
    'ov_npu_native_batched_costadaptive'
) -join ','
& $python -u -m cats_npu bench --methods $methods --ns 4096,16384 --ks 4,10 `
    --distributions normal,tied,clustered,adversarial --dtype f16 --tile 256 `
    --batch 32 --warmup $Warmup --repeats $Repeats --blocks $Blocks `
    --compiler-type $CompilerType --out results/phase4/policy_pilot
if ($LASTEXITCODE -ne 0) {
    throw 'Phase 4 policy pilot failed. Inspect results/phase4/policy_pilot.'
}
