param(
    [ValidateRange(2,256)][int]$TileBatch=32,
    [ValidateRange(1,1000)][int]$Repeats=5,
    [ValidateRange(1,100)][int]$Blocks=2,
    [ValidateRange(0,100)][int]$Warmup=2,
    [ValidateSet('PREFER_PLUGIN','PLUGIN','DRIVER')][string]$CompilerType='PLUGIN'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path 'native\build\cats_topk.dll')) {
    throw 'C++ baseline is missing. Run .\scripts\08_build_cpp.ps1 first.'
}

$methods = @(
    'cpu_exact',
    'cpp_exact',
    'numpy_hier',
    'numpy_adaptive',
    'ov_npu_native_sequential_hier',
    'ov_npu_native_batched_hier',
    'ov_npu_native_sequential_adaptive',
    'ov_npu_native_batched_adaptive'
) -join ','
$compilerTag = $CompilerType.ToLowerInvariant()
$out = "results/phase2_pilot/npu_${compilerTag}_b${TileBatch}_strict"
Write-Host "Starting Phase 2 development pilot: tile batch=$TileBatch compiler=$CompilerType"
Write-Host 'This is descriptive tuning evidence, not a confirmatory statistical session.'
& $python -u -m cats_npu bench --methods $methods --ns 512,4096,16384 --ks 4,10 `
    --distributions normal,tied,clustered,adversarial --dtype f16 --tile 16 `
    --batch $TileBatch --warmup $Warmup --repeats $Repeats --blocks $Blocks `
    --compiler-type $CompilerType --out $out
if ($LASTEXITCODE -ne 0) {
    throw "Phase 2 NPU pilot failed. Inspect $out skips and mismatches."
}

