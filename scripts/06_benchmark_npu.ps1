param(
    [ValidateSet('native','bitonic')][string]$Kernel='native',
    [ValidateSet('PREFER_PLUGIN','PLUGIN','DRIVER')][string]$CompilerType='PLUGIN'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$methods = "cpu_exact,numpy_hier,numpy_adaptive,ov_npu_${Kernel}_hier,ov_npu_${Kernel}_adaptive"
$compilerTag = $CompilerType.ToLowerInvariant()
Write-Host "Starting NPU benchmark: kernel=$Kernel compiler=$CompilerType"
Write-Host 'The first NPU model compilation may take about a minute.'
& $python -u -m cats_npu bench --methods $methods --ns 512,4096 --ks 4,10 `
    --distributions normal,tied,clustered,adversarial --dtype f16 --tile 16 --batch 1 `
    --warmup 3 --repeats 20 --blocks 2 --compiler-type $CompilerType `
    --out "results/bench_npu_${compilerTag}"
if ($LASTEXITCODE -ne 0) { throw 'NPU experiment failed. Inspect skips and mismatches; do not report speedups.' }
