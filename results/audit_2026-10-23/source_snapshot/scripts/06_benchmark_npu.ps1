param([ValidateSet('native','bitonic')][string]$Kernel='native')
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$methods = "cpu_exact,numpy_hier,numpy_adaptive,ov_npu_${Kernel}_hier,ov_npu_${Kernel}_adaptive"
& $python -m cats_npu bench --methods $methods --ns 512,4096 --ks 4,10 `
    --distributions normal,tied,clustered,adversarial --dtype f16 --tile 16 --batch 1 `
    --warmup 3 --repeats 20 --blocks 2 --out results/bench_npu
if ($LASTEXITCODE -ne 0) { throw 'NPU experiment failed. Inspect skips and mismatches; do not report speedups.' }
