$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$methods = 'cpu_exact,numpy_hier,numpy_adaptive'
if (Test-Path 'native\build\Release\cats_topk.dll') { $methods += ',cpp_exact' }
& $python -m cats_npu bench --methods $methods --ns 1024,16384 --ks 1,10 `
    --distributions normal,tied,clustered,adversarial --tile 128 --batch 8 `
    --warmup 3 --repeats 20 --blocks 2 --out results/bench_cpu
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed; check results/bench_cpu' }
