param(
    [ValidateRange(1,1000)][int]$Repeats=10,
    [ValidateRange(1,100)][int]$Blocks=2,
    [ValidateRange(0,100)][int]$Warmup=3
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$out = 'results/phase2_pilot/gpu_cuda_stable'
$methods = 'cpu_exact,cpp_exact,torch_cpu_stable,torch_cuda_stable'
Write-Host 'Starting host-input/host-output CPU and CUDA development pilot.'
Write-Host 'Stable Torch methods record any measured CPU boundary-tie repair.'
& $python -u -m cats_npu bench --methods $methods --ns 512,4096,16384 --ks 4,10 `
    --distributions normal,tied,clustered,adversarial --dtype f16 --tile 16 `
    --batch 32 --warmup $Warmup --repeats $Repeats --blocks $Blocks `
    --out $out
if ($LASTEXITCODE -ne 0) { throw "CUDA pilot failed unexpectedly. Inspect $out." }

