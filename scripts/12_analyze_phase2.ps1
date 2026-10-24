param(
    [string]$NpuFolder='results/phase2_pilot/npu_plugin_b32_strict',
    [string]$GpuFolder='results/phase2_pilot/gpu_cuda_stable'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
foreach ($folder in @($NpuFolder, $GpuFolder)) {
    if (-not (Test-Path -LiteralPath $folder)) { throw "Pilot folder not found: $folder" }
}
& $python -m cats_npu analyze $NpuFolder $GpuFolder --baseline cpp_exact `
    --out results/phase2_pilot/analysis.json
if ($LASTEXITCODE -ne 0) { throw 'Phase 2 pilot analysis failed' }
Write-Host 'Pilot analysis is descriptive until enough independent valid sessions are collected.'
