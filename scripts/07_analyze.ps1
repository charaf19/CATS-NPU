param([string]$DataFolder='results/bench_npu_plugin', [string]$Baseline='cpu_exact')
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $DataFolder)) {
    throw "Benchmark data folder not found: $DataFolder"
}
Write-Host "Analyzing benchmark data in $DataFolder"
& $python -m cats_npu analyze $DataFolder --baseline $Baseline --out results/analysis.json
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed' }
Write-Host 'WARNING: multiple blocks from one script run are NOT independent statistical sessions.'
Write-Host 'Collect six or more separate sessions for inferential testing.'
