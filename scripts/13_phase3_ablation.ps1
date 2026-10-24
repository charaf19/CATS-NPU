param(
    [string]$Label='baseline',
    [ValidateRange(0,100)][int]$Warmup=1,
    [ValidateRange(1,1000)][int]$Repeats=3,
    [ValidateRange(1,100)][int]$Blocks=1
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\run_phase3_ablation.py --label $Label --warmup $Warmup `
    --repeats $Repeats --blocks $Blocks
if ($LASTEXITCODE -ne 0) {
    throw 'Phase 3 ablation produced no valid configuration. Inspect results/phase3 logs.'
}
