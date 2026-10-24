param(
    # Avoid PowerShell's automatic $input enumerator, whose name is
    # case-insensitively identical to $Input.
    [string]$ResultsPath='results/phase4/policy_pilot'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python scripts\summarize_phase4_policy.py $ResultsPath
if ($LASTEXITCODE -ne 0) { throw 'Phase 4 policy summary failed.' }
& $python -m cats_npu analyze $ResultsPath --baseline cpp_exact `
    --out results/phase4/policy_analysis.json
if ($LASTEXITCODE -ne 0) { throw 'Phase 4 statistical analysis failed.' }
