# Optional, requires CMake + MSVC Build Tools (or another C++17 compiler).
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
& cmake -S native -B native/build
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
& cmake --build native/build --config Release
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed' }
Write-Host 'Optional C++ CPU baseline built. Run 05_benchmark_cpu.ps1 again.'
