# Optional portable Windows C++ baseline. Install its local build tools with:
# .\.venv\Scripts\python.exe -m pip install -e '.[native]'
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$venv = Join-Path (Get-Location) '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$localCmake = Join-Path $venv 'Scripts\cmake.exe'
$localNinja = Join-Path $venv 'Scripts\ninja.exe'

if (Test-Path -LiteralPath $localCmake) {
    $cmakePath = $localCmake
} else {
    $cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($null -eq $cmake) {
        throw "CMake is unavailable. Run: $python -m pip install -e '.[native]'"
    }
    $cmakePath = $cmake.Source
}

$zigPath = $null
if ((Test-Path -LiteralPath $python) -and (Test-Path -LiteralPath $localNinja)) {
    $zigCandidate = & $python -c "import pathlib, ziglang; print(pathlib.Path(ziglang.__file__).parent / 'zig.exe')" 2>$null
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $zigCandidate)) {
        $zigPath = (Resolve-Path -LiteralPath $zigCandidate).Path
    }
}

Write-Host "Using CMake: $cmakePath"
if ($null -ne $zigPath) {
    Write-Host "Using portable Zig C++ compiler: $zigPath"
    & $cmakePath -G Ninja -S native -B native/build -DCMAKE_BUILD_TYPE=Release `
        "-DCMAKE_MAKE_PROGRAM=$localNinja" "-DCMAKE_CXX_COMPILER=$zigPath" `
        -DCMAKE_CXX_COMPILER_ARG1=c++
} else {
    Write-Host 'Using the C++ toolchain selected by CMake.'
    & $cmakePath -S native -B native/build
}
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
& $cmakePath --build native/build --config Release
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed' }
& $python -m pytest -q tests/test_native_optional.py
if ($LASTEXITCODE -ne 0) { throw 'C++ baseline correctness test failed' }
Write-Host 'Optional C++ CPU baseline built. Run 05_benchmark_cpu.ps1 again.'
