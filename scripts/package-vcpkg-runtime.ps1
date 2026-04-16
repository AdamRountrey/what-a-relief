param(
    [string]$BuildDir = "build\ninja-vcpkg",
    [string]$OutputDir = "build-vcpkg-direct"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $repo "$BuildDir\vcpkg_installed\x64-windows\bin"
$out = Join-Path $repo $OutputDir
$exe = Join-Path $out "ps_spheres.exe"

if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe. Build first with scripts\build-vcpkg-direct-msvc.ps1."
}
if (-not (Test-Path $bin)) {
    throw "vcpkg runtime DLL directory not found: $bin"
}

Copy-Item -Path (Join-Path $bin "*.dll") -Destination $out -Force
Write-Host "Copied vcpkg runtime DLLs to $out"
