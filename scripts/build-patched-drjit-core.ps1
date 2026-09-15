param(
    [Parameter(Mandatory = $true)]
    [string]$Destination,
    [Parameter(Mandatory = $true)]
    [string]$WorkDirectory,
    [string]$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$CMake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $repo "dist"
$sourceCommit = "213983e47c99db0c6ab5e3dfce952e68bb9a8bd3"
$sourceUrl = "https://github.com/mitsuba-renderer/drjit-core.git"
$patch = Join-Path $repo "tools\mitsuba_backend\drjit-core-1.3.1-large-reductions.patch"
$work = [IO.Path]::GetFullPath($WorkDirectory)
$source = Join-Path $work "source"
$build = Join-Path $work "build"
$buildCommand = Join-Path $work "build-drjit-core.cmd"

function Assert-SafeChildPath {
    param([string]$Parent, [string]$Child)
    $resolvedParent = [IO.Path]::GetFullPath($Parent).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $resolvedChild = [IO.Path]::GetFullPath($Child)
    if (-not $resolvedChild.StartsWith($resolvedParent + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside $resolvedParent`: $resolvedChild"
    }
}

if (-not (Test-Path -LiteralPath $patch)) {
    throw "Dr.Jit Core patch not found: $patch"
}
Assert-SafeChildPath -Parent $dist -Child $work
Assert-SafeChildPath -Parent $work -Child $source
Assert-SafeChildPath -Parent $work -Child $build
if (Test-Path -LiteralPath $work) {
    Remove-Item -LiteralPath $work -Recurse -Force
}
New-Item -ItemType Directory -Path $work -Force | Out-Null

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstall = ""
if (Test-Path -LiteralPath $vswhere) {
    $vsInstall = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
}
if (-not (Test-Path -LiteralPath $VsDevCmd) -and -not [string]::IsNullOrWhiteSpace($vsInstall)) {
    $VsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
}
if (-not (Test-Path -LiteralPath $CMake) -and -not [string]::IsNullOrWhiteSpace($vsInstall)) {
    $CMake = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
}
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "Could not find VsDevCmd.bat: $VsDevCmd"
}
if (-not (Test-Path -LiteralPath $CMake)) {
    throw "Could not find cmake.exe: $CMake"
}
if ([string]::IsNullOrWhiteSpace($vsInstall)) {
    $vsInstall = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $VsDevCmd))
}
$ninja = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path -LiteralPath $ninja)) {
    throw "Could not find ninja.exe: $ninja"
}
$toolset = Get-ChildItem -LiteralPath (Join-Path $vsInstall "VC\Tools\MSVC") -Directory |
    Sort-Object Name -Descending | Select-Object -First 1
if ($null -eq $toolset) {
    throw "Could not find an MSVC toolset under $vsInstall"
}
$compiler = Join-Path $toolset.FullName "bin\Hostx64\x64\cl.exe"
if (-not (Test-Path -LiteralPath $compiler)) {
    throw "Could not find the x64 MSVC compiler: $compiler"
}
$git = Get-Command git.exe -ErrorAction Stop

& $git.Source clone --recursive $sourceUrl $source
if ($LASTEXITCODE -ne 0) {
    throw "Cloning Dr.Jit Core failed with exit code $LASTEXITCODE"
}
& $git.Source -C $source checkout --detach $sourceCommit
if ($LASTEXITCODE -ne 0) {
    throw "Checking out Dr.Jit Core $sourceCommit failed with exit code $LASTEXITCODE"
}
& $git.Source -C $source submodule update --init --recursive
if ($LASTEXITCODE -ne 0) {
    throw "Updating Dr.Jit Core submodules failed with exit code $LASTEXITCODE"
}
$actualCommit = (& $git.Source -C $source rev-parse HEAD).Trim()
if ($actualCommit -ne $sourceCommit) {
    throw "Dr.Jit Core checkout mismatch: expected $sourceCommit, found $actualCommit"
}
& $git.Source -C $source apply --check --whitespace=nowarn $patch
if ($LASTEXITCODE -ne 0) {
    throw "The pinned Dr.Jit Core patch no longer applies cleanly"
}
& $git.Source -C $source apply --whitespace=nowarn $patch
if ($LASTEXITCODE -ne 0) {
    throw "Applying the pinned Dr.Jit Core patch failed with exit code $LASTEXITCODE"
}

$commandText = @"
@echo off
call "$VsDevCmd" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
"$CMake" -S "$source" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$compiler" -DCMAKE_CXX_COMPILER="$compiler" -DCMAKE_MAKE_PROGRAM="$ninja" -DDRJIT_CORE_ENABLE_TESTS=OFF
if errorlevel 1 exit /b %errorlevel%
"$CMake" --build "$build" --config Release --target drjit-core -j 8
if errorlevel 1 exit /b %errorlevel%
"@
Set-Content -LiteralPath $buildCommand -Value $commandText -Encoding ASCII
& $env:ComSpec /d /s /c $buildCommand
if ($LASTEXITCODE -ne 0) {
    throw "Building the patched Dr.Jit Core library failed with exit code $LASTEXITCODE"
}

$library = Join-Path $build "drjit-core.dll"
if (-not (Test-Path -LiteralPath $library) -or (Get-Item -LiteralPath $library).Length -eq 0) {
    throw "The patched Dr.Jit Core build did not produce a nonempty drjit-core.dll"
}
$destinationPath = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path (Split-Path -Parent $destinationPath) -Force | Out-Null
Copy-Item -LiteralPath $library -Destination $destinationPath -Force
$hash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash
Write-Host "Built patched Dr.Jit Core $sourceCommit"
Write-Host "Patched library SHA-256: $hash"
