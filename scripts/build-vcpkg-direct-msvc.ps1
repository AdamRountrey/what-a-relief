param(
    [string]$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$CMake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo "build\ninja-vcpkg"
$out = Join-Path $repo "build-vcpkg-direct"
$obj = Join-Path $out "obj"

function Find-VisualStudioInstall {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($install) {
            return $install.Trim()
        }
    }

    foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
        $candidate = "C:\Program Files\Microsoft Visual Studio\2022\$edition"
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

$vsInstall = Find-VisualStudioInstall
if (-not (Test-Path $VsDevCmd)) {
    if (-not $vsInstall) {
        throw "Could not find a Visual Studio 2022 installation with C++ tools."
    }
    $VsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
}
if (-not (Test-Path $VsDevCmd)) {
    throw "Could not find VsDevCmd.bat: $VsDevCmd"
}

if (-not (Test-Path $CMake)) {
    $cmakeCommand = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($cmakeCommand) {
        $CMake = $cmakeCommand.Source
    } elseif ($vsInstall) {
        $CMake = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    }
}
if (-not (Test-Path $CMake)) {
    throw "Could not find cmake.exe: $CMake"
}

if (-not $env:VCPKG_ROOT) {
    if ($vsInstall) {
        $candidateVcpkg = Join-Path $vsInstall "VC\vcpkg"
        if (Test-Path $candidateVcpkg) {
            $env:VCPKG_ROOT = $candidateVcpkg
        }
    }
}
if (-not $env:VCPKG_ROOT -or -not (Test-Path $env:VCPKG_ROOT)) {
    throw "Could not find vcpkg. Set VCPKG_ROOT to a vcpkg installation."
}

$cmakeTools = Split-Path -Parent (Split-Path -Parent $CMake)
$ninjaDir = Join-Path $cmakeTools "Ninja"

New-Item -ItemType Directory -Force `
    (Join-Path $repo ".vcpkg-downloads"), `
    (Join-Path $repo ".vcpkg-cache"), `
    (Join-Path $repo ".vcpkg-registries"), `
    $obj | Out-Null

$envScript = @"
set VCPKG_FORCE_SYSTEM_BINARIES=1
set VCPKG_ROOT=$env:VCPKG_ROOT
set X_VCPKG_REGISTRIES_CACHE=$repo\.vcpkg-registries
set VCPKG_DOWNLOADS=$repo\.vcpkg-downloads
set VCPKG_DEFAULT_BINARY_CACHE=$repo\.vcpkg-cache
if exist "$ninjaDir" set PATH=$ninjaDir;%PATH%
set PATH=C:\Program Files\Git\cmd;C:\Program Files\Git\bin;%PATH%
call "$VsDevCmd" -arch=x64
"$CMake" --preset ninja-vcpkg
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /EHsc /W4 /permissive- /external:W0 /external:I"$build\vcpkg_installed\x64-windows\include\opencv4" ^
  "$repo\src\args.cpp" "$repo\src\crop_ui.cpp" "$repo\src\gui_workflow.cpp" "$repo\src\image_io.cpp" "$repo\src\main.cpp" "$repo\src\photometric.cpp" "$repo\src\relight_ui.cpp" "$repo\src\sphere_ui.cpp" ^
  /Fe:"$out\what-a-relief.exe" /Fo:"$obj\\" ^
  /link /MANIFEST:NO /LIBPATH:"$build\vcpkg_installed\x64-windows\lib" ^
  opencv_highgui4.lib opencv_videoio4.lib opencv_imgcodecs4.lib opencv_imgproc4.lib opencv_core4.lib comdlg32.lib shell32.lib ole32.lib user32.lib
if errorlevel 1 exit /b %errorlevel%
"@

$cmdFile = Join-Path $out "build.cmd"
Set-Content -Path $cmdFile -Value $envScript -Encoding ASCII
cmd.exe /c "`"$cmdFile`""
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE"
}

Write-Host "Built $out\what-a-relief.exe"
Write-Host "Run with this DLL path first:"
Write-Host "  $build\vcpkg_installed\x64-windows\bin"
