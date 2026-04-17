param(
    [string]$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$CMake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo "build\ninja-vcpkg"
$out = Join-Path $repo "build-vcpkg-direct"
$obj = Join-Path $out "obj"

New-Item -ItemType Directory -Force `
    (Join-Path $repo ".vcpkg-downloads"), `
    (Join-Path $repo ".vcpkg-cache"), `
    (Join-Path $repo ".vcpkg-registries"), `
    $obj | Out-Null

$envScript = @"
set VCPKG_FORCE_SYSTEM_BINARIES=1
set X_VCPKG_REGISTRIES_CACHE=$repo\.vcpkg-registries
set VCPKG_DOWNLOADS=$repo\.vcpkg-downloads
set VCPKG_DEFAULT_BINARY_CACHE=$repo\.vcpkg-cache
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
