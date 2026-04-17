param(
    [string]$BuildDir = "build\ninja-vcpkg",
    [string]$OutputDir = "build-vcpkg-direct"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $repo "$BuildDir\vcpkg_installed\x64-windows\bin"
$share = Join-Path $repo "$BuildDir\vcpkg_installed\x64-windows\share"
$out = Join-Path $repo $OutputDir
$exe = Join-Path $out "what-a-relief.exe"

function Add-LicenseSection {
    param(
        [string]$OutputPath,
        [string]$ShareDir,
        [string]$PackageName,
        [string]$DisplayName
    )

    $copyright = Join-Path $ShareDir "$PackageName\copyright"
    if (-not (Test-Path $copyright)) {
        Write-Warning "License notice not found for $DisplayName at $copyright"
        return
    }

    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value ""
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "================================================================================"
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value $DisplayName
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "vcpkg notice file: share/$PackageName/copyright"
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "================================================================================"
    Get-Content -LiteralPath $copyright | Add-Content -LiteralPath $OutputPath -Encoding UTF8
}

function Write-ThirdPartyLicenseBundle {
    param(
        [string]$OutputPath,
        [string]$ShareDir
    )

    if (-not (Test-Path $ShareDir)) {
        throw "vcpkg notice directory not found: $ShareDir"
    }

    $packages = @(
        @{ Package = "opencv4"; Display = "OpenCV 4" },
        @{ Package = "abseil"; Display = "Abseil" },
        @{ Package = "protobuf"; Display = "Protocol Buffers" },
        @{ Package = "libjpeg-turbo"; Display = "libjpeg-turbo" },
        @{ Package = "libpng"; Display = "libpng" },
        @{ Package = "libwebp"; Display = "libwebp" },
        @{ Package = "liblzma"; Display = "XZ Utils / liblzma" },
        @{ Package = "tiff"; Display = "LibTIFF" },
        @{ Package = "zlib"; Display = "zlib" }
    )

    Set-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "Third-party license notices for the What A Relief Windows runtime package."
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "Generated from vcpkg package copyright files."
    foreach ($package in $packages) {
        Add-LicenseSection -OutputPath $OutputPath -ShareDir $ShareDir -PackageName $package.Package -DisplayName $package.Display
    }
}

if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe. Build first with scripts\build-vcpkg-direct-msvc.ps1."
}
if (-not (Test-Path $bin)) {
    throw "vcpkg runtime DLL directory not found: $bin"
}

Copy-Item -Path (Join-Path $bin "*.dll") -Destination $out -Force
foreach ($doc in @("README.md", "LICENSE", "SECURITY.md", "AI_ATTRIBUTION.md", "THIRD_PARTY_NOTICES.md")) {
    $source = Join-Path $repo $doc
    if (Test-Path $source) {
        Copy-Item -LiteralPath $source -Destination $out -Force
    }
}
Write-ThirdPartyLicenseBundle -OutputPath (Join-Path $out "THIRD_PARTY_LICENSES.txt") -ShareDir $share
Write-Host "Copied vcpkg runtime DLLs to $out"
