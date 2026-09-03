param(
    [string]$Version = "0.2.1",
    [string]$BuildDir = "build-vcpkg-direct"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repo $BuildDir
$dist = Join-Path $repo "dist"
$stage = Join-Path $dist "portable-work"
$archive = Join-Path $dist "What-A-Relief-$Version-portable-windows.zip"

$resolvedDist = [IO.Path]::GetFullPath($dist).TrimEnd([IO.Path]::DirectorySeparatorChar)
$resolvedStage = [IO.Path]::GetFullPath($stage)
if (-not $resolvedStage.StartsWith($resolvedDist + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to stage the portable package outside dist: $resolvedStage"
}
if (Test-Path -LiteralPath $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Path $stage -Force | Out-Null

$exe = Join-Path $source "what-a-relief.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    throw "Executable not found: $exe"
}
Copy-Item -LiteralPath $exe -Destination $stage
Copy-Item -Path (Join-Path $source "*.dll") -Destination $stage
foreach ($doc in @("README.md", "LICENSE", "SECURITY.md", "AI_ATTRIBUTION.md", "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_LICENSES.txt")) {
    $path = Join-Path $source $doc
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required portable-package file not found: $path"
    }
    Copy-Item -LiteralPath $path -Destination $stage
}
Copy-Item -LiteralPath (Join-Path $repo "docs\algorithm.md") -Destination (Join-Path $stage "ALGORITHM.md")
$models = Join-Path $source "models"
if (-not (Test-Path -LiteralPath $models)) {
    throw "Bundled model directory not found: $models"
}
Copy-Item -LiteralPath $models -Destination (Join-Path $stage "models") -Recurse

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $archive -Force
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    $entries = @($zip.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
    foreach ($required in @("what-a-relief.exe", "README.md", "THIRD_PARTY_LICENSES.txt", "models/psfcn_3_normalize.onnx", "models/psfcn_25_normalize.onnx")) {
        if ($entries -notcontains $required) {
            throw "Portable archive is missing $required"
        }
    }
    foreach ($entry in $entries) {
        if ($entry -match '(^|/)(adam-input|obj|smoke[^/]*)(/|$)') {
            throw "Portable archive contains a forbidden build/test path: $entry"
        }
    }
} finally {
    $zip.Dispose()
}

Remove-Item -LiteralPath $stage -Recurse -Force
Write-Host "Created verified portable archive: $archive"
