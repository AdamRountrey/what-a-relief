param(
    [Parameter(Mandatory=$true)]
    [string[]]$Images,

    [string]$Mask = "",
    [string]$Out = "out_example",
    [switch]$Uncalibrated,
    [int[]]$Crop = @()
)

$exe = Join-Path $PSScriptRoot "..\build-vcpkg-direct\what-a-relief.exe"
$args = @()

foreach ($image in $Images) {
    $args += @("--image", $image)
}

$args += @("--out", $Out)

if ($Mask -ne "") {
    $args += @("--mask", $Mask)
}

if ($Uncalibrated) {
    $args += "--uncalibrated"
}

if ($Crop.Count -eq 4) {
    $args += @("--crop", $Crop[0], $Crop[1], $Crop[2], $Crop[3])
} elseif ($Crop.Count -ne 0) {
    throw "Crop must have exactly four integers: x y width height."
}

& $exe @args
