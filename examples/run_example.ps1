param(
    [Parameter(Mandatory=$true)]
    [string[]]$Images,

    [string]$Mask = "",
    [string]$Out = "out_example"
)

$exe = Join-Path $PSScriptRoot "..\build\Release\ps_spheres.exe"
$args = @()

foreach ($image in $Images) {
    $args += @("--image", $image)
}

$args += @("--out", $Out)

if ($Mask -ne "") {
    $args += @("--mask", $Mask)
}

& $exe @args
