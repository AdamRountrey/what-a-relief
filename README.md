# Photometric Stereo Spheres

C++ photometric stereo for 4-image or 8-image directional-light captures that include a chrome or specular highlight sphere. At startup, the program lets the user mark the sphere location and size on the first image, uses the sphere highlights to estimate one light direction per photograph, then solves object normals with direct Lambertian photometric stereo.

This is aimed at microscope-style directional illumination workflows, including Keyence-like captures where there are too few images for conventional RTI or spherical-harmonic fitting.

## Build

Install OpenCV with C++ development files, including the `highgui` module for the interactive sphere picker.

With vcpkg on Windows:

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

With a system OpenCV install:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

The executable is `ps_spheres`.

## Run

Use exactly 4 or 8 images. All images must have the same dimensions and must include the same highlight sphere.

```powershell
.\build\Release\ps_spheres.exe `
  --image capture_01.tif `
  --image capture_02.tif `
  --image capture_03.tif `
  --image capture_04.tif `
  --mask object_mask.png `
  --out out_sample
```

The first image opens in a picker window. Click-drag from the center of the highlight sphere to the edge of the sphere, then press Enter or Space. Press `R` to redraw, or Esc to cancel.

For repeat runs, use the sphere values written to `lights.csv`:

```powershell
.\build\Release\ps_spheres.exe `
  --image capture_01.tif `
  --image capture_02.tif `
  --image capture_03.tif `
  --image capture_04.tif `
  --sphere 1820 315 92 `
  --mask object_mask.png `
  --out out_sample_repeat
```

## Outputs

- `lights.csv`: selected sphere geometry, estimated highlight points, and light vectors.
- `light_vectors.csv`: just the `x,y,z` light vectors, suitable for `--lights-file`.
- `normal_rgb.png`: RGB normal map.
- `normal_x.png`, `normal_y.png`, `normal_z.png`: normal components.
- `albedo.png`: diffuse albedo estimate.
- `height.png`: normalized height preview.
- `height.pfm`: floating-point height field in pixel units.
- `residual.png`: per-pixel fit error.
- `valid_mask.png`: pixels included in the solve.

## Options

- `--mask path`: optional object mask. White pixels are solved. The selected sphere is removed from the solve mask unless `--keep-sphere` is used.
- `--srgb`: convert typical JPEG/PNG sRGB intensities into linear light before solving.
- `--highlight-percentile value`: percentile used to find the specular highlight centroid inside the selected sphere. Default: `99.8`.
- `--min-highlight value`: minimum highlight intensity after normalization. Default: `0.05`.
- `--shadow-threshold value`: ignore observations darker than this value. Default: `0.02`.
- `--integration-iterations n`: iterations for the simple height preview. Default: `800`.
- `--lights-file path`: skip sphere calibration and read known light vectors, one `x,y,z` row per image. Use a previous run's `light_vectors.csv` for batch processing.
- `--no-gui`: require `--sphere` or `--lights-file`; useful for batch processing.

## Coordinate Convention

Image coordinates are `x` right and `y` down. Light and normal vectors are stored as:

- `+x`: image right
- `+y`: image up
- `+z`: toward the camera

For a mirror sphere, the highlight normal `N` bisects the camera view direction `V` and light direction `L`:

```text
L = 2 * dot(N, V) * N - V
```

The default camera view direction is `(0, 0, 1)`.

## Practical Notes

Use raw or consistently exposed images when possible. JPEGs can work, but saturation and gamma compression make the light estimates less stable unless `--srgb` matches the capture.

The normal solve is diffuse Lambertian. Specular object pixels, deep shadows, saturation, and interreflections will show up as higher values in `residual.png`.

The height output is a preview integration of normals, not a calibrated metrology surface. For calibrated depth, add pixel pitch, optical calibration, lens distortion correction, and a more rigorous integration stage.
