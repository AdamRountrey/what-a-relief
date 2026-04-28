# What A Relief

C++ photometric stereo and relief-visualization software for 3- to 25-image directional-light captures. In calibrated mode, captures include a chrome or other mirror-like highlight sphere; the program lets the user mark the sphere location and size on the first image, estimates one light direction per photograph from the sphere highlights, then solves object normals with a direct Lambertian photometric-stereo model. In uncalibrated mode, the program skips sphere marking and estimates a relative normal field from the image stack itself.

This is aimed at microscope-style directional illumination workflows where there are too few images for conventional RTI or spherical-harmonic fitting.

What A Relief was created by Adam Rountrey with the use of AI coding tools. The AI attribution statement is in `AI_ATTRIBUTION.md`.

What A Relief is licensed under the BSD 3-Clause License. See `LICENSE`. Third-party notices for bundled Windows runtime libraries are summarized in `THIRD_PARTY_NOTICES.md`; packaged builds also include the exact vcpkg license texts in `THIRD_PARTY_LICENSES.txt`.

## How It Works

The calibrated workflow is a practical photometric-stereo pipeline with optional experimental neural assistance:

1. Load 3 to 25 registered images of the same specimen, optionally with a mask or crop.
2. Determine lighting either by marking a highlight sphere, loading a previous `lights.csv` or `light_vectors.csv`, or using the uncalibrated unknown-lighting mode.
3. Convert the image stack to grayscale working intensities and solve a classical Lambertian normal field per pixel.
4. In robust calibrated mode, reject very dark observations, reject or downweight bright outliers, and record residual and confidence-style diagnostics.
5. Optionally apply relief flattening, which subtracts a broad low-frequency slope trend so smaller topographic features stand out more clearly.
6. Optionally run experimental PS-FCN neural fusion in calibrated mode. This uses a bundled pretrained neural normal prior, fuses it with the classical normals in slope space, and writes separate classical, neural, and fused normal outputs.
7. Generate visualization products such as `normal_rgb.png`, `albedo.png`, `residual.png`, and `liquid_metal.png`.
8. If requested, integrate the classical normal field into a fast relative height preview and optional PLY mesh.

The important boundary is that neural fusion currently helps the normal-map-style outputs only. Height preview and PLY mesh generation stay on the classical geometry path so the mesh does not inherit exaggerated slopes from the neural prior.

## Scientific Scope

What A Relief is intended for exploratory surface-shape visualization and relative normal estimation. It is not a calibrated height metrology system by itself.

Calibrated sphere mode gives the most physically interpretable normals, but it still assumes a fixed camera and object, fixed exposure response across the image set, a mirror-like calibration sphere at approximately the same imaging geometry as the sample, and mostly diffuse sample reflectance for the photometric solve. Saturation, cast shadows, interreflections, specular sample pixels, changing focus, misregistration, and nonuniform illumination can all affect the result.

Uncalibrated mode is useful when no sphere is available, but its normal field, height preview, and PLY mesh are relative. They can be rotated, stretched, sheared, or slope-stabilized by the unknown-lighting ambiguity and by the program's visual-relief heuristics. Use calibrated lighting, physical scale calibration, and independent validation before interpreting geometry quantitatively.

## Build

The recommended Windows build uses Visual Studio's bundled vcpkg so the project gets its own OpenCV install. This avoids accidentally linking against Anaconda OpenCV.

From PowerShell, run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\build-vcpkg-direct-msvc.ps1
powershell.exe -ExecutionPolicy Bypass -File scripts\package-vcpkg-runtime.ps1
.\build-vcpkg-direct\what-a-relief.exe --help
```

The first build may take a while because vcpkg builds OpenCV. After that, it is cached under `build/ninja-vcpkg/vcpkg_installed`. The packaging step copies the OpenCV runtime DLLs next to `what-a-relief.exe`, so the executable can be run directly from `build-vcpkg-direct` without editing `PATH`.

If you prefer a normal CMake build, open a Visual Studio x64 Native Tools command prompt, or run `VsDevCmd.bat -arch=x64`, then configure and build:

```powershell
cmake --preset ninja-vcpkg
cmake --build --preset ninja-vcpkg-release
```

If you use a standalone OpenCV install instead, set `OpenCV_DIR` to that install's `OpenCVConfig.cmake` directory. The CMake file intentionally rejects Anaconda paths.

```powershell
cmake -S . -B build -DOpenCV_DIR=C:/opencv/build/x64/vc16/lib
cmake --build build --config Release
```

The executable target is `what-a-relief`; on Windows the direct build writes `build-vcpkg-direct\what-a-relief.exe`.

## Windows Installer

To build a per-user Windows installer after the app has been built:

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\build-windows-installer.ps1 -SkipBuild
```

To rebuild the app, package the OpenCV DLLs, and create the installer in one step:

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\build-windows-installer.ps1
```

The installer is written to `dist\What-A-Relief-0.1.0-Setup.exe`. It installs under `%LOCALAPPDATA%\Programs\What A Relief`, creates a Start Menu shortcut, and registers an uninstall entry for the current user. It does not require administrator privileges.

The installer is currently unsigned. Distribute it from a trusted release location, and expect Windows SmartScreen or antivirus tools to warn about new unsigned binaries.

GitHub Actions can also build the Windows installer. Run the **Windows Build** workflow manually to download the installer and portable ZIP as workflow artifacts, or push a version tag such as `v0.1.0` to publish those files on a GitHub Release.

## Run

Use 3 to 25 images. All images must have the same dimensions. Calibrated mode needs the same highlight sphere visible in each image; uncalibrated mode can skip the sphere and should crop or mask to the surface region.

For the GUI workflow, run the packaged executable with no arguments:

```powershell
.\build-vcpkg-direct\what-a-relief.exe
```

The setup window lets you choose the image set, output folder, lighting mode, optional previous `lights.csv` or `light_vectors.csv`, optional near-field ring-light geometry, normal solver, optional relief flattening, sRGB handling, height preview, PLY export, experimental diagnostics, optional experimental neural fusion, and the interactive relight viewer before computation starts. If you use calibrated sphere mode, mark the sphere by clicking three points on its edge, or load a previous calibration CSV to skip sphere marking. For a previous near-field ring run, load the full `lights.csv` so the ring radius, height, and pixel scale are restored along with the light vectors. If you skip the sphere, the program uses an unknown-lighting solve and requires at least 4 images. Uncalibrated mode can also crop to the surface region so shiny fixtures, the calibration sphere, and background do not contaminate the solve.

The program prints progress in the console and shows a completion dialog when outputs are written. Skipping height is faster and still writes normals, relative albedo, residual, light metadata, and `liquid_metal.png`. If experimental neural fusion is enabled, the output folder also includes separate classical, neural, and fused normal-map sets so the result can be reviewed directly. Height preview and PLY export stay on the classical geometry path in that mode. If the interactive specular relight viewer is enabled, drag in the viewer to move the virtual light, press `S` to save the current full-resolution view as `liquid_metal_custom.png`, press `R` to reset the light, or Esc to close. If height preview and PLY export are enabled, the GUI writes `surface.ply`, a 3D mesh made from the reconstructed height field.

```powershell
.\build-vcpkg-direct\what-a-relief.exe `
  --image capture_01.tif `
  --image capture_02.tif `
  --image capture_03.tif `
  --image capture_04.tif `
  --mask object_mask.png `
  --out out_sample
```

The first image opens in a picker window. Use the mouse wheel or `+`/`-` to zoom, right-drag or use `WASD`/arrow keys to pan, then click three points on the sphere edge. Press Enter or Space to accept, Backspace to remove the last point, `R` to reset the points, `0` to fit the image, or Esc to cancel.

For repeat runs, use the sphere values written to `lights.csv`:

```powershell
.\build-vcpkg-direct\what-a-relief.exe `
  --image capture_01.tif `
  --image capture_02.tif `
  --image capture_03.tif `
  --image capture_04.tif `
  --sphere 1820 315 92 `
  --mask object_mask.png `
  --out out_sample_repeat
```

## Outputs

- `lights.csv`: selected sphere geometry, estimated highlight points, light vectors, thresholds, peaks, and selected-highlight pixel counts. In uncalibrated mode this file records that no calibrated light vectors were used.
- `light_vectors.csv`: just the `x,y,z` light vectors, suitable for `--lights-file`. In uncalibrated mode it contains only the header because no physical light vectors are estimated.
- `normal_rgb.png`: 8-bit RGB visualization of the estimated normal map.
- `normal_x.png`, `normal_y.png`, `normal_z.png`: 8-bit visual encodings of the normal components. `x` and `y` are mapped from `[-1, 1]` to `[0, 255]`; `z` is mapped from `[0, 1]` to `[0, 255]`.
- `classical_normal_rgb.png`, `classical_normal_x.png`, `classical_normal_y.png`, `classical_normal_z.png`: written when experimental neural fusion is enabled. These are the classical solver normals before fusion.
- `neural_normal_rgb.png`, `neural_normal_x.png`, `neural_normal_y.png`, `neural_normal_z.png`: written when experimental neural fusion is enabled. These are the bundled PS-FCN neural-prior normals.
- `fused_normal_rgb.png`, `fused_normal_x.png`, `fused_normal_y.png`, `fused_normal_z.png`: written when experimental neural fusion is enabled. These are the slope-domain fused normals.
- `albedo.png`: normalized 8-bit relative albedo or brightness-scale estimate. It is not an absolute reflectance measurement.
- `liquid_metal.png`: normal-map chrome-style render that uses light smoothing to avoid boosting pixel-scale noise.
- `liquid_metal_custom.png`: optional render saved from the interactive relight viewer.
- `height.png`: optional normalized height preview from integrating the normal-derived slopes.
- `height.pfm`: optional floating-point relative height field. With no additional scale calibration, `x` and `y` are image pixels and `z` is a relative integrated height.
- `surface.ply`: optional binary PLY 3D mesh exported from the height preview. Vertex `x` and `y` are image pixel coordinates, `z` is the relative height preview multiplied by `--height-scale`, and vertex RGB color is grayscale albedo.
- `residual.png`: normalized per-pixel root-mean-square fit error. Higher values can indicate specular highlights, shadows, saturation, registration errors, non-Lambertian behavior, or bad light estimates.
- `valid_mask.png`: pixels included in the solve.
- `robust_weight.png`: diagnostic map from the robust calibrated solver. Darker pixels had more downweighted observations.
- `shadow_count.png`: diagnostic map showing how many images were below the shadow threshold at each solved pixel.
- `highlight_outlier_count.png`: diagnostic map showing how many observations were rejected as high-intensity outliers.
- `specular_cue_mask.png`: optional experimental mask of pixels that behaved like shiny or otherwise non-Lambertian outliers. It is a diagnostic, not a calibrated light estimate.

When neural fusion is enabled, the default `normal_rgb.png`, `normal_x.png`, `normal_y.png`, and `normal_z.png` are the fused normals.

## Options

- `--gui`: launch the GUI workflow. Running with no arguments does the same thing.
- `--image path`: add one input image. Use 3 to 25 times, or at least 4 times for `--uncalibrated`.
- `--out dir`: output directory. Default: `out`.
- `--mask path`: optional object mask. White pixels are solved. The selected sphere is removed from the solve mask unless `--keep-sphere` is used.
- `--uncalibrated`: skip sphere calibration and estimate relative normals from unknown lighting. Requires at least 4 images. `--no-sphere` is accepted as an alias.
- `--crop x y width height`: restrict the solve to a rectangular image region. Especially useful in uncalibrated mode.
- `--sphere cx cy radius`: use a known sphere circle in image pixels and skip interactive sphere marking.
- `--srgb`: convert typical JPEG/PNG sRGB intensities into linear light before solving.
- `--solver standard|robust`: calibrated normal solver. `robust` is the default and uses highlight rejection plus Huber-style reweighting when enough observations are available.
- `--high-outlier-threshold value`: normalized intensity cutoff for robust highlight/saturation rejection. Default: `0.98`.
- `--near-field-ring radius height`: use a simple point-light ring model for calibrated solving. Radius and height are in millimeters.
- `--pixel-scale-mm value`: pixel size in millimeters per pixel for near-field solving. Use `0` to auto-read supported TIFF physical scale tags.
- `--specular-diagnostics`: write experimental shiny-cue diagnostics such as `specular_cue_mask.png`.
- `--neural-fusion`: run bundled PS-FCN neural inference and slope-domain fusion after the classical calibrated solve. Supports 3 to 25 calibrated images.
- `--neural-model path`: override the bundled PS-FCN ONNX model path, or point at a directory containing the count-specific ONNX files.
- `--flatten none|gentle|strong`: optional low-frequency slope flattening before outputs are written. Default: `none`.
- `--open-relight`: open the interactive specular relight viewer after GUI processing.
- `--highlight-percentile value`: percentile used to find the specular highlight centroid inside the selected sphere. Default: `99.8`.
- `--min-highlight value`: minimum highlight intensity after normalization. Default: `0.05`.
- `--shadow-threshold value`: ignore observations darker than this value. Default: `0.02`.
- `--integration-iterations n`: legacy option accepted for old scripts; the DCT/Poisson height preview does not iterate.
- `--no-height`: skip `height.png` and `height.pfm`. `liquid_metal.png` does not require height.
- `--mesh path.ply`: export a PLY mesh from the height preview. This forces height calculation.
- `--mesh-step n`: export every nth pixel to reduce PLY size. Default: `1`.
- `--height-scale value`: multiply mesh z coordinates. Default: `1.0`.
- `--lights-file path`: skip sphere calibration and read known light vectors. Use a previous run's full `lights.csv` to restore near-field ring metadata, or `light_vectors.csv` when only directional vectors are needed.
- `--keep-sphere`: keep the calibration sphere in the solve mask. By default, the selected sphere is removed before solving sample normals.
- `--view-dir x y z`: camera view vector used for mirror-sphere light calibration. Default: `0 0 1`.
- `--no-gui`: disable interactive selection. Use with `--sphere`, `--lights-file`, or `--uncalibrated` for batch processing.
- `--help`: show command-line help.

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

Use raw, TIFF, or other minimally processed images when possible. JPEGs can work, but compression artifacts, saturation, and gamma compression make the light estimates less stable unless `--srgb` matches the capture. The program converts color images to luminance before solving; it does not use color channels independently.

The default input interpretation treats pixel values as already linear with intensity. For typical sRGB JPEG/PNG images, use `--srgb`; this applies a simple gamma-2.2 linearization rather than a full camera-profile correction. For scientific imaging, a linear, dark-corrected, consistently exposed stack is preferable.

The calibrated normal solve is diffuse Lambertian. At each pixel it uses only observations above `--shadow-threshold` and requires at least 3 usable observations. The default robust solver estimates a per-pixel high-outlier threshold from the local observation distribution, caps it with `--high-outlier-threshold`, rejects very bright observations when enough images remain, and downweights large residuals with a Huber-style iteratively reweighted fit. Specular object pixels, deep shadows, saturation, interreflections, and misregistration can still show up as higher values in `residual.png`.

The default lighting model treats each image as a single directional light. `--near-field-ring` instead treats the calibrated light azimuths as point lights on a ring, using the given ring radius and light height in millimeters. The program keeps the ring geometry in millimeters and converts image pixel coordinates into millimeters using `--pixel-scale-mm`; if the value is `0`, it tries to read common TIFF physical scale tags from the first input image. This can better approximate close microscope ring lights, but it is still a geometric approximation unless the pixel scale and ring geometry match the actual optical setup.

`--specular-diagnostics` is experimental. It writes a mask of pixels whose observations were rejected or strongly downweighted by the robust solver, which can help identify shiny scene features. The current application does not yet use those pixels to solve lighting directions.

Experimental neural fusion uses bundled PS-FCN ONNX exports for 3 through 25 calibrated images. The network provides a qualitative dense normal prior, which What A Relief fuses with the classical normals in slope space using a confidence term derived from the classical residual and robust diagnostics. Because this neural prior was not trained for near-field microscope metrology, the fused normals are intended for visualization-oriented normal products. Height preview and PLY export remain on the classical geometry path in this mode, and the app warns about that when neural fusion is enabled.

Optional relief flattening removes a broad low-frequency slope trend from the normal field so smaller surface features stand out. This is a visualization control, not calibrated metrological form removal; leave it off when large-scale curvature or tilt is part of the scientific question.

Uncalibrated no-sphere mode is useful for visual relief enhancement when a highlight sphere is not available, but its normals, height, and PLY mesh are relative. The solver stabilizes the visual-relief slopes and removes a best-fit plane from the height preview, but the remaining unknown-lighting ambiguity can still stretch, shear, or rotate the recovered relief. In the current implementation the displayed height preview also has a best-fit plane removed after integration so broad ramps do not dominate the visualization. Use sphere calibration when geometry needs to be physically meaningful.

The height output is an optional DCT/Poisson preview integration of normals, not a calibrated metrology surface. It is fast and useful for visualization, but calibrated depth requires pixel pitch, optical calibration, lens distortion correction, validation of the light directions, and a more rigorous treatment of boundary conditions and missing data.

PLY meshes are written as binary little-endian files for faster export. They use image pixel coordinates for `x` and `y` and the relative height preview for `z`. Use `--height-scale` for visual exaggeration and `--mesh-step` to keep large captures from producing enormous files.

In the interactive specular relight viewer, drag near the image edges for very low raking light that emphasizes broad topographic features.

## References and Credits

This project stands on a long line of photometric stereo and shape-reconstruction work:

- Robert J. Woodham introduced photometric stereo for estimating surface orientation from multiple images with fixed view and varying illumination: "Photometric Method for Determining Surface Orientation from Multiple Images," Optical Engineering 19(1), 139-144, 1980. DOI: https://doi.org/10.1117/12.7972479
- Ronen Basri, David Jacobs, and Ira Kemelmacher developed photometric stereo under general unknown lighting, which informed the no-sphere experimental mode here: "Photometric Stereo with General, Unknown Lighting," International Journal of Computer Vision 72(3), 239-257, 2007. DOI: https://doi.org/10.1007/s11263-006-8815-7
- Peter J. Huber's robust statistics work is the basis for the Huber-style reweighting used by the robust calibrated solver: "Robust Estimation of a Location Parameter," The Annals of Mathematical Statistics 35(1), 73-101, 1964. DOI: https://doi.org/10.1214/aoms/1177703732
- Satoshi Ikehata, David Wipf, Yasuyuki Matsushita, and Kiyoharu Aizawa, and separately Lun Wu, Arvind Ganesh, Boxin Shi, Yasuyuki Matsushita, Yongtian Wang, and Yi Ma, provide important robust photometric-stereo references for treating shadows, specularities, and sparse corruptions. See `docs\references.bib` for full citations.
- Tony Lindeberg's scale-space work and ISO 16610-61's areal Gaussian filtering standard are relevant background for the optional low-frequency relief-flattening control. This implementation is only a practical visualization filter.
- Robert T. Frankot and Rama Chellappa's integrability work is part of the background for turning normal/slope fields into coherent surfaces: "A Method for Enforcing Integrability in Shape from Shading Algorithms," IEEE TPAMI 10(4), 439-451, 1988. DOI: https://doi.org/10.1109/34.3909
- Tal Simchony, Rama Chellappa, and M. Shao's direct Poisson solvers using fast orthogonal transforms inspired the fast DCT/Poisson height preview: "Direct Analytical Methods for Solving Poisson Equations in Computer Vision Problems," IEEE TPAMI 12(5), 435-446, 1990. DOI: https://doi.org/10.1109/34.55103
- Guanying Chen, Kai Han, and Kwan-Yee K. Wong developed PS-FCN, the pretrained neural photometric-stereo model used here as an optional qualitative normal prior for experimental fusion: "PS-FCN: A Flexible Learning Framework for Photometric Stereo," ECCV 2018. DOI: https://doi.org/10.1007/978-3-030-01252-6_1

Relight and RelightLab from the CNR-ISTI Visual Computing Lab are acknowledged as important related RTI software. Comparing against Relight helped identify useful workflow ideas, especially explicit robust-normal and flattening controls. What A Relief does not include Relight source code. Relight is available at https://github.com/cnr-isti-vclab/relight, with software releases archived on Zenodo.

The implementation uses OpenCV for image I/O, image processing, DNN inference, and GUI windows, and vcpkg/CMake to make the Windows OpenCV dependency reproducible. Optional experimental neural-fusion builds bundle PS-FCN-derived ONNX assets under the upstream MIT license; see `THIRD_PARTY_NOTICES.md`, `assets\models\NOTICE.txt`, and `assets\models\LICENSE-PS-FCN.txt`.

BibTeX entries for the academic references are in `docs\references.bib`.
