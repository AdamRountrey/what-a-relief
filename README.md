# what-a-relief

C++ photometric stereo and relief-visualization software for directional-light captures with at least three images. In calibrated mode, captures include a chrome or other mirror-like highlight sphere; the program lets the user mark the sphere location and size on the first image, estimates one light direction per photograph from the sphere highlights, then solves object normals with a direct Lambertian photometric-stereo model. In uncalibrated mode, the program skips sphere marking and estimates a relative normal field from the image stack itself. The classical calibrated and uncalibrated paths have no fixed upper image-count limit; the optional PS-FCN neural path remains limited to 3 through 25 images by its bundled count-specific models.

This is aimed at microscope-style directional illumination workflows. The normal solve does not require dense RTI or spherical-harmonic sampling; optional RTI export uses a reduced appearance model for small image sets.

what-a-relief was created by Adam Rountrey with the use of AI coding tools. The AI attribution statement is in `AI_ATTRIBUTION.md`.

what-a-relief is licensed under the BSD 3-Clause License. See `LICENSE`. Third-party notices for bundled Windows runtime libraries are summarized in `THIRD_PARTY_NOTICES.md`; packaged builds also include the exact vcpkg license texts in `THIRD_PARTY_LICENSES.txt`.

## How It Works

The calibrated workflow is a practical photometric-stereo pipeline with optional experimental neural assistance:

1. Load at least 3 registered images of the same specimen, optionally with a mask or crop.
2. Determine lighting either by marking a highlight sphere, loading a previous `lights.csv` or `light_vectors.csv`, or using the uncalibrated unknown-lighting mode.
3. Convert the image stack to grayscale working intensities and solve a classical Lambertian normal field per pixel.
4. In robust calibrated mode, fit deterministic three-light hypotheses, classify model-relative shadows, highlights, and clipping, refine with Cauchy-weighted least squares, and record confidence-style diagnostics.
5. Optionally apply relief flattening, which subtracts a broad low-frequency slope trend so smaller topographic features stand out more clearly.
6. Optionally run experimental PS-FCN neural fusion in calibrated mode. This uses a bundled pretrained neural normal prior, fuses it with the classical normals in slope space, and writes separate classical, neural, and fused normal outputs.
7. Generate visualization products such as `normal_rgb.png`, `albedo.png`, `residual.png`, and `liquid_metal.png`.
8. If requested, integrate the classical normal field into a relative height preview and optional PLY mesh.
9. Optionally and experimentally, use coherent cast-shadow boundaries to propose a guarded broad correction to height and mesh geometry.
10. As a separate experimental choice, run an isolated Mitsuba differentiable-rendering backend that proposes a smooth correction to the classical height field, models spatially varying diffuse and glossy response, and accepts it only through training and withheld-light checks.
11. If requested, export a PTM-style RTI appearance package for Relight/OpenLIME, including a DeepZoom layout option for tiled web viewing.

The important boundary is that neural fusion currently helps the normal-map-style outputs only. Height preview and PLY mesh generation stay on the classical geometry path so the mesh does not inherit exaggerated slopes from the neural prior.

## Scientific Scope

what-a-relief is intended for exploratory surface-shape visualization and relative normal estimation. It is not a calibrated height metrology system by itself.

Calibrated sphere mode gives the most physically interpretable normals, but it still assumes a fixed camera and object, fixed exposure response across the image set, a mirror-like calibration sphere at approximately the same imaging geometry as the sample, and mostly diffuse sample reflectance for the photometric solve. Saturation, cast shadows, interreflections, specular sample pixels, changing focus, misregistration, and nonuniform illumination can all affect the result.

The experimental cast-shadow height refinement does not make height metrological. It can adjust only broad quadratic-form drift, requires at least six calibrated robust views with coherent cast-shadow evidence, and may deliberately leave height unchanged. Its ray model assumes an opaque single-valued height field and simplified lighting; inspect its audits and validate physical claims independently.

The optional Mitsuba inverse refinement is also not metrology. It retains fixed image-plane vertex coordinates, optimizes only a coarse height correction from the classical initialization, approximates the fixed microscope camera with a long-distance narrow-field perspective camera, and uses idealized directional or isotropic point lights. A lower rendered-image loss does not prove a more accurate surface. The candidate is written separately and may be rejected unchanged when withheld lights or deformation bounds disagree.

Uncalibrated mode is useful when no sphere is available, but its normal field, height preview, and PLY mesh are relative. They can be rotated, stretched, sheared, or slope-stabilized by the unknown-lighting ambiguity and by the program's visual-relief heuristics. Use calibrated lighting, physical scale calibration, and independent validation before interpreting geometry quantitatively.

## Build

The recommended Windows build uses Visual Studio's bundled vcpkg so the project gets its own OpenCV install. This avoids accidentally linking against Anaconda OpenCV.

From PowerShell, run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\build-vcpkg-direct-msvc.ps1
powershell.exe -ExecutionPolicy Bypass -File scripts\package-vcpkg-runtime.ps1
.\build-vcpkg-direct\what-a-relief.exe --help
```

The first build may take a while because vcpkg builds OpenCV. After that, it is cached under `build/ninja-vcpkg/vcpkg_installed`. The direct build compiles the distributable executable with MSVC release optimization, and the packaging step copies the OpenCV runtime DLLs next to `what-a-relief.exe`, so it can be run directly from `build-vcpkg-direct` without editing `PATH`.

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

The executable target is `what-a-relief`; on Windows the direct build writes `build-vcpkg-direct\what-a-relief.exe`. The direct build and installer accept `-Version` (currently `0.2.1`); CMake accepts `-DWHAT_A_RELIEF_VERSION_STRING=0.2.1`. This version is recorded in run manifests. When packaging with `-SkipBuild`, use the same version as the already-built executable.

### Scientific regression tests

The CMake build includes deterministic quantitative checks for radiometry; image-bound calibration reuse; lighting conditioning; 3-, 4-, 8-, 25-, and 64-light normal recovery; hard shadows, penumbrae, saturated highlights, multiple outliers, and broad glossy failure behavior; a committed three-scene Mitsuba development/validation corpus with varied primitives, relief, textures, materials, directional and finite ring lighting, reference normals, and per-light shadow, highlight, and clipping masks; near-field ring correction; guarded cast-shadow height refinement for sphere-derived directional and near-field light geometry; finite-emitter probability and explicit-Z behavior; unit and height-offset invariance; rejection without evidence; and renderer-truth no-op behavior; neural evidence masks; height integration and flattening; TIFF scale units; RTI reconstruction and Deep Zoom placement; checked writes; run manifests; and watertight printable PLY topology. Normal CI does not install or invoke the optional Mitsuba runtime. It uses committed renderer fixtures and a dependency-free fake-process contract test for the inverse handoff. Three additional tests drive the real executable through calibrated classical, bundled-PS-FCN, and uncalibrated workflows, including manifest version checks. Run all release gates after building:

```powershell
ctest --test-dir build\ninja-vcpkg --output-on-failure
```

GitHub Actions runs these checks before creating an installer or portable archive. A successful compile by itself is not considered a passing build. Exact acceptance thresholds, current reference values, known limits, and the reproducible performance benchmark are in [`docs/validation.md`](docs/validation.md); the durable release checklist is in [`docs/quality-roadmap.md`](docs/quality-roadmap.md).

## Windows Installer

To build a per-user Windows installer after the app has been built:

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\build-windows-installer.ps1 -SkipBuild
```

To rebuild the app, package the OpenCV DLLs, and create the installer in one step:

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\build-windows-installer.ps1
```

The installer is written to `dist\what-a-relief-0.2.1-setup.exe` by default. It installs under `%LOCALAPPDATA%\Programs\what-a-relief`, creates a Start Menu shortcut, and registers an uninstall entry for the current user. It does not require administrator privileges.

The installer is currently unsigned. Distribute it from a trusted release location, and expect Windows SmartScreen or antivirus tools to warn about new unsigned binaries.

### Optional Mitsuba backend

The standard installer remains self-contained and does not include the larger inverse-rendering runtime. Experimental inverse refinement uses a separate, self-contained add-on so ordinary users do not inherit it. Build the add-on installer with:

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\build-mitsuba-backend-installer.ps1
```

Run `dist\what-a-relief-0.2.1-mitsuba-backend-setup.exe` after installing the main application. It installs a private CPython 3.13.13 runtime, pinned Mitsuba 3.8.0, Dr.Jit 1.3.1 and NumPy 2.3.3 wheels, and LLVM 18.1.6 under `%LOCALAPPDATA%\Programs\what-a-relief-mitsuba`. Users do not install Python, pip, Conda, Anaconda, or LLVM. The installer verifies pinned build-input hashes and performs real CPU and optional NVIDIA CUDA render probes before replacing an existing backend. The main application finds this location automatically after it is restarted; advanced users may still locate another compatible private runtime explicitly.

GitHub Actions can also build the Windows application installer, portable ZIP, and self-contained Mitsuba backend installer. Run the **Windows Build** workflow manually to download them as workflow artifacts, or push a version tag such as `v0.2.1` to publish those files on a GitHub Release. Tagged builds stamp the executable manifest and installers with the same tag-derived version. The package builders verify required models, backend components, and license files; the portable ZIP is assembled from an explicit runtime whitelist, so old smoke runs or input data in the build folder cannot enter the artifact.

## Run

Use at least 3 images, or at least 4 for uncalibrated mode. All images must have the same dimensions. Calibrated mode needs the same highlight sphere visible in each image unless a previous calibration is loaded; uncalibrated mode can skip the sphere and should crop or mask to the surface region. More images increase image memory, robust-solver work, diagnostic file count, and RTI fitting time.

For the GUI workflow, run the packaged executable with no arguments:

```powershell
.\build-vcpkg-direct\what-a-relief.exe
```

The setup window lets you choose the image set, output folder, lighting mode, optional previous `lights.csv` or `light_vectors.csv`, optional near-field ring-light geometry, image scale, optional specimen mask for height and mesh, normal solver, optional relief flattening, sRGB handling, height solver, height drift correction, height preview, PLY export, optional printable-surface hole filling, optional RTI export, experimental diagnostics, optional cast-shadow height refinement, optional experimental Mitsuba inverse refinement, optional experimental neural fusion, and the interactive relight viewer before computation starts. If you use calibrated sphere mode, mark the sphere by clicking three points on its edge, or load a previous calibration CSV to skip sphere marking. For a previous near-field ring run, load the full `lights.csv` so the ring radius, height, effective LED diameter, and pixel scale are restored along with the light vectors. If its recorded image names do not match the new image set, the GUI can instead apply its rows in order after an explicit warning and confirmation. If you skip the sphere, the program uses an unknown-lighting solve and requires at least 4 images. Uncalibrated mode can also crop to the surface region so shiny fixtures, the calibration sphere, and background do not contaminate the solve. The specimen height mask is separate from the solve mask: it limits only `height.png`, `height.pfm`, `height_mask.png`, and optional PLY export. Mitsuba refinement and cast-shadow height refinement are alternative advanced geometry experiments; the GUI will not run both in one job.

The program prints progress in the console and shows progress in the GUI while long operations run. Skipping height is faster and still writes normals, relative albedo, residual, light metadata, and `liquid_metal.png`. If experimental neural fusion is enabled, the output folder also includes separate classical, neural, and fused normal-map sets so the result can be reviewed directly. Height preview, `surface.ply`, and `printable_surface.ply` stay on the classical geometry path in that mode. If the interactive specular relight viewer is enabled, drag in the viewer to move the virtual light, press `S` to save the current full-resolution view as `liquid_metal_custom.png`, press `R` to reset the light, or Esc to close. If height preview and PLY export are enabled, the GUI writes `surface.ply`, an open inspection mesh made from the reconstructed height field. If printable export is enabled, it also writes `printable_surface.ply`, a watertight solid PLY with a flat base. The optional smart-fill checkbox reconstructs enclosed missing surface pixels only for that printable mesh and writes `printable_fill_mask.png` to disclose every synthesized pixel.

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

The image-scale line picker also uses a zoomable window. Click `Draw Scale Line...`, click two endpoints on the first image, then enter the known length in millimeters. The program computes the shared `mm/pixel` value used by near-field lighting and printable mesh exports.

The GUI height-mask tool uses a similar zoomable window. Click around the specimen boundary and press Enter or Space; the program then shows progress while it searches for a nearby image edge, then previews the edge-refined boundary in green over your original outline in yellow. Press Enter or Space to accept the refined boundary, `B` to use your original outline, `R` to return to editing, or Esc to cancel. Before integration, a custom mask is intersected with valid geometry and eroded with a 5 x 5 elliptical kernel (up to two image pixels) to avoid uncertain boundary slopes, provided at least 100 pixels remain. `height_mask.png` records that final domain.

For repeat runs, use the sphere values written to `lights.csv`. The full file associates vectors with image names and is safely reordered to match the newly selected image order; moved datasets are matched by unique filename. In the GUI, missing or ambiguous names offer an explicit row-order fallback only when the file still contains one valid vector per selected image. The warning defaults to Cancel because accepting it assumes the new image set has exactly the same lighting order as the older run. Command-line use of a named `lights.csv` remains identity-matched. The compact `light_vectors.csv` has no names, so its rows must already be in exactly the same order as the selected images:

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

- `run_manifest.json`: authoritative status and provenance summary for the run. It is written as `in_progress` before computation and changed to `complete` only after every requested output is present and nonempty. It records application version, input paths/sizes/timestamps, parameters, light vectors, solve conditioning and coverage, and generated files/sizes; it is not a cryptographic content-hash record.
- `lights.csv`: selected sphere geometry, estimated highlight points, light vectors, thresholds, peaks, and selected-highlight pixel counts. In uncalibrated mode this file records that no calibrated light vectors were used.
- `light_vectors.csv`: just the `x,y,z` light vectors, suitable for `--lights-file`. In uncalibrated mode it contains only the header because no physical light vectors are estimated.
- `normal_rgb.png`: 8-bit RGB visualization of the estimated normal map.
- `normal_x.png`, `normal_y.png`, `normal_z.png`: 8-bit visual encodings of the normal field. `x` and `y` are mapped from `[-1, 1]` to `[0, 255]`; `z` is mapped from `[0, 1]` to `[0, 255]`.
- `hillshade_ul.png`: cartographic-style hillshade from the image upper-left.
- `classical_normal_rgb.png`, `classical_normal_x.png`, `classical_normal_y.png`, `classical_normal_z.png`, `classical_hillshade_ul.png`: written when experimental neural fusion is enabled. These are the classical solver normals before fusion.
- `neural_normal_rgb.png`, `neural_normal_x.png`, `neural_normal_y.png`, `neural_normal_z.png`, `neural_hillshade_ul.png`: written when experimental neural fusion is enabled. These are the bundled PS-FCN neural-prior normals.
- `neural_valid_mask.png`: pixels where the neural prior had a finite normal and at least three finite, above-shadow-threshold image observations. Neural predictions outside this evidence mask are not fused or exported as valid surface.
- `fused_normal_rgb.png`, `fused_normal_x.png`, `fused_normal_y.png`, `fused_normal_z.png`, `fused_hillshade_ul.png`: written when experimental neural fusion is enabled. These are the slope-domain fused normals.
- `albedo.png`: normalized 8-bit relative albedo or brightness-scale estimate. It is not an absolute reflectance measurement.
- `liquid_metal.png`: normal-map chrome-style render that uses light smoothing to avoid boosting pixel-scale noise.
- `liquid_metal_custom.png`: optional render saved from the interactive relight viewer.
- `height.png`: optional normalized height preview from integrating the normal-derived slopes.
- `height.pfm`: optional floating-point relative height field in image-pixel height units. Pixel scale and `--height-scale` do not rescale this file; physical scaling is applied during printable PLY export.
- `height_mask.png`: pixels used for height integration and PLY export. This matches `valid_mask.png` unless a specimen height mask was supplied or drawn.
- `shadow_height_correction.png` and `shadow_height_correction.pfm`: written when experimental cast-shadow height refinement is requested. The PNG is a neutral-gray signed visualization; the PFM stores the proposed correction in height-pixel units. Both are zero when the safety gate rejects the correction.
- `shadow_constraint_count.png`, `shadow_mismatch_before.png`, and `shadow_mismatch_after.png`: spatial audits of the cast-shadow boundary support and model disagreement. A rejected run keeps the before and after mismatch equal when those measurements are available.
- `shadow_observability.png`: confidence-weighted support from pixels that alternate between lit and shadowed observations across useful illumination azimuths.
- `shadow_edge_support.png`: confidence-weighted support from regularized cast-shadow boundaries. This is an audit map, not a recovered shadow correspondence.
- `shadow_occluder_support.png`: geometry allowed to occlude light during shadow prediction.
- `shadow_refinement/`: per-light regularized observed cast-shadow masks, evidence confidence, and predicted masks and probabilities before and after an evaluated correction. The manifest's `diagnostics.shadow_height_refinement.decision` is authoritative when evidence was insufficient before every audit could be computed.
- `surface.ply`: optional binary PLY 3D mesh exported from the height preview. Vertex `x` and `y` are image pixel coordinates, `z` is the relative height preview multiplied by `--height-scale`, and vertex RGB color is grayscale albedo.
- `printable_surface.ply`: optional binary PLY watertight solid for 3D printing. XY scale is required and coordinates are written in millimeters. The top surface uses the same height preview, the bottom is flat, and boundary edges are closed around the specimen mask.
- `printable_fill_mask.png`: written when printable smart filling is enabled; white pixels identify enclosed gaps synthesized in `printable_surface.ply`. This mask does not alter the height products or open inspection mesh.
- `rti/`: optional RTI package. The `image` and `deepzoom` layouts target Relight/OpenLIME-style PTM exports. In RGB mode, small 3-to-8-image stacks use a stable 3-term PTM and write `plane_0.jpg` through `plane_2.jpg`; better-constrained stacks may use the 6-term PTM and write through `plane_5.jpg`. In LRGB mode, `plane_0` is a base image and the remaining planes store luminance PTM coefficients, so 3-term stacks write `plane_0.jpg` and `plane_1.jpg`. The DeepZoom layout writes `info.json`, `plane_*.dzi`, and `plane_*_files` tile folders. The `webrti` layout targets webRTIViewer and writes `info.xml` plus component tiles named like `1_1.jpg`, `1_2.jpg`, and so on.
- `residual.png`: normalized per-pixel root-mean-square fit error. Higher values can indicate specular highlights, shadows, saturation, registration errors, non-Lambertian behavior, or bad light estimates.
- `valid_mask.png`: pixels included in the solve.
- `robust_weight.png`: diagnostic map from the robust calibrated solver. Darker pixels had more downweighted observations.
- `robust_fallback_mask.png`: white where only three effective inliers remained, so the robust fit had no redundant observation.
- `robust_unsupported_mask.png`: white where fewer than three credible observations remained, the local light geometry was ill-conditioned, or no valid front-facing fit could be supported. These pixels are not force-fit.
- `robust_inlier_count.png`: effective inlier count scaled by the number of input images.
- `robust_local_condition.png`: local weighted light-matrix condition number, clipped at 100 for display. Brighter values indicate less stable local geometry.
- `shadow_count.png`: count of observations classified as fitted-model shadow candidates. Merely falling below the absolute threshold is not enough to receive the label, but low-albedo glossy regions can still be ambiguous; this is a weighting diagnostic rather than semantic segmentation.
- `highlight_outlier_count.png`: count of brighter-than-model observations consistent with a specular half-vector cue, including definite clipping when it lies above the fitted diffuse prediction.
- `saturation_count.png`: count of input samples exactly clipped at the integer container maximum. It is a definite-clipping diagnostic, not a complete detector for sensors whose ADC maximum occupies only part of a wider file container.
- `model_mismatch_count.png`: count of large residual observations that do not have enough physical evidence to label as shadow or highlight.
- `specular_cue_mask.png`: optional experimental union of pixels with classified highlight evidence. It is a diagnostic, not a recovered BRDF or calibrated light estimate.
- `robust_observations/`: written with `--specular-diagnostics`; contains one shadow, highlight, and definite-saturation mask per input light so classifications can be audited against the source stack.

When neural fusion is enabled, the default `normal_rgb.png`, `normal_x.png`, `normal_y.png`, `normal_z.png`, and `hillshade_ul.png` are based on the fused normals.

## Options

- `--gui`: launch the GUI workflow. Running with no arguments does the same thing.
- `--image path`: add one input image. Use at least 3 times, or at least 4 times for `--uncalibrated`. Classical processing has no fixed upper count; neural fusion supports 3 through 25.
- `--out dir`: output directory. Default: `out`.
- `--mask path`: optional object mask. White pixels are solved. The selected sphere is removed from the solve mask unless `--keep-sphere` is used.
- `--height-mask path`: optional specimen mask used only for height integration, `height_mask.png`, and PLY export. It does not change normals, albedo, residuals, liquid-metal renders, relighting, or solve diagnostics.
- `--uncalibrated`: skip sphere calibration and estimate relative normals from unknown lighting. Requires at least 4 images. `--no-sphere` is accepted as an alias.
- `--crop x y width height`: restrict the solve to a rectangular image region. Especially useful in uncalibrated mode.
- `--sphere cx cy radius`: use a known sphere circle in image pixels and skip interactive sphere marking.
- `--srgb`: convert typical JPEG/PNG sRGB intensities into linear light before solving.
- `--solver standard|robust`: calibrated normal solver. `robust` is the default and uses deterministic minimal-subset consensus, physical observation classification, and Cauchy-weighted refinement.
- `--high-outlier-threshold value`: normalized probable-clipping/bright-candidate cutoff used by robust initialization. Exact integer-container clipping is tracked separately. Default: `0.98`.
- `--near-field-ring radius height`: use a simple point-light ring model for calibrated solving. Radius and height are in millimeters.
- `--pixel-scale-mm value`: image pixel size in millimeters per pixel. Use `0` to auto-read supported TIFF physical scale tags when a physical scale is required.
- `--specular-diagnostics`: with the calibrated robust solver, write the aggregate specular cue and per-light shadow, highlight, and saturation masks.
- `--shadow-height-refinement`: experimentally propose a broad height/mesh-only correction from coherent cast-shadow boundaries. Requires at least 6 calibrated images, the robust solver, and height calculation. It may leave height unchanged when evidence or withheld-light validation is insufficient.
- `--shadow-reference-z-mm value`: for the near-field ring model only, set the reference surface height above the ring-calibration datum in millimeters. Default: `0`.
- `--shadow-led-diameter-mm value`: for the near-field ring model only, approximate an effective circular emitter of this diameter in millimeters. Use `0` for a point source. Default: `0`.
- `--neural-fusion`: run bundled PS-FCN neural inference and slope-domain fusion after the classical calibrated solve. Supports 3 to 25 calibrated images.
- `--neural-model path`: override the bundled PS-FCN ONNX model path, or point at a directory containing the count-specific ONNX files.
- `--neural-max-side n`: long-side pixel limit for PS-FCN inference. Default: `2048`; use `0` to try native input size first. If OpenCV DNN fails at the requested size, the program retries at lower resolutions. If every attempt fails, the requested run fails and its manifest remains `in_progress`; it is not reported as a successful neural run with classical-only outputs.
- `--flatten none|gentle|strong`: optional low-frequency slope flattening before outputs are written. Default: `none`.
- `--open-relight`: open the interactive specular relight viewer after GUI processing.
- `--highlight-percentile value`: percentile used to find the specular highlight centroid inside the selected sphere. Default: `99.8`.
- `--min-highlight value`: minimum highlight intensity after normalization. Default: `0.05`.
- `--shadow-threshold value`: ignore observations darker than this value. Default: `0.02`.
- `--integration-iterations n`: controls the work budget for the robust masked height solver. The fast DCT/Poisson solver accepts this option for old scripts but does not iterate.
- `--height-solver robust|fast`: choose height integration method. `robust` is the default masked, weighted solver; `fast` uses the older DCT/Poisson preview.
- `--height-flatten none|plane|radial|quadratic`: optional height/PLY-only form removal after normal integration. `none` leaves height untouched; `plane` subtracts a least-squares affine surface; `radial` jointly subtracts a plane and broad dome term; `quadratic` subtracts a full second-order surface. Default: `none`.
- `--height-slope-cap value`: clamp extreme normal-derived slopes before height integration. Default: `3.0`; use `0` to disable. This affects only `height.png`, `height.pfm`, and PLY export.
- `--no-height`: skip `height.png` and `height.pfm`. `liquid_metal.png` does not require height.
- `--mesh path.ply`: export a PLY mesh from the height preview. This forces height calculation.
- `--printable-mesh path.ply`: export a watertight solid PLY for 3D printing. This forces height calculation and requires `--pixel-scale-mm`, TIFF scale metadata, or a GUI scale line.
- `--printable-fill-holes`: reconstruct enclosed missing surface pixels in `--printable-mesh`; boundary-connected gaps remain open in the top outline and are closed by the normal printable side walls.
- `--mesh-step n`: export every nth pixel to reduce PLY size. Default: `1`.
- `--height-scale value`: multiply mesh z coordinates. Default: `1.0`.
- `--printable-thickness-mm value`: base thickness for `--printable-mesh`. Default: `2.0`.
- `--rti path`: export a Relight/OpenLIME PTM-style RTI package. Requires at least 3 calibrated or loaded light directions.
- `--rti-layout image|deepzoom|webrti`: choose RTI package layout. `image` writes full-size coefficient plane JPEGs; `deepzoom` writes tiled DeepZoom pyramids; `webrti` writes a webRTIViewer-compatible `info.xml` plus quadtree component JPEGs. Default: `image`.
- `--rti-color rgb|lrgb`: choose RTI color model. `rgb` fits separate color coefficient planes. `lrgb` writes a base RGB image plus luminance PTM coefficients, following the PTM LRGB convention used by Relight/OpenLIME. Default: `rgb`.
- `--lights-file path`: skip sphere calibration and read known light vectors. Use a previous run's full `lights.csv` to restore near-field ring metadata, or `light_vectors.csv` when only directional vectors are needed. Named rows are identity-matched in command-line runs; the GUI provides the confirmed row-order fallback for renamed image sets.
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

The default input interpretation treats pixel values as already linear with intensity. For typical sRGB JPEG/PNG images, use `--srgb`; this applies the standard piecewise sRGB decoding function to each color channel before luminance conversion. The complete stack is then normalized by one common multiplicative scale, preserving relative intensities between lighting directions while handling integer-container and floating-point ranges consistently. This does not replace a camera profile or radiometric calibration. For scientific imaging, a linear, dark-corrected, flat-field-corrected, consistently exposed stack is preferable.

The calibrated normal solve is diffuse Lambertian. At each pixel, the standard solver omits observations below `--shadow-threshold` and needs at least three usable observations. The robust solver starts from a deterministic, condition-screened set of three-light hypotheses, scores them against a nonnegative Lambertian prediction, recenters relative albedo from a compact observation consensus, and performs up to ten Cauchy-weighted least-squares refinements. Signed, noise-scaled residuals, neighboring ring-light responses, raw clipping, and illumination-view half-vector agreement distinguish model-supported shadows and highlights from unlabeled model mismatch. A locally weighted light matrix with condition number above 100, or fewer than three credible observations, produces an explicit unsupported pixel instead of a forced normal. Three credible observations remain mathematically solvable but provide no outlier redundancy. This approach is effective for sparse corrupted observations; it does not recover a diffuse normal when a broad glossy lobe affects most lights. Specular object pixels, deep shadows, interreflections, exposure drift, and misregistration can still bias the result.

The default lighting model treats each image as a single directional light. `--near-field-ring` instead treats the calibrated light azimuths as isotropic point lights on a ring, using the given ring radius and light height in millimeters. It recomputes direction per pixel and applies inverse-square distance falloff relative to the image center. The program keeps the ring geometry in millimeters and converts image pixel coordinates into millimeters using `--pixel-scale-mm`; if the value is `0`, it tries to read common TIFF physical scale tags from the first input image. GeoTIFF model pixel scales are converted only when a supported linear model unit is explicitly declared; unitless and angular values are not guessed. In the GUI, the same value can be set by drawing a scale line of known length on the first image. This can better approximate close microscope ring lights, but it still assumes equal LED output, a flat reference plane, orthographic projection, and no LED beam anisotropy unless those effects are separately calibrated.

`--specular-diagnostics` is experimental and requires the calibrated robust solver. It writes the aggregate cue plus auditable per-light classifications. Highlights require a positive model-relative residual and consistency with the corresponding illumination-view half vector; shadows require attached-shadow geometry or a substantial deficit relative to a positive prediction. Neighboring light responses can modestly strengthen evidence on an ordered ring, but do not override the physical tests. The cue is diagnostic only: it does not replace the normal, solve lighting directions, or make broad non-Lambertian normal estimates reliable. The analytic broad-gloss and Mitsuba mixed-material fixtures deliberately record this limitation.

`--shadow-height-refinement` is a separate experimental geometry stage for either calibrated directional lights, including directions recovered from sphere highlights, or the calibrated near-field ring model. It treats confidence-weighted shadow and anti-shadow evidence as nonlocal constraints on the already integrated height and fits only five broad correction modes. The program evaluates exact full-resolution candidate fractions of `0.25`, `0.50`, `0.75`, and `1.0`, accepts the smallest candidate that improves the fit without worsening any withheld light or the photometric normal slopes, and otherwise leaves height byte-for-byte unchanged. At least six light directions must contain coherent cast-shadow evidence; up to sixteen directions, including fully lit negative evidence, are selected around azimuth to bound runtime. Directional mode needs only calibrated light directions and expresses relative height in image-pixel units; XY scale supplies physical units downstream. The near-field ring variant reconstructs source positions from each direction's azimuth plus the ring radius and height, and additionally uses image scale, an explicit reference-surface Z, and optionally an effective finite LED diameter. Arbitrary per-image 3D source positions are not yet an input format. See `docs\algorithm.md` for the model and its limitations.

The optional Mitsuba inverse stage receives the same post-radiometric luminance observations used by the classical solve. what-a-relief encodes them temporarily as globally normalized 16-bit linear grayscale PNGs, so the isolated renderer never needs to understand the original TIFF, PNG, JPEG, or microscope-vendor container. This preserves light-to-light ratios with at most one 16-bit code of transfer quantization, avoids a second color or gamma interpretation, and fixes TIFF compatibility. The job records the original source paths and transfer metadata; the temporary observation files are removed whether the backend succeeds or fails.

Experimental neural fusion uses bundled PS-FCN ONNX exports for 3 through 25 calibrated images. The network provides a qualitative dense normal prior, which what-a-relief fuses with the classical normals in slope space using a confidence term derived from the classical residual and robust diagnostics. Neural normals are admitted only where at least three finite observations exceed the shadow threshold, and the final fused refit applies the same three-observation rule. Because this neural prior was not trained for near-field microscope metrology, the fused normals are intended for visualization-oriented normal products. Height preview and PLY export remain on the classical geometry path in this mode, and the app warns about that when neural fusion is enabled. Neural inference runs with a default long-side limit of 2048 pixels; `--neural-max-side 0` tries native input size first. The app reports the two dense input tensors' approximate memory and retries at lower resolution after allocation or OpenCV DNN failure.

Optional relief flattening removes a broad low-frequency slope trend from the normal field so smaller surface features stand out. This affects the normal visualizations, liquid-metal render, optional height preview, and optional PLY mesh. It is a visualization control, not calibrated metrological form removal; leave it off when large-scale curvature or tilt is part of the scientific question.

Uncalibrated no-sphere mode is useful for visual relief enhancement when a highlight sphere is not available, but its normals, height, and PLY mesh are relative. The solver stabilizes the visual-relief slopes, but the remaining unknown-lighting ambiguity can still stretch, shear, or rotate the recovered relief. Height flattening is explicit: `none` preserves the integrated field, while `plane`, `radial`, or `quadratic` removes the stated fitted form from height and PLY only. Use sphere calibration when geometry needs to be physically meaningful.

The height output is an optional relative integration of normals, not a calibrated metrology surface. Calibrated depth requires pixel pitch, optical calibration, lens distortion correction, validation of the light directions, and careful treatment of boundary conditions and missing data. The default robust masked solver works only inside the height mask, downweights very steep or inconsistent slope constraints, and iteratively solves a weighted Poisson-style problem so large normal errors are less likely to smear into image-wide ramps. The older `--height-solver fast` path remains available when speed matters most; it uses a DCT/Poisson preview and extends the slope field outside the mask before solving to reduce boundary jumps. When the specimen sits above a visible table or background, use the GUI height-mask tool or `--height-mask` to keep the integration and mesh domain on the specimen surface without changing the normal solve. If the integrated height still has broad tilt or edge curl, explicit plane, radial, or quadratic form removal can alter `height.png`, `height.pfm`, and PLY without changing normal maps, albedo, residuals, or liquid-metal renders.

PLY meshes are written as buffered binary little-endian files. When both mesh types are enabled without smart filling, they share one sampled topology; printable boundaries use compact grid bookkeeping rather than retaining every triangle edge. The regular `surface.ply` uses image pixel coordinates for `x` and `y` and the relative height preview for `z`. Use `--height-scale` for visual exaggeration and `--mesh-step` to keep large captures from producing enormous files; step `2` samples one quarter as many grid vertices as step `1`. The printable PLY export writes a closed solid with duplicated bottom vertices, reversed bottom faces, and side faces along every open boundary edge. Printable export requires XY scale so coordinates and base thickness are in millimeters. Optional smart filling identifies invalid components fully enclosed by the geometry domain, keeps every measured height fixed, extends nearby height gradients into each gap, and solves a slope-guided Poisson interpolation there. Normal-derived slopes are used only when a local height derivative is unavailable. Exterior and boundary-connected exclusions are preserved. This is explicitly synthesized geometry for printing, not recovered measurement, and it affects neither `height.pfm` nor `surface.ply`. Image preparation and RTI coefficient or tile generation use OpenCV's worker pool, while PNG, JPEG, and TIFF encoding remains delegated to OpenCV's native codecs with unchanged quality settings.

RTI export writes a practical PTM-style appearance package from the original color image stack and the calibrated or loaded light directions. It uses the same exact selected sRGB decoding as the normal solve and never normalizes images independently, so light-to-light intensity ratios are retained. Three-to-eight-image stacks use a stable first-order three-term subset because small ring-light captures do not constrain the quadratic PTM terms well enough; this is less expressive than a standard six-term quadratic PTM. Larger, better-conditioned stacks use all six terms. RGB mode fits independent color coefficient planes. LRGB mode follows the PTM LRGB model: it stores a base color image and fits the lighting-dependent variation as relative luminance, which can be useful when the viewer should preserve an albedo-like base appearance while relighting. LRGB is still multiplicative in standard RTI viewers, so it is not a separate additive shadow-fill layer. The coefficient fit itself is ordinary least squares and does not inherit the normal solver's robust shadow/highlight rejection. Use the plain image layout for small Relight/OpenLIME data, the Deep Zoom layout for tiled OpenLIME-style viewing, and the webRTIViewer layout when your site calls `createRtiViewer(...)` from jcupitt/webRTIViewer. This is a lossy-JPEG appearance export, not a height, mesh, or archival replacement for the source images.

Output images and data files are first written to temporary siblings and promoted only after a checked close. RTI packages are assembled in a staging directory and replace the previous package only after the new package is complete. RTI destinations must be empty or contain a previous what-a-relief package with `rti_manifest.json`; an unrelated occupied folder is rejected. Calibration and mask files reused from the output folder are preserved during startup cleanup, and output collisions with input images are rejected. If a run fails, treat any products beside an `in_progress` manifest as partial; only a `complete` manifest identifies the verified output set.

In the interactive specular relight viewer, drag near the image edges for very low raking light that emphasizes broad topographic features.

## References and Credits

This project stands on a long line of photometric stereo and shape-reconstruction work:

- Robert J. Woodham introduced photometric stereo for estimating surface orientation from multiple images with fixed view and varying illumination: "Photometric Method for Determining Surface Orientation from Multiple Images," Optical Engineering 19(1), 139-144, 1980. DOI: https://doi.org/10.1117/12.7972479
- Ronen Basri, David Jacobs, and Ira Kemelmacher developed photometric stereo under general unknown lighting, which informed the no-sphere experimental mode here: "Photometric Stereo with General, Unknown Lighting," International Journal of Computer Vision 72(3), 239-257, 2007. DOI: https://doi.org/10.1007/s11263-006-8815-7
- Martin Fischler and Robert Bolles introduced minimal-subset consensus fitting, and Paul Holland and Roy Welsch described robust regression by iteratively reweighted least squares. The implementation uses a deterministic, bounded consensus set followed by Cauchy-weighted IRLS rather than claiming either published algorithm unchanged. See `docs\references.bib` for full citations.
- Kenneth Torrance and Ephraim Sparrow's microfacet reflection analysis, and James Blinn's half-vector highlight model, provide the physical background for the optional specular cue. The implementation uses only a diagnostic half-vector consistency test, not either paper's full BRDF.
- Yvain Queau, Bastien Durix, Tao Wu, Daniel Cremers, Francois Lauze, and Jean-Denis Durou derived the nearby LED point-source model that motivates the ring mode's spatial direction and inverse-square attenuation: "LED-Based Photometric Stereo: Modeling, Calibration and Numerical Solution," Journal of Mathematical Imaging and Vision 60(3), 313-340, 2018. DOI: https://doi.org/10.1007/s10851-017-0761-1
- Satoshi Ikehata, David Wipf, Yasuyuki Matsushita, and Kiyoharu Aizawa, and separately Lun Wu, Arvind Ganesh, Boxin Shi, Yasuyuki Matsushita, Yongtian Wang, and Yi Ma, provide important robust photometric-stereo references for treating shadows, specularities, and sparse corruptions. See `docs\references.bib` for full citations.
- Manmohan Chandraker, Sameer Agarwal, and David Kriegman's ShadowCuts combines light visibility with photometric stereo and shadow-consistent surface reconstruction: "ShadowCuts: Photometric Stereo with Shadows," CVPR 2007. DOI: https://doi.org/10.1109/CVPR.2007.383288
- Austin Abrams, Kylia Miskell, and Robert Pless describe nonlocal geometric constraints from shadow correspondence; Asaf Karnieli, Ohad Fried, and Yacov Hel-Or demonstrate one-shot neural shape supervision from shadow maps; and Markus Worchel and Marc Alexa develop efficient differentiable shadow mapping for inverse graphics. These works motivate the optional cast-shadow experiment, but the implementation here is a smaller guarded height-field correction. Full citations are in `docs\references.bib`.
- Tony Lindeberg's scale-space work and ISO 16610-61's areal Gaussian filtering standard are relevant background for the optional low-frequency relief-flattening control. This implementation is only a practical visualization filter.
- Robert T. Frankot and Rama Chellappa's integrability work is part of the background for turning normal/slope fields into coherent surfaces: "A Method for Enforcing Integrability in Shape from Shading Algorithms," IEEE TPAMI 10(4), 439-451, 1988. DOI: https://doi.org/10.1109/34.3909
- Tal Simchony, Rama Chellappa, and M. Shao's direct Poisson solvers using fast orthogonal transforms inspired the fast DCT/Poisson height preview: "Direct Analytical Methods for Solving Poisson Equations in Computer Vision Problems," IEEE TPAMI 12(5), 435-446, 1990. DOI: https://doi.org/10.1109/34.55103
- Tom Malzbender, Dan Gelb, and Hans Wolters introduced Polynomial Texture Maps and the LRGB PTM representation used by many RTI viewers: "Polynomial Texture Maps," SIGGRAPH 2001, 519-528. DOI: https://doi.org/10.1145/383259.383320
- Yvain Queau, Jean-Denis Durou, and Jean-Francois Aujol's normal-integration survey and variational-methods papers informed the robust masked height solver and its treatment of non-rectangular domains and unreliable gradients. See "Normal Integration: A Survey," Journal of Mathematical Imaging and Vision 60(4), 576-593, 2018, DOI: https://doi.org/10.1007/s10851-017-0773-x, and "Variational Methods for Normal Integration," Journal of Mathematical Imaging and Vision 60(4), 609-632, 2018, DOI: https://doi.org/10.1007/s10851-017-0777-6
- Amit Agrawal, Rama Chellappa, and Ramesh Raskar's work on reconstructing from non-integrable gradient fields is relevant background for containing gradient errors instead of letting them create broad warps: "An Algebraic Approach to Surface Reconstruction from Gradient Fields," ICCV 2005, 174-181.
- Guanying Chen, Kai Han, and Kwan-Yee K. Wong developed PS-FCN, the pretrained neural photometric-stereo model used here as an optional qualitative normal prior for experimental fusion: "PS-FCN: A Flexible Learning Framework for Photometric Stereo," ECCV 2018. DOI: https://doi.org/10.1007/978-3-030-01252-6_1
- Wenzel Jakob and contributors developed Mitsuba 3 and Dr.Jit. Mitsuba 3.8.0 generated the committed mixed-material validation corpus, including GGX rough-plastic reflection, cast shadows, and reference normal AOVs. It is not part of the standard application package or invoked by CTest; the same pinned renderer is distributed only in the optional inverse-rendering add-on. See `tests\fixtures\mitsuba\README.md` and `THIRD_PARTY_NOTICES.md`.
- IEC 61966-2-1 and ITU-R BT.709 define the selected sRGB transfer function and RGB luminance coefficients used in input conversion. TIFF 6.0 and OGC GeoTIFF 1.1 define the physical-resolution and declared model-unit metadata interpreted for pixel scale.

Relight and RelightLab from the CNR-ISTI Visual Computing Lab are acknowledged as important related RTI software. Comparing against Relight helped identify useful workflow ideas, especially explicit robust-normal controls, flattening controls, optional radial/quadratic surface flattening for height or mesh outputs, and Relight/OpenLIME-style RTI package export. what-a-relief does not include Relight source code. Relight is available at https://github.com/cnr-isti-vclab/relight, with software releases archived on Zenodo.

webRTIViewer and webGLRTIMaker from Gianpaolo Palma, Marco Di Benedetto, CNR-ISTI, and John Cupitt's GitHub mirror are acknowledged for the webRTIViewer component-tile layout supported by `--rti-layout webrti`. what-a-relief writes compatible folders but does not include webRTIViewer source code. webRTIViewer is available at https://github.com/jcupitt/webRTIViewer.

The implementation uses OpenCV for image I/O, image processing, DNN inference, and GUI windows, and vcpkg/CMake to make the Windows OpenCV dependency reproducible. Optional experimental neural-fusion builds bundle PS-FCN-derived ONNX assets under the upstream MIT license; see `THIRD_PARTY_NOTICES.md`, `assets\models\NOTICE.txt`, and `assets\models\LICENSE-PS-FCN.txt`.

BibTeX entries for the academic references are in `docs\references.bib`.
