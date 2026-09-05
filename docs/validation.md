# Validation and Release Gates

This document defines the automated evidence required for a passing what-a-relief build. The tests are deterministic synthetic and workflow regressions. They verify implementation behavior against known inputs; they do not establish accuracy on every microscope, camera, material, or specimen.

## Required Test Suites

| CTest name | Scope |
| --- | --- |
| `photometric-core` | Radiometry, calibration binding, lighting conditioning, calibrated normal recovery, corruption handling, near-field lighting, cast-shadow height refinement and rejection, neural evidence masks, height integration, and height-flattening semantics |
| `io-exports` | TIFF/GeoTIFF scale parsing, checked writes, run manifests, RTI reconstruction, Deep Zoom geometry, transactional RTI replacement, and printable PLY topology |
| `mitsuba-backend-contract` | Backend probing and process handoff, source-independent 16-bit linear observations with TIFF-named sources, result parsing, transactional output promotion, and temporary observation cleanup |
| `end-to-end-workflow` | The real executable from input files through calibrated robust solve, guarded cast-shadow refinement, height, open and printable meshes, RTI, audit outputs, and a complete manifest |
| `end-to-end-neural-workflow` | The real executable plus the bundled count-specific PS-FCN model and required classical, neural, fused, confidence, and neural-validity outputs |
| `end-to-end-uncalibrated-workflow` | The real executable through unknown-lighting normals, height, printable geometry, actual solve coverage, and a non-applicable calibrated-light condition |

Run all gates from a configured build:

```powershell
ctest --test-dir build\ninja-vcpkg --output-on-failure
```

The Windows direct-build script runs the same suite before producing the standalone executable. GitHub Actions uses that script, so a compiler-only success is not sufficient for an installer or tagged release.

## Windows Package Gates

The installer build self-extracts its completed artifact and requires a nonempty executable, README, third-party license bundle, and the 3-image and 25-image PS-FCN models. The portable-package script stages only documented runtime files, recursively includes `models/`, inspects the completed ZIP, and rejects entries from local input, object, or smoke-run folders. On the development host, the extracted 0.2.1 installer executable also completed the eight-image neural workflow with a valid 0.2.1 manifest and all 32 expected outputs.

## Quantitative Acceptance Criteria

Angular errors are mean per-pixel angles between recovered and known unit normals. Height errors are RMSE after removing only the unavoidable additive offset and normalizing by the known height range.

| Fixture | Required result |
| --- | --- |
| Clean directional Lambertian, 3, 4, 8, 25, and 64 lights | Mean normal error below `0.05 degrees`; complete solve coverage |
| Linear 8-bit, 12-bit values in a 16-bit container, full 16-bit, floating range, and 8-bit sRGB | Encoding-specific mean normal error from below `0.05` to `0.50 degrees` |
| One hard-shadow observation | Mean normal error below `0.05 degrees`; exactly one omitted shadow observation |
| One 50% penumbra observation | Robust error below `1 degree`, at least 80% below ordinary least squares, reported as shadow rather than specular |
| One saturated highlight with 4 lights | Robust mean normal error below `0.10 degrees` |
| Injected bright outliers with 5, 8, 25, and 64 lights | Robust mean normal error below `0.10 degrees`; every injected outlier reported as a highlight or model mismatch |
| Broad glossy BRDF lobe with 8 lights | Fixture must exceed `10 degrees` ordinary-Lambertian error; robust solve must improve it; specular diagnostic must mark more than 35% of pixels |
| Mitsuba sphere development scene | More than 98% solve coverage; robust mean error below `2 degrees` and below 50% of ordinary least squares overall and in corrupted regions |
| Mitsuba sphere glossy materials | Black narrow-GGX error below `20 degrees` and 50% of ordinary least squares; rough-gloss error below `6 degrees` and 65% of ordinary least squares |
| Mitsuba sphere observation classification | Full-scene shadow precision and recall above `0.90` with F1 above `0.92`; non-floor shadow precision above `0.70`, recall above `0.90`, and F1 above `0.80`; highlight-or-clipping precision above `0.90`, recall above `0.80`, and F1 above `0.85`; definite-clipping recall above `0.999` |
| Mitsuba textured-primitives development scene | More than 94% solve coverage; robust error below `6 degrees`, 85% of ordinary least squares overall, and 90% in corrupted regions; full-scene and non-floor shadow F1 above `0.78` and `0.65`; highlight-or-clipping F1 above `0.68`; clipping recall above `0.999` |
| Mitsuba no-sphere near-field validation scene | More than 90% solve coverage; robust error below `8 degrees` and lower than ordinary least squares overall and in corrupted regions; full-scene and non-floor shadow F1 above `0.75` and `0.65`; highlight-or-clipping F1 above `0.70`; clipping recall above `0.999` |
| Close point-light ring | Corrected mean normal error below `0.05 degrees`, at least 98% below the directional model, and recovered mean albedo within `1e-4` |
| Directional cast-shadow height refinement | Sphere-style calibrated parallel-light directions are sufficient for an accepted guarded correction that reduces balanced shadow mismatch and broad-height RMSE; changing pixel scale or any ring-only geometry value leaves the directional result unchanged within `1e-6` |
| Analytic cast-shadow height refinement | Correction is accepted; balanced cast-shadow mismatch falls by at least 4%; offset-normalized broad-height RMSE falls by at least 3%; high-frequency change RMS stays below `0.055`; multiplying every physical length by the same factor and translating integrated height by a constant leave the result unchanged |
| Finite-emitter and elevated-reference geometry | Seven-sample finite-emitter prediction has at least 15% lower probability RMSE than the point-source model against independent 19-sample truth; the correct reference-surface Z improves mismatch by at least `0.005` over a zero-Z model; observability, edge, and occluder-support audits are populated |
| Mitsuba cast-shadow height validation | At least six coherent shadow-bearing lights; regularized observed cast-shadow F1 above `0.72`; accepted correction does not worsen withheld-light mismatch or cast-shadow F1 and reduces injected broad-height RMSE by at least 3%; starting at renderer-truth height is rejected with an exactly unchanged field |
| Cast-shadow rejection | A stack with no coherent cast-shadow evidence reports an explicit rejection reason and preserves every height value byte-for-byte; its manifest records unavailable measurements as `null` |
| Full rectangular DCT height | Normalized height RMSE below `0.08` |
| Irregular masked robust height | Normalized height RMSE below `0.07` and at least 30% below masked DCT |
| Height flattening | `none` is byte-for-byte unchanged; plane, radial, and quadratic basis fixtures leave less than `1e-4` maximum residual |
| RGB, LRGB, and sRGB-decoded RTI | Mean reconstructed linear error below `0.025`, `0.030`, and `0.030`, respectively |
| Deep Zoom | Every level and edge tile has the expected geometry; stitched full-resolution plane differs by fewer than 3 code values on average |
| Printable PLY | All indices valid, every edge has exactly two incident faces, Euler characteristic is 2 for the rectangular fixture, and the base is planar at the requested millimeter thickness; smart filling closes an enclosed synthetic gap, preserves a boundary-connected notch, follows the known surface height, and emits an audit mask |

The analytic broad-gloss criterion is intentionally a detection and limited-improvement test, not a recovery claim. Broad, multi-image specular structure is not sparse corruption and cannot be repaired reliably by the current robust estimator. The independently rendered Mitsuba fixtures therefore report difficult material and object regions separately rather than allowing strong diffuse-region results to hide them.

## Mitsuba Reference Fixtures

`tests/fixtures/mitsuba` contains three committed offline scenes made with Mitsuba 3.8.0. `robust_v1` and `textured_primitives_v1` are development fixtures; the historically named `holdout_relief_v1` is now a no-sphere validation fixture with finite point lights. Together they cover spheres, smooth relief, a cylinder, tilted cube, disk, constant, checkerboard, and smoothly varying bitmap albedo, diffuse and multiple GGX rough-plastic materials, eight and twelve directional-light arrangements, and a ten-light near-field ring. Every scene includes cast or attached shadows, specular lobes, Poisson shot noise with a small dark signal, Gaussian read noise, 12-bit quantization, and clipping.

Reference positions, normals, albedo, and shape labels come from renderer AOVs. Shadow truth is based on attached-shadow geometry or visibility loss against a matched unoccluded Lambertian prediction; highlight truth is physical specular excess against the same rough-plastic scene with only specular reflection disabled. Highlight classification is scored after unioning physical highlights with definite clipping, because a clipped sample no longer preserves the peak's amplitude; clipping recall is also gated separately. These are operational, reproducible labels rather than universal semantic definitions.

The full-scene shadow score is intentionally supplemented by a non-floor score over renderer-labeled objects so easy background pixels cannot hide ambiguity on low-albedo glossy geometry. These renderer shape labels exist only as test truth; normal application runs neither require nor infer object identities. A shadow output is best read as a fitted-model shadow candidate used for robust weighting, not semantic segmentation. On a black glossy surface, weak diffuse response and a displaced fitted normal can make unilluminated, low-albedo, and geometrically shadowed observations difficult to distinguish from intensity alone.

`holdout_relief_v1` began as an untouched regression judge, but its independent height and cast-shadow truth were inspected while developing the cast-shadow correction. It is therefore accurately treated as validation data now, despite the historical directory name. A future blind generalization claim requires a newly rendered scene with acceptance criteria declared before its results are inspected.

The C++ tests discover each scene's image count and load the committed assets directly. Mitsuba and Python are not required for a normal build or for CI. To intentionally regenerate the corpus, use the pinned command in `tests/fixtures/mitsuba/README.md`, review all settings and hashes in every `manifest.json`, and rerun every release gate. A fixture regeneration is a scientific baseline change and should not be accepted as routine image churn.

## Current Reference Run

On the Windows development host with 20 logical processors, the current Release build produced these representative values:

```text
near_field_directional_mean_degrees=27.1052
near_field_corrected_mean_degrees=0
srgb_8_bit_mean_degrees=0.117034
penumbra_standard_mean_degrees=7.59561
penumbra_robust_mean_degrees=0
narrow_highlight_standard_mean_degrees=10.7489
narrow_highlight_robust_mean_degrees=0.0279765
broad_gloss_standard_mean_degrees=20.5026
broad_gloss_robust_mean_degrees=16.8246
broad_gloss_specular_cue_rate=0.402790
mitsuba_standard_mean_degrees=4.58947
mitsuba_robust_mean_degrees=1.48308
mitsuba_affected_standard_mean_degrees=5.55709
mitsuba_affected_robust_mean_degrees=1.73459
mitsuba_solved_fraction=0.994688
mitsuba_shadow_precision=0.945915
mitsuba_shadow_recall=0.972206
mitsuba_shadow_f1=0.958880
mitsuba_object_shadow_precision=0.727139
mitsuba_object_shadow_recall=0.933712
mitsuba_object_shadow_f1=0.817579
mitsuba_highlight_precision=0.971487
mitsuba_highlight_recall=0.860234
mitsuba_highlight_f1=0.912482
mitsuba_saturation_recall=1
mitsuba_black_gloss_standard_mean_degrees=45.8788
mitsuba_black_gloss_robust_mean_degrees=15.8981
mitsuba_rough_gloss_standard_mean_degrees=8.18015
mitsuba_rough_gloss_robust_mean_degrees=3.90020
mitsuba_textured_primitives_v1_standard_mean_degrees=1.82766
mitsuba_textured_primitives_v1_robust_mean_degrees=0.686827
mitsuba_textured_primitives_v1_shadow_f1=0.993512
mitsuba_textured_primitives_v1_object_shadow_f1=0.966068
mitsuba_textured_primitives_v1_highlight_f1=0.795092
mitsuba_holdout_relief_v1_standard_mean_degrees=3.96961
mitsuba_holdout_relief_v1_robust_mean_degrees=3.67735
mitsuba_holdout_relief_v1_solved_fraction=0.962574
mitsuba_holdout_relief_v1_shadow_f1=0.912880
mitsuba_holdout_relief_v1_object_shadow_f1=0.769673
mitsuba_holdout_relief_v1_highlight_f1=0.771053
mitsuba_holdout_relief_v1_saturation_recall=1
mitsuba_holdout_relief_v1_shape_3_robust_mean_degrees=21.1046
shadow_height_rmse_before=0.0892332
shadow_height_rmse_after=0.0435377
shadow_height_detail_change_rms=0.0180782
mitsuba_cast_observation_f1=0.952124
mitsuba_cast_prediction_f1_before=0.769365
mitsuba_cast_prediction_f1_after=0.776466
mitsuba_cast_prediction_f1_true_height=0.805987
mitsuba_shadow_height_rmse_before=0.0663373
mitsuba_shadow_height_rmse_after=0.0591526
mitsuba_truth_start_applied=0
mitsuba_truth_start_height_rmse=0
masked_dct_normalized_height_rmse=0.109596
masked_robust_normalized_height_rmse=0.0643968
rti_rgb_mean_absolute_linear_code_error=0.000881872
rti_lrgb_mean_absolute_linear_code_error=0.00265214
rti_srgb_mean_absolute_linear_error=0.00129318
deep_zoom_stitched_mean_absolute_code_error=0.0254862
```

Small floating-point variation across compilers and OpenCV builds is expected; the acceptance bounds above, rather than these exact observations, are the release gates.

## Performance Benchmark

`what-a-relief-benchmark` is a developer benchmark, not a timing gate because shared CI hosts are noisy. It renders a deterministic 768 x 1024, eight-light field with shadow and highlight corruption, runs the robust calibrated solver, and prints throughput plus a numerical checksum.

```powershell
cmake --build build\ninja-vcpkg --target what-a-relief-benchmark
$env:Path = "$PWD\build\ninja-vcpkg\vcpkg_installed\x64-windows\bin;$env:Path"
.\build\ninja-vcpkg\what-a-relief-benchmark.exe
```

On the development host, the current Release solver processed the fixed 768 x 1024, eight-light benchmark in `0.213 s`, or `3.69 megapixels/s`, with solved fraction `1.0` and checksum `5857367.868016`. The newer bounded consensus and physical-classification stages do more work than the earlier residual-only robust fit, so this replaces the older timing rather than being compared to it as a behavior-preserving optimization. The 64-hypothesis cap prevents minimal-subset enumeration from growing without bound as image count increases.

`what-a-relief-output-benchmark` writes the normal products, height products, full-resolution open PLY, and watertight printable PLY for a deterministic 768 x 1024 surface. It is also observational rather than a CI timing gate:

```powershell
cmake --build build\ninja-vcpkg --target what-a-relief-output-benchmark
.\build\ninja-vcpkg\what-a-relief-output-benchmark.exe 768 1024 1
```

The optional fourth argument fixes the OpenCV worker count, which is useful for comparing the row-parallel preparation stages. On the development host, buffered binary records, row-batched PFM output, shared sampled topology, compact printable-boundary bookkeeping, and parallel image preparation reduced this 100.754 MB output benchmark from `2.311 s` to `0.214 s` (about 10.8 times faster). The current implementation took `0.286 s` with one worker in the same cached 768 x 1024 test. At 1536 x 2048, it wrote 401.806 MB in `0.953 s` with one worker and `0.553 s` with the normal 20-worker pool. Exact timings depend strongly on storage and filesystem caching.

The standalone executable copied into the installer and portable package must also be a release build. A fixed five-image development fixture took `10.404 s` without MSVC optimization and `2.771 s` with `/O2` for the normal no-height workflow. The same fixture with an LRGB webRTIViewer export took `18.873 s` and `4.396 s`, respectively. These observations guard against accidentally shipping an unoptimized executable; they are not CI timing thresholds.

On that optimized five-image fixture, enabling fast DCT height increased total runtime only from `2.771 s` to `2.989 s`; robust masked height took `13.398 s`. The robust solver deliberately retains ordered SOR updates, so it is expected to remain the slower choice. Changing that update ordering would require separate numerical-quality validation rather than being treated as a behavior-preserving optimization.

PS-FCN still requires two dense `3 * image_count * height * width` float input tensors. Neural preprocessing now uses one image-sized scratch buffer rather than retaining a second padded copy of every input. This removes exactly `4 * image_count * padded_height * padded_width` bytes from that preprocessing working set, about 400 MiB for 25 images at 2048 x 2048. Network activations can require substantially more memory, so the existing lower-resolution retry remains necessary.

## Interpretation Limits

- Analytic and Mitsuba-rendered tests still use registered images, known light vectors, simplified cameras, and chosen material families. Real validation should include a traceable reference surface, repeated captures, and angular or height error statistics.
- Passing corruption tests does not make glossy reconstruction generally solved. Sparse isolated shadows and highlights are the regime supported by the robust estimator.
- Cast-shadow refinement is selective rather than universally corrective. Its acceptance gates establish improvement only under the implemented single-view height-field visibility model; they cannot prove lower physical height error on a real specimen. Overhangs, hidden sidewalls, translucent material, indirect light, and an incorrect near-field Z datum can all violate that model. The seven-sample finite-emitter approximation is tested against denser analytic quadrature, but has not yet been calibrated against a real LED package or diffuser.
- The withheld light directions are excluded from correction fitting, but their images still contribute to the original robust normal and observation estimates. They are an internal overfitting check, not an independent experimental holdout.
- A complete `run_manifest.json` records paths, file sizes, timestamps, parameters, diagnostics, lights, and outputs. It does not currently hash input contents and is not a cryptographic provenance record.
- Near-field normal tests validate the implemented isotropic point-source model on a `z = 0` plane. Experimental shadow refinement assigns the median flat-looking integrated level to a user-entered physical reference Z; that heuristic does not independently establish which specimen surface the entered value describes. Its effective finite-emitter geometry is analytically tested, but neither path validates real LED emitting area, beam anisotropy, camera perspective, lens distortion, or source power differences. Directional refinement needs only calibrated directions and does not use these near-field quantities.
- Watertight topology does not guarantee that a mesh has a useful physical Z scale, adequate wall thickness for a particular printer, no self-intersection on pathological masks, or successful slicing in every tool.
