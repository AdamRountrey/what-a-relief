# Validation and Release Gates

This document defines the automated evidence required for a passing What A Relief build. The tests are deterministic synthetic and workflow regressions. They verify implementation behavior against known inputs; they do not establish accuracy on every microscope, camera, material, or specimen.

## Required Test Suites

| CTest name | Scope |
| --- | --- |
| `photometric-core` | Radiometry, calibration binding, lighting conditioning, calibrated normal recovery, corruption handling, near-field lighting, neural evidence masks, height integration, and height-flattening semantics |
| `io-exports` | TIFF/GeoTIFF scale parsing, checked writes, run manifests, RTI reconstruction, Deep Zoom geometry, transactional RTI replacement, and printable PLY topology |
| `end-to-end-workflow` | The real executable from input files through calibrated robust solve, height, open and printable meshes, RTI, and a complete manifest |
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
| Clean directional Lambertian, 3, 4, 8, and 25 lights | Mean normal error below `0.05 degrees`; complete solve coverage |
| Linear 8-bit, 12-bit values in a 16-bit container, full 16-bit, floating range, and 8-bit sRGB | Encoding-specific mean normal error from below `0.05` to `0.50 degrees` |
| One hard-shadow observation | Mean normal error below `0.05 degrees`; exactly one omitted shadow observation |
| One 50% penumbra observation | Robust error below `1 degree`, at least 80% below ordinary least squares, reported as shadow rather than specular |
| One saturated highlight with 4 lights | Robust mean normal error below `0.10 degrees` |
| Injected bright outliers with 5, 8, and 25 lights | Robust mean normal error below `0.10 degrees`; every injected outlier reported |
| Broad glossy BRDF lobe with 8 lights | Fixture must exceed `10 degrees` Lambertian error; specular diagnostic must mark more than 65% of pixels |
| Close point-light ring | Corrected mean normal error below `0.05 degrees`, at least 98% below the directional model, and recovered mean albedo within `1e-4` |
| Full rectangular DCT height | Normalized height RMSE below `0.08` |
| Irregular masked robust height | Normalized height RMSE below `0.07` and at least 30% below masked DCT |
| Height flattening | `none` is byte-for-byte unchanged; plane, radial, and quadratic basis fixtures leave less than `1e-4` maximum residual |
| RGB, LRGB, and sRGB-decoded RTI | Mean reconstructed linear error below `0.025`, `0.030`, and `0.030`, respectively |
| Deep Zoom | Every level and edge tile has the expected geometry; stitched full-resolution plane differs by fewer than 3 code values on average |
| Printable PLY | All indices valid, every edge has exactly two incident faces, Euler characteristic is 2 for the rectangular fixture, and the base is planar at the requested millimeter thickness |

The broad-gloss criterion is intentionally a detection test, not a recovery claim. In the current deterministic fixture, both ordinary and robust Lambertian solves have about `20.5 degrees` mean error while the half-vector diagnostic marks about 73% of the field. Broad, multi-image specular structure is not sparse corruption and cannot be repaired reliably by the current robust estimator.

## Current Reference Run

On the Windows development host with 20 logical processors, the current Release build produced these representative values:

```text
near_field_directional_mean_degrees=27.1052
near_field_corrected_mean_degrees=0
srgb_8_bit_mean_degrees=0.150658
penumbra_standard_mean_degrees=7.59561
penumbra_robust_mean_degrees=0.721992
narrow_highlight_standard_mean_degrees=10.7489
narrow_highlight_robust_mean_degrees=0.0197823
broad_gloss_standard_mean_degrees=20.5026
broad_gloss_robust_mean_degrees=20.5026
broad_gloss_specular_cue_rate=0.727114
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

On the development host, row-parallel execution reduced the fixed benchmark from `0.733 s` to `0.105 s` (about 7 times faster), with the identical checksum `5857367.868086`.

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

- Synthetic tests use idealized registered images and known light vectors. Real validation should include a traceable reference surface, repeated captures, and angular or height error statistics.
- Passing corruption tests does not make glossy reconstruction generally solved. Sparse isolated shadows and highlights are the regime supported by the robust estimator.
- A complete `run_manifest.json` records paths, file sizes, timestamps, parameters, diagnostics, lights, and outputs. It does not currently hash input contents and is not a cryptographic provenance record.
- Near-field tests validate the implemented isotropic point-source model on a `z = 0` plane. They do not validate finite LED area, beam anisotropy, camera perspective, lens distortion, source power differences, or height-dependent iteration.
- Watertight topology does not guarantee that a mesh has a useful physical Z scale, adequate wall thickness for a particular printer, no self-intersection on pathological masks, or successful slicing in every tool.
