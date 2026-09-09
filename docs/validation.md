# Validation and Release Gates

## Integration And Inverse Update

The current worker adds finer height controls, training-only material refits,
and training-plus-normal-prior checkpoint selection. The [independent pipeline record](inverse-pipeline-test.md#integration-and-measured-normal-update-2026-09-08)
contains full-application CUDA passes for the fixed original and rotated-light
captures, unchanged global error thresholds, exact metrics, and regional
regressions. This supersedes earlier frozen-material limitations below;
historical installer and CPU logs do not certify the updated package.
The original-capture CPU pipeline also passed. However, direct comparison
against the original photometric normal PNG remains worse after inverse
refinement; only height recovery and height-derived normals have improved.
The test reports this as `inverse_improves_classical_normal_product: false`.

The later integration audit corrected center-to-edge slope discretization,
DCT natural boundary terms, and transform padding. Both height integrators
now recover exact analytic planes/quadratics at even and odd/padded sizes
with normalized RMSE below `5.2e-8`, without altering input normals. All eight
CTest suites passed in 171.30 seconds after this update. The worker-numerical
suite contains 11 tests; live CPU tests additionally check the normal-prior
AD gradient against finite differences and tilted-plane recovery with a prior,
while retaining a no-prior truth-start regression. The C++ process contract
verifies XYZ ordering, bottom-up PFM rows, and float32 precision. Three direct
full-application CPU/CUDA cases separately passed the unchanged global
geometry-improvement criteria; these are not a single nine-test CTest run.

This document defines the automated evidence required for a passing what-a-relief build. The tests are deterministic synthetic and workflow regressions. They verify implementation behavior against known inputs; they do not establish accuracy on every microscope, camera, material, or specimen.

The [September 2026 review-fix ledger](review-fixes-2026-09.md) records earlier all-eight CPU CTest passes using the finished installer's extracted runtime and bundled official LLVM 15.0.7, without a DLL override. The final rerun after method-label/cleanup fixes passed in 50.62 seconds; the earlier 61.06-second run supplies the preserved numerical metrics below. The backend installer build and extracted-payload verification succeeded at that stage, when CUDA had only passed its moving-shadow startup probe. Later source and fixture results above supersede that narrower evidence, but do not rebuild those packages or qualify general GPU reconstruction. The installed application/backend remains unchanged. The GUI compiled but remains visually unverified because automation approval timed out.

## Required Test Suites

| CTest name | Scope |
| --- | --- |
| `photometric-core` | Radiometry, calibration binding, lighting conditioning, calibrated normal recovery, corruption handling, near-field lighting, cast-shadow height refinement and rejection, neural evidence masks, height integration, and height-flattening semantics |
| `io-exports` | TIFF/GeoTIFF scale parsing, checked writes, run manifests, RTI reconstruction, Deep Zoom geometry, transactional RTI replacement, and printable PLY topology |
| `mitsuba-backend-contract` | Fake-process probing and schema-2/method-v2 handoff, source-independent 16-bit linear observations plus per-light validity, physical/numerical datum and finite-source parameters, result parsing, transactional output promotion, and temporary observation cleanup; not an actual renderer test |
| `end-to-end-workflow` | The real executable from input files through calibrated robust solve, guarded cast-shadow refinement, height, open and printable meshes, RTI, audit outputs, and a complete manifest |
| `end-to-end-neural-workflow` | The real executable plus the bundled count-specific PS-FCN model and required classical, neural, fused, confidence, and neural-validity outputs |
| `end-to-end-uncalibrated-workflow` | The real executable through unknown-lighting normals, height, printable geometry, actual solve coverage, and a non-applicable calibrated-light condition |

Run all gates from a configured build:

```powershell
ctest --test-dir build\ninja-vcpkg --output-on-failure
```

The Windows direct-build script runs the same suite before producing the standalone executable. GitHub Actions uses that script, so a compiler-only success is not sufficient for an installer or tagged release.

The calibrated workflow also repeats a diagnostics-rich run with ordinary
robust output settings. It requires exactly nine core PNGs, byte-identical
core image hashes, removal of stale aggregate/per-light diagnostic files,
and a complete manifest that does not require disabled diagnostics. Optional
height/mesh/refinement products are checked before this final no-height rerun.

## Inverse Review Verification

An additional [independent curved-scene pipeline acceptance test](inverse-pipeline-test.md)
now starts from actual robust photometric normals and integrated height.
The current source passes the fixed original capture on CPU and CUDA and a
rotated-light capture on CUDA; initial failed attempts remain documented.
Rejection is not counted as success. It is an explicitly enabled ninth CTest
case, separate from the eight suites recorded below.

Keep these evidence tiers separate. Do not infer reconstruction success from a fake process, committed images, or successful import. Startup now includes an actual moving-shadow AD check, but it is narrower than finite-difference agreement or recovery testing.

| Tier | Required evidence | September review status |
| --- | --- | --- |
| Process contract | Schema `2`, method `mitsuba_heightfield_inverse_v2`, validity transfer/cleanup, shared datum and source-size fields, safe output promotion | Passed in the recorded eight-test run; still a fake-process contract, not real rendering |
| Worker numerical | Exact three-coefficient box-constrained optimum with correlated bases and upper bounds; clipping validity before area reduction; masked mesh/normals; original-pixel slope units and resampling; unavailable-backend reporting | Seven numerical unittests passed in the recorded run |
| Startup/device probe | Finite, nonzero AD of an off-camera blocker's shadow on a fixed receiver, finite ring disk, 16 x 16 and 32 spp | Implemented in `tiny_probe`; does not compare with finite differences or reconstruct geometry |
| Real CPU derivatives | Moving shadow on a fixed receiver; finite-difference agreement, finite-source positive and delta-source negative controls; sample/seed and source-size sensitivity | Finite ring and directional-disk AD/FD cases passed; broader controls/sensitivity remain separate work |
| Real inverse recovery | Truth-start no-regression, perturbed-height and mixed-material recovery, irregular masks/holes, resolution/quality comparisons; height/normal error and coverage alongside image loss | Matched-model tilt/internal-hole case passed, including height-RMSE assertions; not the full matrix |
| CUDA compatibility | Repeat applicable derivative/recovery checks on the CUDA runtime with recorded device/driver | Not qualified; CPU results and a startup GPU probe do not certify GPU reconstruction |
| Backend package | Numerical and real CPU tests on the staged LLVM 15.0.7 runtime, successful installer build, artifact fingerprint and extracted-payload verification | Passed; C# self-extractor build exited 0, size/SHA-256 verified, extracted CPU probe and all eight CTest entries passed without a DLL override |
| Application integration | C++ executable through the actual extracted Python/Mitsuba worker and result handling | Eight-image preview CPU directional smoke exited 0; candidate rejected and baseline retained, not an accuracy result |
| GUI workflow | Visual interaction with the compiled application | Not verified; GUI automation approval timed out |
| Physical accuracy | Measured relief phantom, repeated captures, new preregistered unseen specimens, and calibration uncertainty/sensitivity | Not established, especially for black glossy specimens |

Record the code revision and dirty diff, exact command, runtime/library versions, hardware, seeds, image size, source size, sample budget, and output metrics for each live result. Skipping because Mitsuba is unavailable is not a pass. Keep ordinary builds and the baseline C++ tier usable without the optional renderer. The installer/device probe now checks an isolated moving-shadow derivative; a finite nonzero response does not establish numerical derivative agreement, general derivative support, or inverse reconstruction.

The current CMake wiring opts in through `WHAT_A_RELIEF_MITSUBA_PYTHON`, an interpreter with NumPy, Mitsuba, and compatible LLVM. With a rebuilt compatible add-on installed, select that private interpreter from the configured Visual Studio/vcpkg environment and run the two optional entries. The recorded run instead used the finished installer's extracted runtime shown below; it did not update the installed add-on.

```powershell
cmake --preset ninja-vcpkg "-DWHAT_A_RELIEF_MITSUBA_PYTHON=$env:LOCALAPPDATA/Programs/what-a-relief-mitsuba/python.exe"
ctest --test-dir build\ninja-vcpkg -R '^mitsuba-worker-(numerics|rendering)$' --output-on-failure
```

These entries are named `mitsuba-worker-numerics` and `mitsuba-worker-rendering`. Selecting them does not demonstrate that the full matrix above is implemented or passing. An explicitly selected runtime must work; unavailability is a failure, not a silent successful skip.

### Rejected Candidate Review Gates

Finite rejected candidates are now retained for inspection in
`inverse/unvalidated_candidate/`, with an offline report and explicit
unvalidated provenance. The solver and acceptance thresholds are unchanged;
retaining or opening a candidate is not acceptance or reconstruction success.
The numerical suite covers finite retention for all six rejection labels,
unchanged baseline file hashes, nonfinite geometry, failed writes, PFM/PLY units,
mask holes, shared display ranges, and escaped report text. The process contract
covers candidate promotion from staging and rejected-to-accepted reruns. The
live renderer test checks accepted tilt recovery and rejected truth-start
retention. The following list also describes broader regression targets; not
every combination below has an automated gate.

- **Rejected versus accepted outputs:** Force a finite rejected correction and require `accepted:false`, unchanged guarded inverse height, zero guarded correction, matching baseline material outputs, and a distinct candidate height equal to baseline plus correction. Accepted runs must not create an extra candidate directory. Nonfinite supported candidate geometry must not be exported.
- **Artifact contract and provenance:** Verify every file in the [candidate file list](../tools/mitsuba_backend/README.md#rejected-candidate-outputs-and-review), conditional manifest/output checks, candidate export status, matching rejection metrics and selected iteration, `accepted:false` in `candidate.json`, and the PLY warning. Keep absent/skipped validation distinct from a passing metric.
- **Geometry and preview semantics:** Use finite synthetic planes/bumps with a datum offset, mask holes, non-square dimensions, resampling, and nonunit `height_scale`. Check height/correction pixel units, PLY XY/Z conventions, height-derived normals, shared-range review height previews, and shared-range root renders. `render_after.png` must still represent the candidate on rejection, not the guarded baseline.
- **Offline review and GUI consent:** Open the report without a server or network dependency; exercise baseline/candidate/both and product selectors. Candidate links must start hidden, appear after failed-validation acknowledgement, and resolve with the completed output folder relocated. Escape rejection text safely. Verify the GUI open-report prompt defaults to No and that opening, acknowledgement, and file access never change acceptance or default outputs.
- **Reruns and partial exports:** Exercise rejected-to-accepted and repeated rejected runs in the same output folder, with no stale candidate/report advertised. Inject export failures to ensure partial candidate files are not presented as a complete result. Retention is per run, not an archive.

These tests can use small arrays, mocked worker decisions, and fake-process
contracts for export behavior; renderer recovery remains a separate evidence
tier. Candidate normals and normal-prior agreement are not raw normal-map
ground truth, and shared-scale PNG comparisons make no numerical accuracy claim.

Local verification on 2026-09-08: 14 numerical/export tests passed, the live
Mitsuba CPU tilt/truth regression passed, and all six application CTest gates
passed. A synthetic export report was exercised offline in headless Edge at
1440x1100 and 390x844: every product image loaded, baseline/candidate selection
and acknowledgement worked, and there were no script errors or horizontal
overflow. This does not verify native Windows dialog interaction, relocation
of a real network-share run, or physical accuracy on the plant specimen.

### Verified CPU run

On 2026-09-08, all eight entries passed in `build/owned-mitsuba-gates`: the six baseline suites plus worker numerics and live rendering, in 61.06 seconds total. The [preserved log](../build/review-packaged-validation.log) records eight passes, the extracted interpreter path, the 50.23-second renderer gate, the 1.26-second numerical gate, and exact metrics. The passing assertion set includes height RMSE. The final method-label/cleanup rerun also passed all eight entries in 50.62 seconds, as recorded in [LastTest.log](../build/owned-mitsuba-gates/Testing/Temporary/LastTest.log).

The run used `build/packaged-mitsuba-review/python.exe`, extracted from the finished installer, with CPython 3.13.13, Mitsuba 3.8.0, Dr.Jit 1.3.1, NumPy 2.3.3, and bundled official LLVM 15.0.7. No `DRJIT_LIBLLVM_PATH` override was used. CTest invoked the repository's worker/tests using that interpreter. CPU execution used eight workers, without forced Debug kernels, and both primary and indirect silhouette sampling were enabled. In the existing build configured with `WHAT_A_RELIEF_MITSUBA_PYTHON` pointing to that extracted interpreter:

```powershell
Remove-Item Env:DRJIT_LIBLLVM_PATH -ErrorAction SilentlyContinue
ctest --test-dir build\owned-mitsuba-gates --output-on-failure
```

Extraction and testing did not replace installed files. The renderer test launches derivative and recovery methods in separate subprocesses, matching the application's separate probe/job process lifetimes. It does not certify a long-lived shared renderer session. The test scope and thresholds are:

| Check | Implemented scope and criterion |
| --- | --- |
| Shadow derivatives | Finite near-field ring and distant directional disks; 40 x 40 probe, 128 AD spp, 512 finite-difference spp; finite AD, mean absolute FD above `0.01`, AD/FD mean-magnitude ratio between `0.5` and `1.5`, derivative cosine agreement above `0.65` |
| Tilt recovery | 24 x 24 near-field grid, eight lights, six Adam iterations, 16 optimization spp and 128 validation spp; fixed physical reference Z, internal 3 x 3 hole, spatial diffuse and two independently varying glossy coefficients; accept the flat-start correction and reduce mean normal error by at least 15% over the known missing tilt in the eroded fitting region; current source also requires lower offset-normalized height RMSE than the initial field |
| Truth start | Same synthetic truth as initialization; recorded outcome is rejection. Assertions require mean normal error below `1 degree` and offset-normalized height RMSE below `0.05` height pixels, not explicit rejection or byte-for-byte preservation of inverse height |
| Output/support | Finite reconstructed values, zero output outside the mask, retained physical reference Z, and baseline products not modified |

Recorded results, with full-precision source lines preserved in the [durable ledger](review-fixes-2026-09.md#r2-real-renderer-test-coverage):

| Derivative case | Mean absolute AD | Mean absolute FD | Cosine agreement |
| --- | ---: | ---: | ---: |
| Finite ring | 0.46163320541381836 | 0.45899271965026855 | 0.9879823923110962 |
| Directional disk | 0.1200481653213501 | 0.12100214511156082 | 0.9922997951507568 |

| Recovery case | Decision | Mean normal error (degrees) | Offset-normalized height RMSE (original pixels) |
| --- | --- | ---: | ---: |
| Missing tilt | `accepted_train_and_holdout_gate` | 4.313141345977783 | 0.35610827803611755 |
| Truth start | `rejected_insufficient_training_improvement` | 0.0013940532226115465 | 0.0 |

The tilt candidate's relative training/withheld-light improvements were `0.5535522056025525` and `0.6940122433363092`. For the rejected truth-start candidate they were `-74.12578487929359` and `-128.38151942189424`; these are baseline-loss-normalized image metrics, not geometric errors or uncertainties. The reported output geometry for rejection remains the baseline. Height RMSE here removes only the additive offset and is in original height pixels, unlike the range-normalized classical fixture metrics later in this document.

For comparison with the removed historical shadow-height examples, the same log records the following classical cast-shadow refinement results. These are range-normalized height RMSE for injected broad drift, not the inverse-renderer recovery above:

| Classical shadow fixture | Before | After |
| --- | ---: | ---: |
| Analytic broad drift | 0.0892332 | 0.0668985 |
| Committed Mitsuba validation fixture | 0.0663373 | 0.040475 |

Recovery observations are rendered at 256 spp from the worker's own diffuse/two-GGX basis. This is matched-model synthetic evidence, not independently modeled BRDF recovery, integrated-normal end-to-end recovery, physical height calibration, or validation of black glossy specimens. The planar recovery log includes warnings that no valid indirect silhouette samples were found; its improvement is not evidence of cast-shadow-driven height recovery. The separate off-camera-blocker AD/FD probe establishes the measured moving-shadow derivatives. The ring recovery fixture is not a directional reconstruction test. Resolution/preset, source-size, camera, seed, and calibration sensitivity remain separate work.

### Windows LLVM compatibility

Official LLVM 15.0.7 is the empirically verified Windows compatibility pin for Dr.Jit 1.3.1. The final worker retains primary and indirect silhouette sampling, does not force Debug kernels, and caps LLVM workers at eight (or half the logical CPU count if smaller, at least one). Compatibility testing does not establish an internal root-cause explanation for earlier failures.

The built package pins LLVM 15.0.7 with updated hashes. The Windows guard requires LLVM major version 15, rejecting known-unsupported 16-and-newer runtimes and other majors with a reinstall instruction; this does not mean every newer LLVM build was exhaustively tested. Staged numerical/real CPU tests, the installer build, and all eight CTest entries using the extracted runtime passed. The installed application/runtime remains unchanged and GPU reconstruction is not qualified.

Current preview/standard/research settings are 6/18/50 Adam iterations, 8/16/32 optimization spp, 64/128/256 validation spp, and 64/128/256 maximum render sides. `research` is a CLI identifier, not an accuracy certification. Before/after checks use higher-sample renders, and provisionally accepted candidates receive a second-seed withheld-view check. Check convergence and angular/physical source-size sensitivity empirically; these budgets are not prescribed by the projective-sampling paper.

Production inverse scenes retain both primary and indirect silhouette sampling. The data mask is still eroded around fixed XY support. Enabled primary derivatives do not establish free-outline reconstruction or physical accuracy for arbitrary camera occlusion. Indirect silhouette sampling does not mean indirect illumination is modeled.

The baseline-rendered cast-shadow regression below starts from renderer-truth height with injected broad polynomial drift. It tests that perturbation family, not the complete pipeline from biased photometric normals through integration. Its truth-start rejection is not a truth-start test of the Mitsuba inverse worker.

## Windows Package Gates

The installer build self-extracts its completed artifact and requires a nonempty executable, README, third-party license bundle, and the 3-image and 25-image PS-FCN models. The portable-package script stages only documented runtime files, recursively includes `models/`, inspects the completed ZIP, and rejects entries from local input, object, or smoke-run folders. On the development host, the extracted 0.2.1 installer executable also completed the eight-image neural workflow with a valid 0.2.1 manifest and all 32 expected outputs.

The separate Mitsuba backend package built successfully: staged numerical and real CPU shadow/inverse tests passed, then the C# self-extractor build script exited `0`. The artifact `dist/what-a-relief-0.2.1-mitsuba-backend-setup.exe` was verified at `101876591` bytes with the SHA-256 preserved in the [package record](review-fixes-2026-09.md#backend-package). The extracted payload was verified, passed its CPU probe, and supplied the runtime for the final all-eight CTest pass without a DLL override. CUDA's moving-shadow probe returned `True`, not full reconstruction qualification. No installation was performed.

A separate eight-image workflow-fixture smoke exercised the C++ executable through the actual extracted Python/Mitsuba worker with preview CPU directional settings and exited `0`. The [smoke manifest](../build/review-cpp-inverse-smoke/run_manifest.json) records a complete run with `rejected_withheld_lights_worsened` and the baseline retained. This is integration/rejection-handling evidence, not an accepted reconstruction or accuracy claim. The compiled GUI remains visually unverified because GUI automation approval timed out.

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
| Analytic pixel-center slopes | Plane and quadratic surfaces at 32 x 24 and 23 x 17: normalized height RMSE below `1e-4` for both fast and robust integration with slope compression disabled; input normals byte-for-byte unchanged |
| Irregular masked robust height | Normalized height RMSE below `0.07` and at least 30% below masked DCT |
| Height flattening | `none` is byte-for-byte unchanged; plane, radial, and quadratic basis fixtures leave less than `1e-4` maximum residual |
| RGB, LRGB, and sRGB-decoded RTI | Mean reconstructed linear error below `0.025`, `0.030`, and `0.030`, respectively |
| Deep Zoom | Every level and edge tile has the expected geometry; stitched full-resolution plane differs by fewer than 3 code values on average |
| Printable PLY | All indices valid, every edge has two oppositely oriented incident faces, no unused vertices, and one connected component; isolated islands, point contacts, pinched boundaries, downsampling-severed bridges, and 24 deterministic irregular masks are exercised with filling on/off. Discarded islands do not affect base elevation; scientific and open-mesh outputs remain unchanged. Euler characteristic is 2 for the rectangular fixture, and the base is planar at the requested millimeter thickness; smart filling closes an enclosed synthetic gap, preserves a boundary-connected notch, follows the known surface height, and emits an audit mask |

The analytic broad-gloss criterion is intentionally a detection and limited-improvement test, not a recovery claim. Broad, multi-image specular structure is not sparse corruption and cannot be repaired reliably by the current robust estimator. The independently rendered Mitsuba fixtures therefore report difficult material and object regions separately rather than allowing strong diffuse-region results to hide them.

## Mitsuba Reference Fixtures

`tests/fixtures/mitsuba` contains three committed offline scenes made with Mitsuba 3.8.0. `robust_v1` and `textured_primitives_v1` are development fixtures; the historically named `holdout_relief_v1` is now a no-sphere validation fixture with point lights at finite distances, not finite-area emitters. Together they cover spheres, smooth relief, a cylinder, tilted cube, disk, constant, checkerboard, and smoothly varying bitmap albedo, diffuse and multiple GGX rough-plastic materials, eight and twelve directional-light arrangements, and a ten-light near-field ring. Every scene includes cast or attached shadows, specular lobes, Poisson shot noise with a small dark signal, Gaussian read noise, 12-bit quantization, and clipping. These existing assets have not been regenerated to match the finite-disk inverse source model.

Reference positions, normals, albedo, and shape labels come from renderer AOVs. Shadow truth is based on attached-shadow geometry or visibility loss against a matched unoccluded Lambertian prediction; highlight truth is physical specular excess against the same rough-plastic scene with only specular reflection disabled. Highlight classification is scored after unioning physical highlights with definite clipping, because a clipped sample no longer preserves the peak's amplitude; clipping recall is also gated separately. These are operational, reproducible labels rather than universal semantic definitions.

The full-scene shadow score is intentionally supplemented by a non-floor score over renderer-labeled objects so easy background pixels cannot hide ambiguity on low-albedo glossy geometry. These renderer shape labels exist only as test truth; normal application runs neither require nor infer object identities. A shadow output is best read as a fitted-model shadow candidate used for robust weighting, not semantic segmentation. On a black glossy surface, weak diffuse response and a displaced fitted normal can make unilluminated, low-albedo, and geometrically shadowed observations difficult to distinguish from intensity alone.

`holdout_relief_v1` began as an untouched regression judge, but its independent height and cast-shadow truth were inspected while developing the cast-shadow correction. It is therefore accurately treated as validation data now, despite the historical directory name. A future blind generalization claim requires a newly rendered scene with acceptance criteria declared before its results are inspected.

The C++ tests discover each scene's image count and load the committed assets directly. Mitsuba and Python are not required for a normal build or baseline CI; optional worker/live-renderer tests have separate dependencies. To intentionally regenerate the corpus, use the pinned command in `tests/fixtures/mitsuba/README.md`, review all settings and hashes in every `manifest.json`, and rerun every release gate. A fixture regeneration is a scientific baseline change and should not be accepted as routine image churn.

## Historical Reference Run

The pre-review documentation recorded these representative Release values on the Windows development host with 20 logical processors. They lack complete run provenance and are retained as historical observations, not current dirty-tree verification. Shadow-height/mismatch values previously in this block differed from the September review's run and were removed. The [verified CPU run](#verified-cpu-run) above now gives separately attributed shadow-height values from the completed September log; do not mix runs.

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

The pre-review development-host Release measurement processed the fixed 768 x 1024, eight-light benchmark in `0.213 s`, or `3.69 megapixels/s`, with solved fraction `1.0` and checksum `5857367.868016`. The bounded consensus and physical-classification stages do more work than the earlier residual-only robust fit, so this is not a behavior-preserving optimization comparison. The 64-hypothesis cap prevents minimal-subset enumeration from growing without bound as image count increases. This benchmark does not measure the September shadow-memory or inverse-renderer changes.

`what-a-relief-output-benchmark` writes the normal products, height products, full-resolution open PLY, and watertight printable PLY for a deterministic 768 x 1024 surface. It is also observational rather than a CI timing gate:

```powershell
cmake --build build\ninja-vcpkg --target what-a-relief-output-benchmark
.\build\ninja-vcpkg\what-a-relief-output-benchmark.exe 768 1024 1
```

The optional fourth argument fixes the OpenCV worker count, which is useful for comparing the row-parallel preparation stages. On the development host, buffered binary records, row-batched PFM output, shared sampled topology, compact printable-boundary bookkeeping, and parallel image preparation reduced this 100.754 MB output benchmark from `2.311 s` to `0.214 s` (about 10.8 times faster). The current implementation took `0.286 s` with one worker in the same cached 768 x 1024 test. At 1536 x 2048, it wrote 401.806 MB in `0.953 s` with one worker and `0.553 s` with the normal 20-worker pool. Exact timings depend strongly on storage and filesystem caching.

The standalone executable copied into the installer and portable package must also be a release build. A fixed five-image development fixture took `10.404 s` without MSVC optimization and `2.771 s` with `/O2` for the normal no-height workflow. The same fixture with an LRGB webRTIViewer export took `18.873 s` and `4.396 s`, respectively. These observations guard against accidentally shipping an unoptimized executable; they are not CI timing thresholds.

On that optimized five-image fixture, enabling fast DCT height increased total runtime only from `2.771 s` to `2.989 s`; robust masked height took `13.398 s`. The robust solver deliberately retains ordered SOR updates, so it is expected to remain the slower choice. Changing that update ordering would require separate numerical-quality validation rather than being treated as a behavior-preserving optimization.

PS-FCN still requires two dense `3 * image_count * height * width` float input tensors. Neural preprocessing now uses one image-sized scratch buffer rather than retaining a second padded copy of every input. This removes exactly `4 * image_count * padded_height * padded_width` bytes from that preprocessing working set, about 400 MiB for 25 images at 2048 x 2048. Network activations can require substantially more memory, so the existing lower-resolution retry remains necessary.

For September R8/R9, component filtering now maps a keep table in one image pass, and shadow-only runs without full diagnostics use packed 3-bit-class/13-bit-confidence observations (`2 * image_count * pixels` bytes instead of `7 * image_count * pixels`). Regularization streams one light into coarse evidence at a time. This is an allocation/complexity account, not a measured peak-memory or speedup result. The input image stack remains resident; full `--specular-diagnostics` restores larger per-light products. Compare confidence quantization, threshold decisions, geometry, and audit dimensions before reporting equivalence or performance gains.

The Python inverse worker separately streams observation/validity decoding and area reduction, retaining only coarse stacks in prepared state. It reuses three distinct material-basis scenes across lights and accumulates one light's AD contribution at a time before a single Adam update per iteration. Full-resolution geometry, per-image decode buffers, coarse observation/basis arrays, renderer/JIT caches, and the C++ input stack remain. These reductions do not establish total peak memory, runtime improvement, or bitwise equivalence; profile the pinned LLVM 15 configuration used in the verified test.

## Interpretation Limits

- Analytic and Mitsuba-rendered tests still use registered images, known light vectors, simplified cameras, and chosen material families. Real validation should include a traceable reference surface, repeated captures, and angular or height error statistics.
- Passing corruption tests does not make glossy reconstruction generally solved. Sparse isolated shadows and highlights are the regime supported by the robust estimator.
- Cast-shadow refinement is selective rather than universally corrective. Its acceptance gates establish improvement only under the implemented single-view height-field visibility model; they cannot prove lower physical height error on a real specimen. Overhangs, hidden sidewalls, translucent material, indirect light, and an incorrect near-field Z datum can all violate that model. The seven-sample finite-emitter approximation is tested against denser analytic quadrature, but has not yet been calibrated against a real LED package or diffuser.
- The withheld light directions are excluded from correction fitting, but their images still contribute to the original robust normal and observation estimates, and validation participates in candidate acceptance. They are an internal overfitting check, not an independent experimental holdout or accuracy calibration. An inverse validation render with a new Monte Carlo seed is not a new capture or specimen.
- A complete `run_manifest.json` records paths, file sizes, timestamps, parameters, diagnostics, lights, and outputs. It does not currently hash input contents and is not a cryptographic provenance record.
- Near-field normal tests validate the implemented isotropic point-source model on a `z = 0` plane. Shadow and inverse refinement, already mutually exclusive, now share the flat-surface median/tenth-percentile numerical reference assigned to explicit physical Z. That heuristic does not independently identify the entered specimen surface. Shadow-only finite-emitter quadrature tests are not calibration of real emitting area, beam anisotropy, or a diffuser. Directional shadow-only refinement uses calibrated parallel directions; directional inverse refinement instead declares a finite angular disk approximation, default full diameter `1` degree, adjustable with `--mitsuba-light-angle-deg` from `0.1` to `10`.
- Inverse rendering retains an approximate camera and fixed diffuse/two-GGX BRDF basis shapes. Screen-space weights now alternate with height updates using only fitting lights, but there is no camera, lighting, or indirect-illumination calibration. The [independent curved-scene test](inverse-pipeline-test.md) measures reconstruction gains and regional regressions; neither these cases nor corrected algebra/masking/units establish convergence of any preset or recovery of real black glossy specimens.
- Primary and indirect silhouette derivatives are enabled in production inverse scenes. Fixed XY support, an eroded fitting mask, and limited CPU fixtures do not establish free-outline recovery or accurate reconstruction under arbitrary camera occlusion.
- Watertight topology does not guarantee that a mesh has a useful physical Z scale, adequate wall thickness for a particular printer, no self-intersection on pathological masks, or successful slicing in every tool.
