# Scientific Quality Roadmap

This is the durable release checklist for the 0.2.1 quality update. A box is checked only when implementation, automated evidence, and user documentation are all present. Local validation and GitHub CI completion are tracked separately.

Checked release items below record the earlier 0.2.1 baseline, not certification of the current September review changes. The [R1-R9 fix ledger](review-fixes-2026-09.md) separates those dirty-tree implementation changes from evidence still required.

## September Review Verification

- [x] Add and run the [independent curved-scene inverse pipeline test](inverse-pipeline-test.md) with eight ring lights, textured diffuse/glossy materials, sensor noise, and actual classical initialization. Preserve the initial CPU/CUDA failure records.
- [x] Achieve accepted global geometry improvement on that fixed fixture without loosening accuracy thresholds: the alternating-fit worker passes the CUDA original and rotated-light captures. Record regional regressions rather than implying uniform improvement.
- [ ] Qualify unseen shapes/materials, seeds, source sizes, resolutions, and higher presets; address bump/depression and cast-shadow-region height regressions. A small global benchmark gain is not general scientific accuracy validation.
- [x] Correct pixel-center/edge discretization and DCT boundary/padding consistency. Add exact analytic plane/quadratic tests for even and padded sizes in both integrators, without changing photometric normals.
- [x] Feed full-precision classical photometric normals into a confidence-weighted inverse constraint, test its derivatives and handoff, and retain image validation gates. Record the heuristic weights and reduced-resolution scope.
- [ ] Require non-regression against the original photometric normal product, not only against normals derived from integrated height. The constrained inverse normals remain worse than the original classical map on the independent fixture; the test reports this distinction explicitly. Also qualify local height regressions rather than relying solely on global scores.

- [x] Record the completed eight-test CTest run, including all six baseline suites and the schema-2/method-v2 backend contract: 61.06 seconds using the finished installer's extracted runtime and bundled official LLVM 15.0.7, without a DLL override.
- [x] Record the seven passing worker-numerical tests for constrained fitting, observation validity/reduction, masking, units, and actionable unavailable-backend reporting.
- [x] Record the real-CPU renderer gate pass (50.23 seconds, eight workers, no forced Debug flag, primary and indirect silhouette sampling enabled): finite-ring/directional-disk AD versus FD and 24 x 24/eight-light/six-iteration tilt recovery/truth-start with internal-hole, spatial diffuse/glossy, normal-angle, and height-RMSE checks. Exact logs and metrics are in [validation.md](validation.md#verified-cpu-run).
- [ ] Complete the broader CPU matrix: explicit truth-start rejection/preservation assertions, independent geometry/BRDF cases, cast-shadow-driven recovery, irregular boundaries, coverage, and resolution/quality/source-size/seed sensitivity. The passing current gates do not certify this entire matrix.
- [ ] Record separate CUDA derivative/recovery evidence; a startup moving-shadow AD probe or the LLVM 15 CPU pass does not certify full CUDA recovery.
- [x] Verify official LLVM 15.0.7 resolves the tested native failures without the temporary Debug or `sppp=0` workarounds; remove both overrides. Record this as a compatibility pin, not a proven internal root-cause explanation.
- [x] Build the LLVM 15.0.7 backend installer after staged numerical and real CPU shadow/inverse tests pass; record installer build exit 0 and independently verify artifact size/SHA-256 in the [package record](review-fixes-2026-09.md#backend-package). The installed user application/backend remains unchanged.
- [x] Verify the finished installer's extracted runtime with all eight CTests, and complete a real C++-to-worker eight-image CPU smoke run that preserves baseline geometry when the withheld-light check rejects refinement.
- [ ] Quantify fixed-XY/eroded-mask limits with both primary and indirect silhouettes enabled; enabled derivatives do not establish free-outline recovery or arbitrary camera-occlusion accuracy.
- [ ] Verify the corrected R7 scope: preexisting mutual exclusion, shared flat/percentile datum, explicit near-field reference Z, measured positive inverse LED diameter, and declared directional angular-source approximation.
- [ ] Check packed 13-bit confidence versus full diagnostics, near-threshold decisions, coarse audit dimensions, many-component semantics, and measured peak memory before claiming R8/R9 performance gains.
- [ ] Profile inverse per-light input reduction, three-scene reuse, and one-light gradient accumulation on pinned LLVM 15; distinguish removed Python full stacks from the still-resident C++ input stack and geometry/caches.
- [ ] Validate source-size, camera, sample-count/iteration, and calibration sensitivity on new unseen cases and independently measured specimens, including black glossy regions with per-material errors and coverage.

## Scientific Core

- [x] Decode selected sRGB inputs with the standard piecewise transfer function and use one common stack scale so lighting ratios are preserved.
- [x] Cover 8-bit, 12-bit-in-16-bit, 16-bit, floating-point, and sRGB normal recovery with deterministic tests.
- [x] Bind full calibration CSV rows to image identity; reject missing, mixed, duplicate, ambiguous, zero, and ill-conditioned lighting data.
- [x] Support three or more images in classical calibrated and uncalibrated processing, retain the PS-FCN 3-through-25 limit, and make the no-redundancy three-image case explicit.
- [x] Use bounded deterministic minimal-subset consensus and Cauchy IRLS, and leave locally unsupported pixels unsolved rather than force-fitting them.
- [x] Separate model-supported shadows, half-vector-supported highlights, definite container clipping, and unlabeled model mismatch in aggregate and per-light diagnostics.
- [x] Validate sparse shadows, penumbrae, saturated highlights, multiple bright outliers, broad glossy failure behavior, and a 64-light stack quantitatively.
- [x] Add committed Mitsuba development scenes and a distinct no-sphere validation scene spanning primitives, relief, spatial texture, varied materials, directional lights, and a finite point-light ring, with independent normals and per-light shadow, highlight, clipping, and height truth.
- [x] Implement the nearby ring as a documented, center-normalized isotropic point-source model with millimeter geometry and inverse-square attenuation.
- [x] Keep neural predictions outside the evidence mask from entering fused outputs, and test the shipped ONNX workflow.

## Geometry And Scale

- [x] Make height flattening `none` preserve the integrated field and give plane, radial, and quadratic modes distinct documented bases.
- [x] Parse classic TIFF physical resolution conservatively and convert GeoTIFF model scale only when a supported linear unit is declared.
- [x] Quantitatively test full-field and irregular-mask height integration.
- [x] Add opt-in cast-shadow broad-height refinement with low-order corrections, normal/deformation priors, withheld-light validation, exact preservation on rejection, audit products, and analytic plus Mitsuba truth checks.
- [x] Add the first shadow-geometry upgrade: explicit near-field reference-surface Z, finite circular-emitter visibility, confidence-weighted shadow/lit evidence, separate receiver/occluder domains, observability and edge-support audits, and exact final-surface step selection with individual held-out-light checks.
- [x] Parse binary printable PLY output in tests and require a closed genus-zero fixture with a planar millimeter base.
- [x] Make enclosed printable-surface hole reconstruction opt-in, preserve the exterior boundary, and emit a synthesized-pixel audit mask.

## Exports And Reproducibility

- [x] Repair Deep Zoom level numbering and verify a stitched full-resolution coefficient plane.
- [x] Quantitatively reconstruct RGB and LRGB RTI source images from exported JPEG coefficients.
- [x] Write individual files through checked temporary files and replace complete RTI directories transactionally.
- [x] Remove known stale products at run start and retain an earlier valid RTI package when replacement fails.
- [x] Leave `run_manifest.json` marked `in_progress` until every requested artifact is present and nonempty, then record a complete run manifest.
- [x] Exercise the real executable in CTest for calibrated classical, calibrated neural, and uncalibrated workflows.

## Performance And Release

- [x] Keep a reproducible non-gating solver benchmark with a numerical checksum.
- [x] Parallelize independent calibrated pixel solves and record a before/after measurement.
- [x] Remove the redundant all-image padded preprocessing stack from PS-FCN and report dense input-tensor memory.
- [x] Finish the primary-source reference audit and align all user-facing scientific claims with actual implementation limits.
- [x] Pass the final clean Windows build, all CTest gates, diff hygiene, and repository-status review.
- [x] Commit the reviewed 0.2.1 release candidate locally.
- [ ] Push with user approval, then confirm that GitHub Actions passes the same release gates before publishing v0.2.1.
- [ ] Render a newly preregistered blind cast-shadow holdout and validate a physical relief phantom against independently measured height before making quantitative accuracy claims for shadow refinement.

## Shadow And Shape Research Sequence

This sequence preserves the literature-backed plan beyond the current low-order experiment. Checked items describe code and automated evidence already present; unchecked items must not be implied by the GUI or documentation.

- [x] Stage 1 forward-model controls and auditability: a shared directional-ray path for sphere-derived light directions; explicit near-field surface-to-light Z datum and finite LED diameter with sampled penumbra probability; confidence-weighted shadow and anti-shadow evidence; distinct receiver and occluder masks; observability maps; edge-support maps; and conservative per-light validation.
- [ ] Stage 2 calibrated image formation: camera intrinsics, perspective rays, lens distortion, arbitrary positioned-light or ring pose, per-source power, LED angular emission, and radiometric flat-field calibration. Keep the calibrated directional path available when sphere highlights provide directions but not positions, and validate every added term independently before fitting specimen shape.
- [ ] Stage 3 ShadowCuts-style constrained integration: formulate confidence-weighted shadow and anti-shadow inequalities while preserving measured normal gradients, solve with a documented sparse constrained optimizer, and compare against the present five-mode baseline.
- [ ] Stage 4 DeepShadow-style line-of-sight reasoning: implement a perspective cumulative horizon or scan operator with an explicit observability/conditioning test. Do not claim the DeepShadow network unless its published architecture and training protocol are actually reproduced.
- [ ] Stage 5 cross-light edge correspondence: establish geometrically consistent shadow-edge tracks across light positions following the epipolar constraints studied by Abrams, Miskell, and Pless; reject ambiguous or disconnected correspondences.
- [ ] Stage 6 optional differentiable rendering: compare Differentiable Shadow Mapping and Mitsuba Projective Sampling only after stages 2-5 have measurable holdout evidence. Keep this isolated from ordinary CPU-only runs and accept updates only through independent withheld lights and physical-height truth.

## Optional Inverse Rendering Research

- [x] Provide an isolated optional Mitsuba/Dr.Jit process contract and bootstrap path without adding Python or Mitsuba to the normal application runtime.
- [ ] Establish that the current coarse experimental inverse worker improves a preregistered blind holdout and measured physical-height phantom before recommending it for reconstruction.
- [ ] Start from the classical robust normals and fit only low-dimensional nuisance parameters first: ring pose, camera pose, per-light intensity, exposure, and coarse material groups.
- [ ] Use the fitted renderer to predict visibility, clipping, and specular confidence, then rerun the established normal solver with those observations masked or weighted. Measure improvement against held-out rendered and real captures.
- [ ] Consider coarse height or mesh refinement only after calibration/material fitting is identifiable. Avoid unconstrained simultaneous full-resolution optimization of shape, lighting, reflectance, and exposure from one eight-image stack.
- [ ] Compare the deterministic height-field ray audit with Mitsuba or another differentiable visibility model, especially around silhouettes and finite-area penumbrae, without adding a renderer to ordinary application runs.
- [ ] Treat inverse rendering as an experimental auditable stage with saved parameters and before/after metrics, never as a silent replacement for the classical result.

The detailed thresholds and current reference observations are maintained in [validation.md](validation.md).
