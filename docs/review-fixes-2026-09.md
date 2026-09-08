# September 2026 Review-Fix Ledger

Date: 2026-09-08. Scope: the current, uncommitted changes responding to R1-R9
in the external `what-a-relief-review-2026-09-08/REVIEW.md`. That review examined
commit `cc5b7dd7567c2324554b72d4f50ef9d57a7273f6` (0.2.1). This ledger records
implementation changes and required evidence, not a new release certification.
The external review remains historical; its R7 workflow premise is corrected
below.

This ledger preserves the initial review/package state. Subsequent alternating
material fits, measured-normal constraints, corrected height integration, and
CPU/CUDA full-pipeline results are recorded in
[the independent pipeline history](inverse-pipeline-test.md) and
[current validation status](validation.md). Frozen-material budgets and
probe-only CUDA statements below describe the earlier package, not the current
development worker. No installer was rebuilt for these subsequent updates.

## Status and evidence

The recorded 2026-09-08 CPU run passed all eight CTest entries in **61.06 s**;
`mitsuba-worker-rendering` took **50.23 s** and worker numerics **1.26 s**.
The [preserved CTest log](../build/review-packaged-validation.log)
records the finished installer's extracted interpreter at
`build/packaged-mitsuba-review/python.exe` running the repository worker tests.
This run used bundled official LLVM 15.0.7 without a `DRJIT_LIBLLVM_PATH`
override, with eight CPU workers, both primary and indirect projective
silhouette sampling enabled, and no forced Debug kernels. Build-local logs
can be overwritten; exact metrics from this run are preserved below.

After the final method-label and repeat-run cleanup fixes, all eight suites
passed again in **50.62 s** (rendering **40.03 s**, numerics **1.01 s**).
That rerun includes the new v1/v2 inverse-directory cleanup regression and
is recorded in the [latest CTest log](../build/owned-mitsuba-gates/Testing/Temporary/LastTest.log).

The backend installer build, extracted-payload verification, CPU startup probe,
and final extracted-runtime CTest run succeeded; see the package record below.
The installed application/backend remains unchanged. CUDA passed the
moving-shadow startup probe only; full GPU reconstruction remains unqualified.
The GUI compiled, but visual workflow verification remains incomplete because
GUI automation approval timed out.

Source inspection, algebraic tests, fake-process contract tests, committed
rendered fixtures, and live derivative/recovery tests establish different
things. Startup `tiny_probe` is now an actual moving-shadow AD check, not a
forward-only probe. It requires a finite, nonzero response, but does not replace
finite-difference agreement or reconstruction testing. Existing CTest or
packaging success from before these changes does not certify the dirty tree.

For further qualification, record the exact revision/dirty diff, command,
runtime versions, CPU or GPU/driver, seeds, resolution, source size, sample
budget, geometry error, solved coverage, image losses, and acceptance decision.
Keep CPU and CUDA outcomes separate. A skipped or unavailable renderer test is
not a pass. See [validation.md](validation.md) for the required evidence tiers.

## Backend package

Staged-runtime numerical tests and real moving-shadow/inverse tests passed,
followed by backend installer script exit code `0`. The installer is a C#
self-extractor, not an Inno Setup package. Its extracted payload was verified
and passed the CPU startup probe and all eight CTest entries without a DLL
override.
The packaged worker uses official LLVM 15.0.7, at most eight CPU workers,
primary and indirect silhouette sampling enabled, and no forced Debug flag.
Verified artifact size and SHA-256:

```text
Artifact: dist/what-a-relief-0.2.1-mitsuba-backend-setup.exe
Bytes: 101876591
SHA256: 97834F9ECEC0E3181B6BDCFE9D240026D8D95378F207AF6E9145742E34DF1679
```

CUDA's moving-shadow probe returned `True`; this is an isolated AD startup
check, not full CUDA inverse-reconstruction qualification. Successful staging
and packaging do not mean the artifact has been installed: the main user
application and installed backend are unchanged.

A separate C++ executable-to-real-worker smoke completed with exit code `0`
using the extracted interpreter, eight workflow-fixture images, and preview
CPU directional refinement. Its [manifest](../build/review-cpp-inverse-smoke/run_manifest.json)
records `rejected_withheld_lights_worsened`; the baseline was retained.
This verifies integration and rejection handling, not reconstruction accuracy.
The job/result use schema `2` and method `mitsuba_heightfield_inverse_v2`.
The smoke exposed a remaining legacy run-manifest label and cleanup marker;
these are now updated, with a regression for recognizing both v1 and v2
generated directories while refusing unrelated data.

The older installed LLVM 18 CPU runtime was also checked: the current worker
returns an actionable reinstall error without attempting the failing JIT
workload. This is recorded in `build/review-old-backend-probe.json`.

## R1: Projective shadow derivatives

Delta point/directional inverse emitters are replaced by finite disks facing
the reference center, with area emission. Near-field inverse runs require a
measured effective LED emitting diameter greater than zero via the historical
`--shadow-led-diameter-mm` option. Disk shape, emission, and orientation remain
approximations to a real package or diffuser.

Directional inverse runs use an explicitly approximate distant disk, with
`--mitsuba-light-angle-deg` specifying full angular diameter: default `1` degree,
range `0.1` through `10` inclusive. This is not recovered from sphere calibration.
It changes penumbrae and must be included in sensitivity reporting.

The [Mitsuba maintainer's explanation](https://github.com/mitsuba-renderer/mitsuba3/issues/1443#issuecomment-2577492612)
identifies the delta-position boundary-sampling limitation and finite-area
alternative. The [emitter documentation](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#area-light-area)
defines the nonzero-area source model. This motivates finite size for the
chosen estimator, not a universal requirement for differentiating shadows.
[Zhang, Roussel, and Jakob (2023)](https://rgl.epfl.ch/publications/Zhang2023Projective)
provide the projective visibility-derivative method, not a reconstruction
guarantee for this application.

Required evidence: isolate a moving cast shadow on a fixed receiver, compare
AD with central finite differences using finite-source positive and delta-source
negative controls, and examine source-size/sample-count sensitivity. A nonzero
gradient alone is insufficient. Finite-ring and directional-disk AD/FD cases
passed in the recorded eight-test CPU run; broader negative controls and
source-size/sample-count sensitivity remain separate qualification work.

## R2: Real-renderer test coverage

The fake backend tests the process and output contract; worker numerical tests
exercise algebra, masking, sampling, and units. Installer/device `tiny_probe`
now differentiates an off-camera blocker's moving shadow on a fixed receiver
with a finite ring disk (16 x 16, 32 spp). It requires finite, nonzero AD; it
does not compare with finite differences or recover geometry.

The added `mitsuba-worker-rendering` unittest compares AD with central finite
differences for finite ring and distant directional disks. Its recovery case
uses a 24 x 24 near-field grid, eight lights, six Adam iterations, an internal
3 x 3 mask hole, and spatial diffuse/two-glossy coefficients. Optimization uses
16 spp and validation 128 spp; truth observations use 256 spp. The completed
LLVM 15.0.7 CPU run passed these tests, including accepted tilt recovery and
a rejected truth-start outcome. The derivative and recovery methods execute
in separate subprocesses, reflecting the application's isolated probe/job
process lifetimes rather than qualifying persistent shared renderer state.

The passing assertions require at least 15% normal-angle improvement over
missing tilt and lower offset-normalized height RMSE. Truth-start normal error
must be below 1 degree and height RMSE below 0.05 original height pixels. The
assertions do not explicitly require rejection or byte-for-byte inverse-height
preservation; rejection and zero height RMSE are observed in this run.
Observations come from the worker's own basis, so this is matched-model
evidence, not independent BRDF, integrated-normal, or physical-height
calibration. Ring recovery does not certify directional recovery. Thresholds
and scope are recorded in [validation.md](validation.md#verified-cpu-run).

Exact recorded metrics (not rounded or mixed with earlier runs):

```text
SHADOW_DERIVATIVE near_field_ring 0.46163320541381836 0.45899271965026855 0.9879823923110962
SHADOW_DERIVATIVE directional 0.1200481653213501 0.12100214511156082 0.9922997951507568
INVERSE_GEOMETRY tilted accepted_train_and_holdout_gate 4.313141345977783 0.5535522056025525 0.6940122433363092 height_rmse_pixels 0.35610827803611755
INVERSE_GEOMETRY truth rejected_insufficient_training_improvement 0.0013940532226115465 -74.12578487929359 -128.38151942189424 height_rmse_pixels 0.0
```

Derivative columns are mean absolute AD, mean absolute finite difference, and
cosine agreement. Geometry columns after the decision are mean normal error
in degrees, relative training improvement, relative withheld-light
improvement, and offset-normalized height RMSE in original height pixels.
Large negative relative losses near a truth-start minimum are not geometry
errors; the proposed candidate was rejected and the output preserved its
baseline. The planar recovery log reports no valid indirect silhouette
samples in several renders. It is not evidence of cast-shadow-driven height
recovery; the separate off-camera-blocker probe establishes the measured
moving-shadow derivatives.

Remaining evidence: explicit truth-start preservation assertions, independent
material and geometry cases, shadow-driven recovery, irregular boundaries,
coverage, and resolution/quality/source-size/seed sensitivity. Retain a
separate CUDA compatibility check. The eight-test pass certifies only the
implemented checks in the recorded configuration, not this entire research
matrix or every later dirty-tree/package change.

### Windows LLVM compatibility pin

The official LLVM 15.0.7 DLL resolved the tested native failures without
forcing `dr.JitFlag.Debug=True` or suppressing primary silhouettes. Both
temporary workarounds have been removed. Earlier LLVM 18/Dr.Jit 1.3.1 runs
showed native failures with COMDAT diagnostics; the successful compatibility
pin does not identify the internal JIT mechanism or prove a root-cause fix.

The built package now pins LLVM 15.0.7 with matching verified hashes, replacing
18.1.6. The current Windows worker guard requires LLVM major version 15,
rejecting known-unsupported 16-and-newer runtimes as well as other majors and
directing users to reinstall the compatible backend. This is a project
support policy, not proof that every possible build of each newer LLVM was
tested. Staged tests, the installer build, and the final extracted-runtime
CTest run succeeded as recorded above. This is not installed-runtime
verification; the earlier workspace DLL override was not used in the final run.

LLVM workers remain capped at eight, or half the reported logical CPU count
if smaller (minimum one). The verified run used eight. GPU reconstruction has
not been qualified. No change was made to the user's installed runtime.

## R3: Constrained appearance fit

The three-coefficient weighted ridge problem now uses exact box-constrained
active-set enumeration: 27 lower/free/upper combinations, reoptimizing free
coefficients after any binding, then selecting the feasible minimum. It is
exact for that small convex quadratic up to floating-point precision, not
unconstrained least squares with clipped coefficients. The chosen priors,
coefficient bounds, and diffuse/two-GGX basis remain engineering assumptions.

Required evidence: correlated columns, lower and upper bounds, and objective
comparison to known constrained optima. Better appearance optimization does
not itself prove better geometry, particularly while material is frozen.

## R4: Observation validity before reduction

The handoff carries one per-light validity mask derived from original decoded
finite, nonnegative observations and definite clipping masks. Validity is
established before area downsampling and conservatively propagated separately
from intensity; material fitting and geometry loss use the same weights.
Dark but valid shadow evidence is retained. Reduced intensity falling below
the old cutoff is no longer the test for whether a clipped source sample was
usable. Area integration replaces bilinear point sampling of observations.

Required evidence: a clipped highlight whose footprint averages below the old
threshold, nonfinite inputs, and consistency between appearance and height
objectives. Definite integer-container clipping still does not calibrate the
actual sensor saturation level or recover lost highlight amplitudes.

## R5: Slope units

The slope gate uses mask-aware differences of the expanded full-resolution
candidate correction. Height and spatial spacing are both in original-image
pixels. The RMS bound `0.45` therefore means height pixels per original-image
pixel, not per coarse render-grid sample.

Required evidence: identical planar slopes at different input sizes, render
sizes, and rectangular aspect ratios. Correct units remove the prior gate
scaling error; they do not make the complete inverse solution invariant to
quality settings or discretization.

## R6: Supported geometry and normals

Area reduction is mask-aware; coarse mesh triangles require supported
vertices. Exclusions and internal holes are not connected to artificial flat
background geometry. Normal export uses central differences only with valid
neighbors and one-sided differences at supported boundaries.

Required evidence: flat patches at nonzero height, irregular outlines, internal
holes, constant-height offsets, and boundary-adjacent shadows. Conservative
coarse coverage can remove thin features; masked-out real blockers remain
unknown. A supported open height field is not a complete specimen model.

Production inverse scenes again enable both primary and indirect silhouette
sampling; the temporary `sppp=0` override is removed. Fixed XY support and an
eroded fitting mask still limit support-edge evidence. Enabled primary terms
do not make this a free-outline solver or qualify arbitrary camera-occlusion
changes on real specimens. Indirect silhouette derivatives do not imply
indirect illumination is modeled.

## R7: Shared physical reference, corrected premise

**Correction to the review:** shadow and inverse refinement were already
mutually exclusive in CLI and GUI. There was no supported combined run whose
second stage lost the first stage's datum. The actual gap was that inverse-only
near-field geometry did not receive the explicit physical reference used by
shadow-only refinement.

Either mode now uses the shared full-resolution median sufficiently flat
height datum, with a tenth-height-percentile fallback. Near-field geometry
assigns that numerical reference the physical `--shadow-reference-z-mm` in
the same coordinates as ring height. The inverse job records
`reference_height_pixels` and `reference_surface_z_mm`; relative source Z is
`ring_height_mm - reference_surface_z_mm`, with reference Z below the lights.
The GUI enables the shared near-field fields for either alternative mode.

Required evidence: handoff fields, constant integrated-height offsets,
nonzero physical reference Z, scale consistency, and mutually exclusive mode
validation. The flat-surface heuristic does not independently identify the
entered physical surface. The classical ring normal solver still evaluates a
`z = 0` point-source approximation, not the elevated reconstructed geometry.

## R8: Component filtering

A per-label keep table followed by one image pass replaces a full-image
comparison for each retained component. After connected-component labeling,
filtering is `O(P + K)` for `P` pixels and `K` labels, rather than `O(P * K)`.

Required evidence: unchanged component-selection semantics and many-island
stress cases. This is a complexity correction, not a measured speedup claim.

## R9: Compact and streamed shadow evidence

Shadow-only runs without `--specular-diagnostics` retain one packed 16-bit
observation per light/pixel: 3 class bits and 13 confidence bits, decoded as
`code / 8191`. The retained observation allocation becomes `2 * N * P` bytes
instead of `7 * N * P`, for `N` lights and `P` pixels. Full-resolution evidence
is unpacked, regularized, and reduced one light at a time; processed packed
maps are released. Per-light audit masks/probabilities remain coarse unless
full diagnostics are requested. Aggregate and final height products remain
full resolution.

This is streaming of shadow evidence, not out-of-core image loading: the C++
input stack, initial packed maps, geometry, and temporary work buffers still consume
memory. The printed observation-storage estimate is not a total peak-memory
bound. Full diagnostics retain larger full-resolution stacks. The sixteen-light
fit selection does not cap the initial image or observation allocation.

Required evidence: packed/full confidence agreement within quantization
tolerance, decision/geometry regressions near thresholds, audit dimensions,
and measured peak memory on large stacks. No measured memory or timing result
is asserted for the current dirty tree.

### Additional inverse working-set changes

The Python `prepare_job` path now decodes observations and validity one light
at a time and immediately crops/area-reduces them. Prepared state retains
coarse observation/validity stacks, not the earlier full-resolution Python
stack. Full-resolution geometry, per-image decode buffers, and the C++ input
stack remain; this is not an out-of-core application.

`make_scenes` creates three distinct scenes, one for each material basis,
reusing them serially across lights with updated emitter parameters instead
of retaining `3 * N` distinct scene graphs. `optimize_ad` backpropagates each
training light's loss separately, accumulating the summed-objective gradient
and regularization before one Adam update per iteration. It does not take a
separate optimizer step for each light. Coarse multi-light arrays and renderer/
JIT caches still use memory. Peak-memory, runtime, numerical-equivalence, and
large-stack qualification remain to be measured on the pinned LLVM 15 runtime.

## Contract and budgets

The current contract is job schema `2`, method
`mitsuba_heightfield_inverse_v2`. It transfers original-source validity,
physical/numerical datum, effective LED diameter, and angular source diameter.
Update application and worker together; a schema-1 handoff is not equivalent.

| Preset | Maximum render side | Adam iterations | Optimization spp | Validation spp |
| --- | ---: | ---: | ---: | ---: |
| `preview` | 64 | 6 | 8 | 64 |
| `standard` | 128 | 18 | 16 | 128 |
| `research` | 256 | 50 | 32 | 256 |

`standard` is the default. `research` is retained as a CLI identifier; the GUI
calls it High sampling (slowest; experimental). Before/after evaluation uses
higher-sample renders, with a second seed checking provisionally accepted
withheld-view results. The presets are work budgets, not convergence evidence.

## Scientific limits that remain

- Camera: long-distance narrow-field perspective approximation, without
  measured microscope intrinsics, lens distortion, or optimized camera pose.
- Appearance: BRDF basis parameters and fitted screen-space coefficients are
  frozen during height optimization; only rendered responses change with
  geometry. No joint geometry/material/lighting calibration or alternating
  material refit is claimed. Roughness previews are not measured roughness.
- Geometry and lighting: fixed-XY single-valued relief, direct illumination,
  approximate source shapes, and incomplete support cannot represent hidden
  blockers, overhangs, separate depth layers, translucency, or interreflection.
- Visibility derivatives: primary and indirect silhouette sampling are enabled,
  but fixed XY support, an eroded fitting mask, and the limited CPU fixtures do
  not establish free-outline recovery or arbitrary camera-occlusion accuracy.
- Gloss: there is no proof of reliable reconstruction on real black glossy
  specimens. Report per-material/object errors and solved coverage; favorable
  whole-image averages can conceal failure in low-albedo glossy regions.
- Holdout: withheld lights already influenced baseline normals and appearance
  priors, and validation participates in acceptance. A new Monte Carlo seed
  is not a new specimen. The historical `holdout_relief_v1` scene is development
  validation data, not a blind holdout or physical accuracy calibration.
- Metrology: physical XY scale and explicit reference Z give a coordinate
  convention, not proof of correct height. Lower image loss, bounded updates,
  or passing synthetic tests do not establish improved physical surface error.
  Quantitative claims require independently measured phantoms, repeated real
  captures, preregistered unseen cases, and calibration/sampling sensitivity.
