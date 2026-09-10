# Robust Solver Comparison

2026-09-10. Research comparison only. No application solver, existing fixture,
CTest gate, installer, or release was changed by this experiment.

## Decision

There is no universal winner, and the results do not justify replacing the
current robust solver with L1 or SBL as a speckle fix.

- Our current Cauchy solver beats the conservative Huber control on common-pixel
  angular accuracy in all three rendered scenes. Its robust stage is useful.
- The authors' normalized-input SBL implementation improves rendered accuracy
  further, but is substantially more sensitive to tiny changes in the fish photos.
- Huber preserves LS performance on benign diffuse noise and has the best
  dynamic fish stability when every method gets the same near-clipping taper.
- Sensor handling matters independently of the robust loss. Sharing that
  handling with LS removes much of its apparent disadvantage in this replay.
- Neither more images nor a different pointwise loss reliably fixes incorrect
  light brightness or broad, dominant specular reflection.

Recommended next work, not implemented here: share sensor validity/reliability
handling across solvers, investigate continuous treatment of the lower cutoff,
and evaluate a conservative Huber option alongside Cauchy. Retain SBL as an
experimental research candidate requiring noise-aware validation, not a default
replacement. Per-light radiometric calibration remains a separate priority.

## What Was Compared

This is an estimator comparison prompted by the Relight inspection, not an
end-to-end benchmark of the Relight application. The inspected Relight revision
is `21072e4c90cef8649f24501da69c080315727c62`. Its active native robust worker
uses threshold selection and Huber-like IRLS; its native SBL/RPCA functions are
empty. The separate Python reference algorithms are not that GUI solver.
[Relight source](https://github.com/cnr-isti-vclab/relight/blob/21072e4c90cef8649f24501da69c080315727c62/src/normals/normalsworker.cpp)

| Method | Implementation used |
| --- | --- |
| LS | Clipping-aware, headroom-weighted where requested; the application's weighted linear-solve helper. |
| Huber 10 / 80 | Independent implementation, delta 5/255, initial LS plus at most 10 or 80 reweightings, relative scaled-normal change below 1e-6. |
| Cauchy | Actual production `solvePhotometricStereo` API linked from the existing Release core: pseudo-Huber initialization, adaptive-scale Cauchy fitting, and its normal-validity gates. |
| L1 | Unmodified authors' `L1_residual_min`, default 1,000 iterations and absolute 1e-8 tolerance. |
| SBL | Unmodified authors' `sparse_bayesian_learning`, default 1,000 iterations and absolute 1e-8 tolerance. |

Huber is deliberately not an exact Relight reproduction: it uses our shared
observations and stopping rule, does not retain threshold violators to preserve
five samples, and counts reweightings rather than total fits. An 80-update cap
does not guarantee convergence. No Relight source was copied into the driver.

The reference Python repository is pinned to
`f03aa95b57e746a7d31df76b1c0fa0a83584a3c1`. It is imported externally, unmodified,
and is not included in the application or installer. Its author states GPL
distribution terms. This benchmark uses the numerical functions, not their
image loader or a reproduction of every experiment in the publications.
[Authors' implementation](https://github.com/yasumat/RobustPhotometricStereo)

The relevant publication is Ikehata, Wipf, Matsushita, and Aizawa,
*Robust Photometric Stereo using Sparse Regression*, CVPR 2012. Its sparse-error
formulation is relevant to the large-corruption tests; its assumptions about
lighting and linear response still matter. It does not make dense BRDF changes
or calibration errors disappear.
[Paper](https://satoshi-ikehata.github.io/SBLPS_cvpr2012.pdf)

Our Cauchy sequence remains an independent pointwise implementation, not the
full joint variational method of Queau et al. Its existing method mapping is in
[the estimator update](robust-estimator-update-2026-09-09.md).

## Protocol And Controls

The harness and settings were declared before the primary results. There were
28 case/policy combinations and 27,616 sample rows, with eight method/settings
variants per row. Repeated geometry, policy alternatives, and perturbations are
not independent specimens. No constants were tuned against these results.

- All methods receive identical float32 linear luminance and effective light
  rows. They exclude nonfinite values, values at or below 0.02, and known clipping.
- Main arm: binary validity. Secondary arm: shared upper-2% sensor-headroom
  weights. L1 uses linear row preweighting; SBL uses square-root preweighting.
  Weighted SBL is an input adaptation, not a claimed published weighted likelihood.
- L1/SBL are run both in native luminance units and with luminance multiplied by
  255. This adds no quantization. Their constants and absolute tolerances are
  scale-sensitive, so both arms are retained rather than choosing per-scene winners.
- Analytic controls use integrable height-derived normals, varying reflectance,
  8/16/40/64 lights, additive noise, approximate shot/read noise with sRGB
  encoding/decoding, color clipping, unequal light gains, and broad gloss.
  The analytic gloss model is a stress model, not a measured BRDF.
- Each of the three committed Mitsuba scenes contributes 2,000 deterministically
  sampled valid pixels, selected independently of solver output. The rendered
  scenes supply the physically based material/visibility tests. The near-field
  case uses the same planar-reference lighting approximation as production,
  not oracle surface height. These fixtures are reused validation data, not new
  blind holdouts.
- Rendered shadow/highlight truth is used only for evaluation. The known 12-bit
  clipping masks agree with the top ADC code packed as 65534 or 65535 in uint16
  on all sampled valid pixels; this was independently audited.
- Fish: 192 predetermined positions in the existing 640-by-640 crop, 40 lights,
  and eight independent minus-one/zero/plus-one RGB-code perturbations per point.
  sRGB decoding and the original common stack normalization are fixed. There
  are no measured normal truths for these photos.

For the fish photos, "clipping" means an encoded RGB channel equals 255 and
"headroom" refers to integer image codes before sRGB decoding, not camera RAW
measurements. This does not identify physical sensor saturation with certainty.
Inverse sRGB does not undo unknown in-camera tone mapping or sharpening. The
one-code perturbations are a reproducible sensitivity probe, not a calibrated
model of that camera's noise or JPEG processing.

The LS control is intentionally stronger than the application's ordinary LS
path. At the inspected source revision, that path applies the lower intensity
cutoff but does not consume the clipping/headroom arrays used by robust fitting.
Thus these tables do not directly describe selecting LS in the current GUI.
See [the ordinary LS loop](../src/photometric.cpp) and the standalone driver.
This distinction prevents improvements in preprocessing from being credited
entirely to the robust estimator.

Only pointwise normal estimation is assessed. There is no normal smoothing,
height integration, shadow-field refinement, inverse rendering, or image-output
encoding in the comparison. Albedo, height, mesh accuracy, and full-application
throughput are not evaluated here.

## Rendered Accuracy

Mean angular error in degrees, on the intersection of valid normals from all
eight variants. Lower is better. These are binary-validity results; the
headroom alternatives change the scene averages very little. L1/SBL below use
native luminance units and the default iteration budget.

| Scene | Common pixels | LS | Huber 80 | Cauchy | L1 | SBL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shadow/gloss, 8 directional lights | 1,965 | 2.881 | 1.902 | 1.641 | 1.658 | 1.417 |
| Textured primitives, 12 directional lights | 2,000 | 1.846 | 1.055 | 0.884 | 0.994 | 0.686 |
| Relief validation, 10 near-field lights | 1,915 | 3.683 | 3.537 | 3.447 | 3.460 | 3.275 |

Coverage is also important; common-pixel error alone discards difficult cases.

| Scene | LS coverage | Huber 80 | Cauchy | L1 | SBL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shadow/gloss | 98.25% | 99.15% | 99.15% | 99.15% | 99.15% |
| Textured primitives | 100% | 100% | 100% | 100% | 100% |
| Near-field relief | 95.75% | 95.75% | 96.05% | 95.95% | 96.15% |

The CSV also records own-support error and a deliberately severe 180-degree
penalty for unsupported normals. That penalty is a scoring convention, not an
estimate of their unknown errors. Each method retains its own acceptance gates;
Cauchy has additional conditioning/effective-support checks.

Material-level results prevent misleadingly reassuring scene averages. In the
first scene the black glossy object still has approximately 16.9-degree mean
error with Cauchy and 16.3 with SBL among their supported pixels, versus 21.2 for
Huber and 26.2 for LS. This remains a difficult reconstruction, not a solved one.
Conversely, SBL is notably better on the 55 textured-scene pixels shadowed in at
least half the images: 1.08 degrees versus Cauchy's 6.79, with full coverage.
The evaluation groups overlap; they are not shadow/highlight detection scores.

## Noise And Model Mismatch

All methods have full coverage in these analytic controls. Values are mean
ground-truth angular error, in degrees; SBL/L1 use native units.

| Control | LS | Huber 80 | Cauchy | L1 | SBL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diffuse noise, 8 lights | 0.350 | 0.350 | 0.353 | 0.426 | 0.476 |
| Diffuse noise, 16 lights | 0.240 | 0.240 | 0.240 | 0.294 | 0.329 |
| Diffuse noise, 40 lights | 0.151 | 0.151 | 0.153 | 0.189 | 0.216 |
| Diffuse noise, 64 lights | 0.119 | 0.119 | 0.119 | 0.145 | 0.168 |
| Noisy color, 8 lights, headroom | 0.772 | 0.739 | 0.721 | 0.818 | 0.828 |
| Noisy color, 64 lights, headroom | 0.488 | 0.464 | 0.481 | 0.450 | 0.491 |
| Unequal gains/gloss/fill, 8 irregular lights | 1.006 | 2.639 | 3.030 | 2.926 | 5.122 |
| Unequal gains/gloss/fill, 8 ring lights | 6.685 | 6.829 | 6.932 | 6.873 | 7.108 |
| Unequal gains/gloss/fill, 40 irregular lights | 0.762 | 1.006 | 0.897 | 1.464 | 1.708 |
| Unequal gains/gloss/fill, 40 ring lights | 6.684 | 6.739 | 6.768 | 6.860 | 7.083 |
| Broad dominant gloss, 8 ring lights | 18.134 | 18.048 | 17.998 | 18.123 | 16.095 |

Huber exactly matches LS in the benign noise controls. Cauchy is close, but is
not uniformly better. Sparse regression sacrifices noise efficiency here for
its ability to reject stronger corruptions elsewhere. No method adequately
recovers the broad-gloss control. These observations do not establish that
every possible noise-calibrated implementation of SBL has the same tradeoff.

## Fish Stability

99th-percentile angular change from a one-code perturbation, in degrees.
This measures sensitivity, not accuracy against an unknown real surface.

| Input-weight policy | Pairs | LS | Huber 80 | Cauchy | L1 | SBL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed upper clipping/headroom, lower cutoff active | 1,536 | 0.702 | 1.096 | 1.459 | 2.023 | 6.153 |
| Same arm, only pairs with all effective weights unchanged | 1,507 | 0.299 | 0.825 | 0.768 | 1.960 | 6.196 |
| Dynamic clipping, no upper taper | 1,536 | 9.635 | 7.938 | 7.823 | 8.896 | 10.565 |
| Dynamic clipping and shared upper taper | 1,536 | 2.737 | 1.979 | 2.755 | 3.945 | 9.977 |

Clipping status changes in 1,074 pairs, upper-headroom values in 1,169, and the
lower cutoff in 29. The strict unchanged-weight subset separates estimator
sensitivity from those changes. It shows Cauchy and Huber are similarly stable
there, but both remain noisier than LS. SBL's instability persists even without
input-weight changes. The subsets are descriptive analyses, not new fits.

With dynamic headroom, maximum changes are 5.355 degrees for LS, 4.050 for
Huber 80, 9.055 for Cauchy, 5.677 for L1, and 18.462 for SBL. SBL has 16 changes
above 10 degrees; the other four have none in this sample. The earlier, larger
fish replay found such outliers for Cauchy too. This smaller sample does not
establish that those residual problems have disappeared; its positions and
perturbations differ from the earlier experiment.

The lower cutoff contributes to the fixed-upper-policy tail, but it is not the
whole explanation: Cauchy's largest dynamic-headroom jump occurs without a
lower-cutoff crossing. A general fix needs to consider both changing reliability
weights and the estimator's response to them, rather than just blur normals.

## Units, Convergence, And Cost

SBL results depend materially on intensity units. Own-support mean errors for
native-unit versus 255-scaled SBL are 1.564/1.684, 0.686/0.901, and 3.467/3.655
on the three rendered scenes. Dynamic-headroom fish p99 is 9.977/9.318 degrees.
Neither unit convention resolves its fish sensitivity. L1's solutions are
usually close across scales, but its absolute stopping tolerance still matters.

The report separates rejected normals from supported fits whose convergence
was not established. On the 1,728 dynamic-headroom fish fits, those latter counts
are 1,708 for Huber 10, 31 for Huber 80, zero for Cauchy, 33 for native-unit L1,
125 for native-unit SBL, and 1,723 for 255-scaled SBL. The Huber numbers use our
strict scaled-normal tolerance, not Relight's weight-change tolerance.

A secondary check raises SBL's budget to 10,000 on 32 evenly spaced rows from
each of five predeclared cases. The checked fish normals change by at most
0.000160 degrees in native units and 0.000811 in scaled units. More iterations
do not materially move these sampled fish outputs. However, one native-unit
rendered normal changes by 20.825 degrees, improving that 32-row subset's mean
error from 2.923 to 2.272 degrees. Thus default-budget rendered results must not
be described as universally converged. This is a sampled budget check, not a
full 10,000-iteration repeat of every perturbation pair.

Timing is recorded for transparency, not as an application-speed ranking.
Cauchy is called through a one-pixel public API with diagnostics/allocations;
LS/Huber use narrower native paths. Python reference timing includes iteration
instrumentation and dense temporary matrices. Host load was not controlled.
The reference loops' dense diagonal matrices could be avoided in a future
implementation, but these timings do not predict an optimized port's speed.
Iteration counters also have method-specific meanings: LS has no IRLS updates,
while the reference functions count their own loops.

## Evidence And Reproduction

The application checkout is based on `bd66376d5561e98a16109a659677a3ac0da0a8bf`
with the pre-existing, uncommitted continuous-estimator changes. These are
development results, not a benchmark of the unchanged v0.2.4 installer.
The audit verifies all 28 required case/policy combinations, unit normals,
reported coverage, exact-Lambertian sanity, and 27,376 supported native LS fits
against NumPy. Maximum unit-vector disagreement is 6.74e-14 or less.

- [Reproduction tools and commands](../tools/compare_robust/README.md)
- [All summary tables](benchmarks/robust-comparison-2026-09-10/tables.md)
- [Machine-readable metrics](benchmarks/robust-comparison-2026-09-10/metrics.csv)
- [Coverage/convergence and numerical audit](benchmarks/robust-comparison-2026-09-10/audit.json)
- [Shadow/highlight strata](benchmarks/robust-comparison-2026-09-10/strata.json)
- [Strict fish-weight audit](benchmarks/robust-comparison-2026-09-10/fish-audit.json)
- [Source/executable/environment hashes](benchmarks/robust-comparison-2026-09-10/environment.json)
- [Extended-iteration results](benchmarks/robust-comparison-2026-09-10/extended-iterations.json)
- [Extended-iteration provenance](benchmarks/robust-comparison-2026-09-10/extended-environment.json)

The v0.2.5 release cleanup removed the temporary external checkouts, raw input
arrays, individual estimates, checkpoints, and comparison build folders. Summary
metrics, provenance hashes, and the research harness remain in the repository.
Original specimen directories were not modified; fish data are not distributed.
The public analytic/rendered subset can be regenerated using the linked commands
and a separately obtained reference checkout. Historical results used an interim
uncommitted estimator, so a run of the current release is a new comparison, not
a byte-for-byte replay of that estimator. CTest was not rerun during the original
comparison; all seven suites subsequently passed for the v0.2.5 release build.
