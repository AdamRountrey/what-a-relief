# Continuous Robust Estimator Update

This implementation follows the [September 9 review](robust-estimator-review-2026-09-09.md).
It addresses noise amplification in the estimator itself rather than smoothing
the resulting normal map. It is a development update, not a new release or a
claim that general non-Lambertian reconstruction is solved.

## Publication To Implementation

| Reference | Application and status |
| --- | --- |
| [Holland and Welsch, 1977](https://doi.org/10.1080/03610927708827533) | Implemented independently: iteratively reweighted least squares, with weights derived from the actual robust objective. |
| [Queau et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Queau_A_Non-Convex_Variational_CVPR_2017_paper.html) | Adapted idea: continuous redescending photometric loss. Our pointwise pseudo-Huber/Cauchy sequence does not implement their joint depth, lighting, self-shadow, and integrability framework. |
| [Debevec and Malik, SIGGRAPH 1997](https://www.pauldebevec.com/Research/HDR/) | Adapted principle: reduce reliance on sensor-code extremes. Our upper-2% smoothstep is an engineering choice, not their HDR merge, response recovery, or weighting formula. |
| [Ikehata et al., CVPR 2012](https://satoshi-ikehata.github.io/SBLPS_cvpr2012.pdf) | Research prototype only: independently implemented sparse Bayesian regression with an adapted per-observation noise model. Not used in the released normal pipeline. |
| [Verbiest and Van Gool, CVPR 2008](https://doi.org/10.1109/CVPR.2008.4587712) | Research reference for a later spatial inlier/outlier field. Their spatial probabilistic inference is not implemented here. |

No reference implementation source code was copied. The published methods do
not specify our noise constants, stopping tolerances, headroom taper, diagnostic
thresholds, or application acceptance gates. Equations and limits are in
[algorithm.md](algorithm.md); complete bibliographic entries are in
[references.bib](references.bib).

## Implemented

- Removed discrete hypothesis competition and classification-driven refitting.
- Kept one joint scaled-normal variable, without an independent albedo reset.
- Added a convex pseudo-Huber initialization followed by Cauchy IRLS, with
  objective-descent checks, early stopping, and a fixed uncapped scale within
  the final phase. Maximum work is 80 updates per phase, with exact-fit bypass.
- Separated robust fitting weights from shadow/highlight audit labels.
- Propagated a heuristic sensor-noise model through frozen fit weights to
  moderate shadow classification. This is not a calibrated posterior or a
  complete uncertainty estimate for a nonlinear robust fit.
- Tapered raw integer RGB headroom before luminance fitting; retained exact
  clipping masks and ignored alpha. Floating-point white levels are not guessed.
- Added estimator identity, mean iterations, and nonconverged fraction to the
  run manifest. The log warns when final convergence was not established.
- Added portable perturbation, light-permutation, clipping-boundary, and I/O
  regression checks. No spatial normal smoothing or new per-pixel tuning UI.

## Numerical Evidence

The new synthetic perturbation suite covers 8/16/40/64 lights, ring and irregular
directions, varied normals/albedo, gloss, fill light, unknown gain differences,
and sRGB quantization. It deliberately separates noise stability from accuracy
under incorrect calibration. All pixels must remain supported.

| One-code perturbation case | Reviewed hypothesis solver p99 change | Continuous solver p99 change |
| --- | --- | --- |
| 8 lights, ring | 7.33 degrees | 0.97 degrees |
| 8 lights, irregular | 11.50 degrees | 1.21 degrees |
| 16 lights, ring | 2.39 degrees | 0.62 degrees |
| 16 lights, irregular | 4.37 degrees | 0.88 degrees |
| 40 lights, ring | 1.02 degrees | 0.44 degrees |
| 40 lights, irregular | 1.61 degrees | 0.50 degrees |
| 64 lights, ring | 0.78 degrees | 0.34 degrees |
| 64 lights, irregular | 1.18 degrees | 0.40 degrees |

The new RGB clipping-boundary fixture measures 0.929-degree p99 change and
0.410-degree mean ground-truth normal error. Correctly decoded noisy color
relief has lower mean error than both ordinary and clipping-aware LS for every
tested light count; [validation.md](validation.md) gives the actual values.

These tests do not establish universal superiority. On the gain-mismatched
eight-light irregular stress case, the revised mean error is 3.168 degrees
versus LS's 1.498. The analytic dense-gloss case regresses from 16.825 to
20.154 degrees (LS: 20.503). Mixed-scene Mitsuba error rises from 1.483 to 1.755
degrees (LS: 4.255); rough-gloss regional error rises from 3.900 to 5.166 degrees.
The no-sphere near-field validation scene improves from 3.677 to 3.607 degrees.
Sparse rejection cannot reliably distinguish coherent BRDF changes from shape.

All existing renderer-derived acceptance thresholds remain unchanged. One
analytic criterion changed explicitly: broad-gloss diagnostic coverage above
35% was replaced by solve coverage above 99%, while retaining improvement over
LS. On equal-elevation ring data, a constant specular contribution may be
indistinguishable from diffuse amplitude; residual-based labels cannot guarantee
its detection. This is not a claim of improved highlight segmentation.

## Real-Photo Replay And Build

A public-API replay samples 729 positions in the original 640-by-640 fish crop
and applies 16 independent minus-one/zero/plus-one raw RGB code perturbations
per position (11,664 comparisons; RNG seed 718832). It uses explicit sRGB
decoding, the same fixed common stack normalization, and the supplied 40 light
directions. In the fixed case, clipping and headroom weights stay at their
unperturbed values; in the dynamic case both are recomputed from perturbed RGB.

| Estimator | Fixed sensor decisions: p99 / maximum change | Dynamic sensor decisions: p99 / maximum change |
| --- | --- | --- |
| Reviewed robust estimator | 19.769 / 47.484 degrees | 25.285 / 49.294 degrees |
| Continuous robust estimator | 1.291 / 3.754 degrees | 3.097 / 18.105 degrees |
| Clipping-aware LS | 0.609 / 2.491 degrees | 11.138 / 44.043 degrees |

All sampled fits remain valid. In the dynamic replay, 15 continuous-robust
comparisons still exceed 10 degrees, versus 172 for clipping-aware LS. Those
remaining jumps must not be described as solved. The fixed-decision comparison
also shows that robust fitting remains more noise-sensitive than LS when its
outlier handling is not needed. There is no measured real-specimen normal
ground truth, and these code perturbations are not a calibrated sensor model.

The complete crop run takes approximately 3.7 seconds on the development host,
including input, solving, and image output; this is not a controlled throughput
benchmark. It retains 100% coverage, averages 39.94 IRLS updates per solved
pixel, and reports 0.0137% without established final convergence. Visual review
finds substantially less speckle and retained fin edges; it cannot certify
normal accuracy. The original specimen folders were not modified.

Local developer artifacts, ignored by Git:

- `build/fish-noise-review/check-current.cpp` and `current-sensitivity.txt`.
- `build/fish-noise-review/continuous-final-smoke/normal_rgb.png`.
- `build/fish-noise-review/continuous-final-smoke/hillshade_ul.png`.
- The smoke folder's `run_manifest.json` records estimator and convergence.

The final Release build passes all seven baseline CTest suites in 15.46 seconds:
input response, photometric core, I/O exports, Mitsuba process contract, and
classical, neural, and uncalibrated end-to-end workflows. This does not include
the optional live inverse-renderer tier or interactive GUI validation. The
developer executable is refreshed in `build-vcpkg-direct`; no installer,
release tag, commit, or GitHub push is made by this update.

## Deferred Work

- Per-light brightness calibration or constrained global gain refinement.
- Spatial coherence on outlier labels, with preservation of normal detail.
- A shared-normal RGB likelihood with explicit censoring of clipped channels.
- Independent material/lighting benchmarks not used during development.

The SBL prototype used modified noise estimates and a finite iteration budget;
its crop sensitivity was worse than the continuous candidate in the tested
configuration. That is not a general comparison against the published SBL
method. It was not promoted into production. Likewise, shadow uncertainty here
must not be described as Verbiest and Van Gool's spatial posterior inference.

The optional live Mitsuba CPU/CUDA inverse benchmarks have not been rerun for
this estimator update. Existing inverse-rendering reports remain historical.
