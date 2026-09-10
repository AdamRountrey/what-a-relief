# Exposure-Threshold Stability, September 10, 2026

## Scope

Implemented `adaptive_pseudo_huber_cauchy_v2`, retaining the existing robust
workflow and GUI. No extra solver choices, normal-map smoothing, new image
dependencies, or changes to height integration were added. Ordinary LS remains
unchanged. This is a stability improvement, not a guarantee of better normals
on every specimen.

The changed production files are `src/robust_fit.hpp`, `src/photometric.cpp`,
and `src/run_manifest.cpp`. The existing upper-headroom generation in
`src/image_io.cpp` is unchanged by this update.

## Implemented Method

- A measurement enters gradually above the existing low cutoff `T`, with
  `smoothstep(clamp((I-T)/sigma(T,T),0,1))`. Values at or below `T` remain
  excluded. The existing normalized-intensity noise heuristic defines the
  transition width; this is not a newly calibrated camera-noise estimate.
- The product of low-signal and upper-headroom reliability weights initializes
  the fit and defines its weighted mean signal reference. The diagnostic noise
  reference uses that same signal reference, not an independent unweighted
  median that can change when a marginal observation is admitted.
- After the convex pseudo-Huber stage, solve the bounded scale equation
  `sum(h*r^2/(r^2+(c*s)^2))/sum(h) = 0.5`, with
  `c = 0.6120031809624806` and the existing positive noise floor. Input reliability
  `h`, not final residual-dependent IRLS weights, enters this equation.
- The constant satisfies
  `c*sqrt(pi/2)*exp(c*c/2)*erfc(c/sqrt(2)) = 0.5`, giving Gaussian population
  normalization for the unweighted scale equation. Data-dependent reliability,
  few observations, fitted residuals, and model mismatch preclude claiming a
  calibrated finite-sample uncertainty or full-estimator breakdown guarantee.
- Bracketing starts at the noise floor and doubles; bisection has a relative
  interval tolerance of `1e-6` and at most 40 iterations. A nearly zero-weight
  extreme residual cannot inflate the initial search interval.
- The final Cauchy phase still uses a fixed scale, objective-descent checks,
  and up to 80 updates. Conditioning and effective-support checks remain.
  Labels remain diagnostics, not instructions to remove observations and refit.

The bounded M-scale framework is described by
[Yohai (1987)](https://doi.org/10.1214/aos/1176350366). Our rational score and
reliability weighting are application choices, not his complete MM estimator
or its high-breakdown initialization. The IRLS and photometric loss references
remain [Holland and Welsch (1977)](https://doi.org/10.1080/03610927708827533)
and [Queau et al. (2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Queau_A_Non-Convex_Variational_CVPR_2017_paper.html).
Signal-dependent noise and clipping tests are motivated by
[Foi et al. (2008)](https://doi.org/10.1109/TIP.2008.2001399); their automatic
camera-noise estimator is not implemented. No external implementation was
copied or added to the runtime.

## Controlled Tests

The low-boundary fixture uses diffuse and glossy surface normals, spatially
varying albedo, and one partially occluded light. The same observation crosses
the cutoff from `0.02-1e-6` to `0.02+1e-6`, without changing true geometry.
These are analytic observation-level tests, not globally ray-traced shadows.

| Lights/material | Previous maximum normal jump | Revised maximum jump |
| --- | ---: | ---: |
| 8, diffuse | 0.831 deg | below 0.00001 deg |
| 8, glossy | 5.148 deg | below 0.00001 deg |
| 16, glossy | 1.687 deg | below 0.00001 deg |
| 40, glossy | 0.466 deg | below 0.00001 deg |

All 576 samples in each case remain solved. Eight-light diffuse mean error
changes from 0.258 to 0.006 degrees; glossy mean error changes from 3.042 to
2.603 degrees. Very small reported errors are limited by floating-point normal
storage. This does not imply equally tiny sensitivity to realistic sensor noise.

Nine new low-exposure cases use 8/16/40 lights, independently noisy repeated
diffuse samples, a high-count shot-noise approximation (50,000 electrons at
unit linear signal), 0.0005 normalized read-noise sigma, and 8-bit linear,
8-bit sRGB, or 16-bit linear quantization. All pixels remain solved. Eight-light
mean errors are 1.021/0.725/0.666 degrees; LS gives 1.019/0.702/0.648. Robust
fitting is slightly less efficient here, as expected on clean noisy diffuse
data. These tests keep explicit absolute and LS-relative accuracy gates.

The upper-boundary test now covers 8/16/40/64 lights. It rebuilds clipping
masks and headroom after each one-code RGB perturbation. All samples remain
solved. Paired p99 changes are 1.638/1.206/0.929/0.797 degrees and mean errors
are 0.653/0.480/0.408/0.365 degrees. The original 40-light 1.5-degree p99 gate
is preserved; the added eight-light case has a 2.5-degree gate because of its
lower redundancy. Separate tests verify scale normalization, order and unit
invariance, noise floors, and zero/vanishing-weight contamination.

## Saved Comparison Replay

All 28 datasets from the [earlier comparison](robust-solver-comparison-2026-09-10.md)
were replayed against the revised native driver: 27,616 observation rows, not
that many independent specimens. The replay did not alter its baseline results
or original specimen inputs. The external L1/SBL methods were not rerun.

Real-photo sensitivity on 1,536 paired perturbations, with full coverage:

| Input handling | Previous p99 / maximum | Revised p99 / maximum |
| --- | ---: | ---: |
| Dynamic clipping and headroom | 2.755 / 9.055 deg | 1.938 / 4.265 deg |
| Fixed upper clipping/headroom | 1.459 / 2.784 deg | 0.754 / 1.479 deg |

These are one-code perturbations of selected pixels in a 40-light crop, not
camera-noise calibration or ground-truth accuracy. The binary-clipping arm
without the upper taper worsens slightly (p99 7.823 to 8.188 degrees), reinforcing
that stable scale estimation does not replace exposure reliability handling.

Rendered angular accuracy on pixels supported by both robust versions:

| Scene | Common pixels | Previous mean | Revised mean | Coverage before / after |
| --- | ---: | ---: | ---: | ---: |
| Mixed materials, 8 lights | 1,982 | 1.747 deg | 1.798 deg | 99.15% / 99.10% |
| Textured primitives, 12 lights | 2,000 | 0.884 deg | 0.880 deg | 100% / 100% |
| Near-field relief, 10 lights | 1,915 | 3.469 deg | 3.535 deg | 96.05% / 95.80% |

The mixed and near-field scenes therefore have small accuracy regressions on
common support. Near-field own-support mean decreases from 3.614 to 3.547
degrees, but that must not be presented as an accuracy improvement: marginal
pixels were lost. With unsupported normals scored as 180 degrees, mixed-scene
mean changes 3.275 to 3.402, and near-field mean 10.581 to 10.958. The 180-degree
penalty is an audit convention, not a measured error for missing normals.

Clean-noise replay changes are below 0.001 degrees in mean error. Broad-gloss
mean error remains about 18 degrees in that comparison; gain-mismatch failures
remain. The full committed renderer-derived regression gates, including
per-object accuracy and shadow/highlight precision/recall, still pass without
loosening their thresholds. These fixtures are development/validation evidence,
not a fresh blind holdout after parameter selection.

## Candidates Not Kept

A four-noise-sigma dark transition suppressed too much weak diffuse evidence
and failed four existing rendered-scene accuracy/classification gates. It was
replaced by the narrow one-sigma transition; those gates were not weakened.

Fixed 2%, 3%, and 4% upper-headroom tapers were tested on noisy-color cases,
the rendered scenes, and the local crop. Wider tapers improved dynamic photo
sensitivity but not all synthetic-noise metrics. For example, eight-light
noisy-color paired p99 changes from 0.964 to 1.000/1.018 degrees, while
64-light mean error changes from 0.480 to 0.481/0.482 degrees. These differences
are small, not proof of statistical superiority of any width. The existing 2%
default is retained pending a defensible noise-aware camera model, rather than
tuning the production upper taper to one real dataset.

## Pre-Release Verification And Artifacts

- Release build completed with VS2022/vcpkg OpenCV, without Anaconda.
- All seven CTest suites passed in 18.75 seconds. This includes the committed
  Mitsuba-rendered fixtures, not new live-renderer inverse optimization.
- A 640 x 640, 40-light crop smoke run completed with full normal coverage,
  40.327 mean IRLS updates, and 0.003174% unconfirmed convergence. The run ID
  in its manifest is `adaptive_pseudo_huber_cauchy_v2`.
- The developer executable was refreshed at
  `build-vcpkg-direct/what-a-relief.exe` from the tested CMake executable;
  both SHA-256 values are
  `050f2fb60917b381d63ec01a6ad9f085eb4c39e23c6256eae4da82302d3171a0`.
- At this experimental checkpoint, no installer, commit, push, or release was
  made. Optional live Mitsuba
  CPU/GPU numerical and inverse-reconstruction gates were not run.

The smoke products are in `build/threshold-stability/fish-smoke/`, including
`normal_rgb.png`, `hillshade_ul.png`, and `run_manifest.json`. These are inspection
outputs, not truth. Original specimen folders were not modified.

Versionable records are in [benchmarks/threshold-stability-2026-09-10/](benchmarks/threshold-stability-2026-09-10/):
`replay.json` records per-case accuracy, coverage, pair sensitivity, timing,
and source hashes; `headroom-sweep.json` records the exploratory upper-width
comparison; `before-boundaries.txt` and `after-boundaries.txt` record the new
boundary tests. The sweep used the earlier equivalent-scale bisection bracket;
the final replay uses the revised floor-doubling bracket and records its exact
source hash. Timings in the replay include one-pixel public-API overhead and
were not controlled throughput measurements; no processing speedup is claimed.

Original replay commands (requiring the saved baseline arrays and native driver):

```powershell
python tools/compare_robust/replay.py --results build/robust-comparison/results-20260910 --native build/threshold-stability/native/compare-native.exe --out build/threshold-stability/new-replay
python tools/compare_robust/headroom_sweep.py --native build/threshold-stability/native/compare-native.exe --out build/threshold-stability/new-sweep
build/ninja-vcpkg/what-a-relief-tests.exe --exposure-boundaries
ctest --test-dir build/ninja-vcpkg --output-on-failure
```

Set the OpenCV runtime directory on `PATH` for direct executable calls. The
Python analysis tools need NumPy/Pillow; neither is added to the application.
Release cleanup removed the ignored comparison arrays, exploratory runs, and
temporary native driver. It retained the summary records above and the latest
fish smoke products. To run a new comparison, regenerate arrays and build a
driver using [the research harness instructions](../tools/compare_robust/README.md).
Exact historical before/after replay is not available from summary files alone;
the pre-release executable hash above identifies that checkpoint, not v0.2.5.
The regression gates themselves remain self-contained in C++ and committed
rendered assets, requiring neither private images nor Python.
