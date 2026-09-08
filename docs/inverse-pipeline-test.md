# Independent Inverse Pipeline Acceptance Test

`tests/test_mitsuba_pipeline.py` adds a curved-scene improvement test to the
small matched-model tilted-plane regression. It invokes the actual C++
application for robust normals, DCT height integration, and the optional
inverse worker. It never substitutes truth geometry for the photometric
initialization or injects a convenient synthetic height error.

## Fixed Protocol

- Eight finite disk emitters: ring radius 18 mm, height 8 mm, diameter 0.8 mm.
- A 12 mm square field, 24 x 24 pixels, 0.5 mm/pixel.
- A diagonal raised ridge, a rounded bump, and a shallow depression on a
  continuous height field with no overhangs.
- Smooth albedo texture and a glossy patch using nonlinear Beckmann
  rough-plastic reflection with IOR 1.45 and roughness 0.22. This is not the
  inverse worker's diffuse/two-GGX basis construction.
- Scalar CPU forward path tracing, depth limit 4, 256 samples/pixel, seed
  84021 plus fixed per-light offsets. Camera rays are orthographic, while the
  inverse worker uses its existing narrow-perspective approximation.
- Independent Poisson shot noise, 2-electron read noise, a 30,000-electron
  full well, and 12-bit ADC quantization. Linear observations are stored as
  16-bit PNGs without per-image exposure normalization.
- The production preview inverse preset, CPU by default. The initial record
  used six iterations and a 3 x 3 control grid; the updated worker uses twelve
  iterations, a 13 x 13 control grid on this image, and alternating material
  fits. The later integration correction changes the classical initialization
  and adds a measured-normal constraint. Input images and geometry-accuracy
  thresholds remain unchanged; the historical results below identify each stage.

Forward scenes and material construction do not call the inverse worker's
scene, basis, emitter, or prediction helpers. They still use Mitsuba, so this
is not an independent-renderer benchmark. Forward height and shading normals
come from renderer AOVs. Cast-shadow labels use separate center-source
visibility rays from exact surface intersections; they are binary center-ray
labels, not finite-emitter penumbra probabilities. Nonfinite forward pixels
are errors, not silently clamped replacement observations.

The implementation follows the documented [AOV and path integrators](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#arbitrary-output-variables-integrator-aov)
and [orthographic sensor](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html#orthographic-camera-orthographic).
The finite path-depth limit, material model, pixel sampling, and sensor noise
remain simulation assumptions, not a calibration of a particular microscope.

## Pass Criteria

The test requires an accepted correction, at least 5% lower height RMSE, and
at least 5% lower integrated-normal mean angular error. It also requires
finite output, at least 90% solved coverage over the fixed interior, and at
least 12 supported cast-shadow pixels. Rejection, timeout, missing backend,
and process failure are failures, not successful skips or expected failures.
The current test also requires an enabled photometric-normal prior and checks
that an accepted correction satisfies its declared consistency bound.

This is a height-field improvement gate. It does not assert that inverse
normals beat the original, unintegrated photometric normal map. The separate
`normal_products` report decodes both RGB8 normal products without sRGB
conversion and records their angular errors and an explicit improvement flag.
Those quantized diagnostic scores must not be confused with the height-derived
normal gate or full-precision ground-truth normal accuracy.

Before/after errors use the identical supported interior pixels. Height RMSE
is in millimeters and removes only an additive offset, never tilt, scale, or
polynomial drift. Normal error compares the integrated baseline and inverse
height surfaces against independent AOV normals. Separate scores cover cast
shadows, the ridge half, and the bump/depression half. These regional scores
are reported but are not additional pass thresholds.

Passing would establish one synthetic end-to-end improvement, not recovery
of arbitrary black glossy specimens, absolute height, or fine features below
the test's pixel resolution.

## Running

Use a compatible private backend runtime and a built application with its
OpenCV DLLs available. No system Python or Anaconda environment is needed.

```powershell
& .\build\packaged-mitsuba-review\python.exe -B tests\test_mitsuba_pipeline.py `
  --app build-vcpkg-direct\what-a-relief.exe `
  --worker tools\mitsuba_backend\worker.py `
  --output build\inverse-pipeline-test
```

Use `--backend cuda` for a separate GPU run. GPU and CPU results must be
reported separately. Each attempt preserves its inputs, truth, application
log, output products, and `metrics.json` in a new `curved-*` directory.
Application failures or timeouts may occur before metrics are written.
The application phase has an 1800-second limit, with process-tree termination
on timeout so its renderer is not left running. The former 600-second limit
was increased for the larger optimization budget and material-checkpoint
renders, not to weaken accuracy requirements. CTest allows 2100 seconds.

Append `--light-offset-radians 0.39269908169872414` for a separate capture with
all eight lights rotated by 22.5 degrees. The default remains the original
fixed capture. This is a lighting generalization check on the same specimen,
not an unseen-specimen or independent-renderer benchmark.

CTest registration is explicitly opt-in because this is a new scientific
acceptance case, not part of the previously validated eight-suite baseline:

```powershell
cmake -S . -B build\owned-mitsuba-gates `
  -DWHAT_A_RELIEF_MITSUBA_PIPELINE_TEST=ON
ctest --test-dir build\owned-mitsuba-gates `
  -R '^mitsuba-pipeline-reconstruction$' --output-on-failure
```

That build must already set `WHAT_A_RELIEF_MITSUBA_PYTHON`. The CTest entry
uses the compiled application target, runs serially, and preserves a failing
exit status. Enabling or registering the test is not evidence that it passes.

## Recorded Results: 2026-09-08

**The initial worker failed the improvement test on both CPU and CUDA.** Both
completed six iterations, but rejected their proposed correction because
withheld-light error worsened. The test did not relax its acceptance or
geometry thresholds to turn this rejection into a pass.

| Measurement | CPU | CUDA |
| --- | ---: | ---: |
| Training image loss improvement | 1.9403% | 1.8925% |
| Withheld image loss change | 1.9621% worse | 2.6433% worse |
| Height RMSE, before and delivered after | 0.153330 mm | 0.153330 mm |
| Integrated-normal MAE, before and delivered after | 7.275989 degrees | 7.275989 degrees |
| Solved coverage on fixed interior | 100% | 100% |
| Approximate application time from log timestamps | 586 s | 59 s |

These timings include the baseline solve and backend startup. They are not
a controlled speed benchmark; later test reports use a monotonic timer.
Input PNG SHA-256 hashes were identical between the CPU and GPU runs.
There were 324 evaluation pixels, including 163 labeled as cast-shadow
receivers for at least one source-center ray. Before/after equality reflects
baseline preservation following rejection, not an unchanged optimizer state.

Artifacts are under `build/inverse-pipeline-test/`:

- `curved-zd5c4cvj/metrics.json`: CPU result and per-region errors.
- `curved-4nu1tx4f/metrics.json`: CUDA result and per-region errors.
- `curved-29p_kieb/metrics.json`: repeat CUDA run using the final report format,
  including explicit `passed: false`, failed assertions, input hashes, and
  monotonic timings. Fixture generation took 5.00 s and the application
  22.26 s. Training loss improved 1.9680%, while withheld loss worsened
  3.7690%; the delivered geometry and rejection outcome were unchanged.
- Each directory also contains the full inverse `result.json`, images,
  geometry truth, and `application.log`.

The existing eight suites passed separately after adding this test (137.91 s);
that result explicitly excluded `mitsuba-pipeline-reconstruction`. CTest
registration of the opt-in ninth case was verified, while its CPU and CUDA
results above came from direct script execution.

An initial 48 x 48 standard-preset CPU attempt exceeded 600 seconds. An
initial LLVM forward-rendering attempt also produced nonfinite values;
the final fixture uses scalar forward rendering and rejects nonfinite data.
Neither unsuccessful attempt supplies a reconstruction-quality result.

The GPU repeats used the same input images but did not produce bit-identical
Monte Carlo optimization metrics. The repeat confirms this rejection outcome,
not deterministic GPU gradients or a stable hardware speedup factor.

This exposes a gap between the matched-model tilt regression and independent
end-to-end improvement. The result does not identify whether material/model
mismatch, parameterization, weighting, calibration assumptions, or iteration
budget is responsible. Those should be investigated without choosing new
thresholds or an easier scene merely to manufacture an improvement claim.

## Alternating-Fit Update: 2026-09-08

The updated worker keeps the same geometry acceptance thresholds and fixture
images, but uses two-render-pixel height-control spacing, twelve preview Adam
iterations at 16 spp, and training-only material refits every four iterations.
It retains the lowest training-loss checkpoint, including checkpoint zero.
Withheld observations never select the checkpoint or fit its material maps.
Final gates retain the 1% training-improvement requirement, 0.2% maximum
withheld-loss worsening, displacement/slope bounds, and second validation seed.
This is an application-specific schedule, not a new published method;
projective derivatives still follow [Zhang et al. (2023)](https://rgl.epfl.ch/publications/Zhang2023Projective).

The full C++ application passed the unchanged global improvement assertions
on CUDA for the original capture and the separate rotated capture:

| Capture | Height RMSE, before -> after (mm) | Normal MAE, before -> after (degrees) | Withheld loss improvement | Second-seed improvement |
| --- | --- | --- | ---: | ---: |
| Original | 0.153330 -> 0.109714 | 7.275989 -> 5.118115 | 25.76% | 27.09% |
| Lights rotated 22.5 degrees | 0.122351 -> 0.101197 | 6.408334 -> 4.738533 | 26.87% | 26.75% |
| Original, CPU | 0.153330 -> 0.110213 | 7.275989 -> 5.144596 | 27.96% | 29.39% |

All three selected iteration 12 with 100% evaluation coverage. CUDA artifacts are
`build/inverse-pipeline-test/curved-mz0ar4vy/metrics.json` (original) and
`build/inverse-pipeline-test/curved-1bcimb3n/metrics.json` (rotated).
The application took 146.24 s during other tests and 36.44 s respectively;
these are not controlled performance comparisons. The existing eight CTest
suites also passed in 218.80 s, including live CPU tilt/truth-start recovery.
The CPU full-pipeline pass is in `curved-oblpyuei/metrics.json`, with 458.38 s
application time. A final rebuilt-app eight-suite rerun, including checkpoint
selection assertions, passed in 109.06 s. These three full-application cases
were run directly; they are not a claim that a single nine-entry CTest run
was performed.

**The original photometric normals are still better.** The final deployed-app
repeat, `curved-zv21p0sb/metrics.json`, explicitly records RGB8 normal errors
of 2.841694 degrees for the original photometric product versus 5.155149 degrees
for the inverse product. Its height-derived baseline was 7.275989 degrees.
Thus `passed: true` for height recovery coexists with
`inverse_improves_classical_normal_product: false`. Do not replace the
original photometric normal map with inverse normals on the strength of this
test. Feeding reliable original normals into the inverse optimization is an
important next step, not implemented by this update.

One attempt (`curved-dmxijg71`) launched the bare CMake executable without the
OpenCV runtime path and failed with Windows code `0xc0000135` (missing DLL).
That was a test-launch error, not an inverse solve. The successful final repeat
used `build-vcpkg-direct` with adjacent DLLs. The test harness now requests
[inherited noninteractive critical-error handling](https://learn.microsoft.com/en-us/windows/win32/api/errhandlingapi/nf-errhandlingapi-seterrormode)
and supplies an actionable missing-DLL failure; it does not modify the installed
application or a system-wide Windows setting.

**Regional regressions remain.** In the original capture, bump/depression
height RMSE worsened from 0.093758 to 0.126966 mm even though normals improved
from 6.259182 to 5.492931 degrees. In the rotated capture, that region's
height worsened from 0.064188 to 0.113146 mm and normals from 4.960380 to
5.248425 degrees; cast-shadow-region height worsened from 0.070134 to
0.106461 mm. Whole-image gains are dominated by the raised ridge. They must
not be described as uniform improvement or calibrated height recovery.

Diagnostic runs, not full-application acceptance cases, remain under
`build/inverse-pipeline-test/`: `diagnostic-k22br69b` used spacing 4, 24 steps,
and frozen material (about 13% height improvement, less than 5% normal gain);
`diagnostic-m36rjsu7` used spacing 2, 24 steps, and alternating material
(about 30% height and 34% normal gains). These combined-factor experiments
do not isolate each change's contribution. The shorter production preview
was subsequently checked through the real application as recorded above.

Next priorities are unseen shapes/materials and noise seeds, regional
non-regression, normal-confidence priors, larger resolutions, and higher
preset convergence. Fixed camera/material/indirect-lighting/datum assumptions
remain. Older installer packages do not contain this worker update.

## Integration And Measured-Normal Update: 2026-09-08

The preceding results used a defective DCT boundary/discretization scheme.
It treated pixel-center normal slopes as forward-edge differences, retained
outward right/bottom flux terms, and zero-padded divergence. New noiseless
plane and quadratic tests failed with range-normalized RMSE of 0.116 to
0.285. The correction averages endpoint slopes onto edges, excludes edges
leaving the domain, and extends slopes before constructing the padded DCT
system. Both integrators now share the edge convention. Analytic tests cover
even and odd/padded dimensions and preserve the original normal arrays.
The existing smooth cosine fixture was also corrected to supply analytic
pixel-center derivatives rather than forward differences; its accuracy
thresholds were not relaxed. See the cited derivation background in
[Algorithm](algorithm.md#height-preview).

The worker additionally uses the original classical normal field as a
confidence-weighted orientation constraint, streamed in three float32 PFM
components. The reduced-resolution weighted unit-vector loss has coefficient
0.25 and an additional acceptance bound of `1.05 * baseline_loss + 1e-6`.
These heuristic choices and limits are documented in
[Algorithm](algorithm.md#experimental-mitsuba-inverse-refinement). No image
acceptance or geometry-improvement threshold was loosened. This is a combined
integration/prior update, not an isolated ablation of the prior coefficient.

| Capture/backend | Corrected baseline -> inverse height RMSE (mm) | Height-derived normal MAE (degrees) | Withheld improvement | Second-seed improvement |
| --- | --- | --- | ---: | ---: |
| Original/CUDA | 0.106070 -> 0.061006 | 4.455530 -> 3.501712 | 13.25% | 13.47% |
| Original/CPU | 0.106070 -> 0.061011 | 4.455530 -> 3.514866 | 13.96% | 14.49% |
| Rotated/CUDA | 0.070470 -> 0.046699 | 3.388918 -> 3.003634 | 13.16% | 13.77% |

The original baseline had height RMSE 0.153330 mm and integrated-normal MAE
7.275989 degrees. The new baseline improvement is from integration alone;
the photometric normal products are unchanged. All three new full-application
runs accepted iteration 12 with 100% evaluation coverage. Normal-prior loss
fell from 0.003658 to 0.002640 (original/CUDA), 0.003658 to 0.002635
(original/CPU), and 0.002344 to 0.002081 (rotated/CUDA). Artifacts are
`curved-2v_55z20`, `curved-p38scxkp`, and `curved-0cn4qwmq`, respectively,
under `build/inverse-pipeline-test`. Application times were 39.42, 209.05,
and 47.75 seconds; concurrent work makes these unsuitable as speed benchmarks.

**Do not replace original normals based on these scores.** Original-capture
RGB8 photometric normals remain at 2.841694 degrees versus 3.514676 degrees
for the CUDA inverse product (3.526169 CPU). The rotated capture is 2.455031
versus 3.009787 degrees. The improvement flag against original normals remains
false. A height field can improve globally without matching every accurate
local normal, and coarse geometry differentiation introduces additional
discretization error.

**Local regressions remain.** Original/CUDA bump-and-hollow height RMSE is
0.062358 -> 0.065232 mm. Rotated/CUDA is 0.039492 -> 0.053402 mm, with
height-derived normal error 2.284765 -> 2.788586 degrees. Cast-shadow-region
height now improves in both captures, but that does not cancel the regional
bump-and-hollow regression. Broader unseen specimens, larger resolutions,
normal-product non-regression, and convergence remain open qualification work.

The final staged-app CUDA repeat, `curved-xmob4u4w/metrics.json`, also passed,
including the explicit prior-enablement and consistency assertions. Height
RMSE was 0.106070 -> 0.062408 mm, height-derived normal MAE 4.455530 ->
3.570955 degrees, and inverse RGB8 normal MAE 3.588247 degrees. Withheld and
second-seed improvements were 13.00% and 13.43%; prior loss fell to 0.002648.
Bump-and-hollow height still regressed to 0.070079 mm. Application time was
70.55 seconds. GPU Monte Carlo results are not bit-identical across repeats.

All eight CTest suites passed in 171.30 seconds; the preserved log is
`build/integration-prior-validation.log`. The DLL-equipped development folder
`build-vcpkg-direct` contains the matching executable and worker, verified by
this final run without modifying the installed application or backend.
Executable SHA-256:
`B16DFB7E8E5916386FFF556730E7F5D2C6BB22A6802F928EA56CDA2CDA279BD5`.
Worker SHA-256:
`BF9B5C7A984533371836D0ED2F7C9B1DEF26F2CA277A39DA5053EF6A27A86093`.
No installer or public release was created for this update.

## v0.2.2 Installer Verification

The following later checks use the application and private backend extracted
from the newly built 0.2.2 installers, without installing either package.
Both run manifests report version 0.2.2. They pass the global improvement,
coverage, finite-output, normal-prior, and acceptance requirements above:

| Capture/backend | Height RMSE before -> after (mm) | Height-derived normal MAE before -> after (degrees) | Artifact under `build/release-v0.2.2-pipeline` |
| --- | --- | --- | --- |
| Original/CPU | 0.106070 -> 0.059552 | 4.455530 -> 3.480321 | `curved-qfm39fdn/metrics.json` |
| Rotated/CUDA | 0.070470 -> 0.046081 | 3.388918 -> 2.973360 | `curved-2201g6b2/metrics.json` |

These remain small synthetic checks, not metrology qualification. Original
photometric normals still outperform inverse normals. The rotated capture
still regresses in bump-and-hollow height (0.039492 -> 0.050946 mm).
The application build passed all six baseline CTests, including diagnostic
export on/off and same-folder cleanup; the backend packager separately passed
progress/PNG tests, eleven numerical tests, CPU derivatives and tilt/truth
recovery, live backend probes, and extraction verification. The portable ZIP
passed its model/license/runtime and input-exclusion checks. Local logs are
`build/release-v0.2.2-*.log`; CI rebuilds must pass their own gates.
