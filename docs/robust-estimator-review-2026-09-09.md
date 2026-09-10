# Robust estimator review: instability, model mismatch, and next experiments

Date: 2026-09-09. Review target: the uncommitted working tree based on
`bd66376` (v0.2.4), including the recent automatic input-response and robust
initialization changes. Line numbers below describe that working tree, not the
v0.2.4 release tag.

## Conclusion

The remaining speckle cannot reasonably be dismissed as unavoidable image noise.
The current estimator can amplify changes of at most one 8-bit RGB code into
normal changes exceeding 40 degrees, even with clipping decisions held fixed.
Correct sRGB decoding and polishing the initial least-squares hypothesis did not
resolve this instability. More iterations did not resolve most of it either.

The evidence supports reworking the estimation objective and model selection,
not adding normal-map smoothing. It does not establish the true normals of the
fish specimen, or prove that least squares is more accurate for every material.

No production code, original photographs, or original output files were changed
for this review. A standalone diagnostic executable was built and run in the
ignored build directory. The results below are diagnostic experiments, not new
release gates or a claim that an improved algorithm has been implemented.

## Findings, highest priority first

### P1: Discrete hypothesis selection creates large normal jumps

Locations: `src/photometric.cpp:713`, `:757`, `:883`, `:962`.

Candidates compete primarily on an integer count of measurements within 3.5
heuristic noise standard deviations. A bounded residual loss only breaks ties.
Tiny changes can therefore promote a different three-light hypothesis with a
very different normal. The least-squares candidate receives limited local
polishing, but that does not remove the discontinuous selection rule.

The fixed-clipping perturbation experiment below demonstrates this failure in
the current production solver. At the worst sampled location, the selected
initial normals changed by 50.54 degrees; the final normals changed by 47.48
degrees. Starting refinement of the perturbed measurements from the original
final solution instead produced a change of only 0.25 degrees. This strongly
implicates hypothesis selection and distinct optimization basins. It does not
identify which basin contains the physically correct solution.

The correction should be evaluated against a consistent continuous objective,
not a different integer score or more independently tuned thresholds. A
redescending loss remains nonconvex: simply replacing the score with a Cauchy
sum is not, by itself, a demonstrated solution.

### P1: Residual-scale capping forces narrow-inlier behavior under model mismatch

Location: `src/photometric.cpp:1047`.

The code estimates residual scale from median absolute residuals, then caps it
at `max(0.015, 0.05 * max(0.1, intensityScale))`. In the sampled fish region,
712 of 729 locations hit that cap. The median uncapped estimate was 2.38 times
the cap. This restricts the contribution of residual dispersion to the rejection
thresholds precisely where observations disagree substantially with the model.

Those residuals contain some unknown mixture of noise, non-Lambertian response,
illumination error, and other acquisition effects. They are not a measurement of
sensor noise. But forcing them into a narrow noise interpretation encourages
unstable subset selection instead of identifying model inadequacy.

Removing the cap alone is not justified: inflating scale can instead absorb real
shadows and highlights. Separate dense noise/model discrepancy from sparse
corruption, and evaluate both clean-data efficiency and outlier rejection.

### P1: Scoring, rejection, and weighted fitting do not optimize one stated model

Locations: `src/photometric.cpp:637`, `:708`, `:734`, `:1060`, `:1097`, `:1133`.

Hypothesis scoring uses the heuristic observation-noise scale alone. Refinement
uses a different, larger scale that includes capped residual dispersion. It then
adds hard zero-weight decisions, azimuth-neighbor threshold changes, and a
half-vector multiplier that switches on at a standardized residual of 1.5.
Consequently, a candidate preferred by one stage need not be preferred by the
next. No common objective or descent check reconciles these decisions.

There is also a weighting-contract issue. For an intended objective
`sum rho(r_i / sigma_i)`, with fixed per-observation scales during an IRLS step,
the weight multiplying the *raw* squared residual must be proportional to
`(psi(z_i) / z_i) / sigma_i^2`, where `z_i = r_i / sigma_i`. The current solver
uses the dimensionless Cauchy factor without `1 / sigma_i^2`. Because the
scales differ between measurements, this is not IRLS for that standardized
objective. It can be called a heuristic reweighting scheme, but should not be
presented as optimizing that likelihood. Updating scales from the current
prediction requires an explicit optimization contract as well.

This is not a claim that inserting inverse variance alone will cure the speckle.
The noise model and the other threshold rules must be made consistent first.

### P1: Direction calibration does not calibrate per-image light strength

Locations: `src/photometric.cpp:1414`, `:1486`, `:1530`.

The ordinary directional workflow obtains unit directions from the sphere or
normalizes imported CSV vectors. There is no separate relative light-strength
parameter in that solve. Fixed exposure does not guarantee equal incident
illumination, and vector magnitudes in an imported file cannot carry it because
they are discarded. Near-field distance falloff does not independently calibrate
each lamp's emitted intensity either.

A controlled Lambertian experiment below demonstrates a substantial normal bias
from unequal gains with no shadows, highlights, or image misregistration.
Robust rejection is not a substitute for radiometric lighting calibration.

Measured observed/predicted ratios also vary systematically between lights in
the fish crop, but they are confounded by reflectance and normal/model errors.
They are not valid gain estimates and do not prove that unequal light power is
the cause in that dataset.

### P2: A clipped color channel removes the whole luminance measurement

Locations: `src/image_io.cpp:86`; `src/photometric.cpp:1079`.

Clipping in any RGB channel sets a binary mask that removes the entire
luminance observation from robust fitting. This is conservative for a scalar
luminance model, but discards potentially useful unclipped channels. Exact
white-code transitions can also abruptly change the accepted lighting matrix.
Allowing the clipping mask to change in the perturbation experiment destabilized
both robust fitting and clipping-aware least squares.

Do not solve this by accepting clipped luminance as an exact measurement or by
silently replacing luminance with whichever color channel remains. A future RGB
model can share the normal across channels while fitting channel-specific
diffuse amplitudes and treating clipping as a censored observation. This is a
proposal requiring separate validation, not a feature already supported.

### P2: Successful fitting and current tests do not establish reliable normals

Locations: `src/photometric.cpp:1188`, `:1247`;
`tests/test_photometric.cpp:1141`, `:1165`; `docs/validation.md:283`.

The solver accepts a fit based on sufficient weighted observations, finite
front-facing normals, and a local lighting condition threshold. These checks do
not assess sensitivity to measurement perturbations or competing solutions.
The sampled fits typically retained 20 observations, so this is not principally
a three-observation fallback problem.

The iteration loop has a parameter-change stopping rule, but no common-objective
convergence check or explicit iteration-limit status. An additional 100 updates
changed most results negligibly, though nine sampled locations changed by more
than one degree. More iterations alone are not the general remedy.

Existing analytic and rendered tests are useful but insufficient. In particular,
the broad-gloss test gates improvement over LS and diagnostic cue detection,
not accurate normal recovery; its documented robust mean error is still about
16.8 degrees. Clean, correctly calibrated rendering tests also do not expose all
of the discontinuities caused by clipping, uncertain lighting, and dense model
mismatch. Passing those tests must not be equated with scientific adequacy.

### P2: Azimuth-only neighbors are not general lighting neighbors

Locations: `src/photometric.cpp:1099`, `:1312`.

The classifier tightens rejection thresholds using predecessor/successor lights
sorted by azimuth. That is plausible for a single ring but ignores elevation and
angular separation in arbitrary macro-photography setups. It should not be used
as unrestricted evidence that nearby light directions agree. A future coherence
model needs actual angular distances and explicit capture-model assumptions.
This limitation was identified by inspection, not isolated in the fish experiment.

## Experiments performed

### Real-data numerical sensitivity

The existing 640 by 640 lossless crop inputs were used, corresponding to an
original-image crop starting at `(3000, 1900)`. The light file was read from the
original fish output, without modifying it. The stack contains 40 images.

Protocol:

1. Decode RGB with the sRGB inverse transfer function and use the application's
   linear-luminance conversion and common stack normalization.
2. Sample a regular 24-pixel grid beginning at `(12, 12)`: 729 locations.
3. At each location make 16 deterministic random perturbations. Each original
   8-bit B, G, and R value changes by -1, 0, or +1 code, clipped to [0, 255].
   Keep the original stack normalization fixed.
4. Refit through the actual current `solveRobustPixel` implementation. Compare
   against least squares using the same low-signal exclusion and clipping masks.
5. First hold clipping masks fixed to isolate estimator sensitivity. Repeat with
   masks recomputed from perturbed RGB to expose clipping-decision sensitivity.

These are 11,664 trials per method per clipping condition. The perturbations are
a bounded numerical sensitivity test, **not** a calibrated sensor-noise model.
The angles are changes relative to each method's own unperturbed normal, not
errors relative to ground truth. No pixels were spatially filtered. The LS
reference is clipping-aware, not the unmodified ordinary LS UI option.

| Clipping masks | Estimator | Median change | 95th percentile | 99th percentile | Maximum | Changes over 10 degrees |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | Current robust | 0.234 | 4.353 | 19.769 | 47.484 | 382 / 11664 |
| Fixed | Clipping-aware LS | 0.105 | 0.241 | 0.609 | 2.491 | 0 / 11664 |
| Recomputed | Current robust | 0.420 | 10.616 | 25.285 | 49.294 | 623 / 11664 |
| Recomputed | Clipping-aware LS | 0.992 | 6.530 | 11.138 | 44.043 | 172 / 11664 |

All angles are degrees. With recomputed masks, the median and tail comparisons
differ; neither a single average nor a screenshot adequately describes the
failure. There were no baseline robust solve failures in this sample.

Additional observations:

- 521 / 729 locations had at least one clipped observation.
- 712 / 729 had uncapped residual-scale estimates above the implemented cap.
- Median effective inlier count was 20; 727 / 729 had more than 10.
- Adding 100 classification/weighted-fit steps changed the normal by a median
  0.00057 degrees and a 95th percentile of 0.0153 degrees. Nine locations changed
  by more than one degree; the maximum was 9.68 degrees.
- Worst fixed-mask location: crop `(228, 132)`, original `(3228, 2032)`.
  Initial-hypothesis change: 50.54 degrees. Final change: 47.48 degrees.
  Warm-starting from the original final fit: 0.255 degrees.

### Controlled unequal-light-strength test

This separate, generic test has known normals and no fish-specific parameters.
It is a forward Lambertian intensity test, not a new Mitsuba-rendered scene.

For 8 and 40 evenly spaced ring lights, unit directions are
`(0.8 cos(a), 0.8 sin(a), 0.6)`. Relative emitted gains are
`1 + 0.2 cos(a - 0.4) + 0.12 sin(2a)`. For each light count, 1,000 unit normals
are generated from `(x, y, 1)` with x and y uniform in [-0.3, 0.3], and diffuse
amplitude uniform in [0.15, 0.65]. Independent Gaussian intensity noise has
standard deviation 0.001. All surfaces face all lights; there is no clipping,
cast shadow, specular term, or registration error.

| Lights | Robust assuming equal gains: mean error | LS assuming equal gains: mean error | Robust with true gains: mean error |
| ---: | ---: | ---: | ---: |
| 8 | 7.271 degrees | 7.999 degrees | 0.136 degrees |
| 40 | 7.323 degrees | 8.025 degrees | 0.060 degrees |

True gains were supplied directly to the internal fitting helper as scaled light
vectors. This is an experiment with known calibration, not a demonstration of
automatic gain recovery. The production CSV importer currently normalizes away
such magnitudes. With 8 lights, the wrong-gain robust maximum error was 18.39
degrees versus 9.16 for wrong-gain LS, despite robust's lower mean.

### Local artifacts and reproducibility

The ignored directory `build/fish-noise-review` contains `review.cpp`,
`review-console.txt`, `review-samples.csv`, the diagnostic CMake target, and
`bin/review.exe`. The harness uses `cv::RNG(718832)` and directly includes the
current photometric implementation so the tested selection and classification
logic is not a separately reimplemented approximation.

From a Visual Studio x64 developer shell, in the repository:

```powershell
cmake --build build/fish-noise-review/bin --target review
Push-Location build/fish-noise-review
$env:Path = (Resolve-Path '..\ninja-vcpkg\vcpkg_installed\x64-windows\bin').Path + ';' + $env:Path
& '.\bin\review.exe'
Pop-Location
```

These local artifacts depend on the existing crop inputs and saved light file;
they are not a portable checked-in regression suite. The method and numerical
results are recorded here so the conclusions survive build-directory cleanup.
The diagnostic target compiled and completed successfully. Release CTest gates
were not rerun for this documentation-only production change.

## Published methods and what is applicable

### Sparse corruption with a separate noise model

Ikehata, Wipf, Matsushita, and Aizawa's **Robust Photometric Stereo using Sparse
Regression** (CVPR 2012) models ordinary error separately from sparse corrupted
observations and estimates them through sparse Bayesian learning. The small
normal system makes a CPU reference implementation plausible without spatially
blurring normals. It is a strong method to benchmark against our threshold chain,
not a guarantee: optimization is nonconvex, noise choices matter, and sparse
corruption assumptions weaken when most observations contain broad specular
response. Eight-image behavior needs its own validation. [Author paper](https://satoshi-ikehata.github.io/SBLPS_cvpr2012.pdf).

Their **Photometric Stereo Using Sparse Bayesian Regression for General Diffuse
Surfaces** (TPAMI 2014) adds a more flexible monotonic diffuse response instead
of explaining every non-Lambertian deviation as an outlier. This is relevant to
dense material mismatch, but extra response parameters consume limited lighting
constraints. A highly flexible per-pixel model is not automatically identifiable
from eight partially shadowed or clipped observations. [Author-hosted paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/06714613.pdf).

### A coherent objective and imperfect lighting

Queau, Wu, Lauze, Durou, and Cremers' **A Non-Convex Variational Approach to
Photometric Stereo Under Inaccurate Lighting** (CVPR 2017) combines robust
estimation, shadow handling, and lighting refinement in a defined optimization
problem. Its use of a Cauchy-type loss is a reminder that Cauchy itself is not
our error; the surrounding model and solver contract matter. Their depth-based
formulation is not a drop-in replacement for an independent normal solve.
We should investigate bounded relative-intensity refinement without making all
normal outputs depend on height integration. [Official paper page](https://openaccess.thecvf.com/content_cvpr_2017/html/Queau_A_Non-Convex_Variational_CVPR_2017_paper.html).

### Coherent outlier probabilities instead of normal smoothing

Verbiest and Van Gool's **Photometric Stereo with Coherent Outlier Handling and
Confidence Estimation** (CVPR 2008) uses a probabilistic inlier/outlier model,
spatial coherence of the inlier maps, and uncertainty in scaled normals. The
useful distinction is to stabilize observation validity rather than simply blur
the recovered geometry. This is a second-stage candidate, not a substitute for
fixing our pointwise estimator. Spatial classification constraints can still
erase small shadow/highlight structures when too strong, and sparse eight-light
captures need independent assessment. [Author's institutional record](https://www.esat.kuleuven.be/psi/members/00018504),
[paper](https://citeseerx.ist.psu.edu/document?doi=a896b337a36b8c53e369998cb872586264cd3e7e&repid=rep1&type=pdf).

### Color is useful, but not a universal highlight-removal solution

Mallick, Zickler, Kriegman, and Belhumeur's **Beyond Lambert: Reconstructing
Specular Surfaces Using Color** (CVPR 2005) separates specular effects through
a linear color transformation under a dichromatic reflectance model. The paper
also identifies the degeneracy when body and illuminant colors coincide and
the loss of signal-to-noise near that case. It is not a general remedy for
neutral or very dark specimens, and does not recover information destroyed by
saturation. This should remain separate from a channelwise clipping-aware fit.
[Author paper](https://www.eecs.harvard.edu/~zickler/download/photodiff_cvpr05_preprint.pdf).

### Independent validation with relevant geometry

**DiLiGenT-Pi** (ICCV 2023) provides near-planar surfaces with rich detail and
varied materials, a better complement to spheres than another sphere-only
accuracy test. Benchmark predefined eight-light subsets as well as denser sets;
do not report a many-image paper result as evidence for eight-image performance.
[Authors' dataset page](https://photometricstereo.github.io/diligentpi.html).

**DiLiGenRT** (CVPR 2024) adds quantified roughness and translucency variation,
useful for checking where a diffuse-plus-sparse-outlier model ceases to apply.
It complements, rather than replaces, fine-detail objects. Neither dataset was
downloaded or evaluated during this review. [Official paper page](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_DiLiGenRT_A_Photometric_Stereo_Dataset_with_Quantified_Roughness_and_Translucency_CVPR_2024_paper.html).

Reference implementations exist for [sparse robust PS](https://github.com/yasumat/RobustPhotometricStereo)
and [variational robust PS](https://github.com/yqueau/robust_ps). These advertise
GPL licenses, unlike this application's BSD license. Treat them as separately
licensed research references; do not copy their implementation into production
as though they were BSD components. An independent implementation from a paper
still needs appropriate method attribution.

## Recommended development sequence

1. **Establish fair references and instability gates first.** Preserve ordinary
   LS, add an explicitly clipping-aware LS reference, and implement a simple
   continuous robust reference with a documented objective. Keep identical
   radiometry, lighting, masks, and output scaling across comparisons.
2. **Benchmark sparse Bayesian regression against a coherent robust objective.**
   Define residual/noise units and weight semantics; use monitored optimization
   and report convergence, ambiguity, and insufficient evidence. Remove the
   current mixture of scoring and rejection rules only when the replacement
   passes both accuracy and stability gates. Do not merely increase iterations.
3. **Represent relative light strength explicitly.** Support measured gains and
   investigate constrained automatic refinement using high-confidence diffuse
   regions. Fix a gain normalization convention, bound changes, and test
   identifiability. Do not simultaneously free all normals, light directions,
   intensities, and ambient terms from eight observations without constraints.
4. **Preserve useful color information and model clipping.** Evaluate a shared
   normal with channel-specific amplitudes and censored clipped measurements.
   Keep confidence honest where all channels or too many lights are unusable.
5. **Only then test weak outlier-field coherence.** Compare detail retention and
   classification boundaries, not just smoother-looking normal maps. Make no
   automatic geometry-smoothing assumption.

These are proposed experiments, not promises that a named algorithm will be
superior on all specimens. CPU operation should remain the baseline. Inverse
rendering or a neural prior should not conceal instability in the initial solve.

## Required evaluation expansion

- Make the fixed-mask one-code sensitivity experiment portable using synthetic
  observation vectors with known normals and controlled mismatch. Also test
  clipping-boundary transitions, small calibration perturbations, light order,
  deterministic repeatability, and solver failure coverage. Set thresholds from
  independent fixtures before tuning, not from the fish crop alone.
- Retain realistic rendered scenes, materials, textures, occluders, narrow and
  broad gloss, and unseen holdouts. Vary emitted gains, light elevation,
  directional uncertainty, penumbrae, ambient/interreflection, linear camera
  noise, exposure, and clipping. Include 8-light rings and irregular macro-light
  layouts, plus denser sets. Unsupported cases should lose confidence, not be
  hidden by filtering the evaluated pixels.
- Test image formation separately: linear data, sRGB encoding, quantization,
  per-channel saturation, JPEG compression, and subpixel misregistration.
  Knowing the output color encoding does not identify every upstream camera
  processing operation; inverse sRGB is not proof of radiometric calibration.
- Report ground-truth angular error mean, median, 95th/99th percentiles, severe
  errors, and valid coverage per material and object. Include uncertainty
  calibration, shadow/highlight precision and recall, and normal-derived detail
  amplitude. Measure hillshade under fixed display settings, not per-image
  contrast normalization. A quieter image alone is not proof of better normals.
- Require near-LS accuracy and noise efficiency on clean diffuse scenes and
  meaningful improvement over fair LS on corrupted scenes. Establish explicit
  tolerances; avoid an unsupported universal claim of beating LS everywhere.
- Measure CPU time, peak memory, iterations, and per-light scaling. Avoid an
  image-resolution graph or repeated full-stack work unless it produces a
  measured accuracy benefit. Current small weighted solves are inexpensive;
  repeated hypothesis scoring is the first pointwise workload to benchmark.

## Remaining uncertainties

Implementation follow-up: [continuous estimator update](robust-estimator-update-2026-09-09.md).
The findings and prototype recommendations above describe the reviewed version,
not the later implementation. The follow-up records what was implemented,
what was deferred, and measured gains and regressions.

The fish data have no independently measured normal ground truth. This review
does not apportion the observed image artifacts between clipping, BRDF mismatch,
lighting strength/direction error, camera processing, and registration. It does
demonstrate that the current solver magnifies small perturbations far more than
necessary, and that current input/calibration assumptions can independently
produce appreciable geometric bias. Fix those general problems before treating
the remaining artifacts as specimen-specific noise.
