# Robust Estimator Comparison

Standalone developer experiment, not part of the application or its release gates.
It reads an existing Release core library. No solver source or fixture is changed.

Release cleanup removes generated comparison folders and external checkouts,
not this harness or the versioned numerical reports. Obtain the reference source
separately and regenerate inputs before running a new experiment. Historical
reports identify interim source/executable hashes; running the current release
does not reproduce the previous uncommitted estimator exactly.

Protocol declared before results (2026-09-10):

- Compare clipping-aware LS, independently implemented Huber with delta 5/255
  (10 or 80 reweightings), the current Cauchy production API, and the authors'
  unmodified L1/SBL numerical functions. Huber is a controlled comparator, not
  an exact Relight reproduction: it shares our exclusions, does not retain
  excluded samples to maintain five observations, and uses our convergence rule.
  Our caps count reweightings after an initial LS fit; Relight's cap counts ten
  total fits. Eighty updates is an extended budget, not guaranteed convergence.
- Pin Yasuyuki Matsushita's reference repository to
  `f03aa95b57e746a7d31df76b1c0fa0a83584a3c1`; import its `rpsnumerics.py` externally.
  Do not vendor its GPL code into the application. Record SHA-256 in results.
- Main arm: identical finite observations above 0.02, exact clipping exclusions,
  and float32 linear luminance. No renderer shadow/highlight labels enter fitting.
- Secondary arm: common upper-headroom weights. LS/SBL receive square-root
  row preweighting, L1 receives linear row preweighting, Huber/Cauchy include
  headroom in their objectives. Weighted SBL is an adapted input treatment,
  not an assertion of equivalence to a published weighted SBL noise model.
- Run reference functions at their default 1000-iteration budget and 1e-8
  absolute tolerance. Test both normalized luminance and 255-scaled luminance,
  because the reference regularizers/tolerances are not scale invariant. All
  values remain floating point; scaled units do not add 8-bit quantization.
- All three committed rendered scenes use deterministic spatial samples of up
  to 2000 valid pixels per scene, chosen independently of solver output. Report
  per-object results, coverage, common-support error, and unsupported-as-180-degree
  error. These are development/validation data, not new blind holdouts.
- Analytic controls span 8/16/40/64 lights: diffuse sensor noise, color clipping,
  broad gloss, and deliberately mismatched light brightness. Tiny image changes
  are evaluated separately from ground-truth angular error.
- Fish replay uses fixed positions and perturbations, with no normal truth.
  It measures repeatability, not scientific reconstruction accuracy.
- Keep all existing output folders untouched. New results go in an ignored build
  subfolder. Record sample arrays and environment information for reproduction.
- Do not tune constants after seeing these results. A new setting is a new
  experiment. Do not promote a solver based only on average error or appearance.

Run from the repository root in a Visual Studio developer PowerShell, using the
existing Release core library and OpenCV/Zlib package paths:

```powershell
$deps = (Resolve-Path build/ninja-vcpkg/vcpkg_installed/x64-windows).Path
cmake -S tools/compare_robust -B build/robust-comparison/native -G Ninja -DCMAKE_BUILD_TYPE=Release "-DOpenCV_DIR=$deps/share/opencv4" "-DCMAKE_PREFIX_PATH=$deps"
cmake --build build/robust-comparison/native
$env:Path = "$deps/bin;" + $env:Path
python tools/compare_robust/compare.py --reference ../author-rps-20260910 --native build/robust-comparison/native/compare-native.exe --out build/robust-comparison/results
python tools/compare_robust/report.py --results build/robust-comparison/results --out build/robust-comparison/tables
python tools/compare_robust/convergence.py --results build/robust-comparison/results --reference ../author-rps-20260910 --out build/robust-comparison/extended-iterations
```

Use a developer Python with NumPy and Pillow, not Anaconda or the installed
application backend. The native executable needs the existing vcpkg DLL directory
on PATH. BLAS is restricted to one thread by the script. Python versus C++ times
include different wrapper overhead and do not predict a ported solver's speed.

The external reference checkout must be obtained separately from
[RobustPhotometricStereo](https://github.com/yasumat/RobustPhotometricStereo)
at the pinned commit above. Its source is not included in this repository.
The reference functions keep their original constants and stopping rules.
The unit/255 alternatives are sensitivity controls, not per-scene tuning.

`--only pristine` runs the exact-Lambertian sanity check. A full run may take
tens of minutes because several reference fits reach their iteration budgets.
Completed reference arrays are checkpointed. Rerunning with the same environment
resumes them; a changed harness, reference, executable, or data fingerprint
requires a new output directory. Partial runs retain existing summary entries.
`report.py` requires all major datasets and the sanity check to be present.
It independently verifies the native LS normals against NumPy least squares.

The real fish crop is local data, not distributed with the repository; the
runner skips it when absent. For this full comparison report, it is required.
Use `report.py --allow-missing-fish` to audit the complete synthetic/rendered
subset without that crop. The flag is recorded in the report's audit file.
The fish "fixed" replay freezes exact clipping and upper-headroom decisions,
not the common lower-intensity cutoff. No truth normals exist for that replay.
The report also isolates pairs with identical effective input weights, including
the lower cutoff. Its post-hoc shadow/highlight strata never influence fitting.

The secondary convergence script runs 10,000-step SBL on 32 evenly spaced rows
from each of five declared cases, independently of their errors. It records
changes from the default-budget normals without replacing the main results.
This small check is not a fully converged re-evaluation of the entire dataset.

See [the comparison report](../../docs/robust-solver-comparison-2026-09-10.md) for
results, source references, and the important distinction between this
clipping-aware LS control and the application's ordinary LS path.

## Threshold-Stability Follow-Up

`replay.py --results SAVED_RESULTS --native NEW_COMPARE_NATIVE --out NEW_FOLDER`
replays every saved input against a new standalone driver, comparing its Cauchy
normals with the saved baseline and matched-input LS. It reports common-support
accuracy, coverage, unsupported-as-180 scoring, per-region errors, and paired
perturbation sensitivity. It never reruns or imports the external authors' SBL
implementation. Use a separate build folder so the baseline executable remains
available. The output folder must not exist. Native timings include the driver's
per-pixel public-API allocations and are not full-application throughput.

`headroom_sweep.py --native NEW_COMPARE_NATIVE --out NEW_FOLDER` regenerates the
comparison's noisy-color, rendered, and optional local photo cases at fixed
2%, 3%, and 4% upper tapers. This is a research-only change to input weights;
it does not modify the GUI or production defaults. The September 10 evaluation
retains 2% because wider transitions were not consistently beneficial across
the noisy synthetic cases. See the [threshold-stability record](../../docs/threshold-stability-2026-09-10.md).
