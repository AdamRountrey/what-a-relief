# Scientific Quality Roadmap

This is the durable release checklist for the 0.2.1 quality update. A box is checked only when implementation, automated evidence, and user documentation are all present. Local validation and GitHub CI completion are tracked separately.

## Scientific Core

- [x] Decode selected sRGB inputs with the standard piecewise transfer function and use one common stack scale so lighting ratios are preserved.
- [x] Cover 8-bit, 12-bit-in-16-bit, 16-bit, floating-point, and sRGB normal recovery with deterministic tests.
- [x] Bind full calibration CSV rows to image identity; reject missing, mixed, duplicate, ambiguous, zero, and ill-conditioned lighting data.
- [x] Support 3 through 25 images while making the no-redundancy three-image fallback explicit.
- [x] Separate hard-shadow omission, dark residual outliers, bright outliers, and half-vector gloss cues in diagnostics.
- [x] Validate sparse shadows, penumbrae, saturated highlights, multiple bright outliers, and broad glossy failure behavior quantitatively.
- [x] Implement the nearby ring as a documented, center-normalized isotropic point-source model with millimeter geometry and inverse-square attenuation.
- [x] Keep neural predictions outside the evidence mask from entering fused outputs, and test the shipped ONNX workflow.

## Geometry And Scale

- [x] Make height flattening `none` preserve the integrated field and give plane, radial, and quadratic modes distinct documented bases.
- [x] Parse classic TIFF physical resolution conservatively and convert GeoTIFF model scale only when a supported linear unit is declared.
- [x] Quantitatively test full-field and irregular-mask height integration.
- [x] Parse binary printable PLY output in tests and require a closed genus-zero fixture with a planar millimeter base.

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

The detailed thresholds and current reference observations are maintained in [validation.md](validation.md).
