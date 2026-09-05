# Mitsuba robust-solver fixtures

This directory contains an offline, independently rendered regression suite for shadow, highlight, clipping, and normal recovery. The committed assets are consumed directly by CTest. Mitsuba and Python are not runtime or CI dependencies.

The suite deliberately varies geometry, spatial albedo, material response, light count, light elevation, and finite-versus-directional illumination:

- `robust_v1` is a development fixture with eight equal-elevation directional lights, a Lambertian floor and sphere, and two GGX rough-plastic spheres.
- `textured_primitives_v1` is a development fixture with twelve irregular directional lights at varying elevations, a checkerboard floor, a smooth generated relief mesh, a textured cylinder, a tilted sharp-gloss cube, and a textured disk.
- `holdout_relief_v1` is the historically named validation fixture. It has ten finite point lights on a `3.6 mm` radius, `1.25 mm` height ring; a smoothly varying bitmap-albedo floor; a corrugated rough-plastic relief mesh; a dark glossy cylinder; a tilted rough-gloss cube; and a textured rough-plastic disk. It contains no spheres.

This scene began as an untouched holdout, but its hidden height and cast-shadow truth were inspected while developing the experimental cast-shadow correction. It is therefore validation data now. A future blind generalization test must use a newly rendered scene and declare its acceptance criteria before inspecting the results.

Mitsuba supplies visibility, microfacet reflection, world-space positions and shading normals, material albedo, and shape-index AOVs. The generator adds a deterministic linear sensor simulation with Poisson shot noise including a small dark signal, Gaussian read noise, 12-bit quantization, and clipping.

Truth masks come from matched diagnostic renders:

- `shadow_truth/` combines attached shadows with substantial visibility loss measured against an unoccluded Lambertian prediction.
- `attached_shadow_truth/` records only locally back-facing samples.
- `cast_shadow_truth/` records front-facing samples with substantial visibility loss and diffuse deficit.
- Highlights are pixels where a physical rough-plastic render has significant excess over a matched render with only its specular term disabled.
- Saturation is defined from the final simulated ADC code.

The reference AOV exports also include `position_z.png`, a 16-bit container with 12 effective bits encoding world-space surface height from `-0.25` to `1.0` scene millimeters. Invalid pixels are set to the encoding minimum so stochastic subpixel geometry outside the eroded solve mask cannot change the file hash. The scene manifest records that mapping. It is used only as hidden synthetic truth for height-refinement tests.

The committed corpus was generated with the `cuda_ad_rgb` backend and `--spp 128`. Reference AOVs use 128 samples per pixel. Radiance renders use a square 12 by 12 one-sample supersampled film and fixed-order downsampling, giving 144 samples per output pixel while preserving the camera aspect ratio. This avoids nondeterministic parallel accumulation; two independent full-corpus regenerations on the pinned backend must be byte-for-byte identical. Reference normals are stored in 16-bit PNG containers after deterministic 12-bit quantization, which is much finer than the angular acceptance bounds. To regenerate all scenes with Mitsuba 3.8.0 and NumPy 2.3.3:

```powershell
python -m pip install mitsuba==3.8.0 numpy==2.3.3
python tests/fixtures/mitsuba/generate_fixture.py --all --variant cuda_ad_rgb --spp 128
```

Use `--scene NAME` to regenerate one named scene and `--variant llvm_ad_rgb` on a CPU installation with LLVM available. Every scene manifest records its development/validation split, geometry labels, light vectors and any finite positions, physical scale, sensor settings, counts, and SHA-256 hashes. Review all manifests and run the complete CTest suite after regeneration. Changes to a scene, renderer version, truth thresholds, sampling, or sensor model intentionally require review of the resulting scientific acceptance metrics.

Mitsuba 3 is by Wenzel Jakob and contributors and is distributed under its BSD-style license. The exact license supplied with the pinned package is reproduced in `LICENSE-MITSUBA.txt`. See the project third-party notices and cite:

Wenzel Jakob, Sebastien Speierer, Nicolas Roussel, and Delio Vicini. "Dr.Jit: A Just-In-Time Compiler for Differentiable Rendering." *ACM Transactions on Graphics* 41(4), 2022. https://doi.org/10.1145/3528223.3530099

Wenzel Jakob et al. *Mitsuba 3 renderer*, version 3.8.0, 2022-2026. https://mitsuba-renderer.org/
