# Algorithm

What A Relief is an implementation-oriented tool, not a new claim of invention. The calibrated workflow follows the classical photometric stereo model introduced by Woodham, with light directions estimated from a user-marked highlight sphere. The uncalibrated workflow is a pragmatic visual-relief mode inspired by the unknown-lighting literature, especially Basri, Jacobs, and Kemelmacher, but it is intentionally simpler and aimed at useful enhancement rather than metrically unique shape recovery.

The program estimates image-space normals and relative brightness terms from a fixed-view image stack. It does not by itself calibrate physical height, pixel pitch, lens distortion, bidirectional reflectance, or microscope illumination nonuniformity.

## Interactive Sphere Calibration

In calibrated mode, the user marks the highlight sphere once on the first image by clicking three points on the sphere edge. The same circle is then used in every image. This assumes the image stack is registered and that the sphere does not move relative to the image.

Inside that circle, each image is searched for its specular highlight:

1. Convert the image to grayscale linear intensity.
2. Keep pixels inside the selected sphere.
3. Find a high percentile brightness threshold.
4. Compute a weighted centroid of the pixels above that threshold.
5. Convert the centroid into a normal on the sphere.
6. Reflect the camera view vector around that normal to recover the light direction.

For an ideal mirror sphere with orthographic view direction `V`, the surface normal at the highlight point bisects the view and illumination directions:

```text
L = 2 * dot(N, V) * N - V
```

The default view direction is `(0, 0, 1)`. Errors in the sphere outline, highlight centroid, saturation, sphere roughness, or perspective geometry directly affect the estimated light vectors.

## Photometric Stereo

For each object pixel, the calibrated solve uses a Lambertian model:

```text
I_i = dot(L_i, g)
g   = rho * n
```

`L_i` is the light direction for image `i`, `rho` is a relative albedo or brightness scale, and `n` is the unit normal. The program solves `g` by least squares. Observations below `--shadow-threshold` are omitted so that shadowed images do not dominate the fit. A pixel must have at least 3 usable observations, the fitted normal must face the camera, and the resulting residual is stored as a root-mean-square intensity error.

This direct solve supports 3 to 25 images. It does not use conventional RTI fitting because these captures usually have too few lighting samples for that style of spherical-harmonic reconstruction.

## Uncalibrated Unknown Lighting

Uncalibrated mode skips the sphere and estimates a relative normal field from the image stack. It needs at least 4 images. The implementation builds an image covariance over valid pixels, tries a rank-4 first-order unknown-lighting factorization, then falls back to a rank-3 PCA relief estimate when the metric constraint is degenerate.

Because unknown-lighting photometric stereo has an unavoidable ambiguity, uncalibrated mode is intended for visual relief enhancement rather than calibrated geometry. The program orients normals toward the camera, searches a rotation about the view axis that reduces a curl-like integrability cost, stabilizes extreme slopes, and removes a best-fit plane from the height preview.

Uncalibrated mode is sensitive to non-object pixels and to non-Lambertian structure. Use a crop or mask so shiny fixtures, the calibration sphere, and background do not dominate the factorization. Treat uncalibrated `normal_rgb.png`, `height.pfm`, and `surface.ply` as relative visual products.

## Height Preview

The normal field is converted to image-space gradients:

```text
dz/dx   = -nx / nz
dz/drow =  ny / nz
```

The program then solves a Poisson-style integration using a DCT. Invalid pixels have zero gradients in the current preview solver, and the solution is mean-centered over the valid mask. In uncalibrated mode, a best-fit plane is removed after integration.

The result is useful for visual inspection, while `normal_rgb.png`, the component maps, and `residual.png` remain the primary outputs.

If requested, the height preview can also be exported as a binary little-endian PLY mesh.

## Related Work

- Woodham, R. J. "Photometric Method for Determining Surface Orientation from Multiple Images." Optical Engineering 19(1), 139-144, 1980. https://doi.org/10.1117/12.7972479
- Basri, R., Jacobs, D. W., and Kemelmacher, I. "Photometric Stereo with General, Unknown Lighting." International Journal of Computer Vision 72(3), 239-257, 2007. https://doi.org/10.1007/s11263-006-8815-7
- Frankot, R. T., and Chellappa, R. "A Method for Enforcing Integrability in Shape from Shading Algorithms." IEEE Transactions on Pattern Analysis and Machine Intelligence 10(4), 439-451, 1988. https://doi.org/10.1109/34.3909
- Simchony, T., Chellappa, R., and Shao, M. "Direct Analytical Methods for Solving Poisson Equations in Computer Vision Problems." IEEE Transactions on Pattern Analysis and Machine Intelligence 12(5), 435-446, 1990. https://doi.org/10.1109/34.55103

The program also depends on OpenCV for image operations and windowing, and uses vcpkg/CMake to build OpenCV reproducibly on Windows.
