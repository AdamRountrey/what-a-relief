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

`L_i` is the light direction for image `i`, `rho` is a relative albedo or brightness scale, and `n` is the unit normal. The standard solver estimates `g` by least squares. Observations below `--shadow-threshold` are omitted so that shadowed images do not dominate the fit. A pixel must have at least 3 usable observations, the fitted normal must face the camera, and the resulting residual is stored as a root-mean-square intensity error.

The default calibrated solver is a modest robust variant. It estimates a per-pixel high-outlier cutoff from the median and median absolute deviation of the usable observations, caps that cutoff with `--high-outlier-threshold`, omits very bright observations when enough other observations remain, then uses an iteratively reweighted least-squares pass with Huber-style residual weights. This is not the same as the larger sparse-regression or low-rank robust photometric-stereo methods in the literature, but it follows the same practical motivation: shadows, saturation, and specular highlights should not control a diffuse Lambertian normal estimate. When too few observations are available, it falls back to the direct least-squares fit.

The robust solver also writes diagnostic maps for mean robust weight, shadow count, high-intensity outlier count, and, if requested, an experimental specular-cue mask. The mask identifies pixels whose observations behaved like shiny or otherwise non-Lambertian outliers. It is currently a review aid only; those pixels are not used as independent light-calibration constraints.

## Near-Field Ring Lighting

The standard calibrated solve uses one global light direction per image. This is appropriate when illumination is approximately directional at the sample scale. For close microscope or ring-light geometries, `--near-field-ring radius height` can instead treat each image as a point light on a ring. The sphere-derived or file-supplied light vector provides the azimuth of the light, while the user-provided radius and height define the point-light position in millimeters. Image pixel locations are converted into millimeters using `--pixel-scale-mm`, or by reading common TIFF physical scale tags when possible; the light positions themselves are not stored as image-pixel dimensions.

For each solved pixel, the light vector is recomputed from the surface pixel to the corresponding point light:

```text
P = ((x - cx) * pixel_scale_mm, (cy - y) * pixel_scale_mm, 0)
S = (radius_mm * cos(theta), radius_mm * sin(theta), height_mm)
L = normalize(S - P)
```

This is an approximation. It can reduce systematic errors when the light is close, but it does not yet model lens perspective, true 3D sample height, LED size, falloff, or spatial illumination nonuniformity. TIFF metadata support currently covers common classic TIFF resolution tags and the GeoTIFF pixel-scale tag; when those are absent or ambiguous, enter the pixel scale manually.

This direct solve supports 3 to 25 images. It does not use conventional RTI fitting because these captures usually have too few lighting samples for that style of spherical-harmonic reconstruction.

## Relief Flattening

Optional relief flattening removes a low-frequency slope trend from the normal field before writing the normal visualizations, relit image, optional height preview, and optional PLY mesh. The program converts normals to slopes, estimates a broad Gaussian low-pass version of those slopes inside the valid mask, subtracts a fraction of that low-frequency component, and converts the remaining slopes back to normals.

The goal is visual: broad sample tilt or curvature can be reduced so smaller topographic features stand out without sharpening pixel-scale noise. This is best understood as scale separation, related to Gaussian scale-space filtering and surface-texture filtering practice, not as calibrated form removal for metrology. Leave this option off when the large-scale shape itself is scientifically important.

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
- Huber, P. J. "Robust Estimation of a Location Parameter." The Annals of Mathematical Statistics 35(1), 73-101, 1964. https://doi.org/10.1214/aoms/1177703732
- Ikehata, S., Wipf, D., Matsushita, Y., and Aizawa, K. "Robust Photometric Stereo Using Sparse Regression." CVPR 2012, 318-325. https://doi.org/10.1109/CVPR.2012.6247691
- Wu, L., Ganesh, A., Shi, B., Matsushita, Y., Wang, Y., and Ma, Y. "Robust Photometric Stereo via Low-Rank Matrix Completion and Recovery." ACCV 2010, LNCS 6494, 703-717. https://doi.org/10.1007/978-3-642-19318-7_55
- Lindeberg, T. "Scale-Space Theory in Computer Vision." Kluwer/Springer, 1994. https://doi.org/10.1007/978-1-4757-6465-9
- ISO 16610-61:2015 specifies areal Gaussian filters for separating large- and small-scale surface components in surface texture work.
- Frankot, R. T., and Chellappa, R. "A Method for Enforcing Integrability in Shape from Shading Algorithms." IEEE Transactions on Pattern Analysis and Machine Intelligence 10(4), 439-451, 1988. https://doi.org/10.1109/34.3909
- Simchony, T., Chellappa, R., and Shao, M. "Direct Analytical Methods for Solving Poisson Equations in Computer Vision Problems." IEEE Transactions on Pattern Analysis and Machine Intelligence 12(5), 435-446, 1990. https://doi.org/10.1109/34.55103

The program also depends on OpenCV for image operations and windowing, and uses vcpkg/CMake to build OpenCV reproducibly on Windows.

Relight and RelightLab from the CNR-ISTI Visual Computing Lab are acknowledged as important related RTI software. Comparing What A Relief against Relight helped identify useful workflow ideas, such as explicit robust-normal and flattening controls. What A Relief does not include Relight source code.
