# Algorithm

What A Relief is an implementation-oriented tool, not a new claim of invention. The calibrated workflow follows the classical photometric stereo model introduced by Woodham, with light directions estimated from a user-marked highlight sphere. The optional neural-fusion workflow adds a pretrained PS-FCN normal prior after the classical solve, but keeps geometry outputs on the classical path. The uncalibrated workflow is a pragmatic visual-relief mode inspired by the unknown-lighting literature, especially Basri, Jacobs, and Kemelmacher, but it is intentionally simpler and aimed at useful enhancement rather than metrically unique shape recovery.

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

## Experimental Neural Fusion

When the experimental neural-fusion option is enabled, the program still performs the full classical calibrated solve first. It then runs a bundled ONNX export of the pretrained PS-FCN model that matches the number of input images from 3 through 25.

The neural model expects one directional light vector per image, stacks the grayscale image channels as repeated RGB inputs, normalizes the stacked intensities by the per-pixel energy across the image set, and predicts a dense normal map. Because PS-FCN was trained for directional-light photometric stereo rather than microscope-specific near-field metrology, What A Relief treats it as a qualitative normal prior rather than as a replacement physical model.

Fusion is done in slope space rather than by directly averaging RGB normals:

1. Convert the classical and neural normals into slope fields `p = -nx / nz` and `q = ny / nz`.
2. Build low-frequency slope fields for each source with masked Gaussian smoothing.
3. Blend the low-frequency fields with a confidence term derived from the classical residual, robust weight, shadow count, and highlight-outlier count.
4. Reinject a reduced amount of high-frequency classical slope detail.
5. Clamp the fused slope magnitude against the classical slope distribution.
6. Convert the fused slopes back into a normal map.

The fused normals are used for the visualization-oriented normal outputs. To avoid exaggerated geometry, height preview and PLY mesh generation remain on the classical geometry path even when neural fusion is enabled. Fusion runs write three explicit normal sets: classical, neural, and fused, including the RGB normal image, X/Y/Z component images, and an upper-left hillshade image.

## Near-Field Ring Lighting

The standard calibrated solve uses one global light direction per image. This is appropriate when illumination is approximately directional at the sample scale. For close microscope or ring-light geometries, `--near-field-ring radius height` can instead treat each image as a point light on a ring. The sphere-derived or file-supplied light vector provides the azimuth of the light, while the user-provided radius and height define the point-light position in millimeters. Image pixel locations are converted into millimeters using `--pixel-scale-mm`, by reading common TIFF physical scale tags when possible, or in the GUI by drawing a scale line of known length on the first image; the light positions themselves are not stored as image-pixel dimensions.

For each solved pixel, the light vector is recomputed from the surface pixel to the corresponding point light:

```text
P = ((x - cx) * pixel_scale_mm, (cy - y) * pixel_scale_mm, 0)
S = (radius_mm * cos(theta), radius_mm * sin(theta), height_mm)
L = normalize(S - P)
```

This is an approximation. It can reduce systematic errors when the light is close, but it does not yet model lens perspective, true 3D sample height, LED size, falloff, or spatial illumination nonuniformity. TIFF metadata support currently covers common classic TIFF resolution tags and the GeoTIFF pixel-scale tag; when those are absent or ambiguous, enter the pixel scale manually or draw a scale line from a known ruler/scale bar in the image.

This direct solve supports 3 to 25 images. It does not use conventional RTI fitting because these captures usually have too few lighting samples for that style of spherical-harmonic reconstruction.

## Relief Flattening

Optional relief flattening removes a low-frequency slope trend from the normal field before writing the normal visualizations, relit image, optional height preview, and optional PLY mesh. The program converts normals to slopes, estimates a broad Gaussian low-pass version of those slopes inside the valid mask, subtracts a fraction of that low-frequency component, and converts the remaining slopes back to normals.

The goal is visual: broad sample tilt or curvature can be reduced so smaller topographic features stand out without sharpening pixel-scale noise. This is best understood as scale separation, related to Gaussian scale-space filtering and surface-texture filtering practice, not as calibrated form removal for metrology. Leave this option off when the large-scale shape itself is scientifically important.

Height curl correction is separate. It is applied only after height integration and only affects `height.png`, `height.pfm`, and PLY export. The radial mode subtracts a fitted centered dome term plus residual plane. The quadratic mode subtracts a full second-order surface (`1, x, y, x^2, y^2, xy`) over the height mask. These options are intended for broad integration curl, not for calibrated form removal.

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

The default height solver is a masked, weighted Poisson-style integration. It works on the specimen height mask rather than on the full rectangular image, soft-clamps extremely steep normal-derived slopes, assigns lower confidence to grazing normals, and runs a small iteratively reweighted solve so inconsistent gradient constraints have less ability to produce broad ramps.

The faster optional solver uses a DCT/Poisson preview. Because a DCT solve expects a rectangular domain, the program fills the slope field outside the integration mask before solving and then mean-centers the result over the geometry mask.

By default the geometry mask is the solved-pixel mask. If the user supplies or draws a specimen height mask, that mask is intersected with the solved-pixel mask and used only for height integration and PLY export. This is intended for specimens sitting above a visible table or background, where crossing the depth discontinuity can dominate the integrated height. It does not change `normal_rgb.png`, the component maps, albedo, residuals, liquid-metal renders, relighting, or solve diagnostics.

The result is useful for visual inspection, while `normal_rgb.png`, the component maps, and `residual.png` remain the primary outputs. The height preview has a best-fit plane removed after integration so broad image-wide ramps do not dominate the display.

If height-only curl correction is enabled, the program then subtracts a fitted radial dome or quadratic trend over the height mask and removes the best-fit plane again. This follows the same practical idea as RelightLab's optional radial/quadratic flattening controls: reduce broad systematic bias when the specimen is expected to be roughly flat on average, while leaving the primary normal and reflectance products untouched.

If requested, the height preview can also be exported as a binary little-endian PLY mesh. The standard mesh is an open surface for inspection. The printable mesh option creates a second PLY as a watertight solid: it writes the sampled top surface, duplicates the vertices onto a flat bottom plane, writes reversed bottom faces, and closes every boundary edge with a side face. Printable export requires pixel scale and writes coordinates in millimeters.

## RTI Export

Optional RTI export fits a PTM-style polynomial appearance model from the original color image stack and the calibrated or loaded light directions. Small 3-to-8-image stacks use the stable first-order basis:

```text
I(u,v) = a0 + a1*u + a2*v
```

Larger, better-constrained stacks use the quadratic basis when the light geometry is numerically stable:

```text
I(u,v) = a0 + a1*u + a2*v + a3*u^2 + a4*u*v + a5*v^2
```

The coefficient planes can be stored in three layouts. The plain image and DeepZoom layouts use Relight/OpenLIME-style `info.json` plus coefficient JPEGs. RGB mode stores one RGB JPEG per PTM coefficient. LRGB mode stores an unscaled base image in `plane_0` and stores relative luminance coefficients in the remaining JPEGs; this follows the PTM LRGB convention used by Relight/OpenLIME, where the viewer multiplies the base image by the relit luminance. The plain image layout writes full-size coefficient images, and the DeepZoom layout writes one DZI pyramid per stored plane image.

The webRTIViewer layout writes the older `info.xml` plus quadtree component JPEGs expected by `createRtiViewer(...)` in jcupitt/webRTIViewer. For that layout, coefficients are reordered into the shader's expected PTM order:

```text
u^2, v^2, uv, u, v, 1
```

The RGB webRTIViewer export writes six component layers, filling unconstrained high-order terms with zero for first-order exports. The LRGB webRTIViewer export writes three component layers: high-order luminance coefficients, low-order luminance coefficients, and the RGB base image.

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
- Malzbender, T., Gelb, D., and Wolters, H. "Polynomial Texture Maps." SIGGRAPH 2001, 519-528. https://doi.org/10.1145/383259.383320
- Queau, Y., Durou, J.-D., and Aujol, J.-F. "Normal Integration: A Survey." Journal of Mathematical Imaging and Vision 60(4), 576-593, 2018. https://doi.org/10.1007/s10851-017-0773-x
- Queau, Y., Durou, J.-D., and Aujol, J.-F. "Variational Methods for Normal Integration." Journal of Mathematical Imaging and Vision 60(4), 609-632, 2018. https://doi.org/10.1007/s10851-017-0777-6
- Agrawal, A., Chellappa, R., and Raskar, R. "An Algebraic Approach to Surface Reconstruction from Gradient Fields." ICCV 2005, 174-181.
- Chen, G., Han, K., and Wong, K.-Y. K. "PS-FCN: A Flexible Learning Framework for Photometric Stereo." ECCV 2018. https://doi.org/10.1007/978-3-030-01252-6_1

The program also depends on OpenCV for image operations and windowing, and uses vcpkg/CMake to build OpenCV reproducibly on Windows. The optional experimental neural-fusion models are exported from the PS-FCN project and are bundled as third-party MIT-licensed assets.

Relight and RelightLab from the CNR-ISTI Visual Computing Lab are acknowledged as important related RTI software. Comparing What A Relief against Relight helped identify useful workflow ideas, such as explicit robust-normal and flattening controls. What A Relief does not include Relight source code.

webRTIViewer and webGLRTIMaker are acknowledged for the `info.xml` plus component-tile RTI layout supported by the `webrti` export mode. What A Relief writes compatible folders but does not include webRTIViewer source code.
