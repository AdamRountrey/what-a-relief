# Algorithm

What A Relief is an implementation-oriented tool, not a new claim of invention. The calibrated workflow follows the classical photometric stereo model introduced by Woodham, with light directions estimated from a user-marked highlight sphere. The optional neural-fusion workflow adds a pretrained PS-FCN normal prior after the classical solve, but keeps geometry outputs on the classical path. The uncalibrated workflow is a pragmatic visual-relief mode inspired by the unknown-lighting literature, especially Basri, Jacobs, and Kemelmacher, but it is intentionally simpler and aimed at useful enhancement rather than metrically unique shape recovery.

The program estimates image-space normals and relative brightness terms from a fixed-view image stack. It does not by itself calibrate physical height, pixel pitch, lens distortion, bidirectional reflectance, or microscope illumination nonuniformity.

Input images are converted to single-channel floating-point linear luminance. Integer samples are first expressed relative to their container range. When sRGB input is selected, each B, G, and R channel is decoded with the IEC 61966-2-1 piecewise sRGB transfer function before the linear-light RGB coefficients associated with the BT.709 primaries are applied. One common multiplicative scale then maps the brightest finite sample across the complete normal-solve stack to 1.0. Because the same scale is used for every image, inter-image intensity ratios are preserved and normals are unchanged by the choice of integer container; albedo remains relative. RTI color loading uses the same decoding and also preserves one scale across the stack, but does not brighten already normalized source values merely to fill the range. This normalization does not replace dark-frame correction, flat-field correction, matched exposure, a camera profile, or relative LED-output calibration.

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

A full `lights.csv` binds each calibration row to the selected image by normalized path, with a unique filename fallback so a complete dataset can be moved to a new folder. Rows are reordered to the selected-image order, and missing or ambiguous names are rejected. The compact `light_vectors.csv` has no image identities and therefore remains explicitly positional: row `i` applies to selected image `i`.

## Photometric Stereo

For each object pixel, the calibrated solve uses a Lambertian model:

```text
I_i = dot(L_i, g)
g   = rho * n
```

`L_i` is the light direction for image `i`, `rho` is a relative albedo or brightness scale, and `n` is the unit normal. The standard solver estimates `g` by least squares. Observations below `--shadow-threshold` are omitted so that shadowed images do not dominate the fit. A pixel must have at least 3 usable observations, the fitted normal must face the camera, and the resulting residual is stored as a root-mean-square intensity error.

The default calibrated solver is a modest robust variant. It estimates a per-pixel high-outlier cutoff from the median and median absolute deviation of the usable observations, caps that cutoff with `--high-outlier-threshold`, omits very bright observations when enough other observations remain, then performs up to eight iteratively reweighted least-squares passes with Huber-style residual weights. This is not the sparse-regression method of Ikehata et al. or the low-rank recovery method of Wu et al. It follows the same practical motivation but assumes corruptions are sparse among the observations at each pixel. Robust rejection requires at least four usable observations so that one observation can be omitted while retaining a solvable three-light system. A three-image solve has no such redundancy and therefore falls back to direct least squares. Broad specular lobes, interreflection, registration error, or exposure drift across several images can remain coherent enough to bias both ordinary and robust solutions.

The robust solver also writes diagnostic maps for mean robust weight, shadow count, high-intensity outlier count, and, if requested, an experimental specular-cue mask. The shadow count includes both observations below the absolute shadow threshold and darker-than-model residuals that were strongly downweighted. Brighter-than-model residuals contribute to the highlight count and specular cue instead, so penumbrae are not intentionally labeled as shiny. In addition, the specular mask compares the diffuse normal with an intensity-weighted estimate of the illumination-view half vectors. A sufficiently concentrated, asymmetric response that disagrees with the diffuse fit is marked as potentially glossy. This is motivated by Torrance-Sparrow microfacet reflection and Blinn's half-vector formulation, but it is only a diagnostic heuristic: it neither estimates a BRDF nor replaces or repairs the output normal. The broad-gloss regression deliberately requires detection while recording that normal error remains high.

Before solving, the program measures the spectral condition number of the normalized light-direction matrix. Rank-deficient geometry or a condition number above 100 is rejected because small radiometric errors would otherwise be strongly amplified into the recovered normal. The value 100 is a conservative software guard, not a universal scientific boundary; lower condition numbers are preferable and do not prove calibration accuracy. After solving, the fraction of masked pixels with valid normals is reported. A zero or negligible solution is rejected, and coverage below 10% produces a warning so empty masks, deep shadows, exposure errors, and mismatched calibration are visible rather than silently producing plausible-looking empty outputs.

## Experimental Neural Fusion

When the experimental neural-fusion option is enabled, the program still performs the full classical calibrated solve first. It then runs a bundled ONNX export of the pretrained PS-FCN model that matches the number of input images from 3 through 25.

The neural model expects one directional light vector per image, stacks the grayscale image channels as repeated RGB inputs, normalizes the stacked intensities by the per-pixel energy across the image set, and predicts a dense normal map. A network prediction is eligible for fusion only where the solve mask contains the pixel, at least three finite image observations exceed the selected shadow threshold, and the predicted normal is finite and nonzero. The final diffuse refit independently requires three usable observations before marking the fused output valid. Because PS-FCN was trained for directional-light photometric stereo rather than microscope-specific near-field metrology, What A Relief treats it as a qualitative normal prior rather than as a replacement physical model. Its two dense inputs each contain `3 * image_count * padded_height * padded_width` floats; preprocessing uses one image-sized scratch buffer and reports the input-tensor allocation before inference.

Fusion is done in slope space rather than by directly averaging RGB normals:

1. Convert the classical and neural normals into slope fields `p = -nx / nz` and `q = ny / nz`.
2. Build low-frequency slope fields for each source with masked Gaussian smoothing.
3. Blend the low-frequency fields with a confidence term derived from the classical residual, robust weight, shadow count, and highlight-outlier count.
4. Reinject a reduced amount of high-frequency classical slope detail.
5. Clamp the fused slope magnitude against the classical slope distribution.
6. Convert the fused slopes back into a normal map.

The fused normals are used for the visualization-oriented normal outputs. Where the neural evidence mask is false but the classical solve is valid, the classical slope is preserved rather than blended with an invented flat neural normal. Near-field diffuse refitting uses the same center-normalized inverse-square effective light vectors as the classical solve. To avoid exaggerated geometry, height preview and PLY mesh generation remain on the classical geometry path even when neural fusion is enabled. Fusion runs write three explicit normal sets: classical, neural, and fused, including the RGB normal image, X/Y/Z component images, an upper-left hillshade image, and the neural evidence mask.

## Near-Field Ring Lighting

The standard calibrated solve uses one global light direction per image. This is appropriate when illumination is approximately directional at the sample scale. For close microscope or ring-light geometries, `--near-field-ring radius height` can instead treat each image as a point light on a ring. The sphere-derived or file-supplied light vector provides the azimuth of the light, while the user-provided radius and height define the point-light position in millimeters. Image pixel locations are converted into millimeters using `--pixel-scale-mm`, by reading common TIFF physical scale tags when possible, or in the GUI by drawing a scale line of known length on the first image; the light positions themselves are not stored as image-pixel dimensions.

For each solved pixel, the effective light vector is recomputed from the surface pixel to the corresponding point light. It includes both spatial direction and inverse-square irradiance, normalized to 1 at the image center so existing center light calibration retains its scale:

```text
P = ((x - cx) * pixel_scale_mm, (cy - y) * pixel_scale_mm, 0)
S = (radius_mm * cos(theta), radius_mm * sin(theta), height_mm)
d2 = dot(S - P, S - P)
d0_squared = radius_mm^2 + height_mm^2
L_effective = (d0_squared / d2) * normalize(S - P)
```

This is the isotropic point-source specialization of the nearby-LED model described by Queau et al. (2018): direction and inverse-square distance falloff are modeled, while LED angular anisotropy is set to zero. It assumes one active ring segment or source per image, equal source output, an orthographic camera, and a reference surface at `z = 0`. It can reduce systematic errors when the light is close, but it does not model lens perspective, true 3D sample height, LED size, LED optical-axis falloff, source-to-source intensity differences, or spatial flat-field effects. Those terms require additional camera and planar-target radiometric calibration and should not be inferred from ring radius and height alone. TIFF metadata support follows TIFF 6.0 resolution tags with declared inch, centimeter, or recognized description units. A GeoTIFF `ModelPixelScaleTag` is accepted only when `ProjLinearUnitsGeoKey` declares a supported linear EPSG unit, a user-defined linear-unit size is supplied, or the image description declares a recognized unit. This follows OGC GeoTIFF 1.1, in which model pixel scale is expressed in model-space units rather than inherently in millimeters. Unitless, unsupported, and angular model scales are ignored. BigTIFF metadata are not parsed by this helper. When metadata are absent or ambiguous, enter the pixel scale manually or draw a scale line from a known ruler or scale bar in the image.

This direct normal solve supports 3 to 25 images and does not derive normals from the optional RTI polynomial. RTI export is a separate appearance product.

## Relief Flattening

Optional relief flattening removes a low-frequency slope trend from the normal field before writing the normal visualizations, relit image, optional height preview, and optional PLY mesh. The program converts normals to slopes, estimates a broad Gaussian low-pass version of those slopes inside the valid mask, subtracts a fraction of that low-frequency component, and converts the remaining slopes back to normals.

The goal is visual: broad sample tilt or curvature can be reduced so smaller topographic features stand out without sharpening pixel-scale noise. This is best understood as scale separation, related to Gaussian scale-space filtering and surface-texture filtering practice, not as calibrated form removal for metrology. Leave this option off when the large-scale shape itself is scientifically important.

Height drift correction is separate. It is applied only after height integration and only affects `height.png`, `height.pfm`, and PLY export. `None` leaves the integrated field untouched. Plane leveling subtracts a least-squares affine surface (`1, x, y`). Radial mode jointly fits that plane and one centered dome term. Quadratic mode jointly fits a full second-order surface (`1, x, y, x^2, y^2, xy`) over the height mask. These options are intended for broad integration drift, curl, or visual leveling, not calibrated form removal.

## Uncalibrated Unknown Lighting

Uncalibrated mode skips the sphere and estimates a relative normal field from the image stack. It needs at least 4 images. The implementation builds an image covariance over valid pixels, tries a rank-4 first-order unknown-lighting factorization, then falls back to a rank-3 PCA relief estimate when the metric constraint is degenerate.

Because unknown-lighting photometric stereo has an unavoidable ambiguity, uncalibrated mode is intended for visual relief enhancement rather than calibrated geometry. The program orients normals toward the camera, searches a rotation about the view axis that reduces a curl-like integrability cost, and stabilizes extreme slopes. No height drift correction is applied when height flattening is `None`; plane, radial, or quadratic correction must be selected explicitly.

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

The result is useful for visual inspection, while `normal_rgb.png`, the component maps, and `residual.png` remain the primary outputs. With height flattening set to `None`, no fitted surface is subtracted after integration, so broad ramps and specimen tilt remain in the numeric height field.

If height-only form removal is enabled, the program subtracts one jointly fitted plane, radial-plus-plane, or quadratic trend over the height mask. This follows the same practical idea as RelightLab's optional radial/quadratic flattening controls: reduce broad systematic bias when the specimen is expected to be roughly flat on average, while leaving the primary normal and reflectance products untouched.

Custom height masks are intersected with valid geometry and eroded with a 5 x 5 elliptical kernel before integration, provided at least 100 pixels remain. This excludes up to two image pixels of uncertain boundary slopes; the final domain is exported as `height_mask.png`.

If requested, the height preview can also be exported as a binary little-endian PLY mesh. The standard mesh is an open surface for inspection in image-pixel units. The printable mesh option creates a second PLY as a watertight solid: it writes the sampled top surface, duplicates the vertices onto a flat bottom plane, writes reversed bottom faces, and closes every boundary edge with a side face. Printable export requires pixel scale and writes coordinates in millimeters. Its Z scale is the integrated height multiplied by `pixel_scale_mm * height_scale`; `height.pfm` itself remains in image-pixel height units. This unit conversion alone does not establish metric accuracy.

## RTI Export

Optional RTI export fits a PTM-style polynomial appearance model from the original color image stack and the calibrated or loaded light directions. Color samples use the same selected exact sRGB decoding as the normal solve, and no image is normalized independently. Small 3-to-8-image stacks use the stable first-order basis:

```text
I(u,v) = a0 + a1*u + a2*v
```

Larger, better-conditioned stacks use the quadratic basis when the light geometry is numerically stable:

```text
I(u,v) = a0 + a1*u + a2*v + a3*u^2 + a4*u*v + a5*v^2
```

The three-term form is a first-order subset and is less expressive than the six-term quadratic PTM introduced by Malzbender et al.; it is used to avoid fitting unconstrained coefficients from very small stacks. Coefficients are estimated by ordinary least squares. RTI fitting does not use the normal solver's per-pixel robust shadow/highlight rejection, so cast shadows and saturation remain appearance-model limitations.

The coefficient planes can be stored in three layouts. The plain image and Deep Zoom layouts use Relight/OpenLIME-style `info.json` plus coefficient JPEGs. RGB mode stores one RGB JPEG per PTM coefficient. LRGB mode stores an unscaled base image in `plane_0` and stores relative luminance coefficients in the remaining JPEGs; this follows the PTM LRGB convention used by Relight/OpenLIME, where the viewer multiplies the base image by the relit luminance. The plain image layout writes full-size coefficient images, and the Deep Zoom layout writes a standard level-zero-to-full-resolution DZI pyramid per stored plane image. JPEG quantization makes every RTI layout an appearance product rather than a lossless scientific archive.

The webRTIViewer layout writes the older `info.xml` plus quadtree component JPEGs expected by `createRtiViewer(...)` in jcupitt/webRTIViewer. For that layout, coefficients are reordered into the shader's expected PTM order:

```text
u^2, v^2, uv, u, v, 1
```

The RGB webRTIViewer export writes six component layers, filling unconstrained high-order terms with zero for first-order exports. The LRGB webRTIViewer export writes three component layers: high-order luminance coefficients, low-order luminance coefficients, and the RGB base image.

## Output Integrity

At run start, the app writes `run_manifest.json` with status `in_progress` and removes known stale products from the selected output folder. Each image or data file is written to a temporary sibling, checked, and promoted to its final name. RTI export is assembled in a separate staging directory; a previous complete package is replaced only after the new descriptor, planes or tiles, and `rti_manifest.json` have all been written. A run manifest changes to `complete` only after every requested artifact exists and is nonempty.

The complete manifest records application version, UTC times, input paths, input sizes and filesystem timestamps, processing parameters, calibrated light vectors, lighting condition number, solved fraction, and output paths and sizes. It is meant to expose stale or partial runs and support practical reproducibility. It does not hash input bytes, preserve camera metadata independently, or provide cryptographic provenance.

## Related Work

- Woodham, R. J. "Photometric Method for Determining Surface Orientation from Multiple Images." Optical Engineering 19(1), 139-144, 1980. https://doi.org/10.1117/12.7972479
- Basri, R., Jacobs, D. W., and Kemelmacher, I. "Photometric Stereo with General, Unknown Lighting." International Journal of Computer Vision 72(3), 239-257, 2007. https://doi.org/10.1007/s11263-006-8815-7
- Huber, P. J. "Robust Estimation of a Location Parameter." The Annals of Mathematical Statistics 35(1), 73-101, 1964. https://doi.org/10.1214/aoms/1177703732
- Torrance, K. E., and Sparrow, E. M. "Theory for Off-Specular Reflection From Roughened Surfaces." Journal of the Optical Society of America 57(9), 1105-1114, 1967. https://doi.org/10.1364/JOSA.57.001105
- Blinn, J. F. "Models of Light Reflection for Computer Synthesized Pictures." SIGGRAPH 1977, 192-198. https://doi.org/10.1145/563858.563893
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
- IEC 61966-2-1, "Multimedia Systems and Equipment - Colour Measurement and Management - Part 2-1: Default RGB Colour Space - sRGB."
- ITU-R BT.709-6, "Parameter Values for the HDTV Standards for Production and International Programme Exchange," 2015.
- TIFF Revision 6.0, Aldus Corporation, 1992.
- Open Geospatial Consortium, "OGC GeoTIFF Standard 1.1," OGC 19-008r4, 2019. https://docs.ogc.org/is/19-008r4/19-008r4.html

The program also depends on OpenCV for image operations and windowing, and uses vcpkg/CMake to build OpenCV reproducibly on Windows. The optional experimental neural-fusion models are exported from the PS-FCN project and are bundled as third-party MIT-licensed assets.

Relight and RelightLab from the CNR-ISTI Visual Computing Lab are acknowledged as important related RTI software. Comparing What A Relief against Relight helped identify useful workflow ideas, such as explicit robust-normal and flattening controls. What A Relief does not include Relight source code.

webRTIViewer and webGLRTIMaker are acknowledged for the `info.xml` plus component-tile RTI layout supported by the `webrti` export mode. What A Relief writes compatible folders but does not include webRTIViewer source code.
