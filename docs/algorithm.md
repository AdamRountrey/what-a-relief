# Algorithm

## Interactive Sphere Calibration

The user marks the highlight sphere once on the first image by dragging from the sphere center to the sphere edge. The same circle is then used in every image.

Inside that circle, each image is searched for its specular highlight:

1. Convert the image to grayscale linear intensity.
2. Keep pixels inside the selected sphere.
3. Find a high percentile brightness threshold.
4. Compute a weighted centroid of the pixels above that threshold.
5. Convert the centroid into a normal on the sphere.
6. Reflect the camera view vector around that normal to recover the light direction.

For a mirror sphere:

```text
L = 2 * dot(N, V) * N - V
```

## Photometric Stereo

For each object pixel, the intensity vector is modeled as:

```text
I = L g
g = rho * n
```

`L` is the matrix of light directions, `rho` is diffuse albedo, and `n` is the unit normal. The program solves `g` by least squares. Observations below `--shadow-threshold` are omitted so that shadowed images do not dominate the fit.

This direct solve is designed for 4 or 8 images. It does not use spherical harmonics, because those RTI-style methods need many more lighting samples.

## Height Preview

The normal field is converted to gradients:

```text
dz/dx   = -nx / nz
dz/drow =  ny / nz
```

The program then performs an iterative masked integration. The result is useful for visual inspection, while `normal_rgb.png`, the component maps, and `residual.png` remain the primary outputs.
