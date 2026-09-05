#!/usr/bin/env python3
"""Generate the committed robust-photometric-stereo reference fixtures.

Mitsuba is intentionally an offline test-asset dependency. The C++ test suite
loads the generated PNGs and does not import or invoke Mitsuba.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import mitsuba as mi
import numpy as np


WIDTH = 96
HEIGHT = 72
LIGHT_Z = 0.52
LIGHT_ANGLE_OFFSET = math.pi / 8.0
LIGHT_IRRADIANCE = math.pi
FULL_WELL_ELECTRONS = 18_000.0
READ_NOISE_ELECTRONS = 3.0
DARK_SIGNAL_FRACTION = 0.003
ADC_LEVELS = 4095
RNG_SEED = 0x5EED_2026
POSITION_Z_MIN = -0.25
POSITION_Z_MAX = 1.0
SCENE_NAMES = (
    "robust_v1",
    "textured_primitives_v1",
    "holdout_relief_v1",
)


def gray(value: float) -> dict:
    return {"type": "rgb", "value": [value, value, value]}


def material(diffuse: float | dict, roughness: float | None, mode: str) -> dict:
    reflectance = gray(diffuse) if isinstance(diffuse, (int, float)) else diffuse
    if roughness is None or mode == "lambertian":
        return {"type": "diffuse", "reflectance": reflectance}

    result = {
        "type": "roughplastic",
        "distribution": "ggx",
        "alpha": roughness,
        "int_ior": 1.57,
        "ext_ior": "air",
        "diffuse_reflectance": reflectance,
        "nonlinear": False,
    }
    if mode == "no_specular":
        # This nonphysical diagnostic render isolates the specular term while
        # preserving the rough-plastic diffuse transport and visibility.
        result["specular_reflectance"] = gray(0.0)
    return result


def direction(azimuth_degrees: float, z: float) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_degrees)
    radial = math.sqrt(1.0 - z * z)
    return radial * math.cos(azimuth), radial * math.sin(azimuth), z


def light_directions(scene_name: str) -> list[tuple[float, float, float]]:
    if scene_name == "robust_v1":
        radial = math.sqrt(1.0 - LIGHT_Z * LIGHT_Z)
        return [
            (
                radial * math.cos(LIGHT_ANGLE_OFFSET + 2.0 * math.pi * i / 8),
                radial * math.sin(LIGHT_ANGLE_OFFSET + 2.0 * math.pi * i / 8),
                LIGHT_Z,
            )
            for i in range(8)
        ]
    if scene_name == "textured_primitives_v1":
        azimuths = (5, 31, 63, 94, 128, 163, 197, 229, 260, 292, 324, 349)
        elevations = (0.34, 0.48, 0.63, 0.39, 0.72, 0.52, 0.29, 0.57, 0.44, 0.68, 0.36, 0.60)
        return [direction(azimuth, z) for azimuth, z in zip(azimuths, elevations)]
    if scene_name == "holdout_relief_v1":
        azimuths = (13, 47, 82, 121, 166, 207, 251, 294, 329, 353)
        radial = 3.6
        height = 1.25
        z = height / math.sqrt(radial * radial + height * height)
        return [direction(azimuth, z) for azimuth in azimuths]
    raise ValueError(f"Unknown fixture scene: {scene_name}")


def split_name(scene_name: str) -> str:
    return "validation" if scene_name.startswith("holdout_") else "development"


def exposure(scene_name: str) -> float:
    return {
        "robust_v1": 2.35,
        "textured_primitives_v1": 2.10,
        "holdout_relief_v1": 2.20,
    }[scene_name]


def point_light_geometry(scene_name: str) -> tuple[float, float] | None:
    if scene_name == "holdout_relief_v1":
        return 3.6, 1.25
    return None


def checker(low: float, high: float, u_scale: float, v_scale: float) -> dict:
    transform = mi.ScalarTransform4f
    return {
        "type": "checkerboard",
        "color0": gray(low),
        "color1": gray(high),
        "to_uv": transform().scale([u_scale, v_scale, 1.0]),
    }


def bitmap_texture(path: Path, u_scale: float, v_scale: float) -> dict:
    transform = mi.ScalarTransform4f
    return {
        "type": "bitmap",
        "filename": str(path),
        "raw": True,
        "filter_type": "bilinear",
        "wrap_mode": "repeat",
        "to_uv": transform().scale([u_scale, v_scale, 1.0]),
    }


def sensor(scene_name: str, width: int = WIDTH, height: int = HEIGHT) -> dict:
    transform = mi.ScalarTransform4f
    horizontal_scale = 1.35 if scene_name == "robust_v1" else WIDTH / HEIGHT
    return {
        "type": "orthographic",
        "to_world": transform().look_at(
            origin=[0.0, 0.0, 5.0],
            target=[0.0, 0.0, 0.0],
            up=[0.0, 1.0, 0.0],
        ) @ transform().scale([horizontal_scale, 1.0, 1.0]),
        "near_clip": 0.1,
        "far_clip": 10.0,
        "sampler": {"type": "independent", "sample_count": 64},
        "film": {
            "type": "hdrfilm",
            "width": width,
            "height": height,
            "pixel_format": "rgb",
            "component_format": "float32",
            "rfilter": {"type": "box"},
        },
    }


def shape(
    center: tuple[float, float, float],
    radius: float,
    diffuse: float,
    roughness: float | None,
    mode: str,
) -> dict:
    return {
        "type": "sphere",
        "center": list(center),
        "radius": radius,
        "bsdf": material(diffuse, roughness, mode),
    }


def scene_shapes(scene_name: str, mode: str, asset_root: Path) -> dict[str, dict]:
    transform = mi.ScalarTransform4f
    if scene_name == "robust_v1":
        return {
            "floor": {
                "type": "rectangle",
                "to_world": transform().scale([1.35, 1.0, 1.0]),
                "bsdf": material(0.34, None, mode),
            },
            "matte_mound": shape((-0.42, 0.12, 0.21), 0.39, 0.53, None, mode),
            "black_gloss": shape((0.34, -0.12, 0.17), 0.32, 0.035, 0.095, mode),
            "rough_gloss": shape((0.07, 0.50, 0.105), 0.20, 0.22, 0.27, mode),
        }
    if scene_name == "textured_primitives_v1":
        return {
            "floor": {
                "type": "rectangle",
                "to_world": transform().scale([WIDTH / HEIGHT, 1.0, 1.0]),
                "bsdf": material(checker(0.16, 0.48, 7.0, 5.0), None, mode),
            },
            "ripple_patch": {
                "type": "ply",
                "filename": str(asset_root / "relief_mesh.ply"),
                "face_normals": False,
                "bsdf": material(0.46, None, mode),
            },
            "striped_rod": {
                "type": "cylinder",
                "p0": [-0.18, -0.58, 0.11],
                "p1": [0.82, 0.02, 0.11],
                "radius": 0.14,
                "bsdf": material(checker(0.12, 0.55, 10.0, 3.0), 0.22, mode),
            },
            "dark_gloss_block": {
                "type": "cube",
                "to_world": transform().translate([0.52, 0.48, 0.15])
                @ transform().rotate([0.0, 0.0, 1.0], -21.0)
                @ transform().rotate([1.0, 0.0, 0.0], -11.0)
                @ transform().rotate([0.0, 1.0, 0.0], 16.0)
                @ transform().scale([0.27, 0.19, 0.14]),
                "bsdf": material(0.055, 0.075, mode),
            },
            "textured_disk": {
                "type": "disk",
                "to_world": transform().translate([-0.58, 0.57, 0.055])
                @ transform().rotate([1.0, 0.0, 0.0], 7.0)
                @ transform().scale([0.24, 0.24, 0.24]),
                "bsdf": material(checker(0.20, 0.66, 5.0, 5.0), None, mode),
            },
        }
    if scene_name == "holdout_relief_v1":
        texture_path = asset_root / "albedo_texture.png"
        return {
            "floor": {
                "type": "rectangle",
                "to_world": transform().scale([WIDTH / HEIGHT, 1.0, 1.0]),
                "bsdf": material(bitmap_texture(texture_path, 2.0, 2.0), None, mode),
            },
            "corrugated_relief": {
                "type": "ply",
                "filename": str(asset_root / "relief_mesh.ply"),
                "face_normals": False,
                "bsdf": material(0.49, 0.34, mode),
            },
            "dark_rod": {
                "type": "cylinder",
                "p0": [-0.96, 0.37, 0.11],
                "p1": [0.16, 0.67, 0.12],
                "radius": 0.15,
                "bsdf": material(0.028, 0.12, mode),
            },
            "rough_block": {
                "type": "cube",
                "to_world": transform().translate([0.69, 0.34, 0.105])
                @ transform().rotate([0.0, 0.0, 1.0], 31.0)
                @ transform().rotate([1.0, 0.0, 0.0], 9.0)
                @ transform().rotate([0.0, 1.0, 0.0], -13.0)
                @ transform().scale([0.21, 0.26, 0.085]),
                "bsdf": material(0.20, 0.30, mode),
            },
            "textured_medallion": {
                "type": "disk",
                "to_world": transform().translate([0.78, -0.54, 0.045])
                @ transform().rotate([1.0, 0.0, 0.0], -6.0)
                @ transform().scale([0.28, 0.28, 0.28]),
                "bsdf": material(bitmap_texture(texture_path, 3.0, 3.0), 0.18, mode),
            },
        }
    raise ValueError(f"Unknown fixture scene: {scene_name}")


def point_light_position(
    scene_name: str,
    light: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    geometry = point_light_geometry(scene_name)
    if geometry is None:
        return None
    radius, height = geometry
    radial = math.hypot(light[0], light[1])
    return radius * light[0] / radial, radius * light[1] / radial, height


def scene_description(
    scene_name: str,
    mode: str,
    light: tuple[float, float, float] | None,
    asset_root: Path,
    aov: bool = False,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> dict:
    integrator = (
        {
            "type": "aov",
            "aovs": "normal:sh_normal,albedo:albedo,position:position,shape:shape_index",
        }
        if aov
        else {"type": "direct", "hide_emitters": True}
    )
    scene = {
        "type": "scene",
        "integrator": integrator,
        "sensor": sensor(scene_name, width, height),
    }
    scene.update(scene_shapes(scene_name, mode, asset_root))
    if light is not None:
        source = point_light_position(scene_name, light)
        if source is None:
            # Mitsuba's directional vector is the direction in which light travels;
            # photometric-stereo vectors point from the surface toward the light.
            scene["key_light"] = {
                "type": "directional",
                "direction": [-light[0], -light[1], -light[2]],
                "irradiance": gray(LIGHT_IRRADIANCE),
            }
        else:
            radius, height = point_light_geometry(scene_name) or (0.0, 0.0)
            scene["key_light"] = {
                "type": "point",
                "position": list(source),
                "intensity": gray(math.pi * (radius * radius + height * height)),
            }
    return scene


def radiance_sample_grid(spp: int) -> tuple[int, int]:
    scale = max(1, int(math.ceil(math.sqrt(spp))))
    return scale, scale


def render(
    scene_name: str,
    mode: str,
    light: tuple[float, float, float] | None,
    asset_root: Path,
    spp: int,
    seed: int,
    aov: bool = False,
) -> np.ndarray:
    if aov:
        scene = mi.load_dict(scene_description(scene_name, mode, light, asset_root, True))
        return np.asarray(mi.render(scene, spp=spp, seed=seed), dtype=np.float32)

    horizontal_samples, vertical_samples = radiance_sample_grid(spp)
    scene = mi.load_dict(scene_description(
        scene_name,
        mode,
        light,
        asset_root,
        False,
        WIDTH * horizontal_samples,
        HEIGHT * vertical_samples,
    ))
    high_resolution = np.asarray(mi.render(scene, spp=1, seed=seed), dtype=np.float32)
    channels = high_resolution.shape[2]
    reshaped = high_resolution.reshape(
        HEIGHT,
        vertical_samples,
        WIDTH,
        horizontal_samples,
        channels,
    )
    return reshaped.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def erode_same_shape(valid: np.ndarray, shape_index: np.ndarray, iterations: int = 2) -> np.ndarray:
    rounded = np.rint(shape_index).astype(np.int32)
    result = valid.copy()
    for _ in range(iterations):
        previous = result.copy()
        result[:] = False
        center_valid = previous[1:-1, 1:-1]
        center_shape = rounded[1:-1, 1:-1]
        neighborhood_valid = center_valid.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighborhood_valid &= previous[
                    1 + dy : HEIGHT - 1 + dy,
                    1 + dx : WIDTH - 1 + dx,
                ]
                neighborhood_valid &= (
                    rounded[
                        1 + dy : HEIGHT - 1 + dy,
                        1 + dx : WIDTH - 1 + dx,
                    ]
                    == center_shape
                )
        result[1:-1, 1:-1] = neighborhood_valid
    return result


def write_gray(path: Path, values: np.ndarray, component_format: mi.Struct.Type) -> None:
    array = np.ascontiguousarray(values[..., np.newaxis], dtype=np.float32)
    bitmap = mi.Bitmap(array).convert(mi.Bitmap.PixelFormat.Y, component_format, False)
    bitmap.write(str(path))


def write_mask(path: Path, values: np.ndarray) -> None:
    write_gray(path, values.astype(np.float32), mi.Struct.Type.UInt8)


def write_u16(path: Path, values: np.ndarray) -> None:
    write_gray(path, np.clip(values, 0.0, 1.0), mi.Struct.Type.UInt16)


def write_reference_normal(path: Path, values: np.ndarray) -> None:
    # Twelve effective bits are far finer than the angular acceptance bounds
    # and absorb sub-ULP backend reduction differences in multisample AOVs.
    stable = np.rint(np.clip(values, 0.0, 1.0) * ADC_LEVELS) / ADC_LEVELS
    write_u16(path, stable)


def write_reference_position(path: Path, values: np.ndarray, valid: np.ndarray) -> None:
    encoded = (values - POSITION_Z_MIN) / (POSITION_Z_MAX - POSITION_Z_MIN)
    encoded = np.where(valid, encoded, 0.0)
    stable = np.rint(np.clip(encoded, 0.0, 1.0) * ADC_LEVELS) / ADC_LEVELS
    write_u16(path, stable)


def write_u8_codes(path: Path, values: np.ndarray) -> None:
    write_gray(path, np.clip(values, 0.0, 255.0) / 255.0, mi.Struct.Type.UInt8)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relief_height_and_gradient(
    scene_name: str,
    x: float,
    y: float,
) -> tuple[float, float, float]:
    if scene_name == "textured_primitives_v1":
        dx = x + 0.66
        dy = y + 0.24
        mound = math.exp(-4.0 * (dx * dx + dy * dy))
        wave_x = 11.0 * x + 0.4
        wave_y = 8.0 * y
        z = 0.038 + 0.11 * mound + 0.018 * math.sin(wave_x) * math.cos(wave_y)
        dz_dx = -0.88 * dx * mound + 0.198 * math.cos(wave_x) * math.cos(wave_y)
        dz_dy = -0.88 * dy * mound - 0.144 * math.sin(wave_x) * math.sin(wave_y)
        return z, dz_dx, dz_dy
    if scene_name == "holdout_relief_v1":
        dx = x + 0.28
        dy = y + 0.31
        mound = math.exp(-3.2 * (dx * dx + 1.4 * dy * dy))
        ridge_x = 5.3 * x + 0.25
        ridge_y = 4.1 * y - 0.35
        fine = 12.0 * x + 3.0 * y
        z = (
            0.055
            + 0.045 * mound
            + 0.024 * math.sin(ridge_x) * math.sin(ridge_y)
            + 0.009 * math.cos(fine)
        )
        dz_dx = (
            -0.288 * dx * mound
            + 0.1272 * math.cos(ridge_x) * math.sin(ridge_y)
            - 0.108 * math.sin(fine)
        )
        dz_dy = (
            -0.4032 * dy * mound
            + 0.0984 * math.sin(ridge_x) * math.cos(ridge_y)
            - 0.027 * math.sin(fine)
        )
        return z, dz_dx, dz_dy
    raise ValueError(f"Scene {scene_name} does not use a relief mesh")


def write_relief_mesh(path: Path, scene_name: str) -> None:
    if scene_name == "textured_primitives_v1":
        x_min, x_max, y_min, y_max = -1.14, -0.19, -0.72, 0.23
    elif scene_name == "holdout_relief_v1":
        x_min, x_max, y_min, y_max = -1.08, 0.42, -0.76, 0.20
    else:
        raise ValueError(f"Scene {scene_name} does not use a relief mesh")

    columns = 41
    rows = 31
    vertices: list[tuple[float, float, float, float, float, float]] = []
    for row in range(rows):
        y = y_min + (y_max - y_min) * row / (rows - 1)
        for column in range(columns):
            x = x_min + (x_max - x_min) * column / (columns - 1)
            z, dz_dx, dz_dy = relief_height_and_gradient(scene_name, x, y)
            length = math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0)
            vertices.append((x, y, z, -dz_dx / length, -dz_dy / length, 1.0 / length))

    faces: list[tuple[int, int, int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            b = a + 1
            c = a + columns
            d = c + 1
            faces.append((a, b, c))
            faces.append((b, d, c))

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment generated for {scene_name}\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        for vertex in vertices:
            stream.write(struct.pack("<6f", *vertex))
        for face in faces:
            stream.write(struct.pack("<Biii", 3, *face))


def write_albedo_texture(path: Path, scene_name: str) -> None:
    texture_size = 64
    yy, xx = np.mgrid[0:texture_size, 0:texture_size].astype(np.float64)
    u = (xx + 0.5) / texture_size
    v = (yy + 0.5) / texture_size
    if scene_name == "holdout_relief_v1":
        values = (
            0.31
            + 0.12 * np.sin(2.0 * math.pi * (1.3 * u + 0.2 * v))
            + 0.08 * np.cos(2.0 * math.pi * (0.4 * u - 2.1 * v))
            + 0.045 * np.sin(2.0 * math.pi * (4.7 * u + 3.2 * v))
        )
    else:
        raise ValueError(f"Scene {scene_name} does not use a bitmap texture")
    write_u16(path, np.clip(values, 0.07, 0.62).astype(np.float32))


def prepare_scene_assets(output: Path, scene_name: str) -> list[str]:
    assets: list[str] = []
    if scene_name in ("textured_primitives_v1", "holdout_relief_v1"):
        write_relief_mesh(output / "relief_mesh.ply", scene_name)
        assets.append("relief_mesh.ply")
    if scene_name == "holdout_relief_v1":
        write_albedo_texture(output / "albedo_texture.png", scene_name)
        assets.append("albedo_texture.png")
    return assets


def generate(scene_name: str, output: Path, spp: int, variant: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        "images",
        "shadow_truth",
        "attached_shadow_truth",
        "cast_shadow_truth",
        "highlight_truth",
        "saturation_truth",
    ):
        (output / directory).mkdir(exist_ok=True)

    scene_assets = prepare_scene_assets(output, scene_name)
    lights = light_directions(scene_name)
    aov = render(scene_name, "lambertian", None, output, spp, RNG_SEED, aov=True)
    normals = aov[..., 0:3]
    albedo = luminance(aov[..., 3:6])
    positions = aov[..., 6:9]
    shape_index = aov[..., 9]
    normal_length = np.linalg.norm(normals, axis=2)
    normals = normals / np.maximum(normal_length[..., np.newaxis], 1.0e-8)
    valid = (
        (normal_length > 0.985)
        & (normals[..., 2] > 0.08)
        & (np.abs(shape_index - np.rint(shape_index)) < 0.02)
    )
    valid = erode_same_shape(valid, shape_index)

    write_mask(output / "solve_mask.png", valid)
    write_u8_codes(output / "shape_index.png", np.rint(shape_index))
    write_reference_position(output / "position_z.png", positions[..., 2], valid)
    for channel, name in enumerate(("x", "y", "z")):
        write_reference_normal(
            output / f"normal_{name}.png",
            normals[..., channel] * 0.5 + 0.5,
        )

    rng = np.random.default_rng(RNG_SEED)
    input_files: list[str] = []
    mask_files: dict[str, list[str]] = {
        "shadow": [],
        "attached_shadow": [],
        "cast_shadow": [],
        "highlight": [],
        "saturation": [],
    }
    aggregate = {
        "valid_pixels": int(np.count_nonzero(valid)),
        "shadow_observations": 0,
        "attached_shadow_observations": 0,
        "cast_shadow_observations": 0,
        "highlight_observations": 0,
        "saturated_observations": 0,
    }

    for index, light in enumerate(lights):
        full_seed = RNG_SEED + 101 * index + 1
        no_specular_seed = RNG_SEED + 101 * index + 2
        lambertian_seed = RNG_SEED + 101 * index + 3
        full = luminance(
            render(scene_name, "full", light, output, spp, full_seed)
        )
        no_specular = luminance(
            render(
                scene_name,
                "no_specular",
                light,
                output,
                spp,
                no_specular_seed,
            )
        )
        lambertian = luminance(
            render(
                scene_name,
                "lambertian",
                light,
                output,
                spp,
                lambertian_seed,
            )
        )

        source = point_light_position(scene_name, light)
        if source is None:
            n_dot_l = np.maximum(
                0.0,
                normals[..., 0] * light[0]
                + normals[..., 1] * light[1]
                + normals[..., 2] * light[2],
            )
            irradiance = 1.0
        else:
            displacement = np.asarray(source, dtype=np.float32)[None, None, :] - positions
            distance_squared = np.sum(displacement * displacement, axis=2)
            local_light = displacement / np.sqrt(
                np.maximum(distance_squared[..., None], 1.0e-12)
            )
            n_dot_l = np.maximum(0.0, np.sum(normals * local_light, axis=2))
            radius, height = point_light_geometry(scene_name) or (0.0, 0.0)
            irradiance = (radius * radius + height * height) / np.maximum(
                distance_squared,
                1.0e-12,
            )
        unoccluded_lambertian = albedo * n_dot_l * irradiance
        attached_shadow = n_dot_l <= 0.015
        cast_shadow = (
            (unoccluded_lambertian > 0.025)
            & (lambertian < 0.55 * unoccluded_lambertian)
            & ((unoccluded_lambertian - lambertian) > 0.018)
        )
        attached_shadow = valid & attached_shadow
        cast_shadow = valid & cast_shadow
        shadow = attached_shadow | cast_shadow

        specular_excess = np.maximum(0.0, full - no_specular)
        highlight = (
            valid
            & ~shadow
            & (specular_excess > 0.025)
            & (specular_excess > 0.18 * np.maximum(no_specular, 0.025))
        )

        expected_electrons = (
            np.maximum(full * exposure(scene_name) + DARK_SIGNAL_FRACTION, 0.0)
            * FULL_WELL_ELECTRONS
        )
        measured_electrons = rng.poisson(expected_electrons).astype(np.float64)
        measured_electrons += rng.normal(0.0, READ_NOISE_ELECTRONS, full.shape)
        sensor_value = np.clip(measured_electrons / FULL_WELL_ELECTRONS, 0.0, 1.0)
        adc_code = np.rint(sensor_value * ADC_LEVELS).astype(np.uint16)
        quantized = adc_code.astype(np.float32) / ADC_LEVELS
        saturation = valid & (adc_code == ADC_LEVELS)

        image_name = f"images/light_{index:02d}.png"
        shadow_name = f"shadow_truth/light_{index:02d}.png"
        attached_shadow_name = f"attached_shadow_truth/light_{index:02d}.png"
        cast_shadow_name = f"cast_shadow_truth/light_{index:02d}.png"
        highlight_name = f"highlight_truth/light_{index:02d}.png"
        saturation_name = f"saturation_truth/light_{index:02d}.png"
        write_u16(output / image_name, quantized)
        write_mask(output / shadow_name, shadow)
        write_mask(output / attached_shadow_name, attached_shadow)
        write_mask(output / cast_shadow_name, cast_shadow)
        write_mask(output / highlight_name, highlight)
        write_mask(output / saturation_name, saturation)
        input_files.append(image_name)
        mask_files["shadow"].append(shadow_name)
        mask_files["attached_shadow"].append(attached_shadow_name)
        mask_files["cast_shadow"].append(cast_shadow_name)
        mask_files["highlight"].append(highlight_name)
        mask_files["saturation"].append(saturation_name)
        aggregate["shadow_observations"] += int(np.count_nonzero(shadow))
        aggregate["attached_shadow_observations"] += int(np.count_nonzero(attached_shadow))
        aggregate["cast_shadow_observations"] += int(np.count_nonzero(cast_shadow))
        aggregate["highlight_observations"] += int(np.count_nonzero(highlight))
        aggregate["saturated_observations"] += int(np.count_nonzero(saturation))

    lights_path = output / "lights.csv"
    with lights_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("x,y,z\n")
        for light in lights:
            stream.write(f"{light[0]:.10f},{light[1]:.10f},{light[2]:.10f}\n")

    generated_paths = [output / name for name in input_files]
    for names in mask_files.values():
        generated_paths.extend(output / name for name in names)
    generated_paths.extend(output / name for name in scene_assets)
    generated_paths.extend(
        output / name
        for name in (
            "solve_mask.png",
            "shape_index.png",
            "normal_x.png",
            "normal_y.png",
            "normal_z.png",
            "position_z.png",
            "lights.csv",
        )
    )
    point_geometry = point_light_geometry(scene_name)
    shape_names = list(scene_shapes(scene_name, "lambertian", output).keys())
    horizontal_samples, vertical_samples = radiance_sample_grid(spp)
    manifest = {
        "fixture": f"mitsuba-{scene_name.replace('_', '-')}",
        "split": split_name(scene_name),
        "mitsuba_version": mi.__version__,
        "mitsuba_variant": variant,
        "reference_normal_effective_bits": 12,
        "resolution": [WIDTH, HEIGHT],
        "requested_samples_per_pixel": spp,
        "radiance_samples_per_pixel": horizontal_samples * vertical_samples,
        "aov_samples_per_pixel": spp,
        "radiance_sampling": {
            "method": "single-sample spatial supersampling with deterministic downsampling",
            "horizontal_samples": horizontal_samples,
            "vertical_samples": vertical_samples,
        },
        "lighting_model": "near_field_ring" if point_geometry is not None else "directional",
        "light_directions": lights,
        "light_positions": [
            list(point_light_position(scene_name, light))
            for light in lights
        ] if point_geometry is not None else None,
        "light_irradiance": LIGHT_IRRADIANCE if point_geometry is None else None,
        "ring_radius_mm": point_geometry[0] if point_geometry is not None else None,
        "ring_height_mm": point_geometry[1] if point_geometry is not None else None,
        "pixel_scale_mm": 2.0 / HEIGHT if point_geometry is not None else None,
        "sensor_model": {
            "linear": True,
            "exposure": exposure(scene_name),
            "full_well_electrons": FULL_WELL_ELECTRONS,
            "read_noise_electrons": READ_NOISE_ELECTRONS,
            "dark_signal_fraction": DARK_SIGNAL_FRACTION,
            "adc_bits": 12,
            "random_seed": RNG_SEED,
        },
        "truth_definition": {
            "shadow": "attached shadow or at least 45% visibility loss in a matched Lambertian render",
            "attached_shadow": "valid surface sample with local n dot l no greater than 0.015",
            "cast_shadow": "front-facing valid sample with at least 45% visibility loss and an absolute diffuse deficit above 0.018",
            "highlight": "rough-plastic specular excess above absolute and diffuse-relative thresholds",
            "saturation": "measured 12-bit ADC code equals 4095",
        },
        "position_z_encoding": {
            "units": "scene millimeters",
            "minimum": POSITION_Z_MIN,
            "maximum": POSITION_Z_MAX,
            "container_bits": 16,
            "effective_bits": 12,
            "invalid_pixels": "encoded minimum",
        },
        "files": {
            "images": input_files,
            "shadow_truth": mask_files["shadow"],
            "attached_shadow_truth": mask_files["attached_shadow"],
            "cast_shadow_truth": mask_files["cast_shadow"],
            "highlight_truth": mask_files["highlight"],
            "saturation_truth": mask_files["saturation"],
            "solve_mask": "solve_mask.png",
            "shape_index": "shape_index.png",
            "shape_indices": {
                name: index + 1 for index, name in enumerate(shape_names)
            },
            "normal_components": ["normal_x.png", "normal_y.png", "normal_z.png"],
            "position_z": "position_z.png",
            "lights": "lights.csv",
            "scene_assets": scene_assets,
        },
        "counts": aggregate,
        "sha256": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(generated_paths)
        },
    }
    with (output / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(json.dumps(aggregate, indent=2))
    print(f"Wrote fixture to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for one scene, or parent directory with --all",
    )
    parser.add_argument("--scene", choices=SCENE_NAMES, default="robust_v1")
    parser.add_argument("--all", action="store_true", help="Generate every fixture scene")
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument(
        "--variant",
        default="llvm_ad_rgb",
        help="Mitsuba variant; cuda_ad_rgb is much faster when an NVIDIA GPU is available",
    )
    args = parser.parse_args()
    if args.spp < 1:
        parser.error("--spp must be positive")
    mi.set_variant(args.variant)
    fixture_root = Path(__file__).resolve().parent
    if args.all:
        output_root = args.output.resolve() if args.output is not None else fixture_root
        for scene_name in SCENE_NAMES:
            generate(scene_name, output_root / scene_name, args.spp, args.variant)
    else:
        output = args.output.resolve() if args.output is not None else fixture_root / args.scene
        generate(args.scene, output, args.spp, args.variant)


if __name__ == "__main__":
    main()
