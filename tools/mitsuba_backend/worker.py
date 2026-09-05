#!/usr/bin/env python3
"""Optional Mitsuba inverse-rendering worker for what-a-relief.

The C++ application owns the baseline photometric solve and communicates with
this isolated process through a versioned JSON job. This worker never replaces
baseline files. Its guarded result is written to a separate inverse directory.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import platform
import shutil
import struct
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


JOB_SCHEMA_VERSION = 1
METHOD_ID = "mitsuba_heightfield_inverse_v1"
METHOD_REFERENCES = (
    {"id": "zhang2023projective", "doi": "10.1145/3618385"},
    {"id": "jakob2022drjit", "doi": "10.1145/3528223.3530099"},
    {"id": "mitsuba3", "url": "https://mitsuba-renderer.org/"},
)

QUALITY = {
    "preview": {
        "max_side": 64,
        "iterations_ad": 6,
        "spp": 1,
        "control_spacing": 12,
        "learning_rate": 0.018,
    },
    "standard": {
        "max_side": 128,
        "iterations_ad": 18,
        "spp": 2,
        "control_spacing": 10,
        "learning_rate": 0.012,
    },
    "research": {
        "max_side": 256,
        "iterations_ad": 50,
        "spp": 4,
        "control_spacing": 8,
        "learning_rate": 0.008,
    },
}

_DLL_DIRECTORY_HANDLES: list[Any] = []


def configure_runtime_paths() -> None:
    if os.name != "nt":
        return
    runtime_root = Path(sys.executable).resolve().parent
    if runtime_root.name.lower() == "scripts":
        runtime_root = runtime_root.parent
    candidates = [
        runtime_root / "LLVM-C.dll",
        runtime_root / "llvm" / "bin" / "LLVM-C.dll",
    ]
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root) / "LLVM" / "bin" / "LLVM-C.dll")
    if not os.environ.get("DRJIT_LIBLLVM_PATH"):
        llvm = next((path for path in candidates if path.is_file()), None)
        if llvm is not None:
            os.environ["DRJIT_LIBLLVM_PATH"] = str(llvm)
    for directory in {runtime_root, runtime_root / "llvm" / "bin"}:
        if directory.is_dir() and hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


class Progress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._atomic_replace_supported = True

    @staticmethod
    def _is_sharing_violation(error: OSError) -> bool:
        return error.errno in (errno.EACCES, errno.EPERM) or getattr(
            error, "winerror", None
        ) in (5, 32)

    def _write_network_compatible(self, text: str) -> None:
        last_error: OSError | None = None
        for attempt in range(6):
            try:
                self.path.write_text(text, encoding="utf-8")
                return
            except OSError as error:
                if not self._is_sharing_violation(error):
                    raise
                last_error = error
                time.sleep(0.025 * (attempt + 1))
        if last_error is not None:
            raise last_error

    def __call__(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        clean = " ".join(str(message).splitlines())
        text = f"{percent}\n{clean}\n"
        if self._atomic_replace_supported:
            try:
                atomic_text(self.path, text)
            except OSError as error:
                if not self._is_sharing_violation(error):
                    raise
                self._atomic_replace_supported = False
                try:
                    self.path.with_name(self.path.name + ".part").unlink(missing_ok=True)
                except OSError:
                    pass
                print(
                    "Progress-file atomic replacement is unavailable; "
                    "using network-share-compatible updates.",
                    flush=True,
                )
        if not self._atomic_replace_supported:
            self._write_network_compatible(text)
        print(f"[{percent:3d}%] {clean}", flush=True)


def configure_backend(name: str):
    configure_runtime_paths()
    import mitsuba as mi

    if name == "cuda":
        mi.set_variant("cuda_ad_rgb")
        return mi, "cuda_ad_rgb", "adam_projective_autodiff"
    if name == "llvm":
        mi.set_variant("llvm_ad_rgb")
        return mi, "llvm_ad_rgb", "adam_projective_autodiff"
    raise ValueError(f"Unsupported backend selection: {name}")


def tiny_probe(mi, backend: str) -> None:
    transform = mi.ScalarTransform4f
    integrator = "direct_projective"
    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": integrator},
            "sensor": {
                "type": "perspective",
                "to_world": transform().look_at(
                    origin=[0.0, 0.0, 3.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0]
                ),
                "fov": 38.0,
                "fov_axis": "x",
                "sampler": {"type": "independent", "sample_count": 1},
                "film": {
                    "type": "hdrfilm",
                    "width": 2,
                    "height": 2,
                    "pixel_format": "rgb",
                    "component_format": "float32",
                    "rfilter": {"type": "box"},
                },
            },
            "surface": {
                "type": "rectangle",
                "bsdf": {"type": "diffuse", "reflectance": 0.5},
            },
            "emitter": {
                "type": "directional",
                "direction": [0.0, 0.0, -1.0],
                "irradiance": math.pi,
            },
        }
    )
    image = mi.render(scene, spp=1, seed=11)
    value = float(np.asarray(image).mean())
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("Mitsuba live render probe returned no finite illumination")


def probe(backend: str, result_path: Path) -> int:
    try:
        mi, variant, optimizer = configure_backend(backend)
        import drjit as dr

        tiny_probe(mi, backend)
        atomic_json(
            result_path,
            {
                "status": "available",
                "selected_backend": backend,
                "variant": variant,
                "optimizer": optimizer,
                "mitsuba_version": mi.__version__,
                "drjit_version": dr.__version__,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
            },
        )
        return 0
    except Exception as error:
        atomic_json(
            result_path,
            {
                "status": "unavailable",
                "selected_backend": backend,
                "error": str(error),
                "python_version": platform.python_version(),
            },
        )
        traceback.print_exc()
        return 1


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        magic = stream.readline().decode("ascii").strip()
        if magic not in ("Pf", "PF"):
            raise ValueError(f"Unsupported PFM header in {path}: {magic}")
        width, height = [int(value) for value in stream.readline().decode("ascii").split()]
        scale = float(stream.readline().decode("ascii").strip())
        channels = 1 if magic == "Pf" else 3
        dtype = "<f4" if scale < 0.0 else ">f4"
        values = np.fromfile(stream, dtype=dtype, count=width * height * channels)
    if values.size != width * height * channels:
        raise ValueError(f"PFM payload is truncated: {path}")
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.flipud(values.reshape(shape)).astype(np.float32, copy=False)


def write_pfm(path: Path, image: np.ndarray) -> None:
    values = np.asarray(image, dtype="<f4")
    if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[2] != 3):
        raise ValueError("PFM output must be HxW or HxWx3")
    magic = "PF" if values.ndim == 3 else "Pf"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        stream.write(f"{magic}\n{values.shape[1]} {values.shape[0]}\n-1.0\n".encode("ascii"))
        np.flipud(values).tofile(stream)
    os.replace(temporary, path)


def _bitmap_array_local(mi, path: Path) -> np.ndarray:
    bitmap = mi.Bitmap(str(path))
    raw = np.asarray(bitmap)
    if np.issubdtype(raw.dtype, np.integer):
        values = raw.astype(np.float32) / float(np.iinfo(raw.dtype).max)
    else:
        values = raw.astype(np.float32)
    if values.ndim == 3:
        if values.shape[2] == 1:
            values = values[:, :, 0]
        else:
            values = (
                0.2126 * values[:, :, 0]
                + 0.7152 * values[:, :, 1]
                + 0.0722 * values[:, :, 2]
            )
    return np.maximum(values, 0.0).astype(np.float32)


def bitmap_array(mi, path: Path) -> np.ndarray:
    if not _is_remote_path(path):
        return _bitmap_array_local(mi, path)
    with tempfile.TemporaryDirectory(prefix="what-a-relief-mitsuba-input-") as directory:
        suffix = path.suffix if path.suffix else ".bin"
        local_path = Path(directory) / ("input" + suffix)
        shutil.copyfile(path, local_path)
        return _bitmap_array_local(mi, local_path)


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.where(
        clipped <= 0.04045,
        clipped / 12.92,
        np.power((clipped + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def read_stack(mi, paths: list[Path], srgb: bool) -> np.ndarray:
    images = []
    for path in paths:
        try:
            images.append(bitmap_array(mi, path))
        except Exception as error:
            raise RuntimeError(f"Could not decode inverse observation '{path}': {error}") from error
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("All inverse-rendering input images must have matching dimensions")
    stack = np.stack(images, axis=0)
    if srgb:
        stack = srgb_to_linear(stack)
    stack[~np.isfinite(stack)] = 0.0
    peak = float(np.max(stack))
    if peak <= np.finfo(np.float32).eps:
        raise ValueError("Inverse-rendering image stack contains no positive finite samples")
    if peak > 1.0:
        stack /= peak
    return stack


def resize_bilinear(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    old_height, old_width = image.shape[-2:]
    if (old_height, old_width) == (new_height, new_width):
        return image.copy()
    ys = np.linspace(0.0, old_height - 1.0, new_height, dtype=np.float32)
    xs = np.linspace(0.0, old_width - 1.0, new_width, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, old_height - 1)
    x1 = np.minimum(x0 + 1, old_width - 1)
    fy = (ys - y0).reshape((-1, 1))
    fx = (xs - x0).reshape((1, -1))
    top = image[..., y0[:, None], x0[None, :]] * (1.0 - fx) + image[..., y0[:, None], x1[None, :]] * fx
    bottom = image[..., y1[:, None], x0[None, :]] * (1.0 - fx) + image[..., y1[:, None], x1[None, :]] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(np.float32)


def resize_nearest(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    ys = np.rint(np.linspace(0.0, image.shape[-2] - 1.0, new_height)).astype(np.int32)
    xs = np.rint(np.linspace(0.0, image.shape[-1] - 1.0, new_width)).astype(np.int32)
    return image[..., ys[:, None], xs[None, :]].copy()


def crop_bounds(mask: np.ndarray, margin: int = 3) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size < 100:
        raise ValueError("Inverse-rendering mask has fewer than 100 pixels")
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(mask.shape[1], int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(mask.shape[0], int(ys.max()) + margin + 1)
    return x0, y0, x1, y1


def target_size(height: int, width: int, maximum: int) -> tuple[int, int]:
    scale = min(1.0, maximum / float(max(height, width)))
    return max(16, int(round(height * scale))), max(16, int(round(width * scale)))


def normalized_height(height: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    values = height[mask]
    datum = float(np.median(values))
    result = height.astype(np.float32) - datum
    result[~mask] = 0.0
    return result, datum


def surface_normals(height: np.ndarray) -> np.ndarray:
    q, p = np.gradient(height.astype(np.float32))
    normals = np.stack((-p, q, np.ones_like(height)), axis=-1)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(length, 1.0e-8)


def control_mapping(height: int, width: int, spacing: int):
    control_height = max(3, int(math.ceil((height - 1) / spacing)) + 1)
    control_width = max(3, int(math.ceil((width - 1) / spacing)) + 1)
    ys = np.linspace(0.0, control_height - 1.0, height, dtype=np.float32)
    xs = np.linspace(0.0, control_width - 1.0, width, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, control_height - 1)
    x1 = np.minimum(x0 + 1, control_width - 1)
    fy = (ys - y0).reshape((-1, 1))
    fx = (xs - x0).reshape((1, -1))
    index00 = y0[:, None] * control_width + x0[None, :]
    index01 = y0[:, None] * control_width + x1[None, :]
    index10 = y1[:, None] * control_width + x0[None, :]
    index11 = y1[:, None] * control_width + x1[None, :]
    weight00 = (1.0 - fy) * (1.0 - fx)
    weight01 = (1.0 - fy) * fx
    weight10 = fy * (1.0 - fx)
    weight11 = fy * fx
    return (
        control_height,
        control_width,
        tuple(index.astype(np.uint32).ravel() for index in (index00, index01, index10, index11)),
        tuple(weight.astype(np.float32).ravel() for weight in (weight00, weight01, weight10, weight11)),
    )


def expand_control_numpy(control: np.ndarray, mapping) -> np.ndarray:
    _, _, indices, weights = mapping
    flat = control.ravel()
    expanded = sum(flat[index] * weight for index, weight in zip(indices, weights))
    return expanded


def write_grid_ply(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    height, width = z.shape
    xx, yy = np.meshgrid(x, y)
    vertices = np.stack((xx, yy, z), axis=-1).reshape((-1, 3))
    rows = np.arange(height - 1, dtype=np.int32)[:, None]
    cols = np.arange(width - 1, dtype=np.int32)[None, :]
    a = rows * width + cols
    b = a + 1
    c = a + width
    d = c + 1
    faces = np.concatenate(
        (
            np.stack((a, c, b), axis=-1).reshape((-1, 3)),
            np.stack((b, c, d), axis=-1).reshape((-1, 3)),
        ),
        axis=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(
            (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {vertices.shape[0]}\n"
                "property float x\nproperty float y\nproperty float z\n"
                f"element face {faces.shape[0]}\n"
                "property list uchar int vertex_indices\nend_header\n"
            ).encode("ascii")
        )
        vertices.astype("<f4", copy=False).tofile(stream)
        records = np.empty(faces.shape[0], dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]))
        records["count"] = 3
        records["indices"] = faces
        records.tofile(stream)


def write_masked_ply(path: Path, height: np.ndarray, mask: np.ndarray, albedo: np.ndarray, z_scale: float) -> None:
    rows, cols = height.shape
    valid = mask & np.isfinite(height)
    index = np.full((rows, cols), -1, dtype=np.int32)
    index[valid] = np.arange(int(valid.sum()), dtype=np.int32)
    ys, xs = np.nonzero(valid)
    gray = np.clip(albedo[valid] * 255.0, 0.0, 255.0).astype(np.uint8)
    vertices = np.empty(
        ys.size,
        dtype=np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
        ),
    )
    vertices["x"] = xs.astype(np.float32)
    vertices["y"] = -ys.astype(np.float32)
    vertices["z"] = height[valid].astype(np.float32) * float(z_scale)
    vertices["r"] = gray
    vertices["g"] = gray
    vertices["b"] = gray
    a = index[:-1, :-1]
    b = index[:-1, 1:]
    c = index[1:, :-1]
    d = index[1:, 1:]
    first = (a >= 0) & (b >= 0) & (c >= 0)
    second = (b >= 0) & (c >= 0) & (d >= 0)
    faces_first = np.stack((a[first], c[first], b[first]), axis=-1)
    faces_second = np.stack((b[second], c[second], d[second]), axis=-1)
    faces = np.concatenate((faces_first, faces_second), axis=0).astype(np.int32, copy=False)
    with path.open("wb") as stream:
        stream.write(
            (
                "ply\nformat binary_little_endian 1.0\n"
                "comment generated by what-a-relief Mitsuba inverse backend\n"
                f"element vertex {vertices.size}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"element face {faces.shape[0]}\n"
                "property list uchar int vertex_indices\nend_header\n"
            ).encode("ascii")
        )
        vertices.tofile(stream)
        records = np.empty(faces.shape[0], dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]))
        records["count"] = 3
        records["indices"] = faces
        records.tofile(stream)


def scene_dictionary(mi, mesh_path: Path, basis: str, light: np.ndarray, geometry: dict[str, Any], camera: dict[str, Any], backend: str, spp: int):
    transform = mi.ScalarTransform4f
    if basis == "diffuse":
        bsdf = {"type": "diffuse", "reflectance": 1.0}
    else:
        alpha = 0.075 if basis == "narrow" else 0.32
        bsdf = {
            "type": "roughplastic",
            "distribution": "ggx",
            "alpha": alpha,
            "int_ior": 1.57,
            "ext_ior": "air",
            "diffuse_reflectance": 0.0,
            "specular_reflectance": 1.0,
            "nonlinear": False,
        }
    if geometry["lighting_model"] == "near_field_ring":
        radial = math.hypot(float(light[0]), float(light[1]))
        if radial <= 1.0e-8:
            azimuth = geometry["fallback_azimuth"]
            ax, ay = math.cos(azimuth), math.sin(azimuth)
        else:
            ax, ay = float(light[0]) / radial, float(light[1]) / radial
        radius = geometry["ring_radius_world"]
        light_height = geometry["ring_height_world"]
        position = [radius * ax, radius * ay, light_height]
        reference_squared = radius * radius + light_height * light_height
        emitter = {
            "type": "point",
            "position": position,
            "intensity": math.pi * reference_squared,
        }
    else:
        emitter = {
            "type": "directional",
            "direction": [-float(light[0]), -float(light[1]), -float(light[2])],
            "irradiance": math.pi,
        }
    integrator = "direct_projective"
    return {
        "type": "scene",
        "integrator": {"type": integrator},
        "sensor": {
            "type": "perspective",
            "to_world": transform().look_at(
                origin=camera["origin"], target=camera["target"], up=[0.0, 1.0, 0.0]
            ),
            "fov": camera["fov_degrees"],
            "fov_axis": "x",
            "near_clip": camera["near"],
            "far_clip": camera["far"],
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {
                "type": "hdrfilm",
                "width": camera["width"],
                "height": camera["height"],
                "pixel_format": "rgb",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            },
        },
        "surface": {
            "type": "ply",
            "filename": str(mesh_path),
            "face_normals": False,
            "bsdf": bsdf,
        },
        "emitter": emitter,
    }


def scene_parameters(mi, scenes):
    result = []
    for scene in scenes:
        params = mi.traverse(scene)
        keys = [key for key in params.keys() if key.endswith("vertex_positions")]
        if len(keys) != 1:
            raise RuntimeError(f"Expected one differentiable mesh in Mitsuba scene, found: {keys}")
        result.append((params, keys[0]))
    return result


def render_numpy(mi, scenes, parameters, positions: np.ndarray, spp: int, seed: int) -> np.ndarray:
    rendered = []
    flattened = mi.Float(positions.astype(np.float32).ravel())
    for index, (scene, (params, key)) in enumerate(zip(scenes, parameters)):
        params[key] = flattened
        params.update()
        image = np.asarray(mi.render(scene, params, spp=spp, seed=seed + index))
        rendered.append(image.mean(axis=2).astype(np.float32))
    return np.stack(rendered, axis=0)


def fit_material_maps(basis: np.ndarray, observed: np.ndarray, mask: np.ndarray, albedo_prior: np.ndarray, train: list[int]):
    # basis has shape [basis, light, y, x]. Solve three nonnegative coefficients
    # per pixel with a weak prior on classical diffuse albedo.
    x = np.moveaxis(basis[:, train], 0, -1)  # [light, y, x, basis]
    x = np.moveaxis(x, 0, -2)  # [y, x, light, basis]
    y = np.moveaxis(observed[train], 0, -1)[..., None]  # [y, x, light, 1]
    xt = np.swapaxes(x, -1, -2)
    normal = xt @ x
    rhs = (xt @ y)[..., 0]
    ridge = np.array([0.08, 0.14, 0.14], dtype=np.float32)
    normal += np.eye(3, dtype=np.float32) * ridge
    rhs[..., 0] += ridge[0] * albedo_prior
    try:
        coefficients = np.linalg.solve(normal, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        coefficients = np.zeros((*mask.shape, 3), dtype=np.float32)
        for row, col in zip(*np.nonzero(mask)):
            coefficients[row, col] = np.linalg.lstsq(
                normal[row, col], rhs[row, col], rcond=1.0e-5
            )[0]
    coefficients = np.maximum(coefficients, 0.0).astype(np.float32)
    coefficients[..., 0] = np.minimum(coefficients[..., 0], max(2.0, float(np.quantile(albedo_prior[mask], 0.995)) * 2.5))
    specular_limit = max(0.25, float(np.quantile(observed[:, mask], 0.995)) * 3.0)
    coefficients[..., 1:] = np.minimum(coefficients[..., 1:], specular_limit)
    coefficients[~mask] = 0.0
    specular = coefficients[..., 1] + coefficients[..., 2]
    roughness = np.where(
        specular > 1.0e-6,
        (0.075 * coefficients[..., 1] + 0.32 * coefficients[..., 2]) / np.maximum(specular, 1.0e-6),
        0.32,
    ).astype(np.float32)
    return coefficients, roughness


def prediction_numpy(basis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.sum(basis * np.moveaxis(coefficients, -1, 0)[:, None, :, :], axis=0)


def data_loss_numpy(prediction: np.ndarray, observed: np.ndarray, weights: np.ndarray, indices: list[int]) -> float:
    residual = prediction[indices] - observed[indices]
    robust = np.sqrt(residual * residual + 0.0025**2) - 0.0025
    selected = weights[indices]
    denominator = float(selected.sum())
    return float((robust * selected).sum() / max(denominator, 1.0e-8))


def _is_remote_path(path: Path) -> bool:
    if os.name != "nt":
        return False
    anchor = path.anchor
    if anchor.startswith(("\\\\", "//")):
        return True
    if len(anchor) >= 2 and anchor[1] == ":":
        try:
            import ctypes

            return ctypes.windll.kernel32.GetDriveTypeW(anchor) == 4
        except (AttributeError, OSError):
            return False
    return False


def _copy_completed_file(source: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_bitmap(path: Path, bitmap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _is_remote_path(path):
        bitmap.write(str(path))
        return
    with tempfile.TemporaryDirectory(prefix="what-a-relief-mitsuba-png-") as directory:
        local_path = Path(directory) / path.name
        bitmap.write(str(local_path))
        _copy_completed_file(local_path, path)


def save_gray(mi, path: Path, image: np.ndarray) -> None:
    values = np.clip(image, 0.0, 1.0).astype(np.float32)
    bitmap = mi.Bitmap(values).convert(mi.Bitmap.PixelFormat.Y, mi.Struct.Type.UInt8, False)
    _write_bitmap(path, bitmap)


def save_rgb(mi, path: Path, image: np.ndarray) -> None:
    values = np.clip(image, 0.0, 1.0).astype(np.float32)
    bitmap = mi.Bitmap(values).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, False)
    _write_bitmap(path, bitmap)


def display_stretch(values: np.ndarray, mask: np.ndarray, symmetric: bool = False) -> np.ndarray:
    selected = values[mask & np.isfinite(values)]
    if selected.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    if symmetric:
        extent = max(float(np.quantile(np.abs(selected), 0.995)), 1.0e-8)
        output = 0.5 + 0.5 * values / extent
    else:
        low, high = np.quantile(selected, [0.005, 0.995])
        output = (values - float(low)) / max(float(high - low), 1.0e-8)
    output = np.clip(output, 0.0, 1.0).astype(np.float32)
    output[~mask] = 0.0
    return output


def output_products(mi, output: Path, baseline: np.ndarray, correction: np.ndarray, mask: np.ndarray, albedo: np.ndarray, height_scale: float) -> None:
    final_height = baseline + correction
    final_height[~mask] = 0.0
    normals = surface_normals(final_height)
    normals[~mask] = 0.0
    write_pfm(output / "inverse_height.pfm", final_height)
    write_pfm(output / "height_correction.pfm", correction)
    save_gray(mi, output / "inverse_height.png", display_stretch(final_height, mask))
    save_gray(mi, output / "height_correction.png", display_stretch(correction, mask, symmetric=True))
    encoded = normals * 0.5 + 0.5
    encoded[~mask] = 0.0
    save_rgb(mi, output / "inverse_normal_rgb.png", encoded)
    save_gray(mi, output / "inverse_normal_x.png", np.where(mask, normals[..., 0] * 0.5 + 0.5, 0.0))
    save_gray(mi, output / "inverse_normal_y.png", np.where(mask, normals[..., 1] * 0.5 + 0.5, 0.0))
    save_gray(mi, output / "inverse_normal_z.png", np.where(mask, np.maximum(normals[..., 2], 0.0), 0.0))
    light = np.array([-0.5, 0.5, math.sqrt(0.5)], dtype=np.float32)
    hillshade = np.maximum(0.0, np.sum(normals * light, axis=-1))
    save_gray(mi, output / "inverse_hillshade_ul.png", np.where(mask, 0.14 + 0.86 * hillshade, 0.0))
    write_masked_ply(output / "inverse_surface.ply", final_height, mask, albedo, height_scale)


def prepare_job(
    mi,
    job: dict[str, Any],
    settings: dict[str, Any],
    progress: Progress,
    renderer_directory: Path,
):
    inputs = job["inputs"]
    geometry_job = job["geometry"]
    parameters = job["parameters"]
    image_paths = [Path(path) for path in inputs["images"]]
    transfer = inputs.get("image_transfer")
    if transfer is not None:
        if (
            transfer.get("encoding") != "png16"
            or transfer.get("photometry") != "linear_luminance"
            or bool(parameters["srgb_decode"])
        ):
            raise ValueError("Unsupported or inconsistent inverse observation handoff")
    lights = np.asarray(inputs["lights"], dtype=np.float32)
    if len(image_paths) < 6 or lights.shape != (len(image_paths), 3):
        raise ValueError("Inverse rendering requires at least six images and one 3-vector per image")
    progress(5, "Loading linear image stack and baseline geometry")
    images = read_stack(mi, image_paths, bool(parameters["srgb_decode"]))
    height = read_pfm(Path(inputs["height_pfm"]))
    albedo = read_pfm(Path(inputs["albedo_pfm"]))
    mask = bitmap_array(mi, Path(inputs["mask_png"])) > 0.0
    if images.shape[1:] != height.shape or height.shape != albedo.shape or mask.shape != height.shape:
        raise ValueError("Mitsuba handoff image, height, albedo, and mask dimensions differ")
    robust_weight = np.ones_like(height, dtype=np.float32)
    robust_path = inputs.get("robust_weight_pfm")
    if robust_path:
        robust_weight = np.clip(read_pfm(Path(robust_path)), 0.05, 1.0)
    x0, y0, x1, y1 = crop_bounds(mask)
    images_crop = images[:, y0:y1, x0:x1]
    height_crop = height[y0:y1, x0:x1]
    albedo_crop = albedo[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    weight_crop = robust_weight[y0:y1, x0:x1]
    render_height, render_width = target_size(y1 - y0, x1 - x0, int(settings["max_side"]))
    images_small = resize_bilinear(images_crop, render_height, render_width)
    height_small = resize_bilinear(height_crop, render_height, render_width)
    albedo_small = resize_bilinear(albedo_crop, render_height, render_width)
    mask_small = resize_nearest(mask_crop.astype(np.float32), render_height, render_width) > 0.5
    weight_small = resize_bilinear(weight_crop, render_height, render_width)
    height_small, datum = normalized_height(height_small, mask_small)

    full_center_x = 0.5 * (height.shape[1] - 1)
    full_center_y = 0.5 * (height.shape[0] - 1)
    crop = geometry_job.get("crop")
    if crop:
        full_center_x = float(crop["x"]) + 0.5 * (float(crop["width"]) - 1.0)
        full_center_y = float(crop["y"]) + 0.5 * (float(crop["height"]) - 1.0)
    original_x = np.linspace(x0, x1 - 1, render_width, dtype=np.float32)
    original_y = np.linspace(y0, y1 - 1, render_height, dtype=np.float32)
    near_field = geometry_job["lighting_model"] == "near_field_ring"
    pixel_scale = float(geometry_job["pixel_scale_mm_per_pixel"])
    if near_field and pixel_scale <= 0.0:
        raise ValueError("Near-field Mitsuba refinement requires positive mm/pixel scale")
    physical_scale = pixel_scale if near_field else 1.0
    world_x = (original_x - full_center_x) * physical_scale
    world_y = (full_center_y - original_y) * physical_scale
    world_z = height_small * physical_scale
    extent = max(
        float(np.ptp(world_x)) + max(physical_scale, 1.0e-6),
        float(np.ptp(world_y)) + max(physical_scale, 1.0e-6),
        float(geometry_job["ring_height_mm"]) if near_field else 0.0,
        1.0e-3,
    )
    scene_scale = 2.0 / extent
    world_x *= scene_scale
    world_y *= scene_scale
    world_z *= scene_scale
    baseline_positions = np.stack(np.meshgrid(world_x, world_y), axis=-1)
    baseline_positions = np.concatenate((baseline_positions, world_z[..., None]), axis=-1).astype(np.float32)
    renderer_directory.mkdir(parents=True, exist_ok=True)
    mesh_path = renderer_directory / "optimization_mesh.ply"
    write_grid_ply(mesh_path, world_x, world_y, world_z)
    audit_mesh_path = Path(job["outputs"]["directory"]) / "optimization_mesh.ply"
    _copy_completed_file(mesh_path, audit_mesh_path)
    camera_center_x = 0.5 * float(world_x[0] + world_x[-1])
    camera_center_y = 0.5 * float(world_y[0] + world_y[-1])
    center_z = float(np.median(world_z[mask_small]))
    camera_width = float(np.ptp(world_x)) + abs(float(world_x[1] - world_x[0]))
    camera_height = float(np.ptp(world_y)) + abs(float(world_y[1] - world_y[0]))
    camera_extent = max(camera_width, camera_height, 1.0)
    film_aspect = render_width / float(render_height)
    fitted_width = max(camera_width, camera_height * film_aspect)
    camera_distance = 100.0 * camera_extent
    camera = {
        "origin": [camera_center_x, camera_center_y, center_z + camera_distance],
        "target": [camera_center_x, camera_center_y, center_z],
        "fov_degrees": math.degrees(2.0 * math.atan(0.5 * fitted_width / camera_distance)),
        "near": 0.001 * camera_extent,
        "far": camera_distance + 8.0 * camera_extent,
        "width": render_width,
        "height": render_height,
    }
    geometry = {
        "lighting_model": geometry_job["lighting_model"],
        "ring_radius_world": float(geometry_job["ring_radius_mm"]) * scene_scale,
        "ring_height_world": float(geometry_job["ring_height_mm"]) * scene_scale,
        "fallback_azimuth": 0.0,
    }
    weights = (
        mask_small[None, :, :].astype(np.float32)
        * np.clip(weight_small[None, :, :], 0.05, 1.0)
        * (images_small < 0.995).astype(np.float32)
    )
    return {
        "images": images,
        "height": height,
        "albedo": albedo,
        "mask": mask,
        "bounds": (x0, y0, x1, y1),
        "images_small": images_small,
        "height_small": height_small,
        "albedo_small": albedo_small,
        "mask_small": mask_small,
        "weights": weights,
        "lights": lights,
        "baseline_positions": baseline_positions,
        "mesh_path": mesh_path,
        "camera": camera,
        "geometry": geometry,
        "scene_scale": scene_scale,
        "physical_scale": physical_scale,
        "height_datum": datum,
    }


def make_scenes(mi, prepared, backend: str, spp: int):
    scenes_by_basis = []
    parameters_by_basis = []
    for basis in ("diffuse", "narrow", "broad"):
        scenes = []
        for index, light in enumerate(prepared["lights"]):
            geometry = dict(prepared["geometry"])
            geometry["fallback_azimuth"] = 2.0 * math.pi * index / len(prepared["lights"])
            scenes.append(
                mi.load_dict(
                    scene_dictionary(
                        mi,
                        prepared["mesh_path"],
                        basis,
                        light,
                        geometry,
                        prepared["camera"],
                        backend,
                        spp,
                    )
                )
            )
        scenes_by_basis.append(scenes)
        parameters_by_basis.append(scene_parameters(mi, scenes))
    return scenes_by_basis, parameters_by_basis


def render_basis_numpy(mi, scenes_by_basis, parameters_by_basis, positions, spp: int, seed: int):
    return np.stack(
        [
            render_numpy(mi, scenes, parameters, positions, spp, seed + basis_index * 1009)
            for basis_index, (scenes, parameters) in enumerate(zip(scenes_by_basis, parameters_by_basis))
        ],
        axis=0,
    )


def split_lights(count: int) -> tuple[list[int], list[int]]:
    holdout_count = max(1, min(3, count // 5))
    candidates = np.linspace(0, count - 1, holdout_count, endpoint=False)
    holdout = sorted(set(int(round(value)) for value in candidates))
    while len(holdout) < holdout_count:
        candidate = (holdout[-1] + max(1, count // holdout_count)) % count if holdout else 0
        if candidate not in holdout:
            holdout.append(candidate)
    train = [index for index in range(count) if index not in holdout]
    if len(train) < 4:
        raise ValueError("Inverse-rendering holdout requires at least four training lights")
    return train, sorted(holdout)


def optimize_ad(mi, prepared, scenes_by_basis, parameters_by_basis, coefficients, mapping, settings, train, progress):
    import drjit as dr

    control_height, control_width, indices_numpy, weights_numpy = mapping
    optimizer = mi.ad.Adam(lr=float(settings["learning_rate"]))
    optimizer["height_control"] = mi.Float(np.zeros(control_height * control_width, dtype=np.float32))
    index_arrays = [mi.UInt(index) for index in indices_numpy]
    weight_arrays = [mi.Float(weight) for weight in weights_numpy]
    baseline_flat = mi.Float(prepared["baseline_positions"].ravel())
    observed = [mi.TensorXf(image) for image in prepared["images_small"]]
    data_weights = [mi.TensorXf(weight) for weight in prepared["weights"]]
    material = [mi.TensorXf(coefficients[..., index]) for index in range(3)]
    iterations = int(settings["iterations_ad"])
    maximum_delta = 0.22
    render_height, render_width = prepared["height_small"].shape

    def expanded_control():
        control = optimizer["height_control"]
        return sum(
            dr.gather(mi.Float, control, index) * weight
            for index, weight in zip(index_arrays, weight_arrays)
        )

    for iteration in range(iterations):
        expanded = expanded_control()
        positions = baseline_flat + dr.ravel(mi.Vector3f(
            dr.zeros(mi.Float, render_height * render_width),
            dr.zeros(mi.Float, render_height * render_width),
            expanded,
        ))
        rendered_by_basis = []
        for basis_index, (basis_scenes, basis_parameters) in enumerate(zip(scenes_by_basis, parameters_by_basis)):
            rendered_lights = []
            for light_index, (scene, (params, key)) in enumerate(zip(basis_scenes, basis_parameters)):
                params[key] = positions
                params.update()
                image = mi.render(
                    scene,
                    params,
                    spp=int(settings["spp"]),
                    spp_grad=int(settings["spp"]),
                    seed=30000 + iteration * 1009 + basis_index * 101 + light_index,
                )
                rendered_lights.append(dr.mean(image, axis=2))
            rendered_by_basis.append(rendered_lights)

        total = mi.Float(0.0)
        denominator = mi.Float(0.0)
        for light_index in train:
            prediction = sum(
                rendered_by_basis[basis][light_index] * material[basis]
                for basis in range(3)
            )
            residual = prediction - observed[light_index]
            robust = dr.sqrt(residual * residual + 0.0025**2) - 0.0025
            total += dr.sum(robust * data_weights[light_index])
            denominator += dr.sum(data_weights[light_index])

        control = optimizer["height_control"]
        grid = mi.TensorXf(control, shape=(control_height, control_width))
        dx = grid[:, 1:] - grid[:, :-1]
        dy = grid[1:, :] - grid[:-1, :]
        dxx = dx[:, 1:] - dx[:, :-1]
        dyy = dy[1:, :] - dy[:-1, :]
        regularization = (
            0.012 * dr.mean(control * control)
            + 0.025 * (dr.mean(dx * dx) + dr.mean(dy * dy))
            + 0.018 * (dr.mean(dxx * dxx) + dr.mean(dyy * dyy))
        )
        loss = total / dr.maximum(denominator, 1.0e-8) + regularization
        dr.backward(loss)
        optimizer.step()
        optimizer["height_control"] = dr.clip(optimizer["height_control"], -maximum_delta, maximum_delta)
        progress(22 + int(58 * (iteration + 1) / iterations), f"Differentiable inverse iteration {iteration + 1}/{iterations}")
    control = optimizer["height_control"].numpy().reshape((control_height, control_width)).astype(np.float32)
    expanded = expand_control_numpy(control, mapping).reshape(prepared["height_small"].shape)
    return control, expanded, iterations


def run_job(job_path: Path) -> int:
    output = job_path.parent
    progress = Progress(output / "progress.txt")
    result_path = output / "result.json"
    result: dict[str, Any] = {
        "schema_version": JOB_SCHEMA_VERSION,
        "method": METHOD_ID,
        "status": "failed",
        "accepted": False,
        "decision": "worker_failed",
        "method_references": list(METHOD_REFERENCES),
    }
    renderer_workspace = None
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job.get("schema_version") != JOB_SCHEMA_VERSION or job.get("method") != METHOD_ID:
            raise ValueError("Unsupported Mitsuba job schema or method identifier")
        quality = str(job["parameters"]["quality"])
        if quality not in QUALITY:
            raise ValueError(f"Unsupported quality preset: {quality}")
        selected_backend = str(job["parameters"]["backend_selected"])
        mi, variant, optimizer_name = configure_backend(selected_backend)
        import drjit as dr

        result.update(
            {
                "selected_backend": selected_backend,
                "variant": variant,
                "optimizer": optimizer_name,
                "quality": quality,
                "mitsuba_version": mi.__version__,
                "drjit_version": dr.__version__,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
            }
        )
        settings = QUALITY[quality]
        progress(3, f"Starting {variant} inverse-rendering worker")
        renderer_workspace = tempfile.TemporaryDirectory(
            prefix="what-a-relief-mitsuba-renderer-"
        )
        prepared = prepare_job(
            mi, job, settings, progress, Path(renderer_workspace.name)
        )
        render_height, render_width = prepared["height_small"].shape
        result["render_width"] = render_width
        result["render_height"] = render_height
        progress(10, "Building diffuse and spatially varying glossy material bases")
        scenes_by_basis, parameters_by_basis = make_scenes(
            mi, prepared, selected_backend, int(settings["spp"])
        )
        train, holdout = split_lights(len(prepared["lights"]))
        result["training_light_indices"] = train
        result["holdout_light_indices"] = holdout
        progress(15, "Rendering baseline material and visibility bases")
        baseline_basis = render_basis_numpy(
            mi,
            scenes_by_basis,
            parameters_by_basis,
            prepared["baseline_positions"],
            int(settings["spp"]),
            10000,
        )
        coefficients, roughness = fit_material_maps(
            baseline_basis,
            prepared["images_small"],
            prepared["mask_small"],
            prepared["albedo_small"],
            train,
        )
        prediction_before = prediction_numpy(baseline_basis, coefficients)
        train_before = data_loss_numpy(
            prediction_before, prepared["images_small"], prepared["weights"], train
        )
        holdout_before = data_loss_numpy(
            prediction_before, prepared["images_small"], prepared["weights"], holdout
        )
        mapping = control_mapping(render_height, render_width, int(settings["control_spacing"]))
        progress(20, "Optimizing a smooth height correction while preserving baseline detail")
        control, correction_small_world, iterations = optimize_ad(
            mi,
            prepared,
            scenes_by_basis,
            parameters_by_basis,
            coefficients,
            mapping,
            settings,
            train,
            progress,
        )
        del control
        candidate_positions = prepared["baseline_positions"].copy()
        candidate_positions[..., 2] += correction_small_world
        progress(82, "Rendering fixed-seed candidate for training and withheld-light validation")
        candidate_basis = render_basis_numpy(
            mi,
            scenes_by_basis,
            parameters_by_basis,
            candidate_positions,
            int(settings["spp"]),
            10000,
        )
        prediction_after = prediction_numpy(candidate_basis, coefficients)
        train_after = data_loss_numpy(
            prediction_after, prepared["images_small"], prepared["weights"], train
        )
        holdout_after = data_loss_numpy(
            prediction_after, prepared["images_small"], prepared["weights"], holdout
        )
        conversion = prepared["scene_scale"] * prepared["physical_scale"]
        correction_small_pixels = correction_small_world / max(conversion, 1.0e-12)
        masked_correction = correction_small_pixels[prepared["mask_small"]]
        correction_rms = float(np.sqrt(np.mean(masked_correction**2)))
        correction_maximum = float(np.max(np.abs(masked_correction)))
        q, p = np.gradient(correction_small_pixels)
        slope_rms = float(np.sqrt(np.mean((p[prepared["mask_small"]] ** 2 + q[prepared["mask_small"]] ** 2))))
        train_improvement = (train_before - train_after) / max(train_before, 1.0e-8)
        holdout_improvement = (holdout_before - holdout_after) / max(holdout_before, 1.0e-8)
        accepted = (
            math.isfinite(train_after)
            and math.isfinite(holdout_after)
            and train_improvement >= 0.01
            and holdout_improvement >= -0.002
            and slope_rms <= 0.45
            and correction_maximum <= 0.20 * max(prepared["height"].shape)
        )
        if accepted:
            decision = "accepted_train_and_holdout_gate"
        elif train_improvement < 0.01:
            decision = "rejected_insufficient_training_improvement"
        elif holdout_improvement < -0.002:
            decision = "rejected_withheld_lights_worsened"
        elif slope_rms > 0.45:
            decision = "rejected_excessive_slope_change"
        else:
            decision = "rejected_excessive_height_change"

        x0, y0, x1, y1 = prepared["bounds"]
        correction_crop = resize_bilinear(
            correction_small_pixels,
            y1 - y0,
            x1 - x0,
        )
        correction_full = np.zeros_like(prepared["height"], dtype=np.float32)
        if accepted:
            correction_full[y0:y1, x0:x1] = correction_crop
            correction_full[~prepared["mask"]] = 0.0
        progress(90, "Writing guarded inverse geometry, material maps, and audit previews")
        output_products(
            mi,
            output,
            prepared["height"],
            correction_full,
            prepared["mask"],
            prepared["albedo"],
            float(job["parameters"]["height_scale"]),
        )
        diffuse_full = np.zeros_like(prepared["height"], dtype=np.float32)
        specular_full = np.zeros_like(prepared["height"], dtype=np.float32)
        roughness_full = np.zeros_like(prepared["height"], dtype=np.float32)
        diffuse_full[y0:y1, x0:x1] = resize_bilinear(coefficients[..., 0], y1 - y0, x1 - x0)
        specular_full[y0:y1, x0:x1] = resize_bilinear(
            coefficients[..., 1] + coefficients[..., 2], y1 - y0, x1 - x0
        )
        roughness_full[y0:y1, x0:x1] = resize_bilinear(roughness, y1 - y0, x1 - x0)
        save_gray(mi, output / "material_diffuse.png", display_stretch(diffuse_full, prepared["mask"]))
        save_gray(mi, output / "material_specular.png", display_stretch(specular_full, prepared["mask"]))
        save_gray(mi, output / "material_roughness.png", np.where(prepared["mask"], roughness_full / 0.5, 0.0))
        mean_before = np.mean(prediction_before[train], axis=0)
        mean_after = np.mean(prediction_after[train], axis=0)
        save_gray(mi, output / "render_before.png", display_stretch(mean_before, prepared["mask_small"]))
        save_gray(mi, output / "render_after.png", display_stretch(mean_after, prepared["mask_small"]))
        result.update(
            {
                "status": "complete",
                "accepted": accepted,
                "decision": decision,
                "iterations_completed": iterations,
                "train_loss_before": train_before,
                "train_loss_after": train_after,
                "holdout_loss_before": holdout_before,
                "holdout_loss_after": holdout_after,
                "train_relative_improvement": train_improvement,
                "holdout_relative_improvement": holdout_improvement,
                "correction_rms_pixels": correction_rms if accepted else 0.0,
                "correction_maximum_pixels": correction_maximum if accepted else 0.0,
                "candidate_correction_rms_pixels": correction_rms,
                "candidate_correction_maximum_pixels": correction_maximum,
                "candidate_slope_change_rms": slope_rms,
                "height_datum_assumption": "median masked integrated height equals the light-reference plane",
                "material_model": "per-pixel nonnegative diffuse plus narrow- and broad-GGX rough-plastic bases",
                "baseline_outputs_modified": False,
            }
        )
        atomic_json(result_path, result)
        progress(100, "Mitsuba inverse refinement complete")
        return 0
    except Exception as error:
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        try:
            atomic_json(result_path, result)
            progress(100, f"Mitsuba inverse refinement failed: {error}")
        except Exception:
            pass
        traceback.print_exc()
        return 1
    finally:
        if renderer_workspace is not None:
            renderer_workspace.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--backend", choices=("cuda", "llvm"))
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.probe:
        if args.backend is None or args.result is None or args.job is not None:
            parser.error("--probe requires --backend and --result, without --job")
    elif args.job is None or args.backend is not None or args.result is not None:
        parser.error("normal execution requires only --job")
    return args


def main() -> int:
    args = parse_args()
    if args.probe:
        return probe(args.backend, args.result)
    return run_job(args.job.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
