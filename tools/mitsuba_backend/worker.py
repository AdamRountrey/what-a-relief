#!/usr/bin/env python3
"""Optional Mitsuba inverse-rendering worker for what-a-relief.

The C++ application owns the baseline photometric solve and communicates with
this isolated process through a versioned JSON job. This worker never replaces
baseline files. Its guarded result is written to a separate inverse directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
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


JOB_SCHEMA_VERSION = 2
METHOD_ID = "mitsuba_heightfield_inverse_v2"
NORMAL_PRIOR_STRENGTH = 0.25
MIN_LIGHT_SUPPORT = 64
METHOD_REFERENCES = (
    {"id": "zhang2023projective", "doi": "10.1145/3618385"},
    {"id": "jakob2022drjit", "doi": "10.1145/3528223.3530099"},
    {"id": "mitsuba3", "url": "https://mitsuba-renderer.org/"},
)

QUALITY = {
    "preview": {
        "max_side": 64,
        "iterations_ad": 12,
        "spp": 16,
        "validation_spp": 64,
        "control_spacing": 2,
        "material_refit_interval": 4,
        "learning_rate": 0.018,
    },
    "standard": {
        "max_side": 128,
        "iterations_ad": 24,
        "spp": 16,
        "validation_spp": 128,
        "control_spacing": 2,
        "material_refit_interval": 4,
        "learning_rate": 0.012,
    },
    "research": {
        "max_side": 256,
        "iterations_ad": 50,
        "spp": 32,
        "validation_spp": 256,
        "control_spacing": 2,
        "material_refit_interval": 5,
        "learning_rate": 0.008,
    },
    "ultra": {
        "max_side": 1024,
        "iterations_ad": 75,
        "spp": 32,
        "validation_spp": 256,
        "control_spacing": 1,
        "material_refit_interval": 5,
        "learning_rate": 0.004,
        "geometry_mode": "absolute_height",
        "curvature_strength": 0.002,
        "datum_strength": 1.0,
        "absolute_height_limit": 2.0,
    },
}

# Differentiable path records, rather than the height array itself, dominate
# GPU memory at the optimization resolution. Ultra caps the paths in one stochastic
# optimization render. Independent seeds across 75 iterations still accumulate
# samples, while fixed-seed validation retains its separate higher sample count.
ULTRA_AD_SAMPLE_BUDGET = 8_000_000
# Dr.Jit-Core 1.3.1 rounds the block length of a CUDA reduction to the
# next uint32 power of two. A monolithic reduction longer than 2^31 wraps
# that intermediate to zero and terminates the worker with 0xc0000094.
# Hierarchical power-of-two blocks are mathematically the same sum and keep
# every optimization-grid pixel in the objective.
DRJIT_REDUCTION_CHUNK = 1 << 20
DRJIT_CORE_PATCH_ID = "drjit-core-1.3.1-large-cuda-reductions-v1"

_DLL_DIRECTORY_HANDLES: list[Any] = []


def runtime_root_path() -> Path:
    runtime_root = Path(sys.executable).resolve().parent
    if runtime_root.name.lower() == "scripts":
        runtime_root = runtime_root.parent
    return runtime_root


def configure_runtime_paths() -> None:
    if os.name != "nt":
        return
    runtime_root = runtime_root_path()
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
        self.live: dict[str, Any] = {}

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
        if self.live:
            text += json.dumps(self.live, allow_nan=False) + "\n"
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
        import drjit as dr
        if os.name == "nt" and dr.detail.llvm_version()[0] != 15:
            raise RuntimeError("Windows CPU inverse rendering requires the packaged LLVM 15 runtime. Reinstall the current what-a-relief Mitsuba backend; older LLVM 18 packages have native JIT failures.")
        dr.set_thread_count(min(8, max(1, (os.cpu_count() or 1) // 2)))
        return mi, "llvm_ad_rgb", "adam_projective_autodiff"
    raise ValueError(f"Unsupported backend selection: {name}")


def tiny_probe(mi, backend: str) -> None:
    gradient, _ = shadow_gradient_probe(mi, "near_field_ring", side=16, spp=32)
    if not np.all(np.isfinite(gradient)) or float(np.mean(np.abs(gradient))) <= 1.0e-5:
        raise RuntimeError("Mitsuba projective probe returned no finite moving-shadow derivative")


def shadow_gradient_probe(mi, lighting_model: str, side: int = 40, spp: int = 128, finite_difference: bool = False):
    """Only the blocker's shadow is visible, so a material gradient cannot pass."""
    import drjit as dr

    transform = mi.ScalarTransform4f
    geometry = {
        "lighting_model": lighting_model, "ring_radius_world": 3.0,
        "ring_height_world": 5.0, "led_diameter_world": 0.2,
        "directional_distance_world": 100.0, "angular_diameter_degrees": 2.0,
        "fallback_azimuth": math.pi,
    }
    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "direct_projective"},
            "sensor": {
                "type": "perspective",
                "to_world": transform().look_at(
                    origin=[0.0, 0.0, 10.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0]
                ),
                "fov": 8.0,
                "fov_axis": "x",
                "sampler": {"type": "independent", "sample_count": spp},
                "film": {
                    "type": "hdrfilm",
                    "width": side,
                    "height": side,
                    "pixel_format": "rgb",
                    "component_format": "float32",
                    "sample_border": True,
                    "rfilter": {"type": "box"},
                },
            },
            "surface": {
                "type": "rectangle",
                "to_world": transform().scale(10),
                "bsdf": {"type": "diffuse", "reflectance": 0.5},
            },
            "blocker": {
                "type": "sphere", "to_world": transform().translate([-1.5, 0, 2.5]).scale(0.22),
                "bsdf": {"type": "diffuse", "reflectance": 0.5},
            },
            "emitter": finite_emitter(mi, np.array([-0.6, 0, 1.0]), geometry),
        }
    )
    params = mi.traverse(scene)
    z = mi.Float(2.5)
    dr.enable_grad(z)
    params["blocker.to_world"] = mi.Transform4f().translate(mi.Vector3f(-1.5, 0, z)).scale(0.22)
    params.update()
    image = mi.render(scene, params, spp=spp, spp_grad=spp, seed=812)
    dr.forward(z)
    gradient = np.array(dr.grad(image), copy=True)
    numerical = None
    if finite_difference:
        pair = []
        for height in (2.48, 2.52):
            params["blocker.to_world"] = mi.Transform4f().translate([-1.5, 0, height]).scale(0.22)
            params.update()
            pair.append(np.array(mi.render(scene, params, spp=4 * spp, seed=93), copy=True))
        numerical = (pair[1] - pair[0]) / 0.04
    return gradient, numerical


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
                "llvm_version": list(dr.detail.llvm_version()) if backend == "llvm" else None,
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
    ys = np.clip((np.arange(new_height, dtype=np.float32) + 0.5) * old_height / new_height - 0.5, 0, old_height - 1)
    xs = np.clip((np.arange(new_width, dtype=np.float32) + 0.5) * old_width / new_width - 0.5, 0, old_width - 1)
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


def resize_area(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """Integrate source pixel footprints; no new imaging dependency is needed."""
    result = np.asarray(image, dtype=np.float32)
    for axis, count in ((-2, new_height), (-1, new_width)):
        old = result.shape[axis]
        if count == old:
            continue
        if count > old:
            return resize_bilinear(result, new_height, new_width)
        values = np.moveaxis(result, axis, -1)
        cumulative = np.concatenate((np.zeros_like(values[..., :1]), np.cumsum(values, axis=-1, dtype=np.float64)), axis=-1)
        edges = np.linspace(0, old, count + 1)
        index = np.minimum(edges.astype(np.int64), old - 1)
        integrals = cumulative[..., index] + values[..., index] * (edges - index)
        result = np.moveaxis(np.diff(integrals, axis=-1) / (old / count), -1, axis).astype(np.float32)
    return result


def masked_resize(image: np.ndarray, mask: np.ndarray, height: int, width: int) -> np.ndarray:
    coverage = resize_area(mask.astype(np.float32), height, width)
    total = resize_area(np.where(mask, image, 0.0), height, width)
    return np.divide(total, coverage, out=np.zeros_like(total), where=coverage > 1.0e-6)


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
    if maximum <= 0:
        return height, width
    scale = min(1.0, maximum / float(max(height, width)))
    return max(16, int(round(height * scale))), max(16, int(round(width * scale)))


def effective_ultra_spp(pixel_count: int, requested_spp: int) -> int:
    return min(
        requested_spp,
        max(1, ULTRA_AD_SAMPLE_BUDGET // max(1, pixel_count)),
    )


def effective_ultra_learning_rate(world_spacing: tuple[float, float], requested: float) -> float:
    # Adam's first update is approximately the learning rate at every vertex,
    # independent of gradient magnitude. A fixed scene-space step therefore
    # creates extreme pixel-to-pixel curvature when the native pitch is small.
    return min(requested, 0.02 * min(world_spacing))


def safe_dr_sum(dr, value):
    """Sum a Dr.Jit value without a >2^31-element CUDA block reduction."""
    partials = dr.ravel(value)
    while len(partials) > DRJIT_REDUCTION_CHUNK:
        partials = dr.block_sum(partials, DRJIT_REDUCTION_CHUNK)
    return dr.sum(partials)


def drjit_core_patch_status(runtime_root: Path | None = None) -> dict[str, Any]:
    """Verify the native large-reduction fix rather than trusting a marker alone."""
    root = runtime_root.resolve() if runtime_root is not None else runtime_root_path()
    marker_path = root / "drjit-core-patch.json"
    library_path = root / "Lib" / "site-packages" / "drjit" / "drjit-core.dll"
    status: dict[str, Any] = {
        "required_id": DRJIT_CORE_PATCH_ID,
        "marker_found": marker_path.is_file(),
        "library_found": library_path.is_file(),
        "valid": False,
    }
    if not status["marker_found"] or not status["library_found"]:
        return status
    try:
        record = json.loads(marker_path.read_text(encoding="utf-8-sig"))
        expected_hash = str(record.get("library_sha256", "")).lower()
        digest = hashlib.sha256()
        with library_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        status.update(
            {
                "id": record.get("id"),
                "source_commit": record.get("source_commit"),
                "library_sha256": actual_hash,
                "valid": (
                    record.get("id") == DRJIT_CORE_PATCH_ID
                    and len(expected_hash) == 64
                    and actual_hash == expected_hash
                ),
            }
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        status["verification_error"] = str(error)
    return status


def normalized_height(height: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    values = height[mask]
    datum = float(np.median(values))
    result = height.astype(np.float32) - datum
    result[~mask] = 0.0
    return result, datum


def full_height_solution(prepared, optimized_world: np.ndarray, absolute_height: bool):
    """Return a full-size candidate and its signed difference from the classical baseline."""
    baseline = prepared["height"]
    mask = prepared["mask"]
    x0, y0, x1, y1 = prepared["bounds"]
    conversion = prepared["scene_scale"] * prepared["physical_scale"]
    if not math.isfinite(conversion) or abs(conversion) <= 1.0e-12:
        raise ValueError("Invalid scene-to-height conversion")
    if absolute_height:
        candidate_crop = resize_bilinear(
            optimized_world / conversion + prepared["height_datum"], y1 - y0, x1 - x0)
        candidate = np.where(mask, baseline, 0.0).astype(np.float32)
        fit_mask = prepared.get("fit_mask")
        if fit_mask is None:
            fit_mask = mask[y0:y1, x0:x1]
        else:
            fit_mask = np.asarray(fit_mask, dtype=bool)
            if fit_mask.ndim != 2:
                raise ValueError("Absolute-height fitting mask must be two-dimensional")
            if fit_mask.shape != candidate_crop.shape:
                fit_mask = resize_nearest(
                    fit_mask.astype(np.float32), y1 - y0, x1 - x0) >= 0.5
        fit_mask &= mask[y0:y1, x0:x1]
        crop = candidate[y0:y1, x0:x1]
        crop[fit_mask] = candidate_crop[fit_mask]
        candidate[~mask] = 0.0
        difference = candidate - baseline
    else:
        difference = np.zeros_like(baseline, dtype=np.float32)
        difference[y0:y1, x0:x1] = resize_bilinear(
            optimized_world / conversion, y1 - y0, x1 - x0)
        difference[~mask] = 0.0
        candidate = baseline + difference
        candidate[~mask] = 0.0
    difference[~mask] = 0.0
    return candidate.astype(np.float32), difference.astype(np.float32)


def masked_slopes(height: np.ndarray, mask: np.ndarray, spacing_y=1.0, spacing_x=1.0):
    slopes = []
    for axis, spacing in ((0, spacing_y), (1, spacing_x)):
        previous = np.roll(height, 1, axis=axis)
        following = np.roll(height, -1, axis=axis)
        before = np.roll(mask, 1, axis=axis) & mask
        after = np.roll(mask, -1, axis=axis) & mask
        edge = [slice(None)] * 2
        edge[axis] = 0
        before[tuple(edge)] = False
        edge[axis] = -1
        after[tuple(edge)] = False
        difference = np.where(after, following - height, 0.0) + np.where(before, height - previous, 0.0)
        steps = before.astype(np.float32) + after.astype(np.float32)
        slopes.append(np.divide(difference, steps * spacing, out=np.zeros_like(height), where=steps > 0))
    return tuple(slopes)


def surface_normals(height: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = np.ones_like(height, dtype=bool)
    q, p = masked_slopes(height.astype(np.float32), mask)
    normals = np.stack((-p, q, np.ones_like(height)), axis=-1)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(length, 1.0e-8)


def slope_stencils(mask: np.ndarray, spacing_y: float, spacing_x: float):
    center = np.arange(mask.size, dtype=np.uint32).reshape(mask.shape)
    stencils = []
    for axis, spacing in ((0, spacing_y), (1, spacing_x)):
        before = np.roll(mask, 1, axis=axis) & mask
        after = np.roll(mask, -1, axis=axis) & mask
        edge = [slice(None), slice(None)]
        edge[axis] = 0
        before[tuple(edge)] = False
        edge[axis] = -1
        after[tuple(edge)] = False
        left = np.where(before, np.roll(center, 1, axis=axis), center)
        right = np.where(after, np.roll(center, -1, axis=axis), center)
        steps = before.astype(np.float32) + after.astype(np.float32)
        scale = np.divide(1.0, steps * spacing, out=np.zeros_like(steps), where=steps > 0)
        stencils.append((left.ravel(), right.ravel(), scale.ravel()))
    return stencils


def curvature_stencils(mask: np.ndarray, spacing_y: float, spacing_x: float):
    center = np.arange(mask.size, dtype=np.uint32).reshape(mask.shape)
    stencils = []
    for axis, spacing in ((0, spacing_y), (1, spacing_x)):
        before = np.roll(mask, 1, axis=axis)
        after = np.roll(mask, -1, axis=axis)
        edge = [slice(None), slice(None)]
        edge[axis] = 0
        before[tuple(edge)] = False
        edge[axis] = -1
        after[tuple(edge)] = False
        valid = mask & before & after
        left = np.where(valid, np.roll(center, 1, axis=axis), center)
        right = np.where(valid, np.roll(center, -1, axis=axis), center)
        scale = np.where(valid, 1.0 / (spacing * spacing), 0.0).astype(np.float32)
        stencils.append((left.ravel(), center.ravel(), right.ravel(), scale.ravel(),
                         valid.astype(np.float32).ravel()))
    return stencils


def triangle_vertex_support(mask: np.ndarray) -> np.ndarray:
    """Vertices belonging to at least one triangle emitted by write_grid_ply."""
    support = np.zeros_like(mask, dtype=bool)
    first = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:]
    second = mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    support[:-1, :-1] |= first
    support[1:, :-1] |= first | second
    support[:-1, 1:] |= first | second
    support[1:, 1:] |= second
    return support


def absolute_height_regularization_numpy(prepared, height_world: np.ndarray, settings) -> float:
    return absolute_height_regularization_gradient_numpy(
        prepared, height_world, settings)[0]


def absolute_height_regularization_gradient_numpy(
        prepared, height_world: np.ndarray, settings) -> tuple[float, float, np.ndarray]:
    """Evaluate Ultra's priors and exact gradient without dense AD stencils."""
    mask = prepared.get("fit_mask", prepared["mask_small"])
    count = max(float(mask.sum()), 1.0)
    baseline = prepared["baseline_positions"][..., 2]
    datum_offset = float(np.sum((height_world - baseline) * mask) / count)
    datum_strength = float(settings.get("datum_strength", 1.0))
    curvature_strength = float(settings.get("curvature_strength", 0.002))
    gradient = (2.0 * datum_strength * datum_offset / count
                * mask.astype(np.float32))
    loss = datum_strength * datum_offset * datum_offset

    spacing_y, spacing_x = prepared["world_spacing"]
    for axis, spacing in ((0, spacing_y), (1, spacing_x)):
        if height_world.shape[axis] < 3:
            continue
        before = [slice(None), slice(None)]
        center = [slice(None), slice(None)]
        after = [slice(None), slice(None)]
        before[axis] = slice(None, -2)
        center[axis] = slice(1, -1)
        after[axis] = slice(2, None)
        before = tuple(before)
        center = tuple(center)
        after = tuple(after)
        valid = mask[before] & mask[center] & mask[after]
        scale = 1.0 / (spacing * spacing)
        second = ((height_world[before] - 2.0 * height_world[center]
                   + height_world[after]) * scale)
        second = np.where(valid, second, 0.0)
        loss += curvature_strength * float(np.sum(second * second) / count)
        contribution = (2.0 * curvature_strength * second * scale / count).astype(np.float32)
        gradient[before] += contribution
        gradient[center] -= 2.0 * contribution
        gradient[after] += contribution

    prior_loss = 0.0
    prior_weight = prepared.get("normal_prior_weight")
    if prior_weight is None:
        prior_weight = np.zeros_like(height_world, dtype=np.float32)
    if np.any(prior_weight > 0):
        q, p = masked_slopes(height_world, prepared["mask_small"], spacing_y, spacing_x)
        inverse_length = 1.0 / np.sqrt(1.0 + p * p + q * q)
        normal = np.stack((-p * inverse_length, q * inverse_length, inverse_length), axis=-1)
        error = normal - prepared["normal_prior"]
        prior_denominator = max(float(prior_weight.sum()), 1.0e-8)
        prior_loss = float(np.sum(np.sum(error * error, axis=-1) * prior_weight)
                           / prior_denominator)

        # d ||normalize(v)-target||^2 / dv, where v=(-p, q, 1).
        tangent = error - normal * np.sum(normal * error, axis=-1, keepdims=True)
        derivative_v = (2.0 * tangent * inverse_length[..., None]
                        * prior_weight[..., None] / prior_denominator)
        derivative_p = -NORMAL_PRIOR_STRENGTH * derivative_v[..., 0]
        derivative_q = NORMAL_PRIOR_STRENGTH * derivative_v[..., 1]

        for axis, spacing, derivative in (
                (1, spacing_x, derivative_p), (0, spacing_y, derivative_q)):
            before_valid = (np.roll(prepared["mask_small"], 1, axis=axis)
                            & prepared["mask_small"])
            after_valid = (np.roll(prepared["mask_small"], -1, axis=axis)
                           & prepared["mask_small"])
            edge = [slice(None), slice(None)]
            edge[axis] = 0
            before_valid[tuple(edge)] = False
            edge[axis] = -1
            after_valid[tuple(edge)] = False
            steps = before_valid.astype(np.float32) + after_valid.astype(np.float32)
            slope_scale = np.divide(
                1.0, steps * spacing, out=np.zeros_like(steps), where=steps > 0)
            term = derivative * slope_scale
            gradient += term * (before_valid.astype(np.float32) - after_valid.astype(np.float32))
            if axis == 1:
                gradient[:, 1:] += (term * after_valid)[:, :-1]
                gradient[:, :-1] -= (term * before_valid)[:, 1:]
            else:
                gradient[1:, :] += (term * after_valid)[:-1, :]
                gradient[:-1, :] -= (term * before_valid)[1:, :]

    gradient[~mask] = 0.0
    return float(loss), prior_loss, gradient.astype(np.float32, copy=False)


def load_normal_prior(inputs, mask, confidence, bounds, shape):
    paths = inputs.get("normal_prior_pfm")
    if paths is None:
        return np.zeros((*shape, 3), np.float32), np.zeros(shape, np.float32)
    if not isinstance(paths, list) or len(paths) != 3:
        raise ValueError("Normal prior requires three full-precision XYZ component maps")
    z = read_pfm(Path(paths[2]))
    if z.shape != mask.shape:
        raise ValueError("Normal prior dimensions differ from height")
    supported = mask & np.isfinite(z) & (z > 0.15)
    weight = np.where(supported, np.clip(confidence, 0, 1) * np.clip(z, 0, 1)**2, 0)
    x0, y0, x1, y1 = bounds
    crop = np.s_[y0:y1, x0:x1]
    reduced_weight = resize_area(weight[crop], *shape)
    prior = np.zeros((*shape, 3), np.float32)
    for axis in (2, 0, 1):
        component = z if axis == 2 else read_pfm(Path(paths[axis]))
        if component.shape != mask.shape or np.any(~np.isfinite(component[weight > 0])) or np.any(np.abs(component[weight > 0]) > 1.0001):
            raise ValueError("Invalid supported normal-prior component")
        weighted = np.where(weight > 0, component, 0) * weight
        prior[..., axis] = np.divide(resize_area(weighted[crop], *shape), reduced_weight,
                                    out=np.zeros(shape, np.float32), where=reduced_weight > 0)
    length = np.linalg.norm(prior, axis=-1)
    prior /= np.maximum(length[..., None], 1e-8)
    reduced_weight[length < 1e-6] = 0
    return prior, reduced_weight


def normal_prior_loss(prepared, height_world):
    weight = prepared["normal_prior_weight"]
    if not np.any(weight > 0):
        return 0.0
    spacing_y, spacing_x = prepared["world_spacing"]
    q, p = masked_slopes(height_world, prepared["mask_small"], spacing_y, spacing_x)
    n = np.stack((-p, q, np.ones_like(p)), axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return float(np.sum(np.sum((n - prepared["normal_prior"])**2, axis=-1) * weight) / weight.sum())


def normal_prior_loss_ad(mi, dr, z, prior, weight, stencils, denominator):
    q, p = [(dr.gather(mi.Float, z, b) - dr.gather(mi.Float, z, a)) * scale
            for a, b, scale in stencils]
    inv_length = dr.rsqrt(1.0 + p * p + q * q)
    error = ((-p * inv_length - prior[0])**2
             + (q * inv_length - prior[1])**2 + (inv_length - prior[2])**2)
    return safe_dr_sum(dr, error * weight) / denominator


def control_mapping(height: int, width: int, spacing: int):
    control_height = max(3, int(math.ceil((height - 1) / spacing)) + 1)
    control_width = max(3, int(math.ceil((width - 1) / spacing)) + 1)
    if spacing == 1 and control_height == height and control_width == width:
        # The native parameterization is already an identity map. Four dense
        # index planes plus four weight planes only duplicate the whole image.
        return control_height, control_width, None, None
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
    if indices is None:
        return control.ravel()
    flat = control.ravel()
    expanded = sum(flat[index] * weight for index, weight in zip(indices, weights))
    return expanded


def write_grid_ply(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> None:
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
    faces = faces[np.all(mask.ravel()[faces], axis=1)]
    if not len(faces):
        raise ValueError("Inverse mask contains no supported triangles")
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


def write_masked_ply(path: Path, height: np.ndarray, mask: np.ndarray, albedo: np.ndarray, z_scale: float, unvalidated: bool = False) -> None:
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
    validation_comment = "comment UNVALIDATED CANDIDATE - automatic validation failed; see candidate.json\n" if unvalidated else ""
    with path.open("wb") as stream:
        stream.write(
            (
                "ply\nformat binary_little_endian 1.0\n"
                "comment generated by what-a-relief Mitsuba inverse backend\n"
                f"{validation_comment}"
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


def finite_emitter(mi, light: np.ndarray, geometry: dict[str, Any]):
    if geometry["lighting_model"] == "near_field_ring":
        radial = math.hypot(float(light[0]), float(light[1]))
        if radial <= 1.0e-8:
            angle = geometry["fallback_azimuth"]
            ax, ay = math.cos(angle), math.sin(angle)
        else:
            ax, ay = float(light[0]) / radial, float(light[1]) / radial
        position = np.array([geometry["ring_radius_world"] * ax, geometry["ring_radius_world"] * ay, geometry["ring_height_world"]])
        radius = 0.5 * geometry["led_diameter_world"]
    else:
        direction = light / np.linalg.norm(light)
        distance = geometry["directional_distance_world"]
        position = direction * distance
        radius = distance * math.tan(math.radians(geometry["angular_diameter_degrees"] * 0.5))
    if not math.isfinite(radius) or radius <= 0 or not np.all(np.isfinite(position)) or np.linalg.norm(position) <= 1.0e-8:
        raise ValueError("Projective inverse rendering requires a positive finite emitter size")
    # A disk facing the reference origin. Its radiance gives pi irradiance on
    # a perpendicular plane at that origin: E = pi * L * r^2 / (d^2 + r^2).
    direction = -position / np.linalg.norm(position)
    up = [1.0, 0.0, 0.0] if abs(direction[1]) > 0.95 else [0.0, 1.0, 0.0]
    transform = mi.ScalarTransform4f().look_at(origin=position.tolist(), target=[0, 0, 0], up=up).scale(radius)
    return {
        "type": "disk", "to_world": transform,
        "emitter": {"type": "area", "radiance": float((position @ position + radius * radius) / (radius * radius))},
    }


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
    emitter = finite_emitter(mi, light, geometry)
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
                "sample_border": True,
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


def update_scene_parameters(mi, record, positions):
    params, key, emitter = record
    params[key] = positions
    params["emitter.to_world"] = emitter["to_world"]
    params["emitter.emitter.radiance.value"] = mi.Color3f(emitter["emitter"]["radiance"])
    params.update()
    return params


def render_numpy(mi, scenes, parameters, positions: np.ndarray, spp: int, seed: int) -> np.ndarray:
    rendered = []
    flattened = mi.Float(positions.astype(np.float32).ravel())
    for index, (scene, record) in enumerate(zip(scenes, parameters)):
        params = update_scene_parameters(mi, record, flattened)
        image = np.asarray(mi.render(scene, params, spp=spp, seed=seed + index))
        rendered.append(image.mean(axis=2).astype(np.float32))
    return np.stack(rendered, axis=0)


def fit_material_maps(basis: np.ndarray, observed: np.ndarray, mask: np.ndarray, albedo_prior: np.ndarray, train: list[int], weights: np.ndarray | None = None):
    # basis has shape [basis, light, y, x]. Solve three nonnegative coefficients
    # per pixel with a weak prior on classical diffuse albedo.
    x = np.moveaxis(basis[:, train], 0, -1)  # [light, y, x, basis]
    x = np.moveaxis(x, 0, -2)  # [y, x, light, basis]
    y = np.moveaxis(observed[train], 0, -1)[..., None]  # [y, x, light, 1]
    if weights is None:
        weights = np.broadcast_to(mask, observed.shape).astype(np.float32)
    selected = np.moveaxis(weights[train], 0, -1)[..., None]
    xt = np.swapaxes(x * selected, -1, -2)
    normal = xt @ x
    rhs = (xt @ y)[..., 0]
    ridge = np.array([0.08, 0.14, 0.14], dtype=np.float32)
    normal += np.eye(3, dtype=np.float32) * ridge
    rhs[..., 0] += ridge[0] * albedo_prior
    valid_values = observed[train][weights[train] > 0]
    if not valid_values.size:
        raise ValueError("No valid training observations for inverse material fitting")
    specular_limit = max(0.25, float(np.quantile(valid_values, 0.995)) * 3.0)
    upper = np.array([max(2.0, float(np.quantile(albedo_prior[mask], 0.995)) * 2.5), specular_limit, specular_limit])
    coefficients = bounded_material_solve(normal, rhs, upper)
    coefficients[~mask] = 0.0
    return coefficients, material_roughness(coefficients)


def bounded_material_solve(normal: np.ndarray, rhs: np.ndarray, upper: np.ndarray) -> np.ndarray:
    # Three coefficients permit exact enumeration of lower/free/upper active
    # sets. Refit free variables after binding any correlated coefficient.
    best = np.zeros_like(rhs)
    best_loss = np.full(rhs.shape[:-1], np.inf)
    for code in range(27):
        states = [(code // 3**i) % 3 for i in range(3)]
        free = [i for i in range(3) if states[i] == 1]
        candidate = np.zeros_like(rhs)
        for i in range(3):
            if states[i] == 2:
                candidate[..., i] = upper[i]
        if free:
            adjusted = rhs - (normal @ candidate[..., None])[..., 0]
            block = normal[..., free, :][..., :, free]
            candidate[..., free] = np.linalg.solve(block, adjusted[..., free, None])[..., 0]
        feasible = np.all((candidate >= 0) & (candidate <= upper), axis=-1)
        loss = np.sum(candidate * ((normal @ candidate[..., None])[..., 0] - 2 * rhs), axis=-1)
        improve = feasible & (loss < best_loss)
        best[improve] = candidate[improve]
        best_loss[improve] = loss[improve]
    return best.astype(np.float32)


def prediction_numpy(basis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.sum(basis * np.moveaxis(coefficients, -1, 0)[:, None, :, :], axis=0)


def material_roughness(coefficients: np.ndarray) -> np.ndarray:
    specular = coefficients[..., 1] + coefficients[..., 2]
    return np.where(specular > 1.0e-6,
                    (0.075 * coefficients[..., 1] + 0.32 * coefficients[..., 2])
                    / np.maximum(specular, 1.0e-6), 0.32).astype(np.float32)


def data_loss_numpy(prediction: np.ndarray, observed: np.ndarray, weights: np.ndarray, indices: list[int]) -> float:
    residual = prediction[indices] - observed[indices]
    robust = np.sqrt(residual * residual + 0.0025**2) - 0.0025
    selected = weights[indices]
    denominator = float(selected.sum())
    return float((robust * selected).sum() / max(denominator, 1.0e-8))


def refit_training_material(basis, observed, mask, albedo, train, weights, coefficients):
    refitted, _ = fit_material_maps(basis, observed, mask, albedo, train, weights)
    current_loss = data_loss_numpy(prediction_numpy(basis, coefficients), observed, weights, train)
    refitted_loss = data_loss_numpy(prediction_numpy(basis, refitted), observed, weights, train)
    # The bounded least-squares material fit and Charbonnier geometry loss
    # differ. Retain the current material if refitting increases the latter.
    if math.isfinite(refitted_loss) and refitted_loss < current_loss:
        return refitted, refitted_loss
    return coefficients.copy(), current_loss


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


def iteration_preview(height_world, mask, spacing):
    """Fixed-scale geometry display, not a render or an accepted reconstruction."""
    q, p = masked_slopes(height_world, mask, *spacing)
    normal = np.stack((-p, q, np.ones_like(p)), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    rgb = normal * 0.5 + 0.5
    light = np.array([-0.5, 0.5, 2**-0.5], dtype=np.float32)
    shade = 0.15 + 0.85 * np.maximum(0, normal @ light)
    hillshade = np.repeat(shade[..., None], 3, axis=2)
    rgb[~mask] = 0.12
    hillshade[~mask] = 0.12
    return np.concatenate((rgb, hillshade), axis=1)


def fit_preview_frame(frame: np.ndarray, maximum_height: int = 512,
                      maximum_width: int = 1024) -> np.ndarray:
    """Downsample a diagnostic RGB frame to the GUI's guarded decode limits."""
    values = np.asarray(frame, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Iteration preview must be a non-empty H-by-W-by-3 image")
    if maximum_height < 1 or maximum_width < 1:
        raise ValueError("Iteration preview limits must be positive")
    height, width = values.shape[:2]
    scale = min(1.0, maximum_height / height, maximum_width / width)
    if scale >= 1.0:
        return values.copy()
    new_height = max(1, int(np.floor(height * scale)))
    new_width = max(1, int(np.floor(width * scale)))
    channels_first = np.moveaxis(values, -1, 0)
    resized = resize_bilinear(channels_first, new_height, new_width)
    return np.moveaxis(resized, 0, -1)


def publish_iteration(mi, progress, prepared, control, mapping, iteration, total, began, absolute_height=False):
    if not prepared.get("live_preview", False):
        return
    elapsed = max(0.0, time.monotonic() - began)
    progress.live.update(iteration=iteration, total=total,
                         remaining_seconds=elapsed * (total - iteration) / max(iteration, 1))
    if iteration % 5 != 0 and iteration != total:
        return
    try:
        expanded = expand_control_numpy(control, mapping).reshape(prepared["height_small"].shape)
        height_world = expanded if absolute_height else prepared["baseline_positions"][..., 2] + expanded
        frame = iteration_preview(height_world,
                                  prepared["mask_small"], prepared["world_spacing"])
        frame = fit_preview_frame(frame)
        save_rgb(mi, progress.path.parent / f"iteration_{iteration}.png", frame)
        progress.live["preview_iteration"] = iteration
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Iteration preview unavailable: {error}", flush=True)


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


def output_products(mi, output: Path, baseline: np.ndarray, correction_or_height: np.ndarray,
                    mask: np.ndarray, albedo: np.ndarray, height_scale: float,
                    unvalidated: bool = False, absolute_height: bool = False) -> None:
    prefix = "candidate" if unvalidated else "inverse"
    final_height = correction_or_height.copy() if absolute_height else baseline + correction_or_height
    final_height[~mask] = 0.0
    correction = final_height - baseline if absolute_height else correction_or_height
    correction = correction.copy()
    correction[~mask] = 0.0
    normals = surface_normals(final_height, mask)
    normals[~mask] = 0.0
    write_pfm(output / f"{prefix}_height.pfm", final_height)
    write_pfm(output / "height_correction.pfm", correction)
    save_gray(mi, output / f"{prefix}_height.png", display_stretch(final_height, mask))
    save_gray(mi, output / "height_correction.png", display_stretch(correction, mask, symmetric=True))
    encoded = normals * 0.5 + 0.5
    encoded[~mask] = 0.0
    save_rgb(mi, output / f"{prefix}_normal_rgb.png", encoded)
    save_gray(mi, output / f"{prefix}_normal_x.png", np.where(mask, normals[..., 0] * 0.5 + 0.5, 0.0))
    save_gray(mi, output / f"{prefix}_normal_y.png", np.where(mask, normals[..., 1] * 0.5 + 0.5, 0.0))
    save_gray(mi, output / f"{prefix}_normal_z.png", np.where(mask, np.maximum(normals[..., 2], 0.0), 0.0))
    light = np.array([-0.5, 0.5, math.sqrt(0.5)], dtype=np.float32)
    hillshade = np.maximum(0.0, np.sum(normals * light, axis=-1))
    save_gray(mi, output / f"{prefix}_hillshade_ul.png", np.where(mask, 0.14 + 0.86 * hillshade, 0.0))
    write_masked_ply(output / f"{prefix}_surface.ply", final_height, mask, albedo, height_scale, unvalidated)


def comparison_stretch(before: np.ndarray, after: np.ndarray, mask: np.ndarray):
    # One range for both images; independent percentile stretches hide shape changes.
    ranges = [np.quantile(v[mask & np.isfinite(v)], [0.005, 0.995]) for v in (before, after)]
    low = float(min(r[0] for r in ranges))
    high = float(max(r[1] for r in ranges))
    images = [np.where(mask, np.clip((v - low) / max(high - low, 1e-8), 0, 1), 0).astype(np.float32)
              for v in (before, after)]
    return images, [low, high]


def write_candidate_review(directory: Path, result: dict[str, Any]) -> None:
    def metric(name):
        if name.startswith("normal_prior_") and result.get("normal_prior_enabled") is False:
            return "Not used"
        value = result.get(name)
        return f"{value:.6g}" if isinstance(value, (float, int)) and math.isfinite(value) else "Not evaluated"

    rows = "".join(f"<tr><th>{label}</th><td>{metric(before)}</td><td>{metric(after)}</td></tr>"
                   for label, before, after in (
                       ("Training image loss", "train_loss_before", "train_loss_after"),
                       ("Withheld image loss", "holdout_loss_before", "holdout_loss_after"),
                       ("Photometric-normal prior loss", "normal_prior_loss_before", "normal_prior_loss_after")))
    decision = html.escape(str(result["decision"]))
    if result.get("geometry_parameterization") == "absolute_height_grid":
        geometry_note = (
            "The Ultra candidate is an absolute-height field optimized on a grid capped at 1024 pixels "
            "on its longer side. The classical height is its initialization and comparison baseline, "
            "not an additive output layer; larger accepted results are interpolated back to the source grid."
        )
    else:
        geometry_note = (
            "The full-resolution height retains baseline detail plus the upsampled correction."
        )
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>what-a-relief | Unvalidated candidate</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:#202628;font:16px/1.5 system-ui,sans-serif;letter-spacing:0}
main{max-width:1400px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 12px}h2{font-size:18px;margin:12px 0}
.warning{border-left:5px solid #b12d48;background:#fff0f3;padding:12px 16px;margin:16px 0}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}
button,select{font:inherit;min-height:42px;padding:6px 12px;border:1px solid #8b9599;border-radius:4px;background:white;color:inherit}
button{cursor:pointer}button[aria-pressed=true]{background:#12675e;color:white;border-color:#12675e}
.comparison{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.comparison.single{grid-template-columns:minmax(0,1fr)}
figure{margin:0;min-width:0}figcaption{padding:8px 0;font-weight:600}img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#111}
[hidden]{display:none!important}table{border-collapse:collapse;width:100%;max-width:800px}th,td{text-align:left;padding:8px;border-bottom:1px solid #b9c4c8}
a{color:#076a98}code{overflow-wrap:anywhere}.files{display:flex;gap:20px;flex-wrap:wrap}p{max-width:1000px}
@media(max-width:700px){main{padding:16px}.comparison{grid-template-columns:minmax(0,1fr)}th,td{padding:6px;font-size:14px}}
</style></head><body><main>
<h1>what-a-relief | Unvalidated candidate</h1>
<div class="warning"><strong>Automatic validation failed. Baseline retained.</strong>
<p>Reason: <code>__DECISION__</code></p>
<p>This candidate is for investigation, not a confirmed improvement. Viewing or opening it does not accept it or replace the default outputs.</p></div>
<table><thead><tr><th>Lower is better, not ground-truth accuracy</th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>__ROWS__</tbody></table>
<p>The withheld images were excluded from inverse optimization, but were used by the original photometric solve.
The normal prior measures agreement with that solve, not accuracy against measured normals.</p>
<div class="controls"><label for="product">Compare</label><select id="product">
<option value="height">Height (shared display range)</option><option value="normal_rgb">Height-derived RGB normals</option>
<option value="normal_x">Height-derived X normals</option><option value="hillshade_ul">Hillshade</option>
<option value="render">Mean training render (shared display range)</option></select>
<button id="both" type="button" aria-pressed="true">Side by side</button>
<button id="baseline" type="button" aria-pressed="false">Baseline only</button>
<button id="candidate" type="button" aria-pressed="false">Unvalidated candidate only</button></div>
<div class="comparison" id="comparison"><figure id="baselineFigure"><figcaption>Baseline (retained)</figcaption>
<img id="baselineImage" src="review_baseline_height.png" alt="Baseline height"></figure>
<figure id="candidateFigure"><figcaption>Unvalidated candidate</figcaption>
<img id="candidateImage" src="review_candidate_height.png" alt="Unvalidated candidate height"></figure></div>
<p>Normals here are derived from the two height fields, not the original photometric normal maps.
__GEOMETRY_NOTE__ PNGs are visualization products, not quantitative height data.</p>
<h2>Candidate files</h2>
<p><label><input type="checkbox" id="acknowledge"> I understand that this candidate failed automatic validation.</label></p>
<div id="files" class="files" hidden><a href="candidate_surface.ply" download>Unvalidated mesh (PLY)</a>
<a href="candidate_height.pfm" download>Unvalidated height (PFM)</a>
<a href="candidate_normal_rgb.png" download>Unvalidated RGB normals (PNG)</a>
<a href="candidate.json" download>Candidate provenance (JSON)</a></div>
<p>Keep <code>candidate.json</code> with any candidate files used for investigation. PFM heights use pixel units;
mesh X/Y use pixels and mesh Z includes the run's height-scale factor. This is a surface mesh, not a watertight printable solid.</p>
<p>Keep the entire inverse folder together to retain all comparison images and relative links.</p>
<p><a href="../result.json">Full validation report</a></p>
<noscript><p>JavaScript is disabled. Candidate files remain in this folder, with their validation status in candidate.json.</p></noscript>
</main><script>
const product=document.getElementById('product');
product.addEventListener('change',()=>{
 const kind=product.value;
 document.getElementById('baselineImage').src=kind==='height'?'review_baseline_height.png':kind==='render'?'../render_before.png':'../inverse_'+kind+'.png';
 document.getElementById('candidateImage').src=kind==='height'?'review_candidate_height.png':kind==='render'?'../render_after.png':'candidate_'+kind+'.png';
 document.getElementById('baselineImage').alt='Baseline '+product.selectedOptions[0].text;
 document.getElementById('candidateImage').alt='Unvalidated candidate '+product.selectedOptions[0].text;
});
for(const mode of ['both','baseline','candidate'])document.getElementById(mode).addEventListener('click',()=>{
 document.getElementById('baselineFigure').hidden=mode==='candidate';
 document.getElementById('candidateFigure').hidden=mode==='baseline';
 document.getElementById('comparison').classList.toggle('single',mode!=='both');
 for(const id of ['both','baseline','candidate'])document.getElementById(id).setAttribute('aria-pressed',String(id===mode));
});
document.getElementById('acknowledge').addEventListener('change',event=>{document.getElementById('files').hidden=!event.target.checked;});
</script></body></html>
"""
    (directory / "review.html").write_text(
        page.replace("__ROWS__", rows).replace("__DECISION__", decision).replace(
            "__GEOMETRY_NOTE__", html.escape(geometry_note)),
        encoding="utf-8")


def retain_unvalidated_candidate(mi, output: Path, baseline: np.ndarray, correction_or_height: np.ndarray,
                                 mask: np.ndarray, albedo: np.ndarray, height_scale: float,
                                 result: dict[str, Any], absolute_height: bool = False) -> None:
    result["candidate_saved"] = False
    result["candidate_directory"] = None
    result["delivered_geometry"] = (
        "accepted_inverse_solution" if result["accepted"] and absolute_height else
        "accepted_refinement" if result["accepted"] else "baseline"
    )
    if result["accepted"]:
        result["candidate_export_status"] = "not_needed_accepted"
        return
    candidate_height = correction_or_height if absolute_height else baseline + correction_or_height
    if (not np.all(np.isfinite(correction_or_height[mask])) or
            not np.all(np.isfinite(candidate_height[mask]))):
        result["candidate_export_status"] = "not_saved_nonfinite_geometry"
        return
    directory = output / "unvalidated_candidate"
    directory.mkdir(parents=True, exist_ok=True)
    output_products(mi, directory, baseline, correction_or_height, mask, albedo, height_scale,
                    unvalidated=True, absolute_height=absolute_height)
    previews, display_range = comparison_stretch(baseline, candidate_height, mask)
    for name, preview in zip(("baseline", "candidate"), previews):
        save_gray(mi, directory / f"review_{name}_height.png", preview)
    candidate_fields = dict(candidate_saved=True, candidate_directory="unvalidated_candidate",
                            candidate_export_status="saved_unvalidated")
    provenance = {**result, **candidate_fields}
    provenance.update(validation_status="unvalidated_candidate", automatically_accepted=False,
                      selected_for_default_outputs=False, review_display_range_pixels=display_range,
                      height_units="pixels", mesh_xy_units="pixels", mesh_z_scale=height_scale,
                      normal_source="candidate_height_gradient", parent_result="../result.json")
    atomic_json(directory / "candidate.json", provenance)
    write_candidate_review(directory, result)
    result.update(candidate_fields)


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
    if not np.all(np.isfinite(lights)) or np.any(np.linalg.norm(lights, axis=1) <= 1.0e-8):
        raise ValueError("Inverse light directions must be finite and nonzero")
    progress(5, "Loading linear image stack and baseline geometry")
    validity_paths = [Path(path) for path in inputs["observation_validity"]]
    if len(validity_paths) != len(image_paths):
        raise ValueError("Inverse handoff requires one validity mask per observation")
    height = read_pfm(Path(inputs["height_pfm"]))
    albedo = read_pfm(Path(inputs["albedo_pfm"]))
    mask = bitmap_array(mi, Path(inputs["mask_png"])) > 0.0
    if height.shape != albedo.shape or mask.shape != height.shape:
        raise ValueError("Mitsuba handoff image, height, albedo, and mask dimensions differ")
    if not np.all(np.isfinite(height[mask])) or not np.all(np.isfinite(albedo[mask])):
        raise ValueError("Supported inverse height and albedo values must be finite")
    raw_robust_weight = np.ones_like(height, dtype=np.float32)
    robust_weight = raw_robust_weight
    robust_path = inputs.get("robust_weight_pfm")
    if robust_path:
        raw_robust_weight = read_pfm(Path(robust_path))
        robust_weight = np.clip(raw_robust_weight, 0.05, 1.0)
        if robust_weight.shape != height.shape or not np.all(np.isfinite(robust_weight[mask])):
            raise ValueError("Inverse confidence dimensions or values are invalid")
    x0, y0, x1, y1 = crop_bounds(mask)
    height_crop = height[y0:y1, x0:x1]
    albedo_crop = albedo[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    weight_crop = robust_weight[y0:y1, x0:x1]
    render_height, render_width = target_size(y1 - y0, x1 - x0, int(settings["max_side"]))
    coverage = resize_area(mask_crop.astype(np.float32), render_height, render_width)
    mask_small = coverage >= 0.999
    if np.count_nonzero(mask_small) < 64:
        raise ValueError("Too little supported surface at the selected inverse resolution")
    images_small, valid_small = [], []
    peak = 0.0
    for index, (path, validity_path) in enumerate(zip(image_paths, validity_paths)):
        image = bitmap_array(mi, path)
        valid = bitmap_array(mi, validity_path) > 0
        if image.shape != height.shape or valid.shape != height.shape:
            raise ValueError("Inverse observation or validity-mask dimensions differ from height")
        valid &= np.isfinite(image)
        image[~np.isfinite(image)] = 0
        if bool(parameters["srgb_decode"]):
            image = srgb_to_linear(image)
        peak = max(peak, float(np.max(image)))
        images_small.append(resize_area(image[y0:y1, x0:x1], render_height, render_width))
        valid_small.append(resize_area(valid[y0:y1, x0:x1].astype(np.float32), render_height, render_width) >= 0.999)
        progress(5 + int(4 * (index + 1) / len(image_paths)), f"Reduced inverse observation {index + 1}/{len(image_paths)}")
    if peak <= np.finfo(np.float32).eps:
        raise ValueError("Inverse-rendering image stack contains no positive finite samples")
    images_small = np.stack(images_small) / max(1.0, peak)
    valid_small = np.stack(valid_small)
    height_small = masked_resize(height_crop, mask_crop, render_height, render_width)
    albedo_small = masked_resize(albedo_crop, mask_crop, render_height, render_width)
    normalization = float(transfer.get("normalization_divisor", 1.0)) if transfer else 1.0
    if not math.isfinite(normalization) or normalization <= 0:
        raise ValueError("Invalid inverse observation normalization")
    albedo_small /= normalization * max(1.0, peak)
    weight_small = masked_resize(weight_crop, mask_crop, render_height, render_width)
    datum = float(geometry_job["reference_height_pixels"])
    if not math.isfinite(datum):
        raise ValueError("Inverse height reference must be finite")
    height_small -= datum
    height_small[~mask_small] = 0.0

    full_center_x = 0.5 * (height.shape[1] - 1)
    full_center_y = 0.5 * (height.shape[0] - 1)
    crop = geometry_job.get("crop")
    if crop:
        full_center_x = float(crop["x"]) + 0.5 * (float(crop["width"]) - 1.0)
        full_center_y = float(crop["y"]) + 0.5 * (float(crop["height"]) - 1.0)
    original_x = x0 - 0.5 + (np.arange(render_width, dtype=np.float32) + 0.5) * ((x1 - x0) / render_width)
    original_y = y0 - 0.5 + (np.arange(render_height, dtype=np.float32) + 0.5) * ((y1 - y0) / render_height)
    near_field = geometry_job["lighting_model"] == "near_field_ring"
    pixel_scale = float(geometry_job["pixel_scale_mm_per_pixel"])
    if near_field and (not math.isfinite(pixel_scale) or pixel_scale <= 0.0):
        raise ValueError("Near-field Mitsuba refinement requires positive mm/pixel scale")
    reference_z = float(geometry_job["reference_surface_z_mm"])
    led_diameter = float(geometry_job["led_diameter_mm"])
    ring_radius = float(geometry_job["ring_radius_mm"])
    ring_height = float(geometry_job["ring_height_mm"])
    angle = float(geometry_job["angular_diameter_degrees"])
    if not math.isfinite(reference_z) or not math.isfinite(led_diameter) or not math.isfinite(angle):
        raise ValueError("Inverse emitter geometry must be finite")
    if not near_field and not 0.1 <= angle <= 10.0:
        raise ValueError("Directional inverse angular diameter must be between 0.1 and 10 degrees")
    if near_field and (not math.isfinite(ring_radius) or ring_radius <= 0 or not math.isfinite(ring_height) or led_diameter <= 0 or reference_z >= ring_height):
        raise ValueError("Near-field inverse refinement needs positive LED diameter and reference Z below the lights")
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
    write_grid_ply(mesh_path, world_x, world_y, world_z, mask_small)
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
        "ring_height_world": (float(geometry_job["ring_height_mm"]) - reference_z) * scene_scale,
        "led_diameter_world": led_diameter * scene_scale,
        "angular_diameter_degrees": float(geometry_job["angular_diameter_degrees"]),
        "directional_distance_world": 100.0 * camera_extent,
        "fallback_azimuth": 0.0,
    }
    if settings.get("geometry_mode") == "absolute_height":
        # Ultra keeps every optimization-grid pixel that can move a rendered surface triangle,
        # including the silhouette. Isolated/thin mask samples with no emitted
        # triangle are not modeled geometry and cannot supply an equality loss.
        fit_mask = triangle_vertex_support(mask_small)
    else:
        # Coarse refinement excludes a one-pixel rim where area reduction and
        # open-mesh coverage can otherwise masquerade as a shading correction.
        fit_mask = mask_small.copy()
        fit_mask[1:] &= mask_small[:-1]
        fit_mask[:-1] &= mask_small[1:]
        fit_mask[:, 1:] &= mask_small[:, :-1]
        fit_mask[:, :-1] &= mask_small[:, 1:]
        fit_mask[[0, -1], :] = False
        fit_mask[:, [0, -1]] = False
    weights = (
        fit_mask[None, :, :].astype(np.float32)
        * np.clip(weight_small[None, :, :], 0.05, 1.0)
        * valid_small.astype(np.float32)
    )
    normal_prior, normal_prior_weight = load_normal_prior(
        inputs, mask, raw_robust_weight, (x0, y0, x1, y1), (render_height, render_width))
    normal_prior_weight *= fit_mask
    return {
        "height": height,
        "albedo": albedo,
        "mask": mask,
        "bounds": (x0, y0, x1, y1),
        "images_small": images_small,
        "height_small": height_small,
        "albedo_small": albedo_small,
        "mask_small": mask_small,
        "fit_mask": fit_mask,
        "weights": weights,
        "lights": lights,
        "baseline_positions": baseline_positions,
        "mesh_path": mesh_path,
        "camera": camera,
        "geometry": geometry,
        "scene_scale": scene_scale,
        "physical_scale": physical_scale,
        "height_datum": datum,
        "grid_spacing": ((y1 - y0) / render_height, (x1 - x0) / render_width),
        "normal_prior": normal_prior,
        "normal_prior_weight": normal_prior_weight,
        "world_spacing": (abs(float(world_y[1] - world_y[0])), abs(float(world_x[1] - world_x[0]))),
    }


def make_scenes(mi, prepared, backend: str, spp: int):
    scenes_by_basis = []
    parameters_by_basis = []
    for basis in ("diffuse", "narrow", "broad"):
        scene = mi.load_dict(scene_dictionary(
            mi, prepared["mesh_path"], basis, prepared["lights"][0],
            prepared["geometry"], prepared["camera"], backend, spp))
        params = mi.traverse(scene)
        keys = [key for key in params.keys() if key.endswith("vertex_positions")]
        if len(keys) != 1:
            raise RuntimeError(f"Expected one differentiable mesh in Mitsuba scene, found: {keys}")
        records = []
        for index, light in enumerate(prepared["lights"]):
            geometry = dict(prepared["geometry"])
            original_index = prepared.get("light_indices", range(len(prepared["lights"])))[index]
            original_count = prepared.get("input_light_count", len(prepared["lights"]))
            geometry["fallback_azimuth"] = 2.0 * math.pi * original_index / original_count
            records.append((params, keys[0], finite_emitter(mi, light, geometry)))
        # Lights are evaluated serially, so each basis needs only one scene.
        scenes_by_basis.append([scene] * len(prepared["lights"]))
        parameters_by_basis.append(records)
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


def select_supported_lights(prepared, result):
    """Select from input validity only, before rendering or fitting any losses."""
    weights = prepared["weights"]
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Inverse observation weights must be finite and nonnegative")
    support = np.count_nonzero(weights > 0, axis=(1, 2))
    used = np.flatnonzero(support >= MIN_LIGHT_SUPPORT).tolist()
    excluded = np.flatnonzero(support < MIN_LIGHT_SUPPORT).tolist()
    result["light_selection"] = {
        "basis": "pre_fit_geometric_support_and_unclipped_observation_validity",
        "index_base": 0,
        "input_light_count": len(support),
        "used_light_count": len(used),
        "minimum_supported_pixels": MIN_LIGHT_SUPPORT,
        "supported_pixels_per_light": support.tolist(),
        "used_light_indices": used,
        "excluded_light_indices": excluded,
    }
    if len(used) < 6:
        counts = ", ".join(f"{index + 1}: {count}" for index, count in enumerate(support))
        raise ValueError(
            f"Only {len(used)} of {len(support)} lights have at least {MIN_LIGHT_SUPPORT} "
            "supported, unclipped reduced pixels; inverse refinement needs at least six "
            "usable lights, including withheld validation lights. "
            f"Supported pixels by light (1-based): {counts}. "
            "Try higher inverse quality for less aggressive downsampling, or less-clipped "
            "input images. Baseline outputs are unchanged."
        )
    train, holdout = split_lights(len(used))
    result["training_light_indices"] = [used[index] for index in train]
    result["holdout_light_indices"] = [used[index] for index in holdout]
    selected = dict(prepared)
    for key in ("lights", "weights", "images_small"):
        selected[key] = prepared[key][used]
    selected["light_indices"] = used
    selected["input_light_count"] = len(support)
    # Omitting unusable views must not leave a rank-deficient lighting subset.
    directions = selected["lights"][train].astype(np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    singular = np.linalg.svd(directions, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None
    result["light_selection"]["training_direction_condition"] = condition
    if condition is None or condition > 100:
        raise ValueError("Usable inverse training light directions are poorly conditioned "
                         "(condition limit 100). More varied, adequately exposed lighting "
                         "is needed. Baseline outputs are unchanged.")
    return selected, train, holdout


def optimize_ad(mi, prepared, scenes_by_basis, parameters_by_basis, coefficients, mapping, settings, train, progress):
    import drjit as dr

    control_height, control_width, indices_numpy, weights_numpy = mapping
    absolute_height = settings.get("geometry_mode") == "absolute_height"
    if absolute_height and (control_height, control_width) != prepared["height_small"].shape:
        raise ValueError("Absolute-height optimization requires one control per render pixel")
    requested_learning_rate = float(settings["learning_rate"])
    learning_rate = (effective_ultra_learning_rate(
        prepared["world_spacing"], requested_learning_rate)
        if absolute_height else requested_learning_rate)
    if absolute_height:
        prepared["learning_rate_requested"] = requested_learning_rate
        prepared["learning_rate_effective"] = learning_rate
    optimizer = mi.ad.Adam(lr=learning_rate)
    initial_control = (prepared["baseline_positions"][..., 2].ravel().astype(np.float32)
                       if absolute_height else
                       np.zeros(control_height * control_width, dtype=np.float32))
    absolute_limit = max(
        float(settings.get("absolute_height_limit", 2.0)),
        1.25 * float(np.max(np.abs(initial_control))) + 0.22,
    )
    if absolute_height:
        prepared["absolute_height_limit_world"] = absolute_limit
    optimizer["height_control"] = mi.Float(initial_control)
    identity_mapping = indices_numpy is None
    index_arrays = [] if identity_mapping else [mi.UInt(index) for index in indices_numpy]
    weight_arrays = [] if identity_mapping else [mi.Float(weight) for weight in weights_numpy]
    baseline_flat = (None if absolute_height else
                     mi.Float(prepared["baseline_positions"].ravel()))
    baseline_xy = prepared["baseline_positions"].copy()
    baseline_xy[..., 2] = 0.0
    baseline_xy_flat = (mi.Float(baseline_xy.ravel()) if absolute_height else None)
    # Ultra streams observations one light at a time. Keeping every image and
    # validity plane resident on the GPU duplicated a large optimization stack.
    observed = (None if absolute_height else
                [mi.TensorXf(image) for image in prepared["images_small"]])
    data_weights = (None if absolute_height else
                    [mi.TensorXf(weight) for weight in prepared["weights"]])
    material = [mi.TensorXf(coefficients[..., index]) for index in range(3)]
    iterations = int(settings["iterations_ad"])
    maximum_delta = 0.22
    render_height, render_width = prepared["height_small"].shape
    refit_interval = int(settings.get("material_refit_interval", 0))
    history = []
    best = None
    prior_weight = (None if absolute_height else
                    mi.Float(prepared["normal_prior_weight"].ravel()))
    prior = ([] if absolute_height else
             [mi.Float(prepared["normal_prior"][..., axis].ravel()) for axis in range(3)])
    prior_denominator = max(float(prepared["normal_prior_weight"].sum()), 1e-8)
    prior_enabled = bool(np.any(prepared["normal_prior_weight"] > 0))
    stencils = ([] if absolute_height else
                [(mi.UInt(a), mi.UInt(b), mi.Float(scale))
                 for a, b, scale in slope_stencils(
                     prepared["mask_small"], *prepared["world_spacing"])])
    regularization_mask = prepared.get("fit_mask", prepared["mask_small"])
    curvature = ([] if absolute_height else [
        (mi.UInt(left), mi.UInt(center), mi.UInt(right), mi.Float(scale), mi.Float(valid))
        for left, center, right, scale, valid in curvature_stencils(
            regularization_mask, *prepared["world_spacing"])
    ])
    baseline_z = (None if absolute_height else
                  mi.Float(prepared["baseline_positions"][..., 2].ravel()))

    def expanded_control():
        control = optimizer["height_control"]
        if identity_mapping:
            return control
        return sum(
            dr.gather(mi.Float, control, index) * weight
            for index, weight in zip(index_arrays, weight_arrays)
        )

    def checkpoint(iteration):
        nonlocal material, best
        control = optimizer["height_control"].numpy().reshape((control_height, control_width)).astype(np.float32)
        expanded = expand_control_numpy(control, mapping).reshape(prepared["height_small"].shape)
        positions = prepared["baseline_positions"].copy()
        if absolute_height:
            positions[..., 2] = expanded
        else:
            positions[..., 2] += expanded
        basis = render_basis_numpy(mi, scenes_by_basis, parameters_by_basis, positions,
                                   int(settings["validation_spp"]), 20000)
        refitted, current_loss = refit_training_material(
            basis, prepared["images_small"], prepared["mask_small"], prepared["albedo_small"],
            train, prepared["weights"], coefficients)
        coefficients[:] = refitted
        material = [mi.TensorXf(coefficients[..., index]) for index in range(3)]
        prior_loss = normal_prior_loss(prepared, positions[..., 2])
        geometry_regularization = (absolute_height_regularization_numpy(
            prepared, positions[..., 2], settings) if absolute_height else 0.0)
        objective = current_loss + NORMAL_PRIOR_STRENGTH * prior_loss + geometry_regularization
        history.append({"iteration": iteration, "training_loss": current_loss,
                        "normal_prior_loss": prior_loss,
                        "geometry_regularization": geometry_regularization,
                        "objective": objective})
        if math.isfinite(objective) and (best is None or objective < best[0]):
            best = (objective, control.copy(), expanded.copy(), coefficients.copy(), iteration)

    if refit_interval > 0:
        if absolute_height and "initial_training_loss" in prepared:
            expanded = initial_control.reshape(prepared["height_small"].shape).copy()
            prior_loss = normal_prior_loss(prepared, expanded)
            geometry_regularization = absolute_height_regularization_numpy(
                prepared, expanded, settings)
            objective = (float(prepared["initial_training_loss"])
                         + NORMAL_PRIOR_STRENGTH * prior_loss
                         + geometry_regularization)
            history.append({"iteration": 0,
                            "training_loss": float(prepared["initial_training_loss"]),
                            "normal_prior_loss": prior_loss,
                            "geometry_regularization": geometry_regularization,
                            "objective": objective})
            best = (objective, initial_control.reshape((control_height, control_width)).copy(),
                    expanded, coefficients.copy(), 0)
        else:
            checkpoint(0)
    denominator = max(float(prepared["weights"][train].sum()), 1.0e-8)
    iteration_began = time.monotonic()
    for iteration in range(iterations):
        # Accumulate the exact summed-objective gradient one light at a time;
        # retaining every renderer's AD graph scales poorly with image count.
        for training_index, light_index in enumerate(train):
            expanded = expanded_control()
            z_displacement = dr.ravel(mi.Vector3f(
                dr.zeros(mi.Float, render_height * render_width),
                dr.zeros(mi.Float, render_height * render_width),
                expanded,
            ))
            positions = (baseline_xy_flat + z_displacement if absolute_height else
                         baseline_flat + z_displacement)
            if absolute_height:
                # First evaluate the prediction without recording geometry AD.
                # Then replay one material basis at a time with the exact
                # d(loss)/d(prediction). This retains one renderer graph rather
                # than three full-grid graphs and is mathematically identical
                # to differentiating their sum.
                detached_positions = dr.detach(positions)
                dr.eval(detached_positions)
                del positions, z_displacement, expanded
                prediction = None
                for basis_index, (basis_scenes, basis_parameters) in enumerate(
                        zip(scenes_by_basis, parameters_by_basis)):
                    scene = basis_scenes[light_index]
                    params = update_scene_parameters(
                        mi, basis_parameters[light_index], detached_positions)
                    image = mi.render(
                        scene, params, spp=int(settings["spp"]),
                        seed=30000 + iteration * 1009 + basis_index * 101 + light_index)
                    value = dr.detach(dr.mean(image, axis=2)) * material[basis_index]
                    prediction = value if prediction is None else prediction + value
                observed_light = mi.TensorXf(prepared["images_small"][light_index])
                weight_light = mi.TensorXf(prepared["weights"][light_index])
                residual = prediction - observed_light
                upstream = dr.detach(
                    residual / dr.sqrt(residual * residual + 0.0025**2)
                    * weight_light / denominator)
                dr.eval(upstream)
                del prediction, residual, observed_light, weight_light
                for basis_index, (basis_scenes, basis_parameters) in enumerate(
                        zip(scenes_by_basis, parameters_by_basis)):
                    basis_expanded = expanded_control()
                    basis_displacement = dr.ravel(mi.Vector3f(
                        dr.zeros(mi.Float, render_height * render_width),
                        dr.zeros(mi.Float, render_height * render_width),
                        basis_expanded,
                    ))
                    basis_positions = baseline_xy_flat + basis_displacement
                    scene = basis_scenes[light_index]
                    params = update_scene_parameters(
                        mi, basis_parameters[light_index], basis_positions)
                    image = mi.render(
                        scene, params, spp=int(settings["spp"]),
                        spp_grad=int(settings["spp"]),
                        seed=30000 + iteration * 1009 + basis_index * 101 + light_index)
                    dr.backward(safe_dr_sum(
                        dr, dr.mean(image, axis=2)
                        * material[basis_index] * upstream))
                del upstream
            else:
                rendered = []
                for basis_index, (basis_scenes, basis_parameters) in enumerate(
                        zip(scenes_by_basis, parameters_by_basis)):
                    scene = basis_scenes[light_index]
                    params = update_scene_parameters(
                        mi, basis_parameters[light_index], positions)
                    image = mi.render(
                        scene, params, spp=int(settings["spp"]),
                        spp_grad=int(settings["spp"]),
                        seed=30000 + iteration * 1009 + basis_index * 101 + light_index)
                    rendered.append(dr.mean(image, axis=2))
                prediction = sum(
                    rendered[basis] * material[basis]
                    for basis in range(3))
                residual = prediction - observed[light_index]
                robust = dr.sqrt(residual * residual + 0.0025**2) - 0.0025
                dr.backward(safe_dr_sum(
                    dr, robust * data_weights[light_index]) / denominator)
            progress(22 + int(58 * (iteration + (training_index + 1) / len(train)) / iterations),
                     f"Inverse iteration {iteration + 1}/{iterations}, light {training_index + 1}/{len(train)}")

        control = optimizer["height_control"]
        if absolute_height:
            control = optimizer["height_control"]
            height_numpy = control.numpy().reshape((control_height, control_width)).astype(
                np.float32, copy=False)
            geometry_regularization, prior_loss, regularization_gradient = (
                absolute_height_regularization_gradient_numpy(
                    prepared, height_numpy, settings))
            data_gradient = dr.grad(control)
            dr.set_grad(
                control,
                data_gradient + mi.Float(regularization_gradient.ravel()))
            regularization = (geometry_regularization
                              + NORMAL_PRIOR_STRENGTH * prior_loss)
        else:
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
        if prior_enabled and not absolute_height:
            z = expanded_control() if absolute_height else baseline_z + expanded_control()
            regularization += NORMAL_PRIOR_STRENGTH * normal_prior_loss_ad(
                mi, dr, z, prior, prior_weight, stencils, prior_denominator)
        if not absolute_height:
            dr.backward(regularization)
        optimizer.step()
        if absolute_height:
            optimizer["height_control"] = dr.clip(
                optimizer["height_control"], -absolute_limit, absolute_limit)
        else:
            optimizer["height_control"] = dr.clip(
                optimizer["height_control"], -maximum_delta, maximum_delta)
        if refit_interval > 0 and ((iteration + 1) % refit_interval == 0 or iteration + 1 == iterations):
            progress(22 + int(58 * (iteration + 1) / iterations), "Refitting material from training lights")
            checkpoint(iteration + 1)
        if prepared.get("live_preview", False):
            current_control = optimizer["height_control"].numpy().reshape((control_height, control_width))
            publish_iteration(mi, progress, prepared, current_control, mapping, iteration + 1,
                              iterations, iteration_began, absolute_height=absolute_height)
        progress(22 + int(58 * (iteration + 1) / iterations), f"Differentiable inverse iteration {iteration + 1}/{iterations}")
    if best is not None:
        _, control, expanded, best_material, selected_iteration = best
        coefficients[:] = best_material
        prepared["optimization_history"] = history
        prepared["selected_iteration"] = selected_iteration
        return control, expanded, iterations
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
        "candidate_saved": False,
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

        core_patch = drjit_core_patch_status()
        result["drjit_core_patch"] = core_patch
        if quality == "ultra" and selected_backend == "cuda" and not core_patch["valid"]:
            raise RuntimeError(
                "Ultra CUDA requires the patched Dr.Jit Core runtime from the "
                "what-a-relief 0.2.16 Mitsuba backend installer. Reinstall that "
                "backend before retrying; baseline outputs are unchanged."
            )

        result.update(
            {
                "selected_backend": selected_backend,
                "variant": variant,
                "optimizer": optimizer_name,
                "quality": quality,
                "llvm_version": list(dr.detail.llvm_version()) if selected_backend == "llvm" else None,
                "mitsuba_version": mi.__version__,
                "drjit_version": dr.__version__,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
            }
        )
        settings = dict(QUALITY[quality])
        absolute_height = settings.get("geometry_mode") == "absolute_height"
        result.update(
            geometry_parameterization=("absolute_height_grid" if absolute_height else
                                       "coarse_additive_correction"),
            absolute_height_inverse=absolute_height,
            initialization="classical_integrated_height",
        )
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
        x0, y0, x1, y1 = prepared["bounds"]
        source_crop_width = x1 - x0
        source_crop_height = y1 - y0
        optimization_downsampled = (
            render_width != source_crop_width or render_height != source_crop_height)
        result["source_crop_width"] = source_crop_width
        result["source_crop_height"] = source_crop_height
        result["optimization_max_side"] = int(settings["max_side"])
        result["optimization_downsampled"] = optimization_downsampled
        result["full_resolution_inverse"] = absolute_height and not optimization_downsampled
        result["geometry_mask_pixels"] = int(np.count_nonzero(prepared["mask_small"]))
        result["modeled_fit_pixels"] = int(np.count_nonzero(prepared["fit_mask"]))
        result["pixel_inclusion_policy"] = (
            "all_valid_triangle_supported_optimization_grid_vertices" if absolute_height else
            "valid_eroded_geometry_pixels"
        )
        if absolute_height:
            requested_spp = int(settings["spp"])
            effective_spp = effective_ultra_spp(
                render_height * render_width, requested_spp)
            settings["spp"] = effective_spp
            result["optimization_spp_requested"] = requested_spp
            result["optimization_spp_effective"] = effective_spp
            result["optimization_path_budget"] = ULTRA_AD_SAMPLE_BUDGET
            if effective_spp < requested_spp:
                progress(
                    9,
                    f"Ultra memory guard reduced each stochastic AD pass from "
                    f"{requested_spp} to {effective_spp} spp at the optimization resolution; "
                    "fixed-seed validation remains at full quality",
                )
        prepared, train, holdout = select_supported_lights(prepared, result)
        selection = result["light_selection"]
        if selection["excluded_light_indices"]:
            omitted = ", ".join(str(index + 1) for index in selection["excluded_light_indices"])
            progress(9, f"Using {selection['used_light_count']}/{selection['input_light_count']} "
                     f"inverse lights; insufficient reduced support in lights {omitted}. "
                     "Clipping exclusions and withheld-light validation remain enabled.")
        progress(10, "Building diffuse and spatially varying glossy material bases")
        scenes_by_basis, parameters_by_basis = make_scenes(
            mi, prepared, selected_backend, int(settings["spp"])
        )
        progress(15, "Rendering baseline material and visibility bases")
        baseline_basis = render_basis_numpy(
            mi,
            scenes_by_basis,
            parameters_by_basis,
            prepared["baseline_positions"],
            int(settings["validation_spp"]),
            10000,
        )
        coefficients, roughness = fit_material_maps(
            baseline_basis,
            prepared["images_small"],
            prepared["mask_small"],
            prepared["albedo_small"],
            train,
            prepared["weights"],
        )
        baseline_coefficients = coefficients.copy()
        prediction_before = prediction_numpy(baseline_basis, coefficients)
        train_before = data_loss_numpy(
            prediction_before, prepared["images_small"], prepared["weights"], train
        )
        holdout_before = data_loss_numpy(
            prediction_before, prepared["images_small"], prepared["weights"], holdout
        )
        mean_before = np.mean(prediction_before[train], axis=0)
        prepared["initial_training_loss"] = train_before
        # Baseline basis images are large at the optimization resolution and are no
        # longer needed during AD. Release them before allocating the optimizer.
        del baseline_basis, prediction_before
        gc.collect()
        mapping = control_mapping(render_height, render_width, int(settings["control_spacing"]))
        prepared["live_preview"] = job.get("parameters", {}).get("live_preview", False) is True
        progress(20, (f"Optimizing an absolute {render_width}x{render_height} height field "
                      f"({settings['max_side']} px maximum side)" if absolute_height else
                      "Optimizing a smooth height correction while preserving baseline detail"))
        control, optimized_small_world, iterations = optimize_ad(
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
        if absolute_height:
            candidate_positions[..., 2] = optimized_small_world
        else:
            candidate_positions[..., 2] += optimized_small_world
        candidate_height, correction_full = full_height_solution(
            prepared, optimized_small_world, absolute_height)
        progress(82, "Rendering fixed-seed candidate for training and withheld-light validation")
        candidate_basis = render_basis_numpy(
            mi,
            scenes_by_basis,
            parameters_by_basis,
            candidate_positions,
            int(settings["validation_spp"]),
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
        correction_small_world = (optimized_small_world - prepared["baseline_positions"][..., 2]
                                  if absolute_height else optimized_small_world)
        correction_small_pixels = correction_small_world / max(conversion, 1.0e-12)
        masked_correction = correction_small_pixels[prepared["fit_mask"]]
        correction_rms = float(np.sqrt(np.mean(masked_correction**2)))
        correction_maximum = float(np.max(np.abs(masked_correction)))
        correction_crop = correction_full[y0:y1, x0:x1]
        crop_mask = prepared["mask"][y0:y1, x0:x1]
        q, p = masked_slopes(correction_crop, crop_mask)
        slope_rms = float(np.sqrt(np.mean(p[crop_mask] ** 2 + q[crop_mask] ** 2)))
        train_improvement = (train_before - train_after) / max(train_before, 1.0e-8)
        holdout_improvement = (holdout_before - holdout_after) / max(holdout_before, 1.0e-8)
        prior_before = normal_prior_loss(prepared, prepared["baseline_positions"][..., 2])
        prior_after = normal_prior_loss(prepared, candidate_positions[..., 2])
        prior_consistent = math.isfinite(prior_after) and prior_after <= 1.05 * prior_before + 1e-6
        accepted = (
            math.isfinite(train_after)
            and math.isfinite(holdout_after)
            and train_improvement >= 0.01
            and holdout_improvement >= -0.002
            and slope_rms <= 0.45
            and correction_maximum <= 0.20 * max(prepared["height"].shape)
            and prior_consistent
        )
        # A separate Monte Carlo seed must confirm the decision; common random
        # numbers within each pair reduce variance without reusing optimization samples.
        independent_holdout_improvement = None
        if accepted:
            validation_losses = []
            for positions, material in ((prepared["baseline_positions"], baseline_coefficients),
                                        (candidate_positions, coefficients)):
                basis = render_basis_numpy(mi, scenes_by_basis, parameters_by_basis, positions, int(settings["validation_spp"]), 71000)
                prediction = prediction_numpy(basis, material)
                validation_losses.append(data_loss_numpy(prediction, prepared["images_small"], prepared["weights"], holdout))
            independent_holdout_improvement = (validation_losses[0] - validation_losses[1]) / max(validation_losses[0], 1.0e-8)
            accepted = math.isfinite(independent_holdout_improvement) and independent_holdout_improvement >= -0.002
        if accepted:
            decision = "accepted_train_and_holdout_gate"
        elif train_improvement < 0.01:
            decision = "rejected_insufficient_training_improvement"
        elif holdout_improvement < -0.002:
            decision = "rejected_withheld_lights_worsened"
        elif independent_holdout_improvement is not None and (not math.isfinite(independent_holdout_improvement) or independent_holdout_improvement < -0.002):
            decision = "rejected_independent_validation_seed"
        elif slope_rms > 0.45:
            decision = "rejected_excessive_slope_change"
        elif not prior_consistent:
            decision = "rejected_normal_prior_worsened"
        else:
            decision = "rejected_excessive_height_change"

        progress(90, "Writing guarded inverse geometry, material maps, and audit previews")
        output_products(
            mi,
            output,
            prepared["height"],
            (candidate_height if accepted else prepared["height"]) if absolute_height else
            (correction_full if accepted else np.zeros_like(correction_full)),
            prepared["mask"],
            prepared["albedo"],
            float(job["parameters"]["height_scale"]),
            absolute_height=absolute_height,
        )
        delivered_material = coefficients if accepted else baseline_coefficients
        roughness = material_roughness(delivered_material)
        diffuse_full = np.zeros_like(prepared["height"], dtype=np.float32)
        specular_full = np.zeros_like(prepared["height"], dtype=np.float32)
        roughness_full = np.zeros_like(prepared["height"], dtype=np.float32)
        diffuse_full[y0:y1, x0:x1] = resize_bilinear(delivered_material[..., 0], y1 - y0, x1 - x0)
        specular_full[y0:y1, x0:x1] = resize_bilinear(
            delivered_material[..., 1] + delivered_material[..., 2], y1 - y0, x1 - x0
        )
        roughness_full[y0:y1, x0:x1] = resize_bilinear(roughness, y1 - y0, x1 - x0)
        save_gray(mi, output / "material_diffuse.png", display_stretch(diffuse_full, prepared["mask"]))
        save_gray(mi, output / "material_specular.png", display_stretch(specular_full, prepared["mask"]))
        save_gray(mi, output / "material_roughness.png", np.where(prepared["mask"], roughness_full / 0.5, 0.0))
        mean_after = np.mean(prediction_after[train], axis=0)
        render_previews, render_display_range = comparison_stretch(mean_before, mean_after, prepared["mask_small"])
        save_gray(mi, output / "render_before.png", render_previews[0])
        save_gray(mi, output / "render_after.png", render_previews[1])
        result.update(
            {
                "status": "complete",
                "accepted": accepted,
                "decision": decision,
                "iterations_completed": iterations,
                "selected_iteration": prepared.get("selected_iteration", iterations),
                "optimization_history": prepared.get("optimization_history", []),
                "control_grid_shape": list(mapping[:2]),
                "optimization_spp": settings["spp"],
                "loss_reduction_chunk": DRJIT_REDUCTION_CHUNK,
                "material_refit_interval": settings.get("material_refit_interval", 0),
                "learning_rate_requested": prepared.get(
                    "learning_rate_requested", settings["learning_rate"]),
                "learning_rate_effective": prepared.get(
                    "learning_rate_effective", settings["learning_rate"]),
                "normal_prior_enabled": bool(np.any(prepared["normal_prior_weight"] > 0)),
                "normal_prior_strength": NORMAL_PRIOR_STRENGTH,
                "normal_prior_loss_before": prior_before,
                "normal_prior_loss_after": prior_after,
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
                "height_difference_product": "candidate_minus_classical_initialization",
                "absolute_height_limit_world": prepared.get("absolute_height_limit_world"),
                "height_datum_assumption": "shared flat-surface/percentile reference assigned to explicit physical reference Z",
                "reference_height_pixels": prepared["height_datum"],
                "reference_surface_z_mm": job["geometry"]["reference_surface_z_mm"],
                "emitter_model": "finite_reference_facing_disk" if prepared["geometry"]["lighting_model"] == "near_field_ring" else "distant_disk_angular_approximation",
                "led_diameter_mm": job["geometry"]["led_diameter_mm"],
                "angular_diameter_degrees": job["geometry"]["angular_diameter_degrees"],
                "independent_holdout_relative_improvement": independent_holdout_improvement,
                "validation_spp": settings["validation_spp"],
                "primary_visibility_derivatives": True,
                "indirect_visibility_derivatives": True,
                "material_model": "per-pixel nonnegative diffuse plus narrow- and broad-GGX rough-plastic bases",
                "baseline_outputs_modified": False,
                "render_preview_shared_range": render_display_range,
                "pixel_scale_mm_per_pixel": job["geometry"]["pixel_scale_mm_per_pixel"],
            }
        )
        retain_unvalidated_candidate(
            mi, output, prepared["height"],
            candidate_height if absolute_height else correction_full,
            prepared["mask"], prepared["albedo"],
            float(job["parameters"]["height_scale"]), result,
            absolute_height=absolute_height,
        )
        atomic_json(result_path, result)
        progress(100, ("Mitsuba absolute-height inverse solution complete" if absolute_height else
                       "Mitsuba inverse refinement complete"))
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
