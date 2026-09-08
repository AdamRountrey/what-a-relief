"""Independent curved-relief fixture through the real C++ and inverse pipeline.

This is an improvement test, not a test that treats rejection as success.
The forward scene does not use the worker's scene/material construction.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np


SIDE = 24
SPAN_MM = 12.0
SCALE_MM = SPAN_MM / SIDE
RING_RADIUS_MM = 18.0
RING_HEIGHT_MM = 8.0
LED_DIAMETER_MM = 0.8
SEED = 84021


def relief(x, y):
    ridge = 1.4 * np.exp(-((x + 1.4 - 0.20 * y) / 0.75) ** 2 - (y / 3.2) ** 4)
    bump = 0.65 * np.exp(-((x - 2.0) / 1.4) ** 2 - ((y - 1.6) / 1.3) ** 2)
    hollow = -0.35 * np.exp(-((x - 1.4) / 1.5) ** 2 - ((y + 1.9) / 1.0) ** 2)
    return ridge + bump + hollow


def write_truth_mesh(path):
    axis = np.linspace(-7, 7, 225, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    z = relief(x, y)
    dy, dx = np.gradient(z, float(axis[1] - axis[0]))
    normals = np.stack((-dx, -dy, np.ones_like(z)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel(), normals.reshape(-1, 3),
                                (x.ravel() + 7) / 14, (y.ravel() + 7) / 14)).astype('<f4')
    yy, xx = np.mgrid[:len(axis) - 1, :len(axis) - 1]
    a = (yy * len(axis) + xx).ravel()
    faces = np.concatenate((np.column_stack((a, a + 1, a + len(axis))),
                            np.column_stack((a + 1, a + len(axis) + 1, a + len(axis)))))
    records = np.empty(len(faces), dtype=[('count', 'u1'), ('indices', '<i4', (3,))])
    records['count'], records['indices'] = 3, faces
    with path.open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\n'
                      f'element vertex {len(vertices)}\n'
                      'property float x\nproperty float y\nproperty float z\n'
                      'property float nx\nproperty float ny\nproperty float nz\n'
                      'property float s\nproperty float t\n'
                      f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n').encode('ascii'))
        vertices.tofile(stream)
        records.tofile(stream)


def write_gray(mi, path, values, bits=16):
    kind = mi.Struct.Type.UInt16 if bits == 16 else mi.Struct.Type.UInt8
    mi.Bitmap(np.ascontiguousarray(values[..., None], np.float32)).convert(
        mi.Bitmap.PixelFormat.Y, kind, False).write(str(path))


def generate(mi, root, light_offset=0.0):
    transform = mi.ScalarTransform4f
    mesh = root / 'truth_mesh.ply'
    write_truth_mesh(mesh)
    uv_y, uv_x = np.mgrid[:128, :128] / 127.0
    albedo = (0.26 + 0.09 * np.sin(uv_x * 17) ** 2 + 0.05 * np.cos(uv_y * 13) ** 2)
    glossy = ((uv_x > 0.48) & (uv_y > 0.28) & (uv_y < 0.78)).astype(np.float32)
    write_gray(mi, root / 'albedo_texture.png', albedo)
    write_gray(mi, root / 'glossy_patch.png', glossy)
    texture = lambda name: {'type': 'bitmap', 'filename': str(root / name), 'raw': True}
    description = {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 4},
        'sensor': {
            'type': 'orthographic',
            'to_world': transform().look_at(origin=[0, 0, 100], target=[0, 0, 0], up=[0, 1, 0]).scale(SPAN_MM / 2),
            'near_clip': 0.01, 'far_clip': 200,
            'sampler': {'type': 'independent', 'sample_count': 256},
            'film': {'type': 'hdrfilm', 'width': SIDE, 'height': SIDE,
                     'pixel_format': 'rgb', 'rfilter': {'type': 'box'}},
        },
        'surface': {
            'type': 'ply', 'filename': str(mesh), 'face_normals': False,
            'bsdf': {'type': 'blendbsdf', 'weight': texture('glossy_patch.png'),
                     'matte': {'type': 'diffuse', 'reflectance': texture('albedo_texture.png')},
                     'glossy': {'type': 'roughplastic', 'distribution': 'beckmann', 'alpha': 0.22,
                               'int_ior': 1.45, 'nonlinear': True,
                               'diffuse_reflectance': texture('albedo_texture.png')}},
        },
    }
    aov_description = dict(description, integrator={'type': 'aov', 'aovs': 'normal:sh_normal,position:position'})
    truth_scene = mi.load_dict(aov_description)
    aov = np.array(mi.render(truth_scene, spp=256, seed=SEED), dtype=np.float32, copy=True)
    assert aov.shape == (SIDE, SIDE, 6), aov.shape
    normal, position = aov[..., :3].copy(), aov[..., 3:].copy()
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    # Check the camera convention against a known analytic height, independently
    # of inverse geometry and AOV channel ordering.
    assert np.max(np.abs(position[..., 2] - relief(position[..., 0], position[..., 1]))) < 0.6 * SCALE_MM ** 2
    assert position[SIDE // 2, 0, 0] < position[SIDE // 2, -1, 0]
    assert position[0, SIDE // 2, 1] > position[-1, SIDE // 2, 1]
    np.savez_compressed(root / 'truth.npz', height_mm=position[..., 2], normals=normal, positions_mm=position)
    write_gray(mi, root / 'truth_height_preview.png', (position[..., 2] + 0.4) / 1.9)
    observations, lights, shadow_masks = [], [], []
    rng = np.random.default_rng(SEED)
    for i in range(8):
        angle = 2 * math.pi * i / 8 + 0.17 + light_offset
        source = np.array([RING_RADIUS_MM * math.cos(angle), RING_RADIUS_MM * math.sin(angle), RING_HEIGHT_MM])
        lights.append(source / np.linalg.norm(source))
        radius = LED_DIAMETER_MM / 2
        emitter = {'type': 'disk', 'to_world': transform().look_at(origin=source.tolist(), target=[0, 0, 0], up=[0, 1, 0]).scale(radius),
                   'emitter': {'type': 'area', 'radiance': float((source @ source + radius ** 2) / radius ** 2)}}
        scene = mi.load_dict(dict(description, emitter=emitter))
        rgb = np.array(mi.render(scene, spp=256, seed=SEED + 101 * i), dtype=np.float32, copy=True)
        intensity = rgb.mean(axis=-1)
        if not np.all(np.isfinite(intensity)) or np.max(intensity) > 100:
            raise AssertionError(f'Invalid forward radiance for light {i}: {intensity.min()}, {intensity.max()}')
        electrons = rng.poisson(np.maximum(intensity, 0) * 30000) + rng.normal(0, 2.0, intensity.shape)
        observed = np.rint(np.clip(electrons / 30000, 0, 1) * 4095) / 4095
        path = root / f'image_{i}.png'
        write_gray(mi, path, observed)
        observations.append(path)
        blocked = np.zeros((SIDE, SIDE), bool)
        facing = np.zeros((SIDE, SIDE), bool)
        for y in range(SIDE):
            for x in range(SIDE):
                camera_ray = mi.Ray3f(mi.Point3f(float(position[y, x, 0]), float(position[y, x, 1]), 100),
                                     mi.Vector3f(0, 0, -1))
                hit = truth_scene.ray_intersect(camera_ray)
                ray = hit.spawn_ray_to(mi.Point3f(source))
                blocked[y, x] = truth_scene.ray_test(ray)
                facing[y, x] = float(np.dot(np.array(hit.sh_frame.n), source - np.array(hit.p))) > 0
        shadow = blocked & facing
        shadow_masks.append(shadow)
        write_gray(mi, root / f'cast_shadow_{i}.png', shadow.astype(np.float32), bits=8)
        print(f'Forward fixture light {i + 1}/8; cast-shadow pixels={int(shadow.sum())}', flush=True)
    with (root / 'light_vectors.csv').open('w', encoding='ascii') as stream:
        stream.write('x,y,z\n')
        np.savetxt(stream, lights, delimiter=',', fmt='%.10g')
    np.save(root / 'cast_shadow_union.npy', np.any(shadow_masks, axis=0))
    return observations


def metric(height_mm, truth, mask):
    error = height_mm[mask] - truth['height_mm'][mask]
    q, p = np.gradient(height_mm, SCALE_MM)
    normal = np.stack((-p, q, np.ones_like(height_mm)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip(np.sum(normal * truth['normals'], axis=-1), -1, 1)))
    return {'height_rmse_mm': float(np.std(error)), 'normal_mae_deg': float(angle[mask].mean()), 'pixels': int(mask.sum())}


def normal_product_metrics(mi, output, truth, regions):
    scores = {}
    for name, relative in [('classical', 'normal_rgb.png'), ('inverse', 'inverse/inverse_normal_rgb.png')]:
        raw = np.asarray(mi.Bitmap(str(output / relative)))
        if raw.dtype != np.uint8 or raw.shape != truth['normals'].shape:
            raise AssertionError('Unexpected encoded normal-map shape or type')
        normals = raw.astype(np.float32) * (2.0 / 255.0) - 1.0
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
        error = np.degrees(np.arccos(np.clip(np.sum(normals * truth['normals'], axis=-1), -1, 1)))
        scores[name] = {region: float(error[roi].mean()) for region, roi in regions.items() if np.any(roi)}
    return {'encoding': 'raw RGB8 vector components; quantized diagnostic, no sRGB decoding',
            'mae_degrees': scores,
            'inverse_improves_classical_normal_product': scores['inverse']['all'] < scores['classical']['all']}


def main():
    if os.name == 'nt':
        import ctypes
        # Only this test process and its children inherit the loader-error mode.
        # Missing dependencies must fail the test, not open desktop dialogs.
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetErrorMode.restype = ctypes.c_uint
        kernel.SetErrorMode.argtypes = [ctypes.c_uint]
        kernel.SetErrorMode(kernel.GetErrorMode() | 0x0001)
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--light-offset-radians', type=float, default=0.0,
                        help='Rotate the eight-light capture for a separate generalization check')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='curved-', dir=args.output.resolve()))
    print(f'Fixture and result artifacts: {root}', flush=True)
    spec = importlib.util.spec_from_file_location('worker', args.worker.resolve())
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.configure_runtime_paths()
    import mitsuba as mi
    mi.set_variant('scalar_rgb')
    render_started = time.perf_counter()
    if not math.isfinite(args.light_offset_radians):
        raise ValueError('Light offset must be finite')
    observations = generate(mi, root, args.light_offset_radians)
    render_seconds = time.perf_counter() - render_started
    command = [str(args.app.resolve())]
    for path in observations:
        command += ['--image', str(path)]
    output = root / 'output'
    command += ['--out', str(output), '--no-gui', '--solver', 'robust', '--height-solver', 'fast',
                '--height-flatten', 'none', '--lights-file', str(root / 'light_vectors.csv'),
                '--pixel-scale-mm', str(SCALE_MM), '--near-field-ring', str(RING_RADIUS_MM), str(RING_HEIGHT_MM),
                '--shadow-led-diameter-mm', str(LED_DIAMETER_MM), '--mitsuba-inverse',
                '--mitsuba-python', sys.executable, '--mitsuba-worker', str(args.worker.resolve()),
                '--mitsuba-backend', args.backend, '--mitsuba-quality', 'preview']
    application_started = time.perf_counter()
    with (root / 'application.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name != 'nt'))
        try:
            return_code = process.wait(timeout=1800)
        except BaseException:
            # Kill the owned process tree while its parent still exists, so a
            # test timeout cannot leave the separate renderer using the CPU.
            if os.name == 'nt':
                subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if return_code:
        hint = ' Missing DLL: use the deployed app folder or the CTest runtime-path wrapper.' if (return_code & 0xffffffff) == 0xc0000135 else ''
        raise RuntimeError(f'Application failed ({return_code}).{hint} Inspect {root / "application.log"}')
    application_seconds = time.perf_counter() - application_started
    truth = np.load(root / 'truth.npz')
    valid = np.asarray(mi.Bitmap(str(output / 'valid_mask.png'))).squeeze() > 0
    interior = np.zeros((SIDE, SIDE), bool)
    interior[3:-3, 3:-3] = True
    coverage = float(valid[interior].mean())
    # Compare identical pixels, including only neighborhoods supporting normals.
    support = valid.copy()
    support[1:] &= valid[:-1]
    support[:-1] &= valid[1:]
    support[:, 1:] &= valid[:, :-1]
    support[:, :-1] &= valid[:, 1:]
    region = interior & support
    baseline = worker.read_pfm(output / 'height.pfm') * SCALE_MM
    inverse = worker.read_pfm(output / 'inverse' / 'inverse_height.pfm') * SCALE_MM
    audit = json.loads((output / 'inverse' / 'result.json').read_text())
    shadows = np.load(root / 'cast_shadow_union.npy')
    regions = {'all': region, 'cast_shadow': region & shadows,
               'ridge': region & (truth['positions_mm'][..., 0] < 0),
               'bump_and_hollow': region & (truth['positions_mm'][..., 0] >= 0)}
    report = {
        'fixture': 'curved_ring_pipeline_v1', 'seed': SEED, 'image_count': 8, 'image_side': SIDE, 'render_spp': 256,
        'pixel_scale_mm': SCALE_MM, 'render_model': 'scalar_rgb; orthographic/path depth 4; textured diffuse + Beckmann roughplastic',
        'light_offset_radians': args.light_offset_radians,
        'camera_noise': {'full_well_electrons': 30000, 'read_noise_electrons': 2, 'adc_bits': 12},
        'solved_coverage': coverage, 'decision': audit['decision'], 'accepted': audit['accepted'],
        'metrics': {name: {'before': metric(baseline, truth, roi), 'after': metric(inverse, truth, roi)}
                    for name, roi in regions.items() if np.any(roi)},
        'normal_products': normal_product_metrics(mi, output, truth, regions),
        'command': command, 'app_sha256': hashlib.sha256(args.app.read_bytes()).hexdigest(),
        'worker_sha256': hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        'input_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in observations},
        'timing_seconds': {'fixture_render': render_seconds, 'application': application_seconds},
        'thresholds': {'minimum_solved_coverage': 0.9, 'minimum_cast_shadow_pixels': 12,
                       'maximum_height_rmse_ratio': 0.95, 'maximum_normal_mae_ratio': 0.95},
        'mitsuba_version': mi.__version__, 'inverse_audit': audit,
    }
    before, after = report['metrics']['all']['before'], report['metrics']['all']['after']
    failures = []
    if coverage < 0.9:
        failures.append('Fewer than 90% of fixed evaluation pixels were solved')
    if int((region & shadows).sum()) < 12:
        failures.append('Fixture did not produce enough independently ray-tested cast-shadow pixels')
    if not audit['accepted']:
        failures.append('Inverse correction was not accepted')
    if not audit.get('normal_prior_enabled', False):
        failures.append('Classical photometric normals did not reach the inverse constraint')
    elif not audit['normal_prior_loss_after'] <= 1.05 * audit['normal_prior_loss_before'] + 1e-6:
        failures.append('Accepted correction violated the measured-normal consistency guard')
    if not after['height_rmse_mm'] < 0.95 * before['height_rmse_mm']:
        failures.append('Height RMSE did not improve by at least 5%')
    if not after['normal_mae_deg'] < 0.95 * before['normal_mae_deg']:
        failures.append('Integrated-normal MAE did not improve by at least 5%')
    if not np.all(np.isfinite(inverse)):
        failures.append('Inverse height contains nonfinite values')
    report['passed'], report['failures'] = not failures, failures
    (root / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('passed', 'failures', 'solved_coverage', 'decision', 'metrics', 'normal_products')}, indent=2), flush=True)
    if failures:
        raise AssertionError('; '.join(failures) + f'; artifacts: {root}')


if __name__ == '__main__':
    main()
