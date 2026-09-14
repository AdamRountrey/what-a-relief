"""Real CPU renderer regression gates, not a mock or a clinical accuracy claim."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np

worker_path = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'tools/mitsuba_backend/worker.py'
spec = importlib.util.spec_from_file_location('worker', worker_path)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mi, _, _ = w.configure_backend('llvm')

    def test_finite_sources_have_shadow_derivatives(self):
        for model in ('near_field_ring', 'directional'):
            with self.subTest(model=model):
                ad, fd = w.shadow_gradient_probe(self.mi, model, finite_difference=True)
                ad_norm = float(np.mean(np.abs(ad)))
                fd_norm = float(np.mean(np.abs(fd)))
                cosine = float(np.sum(ad * fd) / np.sqrt(np.sum(ad * ad) * np.sum(fd * fd)))
                print('SHADOW_DERIVATIVE', model, ad_norm, fd_norm, cosine, flush=True)
                self.assertTrue(np.all(np.isfinite(ad)))
                self.assertGreater(fd_norm, 0.01)
                self.assertGreater(ad_norm / fd_norm, 0.5)
                self.assertLess(ad_norm / fd_norm, 1.5)
                self.assertGreater(cosine, 0.65)

    def test_normal_prior_ad_gradient_matches_finite_difference(self):
        import drjit as dr
        mi = self.mi
        rng = np.random.default_rng(935)
        height = rng.normal(0, 0.1, (7, 9)).astype(np.float32)
        mask = np.ones(height.shape, bool)
        mask[3, 4] = False
        weight = rng.uniform(0.1, 1, height.shape).astype(np.float32) * mask
        prior = np.tile([0.0, 0.0, 1.0], (*height.shape, 1)).astype(np.float32)
        spacing = (0.4, 0.7)
        prepared = {'mask_small': mask, 'world_spacing': spacing,
                    'normal_prior': prior, 'normal_prior_weight': weight}
        stencils = [(mi.UInt(a), mi.UInt(b), mi.Float(scale))
                    for a, b, scale in w.slope_stencils(mask, *spacing)]
        z = mi.Float(height.ravel())
        dr.enable_grad(z)
        loss = w.normal_prior_loss_ad(mi, dr, z,
                                     [mi.Float(prior[..., axis].ravel()) for axis in range(3)],
                                     mi.Float(weight.ravel()), stencils, float(weight.sum()))
        self.assertAlmostEqual(float(loss[0]), w.normal_prior_loss(prepared, height), delta=1e-6)
        dr.backward(loss)
        gradient = dr.grad(z).numpy().reshape(height.shape)
        epsilon = 1e-3
        for y, x in ((0, 0), (3, 3), (3, 4), (4, 4), (6, 8)):
            perturbation = np.zeros_like(height)
            perturbation[y, x] = epsilon
            fd = (w.normal_prior_loss(prepared, height + perturbation)
                  - w.normal_prior_loss(prepared, height - perturbation)) / (2 * epsilon)
            self.assertAlmostEqual(float(gradient[y, x]), fd, delta=2e-5)
        self.assertEqual(gradient[3, 4], 0)
        self.assertLess(w.normal_prior_loss(prepared, height - 0.1 * gradient), float(loss[0]))

    def test_inverse_tilt_recovery_and_truth_guard(self):
        mi = self.mi
        side = 24
        yy, xx = np.mgrid[:side, :side].astype(np.float32)
        truth = 7.0 + 0.15 * (xx - (side - 1) / 2) + 0.06 * (yy - (side - 1) / 2)
        albedo = (0.35 + 0.08 * ((xx.astype(int) // 5 + yy.astype(int) // 7) % 2)).astype(np.float32)
        mask = np.ones((side, side), bool)
        mask[3:6, 3:6] = False
        lights = np.array([[np.cos(a), np.sin(a), 1.0] for a in np.arange(8) * np.pi / 4], np.float32)
        lights /= np.linalg.norm(lights, axis=1, keepdims=True)
        settings = dict(w.QUALITY['standard'], max_side=side, control_spacing=8, iterations_ad=6)
        with tempfile.TemporaryDirectory(prefix='war-inverse-regression-') as temporary:
            root = Path(temporary)
            inputs = root / 'inputs'
            inputs.mkdir()
            w.write_pfm(inputs / 'height.pfm', truth)
            w.write_pfm(inputs / 'albedo.pfm', albedo)
            w.save_gray(mi, inputs / 'mask.png', mask.astype(np.float32))
            paths = [inputs / f'image_{i}.exr' for i in range(8)]
            validity = [inputs / f'valid_{i}.png' for i in range(8)]
            for path, valid in zip(paths, validity):
                mi.Bitmap(np.ones_like(truth) * 0.3).write(str(path))
                w.save_gray(mi, valid, np.ones_like(truth))
            job = {
                'schema_version': w.JOB_SCHEMA_VERSION, 'method': w.METHOD_ID,
                'inputs': {'images': list(map(str, paths)), 'observation_validity': list(map(str, validity)),
                           'height_pfm': str(inputs / 'height.pfm'), 'albedo_pfm': str(inputs / 'albedo.pfm'),
                           'mask_png': str(inputs / 'mask.png'), 'lights': lights.tolist()},
                'geometry': {'lighting_model': 'near_field_ring', 'pixel_scale_mm_per_pixel': 0.25,
                             'ring_radius_mm': 35, 'ring_height_mm': 45, 'reference_surface_z_mm': 10,
                             'reference_height_pixels': 7, 'led_diameter_mm': 2, 'angular_diameter_degrees': 1},
                'parameters': {'quality': 'regression', 'backend_selected': 'llvm', 'srgb_decode': False, 'height_scale': 1},
                'outputs': {'directory': str(root)},
            }
            prepared = w.prepare_job(mi, job, settings, lambda *args: None, root / 'renderer')
            self.assertAlmostEqual(prepared['geometry']['ring_height_world'] / prepared['scene_scale'], 35)
            self.assertEqual(int(np.count_nonzero(prepared['mask_small'])), side * side - 9)
            scenes, params = w.make_scenes(mi, prepared, 'llvm', 16)
            basis = w.render_basis_numpy(mi, scenes, params, prepared['baseline_positions'], 256, 901)
            # Spatial texture and two glossy coefficients vary independently.
            material = np.stack((albedo, 0.02 * (xx > 12), 0.03 * (yy > 12)), axis=-1)
            observations = w.prediction_numpy(basis, material)
            self.assertGreater(float(observations[:, 8:-3, 8:-3].mean()), 0.1)
            for path, observed in zip(paths, observations):
                mi.Bitmap(observed.astype(np.float32)).write(str(path))
            roi = prepared['weights'][0] > 0
            del scenes, params, basis, prepared
            w.QUALITY['regression'] = settings
            normal_paths = [inputs / f'normal_{axis}.pfm' for axis in range(3)]
            for axis, path in enumerate(normal_paths):
                w.write_pfm(path, w.surface_normals(truth, mask)[..., axis])
            w.save_gray(mi, inputs / 'clipped.png', np.ones_like(truth))
            w.save_gray(mi, inputs / 'invalid.png', np.zeros_like(truth))
            try:
                for name, baseline in [('tilted', np.full_like(truth, 7)),
                                       ('tilted_clipped_views', np.full_like(truth, 7)), ('truth', truth)]:
                    with self.subTest(start=name):
                        output = root / name
                        output.mkdir()
                        w.write_pfm(inputs / 'height.pfm', baseline)
                        if name.startswith('tilted'):
                            job['inputs']['normal_prior_pfm'] = list(map(str, normal_paths))
                        else:
                            job['inputs'].pop('normal_prior_pfm', None)
                        job['outputs']['directory'] = str(output)
                        job_path = output / 'job.json'
                        run_job = json.loads(json.dumps(job))
                        run_job['parameters']['live_preview'] = name == 'tilted'
                        if name == 'tilted_clipped_views':
                            for index in (1, 4):
                                run_job['inputs']['images'].insert(index, str(inputs / 'clipped.png'))
                                run_job['inputs']['observation_validity'].insert(index, str(inputs / 'invalid.png'))
                                run_job['inputs']['lights'].insert(index, lights[0].tolist())
                        job_path.write_text(json.dumps(run_job), encoding='utf-8')
                        self.assertEqual(w.run_job(job_path), 0)
                        if name == 'tilted':
                            for iteration in (5, 6):
                                preview = np.asarray(mi.Bitmap(str(output / f'iteration_{iteration}.png')))
                                self.assertEqual(preview.shape, (side, 2 * side, 3))
                                self.assertGreater(int(preview.max()) - int(preview.min()), 20)
                            self.assertFalse((output / 'iteration_1.png').exists())
                            live = json.loads((output / 'progress.txt').read_text().splitlines()[2])
                            self.assertEqual(live['iteration'], 6)
                            self.assertEqual(live['remaining_seconds'], 0)
                        else:
                            self.assertFalse(list(output.glob('iteration_*.png')))
                        result = json.loads((output / 'result.json').read_text())
                        reconstructed = w.read_pfm(output / 'inverse_height.pfm')
                        expected_normals = w.surface_normals(truth, mask)
                        actual_normals = w.surface_normals(reconstructed, mask)
                        error = np.degrees(np.arccos(np.clip(np.sum(expected_normals * actual_normals, axis=-1), -1, 1)))
                        mean_error = float(error[roi].mean())
                        delta = reconstructed[roi] - truth[roi]
                        height_rmse = float(np.sqrt(np.mean((delta - delta.mean()) ** 2)))
                        print('INVERSE_GEOMETRY', name, result['decision'], mean_error,
                              result['train_relative_improvement'], result['holdout_relative_improvement'],
                              'height_rmse_pixels', height_rmse, flush=True)
                        if name.startswith('tilted'):
                            baseline_error = np.degrees(np.arctan(np.hypot(0.15, 0.06)))
                            self.assertTrue(result['accepted'])
                            self.assertFalse(result['candidate_saved'])
                            self.assertFalse((output / 'unvalidated_candidate').exists())
                            self.assertLess(mean_error, 0.85 * baseline_error)
                            initial_delta = baseline[roi] - truth[roi]
                            self.assertLess(height_rmse, float(np.std(initial_delta)))
                            self.assertTrue(result['normal_prior_enabled'])
                            self.assertLess(result['normal_prior_loss_after'], result['normal_prior_loss_before'])
                            if name == 'tilted_clipped_views':
                                selection = result['light_selection']
                                self.assertEqual(selection['used_light_count'], 8)
                                self.assertEqual(selection['excluded_light_indices'], [1, 4])
                                self.assertTrue(result['holdout_light_indices'])
                                self.assertFalse({1, 4} & set(result['training_light_indices'] + result['holdout_light_indices']))
                        else:
                            self.assertLess(mean_error, 1.0)
                            self.assertLess(height_rmse, 0.05)
                            self.assertFalse(result['normal_prior_enabled'])  # Legacy jobs remain supported.
                            self.assertFalse(result['accepted'])
                            np.testing.assert_array_equal(reconstructed, np.where(mask, baseline, 0))
                            self.assertTrue(result['candidate_saved'])
                            candidate = output / result['candidate_directory']
                            candidate_height = w.read_pfm(candidate / 'candidate_height.pfm')
                            correction = w.read_pfm(candidate / 'height_correction.pfm')
                            np.testing.assert_array_equal(candidate_height, np.where(mask, baseline + correction, 0))
                            self.assertFalse(json.loads((candidate / 'candidate.json').read_text())['accepted'])
                            self.assertTrue((candidate / 'review.html').is_file())
                        self.assertTrue(np.all(np.isfinite(reconstructed)))
                        self.assertTrue(np.all(reconstructed[~mask] == 0))
                        self.assertEqual(result['reference_surface_z_mm'], 10)
                        self.assertFalse(result['baseline_outputs_modified'])
                        self.assertEqual(result['material_refit_interval'], settings['material_refit_interval'])
                        history = result['optimization_history']
                        self.assertEqual(history[0]['iteration'], 0)
                        self.assertEqual(history[-1]['iteration'], settings['iterations_ad'])
                        chosen = next(step for step in history if step['iteration'] == result['selected_iteration'])
                        self.assertEqual(chosen['objective'], min(step['objective'] for step in history))
            finally:
                del w.QUALITY['regression']


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Match the application: the device probe and reconstruction jobs run
        # in separate worker processes, with no shared renderer/AD scene state.
        for method in ('test_normal_prior_ad_gradient_matches_finite_difference',
                       'test_finite_sources_have_shadow_derivatives', 'test_inverse_tilt_recovery_and_truth_guard'):
            result = subprocess.run([sys.executable, '-X', 'faulthandler', '-B', str(Path(__file__).resolve()),
                                     str(worker_path.resolve()), 'RenderingTests.' + method], timeout=420)
            if result.returncode:
                raise SystemExit(1)
    else:
        unittest.main(verbosity=2)
