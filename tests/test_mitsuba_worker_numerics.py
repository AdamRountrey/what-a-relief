"""Fast numerical gates; requires NumPy but does not import a renderer."""
import importlib.util
import contextlib
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

worker_path = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'tools/mitsuba_backend/worker.py'
spec = importlib.util.spec_from_file_location('worker', worker_path)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class NumericalTests(unittest.TestCase):
    def test_rejected_candidate_is_retained_without_promoting_geometry(self):
        yy, xx = np.mgrid[:9, :11].astype(np.float32)
        baseline = 7 + 0.1 * xx - 0.2 * yy
        correction = 0.4 * np.exp(-((xx - 5)**2 + (yy - 4)**2) / 6)
        mask = np.ones(baseline.shape, bool)
        mask[3, 4] = False
        correction[~mask] = 0
        albedo = np.full_like(baseline, 0.4)
        original = baseline.copy()
        for decision in ('rejected_withheld_lights_worsened', 'rejected_insufficient_training_improvement',
                         'rejected_normal_prior_worsened', 'rejected_independent_validation_seed',
                         'rejected_excessive_slope_change', 'rejected_excessive_height_change'):
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                # Exercise the real PFM/PLY writers; only renderer-dependent PNG encoding is stubbed.
                def png(_mi, path, values):
                    path.write_bytes(np.asarray(values, np.float32).tobytes())
                result = dict(accepted=False, status='complete', decision=decision, selected_iteration=12)
                with patch.object(w, 'save_gray', side_effect=png), patch.object(w, 'save_rgb', side_effect=png):
                    w.output_products(None, output, baseline, np.zeros_like(baseline), mask, albedo, 2)
                    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
                    w.retain_unvalidated_candidate(None, output, baseline, correction, mask, albedo, 2, result)
                self.assertFalse(result['accepted'])
                self.assertEqual(result['decision'], decision)
                self.assertEqual(result['delivered_geometry'], 'baseline')
                self.assertTrue(result['candidate_saved'])
                candidate = output / result['candidate_directory']
                expected = np.where(mask, baseline + correction, 0)
                np.testing.assert_array_equal(w.read_pfm(candidate / 'candidate_height.pfm'), expected)
                np.testing.assert_array_equal(w.read_pfm(candidate / 'height_correction.pfm'), correction)
                np.testing.assert_array_equal(baseline, original)
                for name, digest in hashes.items():
                    self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
                provenance = json.loads((candidate / 'candidate.json').read_text())
                self.assertFalse(provenance['accepted'])
                self.assertFalse(provenance['selected_for_default_outputs'])
                self.assertEqual(provenance['validation_status'], 'unvalidated_candidate')
                self.assertEqual(provenance['selected_iteration'], 12)
                self.assertEqual(provenance['mesh_z_scale'], 2)
                header, data = (candidate / 'candidate_surface.ply').read_bytes().split(b'end_header\n', 1)
                self.assertIn(b'UNVALIDATED CANDIDATE', header)
                vertices = np.frombuffer(data, dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))], count=int(mask.sum()))
                ys, xs = np.nonzero(mask)
                np.testing.assert_allclose(vertices['xyz'], np.stack((xs, -ys, 2 * expected[mask]), axis=-1))
                page = (candidate / 'review.html').read_text()
                self.assertIn(decision, page)
                self.assertIn('id="files" class="files" hidden', page)
                self.assertIn('not the original photometric normal maps', page)
                self.assertNotIn('https://', page)

    def test_candidate_not_advertised_on_acceptance_nonfinite_or_write_failure(self):
        height = np.ones((5, 7), np.float32)
        mask = np.ones(height.shape, bool)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for accepted, correction, status in ((True, height, 'not_needed_accepted'),
                                                (False, height * np.nan, 'not_saved_nonfinite_geometry'),
                                                (False, height * np.inf, 'not_saved_nonfinite_geometry')):
                result = dict(accepted=accepted, decision='test')
                w.retain_unvalidated_candidate(None, output, height, correction, mask, height, 1, result)
                self.assertEqual(result['accepted'], accepted)
                self.assertFalse(result['candidate_saved'])
                self.assertEqual(result['candidate_export_status'], status)
                self.assertFalse((output / 'unvalidated_candidate').exists())
            result = dict(accepted=False, decision='test')
            with patch.object(w, 'output_products', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    w.retain_unvalidated_candidate(None, output, height, height, mask, height, 1, result)
            self.assertFalse(result['candidate_saved'])
            self.assertFalse(result['accepted'])

    def test_comparison_uses_shared_range_and_report_escapes_text(self):
        baseline = np.arange(20, dtype=np.float32).reshape(4, 5)
        mask = np.ones(baseline.shape, bool)
        mask[0, 0] = False
        previews, limits = w.comparison_stretch(baseline, baseline + 5, mask)
        self.assertTrue(np.all(previews[1][mask] >= previews[0][mask]))
        self.assertGreater(float(np.mean(previews[1][mask] - previews[0][mask])), 0.1)
        self.assertEqual(previews[0][0, 0], 0)
        self.assertGreater(limits[1], float(baseline.max()))
        with tempfile.TemporaryDirectory() as directory:
            w.write_candidate_review(Path(directory), dict(decision='<script>alert(1)</script>', train_loss_after=float('nan')))
            page = (Path(directory) / 'review.html').read_text()
            self.assertNotIn('<script>alert(1)</script>', page)
            self.assertIn('&lt;script&gt;', page)
            self.assertIn('Not evaluated', page)
            w.write_candidate_review(Path(directory), dict(decision='test', normal_prior_enabled=False,
                                                           normal_prior_loss_before=0, normal_prior_loss_after=0))
            self.assertIn('Not used', (Path(directory) / 'review.html').read_text())

    def test_unavailable_runtime_produces_actionable_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / 'probe.json'
            with patch.object(w, 'configure_backend', side_effect=RuntimeError('Reinstall the current backend')), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(w.probe('llvm', result), 1)
            data = json.loads(result.read_text())
            self.assertEqual(data['status'], 'unavailable')
            self.assertIn('Reinstall', data['error'])

    def test_bounded_fit_kkt(self):
        rng = np.random.default_rng(82)
        x = rng.uniform(0, 1, (300, 8, 3)).astype(np.float32)
        x[..., 1] *= x[..., 0]
        y = rng.uniform(0, 0.7, (300, 8, 1)).astype(np.float32)
        a = np.swapaxes(x, -1, -2) @ x + np.diag([0.08, 0.14, 0.14])
        b = (np.swapaxes(x, -1, -2) @ y)[..., 0] + np.array([0.032, 0, 0])
        upper = np.array([0.55, 0.25, 0.25])
        c = w.bounded_material_solve(a, b, upper)
        gradient = (a @ c[..., None])[..., 0] - b
        self.assertTrue(np.all((c >= 0) & (c <= upper + 1e-7)))
        free = (c > 1e-6) & (c < upper - 1e-6)
        self.assertLess(float(np.max(np.abs(gradient[free]))), 1e-5)
        self.assertTrue(np.all(gradient[c <= 1e-6] >= -1e-5))
        self.assertTrue(np.all(gradient[c >= upper - 1e-6] <= 1e-5))

    def test_excluded_clipping_cannot_change_material(self):
        rng = np.random.default_rng(3)
        basis = rng.uniform(0, 1, (3, 8, 2, 2)).astype(np.float32)
        images = 0.4 * basis[0]
        weights = np.ones_like(images)
        weights[1] = 0
        mask = np.ones((2, 2), bool)
        before, _ = w.fit_material_maps(basis, images, mask, np.full((2, 2), 0.4), list(range(7)), weights)
        images[1] = 100
        images[7] = 200  # Neither invalid nor held-out values may set fit bounds.
        after, _ = w.fit_material_maps(basis, images, mask, np.full((2, 2), 0.4), list(range(7)), weights)
        np.testing.assert_allclose(before, after, atol=1e-6)

    def test_mask_normals_are_datum_invariant(self):
        mask = np.zeros((20, 20), bool)
        mask[3:17, 4:16] = True
        mask[8:10, 8:10] = False
        for height in (0, 100, -100):
            normals = w.surface_normals(np.where(mask, height, 0).astype(np.float32), mask)
            np.testing.assert_allclose(normals[mask], np.tile([0, 0, 1], (mask.sum(), 1)), atol=1e-7)

    def test_normal_prior_stencils_match_masked_slopes(self):
        rng = np.random.default_rng(632)
        height = rng.normal(size=(9, 11)).astype(np.float32)
        mask = rng.uniform(size=height.shape) > 0.25
        for sy, sx in ((1, 1), (0.3, 0.7)):
            expected = w.masked_slopes(height, mask, sy, sx)
            for (a, b, scale), derivative in zip(w.slope_stencils(mask, sy, sx), expected):
                actual = ((height.ravel()[b] - height.ravel()[a]) * scale).reshape(height.shape)
                np.testing.assert_allclose(actual, derivative, atol=1e-6)
                self.assertTrue(np.all(actual[~mask] == 0))

    def test_normal_prior_reduction_and_invalid_support(self):
        mask = np.ones((8, 10), bool)
        mask[:2] = False
        confidence = np.ones(mask.shape, np.float32)
        confidence[2:4] = 0
        normal = np.array([-0.3, 0.4, np.sqrt(0.75)], np.float32)
        components = [np.full(mask.shape, n, np.float32) for n in normal]
        # Unsupported values must not leak through area reduction.
        for component in components:
            component[:4] = np.nan
        components[2][:4] = normal[2]
        components[2][4:6, 2:4] = 0.1  # Grazing normals are not a reliable prior.
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'normal_{axis}.pfm' for axis in range(3)]
            for path, component in zip(paths, components):
                w.write_pfm(path, component)
            inputs = {'normal_prior_pfm': list(map(str, paths))}
            prior, weight = w.load_normal_prior(inputs, mask, confidence, (2, 2, 10, 8), (3, 4))
            self.assertTrue(np.all(weight[0] == 0))
            self.assertEqual(weight[1, 0], 0)
            np.testing.assert_allclose(weight[weight > 0], normal[2]**2, atol=1e-6)
            np.testing.assert_allclose(prior[weight > 0], np.tile(normal, (np.count_nonzero(weight), 1)), atol=1e-6)
            self.assertTrue(np.all(np.isfinite(prior)))
            empty, empty_weight = w.load_normal_prior({}, mask, confidence, (0, 0, 10, 8), (3, 4))
            self.assertFalse(np.any(empty))
            self.assertFalse(np.any(empty_weight))
            with self.assertRaises(ValueError):
                w.load_normal_prior({'normal_prior_pfm': paths[:2]}, mask, confidence, (0, 0, 10, 8), (3, 4))
            components[0][6, 6] = np.nan
            w.write_pfm(paths[0], components[0])
            with self.assertRaises(ValueError):
                w.load_normal_prior(inputs, mask, confidence, (0, 0, 10, 8), (3, 4))
            w.write_pfm(paths[0], np.ones((2, 2), np.float32))
            with self.assertRaises(ValueError):
                w.load_normal_prior(inputs, mask, confidence, (0, 0, 10, 8), (3, 4))

    def test_normal_prior_loss_uses_physical_spacing_and_preserves_datum(self):
        yy, xx = np.mgrid[:9, :11].astype(np.float32)
        height = 0.2 * xx * 0.7 - 0.1 * yy * 0.3
        mask = np.ones(height.shape, bool)
        mask[4, 5] = False
        normal = np.array([-0.2, -0.1, 1], np.float32)
        normal /= np.linalg.norm(normal)
        prepared = {'mask_small': mask, 'world_spacing': (0.3, 0.7),
                    'normal_prior': np.broadcast_to(normal, (*height.shape, 3)),
                    'normal_prior_weight': mask.astype(np.float32)}
        for datum in (0, 7, -7):
            self.assertLess(w.normal_prior_loss(prepared, height + datum), 1e-10)
        self.assertGreater(w.normal_prior_loss(prepared, np.zeros_like(height)), 0.01)

    def test_material_refit_uses_training_only_and_never_increases_data_loss(self):
        rng = np.random.default_rng(734)
        basis = rng.uniform(0.01, 1, (3, 8, 5, 6)).astype(np.float32)
        truth = rng.uniform(0.03, 0.5, (5, 6, 3)).astype(np.float32)
        observed = w.prediction_numpy(basis, truth)
        weights = np.ones_like(observed)
        weights[2, 1:3, 2:4] = 0
        mask = np.ones((5, 6), bool)
        initial = np.full_like(truth, 0.15)
        train = list(range(1, 8))
        albedo = np.full((5, 6), 0.4, np.float32)
        result, loss = w.refit_training_material(basis, observed, mask, albedo, train, weights, initial)
        before = w.data_loss_numpy(w.prediction_numpy(basis, initial), observed, weights, train)
        self.assertLessEqual(loss, before)
        changed = observed.copy()
        changed[0] = 1000
        changed[2, 1:3, 2:4] = 500
        other, other_loss = w.refit_training_material(basis, changed, mask, albedo, train, weights, initial)
        np.testing.assert_array_equal(result, other)
        self.assertEqual(loss, other_loss)
        # An exact current material must not be displaced by a biased ridge fit.
        retained, retained_loss = w.refit_training_material(basis, observed, mask, albedo, train, weights, truth)
        np.testing.assert_array_equal(retained, truth)
        self.assertLess(retained_loss, 1e-8)

    def test_slope_units_across_resolution(self):
        for size in (128, 1024, 4096):
            # Coarse values are original-height pixels at render-pixel centers.
            samples = (np.arange(128) + 0.5) * size / 128 - 0.5
            coarse = np.tile((0.02 * samples).astype(np.float32), (16, 1))
            full = w.resize_bilinear(coarse, 16, size)
            q, p = w.masked_slopes(full, np.ones(full.shape, bool))
            rms = np.sqrt(np.mean(p * p + q * q))
            self.assertAlmostEqual(float(rms), 0.02, delta=0.001)

    def test_area_reduction_and_mask_support(self):
        image = (np.indices((512, 512)).sum(axis=0) % 2).astype(np.float32)
        np.testing.assert_allclose(w.resize_area(image, 32, 32), 0.5, atol=1e-7)
        np.testing.assert_allclose(w.resize_area(np.roll(image, 1, 0), 32, 32), 0.5, atol=1e-7)
        mask = np.ones((12, 12), bool)
        mask[4:6, 4:6] = False
        result = w.masked_resize(np.where(mask, 12, -1000), mask, 6, 6)
        np.testing.assert_allclose(result[result != 0], 12, atol=1e-6)

    def test_mesh_never_spans_unsupported_vertices(self):
        mask = np.ones((5, 5), bool)
        mask[2, 2] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mesh.ply'
            w.write_grid_ply(path, np.arange(5), -np.arange(5), np.zeros((5, 5)), mask)
            with path.open('rb') as stream:
                header = b''
                while not header.endswith(b'end_header\n'):
                    header += stream.readline()
                nfaces = int(header.split(b'element face ')[1].splitlines()[0])
                np.fromfile(stream, '<f4', count=25 * 3)
                records = np.fromfile(stream, dtype=[('count', 'u1'), ('indices', '<i4', (3,))], count=nfaces)
                self.assertLess(nfaces, 32)
                self.assertTrue(np.all(mask.ravel()[records['indices']]))


if __name__ == '__main__':
    unittest.main()
