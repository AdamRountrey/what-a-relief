"""Fast numerical gates; requires NumPy but does not import a renderer."""
import importlib.util
import contextlib
import io
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
