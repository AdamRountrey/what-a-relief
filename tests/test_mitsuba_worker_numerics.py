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
    def test_ultra_is_1024_capped_absolute_height_mode(self):
        settings = w.QUALITY['ultra']
        self.assertEqual(settings['max_side'], 1024)
        self.assertEqual(settings['control_spacing'], 1)
        self.assertEqual(settings['geometry_mode'], 'absolute_height')
        self.assertEqual(w.target_size(73, 121, settings['max_side']), (73, 121))
        self.assertEqual(w.target_size(1200, 1600, settings['max_side']), (768, 1024))
        self.assertEqual(w.target_size(73, 121, 64), (39, 64))
        mapping = w.control_mapping(73, 121, settings['control_spacing'])
        self.assertEqual(mapping[:2], (73, 121))
        self.assertIsNone(mapping[2])
        control = np.arange(73 * 121, dtype=np.float32).reshape(73, 121)
        self.assertTrue(np.shares_memory(w.expand_control_numpy(control, mapping), control))
        self.assertEqual(w.effective_ultra_spp(1000, 32), 32)
        self.assertEqual(w.effective_ultra_spp(4_000_000, 32), 2)
        self.assertEqual(w.effective_ultra_spp(20_000_000, 32), 1)
        self.assertAlmostEqual(w.effective_ultra_learning_rate((0.1, 0.2), 0.004), 0.002)
        self.assertAlmostEqual(w.effective_ultra_learning_rate((0.5, 0.5), 0.004), 0.004)

    def test_drjit_sum_chunks_before_uint32_power_of_two_overflow(self):
        class FakeArray:
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

        class FakeDr:
            def __init__(self):
                self.blocks = []

            @staticmethod
            def ravel(value):
                return value

            def block_sum(self, value, block_size):
                self.blocks.append((len(value), block_size))
                return FakeArray((len(value) + block_size - 1) // block_size)

            @staticmethod
            def sum(value):
                return len(value)

        fake = FakeDr()
        source_size = (1 << 31) + 17
        partial_count = w.safe_dr_sum(fake, FakeArray(source_size))
        self.assertEqual(fake.blocks, [(source_size, w.DRJIT_REDUCTION_CHUNK)])
        self.assertEqual(partial_count,
                         (source_size + w.DRJIT_REDUCTION_CHUNK - 1)
                         // w.DRJIT_REDUCTION_CHUNK)

    def test_ultra_native_patch_marker_verifies_the_actual_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / 'Lib/site-packages/drjit/drjit-core.dll'
            library.parent.mkdir(parents=True)
            library.write_bytes(b'patched-drjit-core-test-payload')
            digest = hashlib.sha256(library.read_bytes()).hexdigest()
            (root / 'drjit-core-patch.json').write_text(json.dumps({
                'id': w.DRJIT_CORE_PATCH_ID,
                'source_commit': 'test-commit',
                'library_sha256': digest,
            }), encoding='utf-8')

            status = w.drjit_core_patch_status(root)
            self.assertTrue(status['valid'])
            self.assertEqual(status['library_sha256'], digest)

            library.write_bytes(b'tampered')
            self.assertFalse(w.drjit_core_patch_status(root)['valid'])

    def test_ultra_regularization_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(17)
        shape = (7, 9)
        mask = np.ones(shape, bool)
        mask[2, 3] = False
        fit_mask = w.triangle_vertex_support(mask)
        baseline = 0.03 * rng.normal(size=shape).astype(np.float32)
        height = baseline + 0.02 * rng.normal(size=shape).astype(np.float32)
        prior = w.surface_normals(baseline, mask)
        prior_weight = np.where(mask, rng.uniform(0.2, 1.0, size=shape), 0).astype(np.float32)
        prepared = {
            'fit_mask': fit_mask,
            'mask_small': mask,
            'baseline_positions': np.dstack((np.zeros(shape), np.zeros(shape), baseline)),
            'world_spacing': (0.7, 1.3),
            'normal_prior': prior,
            'normal_prior_weight': prior_weight,
        }
        settings = {'datum_strength': 0.8, 'curvature_strength': 0.003}
        geometry, prior_loss, gradient = w.absolute_height_regularization_gradient_numpy(
            prepared, height, settings)
        self.assertAlmostEqual(
            geometry, w.absolute_height_regularization_numpy(prepared, height, settings), places=7)
        self.assertAlmostEqual(prior_loss, w.normal_prior_loss(prepared, height), places=7)

        def objective(values):
            return (w.absolute_height_regularization_numpy(prepared, values, settings)
                    + w.NORMAL_PRIOR_STRENGTH * w.normal_prior_loss(prepared, values))

        epsilon = 2.0e-4
        for y, x in ((0, 0), (1, 4), (3, 5), (5, 7), (6, 8)):
            if not fit_mask[y, x]:
                continue
            plus = height.copy()
            minus = height.copy()
            plus[y, x] += epsilon
            minus[y, x] -= epsilon
            finite_difference = (objective(plus) - objective(minus)) / (2 * epsilon)
            self.assertAlmostEqual(float(gradient[y, x]), finite_difference, delta=2.0e-4)

    def test_absolute_solution_is_not_added_to_classical_height(self):
        baseline = np.full((6, 8), 11.0, np.float32)
        mask = np.zeros_like(baseline, bool)
        mask[1:5, 1:7] = True
        prepared = {
            'height': baseline,
            'mask': mask,
            'bounds': (1, 1, 7, 5),
            'scene_scale': 0.5,
            'physical_scale': 0.25,
            'height_datum': 3.0,
        }
        yy, xx = np.mgrid[:4, :6].astype(np.float32)
        intended = 4.0 + 0.2 * xx - 0.1 * yy
        optimized_world = (intended - prepared['height_datum']) * 0.125
        candidate, difference = w.full_height_solution(prepared, optimized_world, True)
        np.testing.assert_allclose(candidate[1:5, 1:7], intended, atol=1e-6)
        np.testing.assert_allclose(difference[mask], candidate[mask] - baseline[mask], atol=1e-6)
        self.assertFalse(np.any(candidate[~mask]))
        # The same values treated as a correction would produce a different result.
        additive, _ = w.full_height_solution(prepared, optimized_world, False)
        self.assertGreater(float(np.max(np.abs(additive[mask] - candidate[mask]))), 1.0)
        prepared['fit_mask'] = np.ones((4, 6), bool)
        prepared['fit_mask'][2, 3] = False
        candidate, difference = w.full_height_solution(prepared, optimized_world, True)
        self.assertEqual(candidate[3, 4], baseline[3, 4])
        self.assertEqual(difference[3, 4], 0)

    def test_absolute_solution_resamples_fitting_mask_to_source_crop(self):
        source_height, source_width = 15, 20
        yy, xx = np.mgrid[:source_height, :source_width].astype(np.float32)
        baseline = 2.0 + 0.03 * xx - 0.02 * yy
        mask = np.ones_like(baseline, dtype=bool)
        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False
        fit_mask = np.ones((6, 8), dtype=bool)
        fit_mask[2, 3] = False
        optimized = np.arange(48, dtype=np.float32).reshape(6, 8) / 10.0
        prepared = {
            'height': baseline,
            'mask': mask,
            'bounds': (0, 0, source_width, source_height),
            'scene_scale': 1.0,
            'physical_scale': 1.0,
            'height_datum': 0.0,
            'fit_mask': fit_mask,
        }

        candidate, difference = w.full_height_solution(prepared, optimized, True)
        expanded_height = w.resize_bilinear(optimized, source_height, source_width)
        expanded_support = (
            w.resize_nearest(fit_mask.astype(np.float32), source_height, source_width) >= 0.5)
        expanded_support &= mask
        expected = np.where(mask, baseline, 0.0).astype(np.float32)
        expected[expanded_support] = expanded_height[expanded_support]
        np.testing.assert_allclose(candidate, expected, atol=1e-6)
        np.testing.assert_allclose(difference[mask], candidate[mask] - baseline[mask], atol=1e-6)
        self.assertFalse(np.any(candidate[~mask]))
        self.assertFalse(np.any(difference[~mask]))

    def test_iteration_preview_uses_world_spacing_and_fixed_normal_encoding(self):
        yy, xx = np.mgrid[:12, :16].astype(np.float32)
        mask = np.ones_like(xx, dtype=bool)
        mask[3:5, 6:8] = False
        height = 10 + 0.2 * xx * 0.3 - 0.1 * yy * 0.7
        original = height.copy()
        image = w.iteration_preview(height, mask, (0.7, 0.3))
        expected = np.array([-0.2, -0.1, 1.0])
        expected = 0.5 + 0.5 * expected / np.linalg.norm(expected)
        np.testing.assert_allclose(image[2, 2], expected, atol=1e-5)
        self.assertEqual(image.shape, (12, 32, 3))
        self.assertTrue(np.all(np.isfinite(image)))
        np.testing.assert_allclose(image[3, 6], 0.12)
        np.testing.assert_array_equal(height, original)
        np.testing.assert_allclose(w.iteration_preview(height + 100, mask, (0.7, 0.3)), image, atol=1e-5)

    def test_iteration_preview_is_resized_to_gui_decode_limits(self):
        yy, xx = np.mgrid[:60, :200].astype(np.float32)
        frame = np.stack((xx / 199, yy / 59, np.full_like(xx, 0.25)), axis=-1)
        resized = w.fit_preview_frame(frame, maximum_height=50, maximum_width=80)
        self.assertEqual(resized.shape, (24, 80, 3))
        self.assertEqual(resized.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(resized)))
        self.assertGreaterEqual(float(np.min(resized)), 0.0)
        self.assertLessEqual(float(np.max(resized)), 1.0)
        small = frame[:20, :30]
        np.testing.assert_array_equal(w.fit_preview_frame(small, 50, 80), small)

    @staticmethod
    def light_support_fixture(count=8):
        angle = 2 * np.pi * np.arange(count) / count
        lights = np.column_stack((0.6 * np.cos(angle), 0.6 * np.sin(angle), np.full(count, 0.8)))
        return dict(lights=lights.astype(np.float32), weights=np.ones((count, 10, 10), np.float32),
                    images_small=np.arange(count * 100, dtype=np.float32).reshape(count, 10, 10))

    def test_inverse_selects_supported_lights_before_splitting(self):
        prepared = self.light_support_fixture(10)
        prepared['weights'][1] = 0
        prepared['weights'][4].flat[63:] = 0
        prepared['weights'][7].flat[64:] = 0
        original = {key: value.copy() for key, value in prepared.items()}
        result = {}
        selected, train, holdout = w.select_supported_lights(prepared, result)
        used = [0, 2, 3, 5, 6, 7, 8, 9]
        self.assertEqual(result['light_selection']['excluded_light_indices'], [1, 4])
        self.assertEqual(result['light_selection']['supported_pixels_per_light'][4], 63)
        self.assertEqual(result['light_selection']['supported_pixels_per_light'][7], 64)
        self.assertEqual(selected['light_indices'], used)
        self.assertEqual(selected['input_light_count'], 10)
        self.assertEqual(result['training_light_indices'], [used[i] for i in train])
        self.assertEqual(result['holdout_light_indices'], [used[i] for i in holdout])
        self.assertTrue(holdout)
        self.assertFalse(set(train) & set(holdout))
        self.assertEqual(sorted(train + holdout), list(range(len(used))))
        for key in original:
            np.testing.assert_array_equal(prepared[key], original[key])
            np.testing.assert_array_equal(selected[key], original[key][used])
        # Selection cannot inspect image brightness, residuals, or validation scores.
        prepared['images_small'] *= -100
        second = {}
        w.select_supported_lights(prepared, second)
        self.assertEqual(second, result)

    def test_inverse_support_selection_keeps_guards_and_failure_audit(self):
        prepared = self.light_support_fixture()
        unchanged, train, holdout = w.select_supported_lights(prepared, {})
        self.assertEqual((train, holdout), w.split_lights(8))
        np.testing.assert_array_equal(unchanged['lights'], prepared['lights'])
        for count in (0, 5):
            with self.subTest(usable=count):
                prepared = self.light_support_fixture()
                prepared['weights'][count:] = 0
                result = {}
                with self.assertRaisesRegex(ValueError, f'Only {count} of 8 lights'):
                    w.select_supported_lights(prepared, result)
                self.assertEqual(result['light_selection']['used_light_count'], count)
                self.assertNotIn('accepted', result)
        prepared = self.light_support_fixture()
        prepared['lights'][:] = [0, 0, 1]
        with self.assertRaisesRegex(ValueError, 'poorly conditioned'):
            w.select_supported_lights(prepared, {})
        prepared = self.light_support_fixture()
        prepared['weights'][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
            w.select_supported_lights(prepared, {})

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

    def test_curvature_stencils_preserve_planes_and_do_not_cross_holes(self):
        yy, xx = np.mgrid[:9, :11].astype(np.float32)
        height = 2.0 + 0.3 * xx - 0.2 * yy
        mask = np.ones(height.shape, bool)
        mask[4, 5] = False
        for left, center, right, scale, valid in w.curvature_stencils(mask, 0.7, 0.3):
            second = (height.ravel()[left] - 2 * height.ravel()[center] +
                      height.ravel()[right]) * scale
            np.testing.assert_allclose(second[valid > 0], 0, atol=1e-5)
            self.assertTrue(np.all(second[valid == 0] == 0))
        prepared = {'mask_small': mask, 'world_spacing': (0.7, 0.3),
                    'baseline_positions': np.dstack((xx, yy, height))}
        settings = {'datum_strength': 1.0, 'curvature_strength': 0.002}
        self.assertLess(w.absolute_height_regularization_numpy(prepared, height, settings), 1e-10)
        self.assertAlmostEqual(
            w.absolute_height_regularization_numpy(prepared, height + 3, settings), 9.0, places=5)

    def test_ultra_pixel_support_includes_mesh_boundary_but_not_isolated_pixels(self):
        mask = np.zeros((8, 10), bool)
        mask[1:7, 2:8] = True
        mask[3:5, 4:6] = False
        mask[0, 0] = True
        support = w.triangle_vertex_support(mask)
        self.assertFalse(support[0, 0])
        self.assertTrue(np.all(support[1, 2:8]))
        self.assertTrue(np.all(support[6, 2:8]))
        self.assertFalse(np.any(support[3:5, 4:6]))
        self.assertTrue(np.all(~support | mask))

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
