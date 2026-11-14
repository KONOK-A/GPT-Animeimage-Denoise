import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt_animeimage_denoise.filters import (
    blend,
    build_weight,
    gradient_preserving_candidate,
    keep_prompt_components,
)


class GradientPreservingTest(unittest.TestCase):
    def test_partial_restoration_scales_residual_before_clipping(self):
        original = np.full((1, 2, 3), 204, np.uint8)
        candidate = np.broadcast_to(np.array([0, 0.9], np.float32)[None, :, None], original.shape).copy()
        # At this radius the two residual samples are averaged: (0.8 - 0.1) / 2.
        partial = gradient_preserving_candidate(original, candidate, 12, restoration_gain=0.25)
        np.testing.assert_allclose(partial[0, :, 0], [0.0875, 0.9875], atol=0.001)
        skipped = gradient_preserving_candidate(original, candidate, 12, restoration_gain=0)
        np.testing.assert_array_equal(skipped, candidate)

    def test_invalid_restoration_gains_are_rejected(self):
        original = np.zeros((1, 1, 3), np.uint8)
        candidate = np.zeros_like(original, dtype=np.float32)
        for gain in (-0.1, 1.1, np.nan, np.inf, True, np.bool_(True)):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                gradient_preserving_candidate(original, candidate, 1, restoration_gain=gain)

    def test_restores_broad_skin_gradient_while_reducing_fine_noise(self):
        height, width = 96, 192
        ramp = np.linspace(-0.10, 0.10, width, dtype=np.float32)[None, :, None]
        clean = np.broadcast_to(np.array([0.72, 0.52, 0.40], np.float32) + ramp,
                                (height, width, 3)).copy()
        yy, xx = np.indices((height, width))
        noise = np.where((xx + yy) % 2, 0.018, -0.018)[..., None]
        original = np.rint((clean + noise) * 255).astype(np.uint8)
        flat = np.broadcast_to(np.array([0.72, 0.52, 0.40], np.float32), clean.shape).copy()
        restored = gradient_preserving_candidate(original, flat, sigma=6)
        region = np.s_[12:-12, 20:-20, :]
        low_error_before = np.mean(np.abs(gaussian_filter(flat - clean, (8, 8, 0))[region]))
        low_error_after = np.mean(np.abs(gaussian_filter(restored - clean, (8, 8, 0))[region]))
        noise_before = np.mean((original.astype(np.float32) / 255 - clean)[region] ** 2)
        noise_after = np.mean((restored - clean)[region] ** 2)
        self.assertLess(low_error_after, low_error_before * 0.1)
        self.assertLess(noise_after, noise_before * 0.1)
        self.assertEqual(restored.dtype, np.float32)

    def test_matching_candidate_is_unchanged_and_inputs_are_not_modified(self):
        original = np.random.default_rng(7).integers(0, 256, (23, 31, 3), np.uint8)
        candidate = original.astype(np.float32) / 255
        source_copy, candidate_copy = original.copy(), candidate.copy()
        actual = gradient_preserving_candidate(original, candidate, 4)
        np.testing.assert_array_equal(actual, candidate)
        np.testing.assert_array_equal(original, source_copy)
        np.testing.assert_array_equal(candidate, candidate_copy)

    def test_channels_do_not_bleed_and_output_stays_in_range(self):
        original = np.full((19, 29, 3), [26, 128, 230], dtype=np.uint8)
        candidate = np.full(original.shape, 0.5, np.float32)
        result = gradient_preserving_candidate(original, candidate, 12)
        np.testing.assert_allclose(result, original.astype(np.float32) / 255, atol=1e-6)
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)

    def test_matching_step_boundary_is_not_blurred_by_ramp_restoration(self):
        height, width = 32, 160
        ramp = np.linspace(-0.06, 0.06, width, dtype=np.float32)
        step = np.where(np.arange(width) < 80, 0.25, 0.70).astype(np.float32)
        clean = np.broadcast_to((step + ramp)[None, :, None], (height, width, 3))
        candidate = np.broadcast_to(step[None, :, None], clean.shape).copy()
        original = np.rint(clean * 255).astype(np.uint8)
        result = gradient_preserving_candidate(original, candidate, 5)
        self.assertGreater(float((result[:, 80] - result[:, 79]).mean()), 0.44)
        self.assertLess(float(result[:, :80].max()), 0.32)
        self.assertGreater(float(result[:, 80:].min()), 0.68)

    def test_small_and_narrow_images(self):
        for shape in [(1, 1, 3), (1, 17, 3), (13, 1, 3), (5, 7, 3)]:
            with self.subTest(shape=shape):
                source = np.full(shape, 127, np.uint8)
                result = gradient_preserving_candidate(source, np.full(shape, 0.4, np.float32), 3)
                np.testing.assert_allclose(result, 127 / 255, atol=1e-6)

    def test_rejects_invalid_images_and_parameters(self):
        source = np.zeros((5, 7, 3), np.uint8)
        candidate = np.zeros(source.shape, np.float32)
        for sigma in [0, -1, np.nan, np.inf]:
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                gradient_preserving_candidate(source, candidate, sigma)
        for bad in [candidate[..., :1], candidate[:-1], np.full_like(candidate, np.nan),
                    np.full_like(candidate, -0.1), np.full_like(candidate, 1.1)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                gradient_preserving_candidate(source, bad, 3)
        with self.assertRaises(TypeError):
            gradient_preserving_candidate(source.astype(np.float32), candidate, 3)
        with self.assertRaises(TypeError):
            gradient_preserving_candidate(source, source, 3)


class BlendTest(unittest.TestCase):
    def test_zero_weights_copy_original_and_full_weights_use_candidate(self):
        source = np.random.default_rng(9).integers(0, 256, (11, 13, 3), np.uint8)
        candidate = np.full(source.shape, [0.0, 0.5, 1.0], np.float32)
        weight = np.zeros(source.shape[:2], np.float32)
        weight[3:8, 4:9] = 1
        actual = blend(source, candidate, weight)
        np.testing.assert_array_equal(actual[weight == 0], source[weight == 0])
        np.testing.assert_array_equal(actual[weight == 1],
                                      np.rint(candidate[weight == 1] * 255).astype(np.uint8))
        self.assertEqual(actual.dtype, np.uint8)

    def test_midpoint_uses_linear_light(self):
        source = np.zeros((1, 1, 3), np.uint8)
        candidate = np.ones((1, 1, 3), np.float32)
        actual = blend(source, candidate, np.full((1, 1), 0.5, np.float32))
        np.testing.assert_array_equal(actual, np.full((1, 1, 3), 188, np.uint8))

    def test_inputs_are_read_only_and_same_candidate_round_trips(self):
        source = np.random.default_rng(4).integers(0, 256, (20, 15, 3), np.uint8)
        candidate = source.astype(np.float32) / 255
        weight = np.random.default_rng(6).random(source.shape[:2], dtype=np.float32)
        source.setflags(write=False); candidate.setflags(write=False); weight.setflags(write=False)
        np.testing.assert_array_equal(blend(source, candidate, weight), source)

    def test_invalid_weights_are_rejected(self):
        source = np.zeros((3, 5, 3), np.uint8)
        candidate = np.zeros(source.shape, np.float32)
        for weight in [np.zeros((3, 4), np.float32), np.full((3, 5), np.nan),
                       np.full((3, 5), -0.1), np.full((3, 5), 1.1)]:
            with self.subTest(shape=weight.shape), self.assertRaises(ValueError):
                blend(source, candidate, weight)


class WeightTest(unittest.TestCase):
    def test_line_protection_can_be_weakened_without_weakening_hard_protection(self):
        source = np.full((41, 81, 3), 40, np.uint8)
        source[:, 40:] = 210
        allowed = np.ones(source.shape[:2], bool)
        allowed[:4] = False
        protected = np.zeros_like(allowed)
        protected[12:16, 38:42] = True
        for protection in (0, 0.5, 1):
            with self.subTest(protection=protection):
                weight = build_weight(allowed, protected, source, feather_radius=0,
                                      structure_protection=protection)
                self.assertEqual(float(weight[20, 40]), 1 - protection)
                self.assertEqual(float(weight[20, 20]), 1)
                self.assertFalse(weight[~allowed | protected].any())

    def test_invalid_line_protection_strengths_are_rejected(self):
        source = np.zeros((1, 1, 3), np.uint8)
        mask = np.ones((1, 1), bool)
        for protection in (-0.1, 1.1, np.nan, np.inf, True, np.bool_(True)):
            with self.subTest(protection=protection), self.assertRaises(ValueError):
                build_weight(mask, ~mask, source, structure_protection=protection)

    def test_permission_boundary_and_protected_hole_do_not_leak(self):
        source = np.full((64, 72, 3), 150, np.uint8)
        allowed = np.zeros(source.shape[:2], bool); allowed[5:59, 5:67] = True
        protected = np.zeros_like(allowed); protected[25:35, 28:39] = True
        weight = build_weight(allowed, protected, source, feather_radius=8)
        self.assertEqual(weight.dtype, np.float32)
        self.assertTrue(np.isfinite(weight).all())
        self.assertEqual(float(weight[~allowed | protected].max()), 0)
        self.assertGreater(float(weight[5, 10]), 0)
        self.assertLess(float(weight[5, 10]), float(weight[15, 10]))
        self.assertEqual(float(weight[15, 15]), 1)
        output = blend(source, np.zeros(source.shape, np.float32), weight)
        np.testing.assert_array_equal(output[~allowed | protected], source[~allowed | protected])

    def test_persistent_step_edge_is_protected_but_flat_interiors_are_processed(self):
        source = np.full((41, 81, 3), 40, np.uint8); source[:, 40:] = 210
        allowed = np.ones(source.shape[:2], bool)
        weight = build_weight(allowed, np.zeros_like(allowed), source)
        self.assertTrue((weight[:, 39:41] == 0).all())
        self.assertTrue((weight[:, 10] == 1).all())
        self.assertTrue((weight[:, 70] == 1).all())
        changed = blend(source, np.full(source.shape, 0.5, np.float32), weight)
        np.testing.assert_array_equal(changed[:, 39:41], source[:, 39:41])

    def test_fine_noise_does_not_protect_an_entire_flat_surface(self):
        yy, xx = np.indices((64, 64))
        gray = np.where((xx + yy) % 2, 136, 120).astype(np.uint8)
        source = np.repeat(gray[..., None], 3, axis=2)
        allowed = np.ones(gray.shape, bool)
        weight = build_weight(allowed, np.zeros_like(allowed), source)
        self.assertTrue((weight[8:-8, 8:-8] == 1).all())

    def test_thin_high_contrast_line_center_remains_protected(self):
        source = np.full((31, 81, 3), 220, np.uint8); source[:, 40] = 20
        allowed = np.ones(source.shape[:2], bool)
        weight = build_weight(allowed, np.zeros_like(allowed), source)
        self.assertTrue((weight[:, 40] == 0).all())
        result = blend(source, np.full(source.shape, 0.5, np.float32), weight)
        np.testing.assert_array_equal(result[:, 40], source[:, 40])

    def test_all_or_no_permission_and_narrow_images(self):
        for shape in [(1, 1), (1, 19), (17, 1), (11, 13)]:
            with self.subTest(shape=shape):
                source = np.full((*shape, 3), 100, np.uint8)
                allowed = np.ones(shape, bool); empty = np.zeros(shape, bool)
                np.testing.assert_array_equal(build_weight(allowed, empty, source), np.ones(shape, np.float32))
                np.testing.assert_array_equal(build_weight(empty, empty, source), np.zeros(shape, np.float32))
                np.testing.assert_array_equal(build_weight(allowed, allowed, source), np.zeros(shape, np.float32))

    def test_zero_radii_disable_softening_but_keep_hard_veto(self):
        source = np.full((8, 10, 3), 0, np.uint8); source[:, 5:] = 255
        allowed = np.ones((8, 10), bool); protected = np.zeros_like(allowed); protected[2:4, 3:6] = True
        weight = build_weight(allowed, protected, source, feather_radius=0, structure_radius=0)
        np.testing.assert_array_equal(weight, (~protected).astype(np.float32))

    def test_rejects_invalid_masks_and_radius(self):
        source = np.zeros((5, 7, 3), np.uint8); allowed = np.ones((5, 7), bool)
        with self.assertRaises(ValueError):
            build_weight(allowed[:, :-1], ~allowed, source)
        with self.assertRaises(TypeError):
            build_weight(allowed.astype(np.uint8), ~allowed, source)
        for kwargs in [{'feather_radius': -1}, {'structure_radius': np.inf}, {'gradient_threshold': np.nan}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_weight(allowed, ~allowed, source, **kwargs)


class PromptComponentTest(unittest.TestCase):
    def test_keeps_multiple_prompted_components_and_discards_larger_unrelated_component(self):
        mask = np.zeros((20, 30), bool)
        mask[1:4, 2:5] = True; mask[15:18, 25:28] = True; mask[4:15, 8:22] = True
        original = mask.copy()
        result = keep_prompt_components(mask, [[3, 2], [26, 16]])
        self.assertEqual(int(result.sum()), 18)
        self.assertTrue(result[2, 3] and result[16, 26])
        self.assertFalse(result[8, 12])
        np.testing.assert_array_equal(mask, original)

    def test_diagonal_connectivity_and_background_points(self):
        mask = np.eye(5, dtype=bool)
        np.testing.assert_array_equal(keep_prompt_components(mask, [[0.2, 0.2]]), mask)
        self.assertFalse(keep_prompt_components(mask, [[4, 0]]).any())
        self.assertFalse(keep_prompt_components(mask, []).any())

    def test_rejects_invalid_points(self):
        mask = np.ones((3, 5), bool)
        for points in [[[-1, 0]], [[5, 0]], [[0, 3]], [[np.nan, 1]], [[1]], [[1, 2, 3]]]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                keep_prompt_components(mask, points)


if __name__ == '__main__':
    unittest.main()
