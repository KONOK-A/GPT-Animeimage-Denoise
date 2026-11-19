"""Probability fusion behavior, using tiny original-grid fixtures (no models)."""
import importlib
import unittest
from unittest.mock import patch

import numpy as np

from gpt_animeimage_denoise.grouping import processing_regions


def evidence(labels, original=None, see=None, native=None):
    labels = np.asarray(labels, np.uint8)
    def scores(values):
        values = np.asarray(values)
        result = np.full((12, *values.shape), .001, np.float32)
        np.put_along_axis(result, values[None], .989, axis=0)
        return np.log(result).astype(np.float32)
    return {'alpha': np.ones(labels.shape, np.float32),
            'see_logits': np.full((19, *labels.shape), -4, np.float32) if see is None else see,
            'm2f_white': scores(labels) if native is None else native,
            'm2f_original': None if original is None else scores(original),
            'reference_labels12': labels.copy(), 'provenance': {}}


class FusionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('gpt_animeimage_denoise.fusion')
        except ModuleNotFoundError:
            self.module = None
        self.assertIsNotNone(self.module, 'Probability-preserving fusion module must exist')

    def legacy(self, ev, policy=None):
        return self.module.fuse_evidence(ev, {**(policy or {}), 'semantic_repair': 'off'})

    def test_reference_reproduces_existing_regions_and_exports_new_schema(self):
        labels = np.arange(12, dtype=np.uint8)[None]
        ev = evidence(labels)
        ev['see_logits'][3, 0, 4:6] = .25
        ev['see_logits'][18, 0, 10] = .25
        old = processing_regions(labels, ev['see_logits'][3] > 0, ev['see_logits'][18] > 0)
        result = self.module.reference_regions(ev)
        for key in ('person_labels', 'allowed', 'protected'):
            np.testing.assert_array_equal(result[key], old[key])
        np.testing.assert_array_equal(result['raw_labels'], old['labels'])
        for key in ('consensus', 'conflict', 'uncertain', 'objects'):
            self.assertEqual(result[key].dtype, bool)
        self.assertEqual(result['source_map'].dtype, np.uint8)
        self.assertEqual(result['sam_records'], [])

    def test_six_groups_brows_nose_and_mouth_keep_business_mapping(self):
        result = self.legacy(evidence([list(range(12))]))
        np.testing.assert_array_equal(result['person_labels'], [[0, 1, 2, 3, 4, 4, 3, 3, 2, 5, 6, 6]])
        np.testing.assert_array_equal(result['allowed'], [[False, True, True, True, False, False, True, True, True, False, False, False]])

    def test_auxiliary_absence_does_not_erase_main_skin_hair_or_outfit(self):
        ev = evidence([[1, 3, 10]], original=[[0, 0, 0]])
        ev['m2f_original'][:] = np.log(np.float32(1 / 12))
        result = self.legacy(ev)
        np.testing.assert_array_equal(result['person_labels'], [[1, 3, 6]])
        self.assertFalse(result['conflict'].any())

    def test_same_group_subclass_probability_shares_are_combined_for_confidence(self):
        for shares, raw, group in [({4: .45, 5: .45, 0: .1}, 4, 4),
                                   ({10: .4, 11: .4, 0: .2}, 10, 6),
                                   ({2: .4, 8: .4, 0: .2}, 2, 2)]:
            with self.subTest(shares=shares):
                ev = evidence([[raw]])
                probabilities = np.full((12, 1, 1), 1e-6, np.float32)
                for index, value in shares.items():
                    probabilities[index] = value
                probabilities /= probabilities.sum(axis=0, keepdims=True)
                ev['m2f_white'] = np.log(probabilities)
                result = self.legacy(ev)
                self.assertEqual(result['person_labels'][0, 0], group)
                self.assertEqual(result['_m2f_labels'][0, 0], raw)
                self.assertFalse(result['uncertain'][0, 0])

    def test_upsampled_view_keeps_exact_confidence_with_one_resize_buffer(self):
        module = self.module
        native = np.random.default_rng(32).normal(size=(12, 5, 7)).astype(np.float32)
        native[4] = native[5]  # Keep ties within the same business group.
        unmodified = native.copy()
        shape = (13, 19)
        resize = module.cv2.resize
        scores = np.stack([resize(plane, shape[::-1], interpolation=module.cv2.INTER_LINEAR) for plane in native])
        expected = module._view(scores, shape, module._policy(), return_scores=True)
        with patch.object(module.cv2, 'resize', wraps=resize) as resize_into:
            actual = module._view(native, shape, module._policy(), return_scores=True)
            for reference, result in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(result, reference)
            self.assertEqual(resize_into.call_count, 12)
            self.assertTrue(all(call.kwargs.get('dst') is not None for call in resize_into.call_args_list))
        np.testing.assert_array_equal(native, unmodified)

    def test_different_group_probability_competition_stays_unknown(self):
        ev = evidence([[4]])
        probabilities = np.full((12, 1, 1), 1e-6, np.float32)
        probabilities[4] = .4
        probabilities[5] = .1
        probabilities[10] = .3
        probabilities[11] = .2
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        ev['m2f_white'] = np.log(probabilities)
        result = self.legacy(ev)
        self.assertEqual(result['person_labels'][0, 0], 255)
        self.assertTrue(result['protected'][0, 0])

    def test_group_confidence_keeps_native_argmax_mapping_when_another_group_sums_higher(self):
        ev = evidence([[1]])
        probabilities = np.full((12, 1, 1), 1e-6, np.float32)
        probabilities[1] = .4
        probabilities[4] = .3
        probabilities[5] = .3
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        ev['m2f_white'] = np.log(probabilities)
        result = self.legacy(ev, {'m2f': .35, 'margin': 0})
        self.assertEqual(result['_m2f_labels'][0, 0], 1)
        self.assertEqual(result['raw_labels'][0, 0], 1)
        self.assertEqual(result['person_labels'][0, 0], 255)

    def test_clear_incompatible_views_are_unknown_and_protected(self):
        result = self.legacy(evidence([[1, 3, 10]], original=[[10, 2, 3]]))
        np.testing.assert_array_equal(result['person_labels'], [[255, 255, 255]])
        self.assertTrue(result['conflict'].all())
        self.assertTrue(result['uncertain'].all())
        self.assertTrue(result['protected'].all())
        self.assertFalse(result['allowed'].any())

    def test_dual_view_and_class_aligned_see_evidence_build_consensus(self):
        ev = evidence([[1, 2, 3, 10]], original=[[1, 2, 3, 10]])
        ev['see_logits'][0, 0, 2] = 3
        ev['see_logits'][2, 0, 1] = 3
        result = self.legacy(ev)
        self.assertTrue(result['consensus'].all())

    def test_handwear_and_legwear_are_not_body_skin_or_hard_outfit_evidence(self):
        ev = evidence([[1, 1]])
        ev['see_logits'][12, 0, 0] = 9
        ev['see_logits'][14, 0, 1] = 9
        result = self.legacy(ev)
        np.testing.assert_array_equal(result['person_labels'], [[1, 1]])
        self.assertFalse(result['consensus'].any())
        self.assertFalse(result['conflict'].any())

    def test_ambiguous_limbwear_cannot_confirm_an_erroneous_outfit_core(self):
        ev = evidence([[10, 10]])
        ev['see_logits'][12, 0, 0] = 9
        ev['see_logits'][14, 0, 1] = 9
        result = self.legacy(ev)
        self.assertFalse(result['consensus'].any())
        self.assertFalse(result['_trusted_cores'][6].any())

    def test_explicit_wear_support_is_separate_from_m2f_outfit_or_ambiguous_limbs(self):
        ev = evidence([[10, 10, 10, 10]], original=[[10, 10, 10, 10]])
        ev['see_logits'][11, 0, 0] = 4
        ev['see_logits'][4, 0, 1] = 4
        ev['see_logits'][12, 0, 2] = 4
        ev['see_logits'][14, 0, 3] = 4
        result = self.legacy(ev)
        np.testing.assert_array_equal(result['_wear_support'], [[True, True, False, False]])

    def test_matching_see_vote_does_not_hide_another_incompatible_positive_vote(self):
        ev = evidence([[3]])
        ev['see_logits'][0] = 4
        ev['see_logits'][11] = 4
        result = self.legacy(ev)
        self.assertEqual(result['person_labels'][0, 0], 255)
        self.assertTrue(result['conflict'][0, 0])

    def test_strong_isolated_see_eyes_without_primary_anchor_stay_unknown(self):
        ev = evidence(np.zeros((5, 5), np.uint8))
        ev['m2f_white'][:] = np.log(np.float32(1 / 12))
        ev['see_logits'][3, 2, 2] = 9
        result = self.legacy(ev)
        self.assertFalse(result['_class_support'][4].any())

    def test_low_confidence_is_unknown_rather_than_false_background(self):
        ev = evidence([[1]])
        ev['m2f_white'][:] = np.log(np.float32(1 / 12))
        result = self.legacy(ev)
        self.assertEqual(result['person_labels'][0, 0], 255)
        self.assertTrue(result['protected'][0, 0])

    def test_interpolate_all_twelve_scores_before_final_class_decision(self):
        # Left native pixel weakly favors skin, right strongly favors hair.
        # Nearest argmax would keep the first two output pixels skin; score
        # interpolation makes the second pixel hair (at low confidence here).
        probabilities = np.full((12, 1, 2), .0001, np.float32)
        probabilities[1, 0] = [.51, .001]
        probabilities[3, 0] = [.489, .998]
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        ev = evidence([[1, 1, 3, 3]], native=np.log(probabilities))
        result = self.legacy(ev, {'m2f': .5, 'margin': 0})
        self.assertEqual(result['_m2f_labels'][0, 1], 3)

    def test_object_conflict_preserves_trusted_skin_but_can_veto_outfit(self):
        ev = evidence([[1, 3, 10, 9, 4, 2]], original=[[1, 3, 10, 9, 4, 2]])
        ev['see_logits'][18] = 4
        result = self.legacy(ev)
        # Semantic disagreement is sent to SAM; no unreviewed false exclusion.
        self.assertFalse(result['objects'][0, :2].any())
        self.assertTrue(result['conflict'].all())
        self.assertTrue(result['_object_support'].all())
        self.assertFalse(result['allowed'].any())

    def test_eye_contour_priority_retains_whole_anchored_component_without_head_crop(self):
        labels = np.full((8, 16), 2, np.uint8)
        labels[3:5, 5:7] = 4
        labels[0, :] = 3
        labels[1, 5:7] = 6
        labels[6, 5] = 8
        labels[7, 5] = 9
        ev = evidence(labels)
        ev['see_logits'][3, 2:5, 3:9] = 3
        result = self.legacy(ev)
        self.assertTrue((result['person_labels'][2:5, 3:9] == 4).all())
        self.assertTrue((result['person_labels'][0] == 3).all())
        self.assertTrue((result['person_labels'][1, 5:7] == 3).all())
        self.assertEqual(result['person_labels'][6, 5], 2)
        self.assertEqual(result['person_labels'][7, 5], 5)

    def test_nonfinite_or_non_float32_scores_rejected(self):
        ev = evidence([[1]])
        ev['m2f_white'] = ev['m2f_white'].astype(np.float16)
        with self.assertRaises(ValueError):
            self.legacy(ev)
        ev = evidence([[1]])
        ev['see_logits'][0] = np.nan
        with self.assertRaises(ValueError):
            self.legacy(ev)


class UnknownResolutionTests(unittest.TestCase):
    def setUp(self):
        from gpt_animeimage_denoise import fusion

        self.fusion = fusion

    def fixture(self, white, original=None, colors=None):
        def view(row):
            values = np.full((12, 3, len(row)), .0001, np.float32)
            for x, (fine, score) in enumerate(row):
                values[fine, :, x] = score
                values[2 if fine != 2 else 0, :, x] = 1 - score
            values /= values.sum(axis=0, keepdims=True)
            return np.log(values)

        ev = evidence(np.zeros((3, len(white)), np.uint8))
        ev['m2f_white'] = view(white)
        ev['m2f_original'] = view(original or white)
        ev['see_original_logits'] = ev['see_logits'].copy()
        rgb = np.full((3, len(white), 3), 45, np.uint8)
        if colors is not None:
            rgb[:] = np.asarray(colors, np.uint8)[None, :, None]
        return rgb, ev

    def resolve(self, rgb, ev):
        initial = self.fusion.fuse_evidence(ev)
        return initial, self.fusion.resolve_unknown_regions(rgb, ev, initial)

    def test_same_class_gap_resolves_and_preserves_old_labels(self):
        rgb, ev = self.fixture([(3, .95), (3, .62), (3, .95), (4, .95)])
        initial, result = self.resolve(rgb, ev)
        np.testing.assert_array_equal(initial['person_labels'][1], [3, 255, 3, 4])
        np.testing.assert_array_equal(result['person_labels'][1], [3, 3, 3, 4])
        self.assertEqual(result['source_map'][1, 1], result['policy']['source_codes']['unknown_agreement'])
        self.assertTrue(result['allowed'][1, 1])
        self.assertFalse(result['protected'][1, 1])
        self.assertFalse(result['uncertain'][1, 1])

    def test_boundary_can_use_one_mid_vote_but_stops_at_a_sharp_edge(self):
        white = [(3, .95), (3, .63), (3, .63), (10, .95)]
        original = [(3, .95), (3, .56), (3, .56), (10, .95)]
        # The last uncertain column is visually on the outfit side.
        rgb, ev = self.fixture(white, original, [38, 40, 185, 186])
        ev['m2f_original'][3, :, 1:3] = np.log(.54)
        ev['m2f_original'][2, :, 1:3] = np.log(.40)
        initial, result = self.resolve(rgb, ev)
        np.testing.assert_array_equal(initial['person_labels'][1], [3, 255, 255, 6])
        np.testing.assert_array_equal(result['person_labels'][1], [3, 3, 255, 6])
        self.assertEqual(result['source_map'][1, 1], result['policy']['source_codes']['unknown_boundary'])
        self.assertTrue(result['protected'][1, 2])

    def test_nearby_competitor_prevents_cross_class_growth_even_for_mid_agreement(self):
        rgb, ev = self.fixture([(3, .95), (3, .62), (10, .95)])
        initial, result = self.resolve(rgb, ev)
        np.testing.assert_array_equal(result['person_labels'], initial['person_labels'])
        self.assertTrue(result['uncertain'][1, 1])

    def test_agreement_fills_a_connected_gap_with_mild_surface_variation(self):
        white = [(3, .95), (3, .62), (3, .62), (3, .95)]
        rgb, ev = self.fixture(white, colors=[45, 64, 65, 45])
        initial, result = self.resolve(rgb, ev)
        np.testing.assert_array_equal(initial['person_labels'][1], [3, 255, 255, 3])
        np.testing.assert_array_equal(result['person_labels'][1], [3, 3, 3, 3])
        self.assertEqual(result['unknown_resolution']['agreement_pixels'], 6)

    def test_sky_and_object_vetoes_keep_uncertain_pixels_protected(self):
        rgb, ev = self.fixture([(0, .95), (0, .63), (0, .63), (3, .63), (3, .95), (3, .63)])
        ev['alpha'][:, :2] = .02
        ev['alpha'][:, 3] = .5
        ev['see_original_logits'][18, :, 5] = 3
        initial, result = self.resolve(rgb, ev)
        np.testing.assert_array_equal(initial['person_labels'][1], [0, 255, 255, 255, 3, 0])
        np.testing.assert_array_equal(result['person_labels'][1], [0, 0, 255, 255, 3, 0])
        np.testing.assert_array_equal(result['objects'], initial['objects'])
        self.assertTrue(result['objects'][1, 5])
        self.assertFalse(result['allowed'][1, 1])
        self.assertTrue(result['protected'][1, 1])

    def test_suspected_objects_and_competing_wear_are_not_filled(self):
        rgb, ev = self.fixture([(3, .95), (3, .62), (3, .95), (3, .62), (3, .95)])
        ev['see_original_logits'][11, :, 3] = 3
        initial = self.fusion.fuse_evidence(ev)
        initial['suspected_objects'][:, 1] = True
        result = self.fusion.resolve_unknown_regions(rgb, ev, initial)
        np.testing.assert_array_equal(result['person_labels'][1], [3, 255, 3, 255, 3])
        np.testing.assert_array_equal(result['suspected_objects'], initial['suspected_objects'])

    def test_enclosed_background_hole_does_not_seed_further_background(self):
        rgb, ev = self.fixture([(3, .95), (0, .95), (0, .62), (3, .95)])
        _, all_hair = self.fixture([(3, .95)] * 4)
        for view in ('m2f_white', 'm2f_original'):
            ev[view][:, [0, 2], :] = all_hair[view][:, [0, 2], :]
        ev['alpha'][1, 1:3] = .02
        rgb[:] = 185
        rgb[1, 1:3] = 45
        initial, result = self.resolve(rgb, ev)
        np.testing.assert_array_equal(initial['person_labels'][1], [3, 0, 255, 3])
        np.testing.assert_array_equal(result['person_labels'][1], [3, 0, 255, 3])

    def test_disagreement_single_view_and_historical_policy_stay_unresolved(self):
        white = [(3, .95), (3, .63), (3, .63), (3, .95)]
        rgb, ev = self.fixture(white, [(3, .95), (10, .95), (3, .63), (3, .95)])
        initial, result = self.resolve(rgb, ev)
        self.assertEqual(result['person_labels'][1, 1], 255)
        historical = {key: value for key, value in initial['policy'].items() if key != 'unknown_resolution'}
        legacy = self.fusion.resolve_unknown_regions(rgb, ev, initial, historical)
        np.testing.assert_array_equal(legacy['person_labels'], initial['person_labels'])
        ev['m2f_original'] = None
        initial, result = self.resolve(rgb, ev)
        np.testing.assert_array_equal(result['person_labels'], initial['person_labels'])


if __name__ == '__main__':
    unittest.main()
