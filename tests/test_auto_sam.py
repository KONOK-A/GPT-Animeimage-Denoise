"""Automatic SAM decisions use semantic evidence, never shape scores as labels."""
import importlib
import unittest
from unittest.mock import patch

import numpy as np

from test_fusion import evidence


class CandidateProvider:
    """Replace the GPU predictor at its complete documented array boundary."""
    def __init__(self, masks, *, score=.95, stability=.95, generated=None):
        self.masks = np.asarray(masks, bool)
        self.score = score
        self.stability = stability
        self.prompts = []
        self.generated = generated or []
        self.generations = 0

    def predict_candidates(self, rgb, prompt, *, mask_input=None):
        if 'box' in prompt:
            x0, y0, x1, y1 = prompt['box']
            if x1 <= x0 or y1 <= y0:
                raise ValueError('SAM adapter requires a positive-area XYXY box')
        self.prompts.append((prompt, mask_input))
        masks = self.masks if self.masks.ndim == 3 else np.stack([self.masks] * 3)
        return {'masks': masks.copy(), 'scores': np.full(3, self.score, np.float32),
                'lowres': np.zeros((3, 256, 256), np.float32),
                'stability': np.full(3, self.stability, np.float32)}

    def generate_candidates(self, rgb, box):
        self.generations += 1
        return self.generated


class AutoSamTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('gpt_animeimage_denoise.auto_sam')
        except ModuleNotFoundError:
            self.module = None
        self.assertIsNotNone(self.module, 'Bounded automatic SAM refinement must exist')
        self.fusion = importlib.import_module('gpt_animeimage_denoise.fusion')

    def run_refine(self, ev, provider, policy=None):
        regions = self.fusion.fuse_evidence(ev, {'semantic_repair': 'off'})
        before = {key: value.copy() for key, value in regions.items() if isinstance(value, np.ndarray)}
        result = self.module.refine_regions(np.zeros((*ev['alpha'].shape, 3), np.uint8), ev, regions, provider, policy)
        for key, value in before.items():
            np.testing.assert_array_equal(regions[key], value)
        return result

    def test_manual_selection_keeps_all_prompted_components_and_discards_unprompted(self):
        shape = (8, 12)
        masks = np.zeros((3, *shape), bool)
        masks[0, 1:3, 1:3] = True
        masks[0, 5:7, 9:11] = True
        masks[0, 1:3, 9:11] = True  # Unsupported disconnected spill.
        provider = CandidateProvider(masks)
        pred = provider.predict_candidates(None, {})
        selected, record = self.module.select_manual_candidate(pred, {'positive': [[1, 1], [9, 5]], 'negative': [], 'box': [0, 0, 11, 7]})
        self.assertEqual(int(selected.sum()), 8)
        self.assertTrue(selected[1, 1] and selected[5, 9])
        self.assertFalse(selected[1, 9])
        self.assertTrue(record['accepted'])

    def test_manual_high_score_cannot_override_negative_or_missing_positive(self):
        masks = np.ones((3, 5, 5), bool)
        pred = CandidateProvider(masks).predict_candidates(None, {})
        selected, record = self.module.select_manual_candidate(pred, {'positive': [[2, 2]], 'negative': [[1, 1]], 'box': [0, 0, 4, 4]}, candidate=0)
        self.assertIsNone(selected)
        self.assertFalse(record['accepted'])

    def test_object_refinement_overrides_outfit_error_and_cleans_all_six_groups(self):
        # All classes in the object region have primary-only support. A trusted
        # skin island elsewhere must be protected by negative prompts.
        ev = evidence([[1, 1, 2, 3, 4, 9, 10, 11]])
        ev['m2f_original'] = ev['m2f_white'].copy()
        ev['m2f_original'][:, :, 1:] = np.log(np.float32(1 / 12))
        ev['see_logits'][18, :, 1:] = 4
        mask = np.ones((1, 8), bool)
        mask[0, 0] = False
        provider = CandidateProvider(mask)
        result = self.run_refine(ev, provider)
        self.assertTrue(result['objects'][0, 1:].all())
        self.assertTrue((result['person_labels'][0, 1:] == 0).all())
        self.assertFalse(result['allowed'][0, 1:].any())
        self.assertTrue(result['protected'][0, 1:].all())
        self.assertEqual(result['person_labels'][0, 0], 1)
        self.assertTrue(any(record['accepted'] for record in result['sam_records']))

    def test_object_prompt_does_not_treat_wrong_outfit_inside_proposal_as_negative(self):
        ev = evidence([[10, 10, 1]], original=[[10, 10, 1]])
        ev['see_logits'][18, 0, :2] = 4
        provider = CandidateProvider([[True, True, False]])
        result = self.run_refine(ev, provider)
        self.assertTrue(result['objects'][0, :2].all())
        self.assertEqual(result['person_labels'][0, 2], 1)
        prompt = provider.prompts[0][0]
        self.assertTrue(all(x >= 2 for x, y in prompt['negative']))

    def test_small_object_seed_can_complete_a_much_larger_connected_object(self):
        ev = evidence(np.zeros((32, 32), np.uint8))
        ev['see_logits'][18, 15:17, 15:17] = 4
        complete = np.zeros((32, 32), bool)
        complete[8:24, 8:24] = True
        result = self.run_refine(ev, CandidateProvider(complete))
        np.testing.assert_array_equal(result['objects'], complete)
        record = result['sam_records'][0]
        self.assertTrue(record['accepted'])
        self.assertEqual(record['seed_pixels'], 4)
        self.assertEqual(record['candidate_pixels'], 256)
        self.assertAlmostEqual(record['semantic_support_fraction'], 4 / 256)

    def test_only_m2f_outfit_agreement_inside_object_context_cannot_block_completion(self):
        labels = np.full((40, 40), 10, np.uint8)
        ev = evidence(labels, original=labels)
        # A narrow semantic object stripe supplies fragmented-area evidence;
        # its contextual box includes the full wider SAM object.
        ev['see_logits'][18, 10:30, 19:21] = 4
        complete = np.zeros_like(labels, bool)
        complete[10:30, 18:22] = True
        result = self.run_refine(ev, CandidateProvider(complete))
        np.testing.assert_array_equal(result['objects'], complete)
        self.assertTrue((result['person_labels'][complete] == 0).all())
        self.assertTrue((result['person_labels'][:5] == 6).all())

    def test_verified_clothing_and_worn_accessory_inside_context_are_protected(self):
        for see_class in (1, 4, 6, 10, 11, 13, 15):
            with self.subTest(see_class=see_class):
                labels = np.full((40, 40), 10, np.uint8)
                ev = evidence(labels, original=labels)
                ev['see_logits'][18, 10:30, 19:21] = 4
                ev['see_logits'][see_class, 10:30, 21:23] = 4
                candidate = np.zeros_like(labels, bool)
                candidate[10:30, 17:23] = True
                result = self.run_refine(ev, CandidateProvider(candidate))
                self.assertFalse(result['objects'][10:30, 21:23].any())
                self.assertTrue((result['person_labels'][10:30, 21:23] == 6).all())
                self.assertFalse(any(record['accepted'] for record in result['sam_records']))

    def test_only_m2f_clothing_outside_context_keeps_its_core_protection(self):
        labels = np.full((40, 40), 10, np.uint8)
        ev = evidence(labels, original=labels)
        ev['see_logits'][18, 10:30, 19:21] = 4
        candidate = np.zeros_like(labels, bool)
        candidate[10:30, 10:30] = True  # Extends beyond the generic context box.
        result = self.run_refine(ev, CandidateProvider(candidate))
        self.assertFalse(result['objects'].any())
        self.assertTrue((result['person_labels'][:, :10] == 6).all())

    def test_trustworthy_skin_and_hair_cores_cannot_be_swallowed_by_object_candidate(self):
        ev = evidence([[1, 3, 10, 10]], original=[[1, 3, 10, 10]])
        ev['see_logits'][18, 0, 2:] = 4
        provider = CandidateProvider(np.ones((1, 4), bool))
        result = self.run_refine(ev, provider)
        self.assertEqual(result['person_labels'][0, 0], 1)
        self.assertEqual(result['person_labels'][0, 1], 3)
        self.assertFalse(result['objects'][0, :2].any())
        self.assertFalse(any(record['accepted'] for record in result['sam_records']))

    def test_every_disconnected_supported_object_component_is_seeded_and_preserved(self):
        ev = evidence(np.full((8, 20), 10, np.uint8))
        ev['see_logits'][18, 1:3, 1:3] = 4
        ev['see_logits'][18, 5:7, 17:19] = 4
        mask = ev['see_logits'][18] > 1
        provider = CandidateProvider(mask)
        result = self.run_refine(ev, provider)
        np.testing.assert_array_equal(result['objects'], mask)
        self.assertEqual(len(provider.prompts[0][0]['positive']), 2)
        self.assertFalse(result['objects'][3, 10])

    def test_single_pixel_object_has_non_degenerate_original_grid_prompt_box(self):
        ev = evidence([[10]])
        ev['see_logits'][18] = 4
        result = self.run_refine(ev, CandidateProvider([[True]]))
        self.assertTrue(result['objects'][0, 0])
        self.assertTrue(result['sam_records'][0]['accepted'])

    def test_large_supported_component_cannot_hide_loss_of_a_small_component(self):
        ev = evidence(np.full((16, 32), 10, np.uint8))
        ev['see_logits'][18, 1:14, 1:14] = 4
        ev['see_logits'][18, 4:7, 25:28] = 4
        mask = ev['see_logits'][18] > 1
        # Keep the small component's prompt but destroy most of its core. A
        # global coverage percentage still exceeds90%; each component must pass.
        mask[4:7, 25:28] = False
        mask[5, 26] = True
        provider = CandidateProvider(mask)
        result = self.run_refine(ev, provider)
        self.assertFalse(result['objects'].any())
        self.assertFalse(any(record['accepted'] for record in result['sam_records']))

    def test_fragmented_objects_bound_points_and_report_components_beyond_task_budget(self):
        ev = evidence(np.full((103, 103), 10, np.uint8))
        ev['see_logits'][18, 1::3, 1::3] = 4
        mask = ev['see_logits'][18] > 1  # 34x34 =1156 separate components.
        provider = CandidateProvider(mask)
        result = self.run_refine(ev, provider)
        self.assertEqual(len(provider.prompts), 16)
        self.assertTrue(all(len(prompt['positive']) <= 64 and len(prompt['negative']) <= 64
                            for prompt, _ in provider.prompts))
        self.assertEqual(int(result['objects'].sum()), 1024)
        self.assertEqual(int(result['_sam_pending'].sum()), 132)
        self.assertTrue((result['person_labels'][result['_sam_pending']] == 255).all())
        self.assertTrue(result['protected'][result['_sam_pending']].all())
        pending = [record for record in result['sam_records'] if record['action'] == 'budget']
        self.assertEqual(pending[0]['pending_components'], 132)

    def test_negative_prompt_budget_keeps_full_trusted_core_protection(self):
        labels = np.full((40, 40), 10, np.uint8)
        labels[1::3, 1::3] = 1
        ev = evidence(labels, original=labels)
        ev['see_logits'][18, labels != 1] = 4
        mask = labels != 1
        # The second skin component is outside the sampled64 negatives. Even
        # a tiny (<2%) unprompted core leak must be vetoed by the full mask.
        mask[1, 4] = True
        provider = CandidateProvider(mask)
        result = self.run_refine(ev, provider)
        self.assertLessEqual(len(provider.prompts[0][0]['negative']), 64)
        self.assertTrue((result['person_labels'][labels == 1] == 1).all())
        self.assertFalse(result['objects'][1, 4])
        self.assertTrue(result['objects'][39, 39])

    def test_automatic_box_adds_context_and_clips_at_original_grid(self):
        labels = np.full((80, 120), 10, np.uint8)
        ev = evidence(labels)
        ev['see_logits'][18, 30:50, 30:50] = 4
        provider = CandidateProvider(ev['see_logits'][18] > 1)
        self.run_refine(ev, provider)
        # 20-pixel side *12% =2.4, rounded up to3 pixels per edge.
        self.assertEqual(provider.prompts[0][0]['box'], [27, 27, 53, 53])
        ev['see_logits'][18] = -4
        ev['see_logits'][18, 0, 0] = 4
        provider = CandidateProvider(ev['see_logits'][18] > 1)
        self.run_refine(ev, provider)
        self.assertEqual(provider.prompts[0][0]['box'], [0, 0, 2, 2])

    def test_high_stability_roi_proposal_without_semantic_evidence_cannot_label_unknown(self):
        ev = evidence([[1, 1], [1, 1]])
        ev['m2f_white'][:] = np.log(np.float32(1 / 12))
        provider = CandidateProvider(np.ones((2, 2), bool), generated=[{'mask': np.ones((2, 2), bool), 'score': .99, 'stability': .99, 'bbox': [0, 0, 2, 2]}])
        result = self.run_refine(ev, provider, {'roi_proposals': True})
        self.assertTrue((result['person_labels'] == 255).all())
        self.assertTrue(result['protected'].all())
        self.assertFalse(result['allowed'].any())
        self.assertEqual(provider.generations, 0)

    def test_shape_refinement_cannot_extend_person_class_into_unsupported_unknown(self):
        ev = evidence([[1, 1, 1, 1]])
        ev['m2f_white'][:] = np.log(np.float32(1 / 12))
        ev['see_logits'][0, 0, :2] = 4
        provider = CandidateProvider(np.ones((1, 4), bool))
        result = self.run_refine(ev, provider)
        self.assertTrue((result['person_labels'][0, :2] == 3).all())
        self.assertTrue((result['person_labels'][0, 2:] == 255).all())
        self.assertTrue(result['protected'][0, 2:].all())

    def test_failed_tasks_stop_after_two_rounds_and_policy_cannot_exceed_hard_limits(self):
        ev = evidence(np.full((4, 4), 10, np.uint8))
        ev['see_logits'][18] = 4
        provider = CandidateProvider(np.ones((4, 4), bool), score=.1)
        result = self.run_refine(ev, provider)
        self.assertLessEqual(len(provider.prompts), 32)
        self.assertEqual(len(provider.prompts), 2)
        self.assertEqual(result['sam_records'][0]['rounds'], 2)
        with self.assertRaises(ValueError):
            self.run_refine(ev, provider, {'max_tasks': 17})
        with self.assertRaises(ValueError):
            self.run_refine(ev, provider, {'max_rounds': 3})

    def test_roi_generation_supplements_object_evidence_when_prompt_candidates_fail(self):
        ev = evidence([[10, 10, 0]])
        ev['see_logits'][18, 0, :2] = 4
        mask = np.array([[True, True, False]])
        provider = CandidateProvider(np.zeros((1, 3), bool), generated=[{'mask': mask, 'score': .95, 'stability': .95, 'bbox': [0, 0, 2, 1]}])
        result = self.run_refine(ev, provider, {'roi_proposals': True})
        np.testing.assert_array_equal(result['objects'], mask)
        self.assertTrue(result['sam_records'][0]['accepted'])
        self.assertEqual(provider.generations, 1)

    def test_conflicting_person_semantics_stay_unknown_even_with_perfect_sam(self):
        ev = evidence([[1, 1]], original=[[3, 3]])
        provider = CandidateProvider(np.ones((1, 2), bool))
        result = self.run_refine(ev, provider)
        self.assertTrue((result['person_labels'] == 255).all())
        self.assertTrue(result['protected'].all())

    def test_unknown_resolution_runs_after_sam_and_outfit_reference(self):
        ev = evidence([[3, 3, 0, 0, 10]], original=[[3, 3, 0, 0, 10]])
        weak = np.full(12, .46 / 11, np.float32)
        weak[3] = .54
        ev['m2f_white'][:, 0, 1] = np.log(weak)
        ev['m2f_original'][:, 0, 1] = np.log(weak)
        ev['see_logits'][18, 0, 2] = 4
        regions = self.fusion.fuse_evidence(ev)
        regions['_outfit_reference_mask'][0, 4] = False
        rgb = np.zeros((1, 5, 3), np.uint8)
        provider = CandidateProvider([[False, False, True, True, False]])

        def resolve(received_rgb, received_evidence, result, policy):
            self.assertIs(received_rgb, rgb)
            self.assertIs(received_evidence, ev)
            self.assertEqual(policy['unknown_resolution'], 'edge-guided')
            self.assertEqual(result['sam_records'][0]['applied_pixels'], 2)
            np.testing.assert_array_equal(result['objects'], [[False, False, True, True, False]])
            np.testing.assert_array_equal(result['person_labels'], [[3, 255, 0, 0, 255]])
            result['person_labels'][0, 1] = 3
            result['uncertain'][0, 1] = False
            result['source_map'][0, 1] = 99
            return self.module._refresh_regions(result)

        with patch.object(self.fusion, 'resolve_unknown_regions', side_effect=resolve, create=True) as resolution, \
                patch.object(self.module, '_refresh_regions', wraps=self.module._refresh_regions) as refresh:
            result = self.module.refine_regions(rgb, ev, regions, provider, {'unknown_resolution': 'edge-guided'})
        resolution.assert_called_once()
        refresh.assert_called_once()
        np.testing.assert_array_equal(result['allowed'], [[True, True, False, False, False]])
        np.testing.assert_array_equal(result['protected'], [[False, False, True, True, True]])
        self.assertEqual(result['source_map'][0, 1], 99)
        self.assertEqual(regions['person_labels'][0, 1], 255)

    def test_unknown_resolution_skips_saved_legacy_and_reference_policies(self):
        ev = evidence([[3]], original=[[3]])
        rgb = np.zeros((1, 1, 3), np.uint8)
        for regions in (self.fusion.fuse_evidence(ev, {'part_cooperation': 'guarded'}),
                        self.fusion.reference_regions(ev)):
            with patch.object(self.fusion, 'resolve_unknown_regions', create=True) as resolution:
                self.module.refine_regions(rgb, ev, regions, CandidateProvider([[False]]))
            resolution.assert_not_called()


if __name__ == '__main__':
    unittest.main()
