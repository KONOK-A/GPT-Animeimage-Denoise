import unittest
from unittest.mock import patch

import numpy as np

from gpt_animeimage_denoise.corrections import apply_corrections
from gpt_animeimage_denoise.editor import DEFAULT_MODULES, MODULES, SelectionEditor


class WorkbenchMasksTests(unittest.TestCase):
    def setUp(self):
        shape = (64, 96)
        labels = np.zeros(shape, np.uint8)
        labels[:, 12:36] = MODULES['skin']
        labels[:, 36:72] = MODULES['hair']
        labels[20, 20] = 255
        objects = np.zeros(shape, bool)
        objects[20, 21] = True
        alpha = np.full(shape, 255, np.uint8)
        alpha[20, 22] = 0
        rgb = np.random.default_rng(21).integers(0, 256, (*shape, 3), dtype=np.uint8)
        self.editor = SelectionEditor(rgb, alpha, labels, objects, np.zeros_like(objects))

    def selection(self, edits=(), *, cursor=None, **settings):
        from gpt_animeimage_denoise.workbench.masks import render_selection

        return render_selection(self.editor, settings, list(edits), len(edits) if cursor is None else cursor)

    def test_untouched_defaults_match_v21_exactly(self):
        result = self.selection()
        np.testing.assert_array_equal(result.weight, self.editor.weights(DEFAULT_MODULES))
        np.testing.assert_array_equal(result.labels, self.editor.labels)
        np.testing.assert_array_equal(result.selected, self.editor.selection(DEFAULT_MODULES))

    def test_workbench_weight_cache_reuses_lines_and_geometry_without_changing_weights(self):
        from gpt_animeimage_denoise.workbench import masks

        cache = masks.WorkbenchWeightCache(self.editor)
        brush = {'tool': 'protect', 'points': [[52, 30]], 'radius': 6, 'strength': .75}
        exclude = dict(brush, tool='exclude')
        history = [[], [brush], [brush, exclude], [brush, exclude, dict(brush, strength=.9)]]
        expected = [self.selection(edits).weight for edits in history]
        with patch.object(masks, '_line_weight', wraps=masks._line_weight) as lines, \
             patch.object(masks, '_compose_weight', wraps=masks._compose_weight) as geometry:
            for edits, exact in zip(history, expected, strict=True):
                actual = masks.render_selection(self.editor, {}, edits, len(edits), weight_cache=cache)
                np.testing.assert_array_equal(actual.weight, exact)
            self.assertEqual(lines.call_count, 1)
            self.assertEqual(geometry.call_count, 2)
            self.assertLess(geometry.call_args.args[0].size, self.editor.labels.size)

    def test_weight_cache_matches_full_feather_across_image_edges_and_large_changes(self):
        from gpt_animeimage_denoise.filters import build_weight
        from gpt_animeimage_denoise.workbench.masks import WorkbenchWeightCache

        shape = (310, 480)
        rgb = np.random.default_rng(2).integers(0, 256, (*shape, 3), dtype=np.uint8)
        empty = np.zeros(shape, bool)
        editor = SelectionEditor(rgb, None, np.ones(shape, np.uint8), empty, empty)
        cache = WorkbenchWeightCache(editor)
        selection = np.ones(shape, bool)
        for tile, mark in [((slice(0, 4), slice(0, 5)), False), ((slice(29, 33), slice(0, 3)), False),
                           ((slice(28, 32), slice(2, 6)), True), ((slice(307, 310), slice(477, 480)), False),
                           ((slice(50, 280), slice(20, 460)), False), ((slice(None), slice(None)), True)]:
            selection[tile] = mark
            expected = build_weight(selection, empty, rgb, 16 * cache.scale, 5 * cache.scale,
                                    structure_protection=.2)
            np.testing.assert_array_equal(cache.weight(selection, .2), expected)

    def test_ordinary_edits_cannot_unlock_predictions_but_override_can(self):
        include = {'tool': 'include', 'shape': 'brush', 'points': [[21, 20]], 'radius': 3}
        invert = {'tool': 'invert'}
        ordinary = self.selection([include, invert])
        self.assertFalse(ordinary.selected[20, 20:23].any())
        correct = dict(include, tool='override', module='hair')
        result = self.selection([include, correct])
        self.assertTrue(result.selected[20, 20:22].all())
        self.assertFalse(result.objects[20, 21])
        self.assertFalse(result.unknown[20, 20])
        self.assertEqual(result.labels[20, 20], MODULES['hair'])
        self.assertFalse(result.selected[20, 22])
        self.assertTrue(self.editor.objects[20, 21])
        protection = dict(include, tool='protect', strength=1)
        protected = self.selection([protection, correct])
        self.assertFalse(protected.objects[20, 21])
        self.assertEqual(protected.soft_protection[20, 21], 1)
        self.assertEqual(protected.weight[20, 21], 0)

    def test_soft_protection_eraser_and_cursor_apply_exactly(self):
        brush = {'tool': 'protect', 'shape': 'brush', 'points': [[52, 30]], 'radius': 12, 'strength': .9}
        erase = {'tool': 'erase', 'shape': 'brush', 'points': [[52, 30]], 'radius': 2}
        baseline = self.selection(structure_protection=0)
        painted = self.selection([brush, erase], cursor=1, structure_protection=0)
        self.assertAlmostEqual(float(painted.soft_protection[30, 52]), .9)
        self.assertGreater(painted.soft_protection[30, 59], painted.soft_protection[30, 62])
        self.assertAlmostEqual(float(painted.weight[30, 52]), float(baseline.weight[30, 52]) * .1)
        self.assertIsNone(painted.edit_feedback)
        erased = self.selection([brush, erase], structure_protection=0)
        self.assertEqual(erased.soft_protection[30, 52], 0)
        self.assertEqual(erased.weight[30, 52], baseline.weight[30, 52])
        self.assertGreater(erased.soft_protection[30, 59], 0)
        self.assertEqual(erased.edit_feedback, {'tool': 'erase', 'changed_pixels': 13, 'reason': 'changed'})

    def test_include_feedback_distinguishes_new_pixels_automatic_selection_and_protection(self):
        include = {'tool': 'include', 'shape': 'brush', 'points': [[52, 30]], 'radius': 2}
        result = self.selection([include])
        self.assertEqual(result.edit_feedback, {'tool': 'include', 'changed_pixels': 0,
                                                'reason': 'already_selected'})
        outside = dict(include, points=[[82, 30]])
        result = self.selection([outside])
        self.assertEqual(result.edit_feedback, {'tool': 'include', 'changed_pixels': 13,
                                                'reason': 'changed'})
        protect = dict(outside, tool='protect', radius=5, strength=1)
        result = self.selection([protect, outside])
        self.assertEqual(result.edit_feedback, {'tool': 'include', 'changed_pixels': 0,
                                                'reason': 'protected'})

    def test_quick_unknown_resolution_uses_only_last_brush_skipped_pixels(self):
        from gpt_animeimage_denoise.workbench.masks import binary_corrections

        # Every veto is touched by the brush. Only x=20 is eligible.
        unknown = self.editor.unknown.copy()
        unknown[20, [21, 22, 24]] = True
        self.editor = SelectionEditor(self.editor.rgb, self.editor.alpha, self.editor.labels,
                                      self.editor.objects, unknown)
        protect = {'tool': 'protect', 'points': [[24, 20]], 'radius': 1, 'strength': 1}
        brush = {'tool': 'include', 'points': [[22, 20]], 'radius': 4}
        before = self.selection([protect, brush])
        self.assertEqual(before.edit_feedback['skipped_unknown_pixels'], 1)
        self.assertTrue(before.unknown[20, 20:23].all())

        unlocked = self.selection([protect, brush, {'tool': 'resolve_unknown'}])
        self.assertTrue(unlocked.selected[20, 20])
        self.assertEqual(unlocked.labels[20, 20], 255)
        self.assertTrue(unlocked.objects[20, 21])
        self.assertTrue(unlocked.unknown[20, 21:23].all())
        self.assertTrue(unlocked.unknown[20, 24])
        self.assertEqual(unlocked.weight[20, 21:23].sum(), 0)
        with self.assertRaisesRegex(ValueError, 'explicit class'):
            binary_corrections(self.editor, unlocked, {})

        assign = {'tool': 'resolve_unknown', 'module': 'skin'}
        disabled = self.selection([protect, brush, assign], modules=[])
        self.assertFalse(disabled.unknown[20, 20])
        self.assertEqual(disabled.labels[20, 20], MODULES['skin'])
        self.assertFalse(disabled.selected[20, 20])
        self.assertEqual(disabled.soft_protection[20, 24], 1)
        enabled = self.selection([protect, brush, assign])
        self.assertTrue(enabled.selected[20, 20])
        self.assertTrue(binary_corrections(self.editor, enabled, {})['module_overrides']['skin'][20, 20])
        undone = self.selection([protect, brush, assign], cursor=2)
        np.testing.assert_array_equal(undone.unknown, before.unknown)
        np.testing.assert_array_equal(undone.weight, before.weight)

    def test_pending_feedback_only_counts_unknown_not_already_unlocked_by_history(self):
        brush = {'tool': 'include', 'points': [[20, 20]], 'radius': 1}
        self.assertEqual(self.selection([brush]).edit_feedback['skipped_unknown_pixels'], 1)
        self.assertNotIn('skipped_unknown_pixels',
                         self.selection([brush, {'tool': 'resolve_unknown'}, brush]).edit_feedback)
        self.assertNotIn('skipped_unknown_pixels',
                         self.selection([dict(brush, allow_protected=True)]).edit_feedback)
        polygon = {'tool': 'include', 'shape': 'polygon', 'points': [[19, 19], [22, 19], [21, 22]]}
        self.assertNotIn('skipped_unknown_pixels', self.selection([polygon]).edit_feedback)

    def test_erase_feedback_distinguishes_automatic_selection_from_manual_correction(self):
        erase = {'tool': 'erase', 'shape': 'brush', 'points': [[21, 20]], 'radius': 2}
        result = self.selection([erase])
        self.assertEqual(result.edit_feedback, {'tool': 'erase', 'changed_pixels': 0, 'reason': 'no_marks'})
        self.assertTrue(result.selected[20, 23])
        override = dict(erase, tool='override', module='hair')
        result = self.selection([override, erase])
        self.assertEqual(result.edit_feedback, {'tool': 'erase', 'changed_pixels': 2, 'reason': 'changed'})
        self.assertTrue(result.objects[20, 21])
        self.assertTrue(result.unknown[20, 20])

    def test_exclude_brush_removes_automatic_selection_without_changing_soft_protection(self):
        exclude = {'tool': 'exclude', 'shape': 'brush', 'points': [[52, 30]], 'radius': 2}
        protect = dict(exclude, tool='protect', radius=5, strength=.9)
        baseline = self.selection([protect])
        removed = self.selection([protect, exclude])
        self.assertFalse(removed.selected[30, 52])
        self.assertEqual(removed.weight[30, 52], 0)
        self.assertEqual(removed.edit_feedback, {'tool': 'exclude', 'changed_pixels': 13, 'reason': 'changed'})
        np.testing.assert_array_equal(removed.soft_protection, baseline.soft_protection)
        undone = self.selection([protect, exclude], cursor=1)
        np.testing.assert_array_equal(undone.weight, baseline.weight)
        included = self.selection([protect, exclude, dict(exclude, tool='include')])
        np.testing.assert_array_equal(included.weight, baseline.weight)
        erased = self.selection([exclude, dict(exclude, tool='erase')])
        np.testing.assert_array_equal(erased.weight, self.selection().weight)

    def test_exclude_polygon_feedback_counts_only_previously_selected_pixels(self):
        exclude = {'tool': 'exclude', 'shape': 'polygon', 'points': [[18, 18], [24, 18], [24, 24], [18, 24]]}
        removed = self.selection([exclude])
        self.assertFalse(removed.selected[18:25, 18:25].any())
        self.assertEqual(removed.edit_feedback, {'tool': 'exclude', 'changed_pixels': 46, 'reason': 'changed'})
        repeated = self.selection([exclude, exclude])
        self.assertEqual(repeated.edit_feedback, {'tool': 'exclude', 'changed_pixels': 0, 'reason': 'already_excluded'})
        included = self.selection([exclude, dict(exclude, tool='include')])
        np.testing.assert_array_equal(included.selected, self.selection().selected)

    def test_mix_sets_absolute_center_ratio_and_replays_overwrite_erase(self):
        from gpt_animeimage_denoise.workbench.masks import binary_corrections

        settings = {'strength': .2, 'part_strengths': {'hair': .5}, 'structure_protection': 0}
        baseline = self.selection(**settings)
        mix = {'tool': 'mix', 'shape': 'brush', 'points': [[52, 30]], 'radius': 6, 'strength': .6}
        painted = self.selection([mix], **settings)
        self.assertEqual(painted.weight[30, 52], np.float32(.4))
        self.assertAlmostEqual(float(painted.weight[30, 56]), (7 / 27) * .1 + (20 / 27) * .4, places=7)
        overwritten = self.selection([mix, dict(mix, strength=.8)], **settings)
        self.assertEqual(overwritten.weight[30, 52], np.float32(.2))
        erased = self.selection([mix, dict(mix, tool='erase')], **settings)
        np.testing.assert_array_equal(erased.weight, baseline.weight)
        self.assertEqual(self.selection([dict(mix, strength=.999)], **settings).weight[30, 52], np.float32(.001))
        with self.assertRaisesRegex(ValueError, 'local mixing'):
            binary_corrections(self.editor, painted, {'strength': .2})

    def test_mix_cannot_bypass_exclusions_or_protection(self):
        mix = {'tool': 'mix', 'shape': 'polygon', 'points': [[16, 17], [25, 17], [25, 23], [16, 23]], 'strength': 0}
        exclude = {'tool': 'exclude', 'points': [[17, 20]], 'radius': 1}
        protect = {'tool': 'protect', 'points': [[24, 20]], 'radius': 1, 'strength': 1}
        result = self.selection([exclude, protect, mix])
        self.assertTrue((result.weight[20, [17, 20, 21, 22, 24]] == 0).all())
        self.assertEqual(result.weight[20, 19], 1)

    def test_explicit_include_unlocks_model_veto_without_inventing_a_label(self):
        from gpt_animeimage_denoise.workbench.masks import binary_corrections, render_selection

        include = {'tool': 'include', 'points': [[21, 20]], 'radius': 2, 'allow_protected': True}
        result = self.selection([include])
        self.assertTrue(result.selected[20, 20:22].all())
        self.assertFalse(result.selected[20, 22])
        self.assertEqual(result.labels[20, 20], 255)
        self.assertEqual(result.labels[20, 21], self.editor.labels[20, 21])
        self.assertEqual(result.edit_feedback['changed_pixels'], 2)
        with self.assertRaisesRegex(ValueError, 'explicit class'):
            binary_corrections(self.editor, result, {})
        override = {'tool': 'override', 'points': [[20, 20]], 'radius': 1, 'module': 'hair'}
        assigned = self.selection([include, override])
        masks = binary_corrections(self.editor, assigned, {})
        self.assertTrue(masks['module_overrides']['hair'][20, 20])
        np.testing.assert_array_equal(self.selection([include, {'tool': 'reset'}]).weight,
                                      self.selection().weight)
        np.testing.assert_array_equal(self.selection([include, dict(include, tool='erase')]).weight,
                                      self.selection().weight)

        empty = np.zeros_like(self.editor.objects)
        correction = {name: empty.copy() for name in ('allow_mask', 'protect_mask',
                                                       'object_add_mask', 'object_remove_mask')}
        correction['protect_mask'][20, 20] = True
        correction['object_add_mask'][20, 21] = True
        correction['module_overrides'] = {}
        protected = render_selection(self.editor, {}, [include], 1, correction=correction)
        self.assertFalse(protected.selected[20, 20:23].any())
        self.assertTrue(protected.unknown[20, 20])
        self.assertTrue(protected.objects[20, 21])

    def test_object_only_unlock_exports_binary_allow_and_remove_together(self):
        from gpt_animeimage_denoise.workbench.masks import binary_corrections

        include = {'tool': 'include', 'shape': 'polygon', 'allow_protected': True,
                   'points': [[21, 19], [22, 19], [22, 20], [21, 20]]}
        result = self.selection([include])
        edits = binary_corrections(self.editor, result, {})
        baseline = self.editor.selection(DEFAULT_MODULES)
        regions = {'person_labels': self.editor.labels, 'allowed': baseline,
                   'protected': ~baseline, 'objects': self.editor.objects}
        rebuilt = apply_corrections(regions, edits)
        np.testing.assert_array_equal(rebuilt['allowed'], result.selected)

    def test_binary_corrections_round_trip_and_refuse_fractional_adjustments(self):
        from gpt_animeimage_denoise.workbench.masks import binary_corrections

        geometry = {'shape': 'polygon', 'points': [[18, 18], [23, 18], [23, 24], [18, 24]]}
        correction = dict(geometry, tool='override', module='hair')
        protect = dict(geometry, tool='protect', strength=1)
        result = self.selection([correction, protect], modules=['background', 'hair'])
        masks = binary_corrections(self.editor, result, {'modules': ['background', 'hair']})
        self.assertTrue(masks['object_remove_mask'][20, 21])
        self.assertTrue(masks['module_overrides']['hair'][20, 20])
        defaults = self.editor.selection(DEFAULT_MODULES)
        regions = {'person_labels': self.editor.labels, 'allowed': defaults,
                   'protected': ~defaults, 'objects': self.editor.objects}
        edits = {**masks, 'object_add_mask': np.zeros_like(defaults), 'sam_regions': []}
        rebuilt = apply_corrections(regions, edits)
        np.testing.assert_array_equal(rebuilt['allowed'], result.selected)
        fractional = self.selection([dict(protect, strength=.5)])
        with self.assertRaisesRegex(ValueError, 'fractional'):
            binary_corrections(self.editor, fractional, {})
        with self.assertRaisesRegex(ValueError, 'fractional'):
            binary_corrections(self.editor, self.selection(part_strengths={'hair': .5}), {'part_strengths': {'hair': .5}})

    def test_part_strength_and_reset_geometry_do_not_clear_protection(self):
        paint = {'tool': 'include', 'shape': 'polygon', 'points': [[1, 1], [9, 1], [9, 9], [1, 9]]}
        protect = {'tool': 'protect', 'shape': 'brush', 'points': [[53, 30]], 'radius': 4, 'strength': 1}
        reset = self.selection([paint, protect, {'tool': 'invert'}, {'tool': 'reset'}],
                               part_strengths={'hair': .5}, structure_protection=0)
        self.assertFalse(reset.selected[3, 3])
        self.assertEqual(reset.weight[30, 53], 0)
        self.assertAlmostEqual(float(reset.weight[40, 53]), .5)
        self.assertAlmostEqual(float(reset.weight[40, 24]), 1)

    def test_legacy_import_retains_manual_marks_without_bypassing_vetoes(self):
        from gpt_animeimage_denoise.workbench.masks import binary_corrections, render_selection

        empty = np.zeros_like(self.editor.objects)
        correction = {name: empty.copy() for name in ('allow_mask', 'protect_mask',
                                                       'object_add_mask', 'object_remove_mask')}
        correction['allow_mask'][20, [4, 20, 22]] = True
        correction['protect_mask'][20, 23] = True
        correction['object_add_mask'][20, 24] = True
        correction['object_remove_mask'][20, 21] = True
        hair = empty.copy()
        hair[20, 21] = True
        correction['module_overrides'] = {'hair': hair}
        settings = {'modules': list(DEFAULT_MODULES)}
        result = render_selection(self.editor, settings, [], 0, correction=correction)
        self.assertTrue(result.selected[20, 4])
        self.assertFalse(result.selected[20, 20])
        self.assertTrue(result.selected[20, 21])
        self.assertFalse(result.selected[20, 22:25].any())
        masks = binary_corrections(self.editor, result, settings)
        self.assertTrue(masks['object_add_mask'][20, 24])
        self.assertTrue(masks['object_remove_mask'][20, 21])
        self.assertTrue(masks['protect_mask'][20, 23])
