from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

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

    def test_stage_erasers_keep_the_other_layers_and_classify_any_semantic_part(self):
        region = {'shape': 'polygon', 'points': [[8, 18], [50, 18], [50, 23], [8, 23]]}
        part = {**region, 'stage': 'parts', 'tool': 'classify', 'module': 'outfit'}
        exclude = {**region, 'stage': 'range', 'tool': 'exclude'}
        for name, code in {**MODULES, 'object': 0, 'unknown': 255}.items():
            classified = self.selection([{**part, 'module': name}])
            for x in (9, 19, 20, 21, 45):
                self.assertEqual(classified.labels[20, x], code, name)
                self.assertEqual(classified.objects[20, x], name == 'object')
                self.assertEqual(classified.unknown[20, x], name == 'unknown')
            self.assertEqual(classified.labels[20, 22], self.editor.labels[20, 22])
        cleared_parts = self.selection([part, exclude, {**region, 'stage': 'parts', 'tool': 'erase'}])
        np.testing.assert_array_equal(cleared_parts.labels, self.editor.labels)
        np.testing.assert_array_equal(cleared_parts.objects, self.editor.objects)
        np.testing.assert_array_equal(cleared_parts.unknown, self.editor.unknown)
        self.assertFalse(cleared_parts.selected[20, 19])
        cleared_range = self.selection([part, exclude, {**region, 'stage': 'range', 'tool': 'erase'}],
                                       modules=['outfit'])
        self.assertTrue(cleared_range.selected[20, 21])
        self.assertEqual(cleared_range.labels[20, 21], MODULES['outfit'])

    def test_fractional_range_brushes_reach_final_weights_and_replay(self):
        from gpt_animeimage_denoise.workbench.masks import binary_corrections

        brush = {'stage': 'range', 'shape': 'brush', 'points': [[50, 40]], 'radius': 8, 'feather': 4}
        for tool, modules, extras, expected in (
                ('include', [], {}, [1, .5, 0]), ('exclude', ['hair'], {}, [0, .5, 1]),
                ('protect', ['hair'], {'strength': .8}, [.2, .6, 1]),
                ('mix', ['hair'], {'strength': 1}, [0, .5, 1])):
            edit = {**brush, **extras, 'tool': tool}
            result = self.selection([edit], modules=modules, structure_protection=0)
            np.testing.assert_allclose(result.weight[40, [50, 56, 58]], expected, atol=1e-6, err_msg=tool)
            with self.assertRaises(ValueError):
                binary_corrections(self.editor, result, {'modules': modules})
            hard = self.selection([{**edit, 'feather': 0}], modules=modules, structure_protection=0)
            self.assertNotEqual(result.weight[40, 56], hard.weight[40, 56], tool)
        hard = {**brush, 'tool': 'exclude', 'feather': 0}
        erase = {**brush, 'tool': 'erase'}
        result = self.selection([hard, erase], structure_protection=0)
        np.testing.assert_allclose(result.weight[40, [50, 56, 58]], [1, .5, 0])
        inverted = self.selection([hard, erase, {'stage': 'range', 'tool': 'invert'}], structure_protection=0)
        np.testing.assert_allclose(inverted.weight[40, [50, 56, 58]], [0, .5, 1])
        undo = self.selection([hard, erase], cursor=1, structure_protection=0)
        np.testing.assert_array_equal(undo.weight, self.selection([hard], structure_protection=0).weight)

    def test_adaptive_include_gives_thin_brushes_a_full_core_and_preserves_outer_blending(self):
        from gpt_animeimage_denoise.workbench.masks import binary_corrections, render_selection

        shape = (768, 768)
        labels = np.full(shape, MODULES['outfit'], np.uint8)
        empty = np.zeros(shape, bool)
        editor = SelectionEditor(np.full((*shape, 3), 100, np.uint8), None, labels, empty, empty)
        settings = {'modules': [], 'strength': .8, 'structure_protection': 0}
        brush = {'stage': 'range', 'tool': 'include', 'shape': 'brush', 'geometry': 'continuous',
                 'radius': 2, 'feather': 0, 'points': [[100, 100], [100, 140]]}
        old = render_selection(editor, settings, [brush], 1)
        new = render_selection(editor, settings, [{**brush, 'adaptive_edge': True}], 1)
        np.testing.assert_array_equal(new.selected, old.selected)
        self.assertAlmostEqual(float(old.weight[120, 100]), .4)
        self.assertAlmostEqual(float(new.weight[120, 100]), .8)
        self.assertEqual(new.weight[120, 103], 0)
        with self.assertRaisesRegex(ValueError, '随笔径'):
            binary_corrections(editor, new, settings)
        binary_corrections(editor, old, settings)

        finer_edge = render_selection(editor, settings, [{**brush, 'radius': 9, 'adaptive_edge': True}], 1)
        np.testing.assert_allclose(finer_edge.weight[120, [106, 107, 108, 109, 110]], [.8, .8, .6, .2, 0], atol=1e-6)
        feathered = render_selection(editor, settings, [{**brush, 'radius': 9, 'feather': 4, 'adaptive_edge': True}], 1)
        self.assertAlmostEqual(float(feathered.weight[120, 100]), .8)
        self.assertTrue(0 < feathered.weight[120, 107] < .4)
        self.assertEqual(feathered.weight[120, 109], 0)

    def test_adaptive_edge_cache_updates_unchanged_selection_and_restores_on_undo(self):
        from gpt_animeimage_denoise.workbench.masks import WorkbenchWeightCache, render_selection

        shape = (512, 512)
        empty = np.zeros(shape, bool)
        editor = SelectionEditor(np.full((*shape, 3), 100, np.uint8), None, empty.astype(np.uint8), empty, empty)
        settings = {'modules': [], 'structure_protection': 0}
        brush = {'stage': 'range', 'tool': 'include', 'radius': 2, 'feather': 0, 'points': [[66, 70], [66, 90]]}
        adaptive = {**brush, 'adaptive_edge': True}
        reset = {'stage': 'range', 'tool': 'reset'}
        cache = WorkbenchWeightCache(editor)
        edits = [brush, adaptive, reset]
        for cursor, expected in [(1, .853553), (2, 1), (1, .853553), (2, 1), (3, 0), (2, 1)]:
            actual = render_selection(editor, settings, edits, cursor, weight_cache=cache)
            reference = render_selection(editor, settings, edits, cursor)
            self.assertAlmostEqual(float(actual.weight[80, 66]), expected, places=6)
            np.testing.assert_array_equal(actual.weight, reference.weight)

    def test_adaptive_edge_join_blends_across_shared_selection_at_equal_depth(self):
        from gpt_animeimage_denoise.workbench.masks import WorkbenchWeightCache, render_selection

        shape = (1280, 1280)
        empty = np.zeros(shape, bool)
        editor = SelectionEditor(np.full((*shape, 3), 100, np.uint8), None, empty.astype(np.uint8), empty, empty)
        settings = {'modules': [], 'structure_protection': 0}
        brush = {'stage': 'range', 'tool': 'include', 'shape': 'brush', 'geometry': 'continuous', 'adaptive_edge': True}
        wide = {**brush, 'radius': 30, 'points': [[640, 500], [640, 750]]}
        thin = {**brush, 'radius': 3, 'points': [[663, 629], [663, 656]]}
        cache = WorkbenchWeightCache(editor)
        before = render_selection(editor, settings, [wide, thin], 1, weight_cache=cache)
        joined = render_selection(editor, settings, [wide, thin], 2, weight_cache=cache)
        np.testing.assert_array_equal(joined.selected, before.selected)
        section = joined.weight[610:644, 666]
        self.assertAlmostEqual(float(section[0]), .5)
        self.assertEqual(section[-1], 1)
        self.assertLessEqual(float(np.abs(np.diff(section)).max()), .25)
        self.assertTrue(((section > .5) & (section < 1)).any())
        full = render_selection(editor, settings, [wide, thin], 2)
        np.testing.assert_array_equal(joined.weight, full.weight)
        undone = render_selection(editor, settings, [wide, thin], 1, weight_cache=cache)
        np.testing.assert_array_equal(undone.weight, before.weight)

        detached = {**thin, 'points': [[675, 629], [675, 656]]}
        separate = render_selection(editor, settings, [wide, detached], 2, weight_cache=cache)
        np.testing.assert_array_equal(separate.weight[:, :672], before.weight[:, :672])
        self.assertEqual(separate.weight[640, 675], 1)

    def test_continuous_brush_uses_distance_to_the_segment_at_subpixel_positions(self):
        from gpt_animeimage_denoise.workbench.masks import _rasterize

        shape = (130, 200)
        points = [[22.25, 24.625], [164.25, 87.625]]
        brush = {'stage': 'range', 'tool': 'include', 'shape': 'brush', 'points': points,
                 'radius': 14, 'geometry': 'continuous'}
        start, end = np.asarray(points)
        dx, dy = end - start
        y, x = np.indices(shape, dtype=np.float64) - start[::-1, None, None]
        along = np.clip((x * dx + y * dy) / (dx ** 2 + dy ** 2), 0, 1)
        distance = np.hypot(x - dx * along, y - dy * along)
        edge = np.clip((14 - distance) / 6, 0, 1)
        for feather, expected in [(0, distance <= 14), (6, edge ** 2 * (3 - 2 * edge))]:
            with self.subTest(feather=feather):
                tile, local = _rasterize({**brush, 'feather': feather}, shape)
                actual = np.zeros(shape, np.float32)
                actual[tile] = local
                np.testing.assert_allclose(actual, expected, atol=3e-6)

        old_brush = {key: value for key, value in brush.items() if key != 'geometry'}
        previous = _rasterize({**old_brush, 'points': [[22, 24], [164, 87]]}, shape)[1]
        np.testing.assert_array_equal(_rasterize(old_brush, shape)[1], previous)

    def test_part_protection_filters_the_current_source_labels(self):
        protected = [*MODULES, 'object']
        first = {'stage': 'parts', 'tool': 'classify', 'module': 'outfit',
                 'shape': 'polygon', 'points': [[8, 18], [50, 18], [50, 23], [8, 23]],
                 'protected_parts': protected}
        result = self.selection([first])
        self.assertEqual(result.labels[20, 20], MODULES['outfit'])
        self.assertEqual(result.labels[20, 19], MODULES['skin'])
        self.assertEqual(result.labels[20, 45], MODULES['hair'])
        self.assertEqual(result.labels[20, 9], MODULES['background'])
        self.assertTrue(result.objects[20, 21])
        self.assertEqual(result.edit_feedback['changed_pixels'], 1)
        self.assertGreater(result.edit_feedback['skipped_protected_pixels'], 0)

        second = {**first, 'module': 'mouth', 'protected_parts': [name for name in protected if name != 'skin']}
        result = self.selection([first, second])
        self.assertEqual(result.labels[20, 19], MODULES['mouth'])
        self.assertEqual(result.labels[20, 20], MODULES['outfit'])
        self.assertTrue(result.objects[20, 21])
        undo = self.selection([first, second], cursor=1)
        self.assertEqual(undo.labels[20, 19], MODULES['skin'])

        all_protected = self.selection([{**first, 'protected_parts': [*protected, 'unknown']}])
        np.testing.assert_array_equal(all_protected.labels, self.editor.labels)
        self.assertEqual(all_protected.edit_feedback['reason'], 'protected')

    def test_range_locks_mask_local_edits_and_saved_sam_outlines(self):
        from gpt_animeimage_denoise.workbench.masks import render_selection

        brush = {'stage': 'range', 'points': [[36, 40]], 'radius': 9}
        locked = {**brush, 'protected_parts': ['skin']}
        for tool, settings, prior, expected in (
            ('include', {'modules': []}, [], [0, 1]),
            ('exclude', {}, [], [1, 0]),
            ('protect', {}, [], [1, .25]),
            ('mix', {}, [], [1, .25]),
            ('erase', {}, [{**brush, 'tool': 'exclude'}], [0, 1]),
        ):
            with self.subTest(tool=tool):
                edit = {**locked, 'tool': tool}
                if tool in ('protect', 'mix'):
                    edit['strength'] = .75
                result = self.selection([*prior, edit], structure_protection=0, **settings)
                np.testing.assert_allclose(result.weight[40, [31, 41]], expected)
                self.assertGreater(result.edit_feedback['skipped_protected_pixels'], 0)

        outline = np.zeros(self.editor.labels.shape, bool)
        outline[37:44, 29:43] = True
        sam = {'stage': 'range', 'tool': 'exclude', 'shape': 'sam', 'token': 'verified', 'protected_parts': ['skin']}
        result = render_selection(self.editor, {'structure_protection': 0}, [sam], 1, object_mask=lambda _: outline)
        np.testing.assert_allclose(result.weight[40, [31, 41]], [1, 0])

    def test_range_locks_preserve_manual_range_through_reset_and_invert(self):
        exclude = {'stage': 'range', 'tool': 'exclude', 'points': [[36, 40]], 'radius': 9}
        for tool in ('reset', 'invert'):
            with self.subTest(tool=tool):
                bulk = {'stage': 'range', 'tool': tool, 'protected_parts': ['skin']}
                result = self.selection([exclude, bulk], structure_protection=0)
                np.testing.assert_allclose(result.weight[40, [31, 41]], [0, 1])
                self.assertTrue(result.selected[40, 25])
                self.assertEqual(result.selected[40, 63], tool == 'reset')
        undone = self.selection([exclude, {'stage': 'range', 'tool': 'reset', 'protected_parts': ['skin']}],
                                cursor=1, structure_protection=0)
        np.testing.assert_allclose(undone.weight[40, [31, 41]], [0, 0])

    def test_range_locks_take_priority_over_model_override_and_unknown_confirmation(self):
        brush = {'stage': 'range', 'tool': 'include', 'points': [[20, 20]], 'radius': 3}
        protected = ['unknown', 'object']
        include = {**brush, 'allow_protected': True, 'protected_parts': protected}
        included = self.selection([include], structure_protection=0)
        self.assertFalse(included.selected[20, 20:22].any())
        self.assertEqual(included.edit_feedback['skipped_protected_pixels'], 2)

        confirm = {'stage': 'range', 'tool': 'resolve_unknown', 'protected_parts': ['unknown']}
        confirmed = self.selection([brush, confirm], structure_protection=0)
        self.assertFalse(confirmed.selected[20, 20])
        self.assertEqual(confirmed.labels[20, 20], 255)

    def test_remove_part_brush_and_lasso_only_unassign_the_chosen_part(self):
        geometries = [
            {'shape': 'brush', 'points': [[36, 20]], 'radius': 5},
            {'shape': 'polygon', 'points': [[8, 18], [50, 18], [50, 23], [8, 23]]},
        ]
        for geometry in geometries:
            with self.subTest(shape=geometry['shape']):
                edit = {**geometry, 'stage': 'parts', 'tool': 'remove_part', 'module': 'hair'}
                result = self.selection([edit])
                changed = result.labels != self.editor.labels
                self.assertTrue(changed[20, 38])
                self.assertFalse(changed[20, 34])
                self.assertFalse(changed[20, 57])
                self.assertTrue(np.all(self.editor.labels[changed] == MODULES['hair']))
                self.assertTrue(np.all(result.labels[changed] == 255))
                np.testing.assert_array_equal(result.unknown, self.editor.unknown | changed)
                np.testing.assert_array_equal(result.objects, self.editor.objects)
                self.assertEqual(result.edit_feedback['changed_pixels'], int(changed.sum()))
                np.testing.assert_array_equal(self.selection([edit], cursor=0).labels, self.editor.labels)

    def test_semantic_veto_override_in_new_range_keeps_labels_and_refuses_lossy_cli(self):
        from gpt_animeimage_denoise.workbench.masks import binary_corrections

        edit = {'stage': 'range', 'tool': 'include', 'allow_protected': True, 'radius': 2, 'points': [[21, 20]]}
        selected = self.selection([edit])
        self.assertTrue(selected.selected[20, 20:22].all())
        self.assertTrue(selected.objects[20, 21])
        self.assertTrue(selected.unknown[20, 20])
        with self.assertRaisesRegex(ValueError, '二值'):
            binary_corrections(self.editor, selected, {})
        for name in ('unknown', 'background'):
            classified = self.selection([{'stage': 'parts', 'tool': 'classify', 'module': name,
                                          'radius': 2, 'points': [[50, 40]]}])
            with self.assertRaisesRegex(ValueError, '二值'):
                binary_corrections(self.editor, classified, {})

    def test_feathered_exclusion_reports_partial_reduction_separately(self):
        settings = {'modules': ['hair'], 'structure_protection': 0}
        before = self.selection(**settings)
        edit = {'stage': 'range', 'tool': 'exclude', 'points': [[52, 40]], 'radius': 12, 'feather': 12}
        after = self.selection([edit], **settings)
        fully_removed = int(np.count_nonzero((before.weight > 0) & (after.weight == 0)))
        partially_reduced = int(np.count_nonzero((after.weight > 0) & (after.weight < before.weight)))
        self.assertEqual((fully_removed, partially_reduced), (1, 436))
        self.assertEqual(after.edit_feedback, {'tool': 'exclude', 'changed_pixels': 437,
                                              'partially_reduced_pixels': 436, 'reason': 'changed'})
        hard_result = self.selection([{**edit, 'feather': .0001}], **settings)
        self.assertEqual(hard_result.edit_feedback.get('partially_reduced_pixels'), 0)

    def test_upstream_edit_replays_existing_range_strokes_on_corrected_labels(self):
        brush = {'radius': 2, 'points': [[21, 20]]}
        include = {**brush, 'stage': 'range', 'tool': 'include'}
        part = {**brush, 'stage': 'parts', 'tool': 'classify', 'module': 'outfit'}
        sequence = self.selection([include, part])
        np.testing.assert_array_equal(sequence.weight, self.selection([part, include]).weight)
        self.assertTrue(sequence.selected[20, 20:22].all())
        self.assertEqual(sequence.edit_feedback['tool'], 'classify')
        self.assertFalse(self.selection([include, part], cursor=1).selected[20, 20:22].any())

    def test_confirm_staged_unknown_preserves_class_vetoes_and_feather(self):
        brush = {'stage': 'range', 'tool': 'include', 'radius': 4, 'feather': 4, 'points': [[18, 20]]}
        confirm = {'stage': 'range', 'tool': 'resolve_unknown'}
        before = self.selection([brush])
        self.assertEqual(before.edit_feedback['skipped_unknown_pixels'], 1)
        result = self.selection([brush, confirm], structure_protection=0)
        self.assertEqual(result.labels[20, 20], 255)
        self.assertTrue(result.unknown[20, 20])
        self.assertTrue(result.selected[20, 20])
        self.assertAlmostEqual(result.weight[20, 20], .5)
        self.assertFalse(result.selected[20, 21:23].any())
        with self.assertRaises(ValueError):
            self.selection([brush, dict(confirm, module='hair')])

    def test_staged_soft_protection_also_attenuates_absolute_mix(self):
        brush = {'stage': 'range', 'points': [[50, 40]], 'radius': 8}
        protect = {**brush, 'tool': 'protect', 'strength': .8, 'feather': 4}
        mix = {**brush, 'tool': 'mix', 'strength': 0, 'feather': 0}
        for edits in ([protect, mix], [mix, protect]):
            result = self.selection(edits, structure_protection=0, strength=.25)
            np.testing.assert_allclose(result.weight[40, [50, 56]], [.2, .6], atol=1e-6)

    def test_generic_sam_applies_each_stage_tool_exactly_as_its_saved_outline(self):
        from gpt_animeimage_denoise.workbench.masks import render_selection

        outline = np.zeros(self.editor.labels.shape, bool)
        outline[32:41, 15:53] = True
        polygon = {'shape': 'polygon', 'points': [[15, 32], [52, 32], [52, 40], [15, 40]]}
        for stage, tool, params in [('parts', 'classify', {'module': 'eyes'}), ('parts', 'erase', {}),
                ('range', 'include', {}), ('range', 'exclude', {}), ('range', 'protect', {'strength': .6}),
                ('range', 'mix', {'strength': .4}), ('range', 'erase', {})]:
            edit = {'stage': stage, 'tool': tool, **params}
            prefix = [{**polygon, 'stage': 'parts', 'tool': 'classify', 'module': 'mouth'},
                      {**polygon, 'stage': 'range', 'tool': 'exclude'}] if tool == 'erase' else []
            expected = self.selection([*prefix, {**edit, **polygon}], modules=['hair'])
            records = [*prefix, {**edit, 'shape': 'sam', 'token': 'verified'}]
            actual = render_selection(self.editor, {'modules': ['hair']}, records, len(records), object_mask=lambda _: outline)
            for name in ('labels', 'objects', 'unknown', 'selected', 'weight', 'soft_protection'):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=stage + ' ' + tool)

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

    def test_coverage_is_reused_across_history_and_settings_with_exact_replay(self):
        from gpt_animeimage_denoise.workbench import masks

        old = {'tool': 'protect', 'points': [[52, 37]], 'radius': 7, 'strength': .8}
        lasso = {'stage': 'parts', 'tool': 'classify', 'module': 'outfit', 'shape': 'polygon',
                 'points': [[25, 23], [45, 23], [45, 33], [25, 33]]}
        stroke = {'stage': 'range', 'tool': 'exclude', 'geometry': 'continuous', 'radius': 9,
                  'points': [[48.5, 36.25], [61.25, 42.5]], 'feather': 4}
        edits = [old, lasso, stroke]
        cases = [
            (edits, 3, {}, 3),
            (edits, 2, {}, 3),
            (edits, 3, {}, 3),
            (edits, 3, {'modules': ['hair', 'outfit'], 'strength': .47, 'structure_protection': .5,
                        'part_strengths': {'hair': .73}}, 3),
            ([{**old, 'strength': .3}, lasso, stroke], 3, {}, 3),
            ([old, lasso, {**stroke, 'feather': 2}], 3, {}, 4),
            ([{**old, 'tool': 'exclude'}, lasso, stroke], 3, {}, 5),
            (edits, 3, {}, 5),
        ]
        expected = [masks.render_selection(self.editor, settings, records, cursor)
                    for records, cursor, settings, _ in cases]
        cache = masks.WorkbenchWeightCache(self.editor)
        with patch.object(masks, '_rasterize', wraps=masks._rasterize) as rasterize:
            for (records, cursor, settings, calls), reference in zip(cases, expected, strict=True):
                actual = masks.render_selection(self.editor, settings, deepcopy(records), cursor, weight_cache=cache)
                self.assertEqual(rasterize.call_count, calls)
                for field in ('labels', 'objects', 'unknown', 'selected', 'weight', 'soft_protection'):
                    np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field))
                self.assertEqual(actual.edit_feedback, reference.edit_feedback)

    def test_saved_sam_coverage_is_reused_only_with_the_bound_mask_reader(self):
        from gpt_animeimage_denoise.workbench import masks

        first = np.zeros_like(self.editor.objects)
        second = first.copy()
        first[31:36, 43:47] = True
        second[37:42, 55:59] = True
        reader = Mock(side_effect=lambda token: first if token == 'legacy' else second)
        cache = masks.WorkbenchWeightCache(self.editor, object_mask=reader)
        records = [{'tool': 'sam_object', 'token': 'legacy'},
                   {'tool': 'exclude', 'stage': 'range', 'shape': 'sam', 'token': 'range'}]
        reference = masks.render_selection(self.editor, {}, records, 2,
                                           object_mask=lambda token: first if token == 'legacy' else second)
        for cursor in (2, 1, 2):
            actual = masks.render_selection(self.editor, {}, records, cursor, object_mask=reader, weight_cache=cache)
        self.assertEqual(reader.call_count, 2)
        np.testing.assert_array_equal(actual.objects, reference.objects)
        np.testing.assert_array_equal(actual.weight, reference.weight)

        other_reader = lambda token: second if token == 'legacy' else first
        other = masks.render_selection(self.editor, {}, records, 2, object_mask=other_reader, weight_cache=cache)
        expected = masks.render_selection(self.editor, {}, records, 2, object_mask=other_reader)
        np.testing.assert_array_equal(other.objects, expected.objects)
        np.testing.assert_array_equal(other.weight, expected.weight)

    def test_coverage_cache_stays_bounded_without_evicting_reused_strokes(self):
        from gpt_animeimage_denoise.workbench import masks

        cache = masks.WorkbenchWeightCache(self.editor)
        edits = [{'tool': 'exclude', 'points': [[42, 22]], 'radius': radius} for radius in (1, 12)]
        with patch.object(masks, '_MAX_COVERAGE_CACHE_BYTES', 256), \
                patch.object(masks, '_rasterize', wraps=masks._rasterize) as rasterize:
            first = masks.render_selection(self.editor, {}, edits, 2, weight_cache=cache)
            again = masks.render_selection(self.editor, {}, edits, 2, weight_cache=cache)
            self.assertEqual(rasterize.call_count, 3)
            np.testing.assert_array_equal(first.weight, again.weight)

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
