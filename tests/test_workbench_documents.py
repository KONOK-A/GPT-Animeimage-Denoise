from base64 import b64decode
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
from threading import Event, current_thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from gpt_animeimage_denoise.editor import SelectionEditor
from gpt_animeimage_denoise.workbench.documents import Documents, RevisionConflict


class DocumentsTest(unittest.TestCase):
    def test_warm_selection_reuses_unchanged_parts_overlay(self):
        from unittest.mock import patch
        from gpt_animeimage_denoise.workbench import documents as docs

        session = self.open_session('source.png', self.workspace)
        settings = docs.DEFAULTS
        before = docs._selection(session.editor, settings, [], 0)
        images = docs._selection_images(session, before)
        after = docs._selection(session.editor, settings, [{'tool': 'exclude', 'points': [[10, 10]], 'radius': 3}], 1)
        exact = docs._selection_images(session, after)
        with patch.object(docs, '_indexed_png', wraps=docs._indexed_png) as png:
            reused = docs._selection_images(session, after, previous=(before, images))
        self.assertEqual(reused, exact)
        self.assertEqual(png.call_count, 1)

    def test_adaptive_edge_marker_is_saved_only_for_range_include_brushes(self):
        opened = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = opened['id']
        ready = self.ready(doc, opened['revision'])
        brush = {'stage': 'range', 'tool': 'include', 'shape': 'brush', 'radius': 2,
                 'points': [[10, 10]], 'adaptive_edge': True}
        payload = {**self.update(ready), 'edits': [brush], 'cursor': 1}
        saved = self.docs.update(doc, payload)
        self.assertEqual(saved['edits'], [brush])
        for invalid in ({**brush, 'tool': 'exclude'}, {**brush, 'adaptive_edge': 1},
                        {**brush, 'shape': 'polygon', 'points': [[10, 10], [14, 10], [14, 14]]}):
            with self.assertRaises(ValueError):
                self.docs.update(doc, {**self.update(saved), 'edits': [invalid], 'cursor': 1})

    def test_history_accepts_range_protection_with_validated_part_names(self):
        opened = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = opened['id']
        ready = self.ready(doc, opened['revision'])
        edits = [
            {'stage': 'range', 'tool': 'exclude', 'radius': 2, 'points': [[10, 10]], 'protected_parts': ['skin']},
            {'stage': 'range', 'tool': 'reset', 'protected_parts': ['hair', 'unknown']},
        ]
        saved = self.docs.update(doc, {**self.update(ready), 'edits': edits, 'cursor': 2})
        self.assertEqual(saved['edits'], edits)
        for invalid in (['invalid'], ['skin', 'skin'], 'skin'):
            with self.assertRaisesRegex(ValueError, '保护分区'):
                self.docs.update(doc, {**self.update(saved), 'edits': [{**edits[0], 'protected_parts': invalid}], 'cursor': 1})

    def test_saved_adaptive_preview_refreshes_once_without_replacing_the_pinned_image(self):
        opened = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = opened['id']
        ready = self.ready(doc, opened['revision'])
        brush = {'stage': 'range', 'tool': 'include', 'radius': 2, 'points': [[10, 10]], 'adaptive_edge': True}
        queued = self.docs.update(doc, {**self.update(ready), 'edits': [brush], 'cursor': 1})
        ready = self.ready(doc, queued['revision'])
        self.docs.pin(doc, ready['revision'])
        pinned = self.docs.asset(doc, 'a', revision=ready['revision'])
        self.docs.close()
        draft = self.workspace / 'documents' / doc / 'draft.json'
        stored = json.loads(draft.read_text(encoding='utf-8'))
        stored.pop('_range_edge_version', None)
        draft.write_text(json.dumps(stored), encoding='utf-8')

        with Documents(self.workspace, self.open_session) as reopened:
            with self.assertRaises(RevisionConflict):
                reopened.changes(doc, ready['revision'], stage='range')
            next_revision = reopened.get(doc)['revision']
            self.assertEqual(next_revision, ready['revision'] + 1)
            published = self.ready(doc, next_revision, docs=reopened)
            self.assertEqual(reopened.get(doc), published)
            self.assertEqual(published['edits'], [brush])
            self.assertEqual(published['selection_revision'], ready['selection_revision'] + 1)
            self.assertEqual(reopened.asset(doc, 'a', revision=ready['revision']), pinned)
        self.assertEqual(json.loads(draft.read_text(encoding='utf-8'))['_source_epoch'], stored['_source_epoch'])

    def test_weight_overlay_distinguishes_weak_medium_and_strong_weights(self):
        from gpt_animeimage_denoise.workbench import documents as docs

        session = self.open_session('source.png', self.workspace)
        mask = docs._selection(session.editor, docs.DEFAULTS, [], 0)
        mask.weight[12, 4:8] = [0, .1, .5, 1]
        encoded = docs._selection_images(session, mask)['weight_overlay.png']
        with Image.open(BytesIO(encoded)) as image:
            self.assertEqual(image.mode, 'P')
            pixels = np.asarray(image.convert('RGBA'))[12, 4:8]
        transparent, weak, middle, strong = pixels.tolist()
        self.assertEqual(transparent[3], 0)
        self.assertGreaterEqual(weak[3], 50)
        self.assertGreater(weak[1], 190)
        self.assertLess(middle[1], 140)
        self.assertGreater(middle[2], 240)
        self.assertEqual(strong[:3], [241, 72, 169])
        self.assertLess(weak[3], middle[3])
        self.assertLess(middle[3], strong[3])

    def test_legacy_indexed_weight_overlay_gets_the_new_palette_without_a_new_revision(self):
        from gpt_animeimage_denoise.workbench.documents import _indexed_png

        opened = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = opened['id']
        ready = self.ready(doc, opened['revision'])
        revision = ready['revision']
        indices = np.zeros(self.alpha.shape, np.uint8)
        indices[12, 4:7] = [0, 28, 140]
        old_png = _indexed_png(Image.fromarray(indices), bytes([79, 159, 232]) * 256, range(256))
        path = self.docs._render_path(doc, revision) / 'weight_overlay.png'
        path.write_bytes(old_png)
        with patch.object(self.docs, '_open_session') as open_session:
            data = self.docs.asset(doc, 'weight_overlay', revision=revision)
            open_session.assert_not_called()
        with Image.open(BytesIO(data)) as image:
            pixels = np.asarray(image.convert('RGBA'))[12, 4:7]
        self.assertEqual(pixels[0, 3], 0)
        self.assertGreater(pixels[1, 1], 170)
        np.testing.assert_allclose(pixels[2, :3], [241, 72, 169], atol=1)
        self.assertEqual(path.read_bytes(), old_png)
        self.assertEqual(self.docs.get(doc), ready)

    def test_changes_palette_prioritizes_weight_and_hides_transparent_pixels(self):
        from gpt_animeimage_denoise.workbench.documents import _changes_overlay

        auto_weight = np.array([[0, .5, .4, .8, .4, .5, 0, 0]], np.float32)
        weight = np.array([[.2, 0, .7, .3, .4, .5, .4, 0]], np.float32)
        editor = SimpleNamespace(labels=np.ones(weight.shape, np.uint8),
                                 unknown=np.zeros(weight.shape, bool),
                                 objects=np.zeros(weight.shape, bool),
                                 alpha=np.array([[255, 128, 255, 255, 255, 255, 0, 255]], np.uint8))
        labels = editor.labels.copy()
        labels[0, [0, 4]] = 2
        mask = SimpleNamespace(weight=weight, labels=labels, unknown=editor.unknown,
                               objects=editor.objects.copy())
        mask.objects[0, 7] = True

        counts, overlay = _changes_overlay(editor, auto_weight, mask)
        self.assertEqual(counts, {'added': 1, 'removed': 1, 'stronger': 1, 'weaker': 1, 'reclassified': 2})
        self.assertTrue(overlay.startswith('data:image/png;base64,'))
        pixels = np.asarray(Image.open(BytesIO(b64decode(overlay.split(',', 1)[1]))))
        self.assertEqual(pixels.shape, (1, 8, 4))
        self.assertEqual(pixels[0, [0, 1, 2, 3, 4, 7], :3].tolist(), [
            [56, 189, 248], [244, 119, 118], [81, 223, 156], [248, 191, 86], [190, 154, 255], [190, 154, 255]])
        self.assertGreater(pixels[0, 0, 3], pixels[0, 1, 3])
        self.assertEqual(pixels[0, [5, 6], 3].tolist(), [0, 0])

    def test_part_changes_use_destination_colors_and_hatch_only_new_unknown_pixels(self):
        from gpt_animeimage_denoise.workbench.documents import _part_changes_overlay

        empty = np.zeros((1, 12), bool)
        editor = SimpleNamespace(labels=np.ones(empty.shape, np.uint8), objects=empty, unknown=empty,
                                 alpha=np.array([[255, 128, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255]], np.uint8))
        labels = np.array([[3, 6, 0, 0, 255, 1, 1, 1, 255, 1, 5, 1]], np.uint8)
        objects, unknown = empty.copy(), empty.copy()
        objects[0, 3] = True
        unknown[0, [4, 8]] = True

        changes = _part_changes_overlay(editor, labels, objects, unknown)
        counts, overlay = changes[:2]
        self.assertEqual(counts['reclassified'], 6)
        with Image.open(BytesIO(b64decode(overlay.split(',', 1)[1]))) as encoded:
            pixels = np.asarray(encoded)
            self.assertEqual(pixels[0, [0, 1, 2, 3], :3].tolist(),
                             [[101, 135, 223], [139, 120, 199], [56, 151, 188], [224, 136, 68]])
            self.assertEqual(pixels[0, [4, 8]].tolist(), [[22, 22, 22, 158], [245, 245, 245, 158]])
            self.assertEqual(pixels[0, [5, 6, 7, 9, 10, 11], 3].tolist(), [0] * 6)
            self.assertEqual(pixels[0, [0, 1], 3].tolist(), [180, 90])
        self.assertEqual(changes[2], ['background', 'hair', 'outfit', 'object', 'unknown'])

    def test_edit_render_defers_corrections_and_saves_small_overlay_assets(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        with patch.object(self.docs, '_encode_corrections', wraps=self.docs._encode_corrections) as encode:
            state = self.ready(doc, state['revision'])
            edit = {'stage': 'range', 'tool': 'exclude', 'radius': 2, 'points': [[12, 12]]}
            state = self.docs.update(doc, {**self.update(state), 'edits': [edit], 'cursor': 1})
            ready = self.ready(doc, state['revision'])
            self.assertEqual(encode.call_count, 0)
            revision = ready['revision']
            self.assertIs(ready['b']['images']['parts_overlay'], True)
            self.assertIs(ready['b']['images']['weight_overlay'], True)
            self.assertFalse((self.docs._render_path(doc, revision) / 'weight.png').exists())
            for view in ('weight_overlay', 'parts_overlay'):
                with Image.open(BytesIO(self.docs.asset(doc, view, revision=revision))) as image:
                    self.assertEqual(image.mode, 'P')
                    self.assertEqual(image.size, (28, 24))
            with Image.open(BytesIO(self.docs.asset(doc, 'parts_overlay', revision=revision, hidden_parts=('skin',)))) as image:
                self.assertFalse(np.asarray(image.convert('RGBA'))[..., 3].any())
            for view in ('parts', 'weight'):
                with Image.open(BytesIO(self.docs.asset(doc, view, revision=revision))) as image:
                    np.testing.assert_array_equal(np.asarray(image)[..., 3], self.alpha)
                    self.assertEqual(image.info['icc_profile'], b'fixture-icc')
            exported = self.docs.corrections(doc, revision)
            self.assertEqual(encode.call_count, 1)
            record = json.loads(exported.read_text(encoding='utf-8'))
            self.assertEqual(record['source_sha256'], doc)
            with Image.open(exported.parent / record['protect_mask']) as image:
                self.assertEqual(image.getpixel((12, 12)), 255)

    def test_changes_compare_same_settings_and_track_undo_without_render_side_work(self):
        from gpt_animeimage_denoise.workbench import documents as module

        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        ready = self.ready(doc, state['revision'])
        with patch.object(module, '_changes_overlay', wraps=module._changes_overlay) as overlay:
            settings = {**ready['settings'], 'strength': .5}
            noop = {'tool': 'include', 'points': [[13, 12]], 'radius': 3}
            ready = self.docs.update(doc, {'revision': ready['revision'], 'settings': settings,
                                          'edits': [noop], 'cursor': 1})
            ready = self.ready(doc, ready['revision'])
            overlay.assert_not_called()
            result = self.docs.changes(doc, ready['revision'])
            self.assertIsNone(result['overlay'])
            self.assertFalse(any(result['counts'].values()))
            self.docs.changes(doc, ready['revision'])
            overlay.assert_called_once()
            excluded = {'tool': 'exclude', 'points': [[13, 12]], 'radius': 3}
            updated = self.docs.update(doc, {'revision': ready['revision'], 'settings': settings,
                                            'edits': [noop, excluded], 'cursor': 2})
            updated = self.ready(doc, updated['revision'])
            self.assertEqual(overlay.call_count, 1)
            with self.assertRaises(RevisionConflict):
                self.docs.changes(doc, ready['revision'])
            result = self.docs.changes(doc, updated['revision'])
            self.assertEqual(result['revision'], updated['revision'])
            self.assertEqual(result['counts']['removed'], 29)
            self.assertEqual(sum(result['counts'].values()), 29)
            reverted = self.docs.update(doc, {'revision': updated['revision'], 'settings': settings,
                                             'edits': [noop, excluded], 'cursor': 0})
            reverted = self.ready(doc, reverted['revision'])
            self.assertIsNone(self.docs.changes(doc, reverted['revision'])['overlay'])
            self.assertEqual(overlay.call_count, 2)

    def test_changes_computation_does_not_block_render_and_rejects_stale_snapshot(self):
        from gpt_animeimage_denoise.workbench import documents as module

        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        ready = self.ready(doc, state['revision'])
        ready = self.docs.update(doc, {**self.update(ready), 'cursor': 1,
                                     'edits': [{'tool': 'exclude', 'points': [[13, 12]], 'radius': 3}]})
        ready = self.ready(doc, ready['revision'])
        overlay = module._changes_overlay

        def gated_overlay(*args):
            self.entered.set()
            self.assertTrue(self.release.wait(5))
            return overlay(*args)

        with patch.object(module, '_changes_overlay', side_effect=gated_overlay), \
                ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.docs.changes, doc, ready['revision'])
            try:
                self.assertTrue(self.entered.wait(3))
                undo = self.docs.update(doc, {**self.update(ready), 'cursor': 0})
                self.ready(doc, undo['revision'])
            finally:
                self.release.set()
            with self.assertRaises(RevisionConflict):
                pending.result(timeout=3)

    def test_node_changes_use_corrected_parts_as_the_range_baseline(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        ready = self.ready(doc, state['revision'])
        classify = {'stage': 'parts', 'tool': 'classify', 'module': 'outfit', 'points': [[8, 12]], 'radius': 2}
        exclude = {'stage': 'range', 'tool': 'exclude', 'points': [[20, 12]], 'radius': 2}
        ready = self.docs.update(doc, {**self.update(ready), 'edits': [classify], 'cursor': 1})
        ready = self.ready(doc, ready['revision'])
        parts = self.docs.changes(doc, ready['revision'], stage='parts')
        range_only = self.docs.changes(doc, ready['revision'], stage='range')
        self.assertEqual(parts['counts'], {'reclassified': 13})
        self.assertEqual(parts['parts'], ['outfit'])
        self.assertEqual(range_only['counts'], {'added': 0, 'removed': 0, 'stronger': 0, 'weaker': 0})
        self.assertIsNone(range_only['overlay'])

        ready = self.docs.update(doc, {**self.update(ready), 'edits': [classify, exclude], 'cursor': 2})
        ready = self.ready(doc, ready['revision'])
        range_only = self.docs.changes(doc, ready['revision'], stage='range')
        parts = self.docs.changes(doc, ready['revision'], stage='parts')
        self.assertEqual(parts['counts'], {'reclassified': 13})
        self.assertEqual(range_only['counts'], {'added': 0, 'removed': 13, 'stronger': 0, 'weaker': 0})
        for changes, painted_x, other_x in [(parts, 8, 20), (range_only, 20, 8)]:
            with Image.open(BytesIO(b64decode(changes['overlay'].split(',', 1)[1]))) as overlay:
                self.assertGreater(overlay.getpixel((painted_x, 12))[3], 0)
                self.assertEqual(overlay.getpixel((other_x, 12))[3], 0)
        self.assertEqual(self.docs.changes(doc, ready['revision'], stage='range'), range_only)

        undone = self.docs.update(doc, {**self.update(ready), 'cursor': 1})
        undone = self.ready(doc, undone['revision'])
        self.assertIsNone(self.docs.changes(doc, undone['revision'], stage='range')['overlay'])
        self.assertEqual(self.docs.changes(doc, undone['revision'], stage='parts')['counts'], {'reclassified': 13})

    def test_legacy_range_unlock_is_not_reported_as_reclassification(self):
        self.objects[12, 14] = True
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        ready = self.ready(doc, state['revision'])
        unlock = {'tool': 'include', 'points': [[14, 12]], 'radius': 1, 'allow_protected': True}
        ready = self.docs.update(doc, {**self.update(ready), 'edits': [unlock], 'cursor': 1})
        ready = self.ready(doc, ready['revision'])
        self.assertIsNone(self.docs.changes(doc, ready['revision'], stage='parts')['overlay'])
        self.assertEqual(self.docs.changes(doc, ready['revision'], stage='range')['counts']['added'], 1)

    def test_concurrent_changes_requests_share_ready_revision(self):
        from gpt_animeimage_denoise.workbench import documents as module

        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        ready = self.ready(doc, state['revision'])
        ready = self.docs.update(doc, {**self.update(ready), 'cursor': 1,
                                     'edits': [{'tool': 'exclude', 'points': [[13, 12]], 'radius': 3}]})
        ready = self.ready(doc, ready['revision'])
        overlay = module._changes_overlay
        second_started = Event()

        def gated_overlay(*args):
            self.entered.set()
            self.assertTrue(self.release.wait(5))
            return overlay(*args)

        def second_request():
            second_started.set()
            return self.docs.changes(doc, ready['revision'])

        with patch.object(module, '_changes_overlay', side_effect=gated_overlay) as encode, \
                ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.docs.changes, doc, ready['revision'])
            try:
                self.assertTrue(self.entered.wait(3))
                second = pool.submit(second_request)
                self.assertTrue(second_started.wait(3))
            finally:
                self.release.set()
            self.assertEqual(first.result(timeout=3), second.result(timeout=3))
            encode.assert_called_once()

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.workspace = Path(folder.name)
        settings = patch.dict(os.environ, GAID_WORKBENCH_SETTINGS=str(self.workspace / 'preferences.json'))
        settings.start()
        self.addCleanup(settings.stop)
        self.rgb = np.full((24, 28, 3), 130, np.uint8)
        self.alpha = np.full(self.rgb.shape[:2], 128, np.uint8)
        self.alpha[:, :2] = 0
        self.content = BytesIO()
        Image.fromarray(np.dstack((self.rgb, self.alpha))).save(self.content, 'PNG', icc_profile=b'fixture-icc')
        self.release = Event()
        self.addCleanup(self.release.set)
        self.entered = Event()
        self.opened = []
        self.strengths = []
        self.suspected_objects = None
        self.unknown = np.zeros(self.rgb.shape[:2], bool)
        self.objects = np.zeros_like(self.unknown)
        self.predict_object = None
        self.docs = Documents(self.workspace, self.open_session)
        self.addCleanup(self.docs.close)

    def open_session(self, source, workspace):
        self.opened.append(source)
        labels = np.ones(self.unknown.shape, np.uint8)
        labels[self.unknown] = 255
        labels[self.objects] = 0
        editor = SelectionEditor(self.rgb, self.alpha, labels, self.objects, self.unknown)

        def candidate(method, *, lam, downscale):
            self.strengths.append(lam)
            if lam == .001:
                self.entered.set()
                self.release.wait(5)
            return np.full(self.rgb.shape, .2 if lam == .0005 else .4, dtype=np.float32)

        return SimpleNamespace(editor=editor, icc=b'fixture-icc', candidate=candidate,
                               suspected_objects=self.suspected_objects, predict_object=self.predict_object)

    def test_sam_object_preview_is_read_only_and_apply_preserves_existing_edits(self):
        def predict(positive, negative):
            self.assertEqual(negative, [[15, 12]])
            x, y = positive[0]
            mask = np.zeros(self.rgb.shape[:2], bool)
            mask[y - 1:y + 2, x - 1:x + 2] = True
            return mask

        self.predict_object = predict
        initial = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = initial['id']
        state = self.ready(doc, initial['revision'])
        self.docs.pin(doc, state['revision'])
        protected = BytesIO()
        region = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
        region[14, 6] = 255
        Image.fromarray(region).save(protected, 'PNG')
        record = {'version': 2, 'source_sha256': doc, 'size': [28, 24], 'protect_mask': 'region.png'}
        state = self.docs.import_correction(doc, state['revision'], record, {'region.png': protected.getvalue()})
        state = self.ready(doc, state['revision'])
        brush = {'tool': 'exclude', 'points': [[8, 8]], 'radius': 1}
        state = self.docs.update(doc, {**self.update(state), 'edits': [*state['edits'], brush], 'cursor': 2})
        state = self.ready(doc, state['revision'])
        before = self.docs.asset(doc, 'b', download=True)
        for x in (12, 18):
            revision = state['revision']
            preview = self.docs.preview_object(doc, revision, [[x, 12]], [[15, 12]])
            self.assertEqual(preview['pixels'], 9)
            self.assertEqual(self.docs.get(doc), state)
            state = self.docs.apply_object(doc, revision, preview['token'])
            state = self.ready(doc, state['revision'])
        self.assertEqual([edit['tool'] for edit in state['edits']], ['import', 'exclude', 'sam_object', 'sam_object'])
        self.assertEqual(state['a']['revision'], initial['revision'])
        output = self.docs.corrections(doc, state['revision']).parent / 'object_add_mask.png'
        with Image.open(output) as image:
            objects = np.asarray(image) > 0
        self.assertEqual(int(objects.sum()), 18)
        self.assertFalse(objects[12, 15])
        with Image.open(BytesIO(self.docs.asset(doc, 'b', download=True))) as image:
            for x, y in [(6, 14), (8, 8), (12, 12), (18, 12)]:
                np.testing.assert_array_equal(np.asarray(image)[y, x, :3], self.rgb[y, x])
        state = self.docs.update(doc, {**self.update(state), 'cursor': 2})
        self.ready(doc, state['revision'])
        self.assertEqual(self.docs.asset(doc, 'b', download=True), before)
        state = self.docs.update(doc, {**self.update(state), 'cursor': 4})
        self.ready(doc, state['revision'])
        with Image.open(BytesIO(self.docs.asset(doc, 'b', download=True))) as image:
            np.testing.assert_array_equal(np.asarray(image)[12, 18, :3], self.rgb[12, 18])

    def test_sam_preview_rejects_stale_revision_after_inference(self):
        def predict(positive, negative):
            self.entered.set()
            self.release.wait(5)
            mask = np.zeros(self.rgb.shape[:2], bool)
            mask[12, 12] = True
            return mask

        self.predict_object = predict
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        with ThreadPoolExecutor(max_workers=1) as pool:
            preview = pool.submit(self.docs.preview_object, doc, state['revision'], [[12, 12]], [])
            try:
                self.assertTrue(self.entered.wait(3))
                updated = self.docs.update(doc, self.update(state, strength=.5))
            finally:
                self.release.set()
            with self.assertRaises(RevisionConflict):
                preview.result(timeout=3)
        self.ready(doc, updated['revision'])

    def test_magnetic_source_reuses_unmasked_image_across_brush_revisions(self):
        from gpt_animeimage_denoise.workbench import documents as docs

        opened = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = opened['id']
        initial = self.ready(doc, opened['revision'])
        with patch.object(docs, '_png', wraps=docs._png) as png:
            source = self.docs.asset(doc, 'magnetic_source', revision=initial['revision'])
            self.assertEqual(self.docs.asset(doc, 'magnetic_source', revision=initial['revision']), source)
            self.assertEqual(png.call_count, 1)
        with Image.open(BytesIO(source)) as image:
            self.assertEqual(image.info['icc_profile'], b'fixture-icc')
            np.testing.assert_array_equal(np.asarray(image)[..., 3], self.alpha)
            np.testing.assert_array_equal(np.asarray(image)[..., :3], 71)

        brush = {'tool': 'exclude', 'points': [[8, 8]], 'radius': 2}
        edited = self.docs.update(doc, {**self.update(initial, modules=[]), 'edits': [brush], 'cursor': 1})
        current = self.ready(doc, edited['revision'])
        self.assertEqual(self.docs.asset(doc, 'magnetic_source', revision=current['revision']), source)
        with self.assertRaises(RevisionConflict):
            self.docs.asset(doc, 'magnetic_source', revision=initial['revision'])
        self.assertEqual(self.strengths, [.0005])

        changed = self.docs.update(doc, self.update(current, restoration_gain=0))
        current = self.ready(doc, changed['revision'])
        new_source = self.docs.asset(doc, 'magnetic_source', revision=current['revision'])
        with Image.open(BytesIO(new_source)) as image:
            np.testing.assert_array_equal(np.asarray(image)[..., :3], 51)

    def test_magnetic_source_recovers_candidate_for_reopened_document(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        original = self.docs.asset(doc, 'magnetic_source', revision=state['revision'])
        self.docs.close()
        with Documents(self.workspace, self.open_session) as reopened:
            recovered = reopened.asset(doc, 'magnetic_source', revision=state['revision'])
            self.assertEqual(reopened.get(doc), state)
        self.assertEqual(recovered, original)
        self.assertEqual(self.strengths, [.0005, .0005])

    def test_magnetic_source_rejects_revision_changed_during_png_encoding(self):
        from gpt_animeimage_denoise.workbench import documents as docs

        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        encode = docs._png

        def gated_png(*args):
            if current_thread().name.startswith('magnetic-test'):
                self.entered.set()
                self.assertTrue(self.release.wait(5))
            return encode(*args)

        with patch.object(docs, '_png', side_effect=gated_png), \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix='magnetic-test') as pool:
            pending = pool.submit(self.docs.asset, doc, 'magnetic_source', revision=state['revision'])
            try:
                self.assertTrue(self.entered.wait(3))
                updated = self.docs.update(doc, self.update(state, strength=.5))
                self.ready(doc, updated['revision'])
            finally:
                self.release.set()
            with self.assertRaises(RevisionConflict):
                pending.result(timeout=3)

    def test_suspected_objects_overlay_is_diagnostic_and_reused_after_brush(self):
        self.suspected_objects = np.zeros(self.rgb.shape[:2], bool)
        self.suspected_objects[10:15, 12:17] = True
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        self.assertTrue(state['b']['images']['suspected_objects'])
        overlay = self.docs.asset(doc, 'suspected_objects', revision=state['revision'])
        with Image.open(BytesIO(overlay)) as image:
            self.assertEqual(image.info['icc_profile'], b'fixture-icc')
            pixels = np.asarray(image)
            np.testing.assert_array_equal(pixels[..., 3], self.alpha)
            np.testing.assert_array_equal(pixels[~self.suspected_objects, :3],
                                          self.rgb[~self.suspected_objects])
            self.assertFalse(np.array_equal(pixels[12, 14, :3], self.rgb[12, 14]))

        change = {**self.update(state, modules=[]),
                  'edits': [{'tool': 'include', 'points': [[14, 12]], 'radius': 2}], 'cursor': 1}
        state = self.docs.update(doc, change)
        state = self.ready(doc, state['revision'])
        self.assertEqual(state['edit_feedback']['changed_pixels'], 13)
        self.assertEqual(self.docs.asset(doc, 'suspected_objects', revision=state['revision']), overlay)
        with self.assertRaises(RevisionConflict):
            self.docs.asset(doc, 'suspected_objects', revision=state['revision'] - 1)

    def test_unknown_overlay_is_independent_revision_safe_and_reusable(self):
        from gpt_animeimage_denoise.workbench import documents as docs

        self.unknown[8:16, 10:17] = True
        self.unknown[8, 0] = True  # Fully transparent source pixel.
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        before = self.ready(doc, state['revision'])
        self.assertTrue(before['b']['images']['unknown'])
        overlay = self.docs.asset(doc, 'unknown', revision=before['revision'])
        with Image.open(BytesIO(overlay)) as image:
            pixels = np.asarray(image)
            self.assertEqual(image.mode, 'RGBA')
            np.testing.assert_array_equal(pixels[..., 3] != 0, self.unknown & (self.alpha > 0))
            colors = np.unique(pixels[..., :3][pixels[..., 3] != 0], axis=0)
            self.assertEqual(len(colors), 2)
            np.testing.assert_array_equal(colors[:, 0], colors[:, 2])
        with Image.open(BytesIO(self.docs.asset(doc, 'parts', revision=before['revision']))) as image:
            np.testing.assert_array_equal(np.asarray(image)[..., :3][self.unknown], self.rgb[self.unknown])
        # Paint only outside the unknown mask. Reuse its previous PNG.
        with patch.object(docs, '_unknown_overlay', wraps=docs._unknown_overlay) as encode:
            edited = self.docs.update(doc, {**self.update(before),
                'edits': [{'tool': 'exclude', 'points': [[8, 6]], 'radius': 1}], 'cursor': 1})
            current = self.ready(doc, edited['revision'])
            encode.assert_not_called()
        self.assertEqual(self.docs.asset(doc, 'unknown', revision=current['revision']), overlay)
        with self.assertRaises(RevisionConflict):
            self.docs.asset(doc, 'unknown', revision=before['revision'])

    def test_resolve_unknown_requires_acknowledged_adjacent_brush_and_replays_on_undo(self):
        self.unknown[12, 14] = True
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        initial = self.ready(doc, state['revision'])
        brush = {'tool': 'include', 'points': [[14, 12]], 'radius': 2}
        resolve = {'tool': 'resolve_unknown', 'module': 'skin'}
        with self.assertRaises(RevisionConflict):
            self.docs.update(doc, {**self.update(initial), 'edits': [brush, resolve], 'cursor': 2})
        painted = self.docs.update(doc, {**self.update(initial), 'edits': [brush], 'cursor': 1})
        painted = self.ready(doc, painted['revision'])
        self.assertEqual(painted['edit_feedback']['skipped_unknown_pixels'], 1)
        for invalid in (dict(resolve, points=[[20, 20]]), dict(resolve, module='background')):
            with self.assertRaises(ValueError):
                self.docs.update(doc, {**self.update(painted), 'edits': [brush, invalid], 'cursor': 2})
        with self.assertRaises(RevisionConflict):
            changed = dict(brush, points=[[15, 12]])
            self.docs.update(doc, {**self.update(painted), 'edits': [changed, resolve], 'cursor': 2})
        fixed = self.docs.update(doc, {**self.update(painted), 'edits': [brush, resolve], 'cursor': 2})
        fixed = self.ready(doc, fixed['revision'])
        self.assertFalse(fixed['b'].get('images', {}).get('unknown'))
        with self.assertRaises(KeyError):
            self.docs.asset(doc, 'unknown', revision=fixed['revision'])
        undo = self.docs.update(doc, {**self.update(fixed), 'cursor': 1})
        undo = self.ready(doc, undo['revision'])
        self.assertTrue(undo['b']['images']['unknown'])
        redo = self.docs.update(doc, {**self.update(undo), 'cursor': 2})
        redo = self.ready(doc, redo['revision'])
        self.assertFalse(redo['b'].get('images', {}).get('unknown'))

    def test_old_analysis_refreshes_on_access_with_pinned_a_and_edits_intact(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        self.docs.pin(doc, state['revision'])
        pinned = self.docs.asset(doc, 'a', download=True)
        edits = [{'tool': 'exclude', 'points': [[14, 12]], 'radius': 2}]
        change = {**self.update(state, strength=.4), 'edits': edits, 'cursor': 1}
        state = self.docs.update(doc, change)
        previous = self.ready(doc, state['revision'])
        self.docs.close()

        path = self.workspace / 'documents' / doc / 'draft.json'
        draft = json.loads(path.read_text(encoding='utf-8'))
        draft.pop('_analysis_version', None)
        path.write_text(json.dumps(draft), encoding='utf-8')
        old_render = path.parent / f'render-{previous["revision"]}'
        for filename in ('candidate.png', 'labels.png'):
            (old_render / filename).unlink()

        def waiting_opener(source, workspace):
            self.release.wait(5)
            return self.open_session(source, workspace)

        with Documents(self.workspace, waiting_opener) as recovered:
            try:
                before_thumbnail = path.read_bytes()
                with Image.open(BytesIO(recovered.thumbnail(doc))) as thumbnail:
                    self.assertEqual(thumbnail.size, (previous['width'], previous['height']))
                self.assertEqual(path.read_bytes(), before_thumbnail)
                pending = recovered.get(doc)
                self.assertEqual(pending['revision'], previous['revision'] + 1)
                self.assertEqual(pending['selection_revision'], previous['selection_revision'] + 1)
                self.assertEqual(pending['edits'], edits)
                self.assertEqual(pending['cursor'], 1)
                self.assertEqual(pending['settings'], previous['settings'])
                self.assertEqual(recovered.asset(doc, 'a', download=True), pinned)
                with self.assertRaises(RevisionConflict):
                    recovered.asset(doc, 'candidate', revision=previous['revision'])
                with self.assertRaises(RevisionConflict):
                    recovered.asset(doc, 'b', download=True)
            finally:
                self.release.set()
            current = self.ready(doc, pending['revision'], docs=recovered)
            self.assertEqual(current['a'], previous['a'])
            self.assertEqual(current['b']['revision'], pending['revision'])
            self.assertEqual(recovered.get(doc)['revision'], pending['revision'])
            for view in ('candidate', 'labels'):
                with Image.open(BytesIO(recovered.asset(doc, view, revision=current['revision']))) as image:
                    self.assertEqual(image.size, (previous['width'], previous['height']))

    def test_version_three_refresh_replays_unknown_edits_without_unlocking_objects(self):
        self.unknown[12, 14:16] = True
        self.suspected_objects = self.unknown.copy()
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        initial = self.ready(doc, state['revision'])
        brush = {'tool': 'include', 'points': [[14, 12]], 'radius': 2}
        painted = self.docs.update(doc, {**self.update(initial), 'edits': [brush], 'cursor': 1})
        painted = self.ready(doc, painted['revision'])
        self.assertEqual(painted['edit_feedback']['skipped_unknown_pixels'], 2)
        edits = [brush, {'tool': 'resolve_unknown', 'module': 'skin'}]
        fixed = self.docs.update(doc, {**self.update(painted), 'edits': edits, 'cursor': 2})
        fixed = self.ready(doc, fixed['revision'])
        self.docs.pin(doc, fixed['revision'])
        pinned = self.docs.asset(doc, 'a', download=True)
        self.docs.close()

        folder = self.workspace / 'documents' / doc
        path = folder / 'draft.json'
        draft = json.loads(path.read_text(encoding='utf-8'))
        draft['_analysis_version'] = 3
        path.write_text(json.dumps(draft), encoding='utf-8')
        previous_overlay = folder / 'suspected-objects-3.png'
        latest_overlay = next(folder.glob('suspected-objects-*.png'))
        if latest_overlay != previous_overlay:
            latest_overlay.rename(previous_overlay)
        self.unknown[12, 14] = False
        self.objects[12, 14] = True
        self.suspected_objects[12, 14] = False

        def waiting_opener(source, workspace):
            self.release.wait(5)
            return self.open_session(source, workspace)

        with Documents(self.workspace, waiting_opener) as recovered:
            try:
                queued = recovered.get(doc)
                self.assertEqual(queued['revision'], fixed['revision'] + 1)
                self.assertEqual(queued['edits'], edits)
                self.assertEqual(queued['cursor'], 2)
                self.assertEqual(recovered.asset(doc, 'a', download=True), pinned)
                self.assertEqual(recovered.asset(doc, 'suspected_objects', revision=fixed['revision']),
                                 previous_overlay.read_bytes())
            finally:
                self.release.set()
            upgraded = self.ready(doc, queued['revision'], docs=recovered)
            self.assertFalse(upgraded['b'].get('images', {}).get('unknown'))
            with Image.open(BytesIO(recovered.asset(doc, 'b', download=True))) as image:
                pixels = np.asarray(image)
                np.testing.assert_array_equal(pixels[12, 14, :3], self.rgb[12, 14])
                self.assertFalse(np.array_equal(pixels[12, 15, :3], self.rgb[12, 15]))
            self.assertEqual(recovered.asset(doc, 'a', download=True), pinned)

    def ready(self, doc, revision, docs=None):
        docs = docs or self.docs
        deadline = monotonic() + 8
        while monotonic() < deadline:
            state = docs.get(doc)
            if state['status'] == 'error':
                self.fail(state['error'])
            if state['b'] and state['b']['revision'] == revision:
                return state
            sleep(.01)
        self.fail(f'Revision {revision} not ready: {state}')

    @staticmethod
    def update(state, **settings):
        return {'revision': state['revision'], 'settings': {**state['settings'], **settings},
                'edits': state['edits'], 'cursor': state['cursor']}

    def test_remove_part_edit_is_saved_with_its_source_part(self):
        self.unknown[10, 11] = True
        self.objects[10, 12] = True
        opened = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = opened['id']
        state = self.ready(doc, opened['revision'])
        edit = {'stage': 'parts', 'tool': 'remove_part', 'module': 'skin',
                'shape': 'brush', 'points': [[10, 10]], 'radius': 3, 'feather': 0}
        for invalid_source in ('unknown', 'missing'):
            with self.assertRaises(ValueError):
                self.docs.update(doc, {**self.update(state), 'edits': [{**edit, 'module': invalid_source}], 'cursor': 1})
        queued = self.docs.update(doc, {**self.update(state), 'edits': [edit], 'cursor': 1})
        ready = self.ready(doc, queued['revision'])
        self.assertEqual(ready['edits'], [edit])
        with Image.open(BytesIO(self.docs.asset(doc, 'labels', revision=ready['revision']))) as image:
            self.assertEqual([image.getpixel((x, 10)) for x in (9, 11, 12, 14)], [255, 255, 7, 1])

    def test_original_is_available_during_analysis_and_stale_result_is_not_exported(self):
        state = self.docs.open(self.content.getvalue(), '测试.png')
        doc = state['id']
        with Image.open(BytesIO(self.docs.asset(doc, 'original'))) as image:
            np.testing.assert_array_equal(np.array(image), np.dstack((self.rgb, self.alpha)))
        state = self.ready(doc, state['revision'])
        published = state['revision']
        self.docs.pin(doc, published)
        self.docs.change_preferences({'export_dir': str(self.workspace)})
        old = self.docs.asset(doc, 'b', download=True)
        state = self.docs.update(doc, self.update(state, l0=.001))
        self.assertTrue(self.entered.wait(3))
        next_state = self.docs.update(doc, self.update(state, l0=.002))
        latest = self.docs.update(doc, self.update(next_state, l0=.003))
        self.assertEqual(self.docs.asset(doc, 'b'), old)
        for view in ('parts', 'weight', 'candidate', 'b'):
            with Image.open(BytesIO(self.docs.thumbnail(doc, view, published))) as node:
                self.assertEqual(node.size, (28, 24))
        for view in ('candidate', 'labels', 'highlight'):
            with Image.open(BytesIO(self.docs.asset(doc, view, revision=published, module='hair'))) as image:
                self.assertEqual(image.size, (28, 24))
        self.assertEqual(self.docs.export(doc, 'a', published).read_bytes(), old)
        for revision in (published, latest['revision']):
            with self.assertRaises(RevisionConflict):
                self.docs.pin(doc, revision)
            with self.assertRaises(RevisionConflict):
                self.docs.export(doc, 'b', revision)
        with self.assertRaises(RevisionConflict):
            self.docs.asset(doc, 'b', download=True)
        with self.assertRaises(RevisionConflict):
            self.docs.update(doc, self.update(state, strength=.7))
        self.release.set()
        self.ready(doc, latest['revision'])
        for view in ('parts', 'weight', 'candidate', 'b'):
            with self.assertRaises(RevisionConflict):
                self.docs.thumbnail(doc, view, published)
        for view in ('candidate', 'labels', 'highlight'):
            with self.assertRaises(RevisionConflict):
                self.docs.asset(doc, view, revision=published, module='hair')
        self.assertEqual(len(self.opened), 1)
        self.assertEqual(self.strengths, [.0005, .001, .003])

    def test_pin_and_edit_history_survive_restart(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        self.docs.pin(doc, state['revision'])
        pinned = self.docs.asset(doc, 'a', download=True)
        change = self.update(state, strength=.4)
        change['edits'] = [{'tool': 'protect', 'shape': 'brush', 'points': [[5, 6]], 'strength': .9}]
        change['cursor'] = 1
        state = self.docs.update(doc, change)
        self.ready(doc, state['revision'])
        self.docs.close()
        with Documents(self.workspace, self.open_session) as recovered:
            restored = recovered.get(doc)
            self.assertEqual(restored['cursor'], 1)
            self.assertEqual(restored['edits'], change['edits'])
            self.assertEqual(restored['settings']['strength'], .4)
            self.assertEqual(recovered.asset(doc, 'a', download=True), pinned)
            self.assertNotEqual(recovered.asset(doc, 'b', download=True), pinned)
            self.assertEqual([item['id'] for item in recovered.recent()], [doc])

    def test_direct_exclusion_is_saved_and_undoable(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        baseline = self.docs.asset(doc, 'b', download=True)
        change = self.update(state)
        change.update(edits=[{'tool': 'exclude', 'points': [[15, 12]], 'radius': 2}], cursor=1)
        state = self.docs.update(doc, change)
        state = self.ready(doc, state['revision'])
        self.assertEqual(state['edit_feedback'], {'tool': 'exclude', 'changed_pixels': 13, 'reason': 'changed'})
        with Image.open(BytesIO(self.docs.asset(doc, 'b', download=True))) as image:
            np.testing.assert_array_equal(np.array(image)[12, 15, :3], self.rgb[12, 15])
        state = self.docs.update(doc, {**self.update(state), 'cursor': 0})
        self.ready(doc, state['revision'])
        self.assertEqual(self.docs.asset(doc, 'b', download=True), baseline)
        self.assertEqual(self.strengths, [.0005])

    def test_warm_blend_updates_the_changed_area_and_reuses_an_identical_result(self):
        from gpt_animeimage_denoise import filters

        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        exclude = {'stage': 'range', 'tool': 'exclude', 'points': [[14, 12]], 'radius': 2}
        reassign = {'stage': 'parts', 'tool': 'classify', 'module': 'hair', 'points': [[17, 18]], 'radius': 2}
        full_blend = filters.blend
        with patch.object(filters, 'blend', wraps=full_blend) as blend:
            queued = self.docs.update(doc, {**self.update(state), 'edits': [exclude], 'cursor': 1})
            ready = self.ready(doc, queued['revision'])
            self.assertEqual(blend.call_count, 1)
            self.assertLess(blend.call_args.args[0].size, self.rgb.size)
            weight = self.docs._selection_value[0].weight
            expected = full_blend(self.rgb, self.docs._restored, weight)
            changed = self.docs.asset(doc, 'b', download=True)
            with Image.open(BytesIO(changed)) as image:
                np.testing.assert_array_equal(np.asarray(image)[..., :3], expected)
                np.testing.assert_array_equal(np.asarray(image)[..., 3], self.alpha)
                self.assertEqual(image.info['icc_profile'], b'fixture-icc')

            queued = self.docs.update(doc, {**self.update(ready), 'edits': [exclude, reassign], 'cursor': 2})
            ready = self.ready(doc, queued['revision'])
            self.assertGreater(ready['edit_feedback']['changed_pixels'], 0)
            self.assertEqual(blend.call_count, 1)
            self.assertEqual(self.docs.asset(doc, 'b', download=True), changed)

    def test_first_reopened_selection_is_available_during_candidate_restoration(self):
        from gpt_animeimage_denoise.workbench import documents as module

        opened = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = opened['id']
        original = self.ready(doc, opened['revision'])
        self.docs.close()
        restoring, release = Event(), Event()
        restore = module._restore_candidate

        def hold_restoration(*args):
            restoring.set()
            if not release.wait(5):
                raise RuntimeError('The test did not release candidate restoration')
            return restore(*args)

        with Documents(self.workspace, self.open_session) as reopened, \
                patch.object(module, '_restore_candidate', side_effect=hold_restoration):
            try:
                edit = {'stage': 'range', 'tool': 'exclude', 'points': [[14, 12]], 'radius': 2}
                queued = reopened.update(doc, {**self.update(original), 'edits': [edit], 'cursor': 1})
                self.assertTrue(restoring.wait(3))
                during = reopened.get(doc)
                self.assertEqual(during['selection_preview']['revision'], queued['revision'])
                self.assertEqual(during['b']['revision'], original['revision'])
                with Image.open(BytesIO(reopened.asset(doc, 'weight_overlay', revision=queued['revision']))) as image:
                    self.assertEqual(image.convert('RGBA').getpixel((14, 12))[3], 0)
            finally:
                release.set()
            self.ready(doc, queued['revision'], docs=reopened)

    def test_publishes_each_selection_while_previous_result_encoding_is_blocked(self):
        from gpt_animeimage_denoise.workbench import documents as module

        source = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = source['id']
        initial = self.ready(doc, source['revision'])
        self.docs.pin(doc, initial['revision'])
        original_b = self.docs.asset(doc, 'b', revision=0)
        started = [Event(), Event()]
        release = [Event(), Event()]
        encode = module._png

        def hold_result(rgb, alpha, icc):
            if rgb.shape == self.rgb.shape:
                gate = 0 if not started[0].is_set() else 1
                started[gate].set()
                if not release[gate].wait(6):
                    raise RuntimeError('The test did not release PNG encoding')
            return encode(rgb, alpha, icc)

        def exclude(x):
            return {'stage': 'range', 'tool': 'exclude', 'points': [[x, 12]], 'radius': 2}

        def early(revision, x):
            limit = monotonic() + 3
            while monotonic() < limit:
                current = self.docs.get(doc)
                if (current.get('selection_preview') or {}).get('revision') == revision:
                    break
                sleep(.01)
            else:
                self.fail(f'Selection preview {revision} was not published during encoding')
            self.assertEqual(current['b']['revision'], 0)
            self.assertEqual(current['selection_preview']['cursor'], revision)
            self.assertNotEqual(current['status'], 'ready')
            with Image.open(BytesIO(self.docs.asset(doc, 'weight_overlay', revision=revision))) as image:
                self.assertEqual(image.convert('RGBA').getpixel((x, 12))[3], 0)
            with Image.open(BytesIO(self.docs.asset(doc, 'parts_overlay', revision=revision))) as image:
                self.assertEqual(image.size, (28, 24))
            with self.assertRaises(RevisionConflict):
                self.docs.pin(doc, revision)
            with self.assertRaises(RevisionConflict):
                self.docs.asset(doc, 'b', revision=revision, download=True)
            with self.assertRaises(RevisionConflict):
                self.docs.asset(doc, 'b', revision=0, download=True)
            return current

        with patch.object(module, '_png', side_effect=hold_result):
            try:
                first = self.docs.update(doc, {**self.update(initial), 'edits': [exclude(10)], 'cursor': 1})
                self.assertTrue(started[0].wait(3))
                first = early(first['revision'], 10)
                second = self.docs.update(doc, {**self.update(first), 'edits': [exclude(10), exclude(18)], 'cursor': 2})
                second = early(second['revision'], 18)
                with self.assertRaises(RevisionConflict):
                    self.docs.asset(doc, 'weight_overlay', revision=first['revision'])
                limit = monotonic() + 3
                while self.docs.get(doc)['phase'] != 'encoding' and monotonic() < limit:
                    sleep(.01)
                self.assertEqual(self.docs.get(doc)['phase'], 'encoding')
                third = self.docs.update(doc, {**self.update(second), 'edits': [exclude(10), exclude(18), exclude(23)],
                                               'cursor': 3})
                third = early(third['revision'], 23)
                release[0].set()
                self.assertTrue(started[1].wait(3))
                self.assertEqual(self.docs.get(doc)['b']['revision'], 0)
                release[1].set()
                newest = self.ready(doc, third['revision'])
            finally:
                for event in release:
                    event.set()
        self.assertEqual(newest['b']['revision'], 3)
        self.assertEqual(self.docs.asset(doc, 'a', revision=0, download=True), original_b)
        with Image.open(BytesIO(self.docs.asset(doc, 'b', revision=3, download=True))) as result:
            np.testing.assert_array_equal(np.asarray(result)[12, [10, 18, 23], :3], self.rgb[12, [10, 18, 23]])

    def test_absolute_mix_is_saved_as_an_edit_and_refuses_binary_export(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        edits = [{'tool': 'include', 'points': [[15, 12]], 'allow_protected': True},
                 {'tool': 'mix', 'points': [[15, 12]], 'radius': 2, 'strength': .6}]
        change = {**self.update(state, strength=.2), 'edits': edits, 'cursor': 2}
        state = self.docs.update(doc, change)
        state = self.ready(doc, state['revision'])
        self.assertEqual(state['edits'], edits)
        with self.assertRaisesRegex(ValueError, '校正'):
            self.docs.corrections(doc, state['revision'])
        invalid = {**edits[0], 'allow_protected': 'true'}
        with self.assertRaises(ValueError):
            self.docs.update(doc, {**self.update(state), 'edits': [invalid], 'cursor': 1})

    def test_disabled_cugan_draft_reopens_with_l0_and_keeps_edits(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        with self.assertRaisesRegex(ValueError, 'CUGAN'):
            self.docs.update(doc, self.update(state, method='cugan60_l0'))
        self.docs.close()

        draft_path = self.workspace / 'documents' / doc / 'draft.json'
        draft = json.loads(draft_path.read_text(encoding='utf-8'))
        edits = [{'tool': 'protect', 'points': [[6, 6]], 'radius': 3, 'strength': 1}]
        draft.update(edits=edits, cursor=1)
        draft['settings'].update(method='cugan-l0', strength=.6)
        draft_path.write_text(json.dumps(draft), encoding='utf-8')
        with Documents(self.workspace, self.open_session) as recovered:
            restored = self.ready(doc, state['revision'] + 1, docs=recovered)
            self.assertEqual(restored['settings']['method'], 'l0')
            self.assertEqual(restored['settings']['strength'], .6)
            self.assertEqual(restored['edits'], edits)
            self.assertEqual(restored['cursor'], 1)

    def test_binary_import_and_durable_export_are_source_bound(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        self.ready(doc, state['revision'])
        stream = BytesIO()
        protected = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
        protected[9:15, 8:14] = 255
        Image.fromarray(protected).save(stream, 'PNG')
        files = {'mask.png': stream.getvalue()}
        record = {'version': 2, 'source_sha256': doc, 'size': [28, 24], 'protect_mask': 'mask.png'}
        with self.assertRaises(ValueError):
            self.docs.import_correction(doc, state['revision'], {**record, 'source_sha256': 'other'}, files)
        updated = self.docs.import_correction(doc, state['revision'], record, files)
        state = self.ready(doc, updated['revision'])
        with Image.open(BytesIO(self.docs.asset(doc, 'b', download=True))) as image:
            np.testing.assert_array_equal(np.array(image)[9:15, 8:14],
                                          np.dstack((self.rgb, self.alpha))[9:15, 8:14])
        saved = self.docs.corrections(doc, state['revision'])
        later = self.docs.update(doc, self.update(state, strength=.3))
        self.ready(doc, later['revision'])
        self.assertTrue(saved.is_file())

    def test_import_can_be_undone_and_redone_without_losing_previous_edits(self):
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        change = self.update(state)
        change.update(edits=[{'tool': 'protect', 'points': [[6, 6]], 'radius': 3, 'strength': 1}], cursor=1)
        state = self.docs.update(doc, change)
        state = self.ready(doc, state['revision'])
        before = self.docs.asset(doc, 'b', download=True)
        png = BytesIO()
        region = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
        region[12:18, 12:18] = 255
        Image.fromarray(region).save(png, 'PNG')
        record = {'version': 2, 'source_sha256': doc, 'size': [28, 24], 'protect_mask': 'region.png'}
        state = self.docs.import_correction(doc, state['revision'], record, {'region.png': png.getvalue()})
        state = self.ready(doc, state['revision'])
        imported = self.docs.asset(doc, 'b', download=True)
        self.assertNotEqual(imported, before)
        self.assertEqual(state['edits'][0], change['edits'][0])
        self.assertEqual(state['cursor'], 2)
        self.assertEqual(state['edits'][1]['tool'], 'import')
        invalid = self.update(state)
        invalid['edits'] = [state['edits'][0], {'tool': 'import', 'token': 'correction-' + '0' * 32}]
        with self.assertRaises(ValueError):
            self.docs.update(doc, invalid)
        state = self.docs.update(doc, {**self.update(state), 'cursor': 1})
        state = self.ready(doc, state['revision'])
        self.assertEqual(self.docs.asset(doc, 'b', download=True), before)
        state = self.docs.update(doc, {**self.update(state), 'cursor': 2})
        self.ready(doc, state['revision'])
        self.assertEqual(self.docs.asset(doc, 'b', download=True), imported)

    def test_reopen_repairs_missing_or_corrupt_source_and_reloads_resident(self):
        content = self.content.getvalue()
        state = self.docs.open(content, 'sample.png')
        doc = state['id']
        state = self.ready(doc, state['revision'])
        source = self.opened[0]
        for damage in (lambda: source.unlink(), lambda: source.write_bytes(b'not the source')):
            damage()
            reopened = self.docs.open(content, 'sample.png')
            self.assertGreater(reopened['revision'], state['revision'])
            self.assertEqual(source.read_bytes(), content)
            state = self.ready(doc, reopened['revision'])
        self.assertEqual(self.opened, [source, source, source])

    def test_reopen_retries_failed_document_even_when_source_is_intact(self):
        attempts = []

        def flaky_opener(source, workspace):
            attempts.append(source)
            if len(attempts) == 1:
                raise RuntimeError('Temporary analysis failure')
            return self.open_session(source, workspace)

        self.docs.opener = flaky_opener
        state = self.docs.open(self.content.getvalue(), 'sample.png')
        doc = state['id']
        deadline = monotonic() + 5
        while state['status'] != 'error' and monotonic() < deadline:
            sleep(.01)
            state = self.docs.get(doc)
        self.assertEqual(state['error'], 'Temporary analysis failure')
        reopened = self.docs.open(self.content.getvalue(), 'sample.png')
        self.assertEqual(reopened['revision'], state['revision'] + 1)
        self.ready(doc, reopened['revision'])
        self.assertEqual(attempts, [attempts[0], attempts[0]])
