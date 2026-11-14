from base64 import b64decode
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import json
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
        with patch.object(docs, '_png', wraps=docs._png) as png:
            reused = docs._selection_images(session, after, previous=(before, images))
        self.assertEqual(reused, exact)
        self.assertEqual(png.call_count, 1)

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
                    recovered.asset(doc, 'b', download=True)
            finally:
                self.release.set()
            current = self.ready(doc, pending['revision'], docs=recovered)
            self.assertEqual(current['a'], previous['a'])
            self.assertEqual(current['b']['revision'], pending['revision'])
            self.assertEqual(recovered.get(doc)['revision'], pending['revision'])

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

    def test_original_is_available_during_analysis_and_stale_result_is_not_exported(self):
        state = self.docs.open(self.content.getvalue(), '测试.png')
        doc = state['id']
        with Image.open(BytesIO(self.docs.asset(doc, 'original'))) as image:
            np.testing.assert_array_equal(np.array(image), np.dstack((self.rgb, self.alpha)))
        state = self.ready(doc, state['revision'])
        old = self.docs.asset(doc, 'b', download=True)
        state = self.docs.update(doc, self.update(state, l0=.001))
        self.assertTrue(self.entered.wait(3))
        next_state = self.docs.update(doc, self.update(state, l0=.002))
        latest = self.docs.update(doc, self.update(next_state, l0=.003))
        self.assertEqual(self.docs.asset(doc, 'b'), old)
        with self.assertRaises(RevisionConflict):
            self.docs.asset(doc, 'b', download=True)
        with self.assertRaises(RevisionConflict):
            self.docs.update(doc, self.update(state, strength=.7))
        self.release.set()
        self.ready(doc, latest['revision'])
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
