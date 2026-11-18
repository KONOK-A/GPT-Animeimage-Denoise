import json
import os
from pathlib import Path
import tempfile
from time import monotonic, sleep
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from gpt_animeimage_denoise import fusion_pipeline
from gpt_animeimage_denoise.corrections import load_edits
from gpt_animeimage_denoise.pipeline import _sha
from gpt_animeimage_denoise.workbench.documents import Documents
import test_workbench_documents
from test_fusion_pipeline import Models, regions


class WorkbenchSessionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        settings = mock.patch.dict(os.environ, GAID_WORKBENCH_SETTINGS=str(self.root / 'preferences.json'))
        settings.start()
        self.addCleanup(settings.stop)
        self.source = self.root / 'input.png'
        self.rgb = np.full((32, 40, 3), 96, np.uint8)
        self.alpha = np.full((32, 40), 255, np.uint8)
        self.alpha[-1] = 0
        Image.fromarray(np.dstack((self.rgb, self.alpha))).save(self.source, icc_profile=b'fixture-icc')
        self.baseline = np.full(self.rgb.shape, .400123, np.float32)
        self.models = Models()
        self.models.release = mock.Mock()

    def open_session(self, workspace=None, **options):
        from gpt_animeimage_denoise.workbench.session import WorkbenchSession

        with mock.patch('gpt_animeimage_denoise.fusion_models.FusionModels', return_value=self.models), \
             mock.patch.object(fusion_pipeline, 'fuse_evidence', side_effect=regions), \
             mock.patch.object(fusion_pipeline, 'refine_regions', side_effect=lambda rgb, ev, reg, models, **_: reg), \
             mock.patch.object(fusion_pipeline, 'smooth_candidate', return_value=self.baseline):
            return WorkbenchSession.open(self.source, workspace or self.root / 'work', device='cpu', **options)

    def test_workbench_resumes_a_legacy_case_with_no_completed_report(self):
        legacy = self.open_session(self.root / 'cases')
        (legacy.case_dir / 'report.json').unlink()
        notes = legacy.case_dir / 'notes.txt'
        notes.write_text('keep', encoding='utf-8')
        opener = mock.Mock(return_value=legacy)
        with Documents(self.root, opener) as docs:
            opened = docs.open(self.source.read_bytes(), 'input.png')
            doc, revision = opened['id'], opened['revision']
            test_workbench_documents.DocumentsTest.ready(self, doc, revision, docs=docs)
            opener.assert_called_once()
            self.assertGreater(docs.cache_info()['bytes'], 0)
            self.assertEqual(docs.clear_cache()['bytes'], 0)
            self.assertEqual(notes.read_text(encoding='utf-8'), 'keep')
            self.assertTrue(docs.asset(doc, 'b', revision=revision, download=True).startswith(b'\x89PNG'))

    def test_first_denoiser_failure_identifies_the_candidate_stage(self):
        model_path = self.root / 'esrgan.pth'
        model_path.write_bytes(b'fixture with an invalid checksum')
        identity = {'method': 'esrgan', 'model': {'path': str(model_path), 'expected_sha256': '0' * 64}}
        with mock.patch('gpt_animeimage_denoise.fusion_models.FusionModels', return_value=self.models), \
             mock.patch.object(fusion_pipeline, 'describe_denoiser', return_value=identity), \
             Documents(self.root / 'workbench') as docs:
            docs.change_preferences({'defaults': {'method': 'esrgan'}})
            opened = docs.open(self.source.read_bytes(), 'input.png')
            deadline = monotonic() + 5
            while monotonic() < deadline:
                state = docs.get(opened['id'])
                if state['status'] == 'error':
                    break
                sleep(.01)
            self.assertEqual(state['status'], 'error')
            self.assertIn('Denoiser model checksum', state['error'])
            self.assertEqual(state['failed_phase'], 'candidate')

    def test_non_l0_default_generates_its_candidate_once_and_skips_l0(self):
        from gpt_animeimage_denoise.workbench import session as workbench_session

        sr = np.full(self.rgb.shape, 179, np.uint8)
        identity = {'method': 'esrgan', 'model': {'path': str(self.root / 'mock-model'), 'expected_sha256': '0' * 64}}
        with mock.patch.object(fusion_pipeline, 'describe_denoiser', return_value=identity), \
             mock.patch.object(fusion_pipeline, 'super_resolve', return_value=sr) as generate, \
             mock.patch.object(workbench_session, 'super_resolve', side_effect=AssertionError('duplicate SR')), \
             mock.patch.object(workbench_session, 'smooth_candidate', side_effect=AssertionError('unwanted L0')):
            session = self.open_session(method='esrgan', downscale=.75)
            report = json.loads((session.case_dir / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['parameters']['denoiser'], 'esrgan')
            self.assertFalse(report['parameters']['l0_enabled'])
            np.testing.assert_array_equal(session.candidate('esrgan', downscale=.75), sr.astype(np.float32) / 255)
            generate.assert_called_once()

    def test_magnetic_source_uses_verified_l0_cache_after_reopen(self):
        session = self.open_session(self.root / 'cases')
        with Documents(self.root, lambda *_: session) as docs:
            state = docs.open(self.source.read_bytes(), 'input.png')
            doc, revision = state['id'], state['revision']
            test_workbench_documents.DocumentsTest.ready(self, doc, revision, docs=docs)
            expected = docs.asset(doc, 'magnetic_source', revision=revision)

        unavailable = mock.Mock(side_effect=AssertionError('must not initialize models'))
        with Documents(self.root, unavailable) as reopened, reopened._inference:
            self.assertEqual(reopened.asset(doc, 'magnetic_source', revision=revision), expected)
        unavailable.assert_not_called()

        report_path = session.case_dir / 'report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report['parameters']['lambda'] = .001
        report_path.write_text(json.dumps(report), encoding='utf-8')
        fallback = mock.Mock(return_value=session)
        with Documents(self.root, fallback) as reopened:
            self.assertEqual(reopened.asset(doc, 'magnetic_source', revision=revision), expected)
        fallback.assert_called_once()

    def test_changes_replays_verified_grid_after_reopen_without_inference(self):
        session = self.open_session(self.root / 'cases')
        with Documents(self.root, lambda *_: session) as docs:
            state = docs.open(self.source.read_bytes(), 'input.png')
            doc = state['id']
            ready = test_workbench_documents.DocumentsTest.ready(self, doc, state['revision'], docs=docs)
            changed = docs.update(doc, {'revision': ready['revision'], 'settings': ready['settings'], 'cursor': 1,
                'edits': [{'tool': 'exclude', 'points': [[20, 15]], 'radius': 3}]})
            ready = test_workbench_documents.DocumentsTest.ready(self, doc, changed['revision'], docs=docs)
            expected = docs.changes(doc, ready['revision'])
            self.assertGreater(expected['counts']['removed'], 0)

        draft = self.root / 'documents' / doc / 'draft.json'
        previous = draft.read_bytes()
        unavailable = mock.Mock(side_effect=AssertionError('must not initialize models'))
        with Documents(self.root, unavailable) as reopened, reopened._inference:
            self.assertEqual(reopened.changes(doc, ready['revision']), expected)
        unavailable.assert_not_called()
        self.assertEqual(draft.read_bytes(), previous)

    def test_open_and_switch_algorithms_reuse_analysis_and_exact_candidate(self):
        from gpt_animeimage_denoise.workbench import session as workbench_session

        source_hash = _sha(self.source)
        session = self.open_session()
        self.assertEqual(session.source_sha256, source_hash)
        self.assertEqual((session.width, session.height), (40, 32))
        self.assertEqual(session.icc, b'fixture-icc')
        np.testing.assert_array_equal(session.editor.alpha, self.alpha)
        np.testing.assert_array_equal(session.candidate(), self.baseline)
        np.testing.assert_array_equal(session.candidate('l0_0.0005'), self.baseline)

        sr_rgb = np.full(self.rgb.shape, 120, np.uint8)
        with mock.patch.object(workbench_session, 'super_resolve', return_value=sr_rgb) as sr, \
             mock.patch.object(workbench_session, 'smooth_candidate', return_value=self.baseline) as l0:
            session.candidate('esrgan', downscale=.6)
            session.candidate('esrgan60_l0', lam=.0005, downscale=.6)
            session.candidate('esrgan-l0', lam=.0005, downscale=.6)
            session.candidate('esrgan-l0', lam=.001, downscale=.6)
            session.candidate('esrgan', downscale=.75)
            self.assertEqual(sr.call_count, 2)
            self.assertEqual(l0.call_count, 2)
            np.testing.assert_array_equal(l0.call_args.args[0], sr_rgb)
            np.testing.assert_array_equal(session.candidate(), self.baseline)

        self.assertEqual(len(self.models.calls), 1)
        self.models.release.assert_called()
        self.assertEqual(_sha(self.source), source_hash)

    def test_manual_object_candidate_obeys_points_and_keeps_analysis_immutable(self):
        session = self.open_session()
        shape = self.alpha.shape
        masks = np.zeros((2, *shape), bool)
        masks[0, 10:17, 10:22] = True
        masks[1, 10:17, 10:16] = True
        self.models.predict_candidates = mock.Mock(return_value={
            'masks': masks, 'scores': np.array([.99, .9]), 'stability': np.array([.99, .9])})
        original = session.editor.objects.copy()
        self.models.release.reset_mock()
        mask = session.predict_object([[12, 12]], [[20, 12]])
        np.testing.assert_array_equal(mask, masks[1])
        np.testing.assert_array_equal(session.editor.objects, original)
        self.assertEqual(self.models.predict_candidates.call_args.args[1],
                         {'positive': [[12, 12]], 'negative': [[20, 12]]})
        self.models.release.assert_called_once()
        with self.assertRaisesRegex(ValueError, '没有找到'):
            session.predict_object([[12, 12]], [[13, 12]])
        self.assertEqual(self.models.release.call_count, 2)

    def test_export_corrections_are_source_bound_and_do_not_overwrite(self):
        session = self.open_session()
        session.editor.paint_stroke([(20, 15)], 2, erase=True)
        modules = ('skin', 'outfit')
        output = self.root / 'correction'
        path = session.save_corrections(output, modules)
        edits = load_edits(path, session.editor.labels.shape, _sha(self.source))
        allowed, protected = session.editor.correction_masks(modules)
        np.testing.assert_array_equal(edits['allow_mask'], allowed)
        np.testing.assert_array_equal(edits['protect_mask'], protected)
        self.assertTrue(allowed.any())
        self.assertTrue(protected.any())
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            session.save_corrections(output, ('skin',))
        self.assertEqual(path.read_bytes(), before)

    def test_loads_verified_suspected_objects_separately_from_confirmed_objects(self):
        from gpt_animeimage_denoise.workbench.session import WorkbenchSession

        legacy = self.open_session()
        report = json.loads((legacy.case_dir / 'report.json').read_text(encoding='utf-8'))
        suspected = np.zeros(self.rgb.shape[:2], bool)
        suspected[15:20, 10:15] = True
        mask_path = legacy.case_dir / 'suspected_objects.png'
        Image.fromarray(suspected.astype(np.uint8) * 255).save(mask_path)
        report['output_hashes'][mask_path.name] = _sha(mask_path)

        with mock.patch('gpt_animeimage_denoise.fusion_models.FusionModels', return_value=self.models), \
             mock.patch.object(fusion_pipeline, 'process_image', return_value=report):
            session = WorkbenchSession.open(self.source, self.root / 'work', device='cpu')
            np.testing.assert_array_equal(session.suspected_objects, suspected)
            self.assertFalse((session.editor.objects & suspected).any())
            mask_path.write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                WorkbenchSession.open(self.source, self.root / 'work', device='cpu')
