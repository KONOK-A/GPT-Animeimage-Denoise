import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageOps


class Models:
    signature = 'weights-precision-test'
    device = 'cpu'

    def __init__(self):
        self.calls = []; self.sam_calls = 0

    def segment_evidence(self, rgb, *, dual_view=True):
        self.calls.append((rgb.copy(), dual_view))
        if rgb[0, 0, 0] == 7:
            raise ValueError('intentional model error')
        h, w = rgb.shape[:2]
        return dict(alpha=np.ones((h, w), np.float32), see_logits=np.zeros((19, h, w), np.float32),
                    see_original_logits=np.zeros((19, h, w), np.float32) if dual_view else None,
                    m2f_white=np.zeros((12, 768, 768), np.float32),
                    m2f_original=np.zeros((12, 768, 768), np.float32) if dual_view else None,
                    reference_labels12=np.ones((h, w), np.uint8), provenance={'precision': 'exact-test'})

    def predict_candidates(self, rgb, prompt, *, mask_input=None):
        self.sam_calls += 1
        mask = np.zeros(rgb.shape[:2], bool); mask[12:20, 15:25] = True
        return dict(masks=np.stack([mask] * 3), scores=np.array([.99, .98, .97], np.float32),
                    stability=np.ones(3, np.float32), lowres=np.ones((3, 256, 256), np.float32))


class L0:
    signature = 'constant-float-test'

    def __init__(self):
        self.calls = 0

    def __call__(self, image, lam, *, device):
        self.calls += 1
        return np.full(image.shape, .350123, np.float32)


def regions(evidence, policy=None):
    shape = evidence['alpha'].shape
    labels = np.ones(shape, np.uint8); labels[:, :3] = 255; labels[:, 3:6] = 6
    objects = np.zeros(shape, bool); objects[5:10, 10:15] = True
    return dict(raw_labels=labels.copy(), person_labels=labels,
                allowed=np.ones(shape, bool), protected=np.zeros(shape, bool),
                objects=objects, uncertain=labels == 255, consensus=labels != 255,
                conflict=np.zeros(shape, bool), source_map=np.ones(shape, np.uint8),
                sam_records=[], policy=policy or {'version': 'test'},
                _trusted_cores=np.zeros((7, *shape), bool))


class FusionPipelineTest(unittest.TestCase):
    def test_grouping_rule_change_invalidates_result_fingerprint(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        original_sha=pipeline._sha
        marker=['a']
        def identity(path):
            return marker[0]*64 if Path(path).name=='grouping.py' else original_sha(path)
        with mock.patch.object(pipeline,'_sha',side_effect=identity):
            first=self.run_image()
            marker[0]='b'
            second=self.run_image()
        self.assertNotEqual(first['fingerprint'],second['fingerprint'])
        self.assertFalse(second['resumed'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.source = self.root / 'source.png'; self.output = self.root / 'out'
        rng = np.random.default_rng(4)
        self.rgb = rng.integers(110, 140, (32, 40, 3), dtype=np.uint8)
        Image.fromarray(self.rgb).save(self.source)
        self.models = Models(); self.l0 = L0()

    def tearDown(self):
        self.temp.cleanup()

    def run_image(self, **kwargs):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        with mock.patch.object(pipeline, 'fuse_evidence', side_effect=regions), \
             mock.patch.object(pipeline, 'reference_regions', side_effect=regions), \
             mock.patch.object(pipeline, 'refine_regions', side_effect=lambda rgb, ev, reg, models, policy=None: reg):
            return pipeline.process_image(self.source, self.output, self.models,
                                          case_id='case', l0_fn=self.l0, denoiser='l0', **kwargs)

    def edit(self, **values):
        path = self.root / 'edit.json'
        path.write_text(json.dumps(dict(version=2, source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
                                       size=[40, 32], **values)), encoding='utf-8')
        return path

    def test_protected_unknown_objects_and_nondefault_modules_are_unchanged(self):
        report = self.run_image()
        folder = self.output / 'case'
        result = np.asarray(Image.open(folder / 'result.png'))
        weight = np.load(folder / 'weight.npy')
        labels = np.asarray(Image.open(folder / 'person_labels.png'))
        self.assertFalse(weight[:, :6].any()); self.assertFalse(weight[5:10, 10:15].any())
        np.testing.assert_array_equal(result[weight == 0], self.rgb[weight == 0])
        self.assertTrue((labels[:, :3] == 255).all())
        for name in ['skin', 'face', 'hair', 'eyes', 'mouth', 'outfit']:
            self.assertTrue((folder / f'module_{name}.png').is_file())
        self.assertEqual(report['parameters']['profile'], 'fusion')
        self.assertEqual(report['review_status'], 'needs_review')
        self.assertEqual(report['quality_status'], 'pending_reference_review')
        self.assertEqual(report['provenance']['precision'], 'exact-test')
        self.assertFalse(list(self.output.rglob('*.html')))

    def test_strength_zero_and_rgba_icc_exactly_preserve_original(self):
        alpha = np.full((32, 40), 128, np.uint8); alpha[:, 10:12] = 0
        rgba = np.dstack([self.rgb, alpha]); profile = b'test-icc'
        Image.fromarray(rgba).save(self.source, icc_profile=profile)
        before = self.source.read_bytes()
        self.run_image(strength=0)
        with Image.open(self.output / 'case/result.png') as result:
            np.testing.assert_array_equal(np.asarray(result), rgba)
            self.assertEqual(result.info['icc_profile'], profile)
        self.assertEqual(before, self.source.read_bytes())
        self.assertFalse(np.load(self.output / 'case/weight.npy').any())

    def test_semitransparent_source_is_analyzed_on_white_and_retains_alpha(self):
        alpha = np.full(self.rgb.shape[:2], 128, np.uint8)
        Image.fromarray(np.dstack([self.rgb, alpha])).save(self.source)
        edit = self.edit(sam_regions=[{'action': 'protect', 'positive': [[.5, .5]]}])
        with mock.patch.object(self.models, 'predict_candidates', wraps=self.models.predict_candidates) as predict:
            self.run_image(edits_path=edit)

        white = (self.rgb.astype(np.int32) * 128 + 127 * 255 + 127) // 255
        np.testing.assert_array_equal(self.models.calls[0][0], white)
        np.testing.assert_array_equal(predict.call_args.args[0], white)
        with Image.open(self.output / 'case/result.png') as image:
            result = np.array(image)
        weight = np.load(self.output / 'case/weight.npy', allow_pickle=False)
        np.testing.assert_array_equal(result[..., 3], alpha)
        np.testing.assert_array_equal(result[weight == 0, :3], self.rgb[weight == 0])
        self.assertTrue(np.any(result[weight > 0, :3] != self.rgb[weight > 0]))

    def test_default_finish_reopens_line_pixels_and_keeps_scope_and_cached_inputs(self):
        self.rgb[:, :20] = 115
        self.rgb[:, 20:] = 150
        Image.fromarray(self.rgb).save(self.source)
        folder = self.output / 'case'
        default = self.run_image()
        default_weight = np.load(folder / 'weight.npy')
        default_result = np.asarray(Image.open(folder / 'result.png'))
        labels = (folder / 'person_labels.png').read_bytes()
        protection = (folder / 'protected.png').read_bytes()
        self.assertEqual(default['parameters']['structure_protection'], .2)
        self.assertEqual(default['parameters']['restoration_gain'], .25)
        self.assertEqual(default_weight[18, 19], np.float32(.8))
        self.assertEqual(default_result[18, 30, 0], round(.75 * .350123 * 255 + .25 * 150))

        old = self.run_image(structure_protection=1, restoration_gain=1)
        self.assertEqual(np.load(folder / 'weight.npy')[18, 19], 0)
        self.assertNotEqual(default['fingerprint'], old['fingerprint'])
        self.assertEqual(default['candidate_cache_identity'], old['candidate_cache_identity'])
        self.assertEqual((len(self.models.calls), self.l0.calls), (1, 1))
        self.assertEqual((folder / 'person_labels.png').read_bytes(), labels)
        self.assertEqual((folder / 'protected.png').read_bytes(), protection)
        self.assertTrue(self.run_image(structure_protection=1, restoration_gain=1)['resumed'])

        disabled = self.run_image(structure_protection=0)
        self.assertEqual(disabled['parameters']['structure_protection'], 0)
        self.assertEqual(np.load(folder / 'weight.npy')[18, 19], 1)

        reference = self.run_image(profile='reference')
        self.assertEqual(reference['parameters']['structure_protection'], 1)
        self.assertEqual(reference['parameters']['restoration_gain'], 1)

    def test_batch_forwards_explicit_finish_controls(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        manifest = {'items': [dict(id='case', path=str(self.source), status='eligible')]}
        with mock.patch.object(pipeline, 'process_image', return_value={'id': 'case', 'status': 'complete'}) as process:
            pipeline.process_batch(manifest, self.output, self.models,
                                   structure_protection=.5, restoration_gain=0)
        self.assertEqual(process.call_args.kwargs['structure_protection'], .5)
        self.assertEqual(process.call_args.kwargs['restoration_gain'], 0)

    def test_stages_reuse_float_l0_after_result_damage_sigma_strength_and_edits(self):
        first = self.run_image()
        self.assertTrue(self.run_image()['resumed'])
        (self.output / 'case/result.png').write_bytes(b'broken')
        repaired = self.run_image()
        self.assertFalse(repaired['resumed']); self.assertEqual(self.l0.calls, 1)
        self.run_image(strength=.5, sigma=8)
        self.assertEqual(len(self.models.calls), 1); self.assertEqual(self.l0.calls, 1)
        Image.new('L', (40, 32), 255).save(self.root / 'protect.png')
        edited = self.run_image(edits_path=self.edit(protect_mask='protect.png'))
        self.assertNotEqual(first['fingerprint'], edited['fingerprint'])
        self.assertEqual(len(self.models.calls), 1); self.assertEqual(self.l0.calls, 1)
        self.assertFalse(np.load(self.output / 'case/weight.npy').any())
        with np.load(next((self.output / '.cache').glob('l0-*.npz')), allow_pickle=False) as saved:
            self.assertTrue(any(v.dtype == np.float32 and np.any(v == np.float32(.350123)) for v in saved.values()))

    def test_invalid_strength_and_edits_alias_are_rejected_before_inference(self):
        for strength in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError): self.run_image(strength=strength)
        folder = self.output / 'case'; folder.mkdir(parents=True)
        mask = folder / 'weight.png'; Image.new('L', (40, 32), 255).save(mask)
        edit = self.edit(protect_mask=str(mask)); before = mask.read_bytes()
        with self.assertRaises(ValueError): self.run_image(edits_path=edit)
        self.assertEqual(mask.read_bytes(), before); self.assertFalse(self.models.calls)

    def test_corrupt_evidence_cache_is_visible_and_recomputed(self):
        self.run_image()
        next((self.output / '.cache').glob('evidence-*.npz')).write_bytes(b'corrupt')
        (self.output / 'case/result.png').unlink()
        report = self.run_image()
        self.assertTrue(any('cache rejected' in warning.lower() for warning in report['warnings']))
        self.assertEqual(len(self.models.calls), 2)

    def test_batch_errors_continue_and_new_profiles_only_write_json_csv(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        bad = self.root / 'bad.png'; Image.new('RGB', (40, 32), (7, 7, 7)).save(bad)
        manifest = {'items': [dict(id='bad', path=str(bad), status='eligible', split='tune'),
                              dict(id='good', path=str(self.source), status='eligible', split='tune')]}
        with mock.patch.object(pipeline, 'fuse_evidence', side_effect=regions), \
             mock.patch.object(pipeline, 'refine_regions', side_effect=lambda rgb, ev, reg, models, policy=None: reg):
            reports = pipeline.process_batch(manifest, self.output, self.models, l0_fn=self.l0, denoiser='l0')
        self.assertEqual([r['status'] for r in reports], ['error', 'complete'])
        self.assertTrue((self.output / 'summary.csv').is_file())
        self.assertFalse(list(self.output.rglob('*.html')))

    def test_json_temporary_cannot_alias_a_manual_mask(self):
        folder = self.output / 'case'; folder.mkdir(parents=True)
        mask = folder / 'regions.json.tmp'; Image.new('L', (40, 32), 255).save(mask, format='PNG')
        before = mask.read_bytes()
        with self.assertRaises(ValueError):
            self.run_image(edits_path=self.edit(protect_mask=str(mask)))
        self.assertEqual(mask.read_bytes(), before)

    def test_missing_batch_source_does_not_poison_other_cases(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        manifest = {'items': [dict(id='missing', path=str(self.root / 'missing.png'), status='eligible'),
                              dict(id='good', path=str(self.source), status='eligible')]}
        with mock.patch.object(pipeline, 'fuse_evidence', side_effect=regions), \
             mock.patch.object(pipeline, 'refine_regions', side_effect=lambda rgb, ev, reg, models, policy=None: reg):
            reports = pipeline.process_batch(manifest, self.output, self.models, l0_fn=self.l0, denoiser='l0')
        self.assertEqual([r['status'] for r in reports], ['error', 'complete'])

    def test_manual_sam_prompts_and_float_candidates_survive_result_rebuild(self):
        entry = dict(action='protect', positive=[[.5, .5]], negative=[[0, 0]], box=[.1, .1, .9, .9])
        edit = self.edit(sam_regions=[entry])
        first = self.run_image(edits_path=edit)
        self.assertTrue(first['sam_records'][0]['accepted'])
        self.assertFalse(np.load(self.output / 'case/weight.npy')[12:20, 15:25].any())
        (self.output / 'case/result.png').unlink()
        second = self.run_image(edits_path=edit)
        self.assertEqual(self.models.sam_calls, 1)
        self.assertEqual(second['cache_hits']['sam'], 1)

    def test_real_reference_keeps_canonical_person_labels_without_auto_refinement(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        with mock.patch.object(pipeline, 'refine_regions', side_effect=AssertionError('reference called SAM')):
            report = pipeline.process_image(self.source, self.output, self.models,
                                             profile='reference', case_id='real-reference', l0_fn=self.l0, denoiser='l0')
        self.assertEqual(report['status'], 'complete')
        self.assertFalse(self.models.calls[0][1])
        labels = np.asarray(Image.open(self.output / 'real-reference/person_labels.png'))
        self.assertTrue(np.isin(labels, [0, 1, 2, 3, 4, 5, 6]).all())

    def test_directory_reuses_grouped_inventory_and_matches_edits_by_id_sha(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        source_dir = self.root / 'inputs'; source_dir.mkdir()
        Image.fromarray(self.rgb).save(source_dir / 'same-a.png')
        Image.fromarray(self.rgb).save(source_dir / 'same-b.png')
        edits_dir = self.root / 'edits'; edits_dir.mkdir()
        Image.new('L', (40, 32), 255).save(edits_dir / 'protect.png')
        digest = hashlib.sha256((source_dir / 'same-a.png').read_bytes()).hexdigest()
        (edits_dir / 'any-name.json').write_text(json.dumps(dict(version=2, id='img_0001', source_sha256=digest,
            size=[40, 32], protect_mask='protect.png')), encoding='utf-8')
        with mock.patch.object(pipeline, 'fuse_evidence', side_effect=regions), \
             mock.patch.object(pipeline, 'refine_regions', side_effect=lambda rgb, ev, reg, models, policy=None: reg):
            reports = pipeline.process_batch(directory=source_dir, output_root=self.output, models=self.models,
                                             edits_dir=edits_dir, l0_fn=self.l0, denoiser='l0')
        self.assertEqual([r['status'] for r in reports], ['complete', 'complete'])
        self.assertEqual(reports[0]['split'], reports[1]['split'])
        self.assertFalse(np.load(self.output / 'img_0001/weight.npy').any())
        self.assertTrue(np.load(self.output / 'img_0002/weight.npy').any())
        data = json.loads((edits_dir / 'any-name.json').read_text(encoding='utf-8'))
        data['source_sha256'] = '0' * 64
        (edits_dir / 'any-name.json').write_text(json.dumps(data), encoding='utf-8')
        with mock.patch.object(pipeline, 'fuse_evidence', side_effect=regions), \
             mock.patch.object(pipeline, 'refine_regions', side_effect=lambda rgb, ev, reg, models, policy=None: reg):
            reports = pipeline.process_batch(directory=source_dir, output_root=self.output, models=self.models,
                                             edits_dir=edits_dir, l0_fn=self.l0, denoiser='l0')
        self.assertEqual([r['status'] for r in reports], ['error', 'complete'])
        self.assertIn('sha256', reports[0]['error'])

    def test_exif_grid_and_input_hardlinks_are_preserved(self):
        self.source = self.root / 'oriented.jpg'
        exif = Image.Exif(); exif[274] = 6
        Image.fromarray(self.rgb).save(self.source, exif=exif)
        with Image.open(self.source) as stored:
            oriented = np.asarray(ImageOps.exif_transpose(stored).convert('RGB'))
        before = self.source.read_bytes()
        report = self.run_image()
        self.assertEqual(report['size'], [32, 40])
        np.testing.assert_array_equal(np.asarray(Image.open(self.output / 'case/original.png')), oriented)
        self.assertEqual(before, self.source.read_bytes())
        alias_output = self.root / 'alias'; folder = alias_output / 'case'; folder.mkdir(parents=True)
        os.link(self.source, folder / 'result.png')
        from gpt_animeimage_denoise.fusion_pipeline import process_image
        with self.assertRaises(ValueError):
            process_image(self.source, alias_output, self.models, case_id='case', l0_fn=self.l0, denoiser='l0')
        self.assertEqual(before, self.source.read_bytes())

    def test_model_precision_and_policy_invalidate_only_relevant_stage_caches(self):
        first = self.run_image()
        self.models.signature = 'weights-precision-changed'
        second = self.run_image()
        self.assertNotEqual(first['fingerprint'], second['fingerprint'])
        self.assertEqual(len(self.models.calls), 2); self.assertEqual(self.l0.calls, 1)
        third = self.run_image(policy={'version': 'changed', 'm2f': .8})
        self.assertNotEqual(second['fingerprint'], third['fingerprint'])
        self.assertEqual(len(self.models.calls), 2); self.assertEqual(self.l0.calls, 1)
        fourth = self.run_image(lam=.001)
        self.assertEqual(self.l0.calls, 2)
        self.assertFalse(fourth['resumed'])

    def test_sam_model_is_released_before_l0_stage(self):
        class ReleasableModels(Models):
            def __init__(self):
                super().__init__(); self.released = False
            def release(self):
                self.released = True
        models = ReleasableModels()
        class SerialL0:
            signature = 'serial-release-test'
            def __call__(self, image, lam, *, device):
                if not models.released:
                    raise AssertionError('SAM GPU stage not released before L0')
                return image.copy()
        self.models = models; self.l0 = SerialL0()
        self.assertEqual(self.run_image()['status'], 'complete')

    def test_optional_roi_proposals_are_cached_as_original_grid_masks(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        class ProposalModels(Models):
            def __init__(self):
                super().__init__(); self.proposals = 0
            def generate_candidates(self, rgb, box):
                self.proposals += 1
                mask = np.zeros(rgb.shape[:2], bool); mask[12:20, 15:25] = True
                return [dict(mask=mask, score=.9, stability=.95, bbox=box)]
        self.models = ProposalModels()
        def with_proposals(rgb, evidence, reg, models, policy=None):
            proposals = models.generate_candidates(rgb, [10, 10, 30, 25])
            reg['objects'] |= proposals[0]['mask']
            return reg
        for _ in range(2):
            with mock.patch.object(pipeline, 'fuse_evidence', side_effect=regions), \
                 mock.patch.object(pipeline, 'refine_regions', side_effect=with_proposals):
                report = pipeline.process_image(self.source, self.output, self.models, case_id='roi', l0_fn=self.l0, denoiser='l0')
            (self.output / 'roi/result.png').unlink()
        self.assertEqual(self.models.proposals, 1)
        self.assertEqual(report['cache_hits']['sam'], 1)

    def test_invalid_sam_float_precision_is_rejected_without_caching(self):
        class WrongPrecision(Models):
            def predict_candidates(self, rgb, prompt, *, mask_input=None):
                prediction = super().predict_candidates(rgb, prompt, mask_input=mask_input)
                prediction['lowres'] = prediction['lowres'].astype(np.float64)
                return prediction
        self.models = WrongPrecision()
        edit = self.edit(sam_regions=[dict(action='protect', positive=[[.5, .5]])])
        with self.assertRaisesRegex(ValueError, 'float32'):
            self.run_image(edits_path=edit)
        self.assertFalse(list((self.output / '.cache').glob('sam-*.npz')))

    def test_policy_file_is_validated_and_bound_to_result_bytes(self):
        policy = self.root / 'calibrated.json'
        policy.write_text('{"m2f": 0.86, "max_tasks": 1}', encoding='utf-8')
        first = self.run_image(policy_path=policy)
        self.assertEqual(first['parameters']['policy']['m2f'], .86)
        self.assertEqual(first['parameters']['policy']['max_rounds'], 2)
        self.assertEqual(first['parameters']['policy_sha256'], hashlib.sha256(policy.read_bytes()).hexdigest())
        self.assertIn(str(policy.resolve()), first['input_paths'])
        policy.write_text('{"m2f": 0.86, "max_tasks": 1}\n', encoding='utf-8')
        second = self.run_image(policy_path=policy)
        self.assertNotEqual(first['fingerprint'], second['fingerprint'])
        self.assertFalse(second['resumed'])
        policy.write_text('{"m2f": 0.9}', encoding='utf-8')
        third = self.run_image(policy_path=policy)
        self.assertEqual(third['parameters']['policy']['m2f'], .9)
        self.assertNotEqual(second['fingerprint'], third['fingerprint'])
        self.assertEqual(len(self.models.calls), 1); self.assertEqual(self.l0.calls, 1)

    def test_policy_file_and_memory_policy_are_exclusive_and_invalid_policy_is_rejected(self):
        policy = self.root / 'policy.json'; policy.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'policy'):
            self.run_image(policy_path=policy, policy={})
        for content in ['[]', '{"m2f": 1.2}', '{"max_tasks": 17}']:
            policy.write_text(content, encoding='utf-8')
            with self.assertRaises(ValueError):
                self.run_image(policy_path=policy)
        self.assertFalse(self.models.calls)

    def test_policy_file_cannot_alias_case_output_or_batch_summary(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        folder = self.output / 'case'; folder.mkdir(parents=True)
        policy = folder / 'regions.json.tmp'; policy.write_text('{}', encoding='utf-8')
        before = policy.read_bytes()
        with self.assertRaises(ValueError):
            self.run_image(policy_path=policy)
        self.assertEqual(policy.read_bytes(), before)
        summary = self.output / 'summary.csv'; summary.write_text('{}', encoding='utf-8')
        manifest = {'items': [dict(id='case', path=str(self.source), status='eligible')]}
        with self.assertRaises(ValueError):
            pipeline.process_batch(manifest, self.output, self.models, policy_path=summary, l0_fn=self.l0, denoiser='l0')
        self.assertEqual(summary.read_bytes(), b'{}')
        self.assertFalse(self.models.calls)

    def test_batch_forwards_policy_path_into_case_identity(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        policy = self.root / 'policy.json'; policy.write_text('{"m2f": 0.89}', encoding='utf-8')
        manifest = {'items': [dict(id='case', path=str(self.source), status='eligible')]}
        with mock.patch.object(pipeline, 'fuse_evidence', side_effect=regions), \
             mock.patch.object(pipeline, 'refine_regions', side_effect=lambda rgb, ev, reg, models, policy=None: reg):
            records = pipeline.process_batch(manifest, self.output, self.models, policy_path=policy, l0_fn=self.l0, denoiser='l0')
        self.assertEqual(records[0]['status'], 'complete')
        self.assertEqual(records[0]['parameters']['policy']['m2f'], .89)
        self.assertEqual(records[0]['parameters']['policy_sha256'], hashlib.sha256(policy.read_bytes()).hexdigest())

    def test_changed_policy_during_filtering_does_not_mark_result_complete(self):
        policy = self.root / 'policy.json'; policy.write_text('{}', encoding='utf-8')
        class MutatingL0:
            signature = 'mutating-policy-test'
            def __call__(self, image, lam, *, device):
                policy.write_text('{"m2f": 0.85}', encoding='utf-8')
                return image.copy()
        self.l0 = MutatingL0()
        with self.assertRaisesRegex(RuntimeError, 'changed while processing'):
            self.run_image(policy_path=policy)
        self.assertFalse((self.output / 'case/report.json').exists())

    def test_reference_reuses_valid_full_evidence_without_dual_view_inference(self):
        self.run_image(profile='fusion')
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        with mock.patch.object(pipeline, 'refine_regions', side_effect=AssertionError('reference called auto SAM')):
            report = pipeline.process_image(self.source, self.output, self.models,
                                             profile='reference', case_id='reference', l0_fn=self.l0, denoiser='l0')
        self.assertEqual(len(self.models.calls), 1)
        self.assertTrue(report['cache_hits']['evidence'])
        self.assertTrue((np.asarray(Image.open(self.output / 'reference/person_labels.png')) == 1).all())

    def test_invalid_full_cache_falls_back_to_single_view_and_single_cannot_feed_fusion(self):
        self.run_image(profile='fusion')
        full_cache = next((self.output / '.cache').glob('evidence-*.npz'))
        full_cache.write_bytes(b'broken-full-evidence')
        reference = self.run_image(profile='reference')
        self.assertFalse(self.models.calls[-1][1])
        self.assertFalse(reference['cache_hits']['evidence'])
        self.assertTrue(any('evidence cache rejected' in warning for warning in reference['warnings']))
        full_cache.unlink()
        fusion = self.run_image(profile='fusion')
        self.assertEqual([dual for _, dual in self.models.calls], [True, False, True])
        self.assertFalse(fusion['cache_hits']['evidence'])

    def test_resumed_results_also_verify_policy_file_stayed_unchanged(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        policy = self.root / 'policy.json'; policy.write_text('{}', encoding='utf-8')
        self.run_image(policy_path=policy)
        resume = pipeline._resume
        def mutate_after_resume(*args):
            result = resume(*args)
            policy.write_text('{"m2f": 0.85}', encoding='utf-8')
            return result
        with mock.patch.object(pipeline, '_resume', side_effect=mutate_after_resume):
            with self.assertRaisesRegex(RuntimeError, 'changed while processing'):
                self.run_image(policy_path=policy)

    def test_reviewed_allow_reaches_outfit_background_but_preserves_hard_vetoes(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        alpha = np.full((32, 40), 255, np.uint8); alpha[0] = 0
        Image.fromarray(np.dstack([self.rgb, alpha])).save(self.source)
        Image.new('L', (40, 32), 255).save(self.root / 'allow.png')
        protect = np.zeros((32, 40), np.uint8); protect[20:25, 20:25] = 255
        Image.fromarray(protect).save(self.root / 'protect.png')
        edit = self.edit(allow_mask='allow.png', protect_mask='protect.png')
        def auto_scope(evidence, policy=None):
            reg = regions(evidence, policy)
            reg['person_labels'][12:20, 22:30] = 0
            reg['person_labels'][reg['objects']] = 0
            reg['allowed'] = np.isin(reg['person_labels'], [1, 2, 3]) & ~reg['objects']
            reg['protected'] = ~reg['allowed']
            return reg
        expected = auto_scope({'alpha': np.ones((32, 40), np.float32)})['person_labels']
        with mock.patch.object(pipeline, 'fuse_evidence', side_effect=auto_scope), \
             mock.patch.object(pipeline, 'refine_regions', side_effect=lambda rgb, ev, reg, models, policy=None: reg):
            pipeline.process_image(self.source, self.output, self.models, case_id='allow-scope',
                                   edits_path=edit, sigma=1200, l0_fn=self.l0, denoiser='l0')
        weight = np.load(self.output / 'allow-scope/weight.npy')
        self.assertTrue(weight[2:, 3:6].any())
        self.assertTrue(weight[12:20, 22:30].any())
        self.assertFalse(weight[:, :3].any())
        self.assertFalse(weight[5:10, 10:15].any())
        self.assertFalse(weight[protect > 0].any())
        self.assertFalse(weight[alpha == 0].any())
        np.testing.assert_array_equal(np.asarray(Image.open(self.output / 'allow-scope/person_labels.png')), expected)
        result = np.asarray(Image.open(self.output / 'allow-scope/result.png'))
        np.testing.assert_array_equal(result[..., :3][weight == 0], self.rgb[weight == 0])
        np.testing.assert_array_equal(result[..., 3], alpha)
        self.assertTrue(np.any(result[12:20, 22:30, :3] != self.rgb[12:20, 22:30]))
