import tempfile
import unittest
import json
import hashlib
import os
from PIL import Image
from pathlib import Path

import numpy as np

from gpt_animeimage_denoise.fusion_evaluation import (segmentation_metrics, release_gate,
    require_approved_references, threshold_grid, select_review_cases)
from gpt_animeimage_denoise.fusion_evaluation import calibrate_policies
from gpt_animeimage_denoise.fusion_evaluation import load_reviewed_cases,evaluate_runs,safe_evaluation_output
from gpt_animeimage_denoise.fusion_evaluation import render_denoise_review


class EvaluationTests(unittest.TestCase):
    def test_unknown_prediction_is_missed_class_not_ignored(self):
        gt = np.array([[1, 1, 7, 255]], np.uint8)
        pred = np.array([[1, 255, 6, 1]], np.uint8)
        objects = np.zeros_like(gt, bool)
        result = segmentation_metrics(gt, pred, objects)
        self.assertEqual(result['valid_pixels'], 3)
        self.assertEqual(result['unknown_pixels'], 1)
        self.assertAlmostEqual(result['modules']['skin']['recall'], .5)
        self.assertEqual(result['outfit_object_leakage'], 1)

    def test_false_object_exclusion_counts_real_person(self):
        gt = np.array([[1, 2, 6, 7]], np.uint8)
        pred = np.array([[0, 2, 6, 0]], np.uint8)
        objects = np.array([[True, False, False, True]])
        result = segmentation_metrics(gt, pred, objects)
        self.assertAlmostEqual(result['person_false_exclusion'], 1/3)

    def test_review_gate_blocks_unapproved_and_wrong_split(self):
        with self.assertRaisesRegex(ValueError, 'approval'):
            require_approved_references([{'id': 'x', 'approved': False, 'split': 'tune'}], 'tune')
        with self.assertRaisesRegex(ValueError, 'split'):
            require_approved_references([{'id': 'x', 'approved': True, 'split': 'holdout'}], 'tune')
        result = require_approved_references([{'id': 'x', 'approved': True, 'split': 'tune'}], 'tune')
        self.assertEqual(len(result), 1)

    def test_release_fails_without_object_or_module_reference(self):
        zero = np.zeros((8, 8), np.uint8)
        result = segmentation_metrics(zero, zero, zero.astype(bool))
        gate = release_gate(result, result)
        self.assertFalse(gate['accepted'])
        self.assertIn('insufficient_reference', gate['reasons'])

    def test_release_requires_actual_improvement_and_all_class_f1(self):
        gt = np.array([[1, 2, 3, 4, 5, 6, 7, 7]], np.uint8)
        baseline = np.array([[1, 2, 3, 4, 5, 6, 6, 6]], np.uint8)
        improved = np.array([[1, 2, 3, 4, 5, 6, 0, 6]], np.uint8)
        first = segmentation_metrics(gt, baseline, np.zeros_like(gt, bool))
        second = segmentation_metrics(gt, improved, improved == 0)
        self.assertTrue(release_gate(first, second)['accepted'])
        bad = segmentation_metrics(gt, improved, np.ones_like(gt, bool))
        self.assertFalse(release_gate(first, bad)['accepted'])

    def test_unknown_external_objects_do_not_fake_leakage_improvement(self):
        gt = np.array([[1,2,3,4,5,6,7,7]], np.uint8)
        baseline = np.array([[1,2,3,4,5,6,6,6]], np.uint8)
        abandoned = np.array([[1,2,3,4,5,6,255,255]], np.uint8)
        a=segmentation_metrics(gt,baseline,np.zeros_like(gt,bool))
        b=segmentation_metrics(gt,abandoned,np.zeros_like(gt,bool))
        self.assertEqual(b['unknown_on_objects_pixels'],2)
        self.assertFalse(release_gate(a,b)['accepted'])

    def test_threshold_grid_is_declared_nine_configs(self):
        grid = list(threshold_grid())
        self.assertEqual(len(grid), 9)
        self.assertEqual({p['m2f_threshold'] for p in grid}, {.6, .75, .9})
        self.assertEqual({p['see_threshold'] for p in grid}, {0, 1, 2})

    def test_review_selection_uses_eligible_tune_and_no_group_leakage(self):
        items = [{'id': f'img_{i:04d}', 'status': 'eligible', 'split': 'tune',
                  'group_id': f'g{i}', 'dhash': f'{i:016x}'} for i in range(20)]
        items += [{'id': 'bad', 'status': 'excluded_from_calibration', 'split': 'tune'},
                  {'id': 'hold', 'status': 'eligible', 'split': 'holdout'}]
        chosen = select_review_cases({'items': items}, 12)
        self.assertEqual(len(chosen), 12)
        self.assertTrue(all(item['split'] == 'tune' for item in chosen))
        self.assertEqual(chosen, select_review_cases({'items': items}, 12))

    def test_calibration_blocks_all_model_calls_before_reference_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'review.json'
            path.write_text(json.dumps({'cases':[{'id':'x','split':'tune','approved':False}]}))
            def forbidden(*args):
                self.fail('A model callback ran before reference approval')
            with self.assertRaisesRegex(ValueError,'approval'):
                calibrate_policies(path,forbidden,forbidden)


class EvaluationInputSafety(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cases=[];items=[]
        for index,split in enumerate(['tune','holdout']):
            source=self.root/(split+'.png');Image.fromarray(np.full((2,8,3),80+index,np.uint8)).save(source)
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            item={'id':split,'path':str(source),'sha256':sha,'group_id':split,'split':split,'status':'eligible','width':8,'height':2}
            items.append(item)
            label=self.root/(split+'-gold.png');Image.fromarray(np.tile(np.array([1,2,3,4,5,6,7,7],np.uint8),(2,1))).save(label)
            self.cases.append({'id':split,'source':str(source),'source_sha256':sha,'group_id':split,'split':split,
                'approved':True,'size':[8,2],'reference_labels':label.name,'rois':[{'name':'all','xyxy':[0,0,8,2]}]})
        self.manifest=self.root/'dataset.json';self.manifest.write_text(json.dumps({'items':items}))
        self.review=self.root/'review.json';self.document={'source_manifest':str(self.manifest),
            'source_manifest_sha256':hashlib.sha256(self.manifest.read_bytes()).hexdigest(),'cases':self.cases}
        self.save_review()

    def save_review(self):
        self.review.write_text(json.dumps(self.document),encoding='utf-8')

    def test_frozen_manifest_hash_and_split_membership_are_checked(self):
        self.assertEqual(len(load_reviewed_cases(self.review,'tune')),1)
        self.document['source_manifest_sha256']='fake';self.save_review()
        with self.assertRaisesRegex(ValueError,'manifest'):
            load_reviewed_cases(self.review,'tune')

    def test_cross_split_group_and_source_cannot_be_used_to_calibrate(self):
        self.document['cases'][1]['group_id']='tune';self.save_review()
        called=[]
        with self.assertRaises(ValueError):
            calibrate_policies(self.review,lambda *a:called.append(a),lambda *a:called.append(a))
        self.assertFalse(called)

    def test_wrong_or_incomplete_prediction_profiles_are_rejected(self):
        for profile in ['reference','fusion']:
            folder=self.root/profile/'holdout';folder.mkdir(parents=True)
            pred=np.tile(np.array([1,2,3,4,5,6,0,0],np.uint8),(2,1))
            Image.fromarray(pred).save(folder/'person_labels.png')
            Image.fromarray((pred==0).astype(np.uint8)*255).save(folder/'objects.png')
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.png')}
            report={'status':'error','profile':profile,'source_sha256':self.cases[1]['source_sha256'],'output_hashes':hashes}
            (folder/'report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'complete'):
            evaluate_runs(self.review,self.root/'reference',self.root/'fusion')

    def test_evaluation_output_cannot_overwrite_review_source_or_labels(self):
        for path in [self.review,Path(self.cases[0]['source']),self.root/self.cases[1]['reference_labels']]:
            with self.assertRaises(ValueError):
                safe_evaluation_output(path,self.review)

    def test_complete_correct_profiles_work_but_swapped_or_changed_outputs_fail(self):
        for profile in ['reference','fusion']:
            folder=self.root/profile/'holdout';folder.mkdir(parents=True)
            pred=np.tile(np.array([1,2,3,4,5,6,0,0] if profile=='fusion' else [1,2,3,4,5,6,6,6],np.uint8),(2,1))
            Image.fromarray(pred).save(folder/'person_labels.png')
            Image.fromarray((pred==0).astype(np.uint8)*255).save(folder/'objects.png')
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.png')}
            report={'status':'complete','profile':profile,'source_sha256':self.cases[1]['source_sha256'],'output_hashes':hashes}
            (folder/'report.json').write_text(json.dumps(report))
        result=evaluate_runs(self.review,self.root/'reference',self.root/'fusion')
        self.assertTrue(result['gate']['accepted'])
        candidate=self.root/'fusion/holdout';report=json.loads((candidate/'report.json').read_text())
        report['profile']='reference';(candidate/'report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'profile'):
            evaluate_runs(self.review,self.root/'reference',self.root/'fusion')
        report['profile']='fusion';(candidate/'report.json').write_text(json.dumps(report))
        Image.fromarray(np.full((2,8),2,np.uint8)).save(candidate/'person_labels.png')
        with self.assertRaisesRegex(ValueError,'hash'):
            evaluate_runs(self.review,self.root/'reference',self.root/'fusion')

    def _make_legacy_release_fixture(self):
        from copy import deepcopy
        from gpt_animeimage_denoise.fusion import DEFAULT_POLICY,_policy
        from gpt_animeimage_denoise.fusion_evaluation import reference_digest
        policy=deepcopy(DEFAULT_POLICY);policy.pop('outfit_reference')
        policy['source_codes'].pop('outfit_reference')
        policy.update(provisional=False,calibrated=True,
            calibration_reference_sha256=reference_digest(self.review,'tune'))
        policy_path=self.root/'frozen_policy.json';policy_path.write_text(json.dumps(policy),encoding='utf-8')
        actual=_policy(policy)
        for profile in ['reference','fusion']:
            folder=self.root/profile/'holdout';folder.mkdir(parents=True)
            pred=np.tile(np.array([1,2,3,4,5,6,0,0] if profile=='fusion' else [1,2,3,4,5,6,6,6],np.uint8),(2,1))
            Image.fromarray(pred).save(folder/'person_labels.png')
            Image.fromarray((pred==0).astype(np.uint8)*255).save(folder/'objects.png')
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.png')}
            report={'status':'complete','profile':profile,'source_sha256':self.cases[1]['source_sha256'],
                'parameters':{'policy':actual},'output_hashes':hashes}
            (folder/'report.json').write_text(json.dumps(report),encoding='utf-8')
        return policy_path,actual

    def test_release_accepts_legacy_frozen_policy_normalized_without_changing_its_identity(self):
        from gpt_animeimage_denoise.fusion_evaluation import evaluate_release
        policy_path,actual=self._make_legacy_release_fixture();before=policy_path.read_bytes()
        self.assertEqual(actual['outfit_reference'],'off')
        self.assertEqual(actual['source_codes']['outfit_reference'],25)
        result=evaluate_release(self.review,self.root/'reference',self.root/'fusion',policy_path)
        self.assertTrue(result['gate']['accepted'])
        self.assertEqual(result['frozen_policy_sha256'],hashlib.sha256(before).hexdigest())
        self.assertEqual(policy_path.read_bytes(),before)

    def test_release_still_rejects_changed_threshold_parts_or_outfit_mode(self):
        from gpt_animeimage_denoise.fusion_evaluation import evaluate_release
        policy_path,actual=self._make_legacy_release_fixture()
        report_path=self.root/'fusion/holdout/report.json';report=json.loads(report_path.read_text())
        for changed in [{'m2f':.9},{'see_repair_parts':['face']},{'outfit_reference':'m2f-white'}]:
            with self.subTest(changed_policy=changed):
                report['parameters']['policy']=dict(actual,**changed)
                report_path.write_text(json.dumps(report),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'frozen policy'):
                    evaluate_release(self.review,self.root/'reference',self.root/'fusion',policy_path)

    def test_denoise_review_cannot_change_already_approved_references(self):
        with self.assertRaisesRegex(ValueError,'approved'):
            render_denoise_review(self.review,self.root/'predictions')

    def _make_float_review_fixture(self):
        from gpt_animeimage_denoise.fusion_pipeline import _pack,_payload_hash,_code_hash
        from gpt_animeimage_denoise.filters import blend,gradient_preserving_candidate
        predictions=self.root/'predictions';cache=predictions/'.cache';cache.mkdir(parents=True)
        for case in self.document['cases']:
            case['approved']=False;folder=predictions/case['id'];folder.mkdir()
            rgb=np.array(Image.open(case['source']));pure=np.full(rgb.shape,.2,np.float32)
            weight=np.full(rgb.shape[:2],.5,np.float32);np.save(folder/'weight.npy',weight)
            restored=gradient_preserving_candidate(rgb,pure,1)
            Image.fromarray(blend(rgb,restored,weight)).save(folder/'result.png')
            report={'status':'complete','source_sha256':case['source_sha256'],
                'parameters':{'lambda':.0005,'device':'cpu','l0_implementation':{'kind':'installed_l0'}},'effective_sigma':1,
                'output_hashes':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['weight.npy','result.png']}}
            (folder/'report.json').write_text(json.dumps(report))
            arrays={};tree=_pack(pure,arrays)
            identity={'source_sha256':case['source_sha256'],'lambda':.0005,'device':'cpu','callable':{'kind':'installed_l0'},
                'code':_code_hash(['../denoise.py','../l0_cuda.py','fusion_pipeline.py'])}
            np.savez_compressed(cache/('l0-'+case['id']+'.npz'),metadata=json.dumps({'identity':identity,'tree':tree,
                'payload_hash':_payload_hash(tree,arrays)}),**arrays)
        self.save_review()
        return predictions

    def test_blind_comparisons_reuse_exact_float_candidate_and_same_weight(self):
        predictions=self._make_float_review_fixture()
        result=render_denoise_review(self.review,predictions)
        self.assertTrue(all(c['float_candidate_reproduced_saved_result'] for c in result['cases']))
        self.assertFalse(result['quality_accepted'])
        self.assertTrue((self.root/'denoise_review.json').exists())

    def test_review_replays_reported_quarter_restoration_and_historical_full_restoration(self):
        from scipy.ndimage import gaussian_filter
        from gpt_animeimage_denoise.filters import blend

        predictions=self._make_float_review_fixture()
        self._declare_candidate_cache(predictions)
        folder=predictions/'tune'
        report_path=folder/'report.json'
        report=json.loads(report_path.read_text())
        report['parameters'].update(restoration_gain=.25,structure_protection=0)
        case=next(c for c in self.document['cases'] if c['id']=='tune')
        rgb=np.array(Image.open(case['source']))
        weight=np.load(folder/'weight.npy',allow_pickle=False)
        pure=np.full(rgb.shape,.2,np.float32)
        sigma=report['effective_sigma']
        correction=gaussian_filter(rgb.astype(np.float32)/255-pure,(sigma,sigma,0),mode='reflect')
        restored=np.clip(pure+.25*correction,0,1)
        result_path=folder/'result.png'
        expected=blend(rgb,restored,weight)
        self.assertFalse(np.array_equal(expected,np.array(Image.open(result_path))))
        Image.fromarray(expected).save(result_path)
        report['output_hashes']['result.png']=hashlib.sha256(result_path.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report))

        result=render_denoise_review(self.review,predictions)
        by_id={entry['id']:entry for entry in result['cases']}
        self.assertEqual(by_id['tune']['restoration_gain'],.25)
        self.assertEqual(by_id['tune']['structure_protection'],0)
        self.assertEqual(by_id['holdout']['restoration_gain'],1)
        self.assertEqual(by_id['holdout']['structure_protection'],1)
        self.assertTrue(all(entry['float_candidate_reproduced_saved_result'] for entry in by_id.values()))

    def _write_legacy_review_cache(self,predictions,case_id,*,code='historical-pipeline-code',
                                   candidate=None,suffix='',identity_changes=None):
        from gpt_animeimage_denoise.fusion_pipeline import _pack,_payload_hash
        path=predictions/'.cache'/('l0-'+case_id+'.npz')
        with np.load(path,allow_pickle=False) as saved:
            identity=json.loads(str(saved['metadata'].item()))['identity']
            value=saved['array_0'] if candidate is None else candidate
        identity.update(code=code,**(identity_changes or {}))
        arrays={};tree=_pack(value,arrays)
        target=path if not suffix else path.with_stem(path.stem+suffix)
        np.savez_compressed(target,metadata=json.dumps({'identity':identity,'tree':tree,
            'payload_hash':_payload_hash(tree,arrays)}),**arrays)
        return target

    def test_historical_l0_code_change_requires_report_with_declared_identity(self):
        from gpt_animeimage_denoise.fusion_evaluation import _review_l0
        predictions=self._make_float_review_fixture()
        for case in self.document['cases']:
            self._write_legacy_review_cache(predictions,case['id'])
        report=json.loads((predictions/'tune/report.json').read_text())
        with self.assertRaisesRegex(ValueError,'Exact float L0 cache unavailable'):
            _review_l0(predictions/'.cache',report,(2,8,3))
        with self.assertRaisesRegex(ValueError,'rerun.*candidate_cache_identity'):
            render_denoise_review(self.review,predictions)

    def test_unique_wrong_code_cache_cannot_be_identified_by_zero_weight_replay(self):
        predictions=self._make_float_review_fixture()
        for case in self.document['cases']:
            folder=predictions/case['id'];rgb=np.array(Image.open(case['source']))
            np.save(folder/'weight.npy',np.zeros(rgb.shape[:2],np.float32))
            Image.fromarray(rgb).save(folder/'result.png')
            report_path=folder/'report.json';report=json.loads(report_path.read_text())
            report['output_hashes']={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                for name in ['weight.npy','result.png']}
            report_path.write_text(json.dumps(report))
            self._write_legacy_review_cache(predictions,case['id'],candidate=np.full(rgb.shape,.9,np.float32))
        with self.assertRaisesRegex(ValueError,'rerun.*candidate_cache_identity'):
            render_denoise_review(self.review,predictions)

    def test_historical_replay_rejects_when_no_float_candidate_reproduces(self):
        from gpt_animeimage_denoise.fusion_pipeline import _code_hash
        predictions=self._make_float_review_fixture()
        value=np.repeat(np.where(np.indices((2,8))[1]%2,.8,.1)[...,None],3,axis=2).astype(np.float32)
        self._write_legacy_review_cache(predictions,'tune',candidate=value,
            code=_code_hash(['../denoise.py','../l0_cuda.py','fusion_pipeline.py']))
        with self.assertRaisesRegex(ValueError,'No float L0 candidate reproduces'):
            render_denoise_review(self.review,predictions)

    def test_historical_replay_rejects_distinct_float_candidates_with_same_result(self):
        from gpt_animeimage_denoise.fusion_pipeline import _code_hash
        predictions=self._make_float_review_fixture()
        self._write_legacy_review_cache(predictions,'tune',candidate=np.full((2,8,3),.3,np.float32),suffix='-other',
            code=_code_hash(['../denoise.py','../l0_cuda.py','fusion_pipeline.py']))
        with self.assertRaisesRegex(ValueError,'Ambiguous float L0'):
            render_denoise_review(self.review,predictions)

    def test_current_code_legacy_replay_deduplicates_numerically_identical_candidates(self):
        from gpt_animeimage_denoise.fusion_pipeline import _code_hash
        predictions=self._make_float_review_fixture()
        code=_code_hash(['../denoise.py','../l0_cuda.py','fusion_pipeline.py'])
        for case in self.document['cases']:
            self._write_legacy_review_cache(predictions,case['id'],code=code)
            self._write_legacy_review_cache(predictions,case['id'],code=code,suffix='-copy')
        result=render_denoise_review(self.review,predictions)
        self.assertTrue(all(c['float_candidate_reproduced_saved_result'] for c in result['cases']))

    def test_historical_replay_checks_device_and_rejects_explicit_backend_cache(self):
        from gpt_animeimage_denoise.fusion_pipeline import _code_hash
        predictions=self._make_float_review_fixture()
        for changes in [{'device':'cuda'},{'denoiser':'esrgan-l0'},{'denoiser':'l0'},
                        {'callable':{'kind':'different_l0'}},{'lambda':.005},{'source_sha256':'wrong'}]:
            with self.subTest(identity_changes=changes):
                # Reset each identity instead of accumulating mismatches.
                report=json.loads((predictions/'tune/report.json').read_text())
                reset={'source_sha256':report['source_sha256'],'lambda':.0005,'device':'cpu',
                    'callable':report['parameters']['l0_implementation']}
                path=self._write_legacy_review_cache(predictions,'tune',identity_changes=reset|changes,
                    code=_code_hash(['../denoise.py','../l0_cuda.py','fusion_pipeline.py']))
                with self.assertRaisesRegex(ValueError,'rerun.*candidate_cache_identity'):
                    render_denoise_review(self.review,predictions)
                with np.load(path,allow_pickle=False) as saved:
                    metadata=json.loads(str(saved['metadata'].item()));array=saved['array_0']
                metadata['identity'].pop('denoiser',None)
                np.savez_compressed(path,metadata=json.dumps(metadata),array_0=array)

    def test_historical_replay_checks_result_hash_before_loading_float_cache(self):
        predictions=self._make_float_review_fixture()
        for case in self.document['cases']:
            self._write_legacy_review_cache(predictions,case['id'])
        (predictions/'tune/result.png').write_bytes(b'changed result')
        with self.assertRaisesRegex(ValueError,'Review final result changed'):
            render_denoise_review(self.review,predictions)

    def _declare_candidate_cache(self,predictions,denoiser='esrgan-l0',candidate=None,lam=.0005,downscale=.6):
        from gpt_animeimage_denoise.fusion_pipeline import _pack,_payload_hash
        from gpt_animeimage_denoise.pipeline import _digest
        from gpt_animeimage_denoise.filters import blend,gradient_preserving_candidate
        paths=[]
        for case in self.document['cases']:
            folder=predictions/case['id'];report_path=folder/'report.json'
            report=json.loads(report_path.read_text())
            uses_l0=denoiser=='l0' or denoiser.endswith('-l0')
            config={} if denoiser=='l0' else {'engine':denoiser.split('-')[0],
                'downscale':1 if denoiser=='apisr' else downscale}
            implementation={'kind':'installed_l0'} if uses_l0 else None
            report['parameters'].update(denoiser=denoiser,denoiser_config=config,device='cpu',
                l0_enabled=uses_l0,l0_implementation=implementation,**{'lambda':lam})
            identity={'source_sha256':case['source_sha256'],'lambda':lam if uses_l0 else None,'device':'cpu',
                'denoiser':denoiser,'denoiser_config':config,'code':'reported-candidate-code',
                'callable':report['parameters']['l0_implementation'],'precision':'float32-cache'}
            stage='l0' if denoiser=='l0' else 'candidate'
            report.update(candidate_cache_identity=identity,candidate_cache_stage=stage)
            value=np.full((2,8,3),.2,np.float32) if candidate is None else candidate
            rgb=np.array(Image.open(case['source']));weight=np.load(folder/'weight.npy',allow_pickle=False)
            restored=gradient_preserving_candidate(rgb,value,report['effective_sigma'])
            Image.fromarray(blend(rgb,restored,weight)).save(folder/'result.png')
            report['output_hashes']['result.png']=hashlib.sha256((folder/'result.png').read_bytes()).hexdigest()
            report_path.write_text(json.dumps(report))
            arrays={};tree=_pack(value,arrays)
            path=predictions/'.cache'/(stage+'-'+_digest(identity)+'.npz')
            np.savez_compressed(path,metadata=json.dumps({'identity':identity,'tree':tree,
                'payload_hash':_payload_hash(tree,arrays)}),**arrays)
            paths.append(path)
        return paths

    def test_declared_esrgan_cache_is_selected_without_confusing_old_l0(self):
        from gpt_animeimage_denoise.fusion_evaluation import _review_l0
        predictions=self._make_float_review_fixture()
        paths=self._declare_candidate_cache(predictions)
        report=json.loads((predictions/'tune/report.json').read_text())
        self.assertTrue(np.array_equal(_review_l0(predictions/'.cache',report,(2,8,3)),
            np.full((2,8,3),.2,np.float32)))
        paths[0].unlink()
        with self.assertRaisesRegex(ValueError,'Exact float candidate cache unavailable'):
            _review_l0(predictions/'.cache',report,(2,8,3))

    def test_esrgan_review_requires_report_declared_cache_identity(self):
        from gpt_animeimage_denoise.fusion_evaluation import _review_l0
        predictions=self._make_float_review_fixture()
        report=json.loads((predictions/'tune/report.json').read_text())
        report['parameters']['denoiser']='esrgan-l0'
        with self.assertRaisesRegex(ValueError,'candidate cache identity'):
            _review_l0(predictions/'.cache',report,(2,8,3))

    def test_esrgan_review_does_not_replay_implicit_l0_cache(self):
        predictions=self._make_float_review_fixture();paths=self._declare_candidate_cache(predictions)
        for candidate_path in paths:candidate_path.unlink()
        with self.assertRaisesRegex(ValueError,'Exact float candidate cache unavailable'):
            render_denoise_review(self.review,predictions)

    def test_declared_cache_metadata_and_payload_must_match_exactly(self):
        from gpt_animeimage_denoise.fusion_evaluation import _review_l0
        from gpt_animeimage_denoise.fusion_pipeline import _pack,_payload_hash
        predictions=self._make_float_review_fixture();paths=self._declare_candidate_cache(predictions)
        report=json.loads((predictions/'tune/report.json').read_text())
        identity=dict(report['candidate_cache_identity'],denoiser='l0')
        arrays={};tree=_pack(np.full((2,8,3),.2,np.float32),arrays)
        metadata={'identity':identity,'tree':tree,'payload_hash':_payload_hash(tree,arrays)}
        np.savez_compressed(paths[0],metadata=json.dumps(metadata),**arrays)
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            _review_l0(predictions/'.cache',report,(2,8,3))
        metadata['identity']=report['candidate_cache_identity'];metadata['payload_hash']='changed'
        np.savez_compressed(paths[0],metadata=json.dumps(metadata),**arrays)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            _review_l0(predictions/'.cache',report,(2,8,3))

    def test_declared_cache_requires_post_l0_float32_on_original_grid(self):
        from gpt_animeimage_denoise.fusion_evaluation import _review_l0
        from gpt_animeimage_denoise.fusion_pipeline import _pack,_payload_hash
        predictions=self._make_float_review_fixture();paths=self._declare_candidate_cache(predictions)
        report=json.loads((predictions/'tune/report.json').read_text())
        for invalid in [np.full((2,8,3),51,np.uint8),np.full((2,8,3),.2,np.float64),
                        np.full((1,8,3),.2,np.float32),np.full((2,8,3),1.1,np.float32)]:
            with self.subTest(dtype=invalid.dtype,shape=invalid.shape,max=float(invalid.max())):
                arrays={};tree=_pack(invalid,arrays)
                metadata={'identity':report['candidate_cache_identity'],'tree':tree,
                    'payload_hash':_payload_hash(tree,arrays)}
                np.savez_compressed(paths[0],metadata=json.dumps(metadata),**arrays)
                with self.assertRaisesRegex(ValueError,'Invalid float'):
                    _review_l0(predictions/'.cache',report,(2,8,3))

    def test_esrgan_blind_views_identify_method_and_reproduce_saved_pixels(self):
        from gpt_animeimage_denoise.filters import blend,gradient_preserving_candidate
        predictions=self._make_float_review_fixture()
        candidate=np.where(np.indices((2,8))[1]%2,.6,.1).astype(np.float32)
        candidate=np.repeat(candidate[...,None],3,axis=2)
        self._declare_candidate_cache(predictions,candidate=candidate)
        for case in self.document['cases']:
            case['rois'][0]['name']='skin';(self.root/case['id']).mkdir()
        self.save_review()
        result=render_denoise_review(self.review,predictions)
        keys=json.loads((self.root/'blind_key.json').read_text())
        self.assertIn('ESRGAN',result['method'])
        for entry in result['cases']:
            self.assertEqual(entry['denoiser'],'esrgan-l0')
            self.assertEqual(entry['raw_candidate'],'esrgan_l0')
            self.assertTrue(entry['float_candidate_reproduced_saved_result'])
            self.assertEqual(set(keys[entry['id']].values()),
                {'original','esrgan_l0','restored','half_strength'})
            rgb=np.array(Image.open(self.root/(entry['id']+'.png')))
            pure=candidate;weight=np.full(rgb.shape[:2],.5,np.float32)
            restored=gradient_preserving_candidate(rgb,pure,1)
            expected={'original':rgb,'esrgan_l0':blend(rgb,pure,weight),
                'restored':blend(rgb,restored,weight),'half_strength':blend(rgb,restored,weight*.5)}
            self.assertEqual(len({view.tobytes() for view in expected.values()}),4)
            board=np.array(Image.open(self.root/entry['comparisons'][0]['image']))
            for column,letter in enumerate('ABCD'):
                self.assertTrue(np.array_equal(board[43:45,column*14:column*14+2],
                    expected[keys[entry['id']][letter]][0:2,3:5]))

    def test_explicit_l0_review_keeps_pure_l0_blind_method(self):
        predictions=self._make_float_review_fixture();self._declare_candidate_cache(predictions,'l0')
        result=render_denoise_review(self.review,predictions)
        keys=json.loads((self.root/'blind_key.json').read_text())
        self.assertTrue(all(c['raw_candidate']=='pure_l0' for c in result['cases']))
        self.assertTrue(all(set(v.values())=={'original','pure_l0','restored','half_strength'} for v in keys.values()))

    def test_all_canonical_methods_use_their_declared_candidate_and_blind_names(self):
        methods={'l0':('pure_l0','pure L0'),'esrgan':('esrgan','ESRGAN'),
            'esrgan-l0':('esrgan_l0','ESRGAN + L0'),'cugan':('cugan','CUGAN'),
            'cugan-l0':('cugan_l0','CUGAN + L0'),'apisr':('apisr','APISR')}
        predictions=self._make_float_review_fixture()
        for method,(raw,label) in methods.items():
            with self.subTest(method=method):
                self._declare_candidate_cache(predictions,method,lam=.0027,downscale=.5)
                result=render_denoise_review(self.review,predictions)
                self.assertIn(label,result['method'])
                self.assertNotIn('60%',result['method'])
                keys=json.loads((self.root/'blind_key.json').read_text())
                self.assertTrue(all(c['denoiser']==method and c['raw_candidate']==raw for c in result['cases']))
                self.assertTrue(all(set(v.values())=={'original',raw,'restored','half_strength'} for v in keys.values()))
                self.assertTrue(all(c['float_candidate_reproduced_saved_result'] for c in result['cases']))
                (self.root/'denoise_review.json').unlink();(self.root/'blind_key.json').unlink()

    def test_pure_sr_identity_has_no_lambda_or_l0_implementation(self):
        from gpt_animeimage_denoise.fusion_evaluation import _review_candidate
        predictions=self._make_float_review_fixture()
        for method in ['esrgan','cugan','apisr']:
            with self.subTest(method=method):
                self._declare_candidate_cache(predictions,method,lam=.0031)
                report=json.loads((predictions/'tune/report.json').read_text())
                self.assertIsNone(report['candidate_cache_identity']['lambda'])
                self.assertIsNone(report['parameters']['l0_implementation'])
                value=_review_candidate(predictions/'.cache',report,(2,8,3))
                self.assertTrue(np.array_equal(value,np.full((2,8,3),.2,np.float32)))
                report['candidate_cache_identity']['lambda']=.0031
                with self.assertRaisesRegex(ValueError,'identity differs'):
                    _review_candidate(predictions/'.cache',report,(2,8,3))

    def test_all_non_l0_methods_require_candidate_stage(self):
        from gpt_animeimage_denoise.fusion_evaluation import _review_candidate
        predictions=self._make_float_review_fixture()
        for method in ['esrgan','esrgan-l0','cugan','cugan-l0','apisr']:
            with self.subTest(method=method):
                self._declare_candidate_cache(predictions,method)
                report=json.loads((predictions/'tune/report.json').read_text())
                report['candidate_cache_stage']='l0'
                with self.assertRaisesRegex(ValueError,'candidate cache stage'):
                    _review_candidate(predictions/'.cache',report,(2,8,3))

    def test_blind_key_cannot_alias_prediction_result(self):
        predictions=self._make_float_review_fixture();source=predictions/'tune/result.png'
        before=source.read_bytes();os.link(source,self.root/'blind_key.json')
        with self.assertRaises(ValueError):render_denoise_review(self.review,predictions)
        self.assertEqual(source.read_bytes(),before)

    def test_roi_board_cannot_alias_prediction_result(self):
        predictions=self._make_float_review_fixture();self.document['cases'][0]['rois'][0]['name']='skin';self.save_review()
        source=predictions/'tune/result.png';before=source.read_bytes();folder=self.root/'tune';folder.mkdir()
        os.link(source,folder/'denoise_roi_00.png')
        with self.assertRaises(ValueError):render_denoise_review(self.review,predictions)
        self.assertEqual(source.read_bytes(),before)

    def test_prepare_review_rejects_escaped_case_id(self):
        from gpt_animeimage_denoise.fusion_evaluation import prepare_review
        inventory=json.loads(self.manifest.read_text());inventory['items'][0]['id']='../escaped'
        self.manifest.write_text(json.dumps(inventory))
        with self.assertRaises(ValueError):
            prepare_review(self.manifest,self.root/'no_predictions',self.root/'review_output',1)
        self.assertFalse((self.root/'escaped').exists())


if __name__ == '__main__':
    unittest.main()
