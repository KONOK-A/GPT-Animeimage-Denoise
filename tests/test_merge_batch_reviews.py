"""Append review cohorts without relabeling the original producer."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


class MergeBatchReviews(unittest.TestCase):
    def setUp(self):
        try:self.module=importlib.import_module('tools.batch_acceptance.merge_reviews')
        except ModuleNotFoundError:self.fail('Safe acceptance cohort merger is not implemented')
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.parent=Path(temporary.name)/'run';self.extension=self.parent/'extensions/added-3'
        self.base=self.create_run(self.parent,range(1,7),'original-run')
        self.extra=self.create_run(self.extension,range(7,10),'added-3')
        config=json.loads((self.extension/'log/run-config.json').read_text(encoding='utf-8'))
        config.update(parent_run_id='original-run',parent_run_config_sha256=sha(self.parent/'log/run-config.json'),append_driver_sha256='e'*64)
        dump(self.extension/'log/run-config.json',config)
        self.original_json=(self.parent/'manifest.json').read_bytes()
        self.original_js=(self.parent/'manifest.js').read_bytes()

    def create_run(self,root,numbers,run_id):
        parameters={'profile':'fusion','methods':['l0','esrgan-l0'],'lambda':.0005,'downscale':.6,'sigma_at_2048':12.,'strength':1.}
        policy={'semantic_repair':'anchored','outfit_reference':'m2f-white','see_reference':'selective'}
        cases=[];items=[];validation=[]
        for n in numbers:
            cid='batch_'+str(n).zfill(4);source=root/'inputs'/(cid+'.png');source.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(3,4),(n,80,120)).save(source);source_sha=sha(source)
            asset=root/'assets'/cid;asset.mkdir(parents=True)
            Image.new('RGB',(3,4),(n,81,121)).save(asset/'preview.png')
            methods={};runs={}
            for method in ('l0','esrgan-l0'):
                native=root/'native'/(cid+'--'+method);native.mkdir(parents=True)
                Image.new('RGB',(3,4),(n,81,121)).save(native/'result.png')
                fingerprint=hashlib.sha256((cid+method).encode()).hexdigest()
                report={'id':native.name,'status':'complete','profile':'fusion','source_sha256':source_sha,
                    'source':str(source),'size':[3,4],'fingerprint':fingerprint,
                    'parameters':{**parameters,'denoiser':method,'models':'same-models','policy':policy},
                    'output_hashes':{'result.png':sha(native/'result.png')}}
                dump(native/'report.json',report);dump(asset/(method+'-report.json'),report)
                methods[method]={'final':f'assets/{cid}/preview.png','candidate':f'assets/{cid}/preview.png',
                    'restored':f'assets/{cid}/preview.png','report':f'assets/{cid}/{method}-report.json',
                    'report_fingerprint':fingerprint,'parameters':{'model':{'path':'E:/models/weights.pth'}}}
                runs[method]={'status':'complete','report':f'native/{cid}--{method}/report.json','fingerprint':fingerprint}
            checks={'same_masks':True,'same_weight':True,'source_unchanged':True,'protected_unchanged':True,'float_replay_verified':True}
            case={'id':cid,'title':cid,'source_sha256':source_sha,'width':3,'height':4,
                'original':f'assets/{cid}/preview.png','thumbnail':f'assets/{cid}/preview.png','methods':methods,
                'shared':{'weight':f'assets/{cid}/preview.png','groups':{'skin':f'assets/{cid}/preview.png'},
                    'evidence':{'white':{'see':{'groups_colored':f'assets/{cid}/preview.png','groups':{'hair':f'assets/{cid}/preview.png'}}}},
                    'source_legend':[{'code':25,'name':'outfit_reference','color':'#123456'}]},
                'differences':{'1':f'assets/{cid}/preview.png','8':{'path':f'assets/{cid}/preview.png','gain':8}},
                'trace':[{'stage':'checks','links':[{'label':'config','path':'log/run-config.json'}]}],
                'checks':checks,'note':'do not prefix E:/metadata/model.pth'}
            cases.append(case);items.append({'id':cid,'path':str(source),'sha256':source_sha})
            validation.append({'id':cid,'methods':runs,'checks':checks})
        input_doc={'schema_version':1,'run_id':run_id,'items':items};dump(root/'input-manifest.json',input_doc)
        config={'run_id':run_id,'input_manifest_sha256':sha(root/'input-manifest.json'),
            'parameters':parameters,'policy':policy,'model_signature':'same-models','model_hashes':{'model.pth':'b'*64},
            'code_hashes':{'producer.py':'a'*64}}
        dump(root/'log/run-config.json',config)
        dump(root/'validation.json',{'case_count':len(cases),'method_runs':len(cases)*2,'cases':validation})
        dump(root/'run-manifest.json',{'run_id':run_id,'cases':validation})
        dump(root/'export-validation.json',{'cases':len(cases),'same_masks_and_weight':True,'exact_float_replay':True,'source_preserved':True,'user_quality_accepted':False})
        manifest={'schema_version':1,'kind':'fixed-mask-denoise-batch-acceptance','run_id':run_id,
            'parameters':parameters,'summary':{'cases':len(cases),'methods':['l0','esrgan-l0'],'same_selection_for_both_methods':True,'engineering_checks_passed':True,'user_quality_accepted':False},
            'modules':[{'id':'skin','name':'皮肤'}],'cases':cases}
        dump(root/'manifest.json',manifest)
        (root/'manifest.js').write_text('window.BATCH_ACCEPTANCE_DATA = '+json.dumps(manifest,ensure_ascii=False,indent=2)+';\n',encoding='utf-8')
        return manifest

    def merge(self):return self.module.merge_reviews(self.parent,self.extension)

    def test_single_image_batch_can_merge_a_separate_extension(self):
        parent=self.parent.parent/'one-image'
        extension=parent/'extensions/extra'
        initial=self.create_run(parent,[1],'one-image')
        added=self.create_run(extension,[2],'extra')
        before=(parent/'manifest.json').read_bytes()
        result=self.module.merge_reviews(parent,extension)
        self.assertEqual([item['id'] for item in result['cases']],['batch_0001','batch_0002'])
        self.assertEqual(result['cases'][0],initial['cases'][0])
        self.assertEqual(result['cases'][1]['source_sha256'],added['cases'][0]['source_sha256'])
        self.assertEqual((parent/result['aggregation']['original_review']).read_bytes(),before)
        self.assertEqual(self.module.merge_reviews(parent,extension),result)

    def test_existing_legacy_snapshot_remains_the_original_review(self):
        (self.parent/'manifest.original-6.json').write_bytes(self.original_json)
        (self.parent/'manifest.original-6.js').write_bytes(self.original_js)
        result=self.merge()
        self.assertEqual(result['aggregation']['original_review'],'manifest.original-6.json')
        self.assertFalse((self.parent/'manifest.original.json').exists())
        self.assertEqual(self.merge(),result)

    def test_merge_preserves_initial_cases_and_producer_metadata_and_rebases_only_paths(self):
        before={str(p):p.read_bytes() for root in (self.parent,self.extension)
            for p in [root/'input-manifest.json',root/'log/run-config.json',root/'validation.json',root/'run-manifest.json']}
        result=self.merge()
        self.assertEqual(result['run_id'],self.base['run_id']);self.assertEqual(result['summary']['cases'],9)
        self.assertEqual(result['cases'][:6],self.base['cases'])
        added=result['cases'][6]
        self.assertEqual(added['original'],'extensions/added-3/'+self.extra['cases'][0]['original'])
        self.assertEqual(added['shared']['evidence']['white']['see']['groups']['hair'],
            'extensions/added-3/'+self.extra['cases'][0]['shared']['evidence']['white']['see']['groups']['hair'])
        self.assertEqual(added['differences']['8']['path'],'extensions/added-3/'+self.extra['cases'][0]['differences']['8']['path'])
        self.assertEqual(added['methods']['l0']['parameters']['model']['path'],'E:/models/weights.pth')
        self.assertEqual(added['trace'][0]['links'][0]['path'],'extensions/added-3/log/run-config.json')
        for path,payload in before.items():self.assertEqual(Path(path).read_bytes(),payload)
        self.assertEqual((self.parent/'manifest.original.json').read_bytes(),self.original_json)
        self.assertEqual((self.parent/'manifest.original.js').read_bytes(),self.original_js)
        aggregate=json.loads((self.parent/'aggregate-validation.json').read_text(encoding='utf-8'))
        self.assertTrue(aggregate['complete']);self.assertEqual(aggregate['case_count'],9)
        self.assertFalse(aggregate['user_quality_accepted'])

    def test_previous_review_descriptor_binds_initial_order_source_and_both_fingerprints(self):
        result=self.merge();previous=result['previous_reviews'][0]
        self.assertEqual(previous['run_id'],self.base['run_id'])
        self.assertEqual(previous['storage_key'],'batch-acceptance-v1:'+self.base['run_id']+':'+':'.join(c['source_sha256'] for c in self.base['cases']))
        self.assertEqual([c['id'] for c in previous['cases']],[c['id'] for c in self.base['cases']])
        self.assertEqual(previous['cases'][0]['report_fingerprints']['l0'],self.base['cases'][0]['methods']['l0']['report_fingerprint'])

    def test_repeat_merge_is_idempotent_and_does_not_prefix_twice(self):
        first=self.merge();files={name:(self.parent/name).read_bytes() for name in ['manifest.json','manifest.js','manifest.original.json','manifest.original.js','aggregate-manifest.json','aggregate-validation.json']}
        second=self.merge();self.assertEqual(first,second)
        for name,payload in files.items():self.assertEqual((self.parent/name).read_bytes(),payload)

    def test_parameter_code_model_or_policy_mismatch_is_rejected_before_writes(self):
        path=self.extension/'log/run-config.json';original=path.read_text(encoding='utf-8')
        for key,value in [('parameters',dict(self.base['parameters'],**{'lambda':.001})),
                          ('code_hashes',{'producer.py':'d'*64}),('model_signature','different'),
                          ('policy',{'see_reference':'legacy'})]:
            with self.subTest(key=key):
                document=json.loads(original);document[key]=value;dump(path,document)
                with self.assertRaises(ValueError):self.merge()
                self.assertEqual((self.parent/'manifest.json').read_bytes(),self.original_json)
                self.assertFalse((self.parent/'manifest.original.json').exists())
        path.write_text(original,encoding='utf-8')

    def test_duplicate_case_ids_are_not_silently_overwritten(self):
        path=self.extension/'manifest.json';doc=json.loads(path.read_text(encoding='utf-8'))
        doc['cases'][0]['id']=self.base['cases'][0]['id'];dump(path,doc)
        with self.assertRaises(ValueError):self.merge()
        self.assertEqual((self.parent/'manifest.json').read_bytes(),self.original_json)

    def test_extension_cannot_escape_its_component_or_parent_resource_root(self):
        path=self.extension/'manifest.json';doc=json.loads(path.read_text(encoding='utf-8'))
        doc['cases'][0]['original']='../../assets/'+self.base['cases'][0]['id']+'/preview.png';dump(path,doc)
        with self.assertRaises(ValueError):self.merge()
        self.assertEqual((self.parent/'manifest.json').read_bytes(),self.original_json)

    def test_native_result_tampering_rejects_stale_verified_summary(self):
        p=self.extension/'native/batch_0007--l0/result.png';p.write_bytes(b'changed result')
        with self.assertRaises(ValueError):self.merge()
        self.assertEqual((self.parent/'manifest.json').read_bytes(),self.original_json)

    def test_atomic_temp_alias_cannot_overwrite_original_input_manifest(self):
        original=(self.parent/'input-manifest.json').read_bytes()
        os.link(self.parent/'input-manifest.json',self.parent/'manifest.json.tmp')
        with self.assertRaises(ValueError):self.merge()
        self.assertEqual((self.parent/'input-manifest.json').read_bytes(),original)

    def test_viewer_script_and_json_must_describe_the_same_source_cohort(self):
        (self.parent/'manifest.js').write_text('window.BATCH_ACCEPTANCE_DATA = {};\n',encoding='utf-8')
        with self.assertRaises(ValueError):self.merge()
        self.assertEqual((self.parent/'manifest.json').read_bytes(),self.original_json)

    def test_backup_metadata_cannot_be_silently_replaced_by_a_changed_six_case_view(self):
        self.merge()
        current=dict(self.base);current['notes']=['Changed original metadata']
        dump(self.parent/'manifest.json',current)
        (self.parent/'manifest.js').write_text('window.BATCH_ACCEPTANCE_DATA = '+json.dumps(current)+';\n',encoding='utf-8')
        before=(self.parent/'manifest.json').read_bytes()
        with self.assertRaises(ValueError):self.merge()
        self.assertEqual((self.parent/'manifest.json').read_bytes(),before)

    def test_aggregate_explains_the_two_producers_without_rewriting_case_traces(self):
        merged=self.merge();meta=merged['aggregation']
        self.assertEqual(meta['initial_cases'],6);self.assertEqual(meta['added_cases'],3)
        self.assertEqual(meta['validation'],'aggregate-validation.json')
        self.assertEqual(meta['extension_manifest'],'extensions/added-3/manifest.json')
        self.assertEqual(meta['extension_validation'],'extensions/added-3/validation.json')
        self.assertEqual(merged['cases'][0]['trace'],self.base['cases'][0]['trace'])

    def test_frozen_exporter_folder_alias_is_bound_to_actual_producer_without_rewriting_it(self):
        manifest=self.extension/'input-manifest.json';source=json.loads(manifest.read_text(encoding='utf-8'))
        source.pop('run_id');dump(manifest,source)
        config_path=self.extension/'log/run-config.json';config=json.loads(config_path.read_text(encoding='utf-8'))
        config['run_id']='original-run--added-3';config['input_manifest_sha256']=sha(manifest);dump(config_path,config)
        run_path=self.extension/'run-manifest.json';run=json.loads(run_path.read_text(encoding='utf-8'))
        run['run_id']=config['run_id'];dump(run_path,run)
        before={str(p):p.read_bytes() for p in (manifest,config_path,run_path,self.extension/'manifest.json')}
        self.merge();aggregate=json.loads((self.parent/'aggregate-manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(aggregate['components'][1]['producer_run_id'],'original-run--added-3')
        self.assertEqual(aggregate['components'][1]['viewer_run_id'],'added-3')
        for p,payload in before.items():self.assertEqual(Path(p).read_bytes(),payload)


if __name__=='__main__':unittest.main()
