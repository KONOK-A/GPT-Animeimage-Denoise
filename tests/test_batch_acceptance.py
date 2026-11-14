"""Acceptance exports must show actual candidates and a fair, fixed-mask pair."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from gpt_animeimage_denoise.filters import blend,gradient_preserving_candidate
from gpt_animeimage_denoise.fusion_pipeline import _pack,_payload_hash,OUTPUT_FILES
from gpt_animeimage_denoise.pipeline import _digest,_save_rgb,_sha


class AcceptanceExport(unittest.TestCase):
    def setUp(self):
        try:self.module=importlib.import_module('tools.batch_acceptance.build_review')
        except ModuleNotFoundError:self.fail('Batch acceptance export module is missing')
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.run=self.root/'run';self.native=self.run/'native'
        self.native.mkdir(parents=True);self.out=self.root/'review'
        self.rgb=np.random.default_rng(42).integers(80,160,(8,10,3),np.uint8)
        self.alpha=np.full((8,10),255,np.uint8);self.alpha[1,1]=0
        self.icc=b'test-profile';self.source=self.root/'source.png'
        _save_rgb(self.source,self.rgb,self.alpha,self.icc)
        self.item={'id':'batch_0001','path':str(self.source),'sha256':_sha(self.source)}
        self.weight=np.full((8,10),.6,np.float32);self.weight[:2,:3]=0
        labels=np.ones((8,10),np.uint8);labels[0,0]=255;labels[0,1]=6;labels[0,2]=0
        objects=labels==0;unknown=labels==255;protected=~np.isin(labels,[1,2,3])|(self.alpha==0)
        self.inputs={}
        for method,factor in [('l0',.98),('esrgan-l0',.9)]:
            folder=self.native/(self.item['id']+'--'+method);folder.mkdir()
            candidate=(self.rgb.astype(np.float32)/255*factor+.00002).clip(0,1)
            restored=gradient_preserving_candidate(self.rgb,candidate,.8)
            final=blend(self.rgb,restored,self.weight)
            _save_rgb(folder/'original.png',self.rgb,self.alpha,self.icc)
            _save_rgb(folder/'result.png',final,self.alpha,self.icc)
            np.save(folder/'weight.npy',self.weight,allow_pickle=False)
            Image.fromarray(np.ceil(self.weight*255).astype(np.uint8)).save(folder/'weight.png')
            for name,array in [('person_labels',labels),('raw_labels',labels),('source_map',np.ones_like(labels))]:
                Image.fromarray(array).save(folder/(name+'.png'))
            for name,mask in [('objects',objects),('suspected_objects',np.zeros_like(objects)),
                              ('unknown',unknown),('protected',protected),
                              ('conflict',unknown),('consensus',~unknown)]:
                Image.fromarray(mask.astype(np.uint8)*255).save(folder/(name+'.png'))
            for number,part in enumerate(('skin','face','hair','eyes','mouth','outfit'),1):
                Image.fromarray((labels==number).astype(np.uint8)*255).save(folder/('module_'+part+'.png'))
            Image.fromarray(final).save(folder/'preview.png')
            (folder/'regions.json').write_text(json.dumps({'semantic_repairs':{'evidence_roles':{}},'sam_records':[]}),encoding='utf-8')
            config={'method':method,'device':'cpu'}
            ident={'source_sha256':self.item['sha256'],'lambda':.0005,'device':'cpu','denoiser':method,
                   'denoiser_config':config,'code':'recorded-code','callable':{'kind':'installed_l0'},'precision':'float32-cache'}
            stage='l0' if method=='l0' else 'candidate'
            report={'id':folder.name,'folder':folder.name,'status':'complete','profile':'fusion',
                'source':str(self.source),'source_sha256':self.item['sha256'],'size':[10,8],
                'fingerprint':'fingerprint-'+method,'effective_sigma':.8,
                'parameters':{'profile':'fusion','lambda':.0005,'sigma_at_2048':12.,'strength':1.,'device':'cpu',
                    'models':'same-models','implementation':'same-source','policy':{'see_reference':'selective'},
                    'l0_implementation':{'kind':'installed_l0'},'denoiser':method,'denoiser_config':config},
                'candidate_cache_identity':ident,'candidate_cache_stage':stage,'provenance':{'signature':'same-models'},
                'output_hashes':{name:_sha(folder/name) for name in OUTPUT_FILES},'source_unchanged':True}
            cache=self.native/'.cache';cache.mkdir(exist_ok=True)
            self.cache_value(cache,stage,ident,candidate)
            if method=='esrgan-l0':
                sr_identity={'source_sha256':self.item['sha256'],'device':'cpu',
                    'denoiser_config':{k:v for k,v in config.items() if k!='method'},
                    'precision':'uint8-native-sr-cache','size':list(self.rgb.shape)}
                report['sr_cache_identity']=sr_identity
                self.cache_value(cache,'sr',sr_identity,self.rgb.copy())
            (folder/'report.json').write_text(json.dumps(report),encoding='utf-8')
            self.inputs[method]=(folder,report,candidate)

    def cache_value(self,root,stage,identity,value):
        arrays={};tree=_pack(value,arrays)
        metadata={'identity':identity,'tree':tree,'payload_hash':_payload_hash(tree,arrays)}
        path=root/(stage+'-'+_digest(identity)+'.npz')
        np.savez_compressed(path,metadata=json.dumps(metadata),**arrays)
        return path

    def loaded(self):
        return {method:self.module.load_method(folder,self.item['sha256'],self.rgb,self.alpha,self.icc,method)
                for method,(folder,_,_) in self.inputs.items()}

    def test_acceptance_still_reads_reports_without_the_suspected_object_layer(self):
        for folder,report,_ in self.inputs.values():
            del report['output_hashes']['suspected_objects.png']
            (folder/'suspected_objects.png').unlink()
            (folder/'report.json').write_text(json.dumps(report),encoding='utf-8')
        self.assertEqual(set(self.loaded()),{'l0','esrgan-l0'})

    def test_exact_float_candidate_and_gaussian_replay_preserve_fractional_values(self):
        folder,report,candidate=self.inputs['esrgan-l0']
        actual=self.module.load_exact_candidate(self.native/'.cache',report,self.rgb.shape)
        np.testing.assert_array_equal(actual,candidate)
        final=np.asarray(Image.open(folder/'result.png').convert('RGB'))
        restored=self.module.replay_final(self.rgb,actual,self.weight,.8,final)
        np.testing.assert_array_equal(restored,gradient_preserving_candidate(self.rgb,candidate,.8))

    def test_missing_float_candidate_is_not_reconstructed_from_png(self):
        _,report,candidate=self.inputs['l0'];stage=report['candidate_cache_stage']
        cache=self.native/'.cache'/(stage+'-'+_digest(report['candidate_cache_identity'])+'.npz')
        cache.unlink();Image.fromarray(np.rint(candidate*255).astype(np.uint8)).save(self.native/'candidate.png')
        with self.assertRaises(ValueError):self.module.load_exact_candidate(self.native/'.cache',report,self.rgb.shape)

    def test_final_replay_rejects_rounded_or_wrong_candidate(self):
        _,_,candidate=self.inputs['l0'];expected=blend(self.rgb,gradient_preserving_candidate(self.rgb,candidate,.8),self.weight)
        expected[4,4]=0
        with self.assertRaises(ValueError):self.module.replay_final(self.rgb,candidate,self.weight,.8,expected)

    def test_pair_rejects_different_actual_weight(self):
        pair=self.loaded();pair['esrgan-l0']['weight']=pair['esrgan-l0']['weight'].copy()
        pair['esrgan-l0']['weight'][4,4]=.2
        with self.assertRaises(ValueError):self.module.validate_pair(pair)

    def test_pair_rejects_different_postprocessing_settings_with_identical_weights(self):
        for key in ('restoration_gain','structure_protection'):
            with self.subTest(key=key):
                pair=self.loaded()
                pair['esrgan-l0']['report']['parameters'][key]=.25
                with self.assertRaisesRegex(ValueError,key):self.module.validate_pair(pair)

    def test_export_replays_the_reported_restoration_gain(self):
        expected={}
        for method,(folder,report,candidate) in self.inputs.items():
            report['parameters'].update(restoration_gain=.25,structure_protection=0.)
            residual=self.rgb.astype(np.float32)/255-candidate
            restored=np.clip(candidate+.25*gaussian_filter(residual,(.8,.8,0),mode='reflect'),0,1)
            expected[method]=np.rint(restored*255).astype(np.uint8)
            result=folder/'result.png'
            _save_rgb(result,blend(self.rgb,restored,self.weight),self.alpha,self.icc)
            report['output_hashes']['result.png']=_sha(result)
            (folder/'report.json').write_text(json.dumps(report),encoding='utf-8')
        entry=self.module.export_case(self.item,self.out,self.loaded())
        for method in expected:
            with Image.open(self.out/entry['methods'][method]['restored']) as image:
                np.testing.assert_array_equal(np.asarray(image.convert('RGB')),expected[method])

    def test_pair_rejects_different_masks_or_segmentation_policy(self):
        pair=self.loaded();pair['esrgan-l0']['report']['parameters']['policy']={'see_reference':'legacy'}
        with self.assertRaises(ValueError):self.module.validate_pair(pair)
        pair=self.loaded();folder=self.inputs['esrgan-l0'][0]
        Image.fromarray(np.zeros((8,10),np.uint8)).save(folder/'objects.png')
        pair['esrgan-l0']['report']['output_hashes']['objects.png']=_sha(folder/'objects.png')
        with self.assertRaises(ValueError):self.module.validate_pair(pair)

    def test_export_preserves_source_alpha_icc_and_records_preview_scope(self):
        before=self.source.read_bytes();entry=self.module.export_case(self.item,self.out,self.loaded())
        self.assertEqual(self.source.read_bytes(),before)
        self.assertTrue(entry['checks']['same_masks']);self.assertTrue(entry['checks']['same_weight'])
        for method in ('l0','esrgan-l0'):
            for stage in ('final','candidate','restored'):
                with Image.open(self.out/entry['methods'][method][stage]) as im:
                    self.assertEqual(im.size,(10,8));self.assertEqual(im.info.get('icc_profile'),self.icc)
                    np.testing.assert_array_equal(np.asarray(im.getchannel('A')),self.alpha)
        self.assertEqual(set(entry['differences']),{'1','4','8'})
        self.assertIn('未',entry['methods']['l0']['candidate_note'])
        with Image.open(self.out/entry['shared']['weight']) as im:
            np.testing.assert_array_equal(np.asarray(im.getchannel('A')),np.ceil(self.weight*255).astype(np.uint8))

    def test_esrgan_single_stage_requires_its_exact_uint8_cache(self):
        _,report,_=self.inputs['esrgan-l0']
        actual=self.module.load_exact_sr(self.native/'.cache',report,self.rgb.shape)
        np.testing.assert_array_equal(actual,self.rgb)
        ident=report['sr_cache_identity'];self.cache_value(self.native/'.cache','sr',ident,self.rgb.astype(np.float32))
        with self.assertRaises(ValueError):self.module.load_exact_sr(self.native/'.cache',report,self.rgb.shape)

    def test_sr_preview_cannot_silently_refer_to_another_network_recipe(self):
        _,report,_=self.inputs['esrgan-l0'];report['sr_cache_identity']['denoiser_config']={'network':'cugan'}
        self.cache_value(self.native/'.cache','sr',report['sr_cache_identity'],self.rgb)
        with self.assertRaises(ValueError):self.module.load_exact_sr(self.native/'.cache',report,self.rgb.shape)

    def test_identically_changed_pair_masks_do_not_bypass_saved_hashes(self):
        pair=self.loaded()
        for method in ('l0','esrgan-l0'):
            Image.fromarray(np.zeros((8,10),np.uint8)).save(pair[method]['folder']/'conflict.png')
        with self.assertRaises(ValueError):self.module.validate_pair(pair)

    def test_nonzero_weight_in_a_native_protected_area_is_rejected(self):
        folder,report,candidate=self.inputs['l0']
        weight=self.weight.copy();weight[0,1]=.5
        np.save(folder/'weight.npy',weight,allow_pickle=False)
        final=blend(self.rgb,gradient_preserving_candidate(self.rgb,candidate,.8),weight)
        _save_rgb(folder/'result.png',final,self.alpha,self.icc)
        for name in ('weight.npy','result.png'):report['output_hashes'][name]=_sha(folder/name)
        (folder/'report.json').write_text(json.dumps(report),encoding='utf-8')
        with self.assertRaises(ValueError):
            self.module.load_method(folder,self.item['sha256'],self.rgb,self.alpha,self.icc,'l0')

    def test_saved_build_weight_allowed_argument_is_exported_as_actual_trace(self):
        trace=self.run/'trace'/self.item['id']/'l0/weight';trace.mkdir(parents=True)
        allowed=np.isin(np.array(Image.open(self.inputs['l0'][0]/'person_labels.png')),[1,2,3])
        allowed[self.alpha==0]=False;allowed[1,0]=False
        Image.fromarray(allowed.astype(np.uint8)*255).save(trace/'allowed.png')
        entry=self.module.export_case(self.item,self.out,self.loaded(),run_root=self.run)
        with Image.open(self.out/entry['shared']['allowed']) as image:
            np.testing.assert_array_equal(np.asarray(image.getchannel('A'))>0,allowed)

    def test_export_alias_to_source_is_rejected_before_writes(self):
        import os
        target=self.out/'assets'/self.item['id']/'original.png';target.parent.mkdir(parents=True)
        os.link(self.source,target);before=self.source.read_bytes()
        with self.assertRaises(ValueError):self.module.export_case(self.item,self.out,self.loaded())
        self.assertEqual(self.source.read_bytes(),before)


if __name__=='__main__':unittest.main()
