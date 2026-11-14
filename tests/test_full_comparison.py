"""Truthful model/fixed-fusion assets for the full local comparison page."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


class FullComparisonAssets(unittest.TestCase):
    def setUp(self):
        try:self.module=importlib.import_module('tools.segmentation_review.build_full_comparison')
        except ModuleNotFoundError:self.fail('Full comparison exporter is missing')
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.source=self.root/'source.png';self.native=self.root/'native'
        self.native.mkdir();self.output=self.root/'output'
        Image.fromarray(np.arange(72,dtype=np.uint8).reshape(4,6,3)).save(self.source)
        from gpt_animeimage_denoise.pipeline import _sha
        self.case={'id':'sample','path':str(self.source),'sha256':_sha(self.source),'title':'服饰样本'}
        labels=np.full((4,6),6,np.uint8)
        labels[0,:]=[0,255,3,2,4,5];labels[1,0]=1
        objects=labels==0;unknown=labels==255;conflict=np.zeros_like(objects);conflict[2,2]=True
        sources=np.full(labels.shape,1,np.uint8);sources[objects]=23;sources[unknown]=6
        images={'person_labels.png':labels,'objects.png':objects.astype(np.uint8)*255,
                'unknown.png':unknown.astype(np.uint8)*255,'conflict.png':conflict.astype(np.uint8)*255,
                'source_map.png':sources}
        for number,name in enumerate(('skin','face','hair','eyes','mouth','outfit'),1):
            images['module_'+name+'.png']=(labels==number).astype(np.uint8)*255
        for name,array in images.items():Image.fromarray(array).save(self.native/name)
        (self.native/'regions.json').write_text(json.dumps({'semantic_repairs':{'strategy':'anchored'}}),encoding='utf-8')
        self.report={'id':'sample','status':'complete','profile':'fusion','source_sha256':self.case['sha256'],
            'size':[6,4],'fingerprint':'fixed-fusion-fingerprint','parameters':{'models':'model-identity','policy':{'semantic_repair':'anchored','see':1}},
            'provenance':{'model_hashes':{'see-weight':'identity'}},
            'output_hashes':{name:_sha(self.native/name) for name in [*images,'regions.json']}}
        (self.native/'report.json').write_text(json.dumps(self.report),encoding='utf-8')

    def evidence(self):
        values=np.full((12,2,3),-8,np.float32);values[6]=0
        see=np.full((19,4,6),-4,np.float32);see[11,:,2:]=3;see[18,0,3]=4
        return {'alpha':np.ones((4,6),np.float32),'see_logits':see,'see_original_logits':see.copy(),
            'm2f_white':values,'m2f_original':values.copy(),
            'reference_labels12':np.full((4,6),6,np.uint8),'provenance':{}}

    def test_score_first_projection_differs_from_nearest_without_changing_grid(self):
        scores=np.full((12,1,2),.0001,np.float32)
        scores[1,0]=[.51,.001];scores[3,0]=[.489,.998]
        scores/=scores.sum(axis=0,keepdims=True);scores=np.log(scores)
        nearest=self.module.project_m2f(scores,(1,4),'nearest')
        projected=self.module.project_m2f(scores,(1,4),'score')
        self.assertEqual(nearest.dtype,np.uint8);self.assertEqual(projected.shape,(1,4))
        self.assertEqual(nearest[0,1],1);self.assertEqual(projected[0,1],3)

    def test_fusion_loader_reads_actual_labels_and_checks_export_hashes(self):
        actual=self.module.load_fusion_result(self.native,self.case['sha256'],(4,6))
        self.assertEqual(actual['report']['fingerprint'],'fixed-fusion-fingerprint')
        self.assertTrue(actual['unknown'][0,1]);self.assertTrue(actual['objects'][0,0])
        Image.fromarray(np.zeros((4,6),np.uint8)).save(self.native/'person_labels.png')
        with self.assertRaises(ValueError):self.module.load_fusion_result(self.native,self.case['sha256'],(4,6))

    def test_fusion_loader_rejects_modules_not_derived_from_final_labels(self):
        from gpt_animeimage_denoise.pipeline import _sha
        Image.fromarray(np.zeros((4,6),np.uint8)).save(self.native/'module_outfit.png')
        self.report['output_hashes']['module_outfit.png']=_sha(self.native/'module_outfit.png')
        (self.native/'report.json').write_text(json.dumps(self.report),encoding='utf-8')
        with self.assertRaises(ValueError):self.module.load_fusion_result(self.native,self.case['sha256'],(4,6))

    def test_export_keeps_independent_raw_channels_and_fixed_fusion(self):
        ev=self.evidence();before=self.source.read_bytes()
        native=self.module.load_fusion_result(self.native,self.case['sha256'],(4,6))
        entry=self.module.export_full_case(self.case,self.output,ev,native)
        self.assertEqual(entry['fusion']['report_fingerprint'],'fixed-fusion-fingerprint')
        for view in ('white','original'):
            for projection in ('score','nearest'):
                self.assertEqual(len(entry['views'][view]['m2f']['projections'][projection]['masks']),12)
            for threshold in ('0','1','2'):
                self.assertEqual(len(entry['views'][view]['see']['thresholds'][threshold]['masks']),19)
        with Image.open(self.output/entry['fusion']['groups']['outfit']) as im:
            self.assertEqual(im.mode,'RGBA');alpha=np.asarray(im.getchannel('A'))
            self.assertEqual(alpha[0,1],0);self.assertEqual(alpha[2,2],255)
        with Image.open(self.output/entry['views']['white']['see']['thresholds']['1']['groups']['outfit']) as im:
            self.assertEqual(np.asarray(im.getchannel('A'))[0,3],255) # Objects do not subtract raw wear.
        self.assertEqual(self.source.read_bytes(),before)
        document=self.module.write_manifest(self.output,[entry])
        self.assertEqual(document['defaults']['part'],'outfit')
        self.assertIn('SEG_FULL_COMPARISON_DATA',(self.output/'manifest.js').read_text(encoding='utf-8'))

    def test_new_output_aliases_are_preflighted_before_original_copy(self):
        native=self.module.load_fusion_result(self.native,self.case['sha256'],(4,6))
        target=self.output/'assets/sample/fusion-outfit.png';target.parent.mkdir(parents=True)
        os.link(self.source,target);before=self.source.read_bytes()
        with self.assertRaises(ValueError):self.module.export_full_case(self.case,self.output,self.evidence(),native)
        self.assertEqual(self.source.read_bytes(),before)
        self.assertFalse((target.parent/'original.png').exists())

    def test_unknown_and_object_regions_stay_outside_fusion_all(self):
        native=self.module.load_fusion_result(self.native,self.case['sha256'],(4,6))
        groups=self.module.fusion_groups(native['labels'])
        self.assertFalse(groups['all'][0,0] or groups['all'][0,1])
        self.assertTrue(groups['outfit'][2,2]) # Resolved conflict remains an actual label.

    def test_cache_index_requires_current_code_and_both_model_input_views(self):
        from unittest import mock
        from gpt_animeimage_denoise.pipeline import _digest
        cache=self.root/'cache';cache.mkdir()
        wanted=None
        for number,(code,views) in enumerate((('current',['white','original']),
                                             ('old',['white','original']),
                                             ('current',['white']))):
            identity={'code':code,'dual_view':True,'see_views':views,
                'source_sha256':self.case['sha256'],'models':'model-identity'}
            path=cache/('evidence-'+_digest(identity)+'.npz')
            np.savez_compressed(path,metadata=json.dumps({'identity':identity}))
            if number==0:wanted=path
        with mock.patch.object(self.module,'_code_hash',return_value='current'):
            indexed=self.module.index_evidence(cache)
        self.assertEqual(indexed,{(self.case['sha256'],'model-identity'):[wanted]})

    def test_unsupported_projection_and_nonfinite_scores_are_rejected(self):
        scores=np.zeros((12,2,3),np.float32)
        with self.assertRaises(ValueError):self.module.project_m2f(scores,(4,6),'invented')
        scores[0,0,0]=np.nan
        with self.assertRaises(ValueError):self.module.project_m2f(scores,(4,6),'score')


if __name__=='__main__':unittest.main()
