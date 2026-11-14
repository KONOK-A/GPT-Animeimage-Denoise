"""Truthful display exports for independent segmentation evidence."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


def evidence(shape=(4, 6)):
    scores=np.full((12, 2, 3),-8,np.float32);scores[3]=0
    return {'alpha':np.ones(shape,np.float32),
            'see_logits':np.full((19,*shape),-4,np.float32),
            'm2f_white':scores.copy(),'m2f_original':scores.copy(),
            'reference_labels12':np.full(shape,3,np.uint8),
            'provenance':{'m2f':{'version':4,'precision':'float32'}}}


class ReviewExports(unittest.TestCase):
    def setUp(self):
        try:self.module=importlib.import_module('tools.segmentation_review.build_assets')
        except ModuleNotFoundError:self.fail('Segmentation review export module is missing')
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.source=self.root/'source.png';self.output=self.root/'output'
        rgb=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3)
        Image.fromarray(rgb).save(self.source)
        from gpt_animeimage_denoise.pipeline import _sha
        self.case={'id':'case','path':str(self.source),'sha256':_sha(self.source),
                   'title':'测试图','description':'独立分割对照'}

    def test_m2f_mapping_preserves_brows_nose_mouth_and_no_eye_fusion(self):
        labels=np.arange(12,dtype=np.uint8)[None]
        groups=self.module.m2f_groups(labels)
        self.assertTrue(groups['hair'][0,6] and groups['hair'][0,7])
        self.assertTrue(groups['face'][0,8]);self.assertFalse(groups['mouth'][0,8])
        self.assertTrue(groups['mouth'][0,9])
        np.testing.assert_array_equal(groups['all'],labels!=0)

    def test_see_thresholds_keep_overlap_and_do_not_invent_bare_skin(self):
        logits=np.full((19,2,3),-4,np.float32)
        logits[0,0,0]=3;logits[11,0,0]=2;logits[18,0,0]=4
        logits[12,0,1]=6;logits[14,0,2]=6;logits[9,1,0]=1
        groups=self.module.see_groups(logits,0)
        self.assertTrue(groups['hair'][0,0] and groups['outfit'][0,0])
        self.assertTrue(groups['all'][0,0])  # Objects do not veto independent model displays.
        self.assertFalse(groups['skin'][0,1]);self.assertFalse(groups['outfit'][0,1])
        self.assertFalse(groups['all'][0,1] or groups['all'][0,2])
        self.assertTrue(groups['skin'][1,0])
        self.assertFalse(self.module.see_groups(logits,1)['skin'][1,0])

    def test_export_has_all_raw_masks_transparent_alpha_and_original_grid(self):
        ev=evidence();ev['see_logits'][3,1,1]=2
        before=self.source.read_bytes()
        entry=self.module.export_case(self.case,self.output,ev,see_original=ev['see_logits'])
        self.assertEqual((entry['width'],entry['height']),(6,4))
        for view in ('white','original'):
            data=entry['views'][view]
            self.assertEqual(len(data['m2f']['masks']),12)
            self.assertEqual(len(data['see']['thresholds']['0']['masks']),19)
            with Image.open(self.output/data['m2f']['groups']['eyes']) as mask:
                self.assertEqual(mask.mode,'RGBA');self.assertEqual(mask.size,(6,4))
                self.assertFalse(np.asarray(mask.getchannel('A')).any())
            with Image.open(self.output/data['see']['thresholds']['0']['groups']['eyes']) as mask:
                alpha=np.asarray(mask.getchannel('A'))
                self.assertEqual(alpha[1,1],255);self.assertEqual(int((alpha>0).sum()),1)
        self.assertEqual(self.source.read_bytes(),before)
        self.module.write_manifest(self.output,[entry])
        manifest=json.loads((self.output/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['cases'][0]['source_sha256'],self.case['sha256'])
        self.assertIn('window.SEG_REVIEW_DATA',(self.output/'manifest.js').read_text(encoding='utf-8'))

    def test_export_applies_exif_orientation_and_preserves_alpha_icc(self):
        rgba=np.full((6,4,4),123,np.uint8);rgba[:,:,3]=255;rgba[0,0,3]=0
        image=Image.fromarray(rgba);exif=image.getexif();exif[274]=6
        image.save(self.source,exif=exif,icc_profile=b'test-icc')
        from gpt_animeimage_denoise.pipeline import _sha
        self.case['sha256']=_sha(self.source)
        entry=self.module.export_case(self.case,self.output,evidence((4,6)))
        with Image.open(self.output/entry['original']) as exported:
            self.assertEqual(exported.size,(6,4));self.assertEqual(exported.info['icc_profile'],b'test-icc')
            self.assertEqual(int((np.asarray(exported.getchannel('A'))==0).sum()),1)

    def test_output_alias_is_rejected_before_any_source_write(self):
        target=self.output/'assets/case/original.png';target.parent.mkdir(parents=True)
        os.link(self.source,target);before=self.source.read_bytes()
        with self.assertRaises(ValueError):self.module.export_case(self.case,self.output,evidence())
        self.assertEqual(self.source.read_bytes(),before)

    def test_wrong_hash_or_evidence_grid_is_rejected(self):
        wrong=dict(self.case,sha256='0'*64)
        with self.assertRaises(ValueError):self.module.export_case(wrong,self.output,evidence())
        with self.assertRaises(ValueError):self.module.export_case(self.case,self.output,evidence((5,6)))

    def test_original_see_cache_checks_payload_and_checkpoint(self):
        import hashlib
        logits=np.zeros((19,4,6),np.float32)
        metadata={'source_sha256':self.case['sha256'],'checkpoint_sha256':'checkpoint',
                  'preprocessing':'oriented-original-white-alpha','precision':'bfloat16',
                  'payload_sha256':hashlib.sha256(logits.tobytes()).hexdigest()}
        path=self.root/'see.npz'
        np.savez_compressed(path,see_logits=logits,metadata=json.dumps(metadata))
        actual,_=self.module.load_see_original(path,self.case['sha256'],(4,6),expected_checkpoint_sha256='checkpoint')
        np.testing.assert_array_equal(actual,logits)
        with self.assertRaises(ValueError):
            self.module.load_see_original(path,self.case['sha256'],(4,6),expected_checkpoint_sha256='other')
        logits[0,0,0]=2
        np.savez_compressed(path,see_logits=logits,metadata=json.dumps(metadata))
        with self.assertRaises(ValueError):self.module.load_see_original(path,self.case['sha256'],(4,6))


if __name__=='__main__':unittest.main()
