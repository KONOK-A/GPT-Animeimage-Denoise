"""Float-residual ablations must isolate restoration and structural weights."""
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from gpt_animeimage_denoise.filters import blend,build_weight,gradient_preserving_candidate
from gpt_animeimage_denoise.pipeline import _save_rgb


class DenoiseAblation(unittest.TestCase):
    def setUp(self):
        try:self.module=importlib.import_module('tools.batch_acceptance.denoise_ablation')
        except ModuleNotFoundError:self.fail('Bounded numerical denoise ablation is missing')
        self.rgb=np.random.default_rng(9).integers(70,180,(10,12,3),np.uint8)
        self.candidate=np.clip(self.rgb.astype(np.float32)/255*.7+.12,0,1)
        self.allowed=np.zeros((10,12),bool);self.allowed[1:9,1:11]=True
        self.protected=np.zeros((10,12),bool);self.protected[4:6,5:7]=True

    def test_quarter_restoration_uses_unclipped_float_residual_not_full_restoration_png(self):
        rgb=np.full((5,9,3),204,np.uint8)
        candidate=np.zeros(rgb.shape,np.float32);candidate[:,4]=.92
        actual=self.module.restore_candidate(rgb,candidate,1.2,.25)
        full=gradient_preserving_candidate(rgb,candidate,1.2)
        wrong=candidate+.25*(full-candidate)
        self.assertGreater(float(actual[2,4,0]),.99)
        self.assertGreater(float(actual[2,4,0]-wrong[2,4,0]),.03)

    def test_gamma_zero_is_raw_float_and_gamma_one_matches_existing_restoration(self):
        before_rgb=self.rgb.copy();before_candidate=self.candidate.copy()
        zero=self.module.restore_candidate(self.rgb,self.candidate,.9,0)
        one=self.module.restore_candidate(self.rgb,self.candidate,.9,1)
        np.testing.assert_array_equal(zero,self.candidate)
        np.testing.assert_array_equal(one,gradient_preserving_candidate(self.rgb,self.candidate,.9))
        np.testing.assert_array_equal(self.rgb,before_rgb);np.testing.assert_array_equal(self.candidate,before_candidate)
        zero[:]=0;np.testing.assert_array_equal(self.candidate,before_candidate)

    def test_invalid_gain_and_sigma_are_rejected(self):
        for gain in (-.1,1.1,np.nan,np.inf,True):
            with self.subTest(gain=gain),self.assertRaises(ValueError):
                self.module.restore_candidate(self.rgb,self.candidate,1,gain)
        for sigma in (0,-1,np.nan,np.inf):
            with self.subTest(sigma=sigma),self.assertRaises(ValueError):
                self.module.restore_candidate(self.rgb,self.candidate,sigma,.25)

    def test_soft_weight_uses_actual_allowed_and_keeps_strength_threshold_and_hard_protection(self):
        current=build_weight(self.allowed,self.protected,self.rgb,feather_radius=2,structure_radius=3,gradient_threshold=4)
        before_allowed=self.allowed.copy();before_protected=self.protected.copy()
        actual=self.module.make_soft_weight(self.rgb,self.allowed,self.protected,2,.2)
        expected=build_weight(self.allowed,self.protected,self.rgb,feather_radius=2,structure_radius=0,gradient_threshold=4)
        expected*=np.float32(.2);expected[expected<1/255]=0
        np.testing.assert_array_equal(actual,expected)
        self.assertFalse(actual[~self.allowed|self.protected].any())
        self.assertTrue(np.any((current==0)&(actual>0)))
        np.testing.assert_array_equal(self.allowed,before_allowed);np.testing.assert_array_equal(self.protected,before_protected)

    def test_exact_five_variants_keep_each_own_w0_and_hard_pixels_unchanged(self):
        current=build_weight(self.allowed,self.protected,self.rgb,feather_radius=2,structure_radius=3)
        soft=self.module.make_soft_weight(self.rgb,self.allowed,self.protected,2,1)
        results=self.module.make_variants(self.rgb,self.candidate,current,soft,.9,self.protected)
        self.assertEqual(set(results),{'current','no-restore','restore25','no-structure-no-restore','no-structure-restore25'})
        for name,result in results.items():
            weight=soft if name.startswith('no-structure') else current
            np.testing.assert_array_equal(result[weight==0],self.rgb[weight==0])
            np.testing.assert_array_equal(result[self.protected],self.rgb[self.protected])
        np.testing.assert_array_equal(results['current'],blend(self.rgb,gradient_preserving_candidate(self.rgb,self.candidate,.9),current))

    def test_current_variant_uses_the_recorded_restoration_gain(self):
        weight=self.module.make_soft_weight(self.rgb,self.allowed,self.protected,2,1)
        expected=blend(self.rgb,gradient_preserving_candidate(self.rgb,self.candidate,.9,restoration_gain=.25),weight)
        results=self.module.make_variants(self.rgb,self.candidate,weight,weight,.9,self.protected,current_gain=.25)
        np.testing.assert_array_equal(results['current'],expected)

    def test_export_preserves_source_bytes_size_alpha_and_icc(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);source=root/'source.png';output=root/'out'
        alpha=np.full(self.rgb.shape[:2],255,np.uint8);alpha[4,5]=0
        _save_rgb(source,self.rgb,alpha,b'original-icc')
        before=source.read_bytes();weight=self.module.make_soft_weight(self.rgb,self.allowed,self.protected,2,1)
        values=self.module.make_variants(self.rgb,self.candidate,weight,weight,.9,self.protected)
        self.module.write_results(output,values,alpha,b'original-icc',input_paths=[source])
        self.assertEqual(source.read_bytes(),before)
        for name in values:
            with Image.open(output/(name+'.png')) as im:
                self.assertEqual(im.size,(12,10));self.assertEqual(im.info.get('icc_profile'),b'original-icc')
                np.testing.assert_array_equal(np.array(im.getchannel('A')),alpha)

    def test_protected_pixels_may_not_receive_any_experiment_weight(self):
        weight=np.ones(self.rgb.shape[:2],np.float32)
        with self.assertRaises(ValueError):self.module.make_variants(self.rgb,self.candidate,weight,weight,.9,self.protected)

    def test_aliasing_an_output_to_an_original_is_rejected_before_writing(self):
        import os
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);source=root/'original.png';_save_rgb(source,self.rgb,None,None)
        output=root/'outputs';output.mkdir();os.link(source,output/'current.png');before=source.read_bytes()
        with self.assertRaises(ValueError):
            self.module.write_results(output,{'current':self.rgb.copy()},None,None,input_paths=[source])
        self.assertEqual(source.read_bytes(),before)


if __name__=='__main__':unittest.main()
