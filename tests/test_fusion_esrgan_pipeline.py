"""End-to-end orchestration contracts with neural/FFT boundaries injected."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from test_fusion_pipeline import Models, regions

POSITIONAL_L0_CALLS=[]
def positional_l0(rgb,lam,factor=1,*,device):
    POSITIONAL_L0_CALLS.append(lam)
    return np.full(rgb.shape,.3*factor,np.float32)


class SR:
    signature='fake-sr-v1'
    def __init__(self):self.calls=[];self.value=200
    def __call__(self,rgb,*,device,downscale=.6):
        self.calls.append((rgb.copy(),device,downscale))
        return np.full_like(rgb,self.value)


class RecordingL0:
    signature='recording-l0-v1'
    def __init__(self):self.calls=[]
    def __call__(self,rgb,lam,*,device):
        self.calls.append((rgb.copy(),lam,device))
        return np.clip(rgb*.9+.01,0,1).astype(np.float32)


class CompletePipelineESRGAN(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'原图.png';self.output=self.root/'outputs'
        self.rgb=np.random.default_rng(13).integers(80,150,(32,40,3),np.uint8)
        Image.fromarray(self.rgb).save(self.source)
        self.models=Models();self.sr=SR();self.l0=RecordingL0()

    def run_image(self,**kwargs):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        kwargs.setdefault('denoiser','esrgan-l0')
        with mock.patch.object(pipeline,'fuse_evidence',side_effect=regions), \
             mock.patch.object(pipeline,'reference_regions',side_effect=regions), \
             mock.patch.object(pipeline,'refine_regions',side_effect=lambda rgb,ev,r,models,policy=None:r):
            return pipeline.process_image(self.source,self.output,self.models,case_id='case',
                sr_fn=self.sr,l0_fn=self.l0,**kwargs)

    def test_explicit_esrgan_calls_sr_then_l0_then_existing_restoration(self):
        record=self.run_image()
        self.assertEqual(record['parameters']['denoiser'],'esrgan-l0')
        self.assertEqual(len(self.sr.calls),1)
        np.testing.assert_array_equal(self.sr.calls[0][0],self.rgb)
        self.assertEqual(self.sr.calls[0][1:],('cpu',.6))
        np.testing.assert_allclose(self.l0.calls[0][0],np.full(self.rgb.shape,200/255),atol=1e-15)
        self.assertEqual(self.l0.calls[0][0].dtype,np.float64)
        self.assertEqual(self.l0.calls[0][1:],(.0005,'cpu'))
        self.assertEqual(record['candidate_cache_stage'],'candidate')
        self.assertEqual(record['candidate_cache_identity']['denoiser'],'esrgan-l0')
        self.assertEqual(record['effective_sigma'],12*32/2048)

    def test_lambda_reuses_sr_and_sigma_strength_reuse_float_candidate(self):
        first=self.run_image();self.assertTrue(self.run_image()['resumed'])
        changed=self.run_image(lam=.0007)
        self.assertEqual(len(self.sr.calls),1);self.assertEqual(len(self.l0.calls),2)
        self.assertTrue(changed['cache_hits']['sr'])
        self.run_image(lam=.0007,sigma=8,strength=.5)
        self.assertEqual(len(self.sr.calls),1);self.assertEqual(len(self.l0.calls),2)
        self.assertNotEqual(first['fingerprint'],changed['fingerprint'])

    def test_old_pure_l0_and_new_engine_caches_are_distinct(self):
        new=self.run_image()
        old=self.run_image(denoiser='l0')
        self.assertNotEqual(new['candidate_cache_identity'],old['candidate_cache_identity'])
        np.testing.assert_allclose(self.l0.calls[-1][0],self.rgb.astype(np.float64)/255)
        self.assertEqual(len(self.sr.calls),1)
        self.run_image()
        self.assertEqual(len(self.sr.calls),1);self.assertEqual(len(self.l0.calls),2)

    def test_sr_identity_change_invalidates_sr_and_post_l0(self):
        self.run_image();self.sr.signature='fake-sr-v2';self.sr.value=201
        record=self.run_image()
        self.assertFalse(record['resumed'])
        self.assertEqual(len(self.sr.calls),2);self.assertEqual(len(self.l0.calls),2)
        np.testing.assert_allclose(self.l0.calls[-1][0],np.full(self.rgb.shape,201/255),atol=1e-15)

    def test_esrgan_candidate_only_changes_selected_pixels_preserving_alpha_icc(self):
        alpha=np.full((32,40),255,np.uint8);alpha[20:24,20:24]=0
        Image.fromarray(np.dstack([self.rgb,alpha])).save(self.source,icc_profile=b'profile-data')
        # A32px fixture needs a larger reference sigma to exercise visible
        # processing; sigma12@2048 would reconstruct nearly every tiny ripple.
        before=self.source.read_bytes();self.run_image(sigma=512)
        folder=self.output/'case';weight=np.load(folder/'weight.npy')
        with Image.open(folder/'result.png') as image:
            after=np.asarray(image.convert('RGB'))
            np.testing.assert_array_equal(np.asarray(image.getchannel('A')),alpha)
            self.assertEqual(image.info.get('icc_profile'),b'profile-data')
        np.testing.assert_array_equal(after[weight==0],self.rgb[weight==0])
        self.assertFalse(weight[:,3:6].any())
        self.assertTrue(np.any(after[weight>0]!=self.rgb[weight>0]))
        self.assertEqual(self.source.read_bytes(),before)

    def test_positional_default_callback_identity_is_json_canonical(self):
        POSITIONAL_L0_CALLS.clear();self.l0=positional_l0
        self.run_image();record=self.run_image(sigma=8,strength=.5)
        self.assertEqual(len(POSITIONAL_L0_CALLS),1)
        self.assertTrue(record['cache_hits']['candidate'])
        self.assertEqual(record['parameters']['l0_implementation']['defaults'],[1])

    def test_output_cannot_alias_native_esrgan_model_weights(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        model_path=self.root/'weights.pth';model_path.write_bytes(b'preserved-model-data')
        folder=self.output/'case';folder.mkdir(parents=True)
        import os,hashlib
        os.link(model_path,folder/'result.png')
        description={'method':'esrgan-l0','device':'cpu','downscale':.6,
            'model':{'path':str(model_path),'expected_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest()}}
        with mock.patch.object(pipeline,'describe_denoiser',return_value=description),self.assertRaises(ValueError):
            # Native invocation: no injected SR hook, so its real weight path
            # must join the protected input set before inference or writes.
            pipeline.process_image(self.source,self.output,self.models,case_id='case',
                denoiser='esrgan-l0',l0_fn=self.l0)
        self.assertEqual(model_path.read_bytes(),b'preserved-model-data')
        self.assertFalse(self.models.calls)

    def test_batch_preflights_model_aliases_before_error_or_summary_writes(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        import os,hashlib
        model_path=self.root/'weights.pth';payload=b'preserved-model-data'
        model_path.write_bytes(payload)
        description={'method':'esrgan-l0','device':'cpu','downscale':.6,
            'model':{'path':str(model_path),'expected_sha256':hashlib.sha256(payload).hexdigest()}}
        manifest={'items':[{'id':'case','path':str(self.source),'status':'eligible'}]}
        for name in ('case/report.json.tmp','summary.csv.tmp','records.json'):
            with self.subTest(target=name):
                output=self.root/('batch-'+name.replace('/','-'))
                target=output/name;target.parent.mkdir(parents=True)
                os.link(model_path,target)
                with mock.patch.object(pipeline,'describe_denoiser',return_value=description), \
                     self.assertRaises(ValueError):
                    pipeline.process_batch(manifest,output,self.models,
                        denoiser='esrgan-l0',l0_fn=self.l0)
                self.assertEqual(model_path.read_bytes(),payload)
                self.assertFalse(self.models.calls)

    def test_empty_batch_protects_native_model_from_global_output_alias(self):
        from gpt_animeimage_denoise import fusion_pipeline as pipeline
        import os,hashlib
        model_path=self.root/'weights.pth';payload=b'preserved-model-data'
        model_path.write_bytes(payload);self.output.mkdir()
        os.link(model_path,self.output/'summary.csv.tmp')
        description={'method':'esrgan-l0','device':'cpu','downscale':.6,
            'model':{'path':str(model_path),'expected_sha256':hashlib.sha256(payload).hexdigest()}}
        with mock.patch.object(pipeline,'describe_denoiser',return_value=description), \
             self.assertRaises(ValueError):
            pipeline.process_batch({'items':[]},self.output,self.models,
                denoiser='esrgan-l0',l0_fn=self.l0)
        self.assertEqual(model_path.read_bytes(),payload)


if __name__=='__main__':unittest.main()
