"""One V2.1 segmentation pipeline with selectable candidate algorithms."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from gpt_animeimage_denoise import cli
from gpt_animeimage_denoise import fusion_pipeline as pipeline
from test_fusion_pipeline import Models, regions
from test_fusion_esrgan_pipeline import SR, RecordingL0


class SelectableCandidates(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.source=self.root/'source.png'
        self.output=self.root/'outputs';self.rgb=np.full((32,40,3),120,np.uint8)
        Image.fromarray(self.rgb).save(self.source)
        self.models=Models();self.sr=SR();self.l0=RecordingL0()

    def process(self,**kwargs):
        with mock.patch.object(pipeline,'reference_regions',side_effect=regions):
            return pipeline.process_image(self.source,self.output,self.models,
                profile='reference',case_id='case',sr_fn=self.sr,l0_fn=self.l0,**kwargs)

    def test_default_remains_pure_l0_and_a_trial_does_not_change_it(self):
        first=self.process()
        self.assertEqual(first['parameters']['denoiser'],'l0')
        self.assertEqual(first['parameters']['lambda'],.0005)
        self.assertFalse(self.sr.calls)
        self.process(denoiser='esrgan-l0')
        self.assertEqual(self.process()['parameters']['denoiser'],'l0')

    def test_network_and_network_plus_l0_share_segmentation_and_sr(self):
        sr_only=self.process(denoiser='esrgan')
        before_sr=len(self.sr.calls);before_l0=len(self.l0.calls)
        combined=self.process(denoiser='esrgan-l0',lam=.0008)
        self.assertTrue(combined['cache_hits']['evidence'])
        self.assertTrue(combined['cache_hits']['sr'])
        self.assertEqual(len(self.sr.calls),before_sr)
        self.assertEqual(len(self.l0.calls),before_l0+1)
        self.assertEqual(self.l0.calls[-1][1],.0008)
        self.assertIsNone(sr_only['candidate_cache_identity']['lambda'])
        self.assertFalse(sr_only['parameters']['l0_enabled'])

    def test_lambda_is_ignored_by_pure_sr_candidate_cache(self):
        self.process(denoiser='esrgan',lam=.0005)
        record=self.process(denoiser='esrgan',lam=.002)
        self.assertTrue(record['cache_hits']['candidate'])
        self.assertEqual(len(self.sr.calls),1)
        self.assertFalse(self.l0.calls)

    def test_downscale_change_recomputes_candidate_not_segmentation(self):
        self.process(denoiser='esrgan-l0',downscale=.6)
        record=self.process(denoiser='esrgan-l0',downscale=.75)
        self.assertTrue(record['cache_hits']['evidence'])
        self.assertFalse(record['cache_hits']['sr'])
        self.assertEqual(self.sr.calls[-1][2],.75)
        self.assertEqual(len(self.l0.calls),2)

    def test_apisr_uses_original_grid_and_skips_l0(self):
        record=self.process(denoiser='apisr')
        self.assertEqual(self.sr.calls[-1][2],1)
        self.assertFalse(self.l0.calls)
        self.assertEqual(record['parameters']['denoiser_config']['downscale'],1)
        self.assertEqual(record['candidate_cache_stage'],'candidate')

    def test_upstream_l0_presets_and_explicit_lambda_resolve_without_code_changes(self):
        preset=self.process(denoiser='l0_0.001')
        self.assertEqual(preset['parameters']['denoiser'],'l0')
        self.assertEqual(self.l0.calls[-1][1],.001)
        self.process(denoiser='l0_0.001',lam=.0007)
        self.assertEqual(self.l0.calls[-1][1],.0007)


class SelectableCLI(unittest.TestCase):
    def test_removed_cugan_methods_are_rejected_before_model_construction(self):
        for method in ('cugan', 'cugan-l0', 'cugan60', 'cugan60_l0'):
            with self.subTest(method=method):
                with self.assertRaises(SystemExit) as error:
                    self.call(['--method', method])
                self.assertEqual(error.exception.code, 2)

    def call(self,options,command='process'):
        model=mock.Mock(device='cpu',signature='test')
        module=types.ModuleType('gpt_animeimage_denoise.fusion_models')
        module.FusionModels=mock.Mock(return_value=model)
        record=dict(id='x',folder='x',status='complete',resumed=False)
        target='process_image' if command=='process' else 'process_batch'
        arguments=([command,'input.png'] if command=='process' else [command,'--directory','input'])
        with mock.patch.dict(sys.modules,{'gpt_animeimage_denoise.fusion_models':module}), \
             mock.patch.object(pipeline,target,return_value=record if command=='process' else [record]) as process, \
             mock.patch.object(pipeline,'write_records'):
            result=cli.main(arguments+['--out','output','--profile','reference','--device','cpu']+options)
        self.assertEqual(result,0)
        return process.call_args.kwargs

    def test_trial_options_are_forwarded_by_single_and_batch_entry(self):
        for command in ('process','batch'):
            with self.subTest(command=command):
                actual=self.call(['--method','esrgan60_l0','--l0','0.0008','--downscale','0.7'],command)
                self.assertEqual(actual['denoiser'],'esrgan60_l0')
                self.assertEqual(actual['lam'],.0008)
                self.assertEqual(actual['downscale'],.7)

    def test_no_trial_option_keeps_pure_l0_baseline(self):
        actual=self.call([])
        self.assertEqual(actual['denoiser'],'l0')


if __name__=='__main__':unittest.main()
