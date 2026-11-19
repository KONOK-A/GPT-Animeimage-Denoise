"""CPU contracts; heavyweight model construction is replaced at its IO boundary."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


def adapter():
    from gpt_animeimage_denoise import fusion_models
    return fusion_models


def fixture_models():
    module = adapter()
    with patch.object(module.FusionModels, '_verify_assets', return_value={'fixture': 'abc'}):
        return module.FusionModels(device='cpu')


class FusionAdapterAvailability(unittest.TestCase):
    def test_adapter_is_available_without_importing_experiments(self):
        self.assertIsNotNone(importlib.util.find_spec('gpt_animeimage_denoise.fusion_models'))


class PreprocessingContracts(unittest.TestCase):
    def test_sky_center_padding_and_float32_scale(self):
        tensor, geometry = adapter()._prepare_sky(np.full((4, 2, 3), 255, np.uint8))
        self.assertEqual(tensor.shape, (1, 3, 2048, 2048))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(geometry, (2048, 1024, 0, 512))
        self.assertEqual(float(tensor[0, 0, 0, 511]), 0)
        self.assertEqual(float(tensor[0, 0, 0, 512]), 1)

    def test_m2f_stretches_rgb_with_upstream_imagenet_normalization(self):
        rgb = np.zeros((3, 5, 3), np.uint8)
        rgb[..., 0] = 255
        actual = adapter()._preprocess_m2f(rgb)
        self.assertEqual(actual.shape, (1, 3, 768, 768))
        np.testing.assert_allclose(actual[0, :, 0, 0], [2.2489083, -2.0357143, -1.8044444], atol=1e-6)

    def test_invalid_arrays_rejected_before_loading(self):
        models = fixture_models()
        for bad in [np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3), np.float32)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                models.segment_evidence(bad)


class StrictLoadingContracts(unittest.TestCase):
    def toy(self):
        model = torch.nn.Module()
        model.layer = torch.nn.Linear(2, 2)
        model.criterion = torch.nn.Module()
        model.criterion.register_buffer('empty_weight', torch.ones(13))
        return model

    def test_strict_mapping_preserves_every_tensor_and_unused_loss_buffer(self):
        model = self.toy()
        state = {'_orig_mod.model.layer.weight': torch.full((2, 2), 7.),
                 '_orig_mod.model.layer.bias': torch.full((2,), 3.),
                 '_orig_mod.model.criterion.empty_weight': torch.arange(151.)}
        info = adapter()._strict_load_m2f(model, state, expected_count=3)
        self.assertEqual(info['key_mapping'], 'strip_model')
        self.assertEqual(info['tensor_count'], 3)
        self.assertEqual(model.criterion.empty_weight.shape, (151,))
        self.assertEqual(float(model.layer.weight.detach().sum()), 28.)
        with self.assertRaisesRegex(RuntimeError, 'Training loss'):
            model.criterion()

    def test_missing_extra_shape_and_count_errors_have_diagnostics(self):
        for mutation in ['missing', 'extra', 'shape', 'count']:
            model = self.toy()
            state = dict(model.state_dict())
            if mutation == 'missing':
                del state['layer.bias']
            elif mutation == 'extra':
                state['unexpected'] = torch.ones(1)
            elif mutation == 'shape':
                state['layer.weight'] = torch.ones(3, 2)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(RuntimeError, 'Strict checkpoint diagnostics'):
                adapter()._strict_load_m2f(model, state, expected_count=4 if mutation == 'count' else 3)

    def test_hash_mismatch_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'weights'
            source.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                adapter()._verified_hash(source, hashlib.sha256(b'original').hexdigest())

    def test_signature_covers_precision_sources_and_device(self):
        module = adapter()
        with patch.object(module.FusionModels, '_verify_assets', return_value={'weights': 'a', 'source': 'b'}):
            a = module.FusionModels(device='cpu').signature
            self.assertEqual(a, module.FusionModels(device='cpu').signature)
            self.assertNotEqual(a, module.FusionModels(device='cuda').signature)
        with patch.object(module.FusionModels, '_verify_assets', return_value={'weights': 'a', 'source': 'changed'}):
            self.assertNotEqual(a, module.FusionModels(device='cpu').signature)

    def test_windows_asset_manifest_and_pinned_config_reject_corruption(self):
        module = adapter()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            v2, anime = root/'v2', root/'anime'
            manifest = {}
            for name in ['skytnt/isnet.py', 'skytnt/model.safetensors', 'see_through/checkpoint-18000.pt', 'sam2/sam2.1_hiera_large.pt']:
                path = v2/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'valid')
                manifest[name.replace('/', '\\')] = hashlib.sha256(b'valid').hexdigest()
            (v2/'asset_hashes.json').write_text(json.dumps(manifest))
            config = anime/'base-configs/large/config.json'
            config.parent.mkdir(parents=True); config.write_bytes(b'{}')
            pinned = {'base-configs/large/config.json': hashlib.sha256(b'{}').hexdigest()}
            with patch.object(module, 'PINNED_ANIME', pinned):
                module.FusionModels(v2, 'cpu', anime)
                config.write_bytes(b'{"changed":1}')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    module.FusionModels(v2, 'cpu', anime)

    def test_see_rejects_foreign_cached_implementation_before_loading_weights(self):
        module = adapter()
        models = fixture_models()
        models.identities['v2/see_through/checkpoint-18000.pt'] = 'fixture'
        foreign = types.ModuleType('modules.semanticsam')
        foreign.__file__ = 'C:/foreign-install/modules/semanticsam.py'
        class Compatible(torch.nn.Module):
            def __init__(self, class_num):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
        foreign.SemanticSam = Compatible
        with patch.dict(sys.modules, {'modules.semanticsam': foreign}), \
             patch.object(module, '_verified_hash', return_value='fixture'), \
             patch.object(torch, 'load', return_value={'weight': torch.ones(1)}):
            with self.assertRaisesRegex(RuntimeError, 'See source'):
                models._build_see()

    def test_deferred_checkpoints_are_hashed_before_a_model_can_load_them(self):
        module = adapter()
        with tempfile.TemporaryDirectory() as folder:
            v2, anime = Path(folder) / 'v2', Path(folder) / 'anime'
            content = b'checkpoint'
            expected = hashlib.sha256(content).hexdigest()
            manifest = dict.fromkeys(('skytnt/isnet.py', 'skytnt/model.safetensors',
                                      'see_through/checkpoint-18000.pt', 'sam2/sam2.1_hiera_large.pt'), expected)
            pinned = {'models/mask2former.safetensors': expected}
            for directory, entries in ((v2, manifest), (anime, pinned)):
                for name in entries:
                    path = directory / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
            (v2 / 'asset_hashes.json').write_text(json.dumps(manifest), encoding='utf-8')
            with (patch.object(module, 'PINNED_ANIME', pinned),
                  patch.object(module, '_verified_hash', wraps=module._verified_hash) as verify):
                models = module.FusionModels(v2, 'cpu', anime, defer_checkpoint_checks=True)
                self.assertEqual(models.identities['v2/skytnt/model.safetensors'], expected)
                self.assertEqual(models.identities['anime/models/mask2former.safetensors'], expected)
                self.assertEqual([call.args[0].suffix for call in verify.call_args_list], ['.py'])
                path = v2 / 'skytnt/model.safetensors'
                path.write_bytes(b'bad weights')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    models._build_sky()
                self.assertEqual(verify.call_args.args, (path, expected))

    def test_preloaded_sky_is_used_once_and_rejected_if_the_checkpoint_changes(self):
        models = fixture_models()
        with tempfile.TemporaryDirectory() as folder:
            models.assets = Path(folder)
            checkpoint = models.assets / 'skytnt' / 'model.safetensors'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'original')
            models.identities.update({'v2/skytnt/model.safetensors': 'checkpoint',
                                      'v2/skytnt/isnet.py': 'implementation'})
            first, second = FakeSky(), FakeSky()
            with patch.object(models, '_load_sky_cpu', create=True, side_effect=[first, first, second]) as load:
                models._preloaded_sky = models.preload_sky()
                self.assertIs(models._build_sky(), first)
                self.assertIsNone(models._preloaded_sky)
                load.assert_called_once()

                models._preloaded_sky = models.preload_sky()
                checkpoint.write_bytes(b'replaced checkpoint')
                self.assertIs(models._build_sky(), second)
                self.assertIsNone(models._preloaded_sky)
                self.assertEqual(load.call_count, 3)

    def test_see_rejects_foreign_cached_dependencies_with_local_entry_module(self):
        module = adapter()
        models = fixture_models()
        models.identities['v2/see_through/checkpoint-18000.pt'] = 'fixture'
        root = models.assets/'see_through/common'
        models.identities['v2/see_through/common/modules/semanticsam.py'] = 'fixture'
        class Compatible(torch.nn.Module):
            def __init__(self, class_num):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
        entry = types.ModuleType('modules.semanticsam')
        entry.__file__ = str(root/'modules/semanticsam.py')
        entry.SemanticSam = Compatible
        for name in ['modules.extend_sam', 'utils.torch_utils']:
            foreign = types.ModuleType(name)
            foreign.__file__ = 'C:/foreign-install/' + name.replace('.', '/') + '.py'
            with self.subTest(name=name), \
                 patch.dict(sys.modules, {'modules.semanticsam': entry, name: foreign}), \
                 patch.object(module, '_verified_hash', return_value='fixture'), \
                 patch.object(torch, 'load', return_value={'weight': torch.ones(1)}):
                with self.assertRaisesRegex(RuntimeError, 'See source.*'+name):
                    models._build_see()


class FakeSky(torch.nn.Module):
    def forward(self, tensor):
        return [[torch.zeros((1, 1, 2048, 2048))]]


class FakeSee(torch.nn.Module):
    img_size = 1024

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    def inference(self, image):
        if image.dtype != np.uint8:
            raise AssertionError('See must receive uint8 RGB')
        if torch.is_autocast_enabled('cpu'):
            raise AssertionError('See must use native checkpoint dtype without implicit autocast')
        return [torch.full((19, *image.shape[:2]), 1.2345)]


class FakeM2F(torch.nn.Module):
    def forward(self, tensor):
        if tensor.dtype != torch.float32 or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise AssertionError('M2F requires FP32 and TF32 disabled')
        logits = torch.full((1, 12, 768, 768), -8.)
        logits[:, 3, :, :384] = 2
        logits[:, 10, :, 384:] = 2
        return {'semantic_logits': logits.log_softmax(1)}


class SegmentationContracts(unittest.TestCase):
    def test_see_dual_views_share_one_checkpoint_and_keep_distinct_original_grid_logits(self):
        models = fixture_models()
        rgb = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        before = rgb.copy()
        class RecordingSee(torch.nn.Module):
            img_size = 1024

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
                self.inputs = []

            def inference(self, image):
                if torch.is_autocast_enabled('cpu') or self.weight.dtype != torch.bfloat16:
                    raise AssertionError('See must retain checkpoint precision without autocast')
                self.inputs.append(image.copy())
                if len(self.inputs) == 2:
                    image[:] = 255  # Model preprocessing must not alter caller RGB.
                return [torch.full((19, *image.shape[:2]), len(self.inputs) * 1.25, dtype=torch.bfloat16)]

        see = RecordingSee()
        with patch.object(models, '_build_sky', return_value=FakeSky()), \
             patch.object(models, '_build_see', return_value=see) as build_see, \
             patch.object(models, '_build_m2f', return_value=FakeM2F()), \
             torch.autocast('cpu', dtype=torch.bfloat16):
            result = models.segment_evidence(rgb, dual_view=True)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        build_see.assert_called_once()
        self.assertEqual(len(see.inputs), 2)
        expected_white = np.rint(rgb.astype(np.float32) * .5 + 127.5).astype(np.uint8)
        np.testing.assert_array_equal(see.inputs[0], expected_white)
        np.testing.assert_array_equal(see.inputs[1], before)
        np.testing.assert_array_equal(rgb, before)
        for key, value in [('see_logits', 1.25), ('see_original_logits', 2.5)]:
            self.assertEqual(result[key].dtype, np.float32)
            self.assertEqual(result[key].shape, (19, 3, 5))
            np.testing.assert_array_equal(result[key], np.full((19, 3, 5), value, np.float32))
        info = result['provenance']['see']
        self.assertEqual(info['views'], ['white', 'original'])
        self.assertEqual(set(info['view_preprocessing']), {'white', 'original'})
        self.assertIn('Sky', info['view_preprocessing']['white'])
        self.assertIn('without Sky', info['view_preprocessing']['original'])
        self.assertEqual(info['checkpoint_dtype'], 'bfloat16')
        for name in ['sky', 'parts', 'm2f', 'sam']:
            self.assertIsNone(getattr(models, name))

    def test_original_see_invalid_output_is_visible_and_releases_the_loaded_model(self):
        for invalid in [torch.zeros((19, 3, 4)), torch.full((19, 3, 5), float('nan')),
                        torch.full((19, 3, 5), float('inf'))]:
            models = fixture_models()
            class BrokenSecondView(FakeSee):
                def __init__(self):
                    super().__init__()
                    self.calls = 0

                def inference(self, image):
                    self.calls += 1
                    return [torch.zeros((19, *image.shape[:2]))] if self.calls == 1 else [invalid]

            with self.subTest(shape=tuple(invalid.shape)), \
                 patch.object(models, '_build_sky', return_value=FakeSky()), \
                 patch.object(models, '_build_see', return_value=BrokenSecondView()) as build_see, \
                 patch.object(models, '_build_m2f', return_value=FakeM2F()) as build_m2f, \
                 self.assertRaisesRegex(RuntimeError, 'See.*original'):
                models.segment_evidence(np.zeros((3, 5, 3), np.uint8), dual_view=True)
            build_see.assert_called_once()
            build_m2f.assert_not_called()
            self.assertIsNone(models.parts)
            self.assertIsNone(models.sky)

    def test_original_see_forward_error_propagates_and_releases_model(self):
        models = fixture_models()
        class FailingSecondView(FakeSee):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def inference(self, image):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('original See forward failed')
                return super().inference(image)

        with patch.object(models, '_build_sky', return_value=FakeSky()), \
             patch.object(models, '_build_see', return_value=FailingSecondView()), \
             patch.object(models, '_build_m2f', return_value=FakeM2F()) as build_m2f, \
             self.assertRaisesRegex(RuntimeError, 'original See forward failed'):
            models.segment_evidence(np.zeros((3, 5, 3), np.uint8), dual_view=True)
        self.assertIsNone(models.parts)
        build_m2f.assert_not_called()

    def test_dual_evidence_native_argmax_provenance_and_serial_lifecycle(self):
        models = fixture_models()
        def build(name, instance):
            def operation():
                for other in ['sky', 'parts', 'm2f', 'sam']:
                    self.assertIsNone(getattr(models, other))
                return instance
            return operation
        with patch.object(models, '_build_sky', side_effect=build('sky', FakeSky())), \
             patch.object(models, '_build_see', side_effect=build('parts', FakeSee())), \
             patch.object(models, '_build_m2f', side_effect=build('m2f', FakeM2F())):
            result = models.segment_evidence(np.zeros((3, 5, 3), np.uint8))
        self.assertEqual(result['see_logits'].dtype, np.float32)
        self.assertEqual(result['alpha'].shape, (3, 5))
        np.testing.assert_allclose(result['alpha'], .5)
        for name in ['m2f_white', 'm2f_original']:
            self.assertEqual(result[name].shape, (12, 768, 768))
            self.assertEqual(result[name].dtype, np.float32)
            np.testing.assert_allclose(np.exp(result[name]).sum(0), 1, atol=2e-6)
        expected = np.array(Image.fromarray(np.tile(np.array([3]*384+[10]*384, np.uint8), (768, 1))).resize((5, 3), Image.Resampling.NEAREST))
        np.testing.assert_array_equal(result['reference_labels12'], expected)
        self.assertEqual(result['provenance']['see']['checkpoint_dtype'], 'bfloat16')
        self.assertEqual(result['provenance']['m2f']['input_shape'], [1, 3, 768, 768])
        json.dumps(result['provenance'])
        for name in ['sky', 'parts', 'm2f', 'sam']:
            self.assertIsNone(getattr(models, name))

    def test_forward_error_propagates_without_precision_or_size_fallback(self):
        models = fixture_models()
        class Broken(FakeSky):
            def forward(self, tensor):
                raise RuntimeError('real forward failed')
        with patch.object(models, '_build_sky', return_value=Broken()), self.assertRaisesRegex(RuntimeError, 'real forward failed'):
            models.segment_evidence(np.zeros((3, 5, 3), np.uint8))
        self.assertIsNone(models.sky)

    def test_single_view_skips_original_forward(self):
        models = fixture_models()
        class SingleSee(FakeSee):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def inference(self, image):
                self.calls += 1
                if self.calls != 1 or int(image[0, 0, 0]) != 128:
                    raise AssertionError('Reference must request only rounded Sky white See input')
                return super().inference(image)

        class SingleView(FakeM2F):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, tensor):
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError('original view was requested')
                return super().forward(tensor)
        see = SingleSee()
        with patch.object(models, '_build_sky', return_value=FakeSky()), \
             patch.object(models, '_build_see', return_value=see) as build_see, \
             patch.object(models, '_build_m2f', return_value=SingleView()):
            result = models.segment_evidence(np.zeros((3, 5, 3), np.uint8), dual_view=False)
        self.assertIsNone(result['m2f_original'])
        self.assertIsNone(result['see_original_logits'])
        self.assertEqual(see.calls, 1)
        build_see.assert_called_once()
        self.assertEqual(result['provenance']['see']['views'], ['white'])
        self.assertEqual(set(result['provenance']['see']['view_preprocessing']), {'white'})
        self.assertEqual(result['provenance']['m2f']['views'], ['white'])

    def test_sky_preserves_legacy_tf32_and_see_native_precision(self):
        models = fixture_models()
        class LegacySky(FakeSky):
            def forward(self, tensor):
                if not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
                    raise AssertionError('Sky must preserve legacy TF32 settings')
                return super().forward(tensor)
        initial = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            with patch.object(models, '_build_sky', return_value=LegacySky()), \
                 patch.object(models, '_build_see', return_value=FakeSee()), \
                 patch.object(models, '_build_m2f', return_value=FakeM2F()):
                result = models.segment_evidence(np.zeros((3, 5, 3), np.uint8), dual_view=False)
            self.assertTrue(result['provenance']['sky']['tf32']['matmul'])
            self.assertEqual(result['provenance']['see']['autocast'], 'disabled')
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = initial

    def test_tf32_flags_restored_after_forward_error(self):
        module = adapter()
        initial = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            with self.assertRaisesRegex(RuntimeError, 'forward error'):
                with module._fp32_context('cpu'):
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                    raise RuntimeError('forward error')
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = initial

    def test_enclosing_autocast_cannot_change_sky_or_see_precision(self):
        models = fixture_models()
        class NativeSky(FakeSky):
            def forward(self, tensor):
                if torch.is_autocast_enabled('cpu'):
                    raise AssertionError('Sky must disable inherited autocast')
                return super().forward(tensor)
        with patch.object(models, '_build_sky', return_value=NativeSky()), \
             patch.object(models, '_build_see', return_value=FakeSee()), \
             patch.object(models, '_build_m2f', return_value=FakeM2F()), \
             torch.autocast('cpu', dtype=torch.bfloat16):
            result = models.segment_evidence(np.zeros((3, 5, 3), np.uint8), dual_view=False)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(result['provenance']['sky']['precision'], 'float32')
        self.assertEqual(result['provenance']['see']['autocast'], 'disabled')


class FakeSAM:
    mask_threshold = 0.

    def __init__(self):
        self.image = None
        self.embeddings = 0

    def set_image(self, image):
        self.image = image.copy()
        self.embeddings += 1

    def reset_predictor(self):
        self.image = None

    def predict(self, *, point_coords, point_labels, box, mask_input, multimask_output, return_logits):
        assert multimask_output and return_logits
        if mask_input is not None:
            np.testing.assert_equal(mask_input.shape, (1, 256, 256))
        masks = np.full((3, *self.image.shape[:2]), 2., np.float32)
        masks[0, 0, 0] = .5
        return masks, np.array([.2, .6, .9]), np.full((3, 256, 256), 2., np.float32)


class SAMContracts(unittest.TestCase):
    def test_empty_model_slots_do_not_trigger_garbage_collection(self):
        models = fixture_models()
        module = adapter()
        with patch.object(module.gc, 'collect') as collect:
            models.release()
            collect.assert_not_called()
            models.sam = FakeSAM()
            models.sam.set_image(np.zeros((4, 5, 3), np.uint8))
            models._sam_image_key = 'active'
            models._ensure_sam()
            collect.assert_not_called()
            models.release()
            models.release()
            collect.assert_called_once()
            self.assertIsNone(models.sam)
            self.assertIsNone(models._sam_image_key)

    def test_candidates_stability_and_image_embedding_identity(self):
        models = fixture_models()
        sam = FakeSAM()
        rgb = np.zeros((4, 5, 3), np.uint8)
        with patch.object(models, '_build_sam', return_value=sam):
            result = models.predict_candidates(rgb, {'positive': [[0, 0]], 'negative': [[4, 3]], 'box': [0, 0, 4, 3]})
            models.predict_candidates(rgb.copy(), {'box': [0, 0, 4, 3]}, mask_input=result['lowres'][0:1])
            self.assertEqual(sam.embeddings, 1)
            changed = rgb.copy(); changed[0, 0] = 255
            models.predict_candidates(changed, {'positive': [[1, 1]]})
            self.assertEqual(sam.embeddings, 2)
        self.assertEqual(result['masks'].dtype, np.bool_)
        self.assertEqual(result['masks'].shape, (3, 4, 5))
        self.assertEqual(result['scores'].dtype, np.float32)
        self.assertEqual(result['lowres'].shape, (3, 256, 256))
        np.testing.assert_allclose(result['stability'], [.95, 1, 1])
        models.release()
        self.assertIsNone(models.sam)
        self.assertIsNone(sam.image)

    def test_bad_prompts_and_lowres_rejected_before_model_load(self):
        models = fixture_models()
        rgb = np.zeros((4, 5, 3), np.uint8)
        for prompt in [{}, {'positive': [[np.nan, 1]]}, {'positive': [[5, 1]]}, {'box': [4, 0, 0, 3]}, {'negative': [[0, 0]]}, {'positive': [1, 2]}]:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                models.predict_candidates(rgb, prompt)
        for value in [np.zeros((256, 256), np.float32), np.zeros((1, 256, 256), np.float64), np.full((1, 256, 256), np.nan, np.float32)]:
            with self.assertRaises(ValueError):
                models.predict_candidates(rgb, {'positive': [[1, 1]]}, mask_input=value)
        self.assertIsNone(models.sam)

    def test_roi_generator_processes_only_crop_and_maps_to_original_grid(self):
        models = fixture_models()
        class Generator:
            def generate(self, crop):
                self_shape = crop.shape
                if self_shape != (3, 4, 3):
                    raise AssertionError(self_shape)
                mask = np.zeros((3, 4), bool); mask[1, 2] = True
                return [{'segmentation': mask, 'predicted_iou': .9, 'stability_score': .95, 'bbox': [2, 1, 1, 1]}]
        with patch.object(models, '_build_sam', return_value=FakeSAM()), patch.object(models, '_build_generator', return_value=Generator()):
            result = models.generate_candidates(np.zeros((8, 10, 3), np.uint8), [2, 3, 6, 6])
        self.assertEqual(result[0]['mask'].shape, (8, 10))
        self.assertEqual(np.argwhere(result[0]['mask']).tolist(), [[4, 4]])
        self.assertEqual(result[0]['bbox'], [4, 4, 5, 5])

    def test_nonfinite_or_wrong_candidate_outputs_are_visible(self):
        models = fixture_models()
        class Invalid(FakeSAM):
            def predict(self, **kwargs):
                values, scores, lowres = super().predict(**kwargs)
                lowres[0, 0, 0] = np.nan
                return values, scores, lowres
        with patch.object(models, '_build_sam', return_value=Invalid()), self.assertRaisesRegex(RuntimeError, 'invalid candidate'):
            models.predict_candidates(np.zeros((4, 5, 3), np.uint8), {'positive': [[1, 1]]})


if __name__ == '__main__':
    unittest.main()
