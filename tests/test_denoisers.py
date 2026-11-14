"""Candidate-stage contracts; heavyweight model execution stays behind fakes."""
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DenoiserTest(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('gpt_animeimage_denoise.denoisers')
        except ModuleNotFoundError:
            self.fail('The V2.1 candidate backend has not been implemented')
        self.rgb = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    def test_injected_sr_keeps_original_grid_and_input_and_avoids_model_assets(self):
        before = self.rgb.copy()
        def sr(rgb, *, device, downscale):
            self.assertEqual(device, 'cpu')
            self.assertEqual(downscale, 0.6)
            return np.full_like(rgb, 91)
        with mock.patch.dict(sys.modules, {'denoise': None}):
            result = self.module.super_resolve(self.rgb, device='cpu', sr_fn=sr)
        np.testing.assert_array_equal(result, np.full((5, 7, 3), 91, np.uint8))
        np.testing.assert_array_equal(self.rgb, before)

    def test_injected_sr_cannot_modify_callers_original_rgb(self):
        original = self.rgb.copy()
        def in_place_sr(rgb, *, device, downscale):
            rgb[:] = 91
            return rgb
        result = self.module.super_resolve(self.rgb, device='cpu', sr_fn=in_place_sr)
        np.testing.assert_array_equal(self.rgb, original)
        np.testing.assert_array_equal(result, np.full((5, 7, 3), 91, np.uint8))

    def test_sr_rejects_wrong_output_grid_dtype_and_non_array(self):
        for invalid in [np.zeros((5, 6, 3), np.uint8), np.zeros((5, 7, 3), np.float32),
                        np.zeros((5, 7), np.uint8), Image.fromarray(self.rgb)]:
            with self.subTest(type=type(invalid), shape=getattr(invalid, 'shape', None)):
                with self.assertRaises(ValueError):
                    self.module.super_resolve(self.rgb, device='cpu',
                        sr_fn=lambda rgb, *, device, downscale: invalid)

    def test_invalid_rgb_is_rejected_before_expensive_stage(self):
        def forbidden(*args, **kwargs):
            self.fail('An invalid image reached the expensive stage')
        for invalid in [self.rgb.astype(np.float32), np.zeros((0, 7, 3), np.uint8),
                        np.zeros((5, 7, 4), np.uint8), self.rgb.tolist()]:
            with self.subTest(shape=getattr(invalid, 'shape', None)):
                with self.assertRaises(ValueError):
                    self.module.super_resolve(invalid, device='cpu', sr_fn=forbidden)
                with self.assertRaises(ValueError):
                    self.module.smooth_candidate(invalid, 0.0005, device='cpu', l0_fn=forbidden)

    def test_l0_receives_unrounded_float64_pixels_lambda_and_selected_device(self):
        original = self.rgb.copy()
        def l0(image, lam, *, device):
            self.assertEqual(image.dtype, np.float64)
            self.assertEqual(lam, 0.0005)
            self.assertEqual(device, 'cuda:1')
            self.assertAlmostEqual(image[0, 0, 1], 1 / 255, places=15)
            return image + 0.00001
        candidate = self.module.smooth_candidate(self.rgb, 0.0005, device='cuda:1', l0_fn=l0)
        self.assertEqual(candidate.dtype, np.float32)
        self.assertEqual(candidate.shape, self.rgb.shape)
        self.assertAlmostEqual(float(candidate[0, 0, 0]), 0.00001, places=11)
        self.assertGreater(float(candidate[0, 0, 1]), 1 / 255)
        np.testing.assert_array_equal(self.rgb, original)

    def test_l0_rejects_nonfinite_range_or_misaligned_results(self):
        for invalid in [np.zeros((5, 6, 3)), np.full((5, 7, 3), np.nan),
                        np.full((5, 7, 3), np.inf), np.full((5, 7, 3), -0.01),
                        np.full((5, 7, 3), 1.01)]:
            with self.subTest(shape=invalid.shape, value=invalid.flat[0]):
                with self.assertRaises(ValueError):
                    self.module.smooth_candidate(self.rgb, 0.0005, device='cpu',
                        l0_fn=lambda rgb, lam, *, device: invalid)

    def test_invalid_lambda_is_rejected_before_l0(self):
        def forbidden(*args, **kwargs):
            self.fail('An invalid lambda reached L0')
        for lam in [0, -0.0005, np.nan, np.inf, -np.inf]:
            with self.subTest(lam=lam):
                with self.assertRaises(ValueError):
                    self.module.smooth_candidate(self.rgb, lam, device='cpu', l0_fn=forbidden)

    def test_native_l0_uses_existing_solver_and_retains_float_candidate(self):
        # A tiny CPU solve exercises the real legacy boundary without model assets.
        source = np.full((3, 4, 3), 73, np.uint8)
        result = self.module.smooth_candidate(source, 0.0005, device='cpu')
        np.testing.assert_allclose(result, 73 / 255, rtol=0, atol=1e-7)
        self.assertEqual(result.dtype, np.float32)

    def test_pure_l0_identity_has_no_sr_or_model_dependency(self):
        def forbidden(*args, **kwargs):
            raise AssertionError('Pure L0 touched SR identity')
        forbidden.signature = object()
        with mock.patch.dict(sys.modules, {'denoise': None}), mock.patch.object(
                self.module, '_sha', side_effect=forbidden):
            identity = self.module.describe_denoiser('l0', 'cpu', sr_fn=forbidden)
        self.assertEqual(identity, {'method': 'l0', 'device': 'cpu'})

    def test_native_sr_identity_records_precision_model_and_code(self):
        identity = self.module.describe_denoiser('esrgan-l0', 'cuda:1')
        self.assertEqual(identity['method'], 'esrgan-l0')
        self.assertEqual(identity['device'], 'cuda:1')
        self.assertEqual(identity['downscale'], 0.6)
        self.assertEqual(identity['model']['name'], 'RealESRGAN_x4plus_anime_6B.pth')
        self.assertEqual(identity['model']['expected_sha256'],
                         'f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da')
        self.assertTrue(Path(identity['model']['path']).is_absolute())
        self.assertEqual(identity['precision'], {'dtype': 'float32', 'autocast': False,
            'matmul_allow_tf32': False, 'cudnn_allow_tf32': False})
        self.assertEqual(set(identity['implementation']),
                         {'denoisers.py', 'denoise.py', 'tiled_inference.py'})
        json.dumps(identity, sort_keys=True, allow_nan=False)

    def test_injected_sr_identity_is_stable_and_tracks_explicit_state(self):
        class ConfiguredSR:
            def __init__(self, value):self.signature = {'value': value}
            def __call__(self, rgb, *, device, downscale):return rgb
        first = self.module.describe_denoiser('esrgan-l0', 'cpu', sr_fn=ConfiguredSR(1))
        again = self.module.describe_denoiser('esrgan-l0', 'cpu', sr_fn=ConfiguredSR(1))
        changed = self.module.describe_denoiser('esrgan-l0', 'cpu', sr_fn=ConfiguredSR(2))
        self.assertEqual(first, again)
        self.assertNotEqual(first, changed)
        self.assertEqual(first['sr_implementation']['kind'], 'injected_sr')
        self.assertNotIn('model', first)

    def test_injected_sr_code_and_closure_changes_invalidate_identity(self):
        def configured(value):
            return lambda rgb, *, device, downscale: np.full_like(rgb, value)
        first = self.module.describe_denoiser('esrgan-l0', 'cpu', sr_fn=configured(12))
        second = self.module.describe_denoiser('esrgan-l0', 'cpu', sr_fn=configured(18))
        third = self.module.describe_denoiser('esrgan-l0', 'cpu',
            sr_fn=lambda rgb, *, device, downscale: rgb.copy())
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_opaque_sr_requires_explicit_stable_signature(self):
        with self.assertRaises(ValueError):
            self.module.describe_denoiser('esrgan-l0', 'cpu', sr_fn=np.asarray)

    def test_unknown_method_is_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            self.module.describe_denoiser('unknown', 'cpu')

    def test_methods_and_legacy_aliases_resolve_to_the_actual_l0_strength(self):
        resolve = getattr(self.module, 'resolve_denoiser', None)
        self.assertTrue(callable(resolve), 'The generic method resolver is missing')
        rows = [(None, 'l0', 0.0005), ('l0', 'l0', 0.0005),
                ('l0_0.001', 'l0', 0.001), ('l0_0.0005', 'l0', 0.0005),
                ('esrgan', 'esrgan', 0.0005), ('esrgan-l0', 'esrgan-l0', 0.0005),
                ('esrgan60', 'esrgan', 0.0005), ('esrgan60_l0', 'esrgan-l0', 0.0005),
                ('apisr', 'apisr', 0.0005)]
        for method, canonical, strength in rows:
            with self.subTest(method=method):
                self.assertEqual(resolve(method), (canonical, strength))
                self.assertEqual(resolve(method, 0.0007), (canonical, 0.0007))
                if method is not None:self.assertIn(method, self.module.METHOD_CHOICES)
        for method in self.module.METHOD_CHOICES:
            self.assertIn(resolve(method)[0], self.module.DENOISERS)

    def test_method_stage_predicates_distinguish_sr_and_l0(self):
        for method, sr, l0 in [('l0_0.001', False, True), ('l0', False, True),
                ('esrgan', True, False), ('esrgan60_l0', True, True),
                ('apisr', True, False)]:
            with self.subTest(method=method):
                self.assertEqual(self.module.uses_sr(method), sr)
                self.assertEqual(self.module.uses_l0(method), l0)

    def test_resolver_rejects_invalid_strength_and_unknown_methods(self):
        for lam in [0, -1, np.nan, np.inf, '0.0005']:
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                self.module.resolve_denoiser('l0', lam)
        with self.assertRaises(ValueError):self.module.resolve_denoiser('missing')

    def test_sr_identity_can_be_reused_when_l0_is_added(self):
        first = self.module.describe_denoiser('esrgan', 'cpu', downscale=0.5)
        second = self.module.describe_denoiser('esrgan-l0', 'cpu', downscale=0.5)
        self.assertEqual(first.pop('method'), 'esrgan')
        self.assertEqual(second.pop('method'), 'esrgan-l0')
        self.assertEqual(first, second)
        self.assertEqual(first['network'], 'esrgan')
        self.assertEqual(first['downscale'], 0.5)

    def test_sr_identity_records_each_existing_model_and_apisr_actual_scale(self):
        rows = [('esrgan', 'RealESRGAN_x4plus_anime_6B.pth', self.module.ESRGAN_SHA256, 0.6),
                ('apisr', '2x_APISR_RRDB_GAN_generator.pth',
                 '3b0d2b3a3c0461ac17d00f4f32240666fb832b738ea5a48449b1acf07fbb07e5', 1.0)]
        for method, name, checksum, scale in rows:
            with self.subTest(method=method):
                identity = self.module.describe_denoiser(method, 'cpu', downscale=0.6)
                self.assertEqual(identity['model']['name'], name)
                self.assertEqual(identity['model']['expected_sha256'], checksum)
                self.assertEqual(identity['network'], method)
                self.assertEqual(identity['downscale'], scale)

    def test_injected_sr_uses_requested_network_scale_and_still_protects_input(self):
        for method, scale in [('esrgan60', 0.5), ('apisr', 1.0)]:
            with self.subTest(method=method):
                original = self.rgb.copy()
                def sr(rgb, *, device, downscale):
                    self.assertEqual(downscale, scale)
                    rgb[:] = 57
                    return rgb
                with mock.patch.dict(sys.modules, {'denoise': None}):
                    result = self.module.super_resolve(self.rgb, device='cpu', method=method,
                                                       downscale=0.5, sr_fn=sr)
                np.testing.assert_array_equal(self.rgb, original)
                np.testing.assert_array_equal(result, np.full((5, 7, 3), 57, np.uint8))

    def test_invalid_scale_and_non_sr_method_fail_before_sr_hook(self):
        def forbidden(*args, **kwargs):self.fail('Invalid SR configuration reached inference')
        for method in ['esrgan', 'apisr']:
            for scale in [0, -0.1, 1.01, np.nan, np.inf]:
                with self.subTest(method=method, scale=scale):
                    with self.assertRaises(ValueError):
                        self.module.super_resolve(self.rgb, device='cpu', method=method,
                                                   downscale=scale, sr_fn=forbidden)
                    with self.assertRaises(ValueError):
                        self.module.describe_denoiser(method, 'cpu', downscale=scale)
        with self.assertRaises(ValueError):
            self.module.super_resolve(self.rgb, device='cpu', method='l0', sr_fn=forbidden)


class NativeSRTest(unittest.TestCase):
    def setUp(self):
        DenoiserTest.setUp(self)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.path = self.folder / 'RealESRGAN_x4plus_anime_6B.pth'
        self.path.write_bytes(b'test ESRGAN weights')
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(self.module, 'model_root', return_value=self.folder).start()
        mock.patch.object(self.module, 'ESRGAN_SHA256',
                          hashlib.sha256(self.path.read_bytes()).hexdigest()).start()

    def test_native_sr_checks_weights_before_model_deserialization(self):
        import denoise
        self.path.write_bytes(b'corrupt weights')
        with mock.patch.object(denoise, 'load_model', side_effect=AssertionError('Unsafe model load')):
            with self.assertRaisesRegex(ValueError, 'SHA256|sha256|checksum'):
                self.module.super_resolve(self.rgb, device='cpu')

    def test_native_sr_validates_a_newly_fetched_file_before_loading(self):
        import denoise
        self.path.unlink()
        def fetch(key, folder):
            self.assertEqual(key, 'esrgan')
            self.assertEqual(folder, self.folder)
            self.path.write_bytes(b'unexpected download')
            return self.path
        with mock.patch.object(denoise, 'fetch', side_effect=fetch), mock.patch.object(
                denoise, 'load_model', side_effect=AssertionError('Unsafe model load')):
            with self.assertRaisesRegex(ValueError, 'SHA256|sha256|checksum'):
                self.module.super_resolve(self.rgb, device='cpu')

    def test_native_sr_uses_box_downscale_lanczos_original_grid_and_shared_upscale(self):
        import denoise
        source = Image.fromarray(self.rgb)
        small = np.asarray(source.resize((4, 3), Image.Resampling.BOX))
        output = Image.fromarray(np.repeat(np.repeat(small, 4, axis=0), 4, axis=1))
        model = torch.nn.Identity().half().train()
        def upscale(actual_model, image, device):
            self.assertEqual(image.size, (4, 3))
            np.testing.assert_array_equal(np.asarray(image), small)
            self.assertIs(actual_model, model)
            self.assertFalse(actual_model.training)
            self.assertEqual(device, 'cpu')
            return output
        with mock.patch.object(denoise, 'load_model', return_value=model), mock.patch.object(
                denoise, 'upscale', side_effect=upscale), mock.patch.object(denoise, 'DOWN', 0.9):
            result = self.module.super_resolve(self.rgb, device='cpu')
            self.assertEqual(denoise.DOWN, 0.9)
        expected = np.asarray(output.resize((7, 5), Image.Resampling.LANCZOS))
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.uint8)

    def test_native_sr_clamps_downscale_dimensions_to_one(self):
        import denoise
        source = np.full((1, 1, 3), 83, np.uint8)
        def upscale(model, image, device):
            self.assertEqual(image.size, (1, 1))
            return image.resize((4, 4))
        with mock.patch.object(denoise, 'load_model', return_value=torch.nn.Identity()), mock.patch.object(
                denoise, 'upscale', side_effect=upscale):
            result = self.module.super_resolve(source, device='cpu')
        np.testing.assert_array_equal(result, source)

    def test_three_networks_dispatch_with_their_actual_input_size(self):
        import denoise
        for method, network in [('esrgan', 'esrgan'), ('esrgan-l0', 'esrgan'), ('apisr', 'apisr')]:
            with self.subTest(method=method):
                name = denoise.MODELS[network][0]
                model_path = self.folder / name
                model_path.write_bytes(b'test ESRGAN weights')
                checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
                expected_size = (7, 5) if network == 'apisr' else (4, 2)
                def load_model(key, folder, device):
                    self.assertEqual((key, folder, device), (network, self.folder, 'cpu'))
                    return torch.nn.Identity()
                def upscale(model, image, device):
                    self.assertEqual(image.size, expected_size)
                    if network == 'apisr':
                        np.testing.assert_array_equal(np.asarray(image), self.rgb)
                    return Image.new('RGB', (image.width * 2, image.height * 2), (17, 43, 89))
                with mock.patch.dict(denoise.SHA256, {name: checksum}), mock.patch.object(
                        denoise, 'load_model', new=load_model), mock.patch.object(
                        denoise, 'upscale', new=upscale):
                    result = self.module.super_resolve(self.rgb, device='cpu', method=method,
                                                       downscale=0.5)
                np.testing.assert_array_equal(result, np.broadcast_to([17, 43, 89], (5, 7, 3)))

    def test_apisr_model_is_checked_before_deserialization(self):
        import denoise
        (self.folder / denoise.MODELS['apisr'][0]).write_bytes(b'corrupt weights')
        with mock.patch.object(denoise, 'load_model', side_effect=AssertionError('Unsafe model load')):
            with self.assertRaisesRegex(ValueError, 'SHA256|sha256|checksum'):
                self.module.super_resolve(self.rgb, device='cpu', method='apisr')

    def test_precision_is_scoped_and_gpu_model_is_released_before_l0(self):
        import denoise
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', old_matmul)
        self.addCleanup(setattr, torch.backends.cudnn, 'allow_tf32', old_cudnn)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model_ref = []
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
        def load_model(key, folder, device):
            self.assertEqual((key, folder, device), ('esrgan', self.folder, 'cuda'))
            value = Model()
            model_ref.append(weakref.ref(value))
            return value
        def upscale(model, image, device):
            self.assertEqual(model.weight.dtype, torch.float32)
            self.assertFalse(model.training)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.is_autocast_enabled('cuda'))
            return image.resize((28, 20))
        def empty_cache():
            gc.collect()
            self.assertIsNone(model_ref[0]())
        with mock.patch.object(denoise, 'load_model', side_effect=load_model), mock.patch.object(
                denoise, 'upscale', new=upscale), mock.patch.object(torch.cuda,
                'empty_cache', side_effect=empty_cache):
            result = self.module.super_resolve(self.rgb, device='cuda')
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.module.smooth_candidate(result, 0.0005, device='cpu', l0_fn=lambda x, lam, *, device: x)

    def test_failed_inference_restores_tf32_and_clears_gpu_cache(self):
        import denoise
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', old_matmul)
        self.addCleanup(setattr, torch.backends.cudnn, 'allow_tf32', old_cudnn)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        with mock.patch.object(denoise, 'load_model', return_value=torch.nn.Identity()), mock.patch.object(
                denoise, 'upscale', side_effect=RuntimeError('simulated inference failure')), mock.patch.object(
                torch.cuda, 'empty_cache') as cache:
            with self.assertRaisesRegex(RuntimeError, 'simulated inference failure'):
                self.module.super_resolve(self.rgb, device='cuda')
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(cache.call_count, 1)


if __name__ == '__main__':unittest.main()
