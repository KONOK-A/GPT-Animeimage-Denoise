import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denoise


class L0RoutingTest(unittest.TestCase):
    def test_pure_and_sr_l0_respect_requested_device(self):
        image = Image.new('RGB', (20, 22), (130, 170, 190))
        with patch.object(denoise, 'l0_smooth', side_effect=lambda a, *args, **kwargs: a) as smooth:
            with patch.object(denoise, 'sr_at', return_value=image):
                result = denoise.run(image, ['l0_0.0005', 'esrgan60_l0'], {'esrgan': object()}, 'cuda:0')
        self.assertEqual(set(result), {'l0_0.0005', 'esrgan60_l0'})
        self.assertEqual([call.kwargs.get('device') for call in smooth.call_args_list], ['cuda:0', 'cuda:0'])

    def test_invalid_parameters_fail_before_iteration(self):
        image = np.zeros((3, 5, 3), dtype=np.float64)
        for lam in (0, -0.001, np.nan, np.inf):
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                denoise.l0_smooth(image, lam, device='cpu')
        for kappa in (0, 1, np.nan, np.inf):
            with self.subTest(kappa=kappa), self.assertRaises(ValueError):
                denoise.l0_smooth(image, 0.0005, kappa=kappa, device='cpu')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
class L0GpuTest(unittest.TestCase):
    def test_gpu_matches_cpu_for_odd_even_and_flat_images(self):
        rng = np.random.default_rng(24)
        images = [rng.random((17, 19, 3)), rng.random((32, 48, 3)),
                  np.full((15, 21, 3), 0.37), np.zeros((12, 16, 3)), np.ones((12, 16, 3)),
                  rng.random((1, 9, 3)), rng.random((9, 1, 3))]
        # Low-amplitude texture around flat colors exercises the hard-gradient
        # threshold rather than only testing random high-contrast inputs.
        flat = np.clip(0.45 + rng.normal(0, 0.006, (47, 61, 3)), 0, 1)
        flat[:, 30:] += 0.08
        images.append(flat)
        for image in images:
            for lam in (0.0005, 0.001):
                with self.subTest(shape=image.shape, lam=lam):
                    expected = denoise.l0_smooth(image, lam)
                    actual = denoise.l0_smooth(image, lam, device='cuda')
                    self.assertEqual(actual.shape, image.shape)
                    self.assertEqual(actual.dtype, np.float64)
                    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)

    def test_fft_computation_really_runs_on_cuda(self):
        with patch('torch.fft.rfft2', wraps=torch.fft.rfft2) as fft:
            denoise.l0_smooth(np.full((9, 11, 3), 0.4), 0.0005, device='cuda')
        self.assertTrue(fft.call_args_list)
        self.assertTrue(all(call.args[0].is_cuda for call in fft.call_args_list))
        self.assertTrue(all(call.args[0].dtype == torch.float64 for call in fft.call_args_list))


if __name__ == '__main__':
    unittest.main()
