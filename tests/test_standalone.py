from contextlib import redirect_stderr
from io import StringIO
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denoise


class TrackingUpscale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape[-2:]))
        return x.repeat_interleave(2, 2).repeat_interleave(2, 3)


class StandaloneDenoiseTest(unittest.TestCase):
    def test_invalid_cli_l0_is_rejected_before_loading_models(self):
        for value in ('0', '-0.001', 'nan', 'inf', '-inf'):
            args = ['gaid-whole', 'sample.png', '--method', 'esrgan60_l0', f'--l0={value}']
            with (self.subTest(value=value),
                  mock.patch.object(sys, 'argv', args),
                  mock.patch.object(denoise, 'load_model', side_effect=AssertionError('model loading reached')) as load,
                  redirect_stderr(StringIO()) as errors):
                with self.assertRaises(SystemExit) as result:
                    denoise.main()
                self.assertEqual(result.exception.code, 2)
                self.assertIn('--l0 must be a finite positive value', errors.getvalue())
                load.assert_not_called()

    def test_removed_cugan_is_rejected_even_with_an_injected_model(self):
        source = Image.new('RGB', (8, 8))
        for method in ('cugan60', 'cugan60_l0'):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'Unknown denoiser'):
                denoise.run(source, [method], {'cugan': TrackingUpscale()}, 'cpu')

    def test_auto_device_reaches_the_standalone_processor_on_mps(self):
        with TemporaryDirectory() as folder:
            args = ['gaid-whole', 'sample.png', '--method', 'l0_0.0005', '--out', folder]
            with (mock.patch.object(sys, 'argv', args),
                  mock.patch.object(torch.cuda, 'is_available', return_value=False),
                  mock.patch.object(torch.backends.mps, 'is_available', return_value=True),
                  mock.patch.object(Image, 'open', return_value=Image.new('RGB', (8, 8))),
                  mock.patch.object(denoise, 'run', return_value={}) as run):
                denoise.main()
        self.assertEqual(run.call_args.args[3], 'mps')

    def test_direct_project_entry_uses_bounded_tiles(self):
        model = TrackingUpscale()
        pixels = np.random.default_rng(8).integers(0, 256, (513, 641, 3), dtype=np.uint8)
        output = denoise.upscale(model, Image.fromarray(pixels), 'cpu')
        self.assertGreater(len(model.shapes), 1)
        self.assertLessEqual(max(max(shape) for shape in model.shapes), 384)
        np.testing.assert_array_equal(np.asarray(output), np.repeat(np.repeat(pixels, 2, 0), 2, 1))


if __name__ == '__main__':
    unittest.main()
