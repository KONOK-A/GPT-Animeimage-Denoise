import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tiled_inference as module


class Nearest(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x.repeat_interleave(self.scale, 2).repeat_interleave(self.scale, 3)


class ContextAverage(torch.nn.Module):
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, (1, 81), stride=1, padding=(0, 40))
        return torch.nn.functional.avg_pool2d(x, (81, 1), stride=1, padding=(40, 0))


class TileMean(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x)


class PhaseLocked(torch.nn.Module):
    def forward(self, x):
        height, width = x.shape[2:]
        x = torch.nn.functional.pad(x, (0, width % 2, 0, height % 2), mode='replicate')
        x = torch.nn.functional.avg_pool2d(x, 2)
        return x.repeat_interleave(2, 2).repeat_interleave(2, 3)[:, :, :height, :width]


class TilingTest(unittest.TestCase):
    def test_cpu_rrdb_uses_smaller_bands_when_memory_is_tight(self):
        from spandrel.architectures.ESRGAN.__arch.RRDB import RRDBNet

        pixels = np.random.default_rng(19).integers(0, 256, (75, 27, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        for shuffle in (None, 2):
            with self.subTest(shuffle=shuffle), torch.random.fork_rng(devices=[]):
                torch.manual_seed(19)
                model = RRDBNet(in_nc=3 * (shuffle or 1) ** 2, num_filters=4, num_blocks=0,
                                shuffle_factor=shuffle).eval()
                upsample_heights = []
                handle = model.model[2].register_forward_pre_hook(
                    lambda _, args: upsample_heights.append(args[0].shape[-2]))
                try:
                    with mock.patch.object(module.psutil, 'virtual_memory',
                                           return_value=SimpleNamespace(available=16 * 2**30)):
                        expected = module.tiled_upscale(model, image, 'cpu')
                    self.assertEqual(len(upsample_heights), 1)
                    full_height = upsample_heights.pop()
                    with mock.patch.object(module.psutil, 'virtual_memory',
                                           return_value=SimpleNamespace(available=512 * 2**20)):
                        actual = module.tiled_upscale(model, image, 'cpu')
                finally:
                    handle.remove()

                self.assertGreater(len(upsample_heights), 1)
                self.assertLess(max(upsample_heights), full_height)
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_memory_budget_keeps_full_tiles_when_they_fit(self):
        args = (384, 384, 64, 4, 4)
        for gib, expected in ((12, 384), (3, 128), (2, 128), (1, 32)):
            with self.subTest(available_gib=gib), mock.patch.object(
                    module.psutil, 'virtual_memory', return_value=SimpleNamespace(available=gib * 2**30)):
                self.assertEqual(module._rrdb_band_rows(*args), expected)

    def test_stitching_buffers_are_bounded_by_one_tile_row(self):
        width, height, tile, scale = 77, 105, 32, 2
        pixels = np.random.default_rng(12).integers(0, 256, (height, width, 3), dtype=np.uint8)
        with (mock.patch.object(module.np, 'zeros', wraps=np.zeros) as zeros,
              mock.patch.object(module.np, 'empty', wraps=np.empty) as empty):
            actual = module.tiled_upscale(Nearest(scale), Image.fromarray(pixels), 'cpu', tile=tile, padding=4)
        output_width = width * scale
        for call in zeros.call_args_list + empty.call_args_list:
            shape = call.args[0]
            if isinstance(shape, tuple) and len(shape) >= 2 and shape[1] == output_width:
                self.assertLessEqual(shape[0], tile * scale, shape)
        expected = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
        np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_tile_boundaries_and_non_multiple_dimensions(self):
        pixels = np.arange(13 * 17 * 3, dtype=np.uint8).reshape(13, 17, 3)
        image = Image.fromarray(pixels)
        for scale in (2, 4):
            with self.subTest(scale=scale):
                actual = np.asarray(module.tiled_upscale(Nearest(scale), image, 'cpu', tile=8, padding=2))
                expected = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
                np.testing.assert_array_equal(actual, expected)

    def test_context_matches_full_inference_across_internal_borders(self):
        pixels = np.random.default_rng(42).integers(0, 256, (288, 320, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        model = ContextAverage()
        with torch.inference_mode():
            tensor = torch.from_numpy(pixels.astype(np.float32) / 255).permute(2, 0, 1)[None]
            expected = model(tensor)[0].permute(1, 2, 0).numpy()
        expected = (expected * 255 + 0.5).astype(np.uint8)
        actual = np.asarray(module.tiled_upscale(model, image, 'cpu'))
        self.assertLessEqual(np.abs(actual.astype(float) - expected).max(), 1)

    def test_global_statistics_do_not_create_hard_tile_seams(self):
        ramp = np.linspace(0, 255, 640, dtype=np.uint8)[None, :, None]
        image = Image.fromarray(np.broadcast_to(ramp, (120, 640, 3)).copy())
        actual = np.asarray(module.tiled_upscale(TileMean(), image, 'cpu')).astype(float)
        self.assertLessEqual(np.abs(np.diff(actual, axis=1)).max(), 4)

    def test_outer_borders_do_not_fade(self):
        image = Image.new('RGB', (277, 301), (241, 127, 13))
        actual = np.asarray(module.tiled_upscale(Nearest(2), image, 'cpu'))
        np.testing.assert_array_equal(actual, np.broadcast_to([241, 127, 13], actual.shape))

    def test_small_images_and_narrow_last_tiles_have_complete_coverage(self):
        for width, height in ((1, 1), (31, 19), (255, 257), (257, 255), (513, 641)):
            with self.subTest(width=width, height=height):
                pixels = np.random.default_rng(4).integers(0, 256, (height, width, 3), dtype=np.uint8)
                actual = np.asarray(module.tiled_upscale(Nearest(2), Image.fromarray(pixels), 'cpu'))
                expected = np.repeat(np.repeat(pixels, 2, axis=0), 2, axis=1)
                np.testing.assert_array_equal(actual, expected)

    def test_strided_models_keep_the_same_sampling_phase(self):
        # Multiples of four keep 2x2 means away from half-integer quantization
        # ties, so exact equality tests spatial phase rather than float rounding.
        pixels = np.random.default_rng(7).integers(0, 64, (513, 641, 3), dtype=np.uint8) * 4
        model = PhaseLocked()
        with torch.inference_mode():
            tensor = torch.from_numpy(pixels.astype(np.float32) / 255).permute(2, 0, 1)[None]
            expected = (model(tensor)[0].permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
        actual = np.asarray(module.tiled_upscale(model, Image.fromarray(pixels), 'cpu'))
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
