import sys
import unittest
from pathlib import Path

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
    def test_removed_cugan_is_rejected_even_with_an_injected_model(self):
        source = Image.new('RGB', (8, 8))
        for method in ('cugan60', 'cugan60_l0'):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'Unknown denoiser'):
                denoise.run(source, [method], {'cugan': TrackingUpscale()}, 'cpu')

    def test_direct_project_entry_uses_bounded_tiles(self):
        model = TrackingUpscale()
        pixels = np.random.default_rng(8).integers(0, 256, (513, 641, 3), dtype=np.uint8)
        output = denoise.upscale(model, Image.fromarray(pixels), 'cpu')
        self.assertGreater(len(model.shapes), 1)
        self.assertLessEqual(max(max(shape) for shape in model.shapes), 384)
        np.testing.assert_array_equal(np.asarray(output), np.repeat(np.repeat(pixels, 2, 0), 2, 1))


if __name__ == '__main__':
    unittest.main()
