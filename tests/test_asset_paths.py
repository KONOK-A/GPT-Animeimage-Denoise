from pathlib import Path
import unittest
from unittest import mock

from gpt_animeimage_denoise.denoisers import describe_denoiser


class AssetPathsTest(unittest.TestCase):
    def test_sr_identity_tracks_the_model_root(self):
        with mock.patch.dict('os.environ', {'GPT_ANIMEIMAGE_DENOISE_MODEL_DIR': 'model-assets'}):
            identity = describe_denoiser('esrgan-l0', 'cpu')
        expected = Path('model-assets/RealESRGAN_x4plus_anime_6B.pth').resolve()
        self.assertEqual(Path(identity['model']['path']), expected)
