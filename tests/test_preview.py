from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from gpt_animeimage_denoise import fusion_pipeline
from test_fusion_pipeline import L0, Models, regions


class PreviewTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source.png'
        self.rgb = np.random.default_rng(21).integers(100, 150, (32, 40, 3), dtype=np.uint8)
        self.alpha = np.full((32, 40), 128, np.uint8)
        self.alpha[:, 30:32] = 0
        Image.fromarray(np.dstack((self.rgb, self.alpha))).save(self.source, icc_profile=b'fixture-icc')

    def run_case(self, name, **options):
        output = self.root / name
        with mock.patch.object(fusion_pipeline, 'fuse_evidence', side_effect=regions), \
             mock.patch.object(fusion_pipeline, 'refine_regions', side_effect=lambda rgb, ev, reg, models, **_: reg):
            fusion_pipeline.process_image(self.source, output, Models(), case_id='case', l0_fn=L0(), **options)
        return output / 'case'

    def test_preview_matches_full_reprocess_and_preserves_case(self):
        from gpt_animeimage_denoise.preview import preview_case

        case = self.run_case('base')
        report = (case / 'report.json').read_bytes()
        options = {'strength': .65, 'restoration_gain': .35, 'sigma': 240}
        output = self.root / 'previews/example.png'
        preview_case(case, output, **options)
        expected = self.run_case('expected', **options)
        with Image.open(output) as actual, Image.open(expected / 'result.png') as full:
            np.testing.assert_array_equal(np.array(actual), np.array(full))
            self.assertEqual(actual.info['icc_profile'], b'fixture-icc')
        self.assertEqual((case / 'report.json').read_bytes(), report)

    def test_whole_image_preview_preserves_the_source_grid_and_metadata(self):
        from gpt_animeimage_denoise.preview import preview_case

        segmented = self.run_case('shared')
        segmented_report = (segmented / 'report.json').read_bytes()
        options = {'profile': 'whole', 'case_id': 'case', 'l0_fn': L0(), 'structure_protection': 0}
        report = fusion_pipeline.process_image(self.source, segmented.parent, None, **options)
        case = segmented.parent / report['folder']
        self.assertEqual(case.name, 'case-whole')
        self.assertEqual((segmented / 'report.json').read_bytes(), segmented_report)
        self.assertFalse((case / 'person_labels.png').exists())
        np.testing.assert_array_equal(np.load(case / 'weight.npy'), self.alpha > 0)
        with Image.open(case / 'result.png') as image:
            pixels = np.asarray(image)
            self.assertEqual(image.size, (40, 32))
            self.assertEqual(image.info['icc_profile'], b'fixture-icc')
            np.testing.assert_array_equal(pixels[..., 3], self.alpha)
            np.testing.assert_array_equal(pixels[..., :3][self.alpha == 0], self.rgb[self.alpha == 0])
            self.assertTrue(np.any(pixels[..., :3][self.alpha > 0] != self.rgb[self.alpha > 0]))
        self.assertTrue(fusion_pipeline.process_image(self.source, segmented.parent, None, **options)['resumed'])

        target = self.root / 'whole-preview.png'
        preview_case(case, target, strength=.5)
        expected = fusion_pipeline.process_image(self.source, self.root / 'expected', None, strength=.5, **options)
        with Image.open(target) as actual, Image.open(self.root / 'expected' / expected['folder'] / 'result.png') as full:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(full))
            self.assertEqual(actual.info['icc_profile'], b'fixture-icc')

    def test_rejects_changed_weight_and_native_destination(self):
        from gpt_animeimage_denoise.preview import preview_case

        case = self.run_case('base')
        with self.assertRaises((ValueError, FileExistsError)):
            preview_case(case, case / 'result.png')
        weight_path = case / 'weight.npy'
        weight = np.load(weight_path, allow_pickle=False)
        weight[0, 0] = 1
        np.save(weight_path, weight, allow_pickle=False)
        target = self.root / 'preview.png'
        with self.assertRaisesRegex(ValueError, 'weight.npy'):
            preview_case(case, target)
        self.assertFalse(target.exists())

    def test_nonfull_strength_still_allows_restoration_preview(self):
        from gpt_animeimage_denoise.preview import preview_case

        case = self.run_case('half', strength=.5)
        target = self.root / 'preview.png'
        with self.assertRaisesRegex(ValueError, 'strength 1'):
            preview_case(case, target, strength=.9)
        preview_case(case, target, restoration_gain=.4)
        self.assertTrue(target.is_file())
