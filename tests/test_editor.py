import unittest

import numpy as np

from gpt_animeimage_denoise.editor import DEFAULT_MODULES, MODULES, SelectionEditor
from gpt_animeimage_denoise.filters import blend, build_weight, gradient_preserving_candidate


class SelectionEditorTests(unittest.TestCase):
    def setUp(self):
        shape = (32, 40)
        self.rgb = np.random.default_rng(3).integers(0, 256, (*shape, 3), dtype=np.uint8)
        self.alpha = np.full(shape, 255, np.uint8)
        self.labels = np.zeros(shape, np.uint8)
        self.labels[:, 8:16] = MODULES['skin']
        self.labels[:, 16:24] = MODULES['hair']
        self.labels[:, 24:32] = MODULES['outfit']
        self.labels[:, 32:] = MODULES['eyes']
        self.labels[10, 11] = 255
        self.objects = np.zeros(shape, bool)
        self.objects[10, 12] = True
        self.unknown = np.zeros(shape, bool)
        self.unknown[10, 13] = True
        self.alpha[10, 14] = 0
        self.editor = SelectionEditor(self.rgb, self.alpha, self.labels, self.objects, self.unknown)

    def test_default_compositing_matches_pipeline(self):
        hard = self.objects | self.unknown | (self.labels == 255) | (self.alpha == 0)
        allowed = np.isin(self.labels, (1, 2, 3)) & ~hard
        scale = min(self.labels.shape) / 2048
        expected_weight = build_weight(allowed, ~allowed, self.rgb, 16 * scale, 5 * scale,
                                       structure_protection=.2)
        expected_weight[expected_weight < 1 / 255] = 0
        candidate = np.full(self.rgb.shape, .4, np.float32)
        restored = gradient_preserving_candidate(self.rgb, candidate, 12 * scale, restoration_gain=.25)
        rgb, weight = self.editor.render(candidate, DEFAULT_MODULES)
        np.testing.assert_array_equal(weight, expected_weight)
        np.testing.assert_array_equal(rgb, blend(self.rgb, restored, expected_weight))
        np.testing.assert_array_equal(rgb[hard], self.rgb[hard])

    def test_modules_brush_and_eraser_respect_hard_protection(self):
        self.assertTrue(self.editor.selection(('background', 'outfit'))[0, 0])
        self.assertTrue(self.editor.selection(('background', 'outfit'))[0, 28])
        self.editor.paint_stroke([[2, 10], [30, 10]], 1)
        self.editor.paint_stroke([[20, 10]], 2, erase=True)
        picked = self.editor.selection(DEFAULT_MODULES)
        self.assertTrue(picked[10, 2])
        self.assertTrue(picked[10, 28])
        self.assertFalse(picked[10, 20])
        self.assertTrue(self.editor.erased[10, 20])
        self.assertTrue(self.editor.objects[10, 12])
        self.assertTrue(self.editor.unknown[10, 11])
        self.assertFalse(picked[10, 11:15].any())
        self.editor.paint_stroke([[20, 10]], 1)
        self.assertTrue(self.editor.selection(DEFAULT_MODULES)[10, 20])
        self.assertFalse(self.editor.erased[10, 20])
        self.editor.clear_strokes()
        self.assertFalse(self.editor.selection(DEFAULT_MODULES)[10, 2])

    def test_correction_masks_reproduce_selection_relative_to_default(self):
        self.editor.paint_stroke([[34, 10]], 2)
        self.editor.paint_stroke([[18, 15]], 2, erase=True)
        chosen = ('hair', 'outfit')
        allow, protect = self.editor.correction_masks(chosen)
        default = np.isin(self.labels, (1, 2, 3)) & ~self.editor.hard_protection
        self.assertFalse((allow & protect).any())
        self.assertFalse((allow & self.editor.hard_protection).any())
        np.testing.assert_array_equal((default | allow) & ~protect, self.editor.selection(chosen))
