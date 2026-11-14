import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


class CorrectionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sha = 'a' * 64

    def tearDown(self):
        self.temp.cleanup()

    def edit(self, **fields):
        path = self.root / 'edit.json'
        path.write_text(json.dumps(dict(version=2, source_sha256=self.sha,
                                       size=[8, 6], **fields)), encoding='utf-8')
        return path

    def test_identity_and_grid_must_match_before_loading_masks(self):
        from gpt_animeimage_denoise.corrections import load_edits
        path = self.edit()
        with self.assertRaisesRegex(ValueError, 'sha256'):
            load_edits(path, (6, 8), 'b' * 64)
        with self.assertRaisesRegex(ValueError, 'size'):
            load_edits(path, (8, 6), self.sha)

    def test_overlapping_manual_classes_and_object_actions_are_errors(self):
        from gpt_animeimage_denoise.corrections import load_edits
        Image.new('L', (8, 6), 255).save(self.root / 'mask.png')
        for fields in [dict(module_overrides={'skin': 'mask.png', 'hair': 'mask.png'}),
                       dict(object_add_mask='mask.png', object_remove_mask='mask.png')]:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    load_edits(self.edit(**fields), (6, 8), self.sha)

    def test_allow_never_unlocks_unknown_objects_or_protect(self):
        from gpt_animeimage_denoise.corrections import load_edits, apply_corrections
        Image.new('L', (8, 6), 255).save(self.root / 'allow.png')
        labels = np.ones((6, 8), np.uint8); labels[:, 0] = 255
        objects = np.zeros((6, 8), bool); objects[:, 1] = True
        protected = np.zeros_like(objects); protected[:, 2] = True
        Image.fromarray(protected.astype(np.uint8) * 255).save(self.root / 'protect.png')
        regions = dict(person_labels=labels, raw_labels=labels.copy(), allowed=~protected,
                       protected=protected, objects=objects, source_map=np.full((6, 8), 6, np.uint8))
        result = apply_corrections(regions, load_edits(self.edit(allow_mask='allow.png', protect_mask='protect.png'), (6, 8), self.sha))
        self.assertFalse(result['allowed'][:, :3].any())
        self.assertTrue(result['protected'][:, :3].all())
        self.assertTrue((result['source_map'][:, :2] == 6).all())
        self.assertTrue((result['source_map'][:, 2] == 13).all())
        np.testing.assert_array_equal(regions['person_labels'], labels)

    def test_explicit_part_and_object_remove_restore_reviewed_class_only(self):
        from gpt_animeimage_denoise.corrections import load_edits, apply_corrections
        mask = np.zeros((6, 8), np.uint8); mask[:, 1] = 255
        Image.fromarray(mask).save(self.root / 'mask.png')
        labels = np.full((6, 8), 255, np.uint8)
        objects = mask > 0
        regions = dict(person_labels=labels, raw_labels=labels.copy(), allowed=np.zeros_like(objects),
                       protected=np.ones_like(objects), objects=objects)
        edits = load_edits(self.edit(object_remove_mask='mask.png',
                                    module_overrides={'skin': 'mask.png'}), (6, 8), self.sha)
        result = apply_corrections(regions, edits)
        self.assertTrue(result['allowed'][:, 1].all())
        self.assertTrue((result['person_labels'][:, 1] == 1).all())
        self.assertFalse(result['objects'].any())
        self.assertTrue(result['protected'][:, 0].all())

    def test_manual_sam_validates_optional_candidate_and_normalized_points(self):
        from gpt_animeimage_denoise.corrections import load_edits
        region = dict(action='part', part='hair', positive=[[0.5, 0.5]], box=[0, 0, 1, 1])
        self.assertEqual(load_edits(self.edit(sam_regions=[region]), (6, 8), self.sha)['sam_regions'][0]['part'], 'hair')
        for bad in [dict(region, candidate=True), dict(region, positive=[[1.1, .5]]), dict(region, part='hat')]:
            with self.assertRaises(ValueError):
                load_edits(self.edit(sam_regions=[bad]), (6, 8), self.sha)

    def test_explicit_part_clears_stale_uncertainty_and_keeps_raw_diagnostic(self):
        from gpt_animeimage_denoise.corrections import load_edits, apply_corrections
        Image.new('L', (8, 6), 255).save(self.root / 'mask.png')
        labels = np.full((6, 8), 255, np.uint8); raw = np.full((6, 8), 6, np.uint8)
        regions = dict(person_labels=labels, raw_labels=raw, labels=raw,
                       allowed=np.zeros((6, 8), bool), protected=np.ones((6, 8), bool),
                       objects=np.zeros((6, 8), bool), uncertain=np.ones((6, 8), bool),
                       conflict=np.ones((6, 8), bool), source_map=np.zeros((6, 8), np.uint8),
                       policy={'source_codes': {'outside': 0}})
        result = apply_corrections(regions, load_edits(self.edit(module_overrides={'hair': 'mask.png'}), (6, 8), self.sha))
        self.assertFalse(result['uncertain'].any()); self.assertFalse(result['conflict'].any())
        self.assertTrue((result['labels'] == 6).all())
        self.assertTrue((result['person_labels'] == 3).all())
        self.assertNotEqual(result['source_map'][0, 0], 0)
        self.assertEqual(regions['policy'], {'source_codes': {'outside': 0}})

    def test_allow_expands_default_scope_without_relabeling_or_removing_hard_vetoes(self):
        from gpt_animeimage_denoise.corrections import load_edits, apply_corrections
        labels = np.tile(np.array([0, 1, 4, 5, 6, 255, 0, 1], np.uint8), (6, 1))
        objects = np.zeros((6, 8), bool); objects[:, 6] = True
        default_allowed = (labels == 1) & ~objects
        explicit_protect = np.zeros_like(objects); explicit_protect[:, 7] = True
        Image.fromarray(explicit_protect.astype(np.uint8) * 255).save(self.root / 'protect.png')
        Image.new('L', (8, 6), 255).save(self.root / 'allow.png')
        regions = dict(person_labels=labels.copy(), raw_labels=labels.copy(),
            allowed=default_allowed, protected=~default_allowed, objects=objects)
        for sam in [False, True]:
            with self.subTest(sam=sam):
                fields = {'protect_mask': 'protect.png'}
                if not sam:
                    fields['allow_mask'] = 'allow.png'
                edits = load_edits(self.edit(**fields), (6, 8), self.sha)
                sam_masks = [({'action': 'allow'}, np.ones((6, 8), bool))] if sam else []
                result = apply_corrections(regions, edits, sam_masks)
                self.assertTrue(result['allowed'][:, :5].all())
                self.assertFalse(result['allowed'][:, 5:].any())
                self.assertTrue(result['protected'][:, 5:].all())
                np.testing.assert_array_equal(result['person_labels'], labels)
