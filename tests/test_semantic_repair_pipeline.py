"""Production wiring contracts for anchored See evidence and legacy policies."""
import unittest

import numpy as np

from gpt_animeimage_denoise import fusion, fusion_pipeline, auto_sam
from test_fusion import evidence
from test_auto_sam import CandidateProvider


class SemanticRepairIntegration(unittest.TestCase):
    @staticmethod
    def weaken(ev, x, favored, runner_up):
        for key in ('m2f_white', 'm2f_original'):
            if ev[key] is None:
                continue
            ev[key][:, 0, x] = np.log(np.float32(.001))
            ev[key][favored, 0, x] = np.log(np.float32(.56))
            ev[key][runner_up, 0, x] = np.log(np.float32(.43))

    def test_guarded_neck_fills_compatible_shadow_but_not_hair_or_isolated_skin(self):
        classes = [[1, 2, 2, 3, 2, 0, 2]]
        ev = evidence(classes, original=classes)
        for x in (1, 2, 4, 6):
            self.weaken(ev, x, 2, 1)
        ev['see_logits'][9, 0, :5] = 4
        ev['see_original_logits'] = ev['see_logits'].copy()
        ev['see_original_logits'][10, 0, 2] = 4
        ev['see_logits'][9, 0, 6] = 4
        result = fusion.fuse_evidence(ev)
        np.testing.assert_array_equal(result['person_labels'], [[1, 1, 255, 3, 255, 0, 255]])
        self.assertEqual(result['source_map'][0, 1], result['policy']['source_codes']['see_neck'])
        self.assertEqual(result['semantic_repairs']['parts']['neck']['adopted_pixels'], 1)
        historical = fusion.fuse_evidence(ev, {**result['policy'], 'part_cooperation': 'legacy'})
        self.assertEqual(historical['person_labels'][0, 1], 255)

    def test_guarded_hair_requires_a_local_weak_primary_proposal(self):
        classes = [[3, 3, 2, 10, 0]]
        ev = evidence(classes, original=classes)
        self.weaken(ev, 1, 3, 2)
        self.weaken(ev, 2, 2, 3)
        ev['see_logits'][0, 0, :5] = 4
        ev['see_original_logits'] = ev['see_logits'].copy()
        result = fusion.fuse_evidence(ev)
        np.testing.assert_array_equal(result['person_labels'], [[3, 3, 255, 6, 0]])

    def test_guarded_hair_keeps_dual_view_extensions_connected_to_a_known_lock(self):
        classes = [[3] * 7 + [0, 3, 3]]
        ev = evidence(classes, original=classes)
        for x in (2, 3, 4, 5, 6, 8, 9):
            self.weaken(ev, x, 3, 2)
        ev['see_logits'][0] = 4
        ev['see_original_logits'] = ev['see_logits'].copy()
        result = fusion.fuse_evidence(ev)
        np.testing.assert_array_equal(result['person_labels'], [[3] * 7 + [0, 255, 255]])

    def test_guarded_eyes_do_not_follow_a_detached_eyebrow_or_trim_for_a_fragment(self):
        classes = np.full((9, 18), 2, np.uint8)
        classes[4:6, 3:6] = 4
        classes[4:6, 12:15] = 5
        ev = evidence(classes, original=classes)
        ev['see_logits'][3, 3, 3:6] = 4
        ev['see_logits'][3, 4, 5] = 4
        ev['see_logits'][3, 3, 12:15] = 4
        ev['see_logits'][2] = 4
        result = fusion.fuse_evidence(ev)
        self.assertTrue((result['person_labels'][4:6, 3:6] == 4).all())
        self.assertTrue((result['person_labels'][4:6, 12:15] == 4).all())
        self.assertTrue((result['person_labels'][3, 12:15] == 2).all())

    def test_guarded_mouth_only_trims_when_one_matching_contour_covers_the_core(self):
        classes = np.full((7, 16), 2, np.uint8)
        classes[3, 3:11] = 9
        ev = evidence(classes, original=classes)
        ev['see_logits'][2] = 4
        ev['see_logits'][8, 3, 4:6] = 4
        ev['see_logits'][8, 2, 7] = 4
        weak = fusion.fuse_evidence(ev)
        self.assertTrue((weak['person_labels'][3, 3:11] == 5).all())
        self.assertEqual(weak['person_labels'][2, 7], 2)
        ev['see_logits'][8, 3, 6:10] = 4
        repaired = fusion.fuse_evidence(ev)
        self.assertTrue((repaired['person_labels'][3, 4:10] == 5).all())
        self.assertTrue((repaired['person_labels'][3, [3, 10]] == 2).all())

    def test_guarded_mouth_can_trim_a_narrow_coherent_dual_view_contour(self):
        classes = np.full((7, 16), 2, np.uint8)
        classes[2:5, 3:11] = 9
        ev = evidence(classes, original=classes)
        ev['see_logits'][2] = 4
        ev['see_logits'][8, 3, 3:11] = 4
        ev['see_original_logits'] = ev['see_logits'].copy()
        result = fusion.fuse_evidence(ev)
        self.assertTrue((result['person_labels'][3, 3:11] == 5).all())
        self.assertTrue((result['person_labels'][[2, 4], 3:11] == 2).all())
        del ev['see_original_logits']
        fallback = fusion.fuse_evidence(ev)
        self.assertTrue((fallback['person_labels'][2:5, 3:11] == 5).all())

    def test_original_see_grid_and_dtype_are_validated_when_present(self):
        ev=evidence([[3,3]])
        for value in (np.zeros((19,1,1),np.float32),
                      np.zeros((19,1,2),np.float16),
                      np.full((19,1,2),np.nan,np.float32)):
            ev['see_original_logits']=value
            with self.subTest(shape=value.shape,dtype=value.dtype),self.assertRaises(ValueError):
                fusion.fuse_evidence(ev)

    def test_declared_original_see_view_cannot_use_white_only_cache(self):
        ev=evidence([[3]])
        ev['provenance']={'see':{'views':['white','original']}}
        ev['m2f_white']=np.zeros((12,768,768),np.float32)
        ev['m2f_original']=ev['m2f_white'].copy()
        with self.assertRaises(ValueError):fusion_pipeline._validate_evidence(ev,(1,1),True)

    def test_requested_dual_see_identity_cannot_accept_white_only_producer(self):
        ev=evidence([[3]])
        ev['provenance']={'see':{'views':['white']}}
        ev['m2f_white']=np.zeros((12,768,768),np.float32)
        ev['m2f_original']=ev['m2f_white'].copy()
        with self.assertRaises(ValueError):fusion_pipeline._validate_evidence(ev,(1,1),True)

    def test_legacy_explicit_policy_keeps_frozen_semantics(self):
        ev=evidence([[3]],original=[[3]])
        ev['see_logits'][11]=4
        old=fusion.fuse_evidence(ev,{'version':'fusion-v1'})
        new=fusion.fuse_evidence(ev,{'semantic_repair':'anchored'})
        self.assertEqual(old['person_labels'][0,0],255)
        self.assertEqual(new['person_labels'][0,0],3)
        self.assertFalse(new['uncertain'][0,0])
        self.assertTrue(new['conflict'][0,0])
        self.assertIn('semantic_repairs',new)

    def test_default_policy_is_explicit_and_repair_parts_are_validated(self):
        self.assertEqual(fusion._policy()['semantic_repair'],'anchored')
        self.assertEqual(fusion._policy({'version':'fusion-v1'})['semantic_repair'],'off')
        for policy in ({'semantic_repair':'guess'},
                       {'semantic_repair':'anchored','see_repair_parts':['background']},
                       {'semantic_repair':'anchored','see_repair_parts':'hair'}):
            with self.subTest(policy=policy),self.assertRaises(ValueError):fusion._policy(policy)

    def test_object_prompts_do_not_use_protected_body_as_positive_seeds(self):
        ev=evidence([[1,3,10]],original=[[1,3,10]])
        ev['see_logits'][18]=4
        regions=fusion.fuse_evidence(ev,{'semantic_repair':'anchored'})
        tasks,*_=auto_sam._tasks(regions,regions['policy'])
        objects=[t for t in tasks if t['action']=='object']
        self.assertEqual(len(objects),1)
        np.testing.assert_array_equal(objects[0]['support'],[[False,False,True]])
        self.assertTrue((regions['person_labels'][0,:2]==[1,3]).all())

    def test_object_wear_ambiguity_remains_protected_through_sam_completion(self):
        ev=evidence([[10,10]],original=[[10,10]])
        ev['see_logits'][18]=4;ev['see_logits'][11,0,1]=4
        regions=fusion.fuse_evidence(ev,{'semantic_repair':'anchored'})
        result=auto_sam.refine_regions(np.zeros((1,2,3),np.uint8),ev,regions,
            CandidateProvider([[True,True]]))
        np.testing.assert_array_equal(result['objects'],[[True,False]])
        np.testing.assert_array_equal(result['person_labels'],[[0,6]])

    def test_stable_outfit_disagreement_is_informational_and_never_a_sam_seed(self):
        ev=evidence([[10,10,0,10]],original=[[10,10,0,10]])
        ev['see_logits'][18,0,:3]=4
        ev['see_original_logits']=ev['see_logits'].copy()
        regions=fusion.fuse_evidence(ev)
        np.testing.assert_array_equal(regions['person_labels'],[[6,6,0,6]])
        np.testing.assert_array_equal(regions['suspected_objects'],[[True,True,False,False]])
        tasks,*_=auto_sam._tasks(regions,regions['policy'])
        objects=[task for task in tasks if task['action']=='object']
        self.assertEqual(len(objects),1)
        np.testing.assert_array_equal(objects[0]['support'],[[False,False,True,False]])
        np.testing.assert_array_equal(objects[0]['protect'],[[True,True,False,True]])
        result=auto_sam.refine_regions(np.zeros((1,4,3),np.uint8),ev,regions,
            CandidateProvider([[True,True,True,True]]))
        np.testing.assert_array_equal(result['objects'],[[False,False,True,False]])


if __name__=='__main__':unittest.main()
