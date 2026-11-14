"""See evidence roles must hold through fusion and the SAM task boundary."""
import unittest

import numpy as np

from gpt_animeimage_denoise import auto_sam, fusion
from test_fusion import evidence
from test_auto_sam import CandidateProvider

SELECTIVE={'semantic_repair':'anchored','outfit_reference':'m2f-white','see_reference':'selective'}


class SelectiveSeeReference(unittest.TestCase):
    def test_default_and_saved_policy_meanings_are_explicit(self):
        self.assertEqual(fusion._policy()['see_reference'],'selective')
        self.assertEqual(fusion._policy({'semantic_repair':'anchored'})['see_reference'],'legacy')
        with self.assertRaises(ValueError):fusion._policy({'see_reference':'always-see'})

    def test_see_neck_cannot_create_unanchored_skin_via_sam(self):
        ev=evidence([[1,0]],original=[[1,0]])
        ev['m2f_white'][:,:,1]=np.log(np.float32(1/12));ev['m2f_original'][:,:,1]=ev['m2f_white'][:,:,1]
        ev['see_logits'][9,0,1]=4
        region=fusion.fuse_evidence(ev,SELECTIVE)
        self.assertFalse(region['_class_support'][1,0,1])
        result=auto_sam.refine_regions(np.zeros((1,2,3),np.uint8),ev,region,CandidateProvider([[False,True]]))
        self.assertNotEqual(result['person_labels'][0,1],1)
        self.assertEqual(result['person_labels'][0,0],1)

    def test_unconfirmed_wear_does_not_veto_high_priority_face(self):
        ev=evidence([[2,2]],original=[[2,2]])
        ev['m2f_white'][:,:,1]=np.log(np.float32(1/12));ev['m2f_original'][:,:,1]=ev['m2f_white'][:,:,1]
        ev['see_logits'][2]=4;ev['see_logits'][11,0,1]=4
        result=fusion.fuse_evidence(ev,SELECTIVE)
        np.testing.assert_array_equal(result['person_labels'],[[2,2]])
        self.assertFalse(result['_wear_support'][0,1])

    def test_unconfirmed_wear_cannot_cancel_object_in_m2f_background(self):
        ev=evidence([[0]],original=[[0]]);ev['see_logits'][18]=4;ev['see_logits'][11]=4
        result=fusion.fuse_evidence(ev,SELECTIVE)
        self.assertTrue(result['objects'][0,0])
        self.assertFalse(result['uncertain'][0,0])

    def test_stable_main_outfit_and_strict_wear_still_protect_worn_items(self):
        ev=evidence([[10]],original=[[10]]);ev['see_logits'][18]=4;ev['see_logits'][11]=4
        result=fusion.fuse_evidence(ev,SELECTIVE)
        self.assertFalse(result['objects'][0,0])
        self.assertEqual(result['person_labels'][0,0],6)

    def test_unanchored_see_hair_cannot_bypass_rejection_through_sam(self):
        ev=evidence([[3,0,0,0]],original=[[3,0,0,0]])
        ev['m2f_white'][:,:,3]=np.log(np.float32(1/12));ev['m2f_original'][:,:,3]=ev['m2f_white'][:,:,3]
        ev['see_logits'][0,0,3]=4
        region=fusion.fuse_evidence(ev,SELECTIVE)
        self.assertFalse(region['_class_support'][3,0,3])
        result=auto_sam.refine_regions(np.zeros((1,4,3),np.uint8),ev,region,
            CandidateProvider([[False,False,False,True]]))
        self.assertNotEqual(result['person_labels'][0,3],3)
        self.assertEqual(result['person_labels'][0,0],3)

    def test_accepted_original_view_eye_is_not_a_face_competitor(self):
        ev=evidence([[2]],original=[[2]])
        ev['see_original_logits']=np.full((19,1,1),-4,np.float32)
        ev['see_original_logits'][3]=4
        result=fusion.fuse_evidence(ev,SELECTIVE)
        self.assertEqual(result['person_labels'][0,0],4)
        self.assertTrue(result['_class_support'][4,0,0])
        self.assertFalse(result['_class_support'][2,0,0])


if __name__=='__main__':unittest.main()
