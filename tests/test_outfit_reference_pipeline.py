"""Clothing must not regress merely because auxiliary evidence is uncertain."""
import unittest

import numpy as np

from gpt_animeimage_denoise import auto_sam, fusion
from test_fusion import evidence
from test_auto_sam import CandidateProvider

POLICY={'semantic_repair':'anchored','outfit_reference':'m2f-white'}


class OutfitReferenceIntegration(unittest.TestCase):
    def run_fusion(self,ev):return fusion.fuse_evidence(ev,POLICY)

    def test_default_is_explicit_and_old_policies_keep_old_meaning(self):
        self.assertEqual(fusion._policy()['outfit_reference'],'m2f-white')
        self.assertEqual(fusion._policy({'semantic_repair':'anchored'})['outfit_reference'],'off')
        with self.assertRaises(ValueError):fusion._policy({'outfit_reference':'largest-mask'})

    def test_weak_white_clothing_does_not_disappear_into_strong_original_background(self):
        ev=evidence([[10]],original=[[0]])
        p=np.full((12,1,1),.001,np.float32);p[10]=.51;p[0]=.48;p/=p.sum(axis=0)
        ev['m2f_white']=np.log(p)
        result=self.run_fusion(ev)
        self.assertEqual(result['person_labels'][0,0],6)
        self.assertFalse(result['uncertain'][0,0])
        self.assertEqual(result['source_map'][0,0],25)

    def test_strong_view_conflict_keeps_white_clothing_with_diagnostic_conflict(self):
        result=self.run_fusion(evidence([[10]],original=[[0]]))
        self.assertEqual(result['person_labels'][0,0],6)
        self.assertTrue(result['conflict'][0,0])
        self.assertFalse(result['uncertain'][0,0])

    def test_see_wear_and_sam_cannot_expand_clothing_into_white_non_clothing(self):
        ev=evidence([[10,0,0]],original=[[10,0,0]]);ev['see_logits'][11]=4
        initial=self.run_fusion(ev)
        result=auto_sam.refine_regions(np.zeros((1,3,3),np.uint8),ev,initial,
            CandidateProvider([[True,True,True]]))
        self.assertEqual(result['person_labels'][0,0],6)
        self.assertFalse((result['person_labels'][0,1:]==6).any())

    def test_see_hair_requires_other_m2f_hair_evidence_to_replace_white_clothing(self):
        ev=evidence([[3,10,10]],original=[[3,0,0]]);ev['see_logits'][0]=4
        result=self.run_fusion(ev)
        np.testing.assert_array_equal(result['person_labels'],[[3,6,6]])
        ev['m2f_original']=evidence([[3,3,3]])['m2f_white']
        corrected=self.run_fusion(ev)
        np.testing.assert_array_equal(corrected['person_labels'],[[3,3,3]])

    def test_object_evidence_can_exclude_clothing_but_sam_cannot_swallow_unsupported_neighbor(self):
        ev=evidence([[10,10]],original=[[10,10]]);ev['see_logits'][18,0,0]=4
        initial=self.run_fusion(ev)
        result=auto_sam.refine_regions(np.zeros((1,2,3),np.uint8),ev,initial,
            CandidateProvider([[True,True]]))
        np.testing.assert_array_equal(result['objects'],[[True,False]])
        np.testing.assert_array_equal(result['person_labels'],[[0,6]])


if __name__=='__main__':unittest.main()
