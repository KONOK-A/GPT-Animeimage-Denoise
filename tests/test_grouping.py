import unittest
import numpy as np
from gpt_animeimage_denoise.grouping import coarsen_v4, fuse_see_eyes, default_processing_mask, processing_regions, PERSON_GROUPS
from gpt_animeimage_denoise.filters import build_weight, blend

class GroupingTests(unittest.TestCase):
    def fixture(self):
        label=np.full((24,40),2,np.uint8)
        label[8:12,8:14]=4;label[8:12,25:31]=5
        label[3:5,8:14]=6;label[3:5,25:31]=7
        label[14:17,19:22]=8;label[19:21,17:24]=9
        return label
    def test_mapping(self):
        values=np.arange(12,dtype=np.uint8)[None,:]
        np.testing.assert_array_equal(coarsen_v4(values),[[0,1,2,3,4,4,3,3,2,5,6,6]])
    def test_features_face_and_one_eye_class(self):
        label=self.fixture();see=np.isin(label,[4,5,6,7,8]);before=label.copy()
        merged,info=fuse_see_eyes(label,see)
        self.assertTrue((merged[np.isin(label,[6,7])]==3).all())
        self.assertTrue((merged[label==8]==2).all())
        self.assertTrue((merged[label==9]==5).all())
        self.assertTrue((merged[np.isin(label,[4,5])]==4).all())
        np.testing.assert_array_equal(label,before)
    def test_use_complete_see_component_not_crop_to_v4_eye(self):
        label=self.fixture();see=np.zeros_like(label,bool)
        see[7:13,6:15]=True;see[7:13,24:33]=True
        merged,info=fuse_see_eyes(label,see)
        self.assertEqual(merged[9,6],4);self.assertEqual(merged[9,32],4)
        self.assertEqual(info['fallback_v4_eye_pixels'],0)
    def test_missing_eye_falls_back_and_is_reported(self):
        label=self.fixture();see=np.zeros_like(label,bool);see[8:12,8:14]=True
        merged,info=fuse_see_eyes(label,see)
        self.assertTrue((merged[label==5]==4).all());self.assertEqual(info['fallback_v4_eye_pixels'],24)
    def test_disconnected_false_brow_and_props_not_eye(self):
        label=self.fixture();label[:2,:]=3;label[:,35:]=11
        see=np.isin(label,[4,5]);see[0,8:14]=True;see[8:12,35:39]=True
        merged,info=fuse_see_eyes(label,see)
        self.assertEqual(merged[0,10],3);self.assertEqual(merged[9,36],6)
    def test_no_v4_eye_can_use_face_supported_see(self):
        label=np.full((16,20),2,np.uint8);see=np.zeros_like(label,bool);see[5:8,5:8]=True
        merged,info=fuse_see_eyes(label,see)
        self.assertTrue((merged[see]==4).all());self.assertTrue(info['used_face_anchor_fallback'])
    def test_no_other_regions_changed(self):
        label=self.fixture();label[22:]=10;see=np.isin(label,[4,5]);see[22:]=True
        merged,info=fuse_see_eyes(label,see);base=coarsen_v4(label)
        np.testing.assert_array_equal(merged[base!=4],base[base!=4])
    def test_validation(self):
        with self.assertRaises(ValueError):coarsen_v4(np.array([[12]],np.uint8))
        with self.assertRaises(ValueError):fuse_see_eyes(self.fixture(),np.zeros((1,1),bool))
        with self.assertRaises(ValueError):fuse_see_eyes(self.fixture(),np.zeros_like(self.fixture()))
    def test_default_processes_only_skin_face_hair(self):
        groups=np.arange(7,dtype=np.uint8)[None,:]
        np.testing.assert_array_equal(default_processing_mask(groups),[[False,True,True,True,False,False,False]])
        self.assertEqual(len(PERSON_GROUPS),6)
        with self.assertRaises(ValueError):default_processing_mask(np.array([[7]],np.uint8))
    def test_current_policy_keeps_mouth_independent(self):
        label=self.fixture();result=processing_regions(label,np.isin(label,[4,5,6,7,8,9]))
        self.assertTrue(result['allowed'][np.isin(label,[6,7,8])].all())
        self.assertFalse(result['allowed'][label==9].any())
        self.assertTrue(result['protected'][label==9].all())
        self.assertEqual(result['policy']['model_version'],4)
        self.assertFalse(result['policy']['offline_bundle'])
    def test_objects_override_processing_without_relabeling(self):
        label=np.arange(12,dtype=np.uint8)[None,:]
        eyes=np.zeros_like(label,bool);objects=np.ones_like(label,bool)
        baseline=processing_regions(label,eyes)
        result=processing_regions(label,eyes,see_objects=objects)
        np.testing.assert_array_equal(result['labels'],baseline['labels'])
        np.testing.assert_array_equal(result['object_exclusion'],objects)
        self.assertFalse(result['allowed'].any())
        self.assertTrue(result['protected'].all())
        self.assertFalse(result['character_allowed'].any())
        report=result['object_report']
        self.assertTrue(report['enabled']);self.assertEqual(report['object_pixels'],12)
        self.assertEqual(report['removed_processing_pixels'],6)
        self.assertEqual(report['removed_processing_pixels_by_group']['skin'],1)
        self.assertEqual(report['removed_processing_pixels_by_group']['face'],2)
        self.assertEqual(report['removed_processing_pixels_by_group']['hair'],3)
        self.assertEqual(report['removed_processing_pixels_by_group']['eyes'],0)
        self.assertEqual(report['object_pixels_by_group']['outfit'],2)
        np.testing.assert_array_equal(label,np.arange(12,dtype=np.uint8)[None,:])
        self.assertTrue(objects.all())
    def test_objects_are_exact_mask_and_none_preserves_previous_behavior(self):
        label=self.fixture();eyes=np.isin(label,[4,5]);objects=np.zeros_like(label,bool)
        objects[15,20]=True
        baseline=processing_regions(label,eyes)
        expected,_=fuse_see_eyes(label,eyes)
        np.testing.assert_array_equal(baseline['labels'],expected)
        np.testing.assert_array_equal(baseline['allowed'],default_processing_mask(expected))
        np.testing.assert_array_equal(baseline['character_allowed'],expected!=0)
        self.assertFalse(baseline['object_exclusion'].any())
        self.assertFalse(baseline['object_report']['enabled'])
        result=processing_regions(label,eyes,objects)
        np.testing.assert_array_equal(result['allowed'],baseline['allowed']&~objects)
        np.testing.assert_array_equal(result['character_allowed'],(expected!=0)&~objects)
        np.testing.assert_array_equal(result['protected'],~result['allowed'])
        self.assertTrue(result['allowed'][15,19]);self.assertTrue(result['allowed'][15,21])
        self.assertEqual(result['object_report']['removed_processing_pixels'],1)
    def test_objects_require_aligned_boolean_mask(self):
        label=self.fixture();eyes=np.zeros_like(label,bool)
        with self.assertRaises(ValueError):processing_regions(label,eyes,np.zeros((1,1),bool))
        with self.assertRaises(ValueError):processing_regions(label,eyes,np.zeros_like(label))
    def test_person_labels_remove_objects_from_every_exported_module(self):
        label=np.tile(np.arange(12,dtype=np.uint8),(2,1))
        eyes=np.zeros_like(label,bool);objects=np.zeros_like(label,bool);objects[0]=True
        baseline=processing_regions(label,eyes)
        result=processing_regions(label,eyes,objects)
        self.assertTrue('person_labels' in result,'Missing object-excluded person labels')
        self.assertTrue((result['person_labels'][objects]==0).all())
        np.testing.assert_array_equal(result['person_labels'][~objects],baseline['labels'][~objects])
        np.testing.assert_array_equal(result['labels'],baseline['labels'])
        np.testing.assert_array_equal(result['character_allowed'],result['person_labels']!=0)
        for group_id in range(1,7):
            exported=result['person_labels']==group_id
            self.assertFalse(exported[objects].any())
            self.assertTrue(exported[~objects].any())
        outfit=result['person_labels']==6
        self.assertFalse(outfit[0,10:].any());self.assertTrue(outfit[1,10:].all())
    def test_person_labels_without_objects_preserve_all_module_labels(self):
        label=self.fixture();eyes=np.isin(label,[4,5])
        result=processing_regions(label,eyes)
        self.assertTrue('person_labels' in result,'Missing object-excluded person labels')
        np.testing.assert_array_equal(result['person_labels'],result['labels'])
        result['person_labels'][0,0]=0
        self.assertEqual(result['labels'][0,0],2)
    def test_objects_remain_original_after_feathering_and_blending(self):
        label=np.full((32,32),3,np.uint8);eyes=np.zeros_like(label,bool)
        objects=np.zeros_like(label,bool);objects[10:22,10:22]=True
        regions=processing_regions(label,eyes,objects)
        original=np.full((*label.shape,3),120,np.uint8)
        candidate=np.full(original.shape,.8,np.float32)
        weight=build_weight(regions['allowed'],regions['protected'],original,
                            feather_radius=4,structure_radius=2)
        result=blend(original,candidate,weight)
        self.assertTrue((weight[objects]==0).all())
        np.testing.assert_array_equal(result[objects],original[objects])
        self.assertTrue((result[~objects]!=original[~objects]).any())

if __name__=='__main__':unittest.main()
