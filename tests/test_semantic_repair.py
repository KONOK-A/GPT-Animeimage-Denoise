"""Anchored evidence repairs keep model meanings and primary fallbacks explicit."""
import importlib
import unittest

import numpy as np


def sample(labels, *, strong=None):
    labels=np.asarray(labels,np.uint8)
    strong=np.ones(labels.shape,bool) if strong is None else np.asarray(strong,bool)
    raw=np.array([0,1,2,3,4,9,10],np.uint8)[labels]
    support=np.stack([(labels==group)&strong for group in range(7)])
    shape=labels.shape
    ev={'see_logits':np.full((19,*shape),-4,np.float32)}
    regions={'person_labels':labels.copy(),'raw_labels':labels.copy(),
        'objects':np.zeros(shape,bool),'uncertain':np.zeros(shape,bool),
        'conflict':np.zeros(shape,bool),'consensus':np.zeros(shape,bool),
        'source_map':np.ones(shape,np.uint8),'_class_support':support.copy(),
        '_m2f_class_support':support.copy(),'_trusted_cores':support.copy(),
        '_wear_support':np.zeros(shape,bool),'_object_support':np.zeros(shape,bool),
        '_view_conflict':np.zeros(shape,bool)}
    return ev,regions,labels,strong,raw


class SemanticRepairTests(unittest.TestCase):
    def setUp(self):
        try:self.module=importlib.import_module('gpt_animeimage_denoise.semantic_repair')
        except ModuleNotFoundError:self.fail('Anchored semantic repair module is missing')
        self.policy={'see':1.,'see_repair_parts':['face','mouth','hair','objects'],
            'source_codes':{'see_face':20,'see_mouth':21,'see_hair':22,
                            'see_object_repair':23,'primary_fallback':24}}

    def run_repair(self, data, **kwargs):
        ev,r,primary,strong,raw=data
        return self.module.apply_see_repairs(ev,r,primary_labels=primary,
            primary_strong=strong,primary_raw=raw,policy=kwargs.get('policy',self.policy))

    def test_primary_auxiliary_disagreement_falls_back_without_hiding_conflict(self):
        data=sample([[3,6,1]])
        ev,r,_,_,_=data;r['person_labels'][:]=255;r['uncertain'][:]=True;r['conflict'][:]=True
        ev['see_logits'][11,0,0]=4
        result=self.run_repair(data)
        np.testing.assert_array_equal(result['person_labels'],[[3,6,1]])
        self.assertFalse(result['uncertain'].any());self.assertTrue(result['conflict'].all())
        self.assertTrue((result['source_map']==24).all())

    def test_missing_primary_evidence_stays_unknown(self):
        data=sample([[0]],strong=[[False]])
        data[1]['person_labels'][:]=255;data[1]['uncertain'][:]=True
        result=self.run_repair(data)
        self.assertEqual(result['person_labels'][0,0],255)
        self.assertTrue(result['uncertain'][0,0])

    def test_conflicting_m2f_views_are_not_generically_restored(self):
        data=sample([[3]])
        data[1]['person_labels'][:]=255;data[1]['uncertain'][:]=True
        data[1]['_view_conflict'][:]=True
        result=self.run_repair(data)
        self.assertEqual(result['person_labels'][0,0],255)

    def test_face_fills_an_adjacent_weak_gap_but_preserves_features_and_hair(self):
        labels=np.zeros((16,20),np.uint8);labels[4:12,6:13]=2
        labels[5,7]=4;labels[10,9]=5;labels[3,6:13]=3
        strong=labels!=0;data=sample(labels,strong=strong)
        ev,r,_,_,raw=data;raw[4,8]=6;raw[9,9]=8
        r['person_labels'][4:12,13]=255;r['uncertain'][4:12,13]=True
        ev['see_logits'][2,3:12,6:14]=4
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][4:12,13]==2).all())
        self.assertEqual(result['person_labels'][5,7],4);self.assertEqual(result['person_labels'][10,9],5)
        self.assertTrue((result['person_labels'][3,6:13]==3).all())
        self.assertEqual(result['source_map'][6,13],20)

    def test_face_cannot_fill_unanchored_or_distant_background(self):
        labels=np.zeros((16,24),np.uint8);labels[4:10,3:8]=2
        data=sample(labels,strong=labels!=0);data[0]['see_logits'][2,4:10,3:20]=4
        result=self.run_repair(data)
        self.assertEqual(result['person_labels'][6,18],0)
        self.assertTrue(result['semantic_repairs'])

    def test_weak_nose_uses_the_face_group_instead_of_being_blocked_as_another_feature(self):
        labels=np.full((12,14),2,np.uint8);strong=np.ones(labels.shape,bool);strong[7,7]=False
        data=sample(labels,strong=strong);data[4][7,7]=8
        data[1]['person_labels'][7,7]=255;data[1]['uncertain'][7,7]=True
        data[0]['see_logits'][2,:,:]=4
        result=self.run_repair(data)
        self.assertEqual(result['person_labels'][7,7],2)
        self.assertFalse(result['uncertain'][7,7])

    def test_mouth_adopts_anchored_component_and_does_not_merge_into_face(self):
        labels=np.full((20,24),2,np.uint8);labels[13:15,10:13]=5
        data=sample(labels);data[0]['see_logits'][8,13:15,9:14]=4
        data[0]['see_logits'][2,:,:]=4
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][13:15,9:14]==5).all())
        self.assertEqual(result['source_map'][13,9],21)
        self.assertTrue(result['_class_support'][5,13,9])
        self.assertEqual(int(result['_class_support'][:,13,9].sum()),1)

    def test_false_mouth_inside_face_far_from_existing_mouth_is_rejected(self):
        labels=np.full((40,40),2,np.uint8);labels[29:31,18:22]=5
        data=sample(labels);data[0]['see_logits'][8,29:31,17:23]=4
        data[0]['see_logits'][8,5:7,18:20]=4
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][5:7,18:20]==2).all())
        self.assertTrue((result['person_labels'][29:31,17:23]==5).all())

    def test_mouth_cannot_overwrite_eye_brow_nose_or_hair(self):
        labels=np.full((16,18),2,np.uint8);labels[8,7]=5;labels[8,8]=4;labels[8,11]=3
        data=sample(labels);data[4][8,9]=6;data[4][8,10]=8
        data[0]['see_logits'][8,8,6:12]=4
        result=self.run_repair(data)
        self.assertEqual(result['person_labels'][8,8],4)
        self.assertEqual(result['person_labels'][8,9],2)
        self.assertEqual(result['person_labels'][8,10],2)
        self.assertEqual(result['person_labels'][8,11],3)

    def test_missing_see_mouth_preserves_primary_mouth(self):
        labels=np.full((8,10),2,np.uint8);labels[5,4:6]=5
        data=sample(labels);result=self.run_repair(data)
        np.testing.assert_array_equal(result['person_labels'],labels)

    def test_matching_see_mouth_can_trim_primary_overselection_only_with_face_evidence(self):
        labels=np.full((20,24),2,np.uint8);labels[12:15,8:16]=5
        data=sample(labels);data[0]['see_logits'][8,12:15,9:15]=4
        data[0]['see_logits'][2,12:15,8:16]=4
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][12:15,9:15]==5).all())
        self.assertTrue((result['person_labels'][12:15,[8,15]]==2).all())

    def test_tiny_see_mouth_fragment_does_not_erase_primary_mouth(self):
        labels=np.full((24,28),2,np.uint8);labels[14:18,8:20]=5
        data=sample(labels);data[0]['see_logits'][8,15,13]=4;data[0]['see_logits'][2,:,:]=4
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][14:18,8:20]==5).all())

    def test_mouth_size_allowance_cannot_borrow_a_far_away_large_mouth_anchor(self):
        labels=np.full((64,80),2,np.uint8);labels[10,10]=5;labels[40:48,60:70]=5
        data=sample(labels);data[0]['see_logits'][8,9,9:19]=4
        result=self.run_repair(data)
        self.assertEqual(result['person_labels'][9,12],2)

    def test_see_only_mouth_does_not_self_anchor_through_class_support(self):
        labels=np.full((12,12),2,np.uint8);data=sample(labels)
        data[0]['see_logits'][8,7:9,5:7]=5
        data[1]['_class_support'][5,7:9,5:7]=True
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][7:9,5:7]==2).all())

    def test_two_anchored_long_hair_components_extend_without_head_crop(self):
        labels=np.full((24,28),6,np.uint8);labels[2:5,3:5]=3;labels[2:5,22:24]=3
        data=sample(labels,strong=labels==3);data[0]['see_logits'][0,2:22,3:5]=4;data[0]['see_logits'][0,2:22,22:24]=4
        result=self.run_repair(data)
        self.assertEqual(result['person_labels'][21,3],3);self.assertEqual(result['person_labels'][21,23],3)
        self.assertTrue((result['person_labels'][labels==3]==3).all())

    def test_see_only_hair_does_not_self_anchor_through_class_support(self):
        data=sample(np.full((16,20),6,np.uint8))
        data[0]['see_logits'][0,2:14,5:7]=5;data[1]['_class_support'][3,2:14,5:7]=True
        result=self.run_repair(data)
        self.assertTrue((result['person_labels']==6).all())

    def test_hair_resolves_a_view_conflict_when_an_m2f_view_supplies_anchor(self):
        labels=np.full((16,20),6,np.uint8);data=sample(labels)
        ev,r,_,_,_=data;ev['see_logits'][0,2:14,5:7]=5
        r['_m2f_class_support'][3,2:14,5:7]=True
        r['_view_conflict'][2:14,5:7]=True;r['person_labels'][2:14,5:7]=255;r['uncertain'][2:14,5:7]=True
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][2:14,5:7]==3).all())
        self.assertFalse(result['uncertain'][2:14,5:7].any())
        self.assertFalse(result['_view_conflict'][2:14,5:7].any())

    def test_hair_repair_preserves_scarf_worn_accessory_and_reliable_face(self):
        labels=np.full((18,20),6,np.uint8);labels[2:5,5:7]=3;labels[7:9,5:7]=2
        data=sample(labels);ev=data[0];ev['see_logits'][0,2:16,5:7]=5
        ev['see_logits'][10,10:12,5:7]=5;ev['see_logits'][1,13:15,5:7]=5
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][7:9,5:7]==2).all())
        self.assertTrue((result['person_labels'][10:12,5:7]==6).all())
        self.assertTrue((result['person_labels'][13:15,5:7]==6).all())

    def test_hair_to_outfit_requires_view_disagreement_and_independent_wear(self):
        labels=np.full((16,20),6,np.uint8);labels[2:12,5:9]=3
        data=sample(labels);ev,r,_,_,raw=data
        r['_view_conflict'][6:9,8]=True;r['_m2f_class_support'][6,6:9,8]=True
        ev['see_logits'][11,6:9,8]=4
        ev['see_logits'][11,3:5,8]=4  # No M2F view disagreement here.
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][6:9,8]==6).all())
        self.assertTrue((result['person_labels'][3:5,8]==3).all())

    def test_hair_cannot_swallow_stable_outfit_core_when_see_mislabels_a_glove(self):
        labels=np.full((32,40),6,np.uint8);labels[:,:4]=3
        data=sample(labels);data[0]['see_logits'][0,:,2:28]=5
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][:,4]==3).all())  # One-pixel boundary band in this fixture.
        self.assertTrue((result['person_labels'][:,5:28]==6).all())
        self.assertGreater(result['semantic_repairs']['parts']['hair']['blocked_outfit_core'],0)

    def test_another_m2f_hair_view_can_correct_deep_outfit_proposal(self):
        labels=np.full((32,40),6,np.uint8);labels[:,:4]=3
        data=sample(labels);ev,r,_,_,_=data;ev['see_logits'][0,:,2:28]=5
        r['_m2f_class_support'][3,:,5:28]=True;r['_view_conflict'][:,5:28]=True
        r['person_labels'][:,5:28]=255;r['uncertain'][:,5:28]=True
        result=self.run_repair(data)
        self.assertTrue((result['person_labels'][:,5:28]==3).all())

    def test_outfit_core_guard_does_not_prevent_object_correction(self):
        labels=np.full((32,40),6,np.uint8);labels[:,:4]=3
        data=sample(labels);data[0]['see_logits'][0,:,2:28]=5
        data[0]['see_logits'][18,20,20]=5
        result=self.run_repair(data)
        self.assertTrue(result['objects'][20,20])
        self.assertEqual(result['person_labels'][20,20],0)

    def test_original_view_objects_can_remove_wrong_outfit_but_not_people(self):
        labels=np.array([[1,2,3,4,5,6,0]],np.uint8);data=sample(labels)
        data[0]['see_original_logits']=np.full((19,1,7),-4,np.float32)
        data[0]['see_original_logits'][18]=5
        result=self.run_repair(data)
        self.assertFalse(result['objects'][0,:5].any())
        self.assertTrue(result['objects'][0,5:].all())
        self.assertTrue((result['person_labels'][0,5:]==0).all())
        self.assertFalse(result['_object_seed_support'][0,:5].any())
        self.assertTrue(result['_object_support'].all())

    def test_object_and_independent_wear_overlap_is_not_automatically_removed(self):
        data=sample([[6,6,6]])
        data[0]['see_logits'][18]=5;data[0]['see_logits'][11,0,1]=5
        result=self.run_repair(data)
        self.assertTrue(result['objects'][0,0] and result['objects'][0,2])
        self.assertFalse(result['objects'][0,1]);self.assertEqual(result['person_labels'][0,1],6)
        self.assertFalse(result['_object_seed_support'][0,1])

    def test_wear_object_ambiguity_is_not_resolved_by_weak_primary_background(self):
        data=sample([[0]],strong=[[False]])
        data[0]['see_logits'][18]=4;data[0]['see_logits'][11]=4
        data[1]['objects'][:]=True;data[1]['person_labels'][:]=0
        result=self.run_repair(data)
        self.assertFalse(result['objects'][0,0])
        self.assertEqual(result['person_labels'][0,0],255)
        self.assertFalse(result['_object_seed_support'][0,0])
        self.assertEqual(result['source_map'][0,0],6)

    def test_retracting_object_over_strong_background_wear_conflict_stays_unknown(self):
        data=sample([[0]])
        data[0]['see_logits'][18]=4;data[0]['see_logits'][11]=4
        data[1]['objects'][:]=True;data[1]['person_labels'][:]=0
        data[1]['source_map'][:]=5;data[1]['consensus'][:]=True
        result=self.run_repair(data)
        self.assertFalse(result['objects'][0,0]);self.assertEqual(result['person_labels'][0,0],255)
        self.assertTrue(result['uncertain'][0,0]);self.assertEqual(result['source_map'][0,0],6)
        self.assertFalse(result['consensus'][0,0])

    def test_object_cannot_delete_person_supported_by_conflicting_m2f_view(self):
        for primary in (0,6):
            with self.subTest(primary=primary):
                data=sample([[primary]])
                data[0]['see_logits'][18]=4;data[1]['_m2f_class_support'][3]=True
                data[1]['_view_conflict'][:]=True;data[1]['person_labels'][:]=255
                data[1]['uncertain'][:]=True
                result=self.run_repair(data)
                self.assertFalse(result['objects'][0,0])
                self.assertEqual(result['person_labels'][0,0],255)
                self.assertFalse(result['_object_seed_support'][0,0])
                self.assertTrue(result['_object_protect'][0,0])

    def test_complete_object_guard_contains_body_and_wear_ambiguity(self):
        data=sample([[1,6,6]])
        data[0]['see_logits'][18]=4;data[0]['see_logits'][11,0,2]=4
        result=self.run_repair(data)
        self.assertIn('_object_protect',result)
        np.testing.assert_array_equal(result['_object_protect'],[[True,False,True]])

    def test_original_object_on_primary_background_records_semantic_state_change(self):
        data=sample([[0]])
        data[1]['source_map'][:]=3;data[1]['consensus'][:]=True
        data[0]['see_original_logits']=np.full((19,1,1),-4,np.float32)
        data[0]['see_original_logits'][18]=5
        result=self.run_repair(data)
        self.assertTrue(result['objects'][0,0]);self.assertEqual(result['person_labels'][0,0],0)
        self.assertEqual(result['source_map'][0,0],23)
        self.assertFalse(result['consensus'][0,0])
        self.assertEqual(result['semantic_repairs']['parts']['objects']['adopted_pixels'],1)

    def test_object_mask_preserves_holes_and_multiple_components(self):
        data=sample(np.full((12,18),6,np.uint8))
        obj=np.zeros((12,18),bool);obj[1:6,1:6]=True;obj[2:5,2:5]=False;obj[8:10,14:16]=True
        data[0]['see_logits'][18,obj]=5
        result=self.run_repair(data)
        np.testing.assert_array_equal(result['objects'],obj)

    def test_original_and_white_same_masks_are_not_two_votes(self):
        labels=np.full((12,16),6,np.uint8);labels[1:4,3:5]=3
        data=sample(labels);data[0]['see_logits'][0,1:11,3:5]=4
        first=self.run_repair(data)
        data[0]['see_original_logits']=data[0]['see_logits'].copy()
        second=self.run_repair(data)
        np.testing.assert_array_equal(first['person_labels'],second['person_labels'])

    def test_inputs_are_not_mutated(self):
        data=sample([[1,2,3,4,5,6]])
        data[0]['see_logits'][18]=4
        before={k:v.copy() for k,v in data[1].items() if isinstance(v,np.ndarray)}
        raw=data[0]['see_logits'].copy();result=self.run_repair(data)
        for key,value in before.items():np.testing.assert_array_equal(data[1][key],value)
        np.testing.assert_array_equal(data[0]['see_logits'],raw)
        self.assertIsNot(result['person_labels'],data[1]['person_labels'])

    def test_disabled_part_does_not_apply_its_see_candidate(self):
        data=sample([[6,6]])
        data[0]['see_logits'][18]=4
        policy=dict(self.policy,see_repair_parts=['face','mouth','hair'])
        result=self.run_repair(data,policy=policy)
        self.assertFalse(result['objects'].any())

    def test_bad_score_precision_and_shape_are_rejected(self):
        for key,value in [('see_logits',np.zeros((19,1,1),np.float16)),
                          ('see_original_logits',np.zeros((19,2,1),np.float32))]:
            data=sample([[2]]);data[0][key]=value
            with self.assertRaises(ValueError):self.run_repair(data)


class OutfitReferenceTests(unittest.TestCase):
    def setUp(self):
        self.module=importlib.import_module('gpt_animeimage_denoise.semantic_repair')
        self.policy={'outfit_reference':'m2f-white','source_codes':{'outfit_reference':25,'unknown':6}}

    def enforce(self, regions, policy=None):
        self.assertTrue(hasattr(self.module,'enforce_outfit_reference'),
            'White M2F outfit reference enforcement is missing')
        return self.module.enforce_outfit_reference(regions,self.policy if policy is None else policy)

    def test_white_weak_outfit_survives_original_strong_background_without_confidence_gate(self):
        data=sample([[0]],strong=[[False]]);regions=data[1]
        regions['_outfit_reference_mask']=np.ones((1,1),bool)
        result=self.enforce(regions)
        self.assertEqual(result['person_labels'][0,0],6)
        self.assertEqual(result['source_map'][0,0],25)
        self.assertFalse(result['uncertain'][0,0]);self.assertTrue(result['conflict'][0,0])
        self.assertEqual(result['outfit_reference']['restored_from_background'],1)

    def test_white_outfit_strong_view_conflict_restores_unknown_and_records_disagreement(self):
        regions=sample([[0]])[1];regions['person_labels'][:]=255;regions['uncertain'][:]=True
        regions['conflict'][:]=True;regions['_view_conflict'][:]=True
        regions['_outfit_reference_mask']=np.ones((1,1),bool)
        result=self.enforce(regions)
        self.assertEqual(result['person_labels'][0,0],6)
        self.assertTrue(result['conflict'][0,0]);self.assertFalse(result['uncertain'][0,0])
        self.assertTrue(result['_class_support'][6,0,0]);self.assertTrue(result['_trusted_cores'][6,0,0])
        self.assertEqual(int(result['_class_support'][:,0,0].sum()),1)
        self.assertEqual(result['outfit_reference']['restored_from_unknown'],1)

    def test_confirmed_objects_and_decided_person_modules_take_precedence(self):
        labels=np.array([[0,1,2,3,4,5]],np.uint8);regions=sample(labels)[1]
        regions['objects'][0,0]=True;regions['_outfit_reference_mask']=np.ones(labels.shape,bool)
        result=self.enforce(regions)
        np.testing.assert_array_equal(result['person_labels'],labels)
        self.assertTrue(result['objects'][0,0])
        self.assertEqual(result['outfit_reference']['restored_from_background'],0)

    def test_outside_white_reference_automatic_outfit_becomes_unknown(self):
        regions=sample([[6,6]])[1];regions['_outfit_reference_mask']=np.array([[True,False]])
        regions['consensus'][:]=True
        result=self.enforce(regions)
        np.testing.assert_array_equal(result['person_labels'],[[6,255]])
        self.assertTrue(result['uncertain'][0,1]);self.assertTrue(result['conflict'][0,1])
        self.assertEqual(result['source_map'][0,1],6)
        self.assertFalse(result['consensus'][0,1])
        self.assertFalse(result['_class_support'][6,0,1]);self.assertFalse(result['_trusted_cores'][6,0,1])
        self.assertEqual(result['outfit_reference']['removed_expansion'],1)

    def test_outfit_support_cannot_seed_a_see_or_sam_expansion_outside_reference(self):
        regions=sample([[3,0]])[1];regions['_class_support'][6]=True
        regions['_outfit_reference_mask']=np.array([[True,False]])
        result=self.enforce(regions)
        self.assertFalse(result['_class_support'][6,0,1])
        np.testing.assert_array_equal(result['person_labels'],[[3,0]])

    def test_counts_survive_second_enforcement_and_new_later_repairs_accumulate(self):
        regions=sample([[0,6]])[1];regions['_outfit_reference_mask']=np.ones((1,2),bool)
        first=self.enforce(regions)
        second=self.enforce(first)
        self.assertEqual(first['outfit_reference'],second['outfit_reference'])
        later=dict(first,person_labels=first['person_labels'].copy(),uncertain=first['uncertain'].copy())
        later['person_labels'][0,1]=255;later['uncertain'][0,1]=True
        third=self.enforce(later)
        self.assertEqual(third['outfit_reference']['restored_from_background'],1)
        self.assertEqual(third['outfit_reference']['restored_from_unknown'],1)
        self.assertEqual(first['outfit_reference']['restored_from_unknown'],0)

    def test_old_policy_has_exact_pass_through_without_requiring_a_reference_mask(self):
        regions=sample([[0,6]])[1]
        for policy in ({},{'outfit_reference':'off'}):
            result=self.enforce(regions,policy)
            self.assertNotIn('outfit_reference',result)
            for key,value in regions.items():
                if isinstance(value,np.ndarray):np.testing.assert_array_equal(result[key],value)

    def test_enforcement_does_not_mutate_inputs_or_other_semantic_groups(self):
        regions=sample([[0,6,3]])[1];regions['_outfit_reference_mask']=np.array([[True,False,True]])
        snapshots={key:value.copy() for key,value in regions.items() if isinstance(value,np.ndarray)}
        result=self.enforce(regions)
        for key,value in snapshots.items():np.testing.assert_array_equal(regions[key],value)
        self.assertIsNot(result['person_labels'],regions['person_labels'])
        self.assertEqual(result['person_labels'][0,2],3)

    def test_mode_requires_aligned_bool_white_reference(self):
        for reference in (None,np.ones((1,1),np.uint8),np.ones((2,1),bool)):
            regions=sample([[0]])[1]
            if reference is not None:regions['_outfit_reference_mask']=reference
            with self.assertRaises(ValueError):self.enforce(regions)

    def test_see_hair_cannot_replace_reference_outfit_without_an_m2f_hair_view(self):
        labels=np.zeros((12,18),np.uint8);labels[:,:4]=3
        data=sample(labels);data[0]['see_logits'][0,:,2:8]=4
        data[1]['_outfit_reference_mask']=np.zeros(labels.shape,bool)
        data[1]['_outfit_reference_mask'][:,4:8]=True
        policy={'see':1.,'see_repair_parts':['hair'],'outfit_reference':'m2f-white'}
        repaired=self.module.apply_see_repairs(data[0],data[1],primary_labels=data[2],
            primary_strong=data[3],primary_raw=data[4],policy=policy)
        self.assertTrue((repaired['person_labels'][:,4:8]==0).all())
        final=self.enforce(repaired)
        self.assertTrue((final['person_labels'][:,4:8]==6).all())

    def test_see_hair_can_use_another_m2f_view_to_correct_reference_outfit(self):
        labels=np.zeros((12,18),np.uint8);labels[:,:4]=3
        data=sample(labels);data[0]['see_logits'][0,:,2:8]=4
        data[1]['_outfit_reference_mask']=np.zeros(labels.shape,bool)
        data[1]['_outfit_reference_mask'][:,4:8]=True
        data[1]['_m2f_class_support'][3,:,4:8]=True
        policy={'see':1.,'see_repair_parts':['hair'],'outfit_reference':'m2f-white'}
        repaired=self.module.apply_see_repairs(data[0],data[1],primary_labels=data[2],
            primary_strong=data[3],primary_raw=data[4],policy=policy)
        self.assertTrue((repaired['person_labels'][:,4:8]==3).all())
        final=self.enforce(repaired)
        self.assertTrue((final['person_labels'][:,4:8]==3).all())

    def test_hair_to_outfit_reverse_cannot_expand_beyond_white_reference(self):
        labels=np.full((16,20),6,np.uint8);labels[2:12,5:9]=3
        data=sample(labels);ev,r,_,_,_=data
        r['_outfit_reference_mask']=labels==6
        r['_view_conflict'][6:9,8]=True;r['_m2f_class_support'][6,6:9,8]=True
        ev['see_logits'][11,6:9,8]=4
        policy={'see':1.,'see_repair_parts':['hair'],'outfit_reference':'m2f-white'}
        result=self.module.apply_see_repairs(ev,r,primary_labels=data[2],
            primary_strong=data[3],primary_raw=data[4],policy=policy)
        self.assertTrue((result['person_labels'][6:9,8]==3).all())

    def test_missing_outfit_mode_keeps_the_previous_hair_decision(self):
        labels=np.zeros((12,18),np.uint8);labels[:,:4]=3
        data=sample(labels);data[0]['see_logits'][0,:,2:8]=4
        data[1]['_outfit_reference_mask']=np.zeros(labels.shape,bool)
        data[1]['_outfit_reference_mask'][:,4:8]=True
        policy={'see':1.,'see_repair_parts':['hair']}
        result=self.module.apply_see_repairs(data[0],data[1],primary_labels=data[2],
            primary_strong=data[3],primary_raw=data[4],policy=policy)
        self.assertTrue((result['person_labels'][:,4:8]==3).all())


class SelectiveReferenceTests(unittest.TestCase):
    def setUp(self):
        self.module=importlib.import_module('gpt_animeimage_denoise.semantic_repair')
        self.policy={'see':1.,'see_reference':'selective',
                     'see_repair_parts':['face','mouth','hair','objects']}

    def repair(self,data,policy=None):
        return self.module.apply_see_repairs(data[0],data[1],primary_labels=data[2],
            primary_strong=data[3],primary_raw=data[4],policy=self.policy if policy is None else policy)

    def hair_sample(self):
        labels=np.zeros((16,32),np.uint8);labels[:,:2]=3
        data=sample(labels,strong=labels==3)
        data[0]['see_logits'][0,:,:24]=3
        data[1]['_class_support'][3,:,:24]=True
        return data

    def test_stable_primary_outfit_is_preserved_without_auxiliary_wear(self):
        data=sample([[6]]);data[0]['see_logits'][18]=4;data[0]['see_logits'][11]=1.5
        result=self.repair(data)
        self.assertFalse(result['objects'][0,0])
        self.assertTrue(result['suspected_objects'][0,0])
        self.assertEqual(result['person_labels'][0,0],6)
        self.assertFalse(result['_wear_support'][0,0])

    def test_strict_auxiliary_wear_still_needs_stable_m2f_outfit(self):
        data=sample([[6]],strong=[[False]])
        data[0]['see_logits'][18]=4;data[0]['see_logits'][11]=3
        result=self.repair(data)
        self.assertTrue(result['objects'][0,0])
        self.assertFalse(result['_object_protect'][0,0])

    def test_strong_stable_outfit_and_strict_wear_preserve_object_ambiguity(self):
        data=sample([[6]]);data[0]['see_logits'][18]=4;data[0]['see_logits'][11]=3
        result=self.repair(data)
        self.assertFalse(result['objects'][0,0]);self.assertEqual(result['person_labels'][0,0],6)
        self.assertTrue(result['_wear_support'][0,0]);self.assertTrue(result['_object_protect'][0,0])

    def test_wear_confirmation_respects_the_white_outfit_reference_when_present(self):
        data=sample([[6]]);data[0]['see_logits'][18]=4;data[0]['see_logits'][11]=3
        data[1]['_outfit_reference_mask']=np.zeros((1,1),bool)
        policy=dict(self.policy,outfit_reference='m2f-white')
        result=self.repair(data,policy)
        self.assertTrue(result['objects'][0,0]);self.assertFalse(result['_wear_support'][0,0])
        legacy=self.repair(data,dict(policy,see_reference='legacy'))
        self.assertFalse(legacy['objects'][0,0])

    def test_hair_at_lower_threshold_does_not_expand_or_erase_primary_hair(self):
        data=self.hair_sample();data[0]['see_logits'][0,:,:24]=1.5
        result=self.repair(data)
        self.assertTrue((result['person_labels'][:,:2]==3).all())
        self.assertTrue((result['person_labels'][:,2:24]==0).all())
        self.assertFalse(result['_class_support'][3,:,2:24].any())

    def test_one_see_view_can_repair_near_edge_but_not_a_far_hair_guess_or_sam_support(self):
        data=self.hair_sample();result=self.repair(data)
        self.assertEqual(result['person_labels'][8,2],3)
        self.assertEqual(result['person_labels'][8,20],0)
        self.assertFalse(result['_class_support'][3,8,20])

    def test_two_see_input_stability_allows_anchored_far_hair_without_a_second_vote(self):
        data=self.hair_sample();data[0]['see_original_logits']=data[0]['see_logits'].copy()
        result=self.repair(data)
        self.assertEqual(result['person_labels'][8,20],3)
        self.assertTrue(result['_class_support'][3,8,20])
        self.assertFalse(result['semantic_repairs']['independent_view_votes'])

    def test_another_m2f_hair_view_can_support_a_far_extension(self):
        data=self.hair_sample();data[1]['_m2f_class_support'][3,8,20]=True
        data[1]['_view_conflict'][8,20]=True
        result=self.repair(data)
        self.assertEqual(result['person_labels'][8,20],3)

    def test_tiny_anchor_cannot_approve_a_huge_see_hair_component(self):
        labels=np.zeros((32,32),np.uint8);labels[16,16]=3
        data=sample(labels,strong=labels==3);data[0]['see_logits'][0,6:26,6:26]=3
        data[0]['see_original_logits']=data[0]['see_logits'].copy()
        result=self.repair(data)
        self.assertEqual(result['person_labels'][16,16],3)
        self.assertEqual(result['person_labels'][16,17],0)
        self.assertEqual(result['semantic_repairs']['parts']['hair']['unanchored_components'],1)

    def test_weak_auxiliary_hair_and_wear_do_not_veto_an_anchored_face_gap(self):
        data=sample([[2,0]],strong=[[True,False]])
        data[1]['person_labels'][0,1]=255;data[1]['uncertain'][0,1]=True
        data[0]['see_logits'][2]=3;data[0]['see_logits'][0,0,1]=3;data[0]['see_logits'][11,0,1]=3
        result=self.repair(data)
        np.testing.assert_array_equal(result['person_labels'],[[2,2]])

    def test_hair_cannot_overwrite_face_adopted_earlier_in_the_same_pass(self):
        labels=np.zeros((12,14),np.uint8);labels[4:9,:2]=3;labels[4:9,2:5]=2
        data=sample(labels,strong=labels!=0);data[1]['person_labels'][4:9,5]=255
        data[1]['uncertain'][4:9,5]=True
        data[0]['see_logits'][2,4:9,2:6]=3;data[0]['see_logits'][0,4:9,:6]=3
        data[0]['see_original_logits']=data[0]['see_logits'].copy()
        result=self.repair(data)
        self.assertTrue((result['person_labels'][4:9,5]==2).all())
        self.assertTrue(result['_trusted_cores'][2,4:9,5].all())

    def test_rejected_unanchored_face_and_mouth_cannot_reenter_through_sam_support(self):
        data=sample(np.zeros((12,14),np.uint8),strong=np.zeros((12,14),bool))
        data[0]['see_logits'][2,3:7,3:7]=3;data[0]['see_logits'][8,8:10,8:10]=3
        data[1]['_class_support'][2,3:7,3:7]=True;data[1]['_class_support'][5,8:10,8:10]=True
        result=self.repair(data)
        self.assertFalse(result['_class_support'][2].any())
        self.assertFalse(result['_class_support'][5].any())

    def test_accepted_eye_cannot_regain_a_competing_pure_m2f_face_support(self):
        data=sample([[2]]);ev,r,primary,_,_=data
        primary[:]=4;r['person_labels'][:]=4
        r['_class_support'][2]=False;r['_class_support'][4]=True
        r['_trusted_cores'][2]=False;r['_trusted_cores'][4]=True
        original_pure=r['_m2f_class_support'].copy()
        result=self.repair(data)
        self.assertEqual(result['person_labels'][0,0],4)
        self.assertFalse(result['_class_support'][2,0,0])
        self.assertTrue(result['_class_support'][4,0,0])
        np.testing.assert_array_equal(result['_m2f_class_support'],original_pure)

    def test_unresolved_pixels_can_retain_pure_m2f_alternative_support(self):
        data=sample([[0]],strong=[[False]])
        data[1]['person_labels'][:]=255;data[1]['uncertain'][:]=True
        data[1]['_m2f_class_support'][2]=True;data[1]['_m2f_class_support'][3]=True
        result=self.repair(data)
        self.assertEqual(result['person_labels'][0,0],255)
        self.assertTrue(result['_class_support'][2,0,0]);self.assertTrue(result['_class_support'][3,0,0])

    def test_disabling_hair_repairs_also_disables_raw_see_hair_sam_seeds(self):
        data=self.hair_sample()
        policy=dict(self.policy,see_repair_parts=['face','mouth','objects'])
        result=self.repair(data,policy)
        np.testing.assert_array_equal(result['_class_support'][3],data[1]['_m2f_class_support'][3])

    def test_selective_roles_and_effective_thresholds_are_explicit(self):
        result=self.repair(self.hair_sample())
        report=result['semantic_repairs']
        self.assertEqual(report['effective_thresholds']['hair'],2.)
        self.assertEqual(report['effective_thresholds']['wear'],2.)
        self.assertEqual(report['effective_thresholds']['face'],1.)
        self.assertIn('hair',report['evidence_roles'])
        self.assertIn('reference_rejections',report)

    def test_missing_or_legacy_reference_mode_preserves_old_decisions(self):
        data=self.hair_sample();base={'see':1.,'see_repair_parts':['hair']}
        first=self.repair(data,base);second=self.repair(data,dict(base,see_reference='legacy'))
        np.testing.assert_array_equal(first['person_labels'],second['person_labels'])
        self.assertEqual(first['person_labels'][8,20],3)
        self.assertNotIn('evidence_roles',first['semantic_repairs'])


if __name__=='__main__':unittest.main()
