"""Processing labels: Mask2Former v4 regions plus See-through eye contours.

This is post-argmax grouping, not retraining the model or finding image noise.
Brows belong to hair, nose to face, mouth is independent, and garments/accessories
share the outfit group. Background is not counted among the six person modules.
See-through objects can veto processing without rewriting those group labels.
"""
import numpy as np
from scipy.ndimage import label,find_objects,maximum_filter

NAMES=('background','skin','face','hair','eyes','mouth','outfit')
PERSON_GROUPS=NAMES[1:]
DEFAULT_PROCESS_GROUPS=('skin','face','hair')
DEFAULT_MODEL_VERSION=4
DEFAULT_MODEL_INPUT_SIZE=768
DEFAULT_MODEL_PRECISION='FP32'
DEFAULT_OFFLINE_BUNDLE=False
COLORS=np.array([[0,0,0],[255,220,180],[100,150,255],[255,0,0],
    [0,255,255],[255,0,150],[180,0,255]],np.uint8)
V4_TO_GROUP=np.array([0,1,2,3,4,4,3,3,2,5,6,6],np.uint8)

def coarsen_v4(labels):
    labels=np.asarray(labels)
    if labels.ndim!=2 or not labels.size or labels.dtype.kind not in 'iu' or labels.min()<0 or labels.max()>11:
        raise ValueError('Expected nonempty 2D integer Mask2Former labels 0..11')
    return V4_TO_GROUP[labels]

def fuse_see_eyes(labels,see_eyes):
    """Use whole See eye components near v4 eyes; never crop a kept component.

    V4 eye labels are only position anchors. A missing See eye falls back to
    its v4 component and is reported. Explicit v4 eyebrows belong to hair, nose
    to face, and mouth remains separate. Isolated See eyebrow fragments and clothing/accessory/background
    predictions are not allowed to become eyes. No head box clips hair or skin.
    """
    base=coarsen_v4(labels);labels=np.asarray(labels);see_eyes=np.asarray(see_eyes)
    if see_eyes.dtype!=bool or see_eyes.shape!=labels.shape:raise ValueError('See eye mask must be aligned bool HxW')
    old_eyes=np.isin(labels,[4,5]);brows=np.isin(labels,[6,7]);nose=labels==8;mouth=labels==9
    reassigned_features=brows|nose|mouth
    eligible=see_eyes&np.isin(labels,[1,2,3,4,5])
    radius=max(1,round(min(labels.shape)/768))
    face_fallback=not old_eyes.any()
    anchor=maximum_filter(old_eyes,size=2*radius+1) if not face_fallback else labels==2
    components,n=label(eligible,structure=np.ones((3,3),bool));retained=np.zeros_like(eligible)
    kept_components=0
    for component,box in enumerate(find_objects(components),1):
        if box is None:continue
        part=components[box]==component
        if np.any(part&anchor[box]):retained[box]|=part;kept_components+=1
    fallback=np.zeros_like(eligible)
    old_components,_=label(old_eyes,structure=np.ones((3,3),bool))
    supported=maximum_filter(retained,size=2*radius+1)
    for component,box in enumerate(find_objects(old_components),1):
        if box is None:continue
        part=old_components[box]==component
        if not np.any(part&supported[box]):fallback[box]|=part
    eyes=(retained|fallback)&~reassigned_features
    merged=base.copy();merged[old_eyes]=2;merged[eyes]=4
    merged[brows]=3;merged[nose]=2;merged[mouth]=5
    assert np.all(merged[brows]==3) and np.all(merged[nose]==2) and np.all(merged[mouth]==5)
    assert np.array_equal(merged[~(old_eyes|eyes)],base[~(old_eyes|eyes)])
    report={'v4_eye_pixels':int(old_eyes.sum()),'see_raw_eye_pixels':int(see_eyes.sum()),
        'see_eye_pixels_used':int(retained.sum()),'fallback_v4_eye_pixels':int(fallback.sum()),
        'final_eye_pixels':int(eyes.sum()),'see_pixels_excluded':int((see_eyes&~retained).sum()),
        'see_eye_over_reassigned_features_pixels':int((see_eyes&reassigned_features).sum()),
        'see_components_kept':kept_components,'used_face_anchor_fallback':face_fallback,
        'anchor_margin_pixels':radius,'all_v4_brows_in_hair':True,'all_v4_nose_in_face':True,'mouth_independent':True,
        'unchanged_outside_eye_reassignment':True}
    return merged,report

def default_processing_mask(groups):
    groups=np.asarray(groups)
    if groups.ndim!=2 or not groups.size or groups.dtype.kind not in 'iu' or groups.min()<0 or groups.max()>=len(NAMES):
        raise ValueError('Expected nonempty integer group map 0..6')
    return np.isin(groups,[NAMES.index(name) for name in DEFAULT_PROCESS_GROUPS])

def processing_regions(labels,see_eyes,see_objects=None):
    """Build regions using an optional final object-exclusion mask.

    See raw19 class18 (logit>0) is the default source; the caller may also
    supply a reviewed union with SAM2 object masks. This function does not run
    SAM2 or dilate masks. Keep labels for diagnosis and export person modules
    from person_labels, where excluded pixels are zero (outside the person).
    Objects are also a hard processing veto. None preserves the previous
    two-argument behavior.
    """
    grouped,report=fuse_see_eyes(labels,see_eyes)
    allowed=default_processing_mask(grouped)
    enabled=see_objects is not None
    objects=np.zeros_like(grouped,bool) if not enabled else np.asarray(see_objects)
    if objects.dtype!=bool or objects.shape!=grouped.shape:
        raise ValueError('See object mask must be aligned bool HxW')
    objects=objects.copy()
    removed=allowed&objects
    object_report={'enabled':enabled,'object_pixels':int(objects.sum()),
        'removed_character_pixels':int(((grouped!=0)&objects).sum()),
        'removed_processing_pixels':int(removed.sum()),
        'object_pixels_by_group':{name:int(((grouped==i)&objects).sum()) for i,name in enumerate(NAMES)},
        'removed_processing_pixels_by_group':{name:int(((grouped==i)&removed).sum()) for i,name in enumerate(NAMES)}}
    allowed&=~objects
    person_labels=grouped.copy();person_labels[objects]=0
    return {'labels':grouped,'person_labels':person_labels,'allowed':allowed,'protected':~allowed,'fusion':report,
        'object_exclusion':objects,'character_allowed':person_labels!=0,'object_report':object_report,
        'policy':{'model_version':DEFAULT_MODEL_VERSION,'person_groups':list(PERSON_GROUPS),
                  'default_input_resolution':[DEFAULT_MODEL_INPUT_SIZE,DEFAULT_MODEL_INPUT_SIZE],
                  'default_precision':DEFAULT_MODEL_PRECISION,'objects_exclusion':enabled,
                  'default_process':list(DEFAULT_PROCESS_GROUPS),'offline_bundle':DEFAULT_OFFLINE_BUNDLE}}
