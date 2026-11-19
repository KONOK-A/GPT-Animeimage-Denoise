"""Limited, class-aware See-through repairs on the original image grid.

The two See inputs are alternative views of one checkpoint, not independent
votes. Thresholded masks suggest visible labels; neither a raw score nor a SAM
shape establishes semantic correctness. Public region masks are refreshed by
the caller after this function returns.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label


PARTS=('face','mouth','hair','neck','objects')
WEAR_CHANNELS=(1,4,6,10,11,13,15)
SOURCE_CODES={'see_face':20,'see_mouth':21,'see_hair':22,
              'see_object_repair':23,'primary_fallback':24,'see_neck':26}


def _radius(shape, pixels):
    return max(1.,float(pixels)*min(shape)/2048.)


def _near(anchor, radius):
    nearby = np.zeros_like(anchor)
    rows = np.flatnonzero(anchor.any(axis=1))
    if not rows.size:
        return nearby
    columns = np.flatnonzero(anchor.any(axis=0))
    # The rectangle contains every anchor and every pixel within the radius
    # of an anchor. Integer offsets beyond the padding cannot be in range.
    padding = int(radius)
    window = (slice(max(0, rows[0] - padding), min(anchor.shape[0], rows[-1] + padding + 1)),
              slice(max(0, columns[0] - padding), min(anchor.shape[1], columns[-1] + padding + 1)))
    nearby[window] = distance_transform_edt(~anchor[window]) <= radius
    return nearby


def _outfit_reference_mask(regions, shape, policy):
    if policy.get('outfit_reference')!='m2f-white':return None
    value=regions.get('_outfit_reference_mask')
    if not isinstance(value,np.ndarray) or value.dtype!=bool or value.shape!=shape:
        raise ValueError('White M2F outfit reference must be aligned bool HxW')
    return value


def _components(mask):
    components,_=label(mask,structure=np.ones((3,3),bool))
    for number,box in enumerate(find_objects(components),1):
        if box is not None:yield box,components[box]==number


def _anchored(mask, anchor, radius, minimum_fraction=0.):
    """Keep every sufficiently anchored component, without closing gaps."""
    near=_near(anchor,radius)
    selected=np.zeros_like(mask)
    counts={'components':0,'accepted_components':0,'unanchored_components':0}
    for box,component in _components(mask):
        counts['components']+=1
        overlap=int((component&near[box]).sum())
        if not overlap or overlap/max(1,int(component.sum()))<minimum_fraction:
            counts['unanchored_components']+=1;continue
        selected[box]|=component;counts['accepted_components']+=1
    return selected,counts


def _matched_feature(fine, anchor, evidence, face, radius, *, narrow_evidence=None):
    """Match each feature to a local contour before replacing a primary rim."""
    evidence = evidence & _near(anchor, radius)
    anchors = np.zeros(fine.shape, np.int32)
    count = 0
    for value in np.unique(fine[anchor]):
        ids, found = label(anchor & (fine == value), structure=np.ones((3, 3), bool))
        hits = ids > 0
        anchors[hits] = ids[hits] + count
        count += found
    sizes = np.bincount(anchors.ravel())
    anchor_boxes = find_objects(anchors)
    matches = {}
    for box, component in _components(evidence):
        overlap = anchors[box][component]
        owners = np.unique(overlap[overlap > 0])
        if len(owners) != 1:
            continue
        owner = int(owners[0])
        if int(component.sum()) <= int(sizes[owner]) * 8:
            matches.setdefault(owner, []).append((box, component, int((overlap == owner).sum())))
    accepted = np.zeros_like(anchor)
    trim = np.zeros_like(anchor)
    accepted_count = 0
    for owner, components in matches.items():
        box, component, overlap = max(components, key=lambda item: item[2])
        anchor_box = anchor_boxes[owner - 1]
        if overlap * 2 < sizes[owner]:
            # A closed mouth can be a thin line inside a much thicker primary
            # proposal. Require agreement along its length, not just area.
            if (narrow_evidence is None or len(components) != 1 or overlap * 4 < sizes[owner]
                    or overlap * 4 < 3 * component.sum()
                    or (component & narrow_evidence[box]).sum() * 2 < component.sum()):
                continue
            axis = max(range(2), key=lambda i: anchor_box[i].stop - anchor_box[i].start)
            along = np.nonzero(component & (anchors[box] == owner))[axis]
            extent = int(along.max() - along.min()) + 1
            if extent * 4 < 3 * (anchor_box[axis].stop - anchor_box[axis].start):
                continue
        accepted[box] |= component
        accepted_count += 1
        if len(components) == 1:
            trim[anchor_box] |= ((anchors[anchor_box] == owner) & ~evidence[anchor_box]
                                 & face[anchor_box])
    return accepted, trim, accepted_count, count


def _validate(evidence, regions, primary_labels, primary_strong, primary_raw, policy):
    primary=np.asarray(primary_labels);strong=np.asarray(primary_strong);raw=np.asarray(primary_raw)
    if (primary.ndim!=2 or not primary.size or primary.dtype.kind not in 'iu'
            or np.any(primary>6) or np.any(primary<0)):
        raise ValueError('Primary labels must be nonempty original-grid integer groups 0..6')
    shape=primary.shape
    if strong.dtype!=bool or strong.shape!=shape:
        raise ValueError('Primary strong mask must be aligned bool')
    if raw.dtype.kind not in 'iu' or raw.shape!=shape or np.any(raw>11) or np.any(raw<0):
        raise ValueError('Primary fine labels must be aligned integers 0..11')
    views=[]
    for name in ('see_logits','see_original_logits'):
        value=evidence.get(name)
        if value is None and name=='see_original_logits':continue
        if (not isinstance(value,np.ndarray) or value.dtype!=np.float32
                or value.shape!=(19,*shape) or not np.isfinite(value).all()):
            raise ValueError('See evidence must retain finite float32 raw19 original-grid scores: '+name)
        views.append((name,value))
    for key in ('person_labels','objects','uncertain','conflict','consensus','source_map','_view_conflict'):
        if not isinstance(regions.get(key),np.ndarray) or regions[key].shape!=shape:
            raise ValueError('Missing or misaligned region array: '+key)
    for key in ('objects','uncertain','conflict','consensus','_view_conflict'):
        if regions[key].dtype!=bool:raise ValueError('Region mask must be bool: '+key)
    for key in ('_class_support','_trusted_cores'):
        if (not isinstance(regions.get(key),np.ndarray) or regions[key].shape!=(7,*shape)
                or regions[key].dtype!=bool):
            raise ValueError('Missing or misaligned region support: '+key)
    parts=policy.get('see_repair_parts',PARTS)
    if not isinstance(parts,(list,tuple)) or any(part not in PARTS for part in parts):
        raise ValueError('see_repair_parts must select face, mouth, hair, neck and/or objects')
    threshold=policy.get('see',1.)
    if not isinstance(threshold,(float,int)) or not np.isfinite(threshold):
        raise ValueError('See threshold must be finite')
    return primary,strong,raw,views,set(parts),float(threshold)


def apply_see_repairs(evidence, regions, *, primary_labels, primary_strong,
                      primary_raw, policy, m2f_views=None):
    """Return repaired region arrays while preserving evidence and input arrays.

    Only pure M2F support can anchor a component. ``_class_support`` also
    contains See evidence, so using it as an anchor would let See certify itself.
    Missing ``_m2f_class_support`` falls back to the selected primary view.
    """
    primary,strong,raw,views,parts,threshold=_validate(
        evidence,regions,primary_labels,primary_strong,primary_raw,policy)
    shape=primary.shape
    result=dict(regions)
    for key in ('person_labels','objects','uncertain','conflict','consensus','source_map',
                '_view_conflict','_class_support','_trusted_cores'):
        result[key]=regions[key].copy()
    pure=regions.get('_m2f_class_support')
    if pure is None:pure=np.stack([(primary==group)&strong for group in range(7)])
    if not isinstance(pure,np.ndarray) or pure.dtype!=bool or pure.shape!=(7,*shape):
        raise ValueError('Pure M2F support must be aligned bool 7xHxW')
    outfit_reference=_outfit_reference_mask(regions,shape,policy)
    codes={**SOURCE_CODES,**policy.get('source_codes',{})}
    person=result['person_labels'];uncertain=result['uncertain'];view_conflict=result['_view_conflict']
    selective=policy.get('see_reference')=='selective'
    guarded=policy.get('part_cooperation')=='guarded'
    white,original=m2f_views if m2f_views is not None else (primary,None)
    other=white if original is None else original
    hair_threshold=max(threshold,2.) if selective else threshold
    wear_threshold=max(threshold,2.) if selective else threshold
    approved_hair=np.zeros(shape,bool)

    def candidates(channels, minimum=threshold):
        masks=[];source={}
        for name,values in views:
            mask=np.zeros(shape,bool)
            for channel in channels:mask|=values[channel]>minimum
            masks.append(mask);source['original' if name=='see_original_logits' else 'white']=int(mask.sum())
        combined=masks[0].copy()
        for mask in masks[1:]:combined|=mask
        source['union']=int(combined.sum())
        return combined,source

    face,face_sources=candidates((2,7,5))
    mouth,mouth_sources=candidates((8,))
    raw_hair,raw_hair_sources=candidates((0,))
    hair,hair_sources=candidates((0,),hair_threshold) if selective else (raw_hair,raw_hair_sources)
    raw_wear,wear_sources=candidates(WEAR_CHANNELS)
    wear=raw_wear
    stable_outfit=np.zeros(shape,bool)
    if selective:
        strict_wear,strict_wear_sources=candidates(WEAR_CHANNELS,wear_threshold)
        stable_outfit=(primary==6)&strong&~regions['_view_conflict']&pure[6]
        if outfit_reference is not None:stable_outfit&=outfit_reference
        wear=strict_wear&stable_outfit
        result['_raw_wear_support']=raw_wear
    object_mask,object_sources=candidates((18,))
    eyes,_=candidates((3,))
    result['_wear_support']=wear.copy()
    report={'strategy':'anchored-visible-components','threshold':threshold,
        'views':['original' if name=='see_original_logits' else 'white' for name,_ in views],
        'independent_view_votes':False,'parts':{},
        'note':'Candidate and changed-pixel counts are diagnostics, not accuracy.'}
    if selective:
        report['evidence_roles']={'face':'anchored repair','mouth':'anchored contour repair',
            'eyes':'existing anchored contours','objects':'guarded external-object repair',
            'hair':'strict anchored supplement','skin':'confirmation of primary skin only',
            'outfit':'strict wear confirmation of stable M2F outfit only'}
        report['effective_thresholds']={'face':threshold,'mouth':threshold,'objects':threshold,
            'hair':hair_threshold,'wear':wear_threshold,'hair_anchor_fraction':.05,
            'single_view_hair_growth_at_2048':16.}
        report['reference_rejections']={
            'hair_below_strict_threshold_pixels':int((raw_hair&~hair).sum()),
            'wear_unconfirmed_pixels':int((raw_wear&~wear).sum())}
        report['wear_reference']={'raw_source_pixels':wear_sources,
            'strict_source_pixels':strict_wear_sources,'confirmed_pixels':int(wear.sum())}

    def adopt(mask,group,source):
        mask=mask&~result['objects'] if group else mask
        changed=mask&((person!=group)|uncertain|view_conflict)
        if group==0:changed|=mask&~result['objects']
        result['conflict'][changed&(primary!=group)]=True
        person[mask]=group;uncertain[mask]=False;view_conflict[mask]=False
        result['source_map'][changed]=codes[source]
        result['consensus'][changed]=False
        result['_class_support'][:,mask]=False
        result['_trusted_cores'][:,mask]=False
        if group:
            result['_class_support'][group,mask]=True
            result['_trusted_cores'][group,mask]=True
        return int(changed.sum())

    # Keep a reliable primary proposal when only the auxiliary model disagrees.
    # Genuine disagreement between strong M2F views still needs class evidence.
    fallback=(person==255)&strong&~view_conflict&np.isin(primary,[1,2,3,4,5,6])&~result['objects']
    report['primary_fallback_pixels']=0
    for group in range(1,7):
        report['primary_fallback_pixels']+=adopt(fallback&(primary==group),group,'primary_fallback')

    primary_body=strong&np.isin(primary,[1,2,3,4,5])&~view_conflict
    explicit=np.isin(raw,[4,5,6,7,9])
    # Low-priority raw hair must not veto higher-priority face/mouth evidence.
    # Pure M2F hair and already trusted primary hair still retain protection.
    feature_hair=(pure[3]|(result['_trusted_cores'][3]&(person==3))) if selective else hair

    if guarded:
        report['strategy']='guarded-part-cooperation'
        report.setdefault('evidence_roles',{}).update(
            eyes='matched per-eye contour', mouth='matched mouth contour',
            face='local unknown supplement', hair='local unknown supplement',
            skin='dual-view neck supplement')
        for group,name,feature,blocked_fine in (
                (4,'eyes',eyes,[6,7,8,9]),(5,'mouth',mouth,[4,5,6,7,8])):
            if name=='mouth' and name not in parts:
                continue
            anchor=(person==group)&pure[group]
            blocked=(np.isin(raw,blocked_fine)|feature_hair|pure[6]|view_conflict|
                     object_mask|result['objects'])
            eligible=feature&((person==group)|(person==2)|uncertain)&~blocked
            narrow=None
            if group==5 and len(views)==2:
                narrow=(views[0][1][8]>threshold)&(views[1][1][8]>threshold)
            contour,trim,matched,anchor_count=_matched_feature(
                raw,anchor,eligible,face&~blocked,_radius(shape,8),narrow_evidence=narrow)
            adopted=adopt(contour,group,'see_eyes' if group==4 else 'see_mouth')
            trimmed=adopt(trim,2,'see_eyes' if group==4 else 'see_mouth')
            report['parts'][name]={'source_pixels':int(feature.sum()),
                'anchor_components':anchor_count,'matched_components':matched,
                'adopted_pixels':adopted,'trimmed_to_face_pixels':trimmed,
                'fallback_pixels':int((anchor&~contour&~trim).sum())}
            if group==4:
                result['fusion']={'v4_eye_pixels':int(anchor.sum()),
                    'see_raw_eye_pixels':int(feature.sum()),'see_eye_pixels_used':int(contour.sum()),
                    'fallback_v4_eye_pixels':report['parts'][name]['fallback_pixels'],
                    'final_eye_pixels':int((person==4).sum()),'trimmed_to_face_pixels':trimmed}

        if 'neck' in parts:
            neck_threshold=max(threshold,2.)
            neck=np.zeros(shape,bool)
            if len(views)==2 and original is not None:
                neck=(views[0][1][9]>neck_threshold)&(views[1][1][9]>neck_threshold)
            competing,_=candidates((0,1,6,10,11),neck_threshold)
            compatible=np.isin(white,[1,2])&np.isin(other,[1,2])
            blocked=(competing|feature_hair|pure[6]|object_mask|result['objects']|view_conflict)
            core=neck&compatible&~blocked
            anchored,counts=_anchored(core,(person==1)&pure[1],_radius(shape,5))
            fill=anchored&uncertain
            changed=adopt(fill,1,'see_neck')
            report['parts']['neck']={**counts,'strong_both_views_pixels':int(neck.sum()),
                'threshold':neck_threshold,'competing_or_incompatible_pixels':int((neck&~core).sum()),
                'adopted_pixels':changed}

    if 'face' in parts:
        anchor=pure[2]&~result['objects']
        if guarded:anchor&=person==2
        anchored,counts=_anchored(face,anchor,_radius(shape,8))
        blocked=(explicit|((primary!=2)&primary_body)|((primary==6)&strong)|
                 feature_hair|wear|object_mask|result['objects']|view_conflict)
        eligible=anchored&_near(anchor,_radius(shape,32))&~blocked
        if guarded:
            eligible&=uncertain&((white==2)|(other==2))
            if outfit_reference is not None:eligible&=~outfit_reference
        changed=adopt(eligible,2,'see_face')
        report['parts']['face']={**counts,'source_pixels':face_sources,
            'accepted_evidence_pixels':int(eligible.sum()),'adopted_pixels':changed,
            'blocked_or_distant_pixels':int((anchored&~eligible).sum())}

    if 'mouth' in parts and not guarded:
        # A face alone is too broad an anchor when a false mouth lies elsewhere
        # inside it. Without a M2F mouth position, retain the candidate for review.
        anchor=pure[5]&~result['objects']
        anchored,counts=_anchored(mouth,anchor,_radius(shape,8))
        mouth_components,mouth_count=label(anchor,structure=np.ones((3,3),bool))
        mouth_sizes=np.bincount(mouth_components.ravel())
        accepted=np.zeros(shape,bool);oversized=0
        near_anchor=_near(anchor,_radius(shape,8))
        for box,component in _components(anchored):
            nearby=mouth_components[box][component&near_anchor[box]]
            identities=np.unique(nearby[nearby>0])
            # Adjacency without direct overlap still needs the size of the
            # nearest known mouth, rather than the area of a whole face.
            if not identities.size and mouth_count:
                pad=int(np.ceil(_radius(shape,8)))
                window=tuple(slice(max(0,axis.start-pad),min(shape[index],axis.stop+pad))
                             for index,axis in enumerate(box))
                nearby=mouth_components[window]
                identities=np.unique(nearby[nearby>0])
            allowance=max(4,int(mouth_sizes[identities].sum())*8) if identities.size else 0
            if int(component.sum())>allowance:oversized+=1;continue
            accepted[box]|=component
        blocked=(np.isin(raw,[4,5,6,7,8])|((primary==3)&strong)|
                 ((primary==6)&strong)|eyes|feature_hair|wear|object_mask|result['objects'])
        eligible=accepted&~blocked
        changed=adopt(eligible,5,'see_mouth')
        # Contour replacement may remove an over-selected primary rim, but a
        # tiny See fragment must not erase a substantially larger known mouth.
        trim=np.zeros(shape,bool)
        for box,component in _components(anchor):
            coverage=int((component&eligible[box]).sum())/max(1,int(component.sum()))
            if coverage>=.25:trim[box]|=component&~mouth[box]&face[box]&~eyes[box]&~feature_hair[box]&~object_mask[box]
        trimmed=adopt(trim,2,'see_mouth')
        report['parts']['mouth']={**counts,'source_pixels':mouth_sources,
            'accepted_evidence_pixels':int(eligible.sum()),'adopted_pixels':changed,
            'trimmed_to_face_pixels':trimmed,'oversized_components':oversized,
            'missing_anchor':not bool(anchor.any()),
            'blocked_pixels':int((accepted&~eligible).sum())}

    if 'hair' in parts:
        anchor=pure[3]&~result['objects']
        if guarded:anchor&=person==3
        anchored,counts=_anchored(hair,anchor,_radius(shape,8),.05 if selective else .01)
        # Independent wear evidence can protect a scarf or hair accessory even
        # when the primary labels on that boundary are not trustworthy.
        # See can mistake a glove for connected hair while also missing its
        # limb channel. Limit corrections of confident outfit to its boundary,
        # unless another M2F view independently supports hair at that pixel.
        outfit_core=((primary==6)&strong&~pure[3]&
                     ~_near(primary!=6,_radius(shape,8)))
        reference_block=(outfit_reference&~pure[3] if outfit_reference is not None
                         else np.zeros(shape,bool))
        blocked=((np.isin(primary,[1,2,4,5])&strong)|eyes|mouth|wear|object_mask|
                 result['objects']|outfit_core|reference_block)
        if selective or guarded:
            # Face and mouth admitted earlier in this pass remain higher
            # priority even where the original primary label was weak.
            blocked|=result['_trusted_cores'][[1,2,4,5]].any(axis=0)
            stable_hair=np.zeros(shape,bool)
            if evidence.get('see_original_logits') is not None:
                stable_hair=((evidence['see_logits'][0]>hair_threshold)&
                             (evidence['see_original_logits'][0]>hair_threshold))
            growth=pure[3]|_near(anchor,_radius(shape,16))|stable_hair
        else:growth=np.ones(shape,bool)
        eligible=anchored&~blocked&growth
        if guarded:
            eligible&=uncertain&~view_conflict&((white==3)|(other==3))
            if outfit_reference is not None:eligible&=~outfit_reference
            connected,_=_anchored(eligible|anchor,anchor,_radius(shape,1))
            eligible&=connected
        approved_hair=eligible
        changed=adopt(eligible,3,'see_hair')
        # Reverse correction is restricted to a genuine M2F hair/outfit view
        # disagreement near an existing outfit boundary, corroborated by See.
        reverse_count=0
        if not guarded:
            outfit_anchor=(primary==6)&strong
            reverse=((primary==3)&regions['_view_conflict']&pure[6]&wear&~hair&
                     _near(outfit_anchor,_radius(shape,8))&~np.isin(raw,[6,7])&
                     ~object_mask&~result['objects'])
            if outfit_reference is not None:reverse&=outfit_reference
            reverse_count=adopt(reverse,6,'see_hair')
        report['parts']['hair']={**counts,'source_pixels':hair_sources,
            'accepted_evidence_pixels':int(eligible.sum()),'adopted_pixels':changed,
            'trimmed_to_outfit_pixels':reverse_count,
            'blocked_pixels':int((anchored&~eligible).sum()),
            'blocked_outfit_core':int((anchored&outfit_core).sum()),
            'wear_source_pixels':wear_sources}
        if outfit_reference is not None:
            report['parts']['hair']['blocked_outfit_reference']=int((anchored&reference_block).sum())
        if selective:
            report['parts']['hair']['distance_or_instability_rejected_pixels']=int((anchored&~growth).sum())
            report['parts']['hair']['minimum_anchor_fraction']=.05

    conflicting_person=regions['_view_conflict']&pure[1:6].any(axis=0)
    protected_body=primary_body|result['_trusted_cores'][1:6].any(axis=0)|conflicting_person
    wear_ambiguity=object_mask&wear
    # An auxiliary object prediction is insufficient to erase reliable primary
    # clothing. Keep the same guard when SAM tries to extend a nearby object.
    protected=protected_body|wear_ambiguity|stable_outfit
    result['_object_protect']=protected.copy()
    if 'objects' in parts:
        # Reliable visible person regions and repaired parts form a complete
        # negative guard. Object-positive seeds must not include this same area.
        remove_existing=result['objects']&protected
        if remove_existing.any():
            result['objects'][remove_existing]=False
            restorable=remove_existing&strong&~regions['_view_conflict']&np.isin(primary,[1,2,3,4,5,6])
            for group in range(1,7):
                adopt(restorable&(primary==group),group,'primary_fallback')
            unknown=remove_existing&~restorable
            person[unknown]=255;uncertain[unknown]=True
            result['source_map'][unknown]=codes.get('unknown',6)
            result['consensus'][unknown]=False;result['conflict'][unknown]=True
            result['_trusted_cores'][:,unknown]=False
        seed=object_mask&~protected
        result['suspected_objects']=object_mask&protected
        result['_object_support']=object_mask.copy()
        result['_object_seed_support']=seed.copy()
        changed=adopt(seed,0,'see_object_repair')
        result['objects']|=seed
        result['conflict'][object_mask&protected]=True
        report['parts']['objects']={'source_pixels':object_sources,
            'accepted_evidence_pixels':int(seed.sum()),'adopted_pixels':changed,
            'protected_person_pixels':int((object_mask&protected_body).sum()),
            'ambiguous_wear_pixels':int(wear_ambiguity.sum()),
             'stable_outfit_conflict_pixels':int((object_mask&stable_outfit).sum()),
             'suspected_pixels':int(result['suspected_objects'].sum()),
            'retracted_unconfirmed_pixels':int(remove_existing.sum())}
    else:
        # Disabling new object repairs must not introduce original-view object
        # proposals through a later SAM task as an accidental side effect.
        result['_object_seed_support']=np.asarray(regions.get('_object_support',result['objects']),bool).copy()
        result['_object_seed_support']&=~protected

    if selective or guarded:
        # Rejected auxiliary candidates cannot bypass these decisions through
        # a later SAM part task. M2F support remains usable independently.
        groups=((2,'face'),(3,'hair'),(5,'mouth'))
        if guarded:groups+=((4,'eyes'),)
        for group,name in groups:
            allowed=pure[group]|result['_trusted_cores'][group]
            if group==3:allowed|=approved_hair
            allowed&=(person==group)|(person==0)|(person==255)
            rejected=result['_class_support'][group]&~allowed
            result['_class_support'][group]=allowed
            report.setdefault('reference_rejections',{})[name+'_sam_support_pixels']=int(rejected.sum())
        for name in ('face','mouth','hair'):
            part=report['parts'].get(name)
            if part is not None and 'unanchored_components' in part:
                report['reference_rejections'][name+'_unanchored_components']=part['unanchored_components']
    result['semantic_repairs']=report
    return result


def enforce_outfit_reference(regions, policy):
    """Keep automatic outfit within the white-view M2F proposal.

    The reference is a proposal, not ground truth: decided body labels and
    confirmed external objects retain precedence. Unresolved or background
    holes inside that proposal receive an explicit, traceable outfit fallback.
    Public masks are refreshed by the caller; later manual edits are separate.
    """
    result=dict(regions)
    if policy.get('outfit_reference')!='m2f-white':return result
    person=regions.get('person_labels')
    if (not isinstance(person,np.ndarray) or person.ndim!=2 or not person.size
            or person.dtype.kind not in 'iu' or
            not np.isin(person,[0,1,2,3,4,5,6,255]).all()):
        raise ValueError('Person labels must retain original-grid groups 0..6/255')
    shape=person.shape
    reference=_outfit_reference_mask(regions,shape,policy)
    for key in ('objects','uncertain','conflict','consensus'):
        if (not isinstance(regions.get(key),np.ndarray) or regions[key].dtype!=bool
                or regions[key].shape!=shape):
            raise ValueError('Missing or misaligned outfit region mask: '+key)
    if (not isinstance(regions.get('source_map'),np.ndarray)
            or regions['source_map'].shape!=shape or regions['source_map'].dtype.kind not in 'iu'):
        raise ValueError('Missing or misaligned outfit source map')
    for key in ('_class_support','_trusted_cores'):
        if (not isinstance(regions.get(key),np.ndarray) or regions[key].dtype!=bool
                or regions[key].shape!=(7,*shape)):
            raise ValueError('Missing or misaligned outfit support: '+key)
    for key in ('person_labels','uncertain','conflict','consensus','source_map',
                '_class_support','_trusted_cores'):
        result[key]=regions[key].copy()
    from_unknown=reference&(person==255)&~regions['objects']
    from_background=reference&(person==0)&~regions['objects']
    restored=from_unknown|from_background
    removed=(person==6)&~reference
    codes=policy.get('source_codes',{})
    result['person_labels'][restored]=6
    result['uncertain'][restored]=False
    result['source_map'][restored]=codes.get('outfit_reference',25)
    result['conflict'][restored]=True
    result['consensus'][restored]=False
    result['_class_support'][:,restored]=False
    result['_trusted_cores'][:,restored]=False
    result['_class_support'][6,restored]=True
    result['_trusted_cores'][6,restored]=True
    result['person_labels'][removed]=255
    result['uncertain'][removed]=True
    result['source_map'][removed]=codes.get('unknown',6)
    result['conflict'][removed]=True
    result['consensus'][removed]=False
    result['_class_support'][6]&=reference
    result['_trusted_cores'][6]&=reference
    previous=regions.get('outfit_reference',{})
    previous=previous if isinstance(previous,dict) and previous.get('mode')=='m2f-white' else {}
    report={'mode':'m2f-white','reference_pixels':int(reference.sum()),
        'note':'Fallback/removed-pixel counts are diagnostics, not accuracy.'}
    for key,mask in (('restored_from_unknown',from_unknown),
                     ('restored_from_background',from_background),
                     ('removed_expansion',removed)):
        report[key]=int(previous.get(key,0))+int(mask.sum())
    result['outfit_reference']=report
    return result
