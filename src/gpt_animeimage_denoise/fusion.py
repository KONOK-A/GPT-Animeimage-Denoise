"""Class-aligned evidence fusion on the original, EXIF-oriented image grid.

See raw19 is multilabel and has no background class. In particular, handwear
and legwear do not supply a bare-skin vote. Thresholds here are provisional;
neither SAM quality nor Sky alpha is used as a semantic ground truth.
"""
from copy import deepcopy

import cv2
import numpy as np
from scipy.ndimage import binary_propagation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .grouping import NAMES, V4_TO_GROUP, fuse_see_eyes, processing_regions


DEFAULT_POLICY = {
    'version': 'fusion-v1', 'provisional': True,
    'semantic_repair': 'anchored',
    'outfit_reference': 'm2f-white',
    'see_reference': 'selective',
    'part_cooperation': 'guarded',
    'unknown_resolution': 'edge-guided',
    'see_repair_parts': ['face', 'mouth', 'hair', 'neck', 'objects'],
    'm2f': .75, 'see': 1., 'margin': .15,
    'max_tasks': 16, 'max_rounds': 2,
    'max_positive_points': 64, 'max_negative_points': 64,
    'sam_box_context': .12, 'sam_box_min_at_2048': 16.,
    'sam_min_score': .5, 'sam_min_stability': .8,
    'sam_min_support': .5, 'sam_max_core_leakage': .02,
    'sam_min_component_coverage': .9,
    'roi_proposals': False,
    'default_process': ['skin', 'face', 'hair'],
    'source_codes': {'outside': 0, 'm2f_white': 1, 'm2f_original': 2,
                     'consensus': 3, 'see_eyes': 4, 'see_object': 5,
                     'unknown': 6, 'sam_part': 7, 'sam_object': 8,
                     'reference': 9, 'see_face': 20, 'see_mouth': 21,
                     'see_hair': 22, 'see_object_repair': 23, 'primary_fallback': 24,
                     'outfit_reference': 25, 'see_neck': 26,
                     'unknown_agreement': 27, 'unknown_boundary': 28},
    'see19_labels': ['hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears',
                     'earwear', 'nose', 'mouth', 'neck', 'neckwear',
                     'topwear', 'handwear', 'bottomwear', 'legwear',
                     'footwear', 'tail', 'wings', 'objects'],
    'notes': ['Thresholds require approved-reference calibration.',
              'Handwear/legwear are ambiguous, never bare-skin evidence.',
              'No head crop, default dilation, hole filling, or alpha clipping.',
              'SAM score/stability rank shapes, not semantic classes.'],
}


def _policy(policy=None):
    result = deepcopy(DEFAULT_POLICY)
    if policy is not None:
        result.update(deepcopy(policy))
        # A saved policy predating this option retains its original decisions.
        if 'semantic_repair' not in policy:
            result['semantic_repair'] = 'off'
        if 'outfit_reference' not in policy:
            result['outfit_reference'] = 'off'
        if 'see_reference' not in policy:
            result['see_reference'] = 'legacy'
        if 'part_cooperation' not in policy:
            result['part_cooperation'] = 'legacy'
        if 'unknown_resolution' not in policy:
            result['unknown_resolution'] = 'off'
        result['source_codes'] = {**DEFAULT_POLICY['source_codes'], **policy.get('source_codes', {})}
    if result['semantic_repair'] not in ('off', 'anchored'):
        raise ValueError('Policy semantic_repair must be off or anchored')
    if result['outfit_reference'] not in ('off', 'm2f-white'):
        raise ValueError('Policy outfit_reference must be off or m2f-white')
    if result['see_reference'] not in ('legacy', 'selective'):
        raise ValueError('Policy see_reference must be legacy or selective')
    if result['part_cooperation'] not in ('legacy', 'guarded'):
        raise ValueError('Policy part_cooperation must be legacy or guarded')
    if result['unknown_resolution'] not in ('off', 'edge-guided'):
        raise ValueError('Policy unknown_resolution must be off or edge-guided')
    parts = result['see_repair_parts']
    if (not isinstance(parts, list) or any(part not in ('face', 'mouth', 'hair', 'neck', 'objects') for part in parts)
            or len(set(parts)) != len(parts)):
        raise ValueError('Policy see_repair_parts must list distinct supported parts')
    for name in ('m2f', 'margin', 'sam_min_score', 'sam_min_stability',
                 'sam_min_support', 'sam_max_core_leakage', 'sam_min_component_coverage'):
        value = result[name]
        if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Policy {name} must be finite in 0..1')
    if not np.isfinite(result['see']):
        raise ValueError('Policy see must be finite')
    for name, upper in (('max_tasks', 16), ('max_rounds', 2)):
        if type(result[name]) is not int or not 0 <= result[name] <= upper:
            raise ValueError(f'Policy {name} must be integer 0..{upper}')
    for name, lower in (('max_positive_points', 1), ('max_negative_points', 0)):
        if type(result[name]) is not int or not lower <= result[name] <= 64:
            raise ValueError(f'Policy {name} must be integer {lower}..64')
    for name in ('sam_box_context', 'sam_box_min_at_2048'):
        if not isinstance(result[name], (float, int)) or not np.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f'Policy {name} must be finite and nonnegative')
    return result


def _validate_evidence(evidence, *, scores=True):
    alpha = np.asarray(evidence['alpha'])
    see = np.asarray(evidence['see_logits'])
    if alpha.dtype != np.float32 or alpha.ndim != 2 or not alpha.size or not np.isfinite(alpha).all():
        raise ValueError('Expected nonempty finite float32 alpha HxW')
    if see.dtype != np.float32 or see.shape != (19, *alpha.shape) or not np.isfinite(see).all():
        raise ValueError('Expected finite float32 See logits 19xHxW')
    original_see = evidence.get('see_original_logits')
    if original_see is not None and (not isinstance(original_see, np.ndarray)
            or original_see.dtype != np.float32 or original_see.shape != see.shape
            or not np.isfinite(original_see).all()):
        raise ValueError('Expected finite float32 original-view See logits 19xHxW')
    if scores:
        for name in ('m2f_white', 'm2f_original'):
            value = evidence.get(name)
            if value is None and name == 'm2f_original':
                continue
            value = np.asarray(value)
            if value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 12 or not value.size or not np.isfinite(value).all():
                raise ValueError(f'Expected finite float32 {name} log probabilities 12xHxW')
    return alpha, see


def _view(log_scores, shape, policy, *, return_scores=False):
    """Resize every native log score before computing labels/confidence."""
    h, w = shape
    if log_scores.shape[1:] == shape:
        up = log_scores
    else:
        up = np.stack([cv2.resize(plane, (w, h), interpolation=cv2.INTER_LINEAR)
                       for plane in log_scores]).astype(np.float32)
    # Re-normalize interpolated log probabilities, retaining all twelve classes.
    probabilities = np.exp(up - up.max(axis=0, keepdims=True))
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    raw = probabilities.argmax(axis=0).astype(np.uint8)
    grouped = V4_TO_GROUP[raw]
    # The original fine-class winner determines business mapping and feature
    # anchors. Confidence, however, belongs to the merged business classes:
    # left/right eyes, face/nose and clothes/accessory do not compete internally.
    group_probabilities = np.zeros((7, h, w), np.float32)
    for fine_class, group in enumerate(V4_TO_GROUP):
        group_probabilities[group] += probabilities[fine_class]
    chosen = np.take_along_axis(group_probabilities, grouped[None], axis=0)[0]
    np.put_along_axis(group_probabilities, grouped[None], -np.inf, axis=0)
    other = group_probabilities.max(axis=0)
    margin = chosen - other
    strong = (chosen >= policy['m2f']) & (margin >= policy['margin'])
    if return_scores:
        return grouped, strong, chosen, margin
    return raw, grouped, strong


def _see_support(see, main_groups, policy, *, m2f_support=None, stable_main=None, outfit_scope=None):
    """Map only equivalent semantic classes; preserve independent overlaps."""
    strong = see > policy['see']
    support = np.zeros((7, *main_groups.shape), bool)
    support[1] = strong[9]  # Neck can support skin; arms/legs cannot.
    support[2] = strong[2] | strong[7]
    support[3] = strong[0]
    support[4] = strong[3]
    support[5] = strong[8]
    support[6] = strong[[1, 4, 6, 10, 11, 13, 15]].any(axis=0)
    # Ear tissue may be labeled either face or exposed skin by Mask2Former.
    support[1] |= strong[5] & (main_groups == 1)
    support[2] |= strong[5] & (main_groups == 2)
    if policy.get('see_reference') == 'selective':
        strict = see > max(float(policy['see']), 2.)
        pure = m2f_support if m2f_support is not None else np.zeros_like(support)
        stable = stable_main if stable_main is not None else np.zeros_like(main_groups, bool)
        # Low-priority classes can confirm the primary model, but cannot
        # create naked skin/outfit or veto a different reliable semantic class.
        support[1] = (strict[9] | (strict[5] & (main_groups == 1))) & pure[1] & stable
        support[6] = strict[[1, 4, 6, 10, 11, 13, 15]].any(axis=0) & pure[6] & stable & (main_groups == 6)
        if outfit_scope is not None:
            support[6] &= outfit_scope
        support[3] = strict[0]
    # Handwear/legwear include bare limbs. Even agreement with an outfit label
    # cannot establish a trustworthy outfit core; retain only their raw logits.
    return support


def _refresh_regions(result):
    """Synchronize public masks after a semantic or SAM decision."""
    person = result['person_labels']
    objects = result['objects']
    person[objects] = 0
    result['uncertain'] |= person == 255
    result['uncertain'][objects] = False
    result['allowed'] = np.isin(person, [1, 2, 3]) & ~objects & ~result['uncertain']
    result['protected'] = ~result['allowed']
    result['labels'] = result['raw_labels']
    result['object_exclusion'] = objects
    result['character_allowed'] = (person > 0) & (person != 255) & ~objects
    result['object_report'] = {
        'enabled': True, 'object_pixels': int(objects.sum()),
        'removed_character_pixels': int(((result['raw_labels'] != 0) & objects).sum()),
        'object_pixels_by_group': {name: int(((result['raw_labels'] == index) & objects).sum())
                                   for index, name in enumerate(NAMES)},
    }
    return result


def reference_regions(evidence):
    """Freeze canonical argmax-then-nearest labels and old See >0 behavior."""
    _, see = _validate_evidence(evidence, scores=False)
    labels = np.asarray(evidence['reference_labels12'])
    if labels.shape != see.shape[1:]:
        raise ValueError('Reference labels must match the original grid')
    old = processing_regions(labels, see[3] > 0, see[18] > 0)
    result = dict(old)
    shape = labels.shape
    result.update(raw_labels=old['labels'].copy(), objects=old['object_exclusion'].copy(),
                  consensus=np.zeros(shape, bool), conflict=np.zeros(shape, bool),
                  uncertain=np.zeros(shape, bool),
                  source_map=np.full(shape, DEFAULT_POLICY['source_codes']['reference'], np.uint8),
                  sam_records=[], policy=_policy({'semantic_repair': 'off', 'outfit_reference': 'off'}))
    result['policy']['profile'] = 'reference'
    return result


def fuse_evidence(evidence, policy=None):
    """Fuse independent view votes and compatible multilabel evidence.

    A confident primary class survives absent auxiliary evidence. Competing
    confident classes are protected unknowns. See eyes retain anchored complete
    contours, with explicit brows/nose/mouth priorities from existing grouping.
    """
    alpha, see = _validate_evidence(evidence)
    policy = _policy(policy)
    guarded = policy['part_cooperation'] == 'guarded' and policy['semantic_repair'] == 'anchored'
    raw, white, white_strong = _view(evidence['m2f_white'], alpha.shape, policy)
    original_scores = evidence.get('m2f_original')
    original_raw = None
    if original_scores is None:
        original = white
        original_strong = np.zeros_like(white_strong)
    else:
        original_raw, original, original_strong = _view(original_scores, alpha.shape, policy)
    picked = white.copy()
    original_only = ~white_strong & original_strong
    picked[original_only] = original[original_only]
    supported = white_strong | original_strong
    agreement = white_strong & original_strong & (white == original)
    view_conflict = white_strong & original_strong & (white != original)
    conflict = view_conflict.copy()
    m2f_class_support = np.zeros((7, *alpha.shape), bool)
    for group in range(1, 7):
        m2f_class_support[group] = ((white == group) & white_strong) | ((original == group) & original_strong)
    see_support = _see_support(see, picked, policy, m2f_support=m2f_class_support,
        stable_main=supported & ~conflict,
        outfit_scope=white == 6 if policy['outfit_reference'] == 'm2f-white' else None)
    same_see = np.take_along_axis(see_support, picked[None], axis=0)[0]
    # Face is a broad region in See and may contain eyes or mouth. Their
    # narrower positive masks do not make every overlap a contradiction.
    compatible = see_support.copy()
    compatible[2] &= ~np.isin(picked, [4, 5])
    compatible[4] &= picked != 2
    compatible[5] &= picked != 2
    np.put_along_axis(compatible, picked[None], False, axis=0)
    incompatible_see = compatible.any(axis=0)
    conflict |= supported & incompatible_see

    # Start from original-grid score decisions, then apply established eye
    # contours and feature priorities. No alpha or head-box clipping occurs.
    eye_raw = raw.copy()
    if original_raw is not None:
        eye_raw[original_only] = original_raw[original_only]
    if guarded:
        raw_groups = V4_TO_GROUP[eye_raw]
        eye_report = {}
        see_eyes = np.zeros_like(supported)
    else:
        eye_evidence = see[3] > 0
        strong_eye_evidence = see[3] > policy['see']
        original_see = evidence.get('see_original_logits')
        if policy['semantic_repair'] == 'anchored' and original_see is not None:
            eye_evidence |= original_see[3] > 0
            strong_eye_evidence |= original_see[3] > policy['see']
        raw_groups, eye_report = fuse_see_eyes(eye_raw, eye_evidence)
        # Isolated See eye components must not become SAM part seeds.
        see_support[4] &= raw_groups == 4
        see_eyes = (raw_groups == 4) & eye_evidence & supported
        see_eyes &= ~np.isin(eye_raw, [6, 7, 8, 9])
        picked[see_eyes] = 4
        same_see[see_eyes] = strong_eye_evidence[see_eyes]
        conflict[see_eyes] = view_conflict[see_eyes]
    consensus = supported & (agreement | same_see) & ~conflict
    cores = np.zeros_like(see_support)
    for group in range(1, 7):
        cores[group] = (picked == group) & supported & (agreement | same_see) & ~view_conflict

    object_support = see[18] > policy['see']
    object_conflict = object_support & supported & (picked != 0)
    conflict |= object_conflict
    consensus &= ~object_conflict
    objects = object_support & (~supported | (picked == 0))
    person = picked.copy()
    unresolved = (view_conflict | object_conflict) if guarded else conflict
    uncertain = (~supported | unresolved) & ~objects
    person[uncertain] = 255
    person[objects] = 0
    source = np.full(alpha.shape, policy['source_codes']['m2f_white'], np.uint8)
    source[picked == 0] = policy['source_codes']['outside']
    source[original_only] = policy['source_codes']['m2f_original']
    source[consensus] = policy['source_codes']['consensus']
    source[see_eyes & ~uncertain] = policy['source_codes']['see_eyes']
    source[objects] = policy['source_codes']['see_object']
    source[uncertain] = policy['source_codes']['unknown']
    class_support = see_support.copy()
    for group in range(1, 7):
        class_support[group] |= m2f_class_support[group]
    if policy['see_reference'] == 'selective':
        # An accepted original-view eye contour is a resolved child region,
        # not a competing face label. Keep raw M2F evidence separately.
        accepted_eyes = see_eyes & ~uncertain
        class_support[:, accepted_eyes] = False
        class_support[4, accepted_eyes] = True
        cores[:, accepted_eyes] = False
        cores[4, accepted_eyes] = True
    result = {'raw_labels': raw_groups, 'person_labels': person, 'objects': objects,
              'consensus': consensus, 'conflict': conflict, 'uncertain': uncertain,
              'source_map': source, 'sam_records': [], 'policy': policy, 'fusion': eye_report,
              '_class_support': class_support, '_trusted_cores': cores,
              '_wear_support': see_support[6].copy(),
              '_object_support': object_support, '_m2f_labels': raw,
              '_proposal_labels': picked, '_view_conflict': view_conflict,
              '_m2f_class_support': m2f_class_support,
              '_outfit_reference_mask': white == 6}
    if policy['semantic_repair'] == 'anchored':
        from .semantic_repair import apply_see_repairs
        result = apply_see_repairs(evidence, result, primary_labels=picked,
            primary_strong=supported, primary_raw=eye_raw, policy=policy,
            m2f_views=(white, original if original_scores is not None else None))
    if policy['outfit_reference'] == 'm2f-white':
        from .semantic_repair import enforce_outfit_reference
        result = enforce_outfit_reference(result, policy)
    return _refresh_regions(result)


_RESOLVABLE_GROUPS = (0, 1, 2, 3, 6)
_WEAR_CHANNELS = (1, 4, 6, 10, 11, 13, 15)


def _competing_see(see, original_see, group, threshold):
    # Handwear and legwear are ambiguous and do not veto a naked-skin claim.
    if group == 0:
        channels = (0, 2, 3, 5, 7, 8, 9, *_WEAR_CHANNELS)
    elif group in (1, 2):
        channels = (0, 3, 8, *_WEAR_CHANNELS)
    elif group == 3:
        channels = (2, 3, 8, 9, *_WEAR_CHANNELS)
    else:
        channels = (0, 2, 3, 8, 9)
    blocked = np.zeros(see.shape[1:], bool)
    for channel in channels:
        blocked |= (see[channel] > threshold) | (original_see[channel] > threshold)
    return blocked


def _unknown_graph(rgb, person, proposals, trusted):
    """Connect admissible pixels with weighted 4-neighbor edges.

    Each class has two virtual roots: all existing labels for competition,
    and dual-strong labels only for positive propagation. Edges from the
    roots go one way so a path cannot jump between disconnected seeds.
    """
    height, width = person.shape
    flat = np.flatnonzero(proposals.ravel() != 255)
    count = flat.size
    nodes = np.full(person.size, -1, np.int32)
    nodes[flat] = np.arange(count, dtype=np.int32)
    y, x = np.divmod(flat, width)
    colors = rgb.reshape(-1, 3)
    labels = person.ravel()
    proposed = proposals.ravel()
    strict = trusted.ravel()
    costs = np.full((2, len(NAMES), count), np.inf, np.float32)
    sources, targets, distances = [], [], []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        inside = (y + dy >= 0) & (y + dy < height) & (x + dx >= 0) & (x + dx < width)
        current = flat[inside]
        adjacent = current + dy * width + dx
        delta = np.abs(colors[current].astype(np.int16) - colors[adjacent]).max(axis=1)
        # A hard contrast edge blocks traversal; weaker edges are expensive
        # compared to propagation along the same surface.
        accepted = delta <= 35
        current, adjacent, delta = current[accepted], adjacent[accepted], delta[accepted]
        edge_cost = 1 + delta.astype(np.float32) / 12
        within = (nodes[adjacent] >= 0) & (proposed[current] == proposed[adjacent])
        sources.append(nodes[current[within]])
        targets.append(nodes[adjacent[within]])
        distances.append(edge_cost[within])
        seeded = labels[adjacent] < len(NAMES)
        for layer, hits in enumerate((seeded, seeded & strict[adjacent])):
            classes, indices = labels[adjacent[hits]], nodes[current[hits]]
            costs[layer, classes, indices] = np.minimum(costs[layer, classes, indices], edge_cost[hits])
    layer, classes, indices = np.nonzero(np.isfinite(costs))
    sources.append(count + layer * len(NAMES) + classes)
    targets.append(indices)
    distances.append(costs[layer, classes, indices])
    size = count + costs.shape[0] * len(NAMES)
    graph = csr_matrix((np.concatenate(distances),
                        (np.concatenate(sources), np.concatenate(targets))), shape=(size, size))
    return graph, flat


def resolve_unknown_regions(rgb, evidence, regions, policy=None):
    """Resolve supported post-SAM gaps without changing existing labels."""
    policy = _policy(regions.get('policy') if policy is None else policy)
    result = dict(regions, policy=policy, person_labels=regions['person_labels'].copy(),
                  uncertain=regions['uncertain'].copy())
    person = result['person_labels']
    unknown = (person == 255) & result['uncertain']
    before = int(unknown.sum())
    enabled = policy['unknown_resolution'] == 'edge-guided'
    report = {'enabled': enabled, 'unknown_before': before, 'unknown_after': before,
              'agreement_pixels': 0, 'boundary_pixels': 0,
              'by_group': {NAMES[group]: {'agreement': 0, 'boundary': 0} for group in _RESOLVABLE_GROUPS}}
    result['unknown_resolution'] = report
    if (not enabled or not before or evidence.get('m2f_original') is None
            or evidence.get('see_original_logits') is None):
        return _refresh_regions(result)

    alpha, see = _validate_evidence(evidence)
    original_see = evidence['see_original_logits']
    if rgb.shape != (*person.shape, 3) or rgb.dtype != np.uint8 or alpha.shape != person.shape:
        raise ValueError('Unknown resolution needs original-grid uint8 RGB and aligned evidence')
    white, white_strong, wp, wm = _view(evidence['m2f_white'], person.shape, policy, return_scores=True)
    original, original_strong, op, om = _view(evidence['m2f_original'], person.shape, policy, return_scores=True)
    same = white == original
    trusted = same & white_strong & original_strong & (person == white)
    suspected = regions.get('suspected_objects', False)
    blocked = (regions['objects'] | suspected | regions.get('_view_conflict', False)
               | (see[18] > policy['see']) | (original_see[18] > policy['see']))
    trusted &= ~blocked & np.where(person == 0, alpha <= .08, alpha >= .93)
    if (trusted & (person == 0)).any():
        # Enclosed holes can be false background on a light garment or hair.
        # Only the exterior may seed new background, and existing labels stay.
        background = (person == 0) & (alpha <= .08) & ~blocked
        border = np.zeros_like(background)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        outside = binary_propagation(border & background, mask=background)
        trusted &= (person != 0) | outside
    # Explicitly raising the primary gate in a custom policy also raises the
    # rescue floor, so this step cannot silently defeat a stricter policy.
    mid = max(.55, policy['m2f'] - .20)
    soft = max(.48, mid - .07)
    strict_margin = max(.15, policy['margin'])
    soft_margin = max(.10, strict_margin - .05)
    wmids, omids = (wp >= mid) & (wm >= strict_margin), (op >= mid) & (om >= strict_margin)
    dual_mid = wmids & omids
    weak = (wp >= soft) & (op >= soft) & (wm >= soft_margin) & (om >= soft_margin)
    eligible = unknown & ~blocked & same & (dual_mid | (weak & (wmids | omids)))
    del wp, wm, op, om
    radius = max(2, round(12 * min(person.shape) / 2048))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    proposals = np.full(person.shape, 255, np.uint8)
    for group in _RESOLVABLE_GROUPS:
        seeds = trusted & (person == group)
        if not seeds.any():
            continue
        close = cv2.dilate(seeds.astype(np.uint8), kernel) != 0
        sky = alpha <= .08 if group == 0 else alpha >= .93
        candidate = eligible & (white == group) & close & sky
        if candidate.any():
            candidate &= ~_competing_see(see, original_see, group, max(2, policy['see']))
            proposals[candidate] = group
    if not (proposals != 255).any():
        return _refresh_regions(result)

    graph, flat = _unknown_graph(rgb, person, proposals, trusted)
    count = flat.size
    groups = len(NAMES)
    # Distance includes a color penalty. The geometric cap was already
    # enforced above, so give dual agreement room for mild surface variation.
    agreement_budget = 2 * radius
    nearby = dijkstra(graph, indices=np.arange(count, count + groups), limit=agreement_budget + 2)[:, :count]
    result['source_map'] = regions['source_map'].copy()
    for key in ('conflict', 'consensus'):
        result[key] = regions[key].copy()
    for group in _RESOLVABLE_GROUPS:
        hits = proposals.ravel()[flat] == group
        if not hits.any():
            continue
        own = dijkstra(graph, indices=count + groups + group, limit=agreement_budget)[:count][hits]
        rival = np.min(nearby[np.ix_(np.arange(groups) != group, hits)], axis=0)
        # A path must beat its competitor from actual visible seed regions;
        # softer model evidence additionally needs a short, clearer path.
        middle = dual_mid.ravel()[flat[hits]]
        accept_mid = middle & np.isfinite(own) & (own + 1 < rival)
        accept_soft = ~middle & (own <= radius) & (own + 2 < rival)
        for selection, stage in ((accept_mid, 'agreement'), (accept_soft, 'boundary')):
            resolved = flat[hits][selection]
            person.flat[resolved] = group
            result['uncertain'].flat[resolved] = False
            result['source_map'].flat[resolved] = policy['source_codes']['unknown_' + stage]
            for key in ('conflict', 'consensus'):
                result[key].flat[resolved] = False
            amount = int(resolved.size)
            report[stage + '_pixels'] += amount
            report['by_group'][NAMES[group]][stage] = amount
    report['unknown_after'] = before - report['agreement_pixels'] - report['boundary_pixels']
    return _refresh_regions(result)
