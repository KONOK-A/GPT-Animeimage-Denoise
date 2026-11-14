"""Bounded, evidence-conditioned SAM shape refinement without file/model I/O.

Each semantic task carries every supported connected component as a positive
distance-peak prompt. SAM may propose boundaries, but cannot invent a person
class or settle a disagreement between equally supported semantic classes.
"""
from copy import deepcopy

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label

from .fusion import _policy, _refresh_regions


def _peaks(mask, *, limit=None):
    """One interior XY seed per component, including one-pixel components."""
    components, _ = label(mask, structure=np.ones((3, 3), bool))
    points = []
    boxes = find_objects(components)
    indices = np.arange(len(boxes))
    if limit is not None and len(indices) > limit:
        # Spread negative prompts over the component ordering; the complete
        # protected mask still participates in leakage checks and final vetoes.
        indices = np.linspace(0, len(indices) - 1, limit, dtype=np.int64)
    for offset in indices:
        index, box = int(offset + 1), boxes[offset]
        if box is None:
            continue
        part = components[box] == index
        distance = distance_transform_edt(np.pad(part, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        points.append([int(x + box[1].start), int(y + box[0].start)])
    return points


def _context_box(support, policy):
    yy, xx = np.nonzero(support)
    h, w = support.shape
    x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)
    minimum = max(1, int(np.ceil(min(h, w) / 2048 * policy['sam_box_min_at_2048'])))
    mx = max(minimum, int(np.ceil((x1 - x0) * policy['sam_box_context'])))
    my = max(minimum, int(np.ceil((y1 - y0) * policy['sam_box_context'])))
    return [max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)]


def _prompt(support, negatives, policy):
    return {'positive': _peaks(support),
            'negative': _peaks(negatives, limit=policy['max_negative_points']),
            # Context changes the prompt box, never clips the candidate mask.
            # Exclusive upper bounds retain positive area for one-pixel parts.
            'box': _context_box(support, policy)}


def _points(prompt, shape):
    h, w = shape
    result = []
    for key in ('positive', 'negative'):
        values = np.asarray(prompt.get(key, []), np.float64)
        if values.size == 0:
            values = values.reshape(0, 2)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError('SAM prompts must contain finite pixel XY points')
        if (values < 0).any() or (values[:, 0] > w - 1).any() or (values[:, 1] > h - 1).any():
            raise ValueError('SAM point lies outside original image grid')
        result.append(np.rint(values).astype(np.int64))
    if not len(result[0]):
        raise ValueError('SAM prompt requires at least one positive point')
    return result


def _prompt_components(mask, positive, negative):
    if not mask[positive[:, 1], positive[:, 0]].all():
        return None, 'missing_positive_component'
    components, _ = label(mask, structure=np.ones((3, 3), bool))
    retained = np.isin(components, np.unique(components[positive[:, 1], positive[:, 0]]))
    if len(negative) and retained[negative[:, 1], negative[:, 0]].any():
        return None, 'negative_prompt_leakage'
    return retained, None


def _prediction(prediction):
    masks = np.asarray(prediction['masks'])
    scores = np.asarray(prediction['scores'])
    stability = np.asarray(prediction['stability'])
    if masks.dtype != bool or masks.ndim != 3 or not masks.size:
        raise ValueError('SAM candidates must be nonempty bool NxHxW')
    if scores.shape != (len(masks),) or stability.shape != scores.shape or not np.isfinite(scores).all() or not np.isfinite(stability).all():
        raise ValueError('SAM score/stability arrays must be finite and match candidates')
    lowres = prediction.get('lowres')
    if lowres is not None:
        lowres = np.asarray(lowres)
        if lowres.dtype != np.float32 or lowres.shape != (len(masks), 256, 256) or not np.isfinite(lowres).all():
            raise ValueError('SAM lowres must be finite float32 Nx256x256')
    return masks, scores, stability


def _select(prediction, prompt, policy, *, candidate=None, check=None):
    masks, scores, stability = _prediction(prediction)
    positive, negative = _points(prompt, masks.shape[1:])
    if candidate is not None and (type(candidate) is not int or not 0 <= candidate < len(masks)):
        raise ValueError('SAM candidate index is out of range')
    indices = range(len(masks)) if candidate is None else [candidate]
    rejected, accepted = [], []
    for index in indices:
        reason = None
        if scores[index] < policy['sam_min_score'] or stability[index] < policy['sam_min_stability']:
            reason = 'low_shape_quality'
        if reason is None:
            kept, reason = _prompt_components(masks[index], positive, negative)
        else:
            kept = None
        details = {}
        if reason is None and check is not None:
            reason, details = check(kept)
        if reason is not None:
            rejected.append({'candidate': int(index), 'reason': reason, **details})
        else:
            accepted.append((float(scores[index] + stability[index]), index, kept, details))
    if not accepted:
        return None, {'accepted': False, 'candidate': None, 'reason': 'no_valid_candidate', 'rejected': rejected}
    _, index, kept, details = max(accepted, key=lambda item: (item[0], -item[1]))
    return kept, {'accepted': True, 'candidate': int(index), 'score': float(scores[index]),
                  'stability': float(stability[index]), 'retained_pixels': int(kept.sum()),
                  'rejected': rejected, **details}


def select_manual_candidate(prediction, prompt, *, candidate=None):
    """Select a prompted shape, retaining all positively prompted components.

    Coordinates are original pixel XY. An explicit candidate still has to pass
    shape quality and positive/negative prompts. Semantics come from the user's
    explicit action in the caller, never from SAM's score or stability.
    """
    return _select(prediction, prompt, _policy(), candidate=candidate)


def _tasks(regions, policy):
    support = regions['_class_support']
    cores = regions['_trusted_cores']
    objects = regions['_object_support']
    object_seeds = regions.get('_object_seed_support', objects)
    groups = []
    if object_seeds.any():
        # Outfit inside an object proposal may be the very segmentation error
        # we need to correct. Such labels must never become negative prompts.
        protect = cores[1:6].any(axis=0) | regions.get('_object_protect', False)
        if policy.get('outfit_reference') == 'm2f-white':
            protect |= regions['_outfit_reference_mask'] & ~objects
        groups.append(('object', 0, object_seeds, protect))
    competing = support[1:].sum(axis=0) > 1
    for group in range(1, 7):
        eligible = support[group] & ~objects & ~regions['objects'] & ~regions['_view_conflict'] & ~competing
        needs = eligible & regions['uncertain']
        if not needs.any():
            continue
        components, _ = label(eligible, structure=np.ones((3, 3), bool))
        # Refinement is needed only for components containing unresolved pixels.
        # Known supported components remain intact without spending task slots.
        needed_components = np.unique(components[needs])
        eligible = np.isin(components, needed_components)
        # Keep all supported components for this class, not only the largest
        # component or the one nearest a head/face bounding box.
        protect = cores[1:].any(axis=0) & ~cores[group]
        groups.append(('part', group, eligible, protect))
    tasks = []
    pending = np.zeros_like(objects)
    pending_components = skipped_tasks = 0
    point_budget = policy['max_positive_points']
    for action, group, group_support, protect in groups:
        components, count = label(group_support, structure=np.ones((3, 3), bool))
        # Consecutive component IDs follow spatial scan order. Each component
        # belongs to one batch, so all included positive components are retained.
        for start in range(1, count + 1, point_budget):
            if len(tasks) >= policy['max_tasks']:
                remaining = count - start + 1
                pending |= components >= start
                pending_components += remaining
                skipped_tasks += (remaining + point_budget - 1) // point_budget
                break
            batch_support = (components >= start) & (components < start + point_budget)
            batch_protect = protect
            if action == 'object':
                x0, y0, x1, y1 = _context_box(batch_support, policy)
                context = np.zeros_like(objects)
                context[y0:y1, x0:x1] = True
                # Inside an object proposal's context, agreement between two
                # M2F views alone cannot certify an outfit boundary. Strong,
                # noncontradictory See worn-item evidence is needed there.
                verified_wear = regions['_wear_support'] & ~objects
                outfit_protect = cores[6] & (~context | verified_wear)
                batch_protect = protect | outfit_protect
            tasks.append({'action': action, 'group': group, 'support': batch_support,
                          'protect': batch_protect,
                          'prompt': _prompt(batch_support, batch_protect, policy)})
    return tasks, pending, pending_components, skipped_tasks


def _semantic_check(task, policy):
    support = task['support']
    protect = task['protect']
    components, _ = label(support, structure=np.ones((3, 3), bool))
    component_sizes = np.bincount(components.ravel())[1:]
    def check(mask):
        area = int(mask.sum())
        overlap = int((mask & support).sum())
        fraction = overlap / max(1, area)
        leakage = int((mask & protect).sum()) / max(1, area)
        details = {'semantic_support_fraction': fraction, 'trusted_core_leakage': leakage,
                   'seed_pixels': int(support.sum()), 'candidate_pixels': area}
        # A semantic object seed may cover only a small fragment of the complete
        # object. Area fraction is diagnostic, not a semantic veto on completion.
        if task['action'] != 'object' and fraction < policy['sam_min_support']:
            return 'insufficient_semantic_support', details
        if leakage > policy['sam_max_core_leakage']:
            return 'trusted_core_leakage', details
        # Every supported positive component must survive shape selection.
        # Sparse point-only masks do not qualify as preserving a component.
        kept_sizes = np.bincount(components[mask], minlength=len(component_sizes) + 1)[1:]
        coverage = kept_sizes / component_sizes
        details['semantic_support_coverage'] = float(coverage.min())
        if (coverage < policy['sam_min_component_coverage']).any():
            return 'lost_semantic_support', details
        return None, details
    return check


def _generated_prediction(proposals, shape):
    masks, scores, stability = [], [], []
    for proposal in proposals:
        mask = np.asarray(proposal.get('mask', proposal.get('masks')))
        if mask.dtype != bool or mask.shape != shape:
            raise ValueError('Generated SAM proposal must be bool on original grid')
        masks.append(mask)
        scores.append(proposal['score'])
        stability.append(proposal['stability'])
    if not masks:
        return None
    return {'masks': np.stack(masks), 'scores': np.asarray(scores, np.float32),
            'stability': np.asarray(stability, np.float32)}


def refine_regions(rgb, evidence, regions, models, policy=None):
    """Refine up to 16 semantic tasks, then resolve supported unknown pixels."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*regions['person_labels'].shape, 3):
        raise ValueError('Expected uint8 RGB on the original region grid')
    policy = _policy(regions.get('policy') if policy is None else {**regions.get('policy', {}), **policy})
    result = {key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value)
              for key, value in regions.items()}
    result['policy'] = policy
    if '_class_support' not in regions:
        # Reference is deliberately frozen and has no automatic decisions.
        return result
    tasks, pending, pending_components, skipped_tasks = _tasks(regions, policy)
    result['_sam_pending'] = pending
    for task_index, task in enumerate(tasks):
        prompt = task['prompt']
        checker = _semantic_check(task, policy)
        record = {'task': task_index, 'action': task['action'], 'group': task['group'],
                  'positive_components': len(prompt['positive']), 'prompt': deepcopy(prompt),
                  'accepted': False, 'rounds': 0, 'attempts': []}
        mask_input = None
        chosen = None
        for round_index in range(policy['max_rounds']):
            prediction = models.predict_candidates(rgb, prompt, mask_input=mask_input)
            if np.asarray(prediction['masks']).shape[1:] != rgb.shape[:2]:
                raise ValueError('SAM masks must match the original image grid')
            chosen, selection = _select(prediction, prompt, policy, check=checker)
            selection['round'] = round_index + 1
            selection['provider'] = 'prompt'
            record['attempts'].append(selection)
            record['rounds'] = round_index + 1
            if chosen is not None:
                record.update(selection)
                break
            if round_index == 0 and policy['roi_proposals']:
                generated = _generated_prediction(models.generate_candidates(rgb, prompt['box']), rgb.shape[:2])
                if generated is not None:
                    chosen, selection = _select(generated, prompt, policy, check=checker)
                    selection.update(round=round_index + 1, provider='roi')
                    record['attempts'].append(selection)
                    if chosen is not None:
                        record.update(selection)
                        break
            lowres = prediction.get('lowres')
            if lowres is not None:
                best = int(np.argmax(np.asarray(prediction['scores']) + np.asarray(prediction['stability'])))
                mask_input = np.asarray(lowres)[best:best + 1].copy()
        if chosen is not None:
            # Never alter a trusted opposing person core, even if a tiny core
            # was below a candidate's aggregate leakage tolerance.
            chosen &= ~task['protect']
            if task['action'] == 'object':
                result['objects'] |= chosen
                result['person_labels'][chosen] = 0
                result['source_map'][chosen] = policy['source_codes']['sam_object']
            else:
                # Unsupported unknown stays protected. Shape quality cannot
                # supply the missing semantic label outside this support.
                chosen &= task['support'] & ~result['objects']
                if policy.get('part_cooperation') == 'guarded':
                    chosen &= result['uncertain']
                result['person_labels'][chosen] = task['group']
                result['source_map'][chosen] = policy['source_codes']['sam_part']
            result['uncertain'][chosen] = False
            result['conflict'][chosen] = False
            result['consensus'][chosen] = False
            record['applied_pixels'] = int(chosen.sum())
        result['sam_records'].append(record)
    if skipped_tasks:
        result['sam_records'].append({'accepted': False, 'action': 'budget',
                                      'reason': 'automatic_task_budget_exhausted',
                                      'skipped_tasks': skipped_tasks,
                                      'pending_components': pending_components,
                                      'pending_pixels': int(pending.sum())})
    if policy.get('outfit_reference') == 'm2f-white':
        from .semantic_repair import enforce_outfit_reference
        result = enforce_outfit_reference(result, policy)
    if policy.get('unknown_resolution') == 'edge-guided':
        from .fusion import resolve_unknown_regions
        return resolve_unknown_regions(rgb, evidence, result, policy)
    return _refresh_regions(result)
