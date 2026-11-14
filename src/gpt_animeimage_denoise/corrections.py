"""Source-bound V2.1 manual masks and correction prompts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

MODULES = {'skin': 1, 'face': 2, 'hair': 3, 'eyes': 4, 'mouth': 5, 'outfit': 6}
MASK_KEYS = ('object_add_mask', 'object_remove_mask', 'allow_mask', 'protect_mask')


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _sam_regions(value):
    if not isinstance(value, list):
        raise ValueError('sam_regions must be a list')
    for region in value:
        if not isinstance(region, dict) or region.get('action') not in ('object_add', 'object_remove', 'part', 'allow', 'protect'):
            raise ValueError('Invalid SAM action')
        unknown = set(region) - {'action', 'part', 'positive', 'negative', 'box', 'candidate', 'notes'}
        if unknown:
            raise ValueError('Unknown SAM fields: ' + ', '.join(sorted(unknown)))
        if region['action'] == 'part' and region.get('part') not in MODULES:
            raise ValueError('SAM part must name one of the six modules')
        if 'candidate' in region and (type(region['candidate']) is not int or region['candidate'] not in (0, 1, 2)):
            raise ValueError('SAM candidate must be 0, 1 or 2 when supplied')
        for name in ('positive', 'negative'):
            points = np.asarray(region.get(name, []), dtype=np.float64)
            if not points.size and name == 'negative':
                continue
            if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                    or not np.isfinite(points).all() or np.any((points < 0) | (points > 1))):
                raise ValueError('SAM points must be normalized finite Nx2 coordinates')
        if 'box' in region:
            box = np.asarray(region['box'], dtype=np.float64)
            if (box.shape != (4,) or not np.isfinite(box).all() or np.any((box < 0) | (box > 1))
                    or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError('SAM box must be normalized ordered XYXY')
    return value


def load_edits(path, shape, source_sha256):
    """Validate oriented [W,H], source bytes and every mask before inference."""
    edits = {key: np.zeros(shape, bool) for key in MASK_KEYS}
    edits.update(module_overrides={}, sam_regions=[], hashes={}, input_paths=[])
    if path is None:
        return edits
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Edits JSON must be an object')
    unknown = set(data) - {*MASK_KEYS, 'module_overrides', 'sam_regions', 'notes', 'version', 'source_sha256', 'size', 'id'}
    if unknown:
        raise ValueError('Unknown edit keys: ' + ', '.join(sorted(unknown)))
    if type(data.get('version')) is not int or data['version'] != 2:
        raise ValueError('New profiles require edits version 2')
    if data.get('source_sha256') != source_sha256:
        raise ValueError('Edits source_sha256 does not match source bytes')
    if data.get('size') != [shape[1], shape[0]]:
        raise ValueError('Edits size must match EXIF-oriented [W,H]')
    edits['hashes']['edits_json'] = _sha(path)
    edits['input_paths'].append(path)

    def mask(name, value):
        if not isinstance(value, str) or not value:
            raise ValueError(name + ' must contain a mask path')
        resolved = (path.parent / value).resolve()
        with Image.open(resolved) as image:
            if image.size != (shape[1], shape[0]):
                raise ValueError(name + ' dimensions must match EXIF-oriented source')
            result = np.array(image.convert('L')) > 0
        edits['hashes'][name] = _sha(resolved)
        edits['input_paths'].append(resolved)
        return result

    for key in MASK_KEYS:
        if key in data:
            edits[key] = mask(key, data[key])
    overrides = data.get('module_overrides', {})
    if not isinstance(overrides, dict) or set(overrides) - MODULES.keys():
        raise ValueError('module_overrides must map six module names to mask paths')
    for name, value in overrides.items():
        edits['module_overrides'][name] = mask('module_' + name, value)
    _check_overlaps(edits)
    edits['sam_regions'] = _sam_regions(data.get('sam_regions', []))
    return edits


def _check_overlaps(edits):
    if np.any(edits['object_add_mask'] & edits['object_remove_mask']):
        raise ValueError('object_add and object_remove masks overlap')
    used = np.zeros_like(edits['allow_mask'])
    for name, mask in edits['module_overrides'].items():
        if np.any(used & mask):
            raise ValueError('Manual module masks overlap: ' + name)
        if np.any(mask & edits['object_add_mask']):
            raise ValueError('Manual module and object_add masks overlap: ' + name)
        used |= mask


def apply_corrections(regions, edits, sam_masks=()):
    """Reviewed allow expands processing scope; semantic/manual vetoes still win."""
    result = {key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value) for key, value in regions.items()}
    merged = {key: edits[key].copy() for key in MASK_KEYS}
    merged['module_overrides'] = {name: value.copy() for name, value in edits['module_overrides'].items()}
    for region, mask in sam_masks:
        if mask.dtype != bool or mask.shape != result['person_labels'].shape:
            raise ValueError('Manual SAM mask shape/type mismatch')
        action = region['action']
        if action == 'part':
            name = region['part']
            merged['module_overrides'].setdefault(name, np.zeros_like(mask))
            merged['module_overrides'][name] |= mask
        else:
            merged[{'allow': 'allow_mask', 'protect': 'protect_mask',
                    'object_add': 'object_add_mask', 'object_remove': 'object_remove_mask'}[action]] |= mask
    _check_overlaps(merged)
    labels = result['person_labels']
    protected = result['protected'].copy()
    allowed = result['allowed'].copy()
    objects = (result['objects'] | merged['object_add_mask']) & ~merged['object_remove_mask']
    source_codes = {'manual_object': 10, 'manual_part': 11, 'manual_allow': 12,
                    'manual_protect': 13, 'manual_object_remove': 14}
    if 'policy' in result:
        result['policy'].setdefault('source_codes', {}).update(source_codes)
    if 'source_map' in result:
        result['source_map'][merged['object_remove_mask']] = source_codes['manual_object_remove']
    for name, mask in merged['module_overrides'].items():
        labels[mask] = MODULES[name]
        protected[mask] = MODULES[name] not in (1, 2, 3)
        allowed[mask] = MODULES[name] in (1, 2, 3)
        for key in ('uncertain', 'conflict', 'consensus'):
            if key in result:
                result[key][mask] = False
        if 'source_map' in result:
            result['source_map'][mask & ~objects] = source_codes['manual_part']
    # Removing an object only removes that object veto. It does not invent a part.
    # Automatic protected is the complement of the default processing scope.
    # Allow may lift that default, but never the hard vetoes restored below.
    protected &= ~merged['allow_mask']
    protected |= (labels == 255) | objects | merged['protect_mask']
    allowed = (allowed | merged['allow_mask']) & ~protected & ~objects & (labels != 255)
    labels[objects] = 0
    for key in ('uncertain', 'conflict', 'consensus'):
        if key in result:
            result[key][objects] = False
    if 'source_map' in result:
        result['source_map'][merged['allow_mask'] & allowed] = source_codes['manual_allow']
        result['source_map'][merged['object_add_mask']] = source_codes['manual_object']
        result['source_map'][merged['protect_mask']] = source_codes['manual_protect']
    result.update(person_labels=labels, objects=objects, protected=protected, allowed=allowed)
    if 'object_exclusion' in result:
        result['object_exclusion'] = objects.copy()
    if 'character_allowed' in result:
        result['character_allowed'] = (labels > 0) & (labels != 255) & ~objects
    if 'object_report' in result:
        result['object_report']['object_pixels'] = int(objects.sum())
    return result
