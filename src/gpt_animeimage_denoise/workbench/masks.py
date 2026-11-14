"""Replay workbench edits and compute blend weights on the oriented source grid."""
from dataclasses import dataclass
from math import ceil

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter

from ..editor import DEFAULT_MODULES, MODULES
from ..filters import _compose_weight, _line_weight, build_weight, validate_unit_interval


@dataclass
class Selection:
    weight: np.ndarray
    labels: np.ndarray
    objects: np.ndarray
    unknown: np.ndarray
    soft_protection: np.ndarray
    selected: np.ndarray
    edit_feedback: dict | None = None
    has_local_mix: bool = False
    has_unclassified_selection: bool = False


class WorkbenchWeightCache:
    """Keep original-image line weights and the last geometry for one editor."""

    def __init__(self, editor):
        self.editor = editor
        self.scale = min(editor.labels.shape) / 2048
        self.lines = None
        self.selected = None
        self.structure = None
        self.base = None

    def weight(self, selected, structure):
        if self.structure != structure or not np.array_equal(selected, self.selected):
            lines = None
            if structure > 0 and selected.any():
                if self.lines is None:
                    self.lines = _line_weight(self.editor.rgb, 5 * self.scale, 4)
                lines = self.lines
            if self.structure == structure:
                self._update_geometry(selected, lines, structure)
            else:
                self.base = _compose_weight(selected, 16 * self.scale, lines, structure)
            self.selected = selected.copy()
            self.structure = structure
        return self.base.copy()

    def _update_geometry(self, selected, lines, structure):
        # Feathering saturates beyond its radius. A change only affects pixels
        # in that radius, and a second halo gives the crop all relevant context.
        feather = 16 * self.scale
        halo = ceil(feather)
        changed = selected != self.selected
        rows = np.flatnonzero(changed.any(axis=1))
        columns = np.flatnonzero(changed.any(axis=0))
        height, width = selected.shape
        y0, y1 = max(0, int(rows[0]) - halo), min(height, int(rows[-1]) + 1 + halo)
        x0, x1 = max(0, int(columns[0]) - halo), min(width, int(columns[-1]) + 1 + halo)
        cy0, cy1 = max(0, y0 - halo), min(height, y1 + halo)
        cx0, cx1 = max(0, x0 - halo), min(width, x1 + halo)
        if (cy1 - cy0) * (cx1 - cx0) >= selected.size // 4:
            self.base = _compose_weight(selected, feather, lines, structure)
            return
        context = (slice(cy0, cy1), slice(cx0, cx1))
        local = _compose_weight(selected[context], feather,
                                lines[context] if lines is not None else None, structure)
        self.base[y0:y1, x0:x1] = local[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]


def _part_strengths(settings):
    strengths = settings.get('part_strengths', {})
    if not isinstance(strengths, dict) or strengths.keys() - MODULES.keys():
        raise ValueError('Invalid part strengths')
    return {name: validate_unit_interval(value, name + ' strength') for name, value in strengths.items()}


def _rasterize(edit, shape):
    """Return the affected tile and a float mask with its brush feather."""
    points = np.asarray(edit.get('points'), dtype=np.float64)
    height, width = shape
    if (points.ndim != 2 or points.shape[1:] != (2,) or not len(points)
            or not np.isfinite(points).all() or (points < 0).any()
            or (points[:, 0] >= width).any() or (points[:, 1] >= height).any()):
        raise ValueError('Edit points must be finite original-grid [x, y] pairs')
    form = edit.get('shape', 'brush')
    if form == 'polygon':
        if len(points) < 3:
            raise ValueError('A polygon needs at least three points')
        radius = 0
    elif form == 'brush':
        radius = edit.get('radius', 24)
        if type(radius) is not int or not 1 <= radius <= 256:
            raise ValueError('Brush radius must be an integer in 1..256')
    else:
        raise ValueError('Unknown edit shape')

    pixels = np.floor(points).astype(np.intp)
    x0, y0 = np.maximum(0, pixels.min(axis=0) - radius)
    x1, y1 = np.minimum((width, height), pixels.max(axis=0) + radius + 1)
    tile = (slice(y0, y1), slice(x0, x1))
    image = Image.new('1', (int(x1 - x0), int(y1 - y0)))
    draw = ImageDraw.Draw(image)
    local = [tuple(map(int, pixel)) for pixel in pixels - (x0, y0)]
    if form == 'polygon':
        draw.polygon(local, fill=1)
        return tile, np.asarray(image, dtype=np.float32)
    if len(local) > 1:
        draw.line(local, fill=1, width=1)
    else:
        draw.point(local[0], fill=1)
    distance = distance_transform_edt(~np.asarray(image))
    if edit['tool'] not in ('protect', 'mix'):
        return tile, (distance <= radius).astype(np.float32)
    # Flat within half the radius, easing to zero at the visible brush outline.
    edge = np.clip(2 - 2 * distance / radius, 0, 1).astype(np.float32)
    return tile, edge * edge * (3 - 2 * edge)


def _selected(labels, auto_labels, geometry, blocked):
    return ((np.isin(labels, auto_labels) & (geometry == 0)) | (geometry > 0)) & ~blocked


def _plain_include_brush(edit):
    return (isinstance(edit, dict) and edit.get('tool') == 'include'
            and edit.get('shape', 'brush') == 'brush' and not edit.get('allow_protected', False))


def _resolution_module(edit, previous):
    module = edit.get('module')
    if (set(edit) - {'tool', 'module'} or not _plain_include_brush(previous)
            or ('module' in edit and (not isinstance(module, str) or not MODULES.get(module)))):
        raise ValueError('待确认处理必须紧接普通补选笔画，并指定有效部位或仅加入处理范围。')
    return module


def _region_factor(labels, selected, active, strengths, scale):
    factors = np.ones(256, np.float32)
    for name in active:
        factors[MODULES[name]] = strengths.get(name, 1)
    if np.all(factors == 1):
        return 1
    # A manually added area in a disabled part follows the overall strength.
    field = np.where(selected, factors[labels], 0)
    if not np.any(selected):
        return 1
    # Two sigma on each side gives a transition of roughly 4 px at short edge 2048.
    numerator = gaussian_filter(field, sigma=scale, truncate=2)
    denominator = gaussian_filter(selected.astype(np.float32), sigma=scale, truncate=2)
    np.divide(numerator, denominator, out=field, where=selected)
    np.clip(field, 0, 1, out=field)
    return field


def render_selection(editor, settings, edits, cursor, *, correction=None, object_mask=None, weight_cache=None):
    """Replay only the active prefix of a document's reversible edit history."""
    if type(cursor) is not int or not 0 <= cursor <= len(edits):
        raise ValueError('Invalid edit history cursor')
    modules = settings.get('modules', DEFAULT_MODULES)
    if (not isinstance(modules, (tuple, list))
            or any(not isinstance(name, str) or name not in MODULES for name in modules)):
        raise ValueError('Invalid selection modules')
    strengths = _part_strengths(settings)
    active = {name for name in modules if strengths.get(name, 1) > 0}
    auto_labels = [MODULES[name] for name in active]
    strength = validate_unit_interval(settings.get('strength', 1), 'overall strength')
    structure = validate_unit_interval(settings.get('structure_protection', .2), 'line protection')

    labels = editor.labels.copy()
    objects = editor.objects.copy()
    unknown = editor.unknown.copy()
    transparent = editor.alpha == 0 if editor.alpha is not None else np.zeros_like(objects)
    predicted = editor.objects | editor.unknown
    marked_objects = np.zeros_like(objects)
    geometry = np.zeros(labels.shape, np.int8)
    protection = np.zeros(labels.shape, np.float32)
    unlocked = mix_alpha = mix_value = None
    if correction is not None:
        if correction.get('sam_regions'):
            raise ValueError('Workbench import requires binary corrections without SAM prompts')
        marked_objects = correction['object_add_mask'] & ~transparent
        objects = (objects & ~correction['object_remove_mask']) | marked_objects
        for name, mask in correction['module_overrides'].items():
            hits = mask & ~transparent & ~objects
            labels[hits] = MODULES[name]
            unknown[hits] = False
        geometry[correction['allow_mask'] & ~transparent & ~objects & ~unknown] = 1
        protection[correction['protect_mask'] & ~transparent] = 1

    edit_feedback = None
    skipped_by_brush = None
    for index, edit in enumerate(edits[:cursor], start=1):
        if not isinstance(edit, dict):
            raise ValueError('Invalid edit')
        tool = edit.get('tool')
        if tool == 'resolve_unknown':
            module = _resolution_module(edit, edits[index - 2] if index > 1 else None)
            tile, hits = skipped_by_brush
            if module is not None:
                labels[tile][hits] = MODULES[module]
                unknown[tile][hits] = False
            elif hits.any():
                if unlocked is None:
                    unlocked = np.zeros(labels.shape, bool)
                unlocked[tile][hits] = True
                geometry[tile][hits] = 1
            continue
        if tool == 'sam_object':
            if object_mask is None:
                raise ValueError('SAM object edits require a saved mask')
            hits = object_mask(edit['token']) & ~transparent
            marked_objects |= hits
            objects |= hits
            unknown[hits] = False
            labels[hits] = 0
            if unlocked is not None:
                unlocked[hits] = False
            continue
        if tool == 'reset':
            geometry.fill(0)
            unlocked = None
            continue
        if tool == 'invert':
            blocked = objects | unknown
            if unlocked is not None:
                blocked &= ~unlocked
            blocked |= transparent | (protection == 1)
            selected = _selected(labels, auto_labels, geometry, blocked)
            geometry[~blocked & ~selected] = 1
            geometry[selected] = -1
            continue
        if tool not in ('include', 'exclude', 'protect', 'mix', 'erase', 'override'):
            raise ValueError('Unknown edit tool')
        tile, brush = _rasterize(edit, labels.shape)
        hits = (brush > 0) & ~transparent[tile]
        if tool == 'erase':
            if index == cursor:
                manual = ((geometry[tile] != 0) | (protection[tile] > 0)
                          | (labels[tile] != editor.labels[tile])
                          | (objects[tile] != editor.objects[tile])
                          | (unknown[tile] != editor.unknown[tile]) | marked_objects[tile])
                if unlocked is not None:
                    manual |= unlocked[tile]
                if mix_alpha is not None:
                    manual |= mix_alpha[tile] > 0
                changed = int(np.count_nonzero(hits & manual))
                edit_feedback = {'tool': tool, 'changed_pixels': changed,
                                 'reason': 'changed' if changed else 'no_marks'}
            geometry[tile][hits] = 0
            protection[tile][hits] = 0
            labels[tile][hits] = editor.labels[tile][hits]
            objects[tile][hits] = editor.objects[tile][hits]
            unknown[tile][hits] = editor.unknown[tile][hits]
            marked_objects[tile][hits] = False
            if unlocked is not None:
                unlocked[tile][hits] = False
            if mix_alpha is not None:
                mix_alpha[tile][hits] = mix_value[tile][hits] = 0
        elif tool == 'override':
            name = edit.get('module')
            if not isinstance(name, str) or name not in MODULES or name == 'background':
                raise ValueError('A correction must name a character part')
            hits &= predicted[tile] & ~marked_objects[tile]
            labels[tile][hits] = MODULES[name]
            objects[tile][hits] = False
            unknown[tile][hits] = False
            if unlocked is not None:
                unlocked[tile][hits] = False
        elif tool == 'exclude':
            if index == cursor:
                blocked = objects[tile] | unknown[tile]
                if unlocked is not None:
                    blocked &= ~unlocked[tile]
                blocked |= ~hits | (protection[tile] == 1)
                selected_before = _selected(labels[tile], auto_labels, geometry[tile], blocked)
                changed = int(np.count_nonzero(selected_before))
                edit_feedback = {'tool': tool, 'changed_pixels': changed,
                                 'reason': 'changed' if changed else 'already_excluded'}
            geometry[tile][hits] = -1
        elif tool == 'include':
            allow_protected = edit.get('allow_protected', False)
            if type(allow_protected) is not bool:
                raise ValueError('Explicit model-veto override must be boolean')
            model_blocked = objects[tile] | unknown[tile]
            if unlocked is not None:
                model_blocked &= ~unlocked[tile]
            hits &= protection[tile] < 1
            skipped = None
            if (_plain_include_brush(edit) and (index == cursor or
                    (isinstance(edits[index], dict) and edits[index].get('tool') == 'resolve_unknown'))):
                skipped = hits & unknown[tile] & ~objects[tile]
                if unlocked is not None:
                    skipped &= ~unlocked[tile]
                skipped_by_brush = (tile, skipped)
            already_writable = hits & ~model_blocked
            if allow_protected:
                hits &= ~marked_objects[tile]
                new_unlock = hits & model_blocked
                if new_unlock.any():
                    if unlocked is None:
                        unlocked = np.zeros(labels.shape, bool)
                    unlocked[tile][new_unlock] = True
            else:
                hits = already_writable
            if index == cursor:
                selected_before = _selected(labels[tile], auto_labels, geometry[tile], ~already_writable)
                writable = int(np.count_nonzero(hits))
                changed = writable - int(np.count_nonzero(selected_before))
                edit_feedback = {'tool': tool, 'changed_pixels': changed,
                                 'reason': 'changed' if changed else 'already_selected' if writable else 'protected'}
                skipped_count = int(np.count_nonzero(skipped)) if skipped is not None else 0
                if skipped_count:
                    edit_feedback['skipped_unknown_pixels'] = skipped_count
            geometry[tile][hits] = 1
        elif tool == 'mix':
            original = validate_unit_interval(edit.get('strength', .9), 'original fraction')
            blocked = objects[tile] | unknown[tile]
            if unlocked is not None:
                blocked &= ~unlocked[tile]
            hits &= ~blocked & (protection[tile] < 1)
            if hits.any():
                if mix_alpha is None:
                    mix_alpha = np.zeros(labels.shape, np.float32)
                    mix_value = np.zeros(labels.shape, np.float32)
                coverage = mix_alpha[tile]
                value = mix_value[tile]
                amount = brush[hits]
                coverage[hits] = (1 - amount) * coverage[hits] + amount
                value[hits] = (1 - amount) * value[hits] + amount * np.float32(1 - original)
        else:
            opacity = validate_unit_interval(edit.get('strength', .9), 'brush protection')
            marks = protection[tile]
            marks[hits] = np.maximum(marks[hits], brush[hits] * np.float32(opacity))

    unclassified = None
    if unlocked is not None:
        unclassified = unknown & unlocked
        objects &= ~unlocked
        unknown &= ~unlocked
    blocked = transparent | objects | unknown | (protection == 1)
    selected = _selected(labels, auto_labels, geometry, blocked)
    scale = min(labels.shape) / 2048
    if weight_cache is None:
        weight = build_weight(selected, blocked, editor.rgb, feather_radius=16 * scale,
                              structure_radius=5 * scale, structure_protection=structure)
    else:
        if weight_cache.editor is not editor:
            raise ValueError('Weight cache belongs to a different editor')
        weight = weight_cache.weight(selected, structure)
    weight *= np.float32(strength)
    weight *= _region_factor(labels, selected, active, strengths, scale)
    weight *= 1 - protection
    weight[weight < 1 / 255] = 0
    has_local_mix = False
    if mix_alpha is not None:
        mixed = selected & (mix_alpha > 0)
        has_local_mix = bool(mixed.any())
        weight[mixed] = (1 - mix_alpha[mixed]) * weight[mixed] + mix_value[mixed]
        # Floating-point compositing can overshoot by a few ulps.
        np.clip(weight, 0, 1, out=weight)
    return Selection(weight, labels, objects, unknown, protection, selected, edit_feedback,
                     has_local_mix, unclassified is not None and bool((unclassified & selected).any()))


def binary_corrections(editor, selection, settings):
    """Translate exactly representable workbench masks to CLI version 2."""
    if selection.has_local_mix:
        raise ValueError('CLI edits cannot represent local mixing')
    if selection.has_unclassified_selection:
        raise ValueError('Assign an explicit class to unlocked unknown pixels before CLI export')
    protection = selection.soft_protection
    if (any(value != 1 for value in _part_strengths(settings).values())
            or np.any((protection > 0) & (protection < 1))):
        raise ValueError('CLI edits cannot represent fractional part strengths or soft protection')
    corrected = ((selection.labels != editor.labels) | (editor.unknown & ~selection.unknown)) & ~selection.objects
    overrides = {name: corrected & (selection.labels == label) for name, label in MODULES.items() if label > 0}
    default = np.isin(selection.labels, [MODULES[name] for name in DEFAULT_MODULES])
    default &= ~selection.objects & ~selection.unknown
    if editor.alpha is not None:
        default &= editor.alpha != 0
    removed_objects = editor.objects & ~selection.objects
    # CLI object removal lifts the object veto but leaves its original protection until allowed.
    return {'allow_mask': selection.selected & (~default | removed_objects),
            'protect_mask': (default & ~selection.selected) | (protection == 1),
            'object_add_mask': selection.objects & ~editor.objects,
            'object_remove_mask': removed_objects,
            'module_overrides': {name: mask for name, mask in overrides.items() if mask.any()}}
