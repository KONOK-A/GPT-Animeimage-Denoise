"""Replay workbench edits and compute blend weights on the oriented source grid."""
from dataclasses import dataclass
from itertools import groupby
import json
from math import ceil

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter

from ..editor import DEFAULT_MODULES, MODULES
from ..filters import _compose_weight, _line_weight, build_weight, validate_unit_interval


PART_CODES = {**MODULES, 'object': 7, 'unknown': 255}
PART_ASSIGNMENT_TOOLS = ('classify', 'remove_part')
STAGE_TOOLS = {'parts': (*PART_ASSIGNMENT_TOOLS, 'erase'),
               'range': ('include', 'exclude', 'protect', 'mix', 'erase', 'invert', 'reset', 'resolve_unknown')}
_MAX_COVERAGE_CACHE_BYTES = 64 * 1024 * 1024


def part_ids(labels, objects, unknown):
    ids = labels.copy()
    ids[unknown] = PART_CODES['unknown']
    ids[objects] = PART_CODES['object']
    return ids


def _locked_part_mask(labels, objects, unknown, protected):
    return np.isin(part_ids(labels, objects, unknown), [PART_CODES[part] for part in protected])


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
    fractional_range: bool = False
    legacy_parts_overlay: bool = True
    legacy_semantic_masks: tuple[np.ndarray, np.ndarray] | None = None
    adaptive_range: bool = False


def _join_edge_widths(selected, widths, default_width):
    adapted = selected & (widths < default_width)
    rows = np.flatnonzero(adapted.any(axis=1))
    if not len(rows):
        return widths
    columns = np.flatnonzero(adapted.any(axis=0))
    steps = min(12, ceil(default_width))
    height, width = selected.shape
    crop = (slice(max(0, rows[0] - steps), min(height, rows[-1] + 1 + steps)),
            slice(max(0, columns[0] - steps), min(width, columns[-1] + 1 + steps)))
    inside, local = selected[crop], widths[crop]
    vertical = inside[1:] & inside[:-1]
    horizontal = inside[:, 1:] & inside[:, :-1]
    if not (np.any(vertical & (local[1:] != local[:-1]))
            or np.any(horizontal & (local[:, 1:] != local[:, :-1]))):
        return widths

    # Smooth the rate of the inward fade along shared selection edges.
    # Exchanging only with selected neighbors keeps separate regions independent.
    default_rate = 1 / default_width
    initial = np.where(inside, np.reciprocal(local) - default_rate, 0)
    rates = initial
    for _ in range(steps):
        blended = rates.copy()
        change = (rates[1:] - rates[:-1]) * vertical * .2
        blended[:-1] += change
        blended[1:] -= change
        change = (rates[:, 1:] - rates[:, :-1]) * horizontal * .2
        blended[:, :-1] += change
        blended[:, 1:] -= change
        rates = blended
    joined = widths.copy()
    np.reciprocal(rates + default_rate, out=joined[crop], where=inside & (rates != initial))
    return joined


class WorkbenchWeightCache:
    """Reuse stroke coverage and original-image weights for one editor."""

    def __init__(self, editor, *, object_mask=None):
        self.editor = editor
        self.object_mask = object_mask
        self._coverage = {}
        self._coverage_bytes = 0
        self.scale = min(editor.labels.shape) / 2048
        self.lines = None
        self.selected = None
        self.edge_widths = None
        self.structure = None
        self.base = None

    def stroke_coverage(self, edit):
        def rasterize():
            return _rasterize(edit, self.editor.labels.shape)

        try:
            key = _stroke_coverage_key(edit)
        except TypeError:
            # Direct callers may supply an array rather than JSON draft points.
            return rasterize()
        return self._cached_coverage(key, rasterize)

    def saved_coverage(self, token, object_mask, *, full=False):
        def read():
            outline = object_mask(token)
            return (None, outline) if full else _outline_coverage(outline)

        if object_mask is not self.object_mask or not isinstance(token, str):
            return read()
        prefix = 'sam-outline:' if full else 'sam-selection:'
        return self._cached_coverage(prefix + token, read)

    def _cached_coverage(self, key, calculate):
        cached = self._coverage.get(key)
        if cached is not None:
            return cached
        tile, mask = calculate()
        size = mask.nbytes + len(key)
        # Old strokes are visited first on every replay. Keeping the admitted
        # tiles avoids repeatedly evicting and recalculating the same history.
        if self._coverage_bytes + size <= _MAX_COVERAGE_CACHE_BYTES:
            if tile is None:
                mask = mask.copy()
            mask.setflags(write=False)
            self._coverage[key] = (tile, mask)
            self._coverage_bytes += size
        return tile, mask

    def weight(self, selected, structure, edge_widths=None):
        same_edges = edge_widths is self.edge_widths or np.array_equal(edge_widths, self.edge_widths)
        if self.structure != structure or not np.array_equal(selected, self.selected) or not same_edges:
            lines = None
            if structure > 0 and selected.any():
                if self.lines is None:
                    self.lines = _line_weight(self.editor.rgb, 5 * self.scale, 4)
                lines = self.lines
            if self.structure == structure:
                self._update_geometry(selected, lines, structure, edge_widths)
            else:
                self.base = self._compose(selected, lines, structure, edge_widths)
            self.selected = selected.copy()
            if edge_widths is not None:
                edge_widths.setflags(write=False)
            self.edge_widths = edge_widths
            self.structure = structure
        return self.base.copy()

    def _compose(self, selected, lines, structure, edge_widths):
        default = 16 * self.scale
        feather = default if edge_widths is None else _join_edge_widths(selected, edge_widths, default)
        return _compose_weight(selected, feather, lines, structure)

    def _update_geometry(self, selected, lines, structure, edge_widths):
        # The fade and join blending both have finite support within this radius.
        # A second halo gives the crop the context needed to recompute that area.
        feather = 16 * self.scale
        halo = ceil(feather)
        changed = selected != self.selected
        if edge_widths is not self.edge_widths:
            previous = feather if self.edge_widths is None else self.edge_widths
            current = feather if edge_widths is None else edge_widths
            changed |= (previous != current) & selected
        rows = np.flatnonzero(changed.any(axis=1))
        if not len(rows):
            return
        columns = np.flatnonzero(changed.any(axis=0))
        height, width = selected.shape
        y0, y1 = max(0, int(rows[0]) - halo), min(height, int(rows[-1]) + 1 + halo)
        x0, x1 = max(0, int(columns[0]) - halo), min(width, int(columns[-1]) + 1 + halo)
        cy0, cy1 = max(0, y0 - halo), min(height, y1 + halo)
        cx0, cx1 = max(0, x0 - halo), min(width, x1 + halo)
        if (cy1 - cy0) * (cx1 - cx0) >= selected.size // 4:
            self.base = self._compose(selected, lines, structure, edge_widths)
            return
        context = (slice(cy0, cy1), slice(cx0, cx1))
        local = self._compose(selected[context], lines[context] if lines is not None else None, structure,
                              edge_widths[context] if edge_widths is not None else None)
        self.base[y0:y1, x0:x1] = local[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]


def _part_strengths(settings):
    strengths = settings.get('part_strengths', {})
    if not isinstance(strengths, dict) or strengths.keys() - MODULES.keys():
        raise ValueError('Invalid part strengths')
    return {name: validate_unit_interval(value, name + ' strength') for name, value in strengths.items()}


def _stroke_coverage_key(edit):
    shape = edit.get('shape', 'brush')
    geometry = [shape, edit.get('points')]
    if shape == 'brush':
        staged = 'stage' in edit
        feather = edit.get('feather', 0) if staged else edit.get('tool') in ('protect', 'mix')
        geometry.extend((edit.get('radius', 24), edit.get('geometry'), staged, feather))
    return json.dumps(geometry, separators=(',', ':'))


def _outline_coverage(outline):
    rows, columns = np.flatnonzero(outline.any(axis=1)), np.flatnonzero(outline.any(axis=0))
    if not len(rows):
        raise ValueError('保存的 SAM2 选区为空。')
    tile = (slice(rows[0], rows[-1] + 1), slice(columns[0], columns[-1] + 1))
    return tile, outline[tile].astype(np.float32)


def _continuous_distance(points, shape, radius):
    distance2 = np.full(shape, (radius + 1) ** 2, np.float32)
    height, width = shape
    starts, ends = (points[:-1], points[1:]) if len(points) > 1 else (points, points)
    for first, last in zip(starts, ends):
        vector = last - first
        # Bound intermediate arrays for a long diagonal between input samples.
        steps = max(1, ceil(float(np.max(np.abs(vector))) / 256))
        step = vector / steps
        dx, dy = step.astype(np.float32)
        length2 = dx * dx + dy * dy
        for index in range(steps):
            start = first + index * step
            end = start + step
            lower = np.floor(np.minimum(start, end)).astype(np.intp) - radius
            upper = np.floor(np.maximum(start, end)).astype(np.intp) + radius + 1
            x0, y0 = np.maximum(0, lower)
            x1, y1 = np.minimum((width, height), upper)
            x = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(start[0])
            y = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(start[1])
            if length2 > 0:
                along = np.clip((x * dx + y * dy) / length2, 0, 1)
                x = x - dx * along
                y = y - dy * along
            local = distance2[y0:y1, x0:x1]
            np.minimum(local, x * x + y * y, out=local)
    return np.sqrt(distance2, out=distance2)


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
    if form == 'brush' and edit.get('geometry') == 'continuous':
        distance = _continuous_distance(points - (x0, y0), (int(y1 - y0), int(x1 - x0)), radius)
    else:
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
    if 'stage' in edit:
        feather = edit.get('feather', 0)
        if (type(feather) not in (int, float) or not 0 <= feather <= radius):
            raise ValueError('羽化须在零到笔刷半径之间。')
        if feather == 0:
            return tile, (distance <= radius).astype(np.float32)
        edge = np.clip((radius - distance) / feather, 0, 1).astype(np.float32)
        return tile, edge * edge * (3 - 2 * edge)
    if edit['tool'] not in ('protect', 'mix'):
        return tile, (distance <= radius).astype(np.float32)
    # Flat within half the radius, easing to zero at the visible brush outline.
    edge = np.clip(2 - 2 * distance / radius, 0, 1).astype(np.float32)
    return tile, edge * edge * (3 - 2 * edge)


class _Geometry:
    """Manual geometry composites over the current automatic part selection."""

    def __init__(self, shape, default_edge):
        self.hard = np.zeros(shape, np.int8)
        self.alpha = self.include = None
        self.default_edge = default_edge
        self.edge_widths = None

    def coverage(self, labels, auto_labels, tile):
        auto = np.isin(labels, auto_labels)
        if self.alpha is None:
            marks = self.hard[tile]
            return (auto & (marks == 0)) | (marks > 0)
        return self.include[tile] + (1 - self.alpha[tile]) * auto

    def marked(self, tile):
        return self.hard[tile] != 0 if self.alpha is None else self.alpha[tile] > 0

    def paint(self, tile, hits, direction, brush=None, *, edge_width=None):
        amount = brush[hits] if brush is not None else None
        if amount is not None and np.any(amount != 1) and self.alpha is None:
            self.alpha = (self.hard != 0).astype(np.float32)
            self.include = (self.hard > 0).astype(np.float32)
        if self.alpha is None:
            self.hard[tile][hits] = direction
        elif amount is None:
            self.alpha[tile][hits] = int(direction != 0)
            self.include[tile][hits] = int(direction > 0)
        else:
            inverse = 1 - amount
            self.alpha[tile][hits] = self.alpha[tile][hits] * inverse + amount * (direction != 0)
            self.include[tile][hits] = self.include[tile][hits] * inverse + amount * (direction > 0)
        if edge_width is not None and edge_width < self.default_edge and hits.any():
            if self.edge_widths is None:
                self.edge_widths = np.full(self.hard.shape, self.default_edge, np.float32)
            # Adding a wider stroke over a thin one cannot weaken its core.
            target = np.minimum(self.edge_widths[tile][hits], edge_width)
        elif direction <= 0 and self.edge_widths is not None:
            target = self.default_edge
        else:
            return
        widths = self.edge_widths[tile]
        widths[hits] = target if amount is None else (1 - amount) * widths[hits] + amount * target

    def reset(self):
        self.hard.fill(0)
        self.alpha = self.include = None
        self.edge_widths = None


def _plain_include_brush(edit):
    return (isinstance(edit, dict) and edit.get('tool') == 'include'
            and edit.get('stage') in (None, 'range')
            and edit.get('shape', 'brush') == 'brush' and not edit.get('allow_protected', False))


def _resolution_module(edit, previous):
    module = edit.get('module')
    if not _plain_include_brush(previous):
        raise ValueError('待确认处理必须紧接普通补选笔画，并指定有效部位或仅加入处理范围。')
    if previous.get('stage') == 'range':
        if edit.get('stage') != 'range' or set(edit) - {'stage', 'tool', 'protected_parts'}:
            raise ValueError('处理范围只支持加入这笔待确认像素。更改分类请在部位分区操作。')
    elif set(edit) - {'tool', 'module'} or ('module' in edit and (not isinstance(module, str) or not MODULES.get(module))):
        raise ValueError('请选择有效的部位或仅加入处理范围。')
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


def _ordered_edits(edits, cursor):
    # Old records retain their exact order and form boundaries. In the two-step
    # editor, replay downstream strokes against the latest upstream classification.
    prefix = enumerate(edits[:cursor], start=1)
    for staged, records in groupby(prefix, lambda item: isinstance(item[1], dict) and 'stage' in item[1]):
        if staged:
            yield from sorted(records, key=lambda item: item[1]['stage'] != 'parts')
        else:
            yield from records


def render_selection(editor, settings, edits, cursor, *, correction=None, object_mask=None, weight_cache=None):
    """Replay only the active prefix of a document's reversible edit history."""
    if weight_cache is not None and weight_cache.editor is not editor:
        raise ValueError('Weight cache belongs to a different editor')
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
    scale = min(labels.shape) / 2048
    geometry = _Geometry(labels.shape, 16 * scale)
    protection = np.zeros(labels.shape, np.float32)
    kinds = {(edit.get('stage'), edit.get('tool')) for edit in edits[:cursor] if isinstance(edit, dict)}
    mix_protection = None
    if ('range', 'protect') in kinds and any(tool == 'mix' for _, tool in kinds):
        # Legacy absolute mixes intentionally ignored legacy soft protection.
        # Only a mixed old/new history needs a second protection channel.
        mix_protection = (np.zeros_like(protection) if correction is not None or (None, 'protect') in kinds
                          else protection)
    unlocked = mix_alpha = mix_value = None
    range_unlocked = None
    if correction is not None:
        if correction.get('sam_regions'):
            raise ValueError('Workbench import requires binary corrections without SAM prompts')
        marked_objects = correction['object_add_mask'] & ~transparent
        objects = (objects & ~correction['object_remove_mask']) | marked_objects
        for name, mask in correction['module_overrides'].items():
            hits = mask & ~transparent & ~objects
            labels[hits] = MODULES[name]
            unknown[hits] = False
        geometry.paint(..., correction['allow_mask'] & ~transparent & ~objects & ~unknown, 1)
        protection[correction['protect_mask'] & ~transparent] = 1

    edit_feedback = None
    skipped_by_brush = None
    staged = False
    for index, edit in _ordered_edits(edits, cursor):
        if not isinstance(edit, dict):
            raise ValueError('Invalid edit')
        tool = edit.get('tool')
        stage = edit.get('stage')
        staged |= stage is not None
        if 'stage' in edit and (stage not in STAGE_TOOLS or tool not in STAGE_TOOLS[stage]):
            raise ValueError('此工具不适用于当前编辑阶段。')
        protected_range = edit.get('protected_parts') if stage == 'range' else None
        if tool == 'resolve_unknown':
            module = _resolution_module(edit, edits[index - 2] if index > 1 else None)
            tile, hits, brush, edge_width = skipped_by_brush
            if protected_range:
                hits = hits & ~_locked_part_mask(labels[tile], objects[tile], unknown[tile], protected_range)
            if stage == 'range' and hits.any():
                if range_unlocked is None:
                    range_unlocked = np.zeros(labels.shape, bool)
                range_unlocked[tile][hits] = True
                geometry.paint(tile, hits, 1, brush, edge_width=edge_width)
            elif module is not None:
                labels[tile][hits] = MODULES[module]
                unknown[tile][hits] = False
            elif hits.any():
                if unlocked is None:
                    unlocked = np.zeros(labels.shape, bool)
                unlocked[tile][hits] = True
                geometry.paint(tile, hits, 1)
            continue
        if tool == 'sam_object':
            if object_mask is None:
                raise ValueError('SAM object edits require a saved mask')
            token = edit['token']
            outline = (object_mask(token) if weight_cache is None else
                       weight_cache.saved_coverage(token, object_mask, full=True)[1])
            hits = outline & ~transparent
            marked_objects |= hits
            objects |= hits
            unknown[hits] = False
            labels[hits] = 0
            if unlocked is not None:
                unlocked[hits] = False
            continue
        if tool == 'reset':
            if protected_range:
                locked = _locked_part_mask(labels, objects, unknown, protected_range)
                geometry.paint(..., ~locked, 0)
                if unlocked is not None:
                    unlocked &= locked
                if range_unlocked is not None:
                    range_unlocked &= locked
            else:
                geometry.reset()
                unlocked = range_unlocked = None
            continue
        if tool == 'invert':
            blocked = objects | unknown
            if unlocked is not None:
                blocked &= ~unlocked
            if range_unlocked is not None:
                blocked &= ~range_unlocked
            blocked |= transparent | (protection == 1)
            if protected_range:
                blocked |= _locked_part_mask(labels, objects, unknown, protected_range)
            coverage = geometry.coverage(labels, auto_labels, ...)
            if stage == 'range' and geometry.include is not None:
                geometry.alpha[~blocked] = 1
                geometry.include[~blocked] = 1 - coverage[~blocked]
            else:
                selected = (coverage > 0) & ~blocked
                geometry.paint(..., ~blocked & ~selected, 1)
                geometry.paint(..., selected, -1)
            if geometry.edge_widths is not None:
                geometry.edge_widths[~blocked] = geometry.default_edge
            continue
        if tool not in ('include', 'exclude', 'protect', 'mix', 'erase', 'override', *PART_ASSIGNMENT_TOOLS):
            raise ValueError('Unknown edit tool')
        if edit.get('shape') == 'sam' and stage:
            if object_mask is None:
                raise ValueError('SAM2 选区缺少已保存的蒙版。')
            token = edit['token']
            if weight_cache is None:
                tile, brush = _outline_coverage(object_mask(token))
            else:
                tile, brush = weight_cache.saved_coverage(token, object_mask)
        elif weight_cache is not None:
            tile, brush = weight_cache.stroke_coverage(edit)
        else:
            tile, brush = _rasterize(edit, labels.shape)
        hits = (brush > 0) & ~transparent[tile]
        if stage == 'parts':
            # A semantic label is discrete. Use the midpoint of a feathered edge.
            hits &= brush >= .5
            skipped = 0
            target = None
            if tool == 'remove_part':
                source = edit.get('module')
                if not isinstance(source, str) or source not in PART_CODES or source == 'unknown':
                    raise ValueError('请选择要移出的部位。')
                hits &= part_ids(labels[tile], objects[tile], unknown[tile]) == PART_CODES[source]
                target = 'unknown'
            elif tool == 'classify':
                target = edit.get('module')
                if target not in PART_CODES:
                    raise ValueError('请选择有效的分区类别。')
                protected = edit.get('protected_parts')
                if protected:
                    source_parts = part_ids(labels[tile], objects[tile], unknown[tile])
                    locked = np.isin(source_parts, [PART_CODES[part] for part in protected])
                    if index == cursor:
                        skipped = int(np.count_nonzero(hits & locked & (source_parts != PART_CODES[target])))
                    hits &= ~locked
            if unlocked is not None:
                # Preserve an earlier range override separately when a legacy
                # pixel gains an explicit, independently editable semantic label.
                transfer = unlocked[tile] & hits
                if transfer.any():
                    if range_unlocked is None:
                        range_unlocked = np.zeros(labels.shape, bool)
                    range_unlocked[tile][transfer] = True
                    unlocked[tile][transfer] = False
            if target is not None:
                label = 0 if target == 'object' else PART_CODES[target]
                changed = hits & ((labels[tile] != label) | (objects[tile] != (target == 'object'))
                                  | (unknown[tile] != (target == 'unknown')))
                labels[tile][hits] = label
                objects[tile][hits] = marked_objects[tile][hits] = target == 'object'
                unknown[tile][hits] = target == 'unknown'
            else:
                changed = hits & ((labels[tile] != editor.labels[tile]) | (objects[tile] != editor.objects[tile])
                                  | (unknown[tile] != editor.unknown[tile]) | marked_objects[tile])
                labels[tile][hits] = editor.labels[tile][hits]
                objects[tile][hits] = editor.objects[tile][hits]
                unknown[tile][hits] = editor.unknown[tile][hits]
                marked_objects[tile][hits] = False
            if index == cursor:
                count = int(changed.sum())
                edit_feedback = {'tool': tool, 'changed_pixels': count,
                                 'reason': 'changed' if count else 'protected' if skipped else 'no_marks'}
                if skipped:
                    edit_feedback['skipped_protected_pixels'] = skipped
            continue
        skipped_locked = 0
        if protected_range:
            locked = _locked_part_mask(labels[tile], objects[tile], unknown[tile], protected_range)
            if index == cursor:
                skipped_locked = int(np.count_nonzero(hits & locked))
            hits &= ~locked
        amount = brush if stage == 'range' else None
        if tool == 'erase':
            if index == cursor:
                manual = geometry.marked(tile) | (protection[tile] > 0)
                if stage is None:
                    manual |= ((labels[tile] != editor.labels[tile]) | (objects[tile] != editor.objects[tile])
                               | (unknown[tile] != editor.unknown[tile]) | marked_objects[tile])
                if unlocked is not None:
                    manual |= unlocked[tile]
                if mix_alpha is not None:
                    manual |= mix_alpha[tile] > 0
                changed = int(np.count_nonzero(hits & manual))
                edit_feedback = {'tool': tool, 'changed_pixels': changed,
                                 'reason': 'changed' if changed else 'no_marks'}
            geometry.paint(tile, hits, 0, amount)
            inverse = 0 if amount is None else 1 - amount[hits]
            protection[tile][hits] *= inverse
            if mix_protection is not None and mix_protection is not protection:
                mix_protection[tile][hits] *= inverse
            full = hits if amount is None else hits & (amount == 1)
            if stage is None:
                labels[tile][hits] = editor.labels[tile][hits]
                objects[tile][hits] = editor.objects[tile][hits]
                unknown[tile][hits] = editor.unknown[tile][hits]
                marked_objects[tile][hits] = False
            if unlocked is not None:
                unlocked[tile][full] = False
            if range_unlocked is not None:
                range_unlocked[tile][full] = False
            if mix_alpha is not None:
                mix_alpha[tile][hits] *= inverse
                mix_value[tile][hits] *= inverse
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
                if range_unlocked is not None:
                    blocked &= ~range_unlocked[tile]
                blocked |= ~hits | (protection[tile] == 1)
                before = geometry.coverage(labels[tile], auto_labels, tile)
            geometry.paint(tile, hits, -1, amount)
            if index == cursor:
                after = geometry.coverage(labels[tile], auto_labels, tile)
                reduced = (after < before) & ~blocked
                changed = int(np.count_nonzero(reduced))
                edit_feedback = {'tool': tool, 'changed_pixels': changed,
                                 'reason': 'changed' if changed else 'already_excluded'}
                partial = int(np.count_nonzero(reduced & (after > 0)))
                if edit.get('feather', 0) > 0:
                    edit_feedback['partially_reduced_pixels'] = partial
        elif tool == 'include':
            allow_protected = edit.get('allow_protected', False)
            if type(allow_protected) is not bool:
                raise ValueError('Explicit model-veto override must be boolean')
            model_blocked = objects[tile] | unknown[tile]
            if unlocked is not None:
                model_blocked &= ~unlocked[tile]
            if range_unlocked is not None:
                model_blocked &= ~range_unlocked[tile]
            hits &= protection[tile] < 1
            edge_width = edit.get('radius', 24) / 3 if edit.get('adaptive_edge') else None
            skipped = None
            if (_plain_include_brush(edit) and (index == cursor or
                    (isinstance(edits[index], dict) and edits[index].get('tool') == 'resolve_unknown'))):
                skipped = hits & unknown[tile] & ~objects[tile]
                if unlocked is not None:
                    skipped &= ~unlocked[tile]
                if range_unlocked is not None:
                    skipped &= ~range_unlocked[tile]
                skipped_by_brush = (tile, skipped, brush, edge_width)
            already_writable = hits & ~model_blocked
            if allow_protected:
                hits &= ~marked_objects[tile]
                new_unlock = hits & model_blocked
                if new_unlock.any():
                    if stage == 'range':
                        if range_unlocked is None:
                            range_unlocked = np.zeros(labels.shape, bool)
                        range_unlocked[tile][new_unlock] = True
                    else:
                        if unlocked is None:
                            unlocked = np.zeros(labels.shape, bool)
                        unlocked[tile][new_unlock] = True
            else:
                hits = already_writable
            if index == cursor:
                selected_before = (geometry.coverage(labels[tile], auto_labels, tile) >= 1) & already_writable
                writable = int(np.count_nonzero(hits))
                changed = writable - int(np.count_nonzero(selected_before))
                edit_feedback = {'tool': tool, 'changed_pixels': changed,
                                 'reason': 'changed' if changed else 'already_selected' if writable else 'protected'}
                skipped_count = int(np.count_nonzero(skipped)) if skipped is not None else 0
                if skipped_count:
                    edit_feedback['skipped_unknown_pixels'] = skipped_count
            geometry.paint(tile, hits, 1, amount, edge_width=edge_width)
        elif tool == 'mix':
            original = validate_unit_interval(edit.get('strength', .9), 'original fraction')
            blocked = objects[tile] | unknown[tile]
            if unlocked is not None:
                blocked &= ~unlocked[tile]
            if range_unlocked is not None:
                blocked &= ~range_unlocked[tile]
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
            amount = brush[hits] * np.float32(opacity)
            marks[hits] = np.maximum(marks[hits], amount)
            if stage == 'range' and mix_protection is not None and mix_protection is not protection:
                marks = mix_protection[tile]
                marks[hits] = np.maximum(marks[hits], amount)
        if skipped_locked:
            if edit_feedback is None:
                edit_feedback = {'tool': tool}
            edit_feedback['skipped_protected_pixels'] = skipped_locked

    unclassified = legacy_semantic_masks = None
    if unlocked is not None:
        # A legacy range override cleared these flags for rendering.
        legacy_semantic_masks = (objects, unknown)
        unclassified = unknown & unlocked
        objects = objects & ~unlocked
        unknown = unknown & ~unlocked
    blocked = objects | unknown
    if range_unlocked is not None:
        blocked &= ~(range_unlocked & ~marked_objects)
    blocked |= transparent | (protection == 1)
    fraction = geometry.coverage(labels, auto_labels, ...)
    if range_unlocked is not None and geometry.include is not None:
        lifted = range_unlocked & (objects | unknown)
        fraction[lifted] = geometry.include[lifted]
    selected = (fraction > 0) & ~blocked
    edge_widths = geometry.edge_widths
    if weight_cache is None and edge_widths is None:
        weight = build_weight(selected, blocked, editor.rgb, feather_radius=16 * scale,
                              structure_radius=5 * scale, structure_protection=structure)
    else:
        cache = weight_cache if weight_cache is not None else WorkbenchWeightCache(editor)
        weight = cache.weight(selected, structure, edge_widths)
    weight *= np.float32(strength)
    weight *= _region_factor(labels, selected, active, strengths, scale)
    weight *= 1 - protection
    weight[weight < 1 / 255] = 0
    has_local_mix = False
    if mix_alpha is not None:
        mixed = selected & (mix_alpha > 0)
        has_local_mix = bool(mixed.any())
        target = mix_value[mixed]
        if mix_protection is not None:
            target *= 1 - mix_protection[mixed]
        weight[mixed] = (1 - mix_alpha[mixed]) * weight[mixed] + target
        # Floating-point compositing can overshoot by a few ulps.
        np.clip(weight, 0, 1, out=weight)
    fractional = geometry.alpha is not None and bool(((fraction > 0) & (fraction < 1) & selected).any())
    if fractional:
        weight *= fraction
    adaptive_range = edge_widths is not None and bool(((edge_widths < geometry.default_edge) & selected).any())
    return Selection(weight, labels, objects, unknown, protection, selected, edit_feedback,
                     has_local_mix, unclassified is not None and bool((unclassified & selected).any()), fractional,
                     not staged, legacy_semantic_masks, adaptive_range)


def binary_corrections(editor, selection, settings):
    """Translate exactly representable workbench masks to CLI version 2."""
    if selection.fractional_range:
        raise ValueError('CLI 二值校正无法表示带羽化的处理范围。')
    if selection.adaptive_range:
        raise ValueError('CLI 二值校正无法表示随笔径调整的边缘过渡。')
    if np.any(selection.selected & (selection.objects | selection.unknown)):
        raise ValueError('CLI 二值校正无法在保留外物或待确认标签的同时处理这些像素。')
    if np.any((selection.unknown & ~editor.unknown) | ((selection.labels == 0) & ~selection.objects
              & ((editor.labels != 0) | editor.unknown))):
        raise ValueError('CLI 二值校正无法表示此次背景或待确认分类。')
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
