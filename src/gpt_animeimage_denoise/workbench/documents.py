"""Persist workbench drafts and publish only the newest completed image revision."""
from base64 import b64encode
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from math import inf, isfinite
from numbers import Real
from pathlib import Path, PureWindowsPath
import re
from threading import Condition, Event, Lock, Thread, Timer
from time import time
import traceback
from types import SimpleNamespace
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..corrections import MASK_KEYS, load_edits
from ..denoisers import resolve_denoiser
from ..editor import DEFAULT_MODULES, MODULES
from ..model_paths import part_models_present
from ..pipeline import _digest, _json, _read_source, _sha
from .storage import CaseCache, Preferences, StorageBusy, export_png, write_atomic as _write_atomic


DEFAULTS = {'method': 'l0', 'l0': .0005, 'downscale': .6, 'strength': 1,
            'restoration_gain': .25, 'sigma': 12, 'structure_protection': .2,
            'modules': list(DEFAULT_MODULES), 'part_strengths': {}}
WORKBENCH_METHODS = {'l0': '纯 L0', 'esrgan': 'ESRGAN', 'esrgan-l0': 'ESRGAN + L0', 'apisr': 'APISR'}
# Recognize old drafts so they can resume without the retired model.
_RETIRED_METHODS = {'cugan', 'cugan-l0', 'cugan60', 'cugan60_l0'}
_FORMATS = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp', 'BMP': '.bmp', 'TIFF': '.tif'}
_DOCUMENT_ID = re.compile(r'[0-9a-f]{64}')
_CORRECTION_TOKEN = re.compile(r'correction-[0-9a-f]{32}')
_ANALYSIS_VERSION = 5
_RANGE_EDGE_VERSION = 1
_SUSPECTED_ASSET = f'suspected-objects-{_ANALYSIS_VERSION}.png'
BACKGROUND_TINT = (56, 151, 188)
_BACKGROUND_OVERLAY_OPACITY = 100
_BACKGROUND_HIGHLIGHT_OPACITY = 64
PART_PALETTE = {'background': BACKGROUND_TINT, 'skin': (219, 137, 108), 'face': (225, 180, 80),
                'hair': (101, 135, 223), 'eyes': (54, 165, 160), 'mouth': (217, 120, 164), 'outfit': (139, 120, 199)}
_OBJECT_COLOR = (224, 136, 68)
_WEIGHT_STOPS = ((100, 226, 250), (111, 119, 247), (241, 72, 169))
_LEGACY_WEIGHT_PALETTE = bytes((79, 159, 232)) * 256
_MAX_RETAINED_RESULT_BYTES = 32 * 1024 * 1024
_PRELOADED_SKY_SECONDS = 60
_SELECTION_ASSETS = ('parts', 'weight', 'weight_overlay', 'unknown', 'suspected_objects', 'labels')
_CHANGE_KINDS = ('added', 'removed', 'stronger', 'weaker', 'reclassified')
_CHANGE_STAGES = {None: _CHANGE_KINDS, 'parts': ('reclassified',), 'range': _CHANGE_KINDS[:-1]}
_CHANGE_COLORS = np.array([(0, 0, 0, 0), (56, 189, 248, 180), (244, 119, 118, 180),
                           (81, 223, 156, 180), (248, 191, 86, 180), (190, 154, 255, 180)], np.uint8)


class RevisionConflict(ValueError):
    """An update or export refers to a revision that is no longer current."""


def _require_parts(state):
    if state['status'] == 'awaiting_profile':
        raise ValueError('请先选择处理模式。')
    if state.get('profile') == 'whole':
        raise ValueError('请切换到「按部位精修」后修改选区。')


@dataclass
class _ResultImage:
    rgb: np.ndarray
    png: bytes | None = None


@dataclass(frozen=True)
class _ResultJob:
    revision: int
    image: _ResultImage
    alpha: np.ndarray | None
    icc: bytes | None
    candidate_png: bytes
    publication: dict


def _selection(editor, settings, edits, cursor, *, correction=None, object_mask=None, weight_cache=None):
    from .masks import render_selection

    return render_selection(editor, settings, edits, cursor, correction=correction,
                            object_mask=object_mask, weight_cache=weight_cache)


def _png(rgb, alpha, icc):
    image = Image.fromarray(rgb)
    if alpha is not None:
        image.putalpha(Image.fromarray(alpha))
    stream = BytesIO()
    image.save(stream, 'PNG', compress_level=1, **({'icc_profile': icc} if icc else {}))
    return stream.getvalue()


def _indexed_png(image, palette, opacity):
    image.putpalette(bytes(palette))
    stream = BytesIO()
    image.save(stream, 'PNG', transparency=bytes(opacity), compress_level=1)
    return stream.getvalue()


def _weight_ramp(levels):
    colors = np.column_stack([np.interp(levels, (0, .5, 1), channel) for channel in zip(*_WEIGHT_STOPS)])
    opacity = np.rint(np.sqrt(levels) * (.72 * 255)).astype(np.uint8)
    return np.rint(colors).astype(np.uint8).tobytes(), opacity.tobytes()


_WEIGHT_PALETTE, _WEIGHT_OPACITY = _weight_ramp(np.arange(256) / 255)
# The old indexed image stores up to 55% opacity, rather than the full weight.
_RECOLORED_WEIGHT_PALETTE, _RECOLORED_WEIGHT_OPACITY = _weight_ramp(np.minimum(np.arange(256) / 140, 1))


def _recolor_legacy_weight(data):
    with Image.open(BytesIO(data)) as image:
        if image.mode == 'P' and image.palette and image.palette.getdata()[1] == _LEGACY_WEIGHT_PALETTE:
            return _indexed_png(image, _RECOLORED_WEIGHT_PALETTE, _RECOLORED_WEIGHT_OPACITY)
    return data


def _restoration_key(settings):
    return (settings['method'], settings['l0'], settings['downscale'],
            settings['sigma'], settings['restoration_gain'])


def _restore_candidate(rgb, candidate, settings):
    from ..filters import gradient_preserving_candidate

    sigma = settings['sigma'] * (min(rgb.shape[:2]) / 2048)
    return gradient_preserving_candidate(rgb, candidate, sigma,
                                         restoration_gain=settings['restoration_gain'])


def _overlay(rgb, tint, opacity):
    # Display overlays use byte opacity. Mixing one channel at a time avoids
    # full RGB float temporaries while the actual result still uses float W.
    alpha = np.rint(np.asarray(opacity, dtype=np.float32) * 255).astype(np.uint16)
    inverse = 255 - alpha
    result = np.empty_like(rgb)
    for channel in range(3):
        result[..., channel] = (rgb[..., channel] * inverse + tint[..., channel] * alpha + 127) // 255
    return result


def _paint_unknown_hatch(overlay, visible, alpha=None):
    yy, xx = np.nonzero(visible)
    stripe_width = max(4, round(min(visible.shape) / 192))
    value = np.where(((xx + yy) // stripe_width) % 2 == 0, 245, 22)
    overlay[yy, xx, :3] = value[:, None]
    overlay[yy, xx, 3] = 158 if alpha is None else (alpha[yy, xx].astype(np.uint16) * 158 + 127) // 255


def _unknown_overlay(unknown, alpha):
    visible = unknown if alpha is None else unknown & (alpha != 0)
    if not visible.any():
        return None
    overlay = np.zeros((*unknown.shape, 4), np.uint8)
    _paint_unknown_hatch(overlay, visible, alpha)
    return _png(overlay, None, None)


def _changes_overlay(editor, auto_weight, mask, *, include_parts=True):
    weight = mask.weight
    classes = np.zeros(weight.shape, np.uint8)
    stronger = weight > auto_weight
    weaker = weight < auto_weight
    classes[stronger] = 3
    classes[stronger & (auto_weight == 0)] = 1
    classes[weaker] = 4
    classes[weaker & (weight == 0)] = 2
    if include_parts:
        reclassified = (mask.labels != editor.labels) | (mask.objects != editor.objects) | (mask.unknown != editor.unknown)
        classes[(classes == 0) & reclassified] = 5
    return _encode_changes(editor, classes)


def _part_changes_overlay(editor, labels, objects, unknown):
    from .masks import PART_CODES, part_ids

    parts = part_ids(labels, objects, unknown)
    changed = parts != part_ids(editor.labels, editor.objects, editor.unknown)
    if editor.alpha is not None:
        changed &= editor.alpha != 0
    sizes = np.bincount(parts[changed], minlength=256)
    counts = {'reclassified': int(sizes.sum())}
    destinations = [part for part, code in PART_CODES.items() if sizes[code]]
    if not destinations:
        return counts, None, destinations

    colors = np.zeros((256, 4), np.uint8)
    for part, color in {**PART_PALETTE, 'object': _OBJECT_COLOR}.items():
        colors[PART_CODES[part]] = (*color, 180)
    parts[~changed] = 254  # Transparent palette entry.
    overlay = colors[parts]
    if 'unknown' in destinations:
        _paint_unknown_hatch(overlay, parts == PART_CODES['unknown'])
    return counts, _change_image(overlay, editor.alpha), destinations


def _encode_changes(editor, classes):
    if editor.alpha is not None:
        classes[editor.alpha == 0] = 0
    sizes = np.bincount(classes.ravel(), minlength=len(_CHANGE_COLORS))
    counts = dict(zip(_CHANGE_KINDS, map(int, sizes[1:])))
    if not sizes[1:].any():
        return counts, None
    return counts, _change_image(_CHANGE_COLORS[classes], editor.alpha)


def _change_image(overlay, alpha):
    if alpha is not None:
        overlay[..., 3] = (overlay[..., 3].astype(np.uint16) * alpha + 127) // 255
    return 'data:image/png;base64,' + b64encode(_png(overlay, None, None)).decode('ascii')


def _automatic_weight(editor, settings, lines):
    from .masks import WorkbenchWeightCache

    cache = WorkbenchWeightCache(editor)
    cache.lines = lines
    return _selection(editor, settings, [], 0, weight_cache=cache).weight


def _selection_images(session, mask, *, previous=None):
    editor = session.editor
    old_mask, old_images = previous or (None, {})
    images = {}
    same_unknown = old_mask is not None and np.array_equal(old_mask.unknown, mask.unknown)
    same_parts = same_unknown and all(np.array_equal(getattr(old_mask, key), getattr(mask, key))
                                      for key in ('labels', 'objects'))
    if same_parts:
        images['labels.png'] = old_images['labels.png']
    else:
        from .masks import part_ids

        labels = part_ids(mask.labels, mask.objects, mask.unknown)
        if editor.alpha is not None:
            labels[editor.alpha == 0] = 254
        images['labels.png'] = _png(labels, None, None)
    if mask.legacy_parts_overlay and mask.soft_protection.any():
        part_marks = mask.soft_protection
        if same_parts and 'parts.png' in old_images and np.array_equal(old_mask.soft_protection, part_marks):
            images['parts.png'] = old_images['parts.png']
        else:
            labels = mask.labels
            colors = np.zeros((256, 3), np.uint8)
            for module, color in PART_PALETTE.items():
                colors[MODULES[module]] = color
            tint = colors[labels]
            tint[mask.objects] = _OBJECT_COLOR
            opacity = np.where(mask.objects, np.float32(.56),
                               np.where(mask.unknown | (labels == 0) | (labels == 255),
                                        np.float32(0), np.float32(.42)))
            parts = _overlay(editor.rgb, tint, opacity)
            protected = part_marks > 0
            parts[protected] = _overlay(parts[protected], np.array([220, 76, 106], np.uint8),
                                        part_marks[protected] * .55)
            images['parts.png'] = _png(parts, editor.alpha, session.icc)
    if same_unknown:
        unknown = old_images.get('unknown.png')
    else:
        unknown = _unknown_overlay(mask.unknown, editor.alpha)
    if unknown is not None:
        images['unknown.png'] = unknown
    if old_mask is not None and np.array_equal(old_mask.weight, mask.weight):
        images['weight_overlay.png'] = old_images['weight_overlay.png']
    else:
        levels = np.rint(mask.weight * np.float32(255)).astype(np.uint8)
        images['weight_overlay.png'] = _indexed_png(Image.fromarray(levels), _WEIGHT_PALETTE, _WEIGHT_OPACITY)
    return images


def _setting_number(value, name, *, positive=False, maximum=inf):
    """Validate saved settings without importing the image filters."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f'{name} 必须是数值。')
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f'{name} 超出允许范围。') from None
    if not isfinite(value) or not 0 <= value <= maximum or (positive and value == 0):
        raise ValueError(f'{name} 超出允许范围。')
    return value


def _settings(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('工作台设置无效。')
    result = {**DEFAULTS, **value}
    modules = result['modules']
    if (not isinstance(modules, list) or any(not isinstance(name, str) or name not in MODULES for name in modules)
            or len(set(modules)) != len(modules)):
        raise ValueError('部位选择无效。')
    part = result['part_strengths']
    if not isinstance(part, dict) or set(part) - set(MODULES):
        raise ValueError('部位强度无效。')
    result['part_strengths'] = {name: _setting_number(value, name, maximum=1) for name, value in part.items()}
    if isinstance(result['method'], str) and result['method'] in _RETIRED_METHODS:
        raise ValueError('项目已移除 CUGAN，请选择其他降噪方法。')
    if not isinstance(result['method'], str):
        raise ValueError('请选择工作台支持的降噪方法。')
    result['method'], result['l0'] = resolve_denoiser(result['method'], _setting_number(result['l0'], 'L0', positive=True))
    if result['method'] not in WORKBENCH_METHODS:
        raise ValueError('请选择工作台支持的降噪方法。')
    for name in ('strength', 'structure_protection', 'restoration_gain', 'downscale'):
        result[name] = _setting_number(result[name], name, maximum=1)
    result['sigma'] = _setting_number(result['sigma'], 'sigma', positive=True)
    if not result['downscale'] or result['sigma'] > 256:
        raise ValueError('缩图比例须大于零，sigma 不能超过 256。')
    return result


def _stage_tool(edit):
    from ..filters import validate_unit_interval
    from .masks import PART_ASSIGNMENT_TOOLS, PART_CODES, STAGE_TOOLS

    stage, tool = edit.get('stage'), edit.get('tool')
    if not isinstance(stage, str) or stage not in STAGE_TOOLS or tool not in STAGE_TOOLS[stage]:
        raise ValueError('此工具不适用于当前编辑阶段。')
    module = edit.get('module')
    if tool == 'remove_part' and (not isinstance(module, str) or module not in PART_CODES or module == 'unknown'):
        raise ValueError('请选择要移出的部位。')
    if tool == 'classify' and (not isinstance(module, str) or module not in PART_CODES):
        raise ValueError('请选择有效的分区类别。')
    if 'module' in edit and tool not in PART_ASSIGNMENT_TOOLS:
        raise ValueError('请在归入或移出操作中指定部位。')
    if 'protected_parts' in edit:
        parts = edit['protected_parts']
        if ((stage != 'range' and tool != 'classify') or not isinstance(parts, list)
                or any(not isinstance(part, str) or part not in PART_CODES for part in parts)
                or len(parts) != len(set(parts))):
            raise ValueError('请选择有效且互不重复的保护分区。')
    if 'allow_protected' in edit and (tool != 'include' or type(edit['allow_protected']) is not bool):
        raise ValueError('只有补选可以允许使用自动保护的区域。')
    if 'strength' in edit:
        if tool not in ('protect', 'mix'):
            raise ValueError('此工具不能指定局部强度。')
        validate_unit_interval(edit['strength'], '局部强度')


def _history(edits, cursor, width, height, stored_edits):
    from ..filters import _radius, validate_unit_interval
    from .masks import PART_ASSIGNMENT_TOOLS

    if (not isinstance(edits, list) or len(edits) > 2000 or type(cursor) is not int
            or not 0 <= cursor <= len(edits)):
        raise ValueError('编辑记录无效或已超过上限。')
    count = 0
    seen_tokens = set()
    for index, edit in enumerate(edits):
        if isinstance(edit, dict) and ('token' in edit or edit.get('tool') in ('import', 'sam_object')
                                       or edit.get('shape') == 'sam'):
            token = edit.get('token')
            if not isinstance(token, str) or stored_edits.get(token) != edit or token in seen_tokens:
                raise ValueError('蒙版编辑记录必须来自当前文档。')
            seen_tokens.add(token)
            continue
        fields = {'tool', 'shape', 'points', 'radius', 'strength', 'module', 'allow_protected'}
        if isinstance(edit, dict) and 'stage' in edit:
            _stage_tool(edit)
            fields.update(('stage', 'feather', 'protected_parts', 'geometry', 'adaptive_edge'))
        if not isinstance(edit, dict) or set(edit) - fields:
            raise ValueError('编辑记录无效。')
        tool = edit.get('tool')
        if tool == 'resolve_unknown':
            from .masks import _resolution_module

            _resolution_module(edit, edits[index - 1] if index else None)
            continue
        if 'allow_protected' in edit and (tool != 'include' or type(edit['allow_protected']) is not bool):
            raise ValueError('只有补选可明确允许使用自动保护的区域。')
        if tool in ('invert', 'reset'):
            if 'stage' in edit and set(edit) - {'stage', 'tool', 'protected_parts'}:
                raise ValueError('整体范围操作不能指定笔画或选区。')
            continue
        shape = edit.get('shape', 'brush')
        points = edit.get('points')
        if (tool not in ('include', 'exclude', 'protect', 'mix', 'erase', 'override', *PART_ASSIGNMENT_TOOLS)
                or (tool in PART_ASSIGNMENT_TOOLS and edit.get('stage') != 'parts') or shape not in ('brush', 'polygon')
                or not isinstance(points, list) or len(points) < (3 if shape == 'polygon' else 1)):
            raise ValueError('笔画或套索记录无效。')
        if 'geometry' in edit and (shape != 'brush' or edit['geometry'] != 'continuous'):
            raise ValueError('画笔轨迹格式无效。')
        if 'adaptive_edge' in edit and (edit.get('stage') != 'range' or tool != 'include' or shape != 'brush'
                                        or type(edit['adaptive_edge']) is not bool):
            raise ValueError('只有补选画笔可以按笔径调整边缘过渡。')
        count += len(points)
        if count > 50000:
            raise ValueError('笔画采样点超过上限。')
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError('笔画坐标无效。')
            if _radius(point[0], 'x') >= width or _radius(point[1], 'y') >= height:
                raise ValueError('笔画坐标超出原图。')
        if shape == 'brush':
            radius = edit.get('radius', 24)
            if type(radius) is not int or not 1 <= radius <= 256 or _radius(edit.get('feather', 0), '羽化') > radius:
                raise ValueError('笔刷半径须为 1 至 256 的整数，羽化不能超过半径。')
        elif edit.get('feather', 0) != 0:
            raise ValueError('套索使用明确的轮廓边界。')
        if tool == 'protect':
            validate_unit_interval(edit.get('strength', .9), '保护强度')
        if tool == 'mix':
            validate_unit_interval(edit.get('strength', .9), '保留原图比例')
        if tool == 'override' and edit.get('module') not in set(MODULES) - {'background'}:
            raise ValueError('请选择纠正后的部位。')
    return deepcopy(edits)


def _safe_mask_filename(value):
    if (not isinstance(value, str) or not value or PureWindowsPath(value).name != value
            or value.casefold() in ('corrections.json', 'corrections.json.tmp')
            or PureWindowsPath(value).is_reserved() or value.endswith((' ', '.'))
            or any(c in '/\\:*?"<>|' or ord(c) < 32 for c in value)):
        raise ValueError('校正蒙版必须使用同目录文件名。')
    return value


def _active_edit_layers(state):
    active = state['edits'][:state['cursor']]
    token = state.get('_correction')
    for index in range(len(active) - 1, -1, -1):
        if active[index]['tool'] == 'import':
            token = active[index]['token']
            active = active[index + 1:]
            break
    if token is not None and (not isinstance(token, str) or not _CORRECTION_TOKEN.fullmatch(token)):
        raise ValueError('保存的校正记录无效。')
    return token, active


def _sam_points(points, width, height):
    if not isinstance(points, list) or len(points) > 32:
        raise ValueError('每种点选最多可添加 32 个点。')
    for point in points:
        if (not isinstance(point, list) or len(point) != 2 or any(type(axis) is not int for axis in point)
                or not (0 <= point[0] < width and 0 <= point[1] < height)):
            raise ValueError('点选坐标必须在原图范围内。')
    return {tuple(point) for point in points}


class Documents:
    """Source-bound drafts with one resident analysis and separate render and encoding workers."""

    def __init__(self, workspace, opener=None):
        self.workspace = Path(workspace).resolve()
        self.opener = opener
        self.preferences = Preferences(DEFAULTS, _settings)
        self.cache = CaseCache(self.workspace, self.preferences)
        self.root = self.workspace / 'documents'
        self._condition = Condition()
        self._inference = Lock()
        self._magnetic = Lock()
        self._magnetic_cache = None
        self._changes_lock = Lock()
        self._changes_cache = {}
        self._changes_baseline = None
        self._thumbnail_lock = Lock()
        self._thumbnails = OrderedDict()
        self._documents = {}
        self._pending = OrderedDict()
        self._pending_results = OrderedDict()
        self._prepare_request = None
        self._warmup_stop = Event()
        self._warmup_thread = None
        self._preloaded_sky = None
        self._preload_expiry = None
        self._object_previews = OrderedDict()
        self._closed = False
        self._drop_resident()
        self._restore()
        self._worker = Thread(target=self._run, name='gaid-workbench', daemon=True)
        self._encoder = Thread(target=self._run_encoder, name='gaid-png', daemon=True)
        self._worker.start()
        self._encoder.start()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._condition:
            self._closed = True
            self._warmup_stop.set()
            self._take_preloaded_sky()
            self._pending_results.clear()
            self._condition.notify_all()
        self._worker.join(timeout=2)
        self._encoder.join(timeout=2)
        if self._warmup_thread is not None:
            self._warmup_thread.join(timeout=2)

    def _restore(self):
        if not self.root.is_dir():
            return
        for folder in self.root.iterdir():
            draft = folder / 'draft.json'
            if not _DOCUMENT_ID.fullmatch(folder.name) or not draft.is_file():
                continue
            try:
                state = json.loads(draft.read_text(encoding='utf-8'))
                correction = state.get('_correction')
                if (state['id'] != folder.name or state['_schema'] != 1 or state['_suffix'] not in _FORMATS.values()
                        or (correction is not None and not _CORRECTION_TOKEN.fullmatch(correction))):
                    continue
                if state['settings']['method'] in _RETIRED_METHODS:
                    previous = state.get('b')
                    if not state.get('a') and previous and previous['method'] in _RETIRED_METHODS:
                        state['a'] = deepcopy(previous)
                    state['settings'] = {**state['settings'], 'method': DEFAULTS['method']}
                    state.update(revision=state['revision'] + 1, status='queued', phase='queued', error=None)
                    self._store(state)
                self._documents[folder.name] = state
                if state['status'] in ('queued', 'running'):
                    state.update(status='queued', phase='queued')
                    self._pending[folder.name] = state['revision']
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def _store(self, state):
        if state['status'] != 'error':
            state.pop('failed_phase', None)
        _json(self.root / state['id'] / 'draft.json', state)
        self._documents[state['id']] = state
        preview = self._object_previews.get(state['id'])
        if preview is not None and preview[0] != state['revision']:
            self._object_previews.pop(state['id'])

    def _queue(self, state):
        identifier = state['id']
        self._warmup_stop.set()
        self._pending[identifier] = state['revision']
        self._pending_results.pop(identifier, None)
        self._condition.notify_all()

    def _lookup(self, doc, *, upgrade=True):
        if not isinstance(doc, str) or not _DOCUMENT_ID.fullmatch(doc) or doc not in self._documents:
            raise KeyError('找不到此文档。')
        state = self._documents[doc]
        if upgrade and self._analysis_outdated(state):
            return self._restart_render(state, selection_changed=True, reset_source=True)
        if upgrade and self._range_edge_outdated(state):
            return self._restart_render(state, selection_changed=True)
        return state

    @staticmethod
    def _analysis_outdated(state):
        return (state.get('profile') != 'whole' and state['status'] == 'ready'
                and state.get('_analysis_version') != _ANALYSIS_VERSION)

    @staticmethod
    def _range_edge_outdated(state):
        return (state.get('profile') != 'whole' and state['status'] == 'ready'
                and state.get('_range_edge_version') != _RANGE_EDGE_VERSION
                and any(edit.get('adaptive_edge') for edit in state['edits'][:state['cursor']]))

    def _restart_render(self, state, *, selection_changed, reset_source=False, defer_profile=False):
        # Leave earlier previews available while the new revision is running.
        phase = 'awaiting_profile' if defer_profile else 'queued'
        updated = {**state, 'revision': state['revision'] + 1, 'status': phase,
                   'phase': phase, 'error': None,
                   'selection_revision': state['selection_revision'] + int(selection_changed)}
        if reset_source:
            updated['_source_epoch'] = state.get('_source_epoch', 0) + 1
        self._store(updated)
        if defer_profile:
            self._pending.pop(state['id'], None)
            self._pending_results.pop(state['id'], None)
            self._condition.notify_all()
        else:
            self._queue(updated)
        return updated

    @staticmethod
    def _public(state):
        return deepcopy({key: value for key, value in state.items() if not key.startswith('_')})

    def get(self, doc):
        with self._condition:
            return self._public(self._lookup(doc))

    def document_snapshot(self, doc, instance, known_etag):
        with self._condition:
            state = self._lookup(doc)
            # The revision already identifies the full settings and edit history.
            # Progress and pinned A can also change within a revision.
            metadata = {key: value for key, value in state.items()
                        if not key.startswith('_') and key not in ('settings', 'edits')}
            etag = '"' + _digest([instance, metadata]) + '"'
            return etag, None if etag == known_etag else self._public(state)

    def prepare_edit(self, doc, revision):
        with self._condition:
            state = self._lookup(doc)
            self._current(state, revision)
            if (self.opener is None and state.get('profile') != 'whole' and state['status'] == 'ready' and state.get('b')
                    and ((self._resident_id, self._resident_epoch) != (doc, state.get('_source_epoch', 0))
                         or self._selection_key != self._selection_cache_key(state))):
                self._warmup_stop.set()
                self._prepare_request = (doc, revision)
                self._condition.notify_all()

    def prepare_models(self):
        with self._condition:
            if (self._closed or self.opener is not None or self._warmup_thread is not None
                    or self._warmup_stop.is_set() or self._pending or self._prepare_request is not None
                    or self._resident is not None or self.preferences.read()['default_profile'] in ('whole', 'ask')
                    or not part_models_present()):
                return
            self._warmup_thread = Thread(target=self._warm_model_cache, name='gaid-model-warmup', daemon=True)
            self._warmup_thread.start()

    def _warm_model_cache(self):
        if not self._inference.acquire(blocking=False):
            return
        try:
            if self._warmup_stop.is_set():
                return
            from .session import WorkbenchSession

            prepared = WorkbenchSession.prewarm(self._warmup_stop)
            with self._condition:
                if prepared is not None and not self._closed:
                    self._preloaded_sky = prepared
                    expiry = Timer(_PRELOADED_SKY_SECONDS, self._take_preloaded_sky)
                    self._preload_expiry = expiry
                    expiry.daemon = True
                    expiry.start()
        except Exception:
            traceback.print_exc()  # An optional preload still leaves the regular render path available.
        finally:
            self._inference.release()

    def _take_preloaded_sky(self):
        with self._condition:
            prepared, self._preloaded_sky = self._preloaded_sky, None
            if self._preload_expiry is not None:
                self._preload_expiry.cancel()
                self._preload_expiry = None
            return prepared

    @contextmanager
    def maintenance(self, activity='管理缓存'):
        if not self._inference.acquire(blocking=False):
            raise StorageBusy(f'工作台正在处理图片，请等待空闲后再{activity}。')
        try:
            with self._condition:
                if any(state['status'] in ('queued', 'running') for state in self._documents.values()):
                    raise StorageBusy(f'工作台有待处理图片，请等待空闲后再{activity}。')
            with self.cache.processing(wait=False):
                yield
        finally:
            self._inference.release()

    def environment_diagnosis(self, *, whole, verify_models):
        from ..doctor import diagnose, format_report

        with self.maintenance('运行诊断'):
            report = diagnose(whole=whole, verify_models=verify_models)
        return {'report': report, 'text': format_report(report)}

    def change_preferences(self, patch):
        return self.preferences.change(patch, self.cache, self.maintenance)

    def clear_cache(self, *, mode='all'):
        with self.maintenance():
            return self.cache.clear(mode=mode)

    def cache_info(self):
        try:
            with self.maintenance(), self.cache.lock(wait=False):
                self.cache.switch(self.preferences.read()['cache_dir'])
                return self.cache.info()
        except StorageBusy:
            return self.cache.info()

    def _open_session(self, source, state, *, on_stage=None):
        with self.cache.using(state['id']) as root:
            if self.opener is not None:
                return self.opener(source, root)
            from .session import WorkbenchSession

            saved = self._saved_session(source, root, state)
            if saved is not None:
                return saved
            settings = state['settings']
            return WorkbenchSession.open(source, root, method=settings['method'], lam=settings['l0'],
                                         downscale=settings['downscale'], on_stage=on_stage,
                                         preloaded_sky=self._take_preloaded_sky())

    @staticmethod
    def _saved_session(source, root, state):
        if not state.get('b') or state.get('_analysis_version') != _ANALYSIS_VERSION:
            return None
        from .session import WorkbenchSession

        settings = state['settings']
        try:
            saved = WorkbenchSession.from_cache(source, root, state['id'], [state['width'], state['height']],
                                                method=settings['method'], lam=settings['l0'],
                                                downscale=settings['downscale'])
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
            return None
        if _sha(source) != state['id']:
            raise ValueError('本地原图副本已更改，请重新打开源图。')
        return saved

    def recent(self):
        with self._condition:
            states = sorted(self._documents.values(), key=lambda state: state['_updated'], reverse=True)
            return [{key: state[key] for key in ('id', 'name', 'width', 'height', 'status')}
                    for state in states[:20]]

    def thumbnail(self, doc, view='original', revision=None):
        if view != 'original':
            if view not in ('parts', 'weight', 'candidate', 'b'):
                raise KeyError('找不到此节点的缩略图。')
            key = (doc, view, revision)
            with self._thumbnail_lock:
                with self._condition:
                    state = self._published_b(doc, revision)
                    if key in self._thumbnails:
                        self._thumbnails.move_to_end(key)
                        return self._thumbnails[key]
                    overlay = state['b'].get('images', {}).get(view + '_overlay')
                if overlay:
                    data = self._composited_overlay(doc, view, revision, size=(320, 320))
                else:
                    data = self.asset(doc, view, revision=revision)
                    output = BytesIO()
                    with Image.open(BytesIO(data)) as image:
                        image.thumbnail((320, 320), Image.Resampling.LANCZOS)
                        image.save(output, 'PNG', icc_profile=image.info.get('icc_profile'))
                    data = output.getvalue()
                with self._condition:
                    self._published_b(doc, revision)
                self._thumbnails[key] = data
                if len(self._thumbnails) > 20:
                    self._thumbnails.popitem(last=False)
                return data
        with self._condition:
            self._lookup(doc, upgrade=False)
        original = self.root / doc / 'original.png'
        # Decode one original at a time outside the document lock. Cache only small previews.
        with self._thumbnail_lock:
            try:
                stat = original.stat()
                version = (stat.st_mtime_ns, stat.st_size)
                cached = self._thumbnails.pop(doc, None)
                if cached is not None and cached[0] == version:
                    self._thumbnails[doc] = cached
                    return cached[1]
                stream = BytesIO()
                with Image.open(original) as image:
                    image.thumbnail((192, 192), Image.Resampling.LANCZOS)
                    image.save(stream, 'PNG', icc_profile=image.info.get('icc_profile'))
            except (OSError, Image.DecompressionBombError) as error:
                raise KeyError('找不到可用的缩略图。') from error
            data = stream.getvalue()
            if len(data) <= 256 * 1024:
                self._thumbnails[doc] = (version, data)
                if len(self._thumbnails) > 20:
                    self._thumbnails.popitem(last=False)
            return data

    def _source_path(self, doc, suffix):
        return self.workspace / 'inputs' / doc / ('source' + suffix)

    def _repair_source(self, doc, suffix, content):
        path = self._source_path(doc, suffix)
        if not path.parent.resolve().is_relative_to(self.workspace):
            raise ValueError('本地原图副本目录超出了工作台范围。')
        if not path.is_symlink() and path.is_file() and _sha(path) == doc:
            return False
        _write_atomic(path, content)
        return True

    def _reopen(self, state, content):
        doc = state['id']
        repaired = self._repair_source(doc, state['_suffix'], content)
        original = self.root / doc / 'original.png'
        if not original.is_file():
            _write_atomic(original, _png(*_read_source(BytesIO(content))))
        updated = {**state, '_updated': time()}
        analysis_outdated = self._analysis_outdated(state)
        edge_outdated = self._range_edge_outdated(state)
        reset_source = repaired or state['status'] == 'error' or analysis_outdated
        choose_profile = self.preferences.read()['default_profile'] == 'ask'
        active = state['status'] in ('queued', 'running')
        if state['status'] != 'awaiting_profile' and (reset_source or edge_outdated or (choose_profile and active)):
            updated = self._restart_render(updated, selection_changed=analysis_outdated or edge_outdated,
                                           reset_source=reset_source, defer_profile=choose_profile)
        else:
            self._store(updated)
        result = self._public(updated)
        if choose_profile:
            result['choose_profile'] = True
        return result

    def open(self, content, name):
        if not isinstance(content, bytes) or not content or len(content) > 64 * 1024 * 1024:
            raise ValueError('请选择不超过 64 MiB 的图片。')
        self._warmup_stop.set()
        doc = hashlib.sha256(content).hexdigest()
        with self._condition:
            if doc in self._documents:
                return self._reopen(self._documents[doc], content)
        try:
            with Image.open(BytesIO(content)) as image:
                suffix = _FORMATS.get(image.format)
                if not suffix or getattr(image, 'n_frames', 1) != 1 or image.width * image.height > 40_000_000:
                    raise ValueError('请选择不超过 4000 万像素的单帧 PNG、JPEG、WebP、BMP 或 TIFF。')
                image.verify()
            rgb, alpha, icc = _read_source(BytesIO(content))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError('无法读取图片。请检查文件格式和内容。') from error
        original = _png(rgb, alpha, icc)
        height, width = rgb.shape[:2]
        with self._condition:
            if doc in self._documents:
                return self._reopen(self._documents[doc], content)
            self._repair_source(doc, suffix, content)
            folder = self.root / doc
            _write_atomic(folder / 'original.png', original)
            preferences = self.preferences.read()
            preferred = preferences['default_profile']
            if preferred == 'ask':
                profile = None
            else:
                profile = 'fusion' if preferred != 'whole' and (self.opener is not None or part_models_present()) else 'whole'
            phase = 'awaiting_profile' if profile is None else 'queued'
            state = {'id': doc, 'profile': profile, 'name': (name.replace('\\', '/').split('/')[-1] or 'image' + suffix)[:160],
                     'width': width, 'height': height, 'revision': 0, 'settings': preferences['defaults'],
                     'edits': [], 'cursor': 0, 'status': phase, 'phase': phase, 'error': None,
                     'b': None, 'a': None, 'selection_revision': 0,
                     '_schema': 1, '_updated': time(), '_suffix': suffix, '_correction': None, '_source_epoch': 0}
            self._store(state)
            if profile is not None:
                self._queue(state)
            result = self._public(state)
            if preferred == 'fusion' and profile == 'whole':
                result['profile_fallback'] = True
            return result

    @staticmethod
    def _current(state, revision):
        if type(revision) is not int or revision != state['revision']:
            raise RevisionConflict('文档已在另一页面更新。请重新载入后继续。')

    def switch_profile(self, doc, revision, profile):
        if profile not in ('whole', 'fusion'):
            raise ValueError('请选择整图降噪或按部位精修。')
        with self._condition:
            state = self._lookup(doc, upgrade=False)
            self._current(state, revision)
            if profile == state.get('profile', 'fusion') and state['status'] != 'awaiting_profile':
                return self._public(state)
            if profile == 'fusion' and self.opener is None and not part_models_present():
                raise ValueError('缺少分区模型。请在项目根目录运行 python -m tools.assets fetch，再切换到按部位精修。')
            updated = {**state, 'profile': profile, 'selection_preview': None, '_updated': time()}
            previous = state['b']
            if profile == 'fusion' and not state['a'] and previous and previous.get('profile') == 'whole':
                updated['a'] = deepcopy(previous)
            updated = self._restart_render(updated, selection_changed=True, reset_source=True)
            self._prune(state, updated)
            return self._public(updated)

    def update(self, doc, payload):
        if not isinstance(payload, dict) or set(payload) != {'revision', 'settings', 'edits', 'cursor'}:
            raise ValueError('更新必须包含修订号、设置和编辑记录。')
        with self._condition:
            state = self._lookup(doc)
            self._current(state, payload['revision'])
            if state['status'] == 'awaiting_profile':
                raise ValueError('请先选择处理模式。')
            settings = _settings(payload['settings'])
            if state.get('profile') == 'whole':
                if (payload['edits'] != state['edits'] or type(payload['cursor']) is not int
                        or payload['cursor'] != state['cursor']):
                    _require_parts(state)
                edits = state['edits']
            else:
                stored_edits = {edit['token']: edit for edit in state['edits'] if 'token' in edit}
                edits = _history(payload['edits'], payload['cursor'], state['width'], state['height'], stored_edits)
            new_resolutions = [i for i, edit in enumerate(edits) if edit['tool'] == 'resolve_unknown'
                               and edits[i - 1:i + 1] != state['edits'][i - 1:i + 1]]
            if new_resolutions and (state['status'] != 'ready'
                    or new_resolutions != [state['cursor']]
                    or payload['cursor'] != len(edits)
                    or edits[:-1] != state['edits'][:state['cursor']]
                    or not (state.get('edit_feedback') or {}).get('skipped_unknown_pixels')):
                raise RevisionConflict('这笔补选的待确认结果已更新，请重新补选。')
            if (state['status'] != 'error' and settings == state['settings']
                    and edits == state['edits'] and payload['cursor'] == state['cursor']):
                return self._public(state)
            old, new = state['settings'], settings
            mask_changed = (any(old[key] != new[key] for key in ('modules', 'part_strengths'))
                            or edits[:payload['cursor']] != state['edits'][:state['cursor']])
            updated = {**state, 'revision': state['revision'] + 1, 'settings': settings,
                       'edits': edits, 'cursor': payload['cursor'],
                       'selection_revision': state['selection_revision'] + int(mask_changed),
                       'status': 'queued', 'phase': 'queued', 'error': None, '_updated': time()}
            self._store(updated)
            self._queue(updated)
            return self._public(updated)

    def pin(self, doc, revision):
        with self._condition:
            state = self._published_b(doc, revision)
            if state['status'] != 'ready' or state['revision'] != revision:
                raise RevisionConflict('可固定的结果已更新。请重新选择。')
            pinned = {**state, 'a': deepcopy(state['b']), '_updated': time()}
            self._store(pinned)
            self._prune(state, pinned)
            return self._public(pinned)

    def _render_path(self, doc, revision):
        return self.root / doc / f'render-{revision}'

    def _published_b(self, doc, revision):
        state = self._lookup(doc)
        if type(revision) is not int or not state['b'] or state['b']['revision'] != revision:
            raise RevisionConflict('请求的预览尚未完成或已更新。')
        return state

    @staticmethod
    def _selection_asset(state, revision):
        for rendered in (state.get('selection_preview'), state['b']):
            if (rendered and rendered.get('profile') != 'whole'
                    and (revision is None or (type(revision) is int and rendered['revision'] == revision))):
                return rendered
        raise RevisionConflict('请求的选区预览尚未完成或已更新。')

    def _published_selection(self, doc, revision):
        if type(revision) is not int:
            raise RevisionConflict('请指定选区预览的修订号。')
        return self._selection_asset(self._lookup(doc), revision)

    def _changes_state(self, doc, revision):
        state = self._lookup(doc, upgrade=False)
        self._current(state, revision)
        _require_parts(state)
        if (state['status'] != 'ready' or not state['b'] or state['b']['revision'] != revision
                or self._analysis_outdated(state) or self._range_edge_outdated(state)):
            raise RevisionConflict('当前成品尚未更新，请稍后查看手动改动。')
        return state

    def changes(self, doc, revision, *, stage=None):
        if stage not in _CHANGE_STAGES:
            raise ValueError('请选择要核对的编辑阶段。')
        kinds = _CHANGE_STAGES[stage]
        key = (doc, revision, stage)
        result = {'revision': revision, 'counts': dict.fromkeys(kinds, 0), 'overlay': None}
        if stage is not None:
            result['stage'] = stage
        if stage == 'parts':
            result['parts'] = []
        with self._condition:
            state = self._changes_state(doc, revision)
            if key in self._changes_cache:
                return self._changes_cache[key]
            if not state['cursor'] and not state.get('_correction'):
                return result

        with self._changes_lock:
            with self._condition:
                self._changes_state(doc, revision)
                if key in self._changes_cache:
                    return self._changes_cache[key]
            editor, mask, lines = self._changes_selection(doc, revision, state)
            settings = state['settings']
            objects, unknown = mask.legacy_semantic_masks or (mask.objects, mask.unknown)
            if stage == 'parts':
                counts, overlay, result['parts'] = _part_changes_overlay(editor, mask.labels, objects, unknown)
            elif stage == 'range':
                corrected = SimpleNamespace(rgb=editor.rgb, alpha=editor.alpha, labels=mask.labels,
                                            objects=objects, unknown=unknown)
                baseline = _automatic_weight(corrected, settings, lines)
                counts, overlay = _changes_overlay(editor, baseline, mask, include_parts=False)
            else:
                baseline_key = (doc, state.get('_source_epoch', 0), tuple(settings['modules']),
                                tuple(sorted(settings['part_strengths'].items())), settings['strength'],
                                settings['structure_protection'])
                if self._changes_baseline is None or self._changes_baseline[0] != baseline_key:
                    self._changes_baseline = (baseline_key, _automatic_weight(editor, settings, lines))
                counts, overlay = _changes_overlay(editor, self._changes_baseline[1], mask)
            result.update(counts={kind: counts[kind] for kind in kinds}, overlay=overlay)
            with self._condition:
                self._changes_state(doc, revision)
                if self._changes_cache and next(iter(self._changes_cache))[:2] != key[:2]:
                    self._changes_cache.clear()
                self._changes_cache[key] = result
            return result

    def _changes_selection(self, doc, revision, state):
        epoch = state.get('_source_epoch', 0)
        # Capture the immutable editor and completed selection, then let the
        # worker render other revisions while the comparison runs on CPU.
        if self._inference.acquire(blocking=False):
            try:
                with self._condition:
                    self._changes_state(doc, revision)
                if ((self._resident_id, self._resident_epoch) == (doc, epoch)
                        and self._selection_key == self._selection_cache_key(state)):
                    return self._resident.editor, self._selection_value[0], self._weight_cache.lines
            finally:
                self._inference.release()

        from .session import _cached_editor

        source = self._source_path(doc, state['_suffix'])
        try:
            with self.cache.using() as workspace:
                editor = _cached_editor(source, workspace, doc, [state['width'], state['height']])
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            # Custom openers and missing caches still use the verified session
            # path. Only the analysis, if needed, holds the inference lock.
            if not self._inference.acquire(blocking=False):
                raise RevisionConflict('降噪工作台正在处理图片，请稍后查看。')
            try:
                with self._condition:
                    self._changes_state(doc, revision)
                if _sha(source) != doc:
                    raise ValueError('本地原图副本已更改，请重新打开源图。')
                editor = self._open_session(source, state).editor
            finally:
                self._inference.release()
        cache = self._make_weight_cache(doc, editor)
        return editor, self._replay_selection(doc, state, editor, cache), cache.lines

    def asset(self, doc, view, *, download=False, revision=None, module=None, hidden_parts=()):
        if hidden_parts and (view not in ('parts', 'parts_overlay') or download):
            raise ValueError('只有分区叠加支持隐藏部位。')
        if view == 'parts_overlay' and not download:
            return self._parts_overlay(doc, revision, hidden_parts)
        if view == 'weight_overlay' and not download:
            with self._condition:
                self._published_selection(doc, revision)
                data = (self._render_path(doc, revision) / 'weight_overlay.png').read_bytes()
            overlay = _recolor_legacy_weight(data)
            with self._condition:
                self._published_selection(doc, revision)
            return overlay
        if view == 'magnetic_source' and not download:
            return self._magnetic_source(doc, revision)
        if view == 'highlight' and not download:
            color, opacity = ((BACKGROUND_TINT, _BACKGROUND_HIGHLIGHT_OPACITY) if module == 'background'
                              else ((255, 255, 255), 145))
            return self._part_overlay(doc, revision, module, color, opacity)
        if view == 'background_overlay' and not download:
            return self._part_overlay(doc, revision, 'background', BACKGROUND_TINT, _BACKGROUND_OVERLAY_OPACITY)
        with self._condition:
            state = self._lookup(doc)
            if view == 'original' and not download:
                return (self.root / doc / 'original.png').read_bytes()
            if (view not in ('a', 'b', 'candidate', *_SELECTION_ASSETS) or (download and view not in ('a', 'b'))):
                raise KeyError('找不到此视图。')
            if view in _SELECTION_ASSETS:
                if (view == 'labels' or hidden_parts) and type(revision) is not int:
                    raise RevisionConflict('请指定选区预览的修订号。')
                rendered = self._selection_asset(state, revision)
            else:
                rendered = state['a' if view == 'a' else 'b']
            if (not rendered or (revision is not None and rendered['revision'] != revision)
                    or (download and view == 'b' and (state['status'] != 'ready' or rendered['revision'] != state['revision']))):
                raise RevisionConflict('请求的结果尚未完成或已更新。')
            if view == 'suspected_objects':
                if not rendered.get('images', {}).get(view):
                    raise KeyError('此版本没有疑似外物图层。')
                version = rendered.get('analysis_version', state.get('_analysis_version'))
                filename = f'suspected-objects-{version}.png' if type(version) is int else _SUSPECTED_ASSET
                return (self.root / doc / filename).read_bytes()
            if view == 'unknown' and not rendered.get('images', {}).get(view):
                raise KeyError('此版本没有待确认图层。')
            if view == 'candidate':
                self._published_b(doc, revision)
            composite = view in ('parts', 'weight') and rendered.get('images', {}).get(view + '_overlay')
            published = rendered['revision']
            if not composite and not hidden_parts:
                filename = 'result.png' if view in ('a', 'b') else view + '.png'
                path = self._render_path(doc, published) / filename
                if view in ('candidate', 'labels') and not path.is_file():
                    if state['status'] == 'ready':
                        self._restart_render(state, selection_changed=False, reset_source=True)
                    raise RevisionConflict('正在为这份草稿生成节点预览。')
                return path.read_bytes()
        if composite:
            return self._composited_overlay(doc, view, published, hidden_parts)
        return self._filtered_parts(doc, published, hidden_parts)

    def _parts_overlay(self, doc, revision, hidden_parts, *, background=True):
        from .masks import PART_CODES

        hidden = self._hidden_part_codes(hidden_parts)
        palette = bytearray(256 * 3)
        opacity = bytearray(256)
        for part, color in {**PART_PALETTE, 'object': _OBJECT_COLOR}.items():
            code = PART_CODES[part]
            palette[3 * code:3 * code + 3] = bytes(color)
            if code not in hidden:
                if part == 'background':
                    opacity[code] = _BACKGROUND_OVERLAY_OPACITY if background else 0
                else:
                    opacity[code] = 143 if part == 'object' else 107
        labels = self.asset(doc, 'labels', revision=revision)
        with Image.open(BytesIO(labels)) as image:
            result = _indexed_png(image, palette, opacity)
        with self._condition:
            self._published_selection(doc, revision)
        return result

    def _composited_overlay(self, doc, view, revision, hidden_parts=(), *, size=None):
        if view == 'parts':
            overlay = self._parts_overlay(doc, revision, hidden_parts, background=size is not None)
        else:
            overlay = self.asset(doc, 'weight_overlay', revision=revision)
        original = self.asset(doc, 'original')
        with Image.open(BytesIO(original)) as source, Image.open(BytesIO(overlay)) as mask:
            layer = mask.convert('RGBA')
            if size is not None:
                source.thumbnail(size, Image.Resampling.LANCZOS)
                layer = layer.resize(source.size, Image.Resampling.LANCZOS)
            tint = np.asarray(layer)
            alpha = np.asarray(source.getchannel('A')) if 'A' in source.getbands() else None
            rgb = _overlay(np.asarray(source.convert('RGB')), tint[..., :3], tint[..., 3].astype(np.float32) / 255)
            result = _png(rgb, alpha, source.info.get('icc_profile'))
        with self._condition:
            self._published_selection(doc, revision)
        return result

    @staticmethod
    def _hidden_part_codes(hidden_parts):
        from .masks import PART_CODES

        if len(hidden_parts) > len(PART_CODES) or any(part not in PART_CODES for part in hidden_parts):
            raise ValueError('请选择有效的分区。')
        return {PART_CODES[part] for part in hidden_parts}

    def _filtered_parts(self, doc, revision, hidden_parts):
        from .masks import PART_CODES

        codes = self._hidden_part_codes(hidden_parts)
        with self._condition:
            self._published_selection(doc, revision)
        original = self.asset(doc, 'original')
        if len(codes) == len(PART_CODES):
            return original
        labels = self.asset(doc, 'labels', revision=revision)
        parts = self.asset(doc, 'parts', revision=revision)
        stream = BytesIO()
        with (Image.open(BytesIO(original)) as source, Image.open(BytesIO(parts)) as shaded,
              Image.open(BytesIO(labels)) as ids):
            reveal = ids.point([255 if code in codes else 0 for code in range(256)])
            composite = Image.composite(source, shaded, reveal)
            composite.save(stream, 'PNG', compress_level=1, icc_profile=source.info.get('icc_profile'))
        with self._condition:
            self._published_selection(doc, revision)
        return stream.getvalue()

    def _part_overlay(self, doc, revision, module, color, opacity):
        from .masks import PART_CODES

        if not isinstance(module, str) or module not in PART_CODES:
            raise ValueError('请选择要高亮的部位。')
        labels = self.asset(doc, 'labels', revision=revision)
        with Image.open(BytesIO(labels)) as image:
            transparency = bytearray(256)
            transparency[PART_CODES[module]] = opacity
            result = _indexed_png(image, bytes(color) * 256, transparency)
        with self._condition:
            self._published_selection(doc, revision)
        return result

    def export(self, doc, view, revision):
        if view not in ('a', 'b') or type(revision) is not int or revision < 0:
            raise ValueError('请选择要导出的对照或成品。')
        with self._condition:
            state = self._lookup(doc)
            folder = self.preferences.read()['export_dir']
            if not folder:
                raise ValueError('请先选择本机导出目录。')
            self.cache.validate_base(Path(folder).resolve())
            data = self.asset(doc, view, download=True, revision=revision)
            return export_png(folder, state['name'], data)

    def _magnetic_state(self, doc, revision):
        state = self._lookup(doc)
        self._current(state, revision)
        _require_parts(state)
        if state['status'] != 'ready' or not state['b'] or state['b']['revision'] != revision:
            raise RevisionConflict('当前降噪候选尚未就绪。')
        return state

    def _magnetic_source(self, doc, revision):
        # Encode only on demand. A single cached PNG can follow the active
        # document across brush revisions without retaining every old image.
        if not self._magnetic.acquire(blocking=False):
            raise RevisionConflict('正在准备吸边底图。')
        try:
            with self._condition:
                state = self._magnetic_state(doc, revision)
                restore_key = _restoration_key(state['settings'])
                key = (doc, state.get('_source_epoch', 0), restore_key)
                if self._magnetic_cache is not None and self._magnetic_cache[0] == key:
                    return self._magnetic_cache[1]

            restored, alpha, icc = self._magnetic_candidate(doc, revision, state, restore_key)
            rgb = restored * np.float32(255)
            rgb = np.rint(rgb, out=rgb).astype(np.uint8)
            data = _png(rgb, alpha, icc)
            with self._condition:
                self._magnetic_state(doc, revision)
                self._magnetic_cache = (key, data)
            return data
        finally:
            self._magnetic.release()

    def _magnetic_candidate(self, doc, revision, state, restore_key):
        settings = state['settings']
        epoch = state.get('_source_epoch', 0)
        source = self._source_path(doc, state['_suffix'])
        if settings['method'] == 'l0' and (self._resident_id, self._resident_epoch) != (doc, epoch):
            from .session import _cached_l0_baseline

            if _sha(source) != doc:
                raise ValueError('本地原图副本已更改，请重新打开源图。')
            try:
                with self.cache.using() as cases:
                    rgb, alpha, icc, candidate = _cached_l0_baseline(
                        source, cases, doc, [state['width'], state['height']], settings['l0'])
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
                pass  # Missing or invalid analysis caches use the normal verified session path.
            else:
                return _restore_candidate(rgb, candidate, settings), alpha, icc

        if not self._inference.acquire(blocking=False):
            raise RevisionConflict('降噪工作台正在处理图片。')
        try:
            with self._condition:
                self._magnetic_state(doc, revision)
            if (self._resident_id, self._resident_epoch) == (doc, epoch):
                session = self._resident
                if self._restored_key == restore_key:
                    return self._restored, session.editor.alpha, session.icc
            else:
                if _sha(source) != doc:
                    raise ValueError('本地原图副本已更改，请重新打开源图。')
                session = self._open_session(source, state)
            candidate = session.candidate(settings['method'], lam=settings['l0'], downscale=settings['downscale'])
        finally:
            self._inference.release()
        return _restore_candidate(session.editor.rgb, candidate, settings), session.editor.alpha, session.icc

    def corrections(self, doc, revision):
        data = None
        with self._condition:
            state = self._published_b(doc, revision)
            self._current(state, revision)
            _require_parts(state)
            path = self._render_path(doc, revision) / 'corrections.json'
            if path.is_file():
                record = json.loads(path.read_text(encoding='utf-8'))
                filenames = [record[key] for key in MASK_KEYS if key in record]
                filenames.extend(record.get('module_overrides', {}).values())
                data = {name: (path.parent / name).read_bytes() for name in [*filenames, path.name]}
            elif state.get('correction_error'):
                raise ValueError(state['correction_error'])

        error = None
        if data is None:
            editor, mask, _ = self._changes_selection(doc, revision, state)
            data = {}
            error = self._encode_corrections(data, state, editor, mask)

        with self._condition:
            self._current(self._published_b(doc, revision), revision)
            if error:
                raise ValueError(error)
            folder = self.workspace / 'corrections' / doc / f'revision-{revision}'
            folder.mkdir(parents=True, exist_ok=True)
            for name, content in data.items():
                target = folder / name
                try:
                    with target.open('xb') as stream:
                        stream.write(content)
                except FileExistsError:
                    if target.read_bytes() != content:
                        raise ValueError('先前导出的校正已被修改。')
            return folder / path.name

    def import_correction(self, doc, revision, record, files):
        if not isinstance(record, dict) or not isinstance(files, dict) or record.get('sam_regions'):
            raise ValueError('工作台只支持导入不含 SAM 提示的二值校正。')
        overrides = record.get('module_overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError('部位校正格式无效。')
        names = {_safe_mask_filename(record[key]) for key in MASK_KEYS if key in record}
        names.update(_safe_mask_filename(name) for name in overrides.values())
        if (len(names) != len({name.casefold() for name in names}) or set(files) != names
                or any(not isinstance(content, bytes) or not content for content in files.values())):
            raise ValueError('请选择校正文件及其引用的全部蒙版。')
        with self._condition:
            state = self._lookup(doc)
            self._current(state, revision)
            _require_parts(state)
            if record.get('source_sha256') != doc or record.get('size') != [state['width'], state['height']]:
                raise ValueError('校正文件与当前原图不匹配。')
            if state['cursor'] >= 2000:
                raise ValueError('编辑记录已超过上限。')
        token = 'correction-' + uuid4().hex
        folder = self.root / doc / token
        folder.mkdir()
        record_path = folder / 'corrections.json'
        try:
            for name, content in files.items():
                (folder / name).write_bytes(content)
            _json(record_path, record)
            load_edits(record_path, (state['height'], state['width']), doc)
            with self._condition:
                state = self._lookup(doc)
                self._current(state, revision)
                edits = [*state['edits'][:state['cursor']], {'tool': 'import', 'token': token}]
                updated = {**state, 'revision': revision + 1, 'edits': edits, 'cursor': len(edits),
                           'selection_revision': state['selection_revision'] + 1,
                           'status': 'queued', 'phase': 'queued', 'error': None,
                           '_updated': time()}
                self._store(updated)
                self._queue(updated)
                return self._public(updated)
        except Exception:
            for name in (*files, 'corrections.json'):
                (folder / name).unlink(missing_ok=True)
            folder.rmdir()
            raise

    def preview_object(self, doc, revision, positive, negative):
        with self._condition:
            state = self._lookup(doc)
            self._current(state, revision)
            _require_parts(state)
            width, height = state['width'], state['height']
            included = _sam_points(positive, width, height)
            excluded = _sam_points(negative, width, height)
            if not included or included & excluded:
                raise ValueError('请在选区内点至少一个加入点，加入点和排除点不能重叠。')
            if state['status'] != 'ready':
                raise RevisionConflict('请等待当前结果更新后再点选区域。')
        # The render worker and HTTP requests share the GPU. State queries and
        # edits remain available while inference is in progress.
        with self._inference, self.cache.processing():
            with self._condition:
                state = self._lookup(doc)
                self._current(state, revision)
                self._object_previews.pop(doc, None)
                source = self._source_path(doc, state['_suffix'])
            if _sha(source) != doc:
                raise ValueError('本地原图副本已更改，请重新打开源图。')
            session = self._resident
            if self._resident_id != doc or self._resident_epoch != state.get('_source_epoch', 0):
                session = self._open_session(source, state)
            mask = session.predict_object(positive, negative)
            if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != (height, width):
                raise ValueError('SAM2 蒙版与原图尺寸不一致。')
            if session.editor.alpha is not None:
                mask = mask & (session.editor.alpha > 0)
            if not all(mask[y, x] for x, y in included) or any(mask[y, x] for x, y in excluded):
                raise ValueError('SAM2 蒙版未满足点选范围，请调整加入点或排除点。')
            overlay = np.zeros((height, width, 4), np.uint8)
            overlay[mask] = (224, 136, 68, 146)
            png = _png(mask.astype(np.uint8) * 255, None, None)
            preview = 'data:image/png;base64,' + b64encode(_png(overlay, None, None)).decode('ascii')
            token = 'correction-' + uuid4().hex
            with self._condition:
                self._current(self._lookup(doc), revision)
                self._object_previews[doc] = (revision, token, png)
                if len(self._object_previews) > 4:
                    self._object_previews.popitem(last=False)
        return {'revision': revision, 'token': token, 'preview': preview, 'pixels': int(mask.sum())}

    def apply_object(self, doc, revision, token, selection=None):
        with self._condition:
            state = self._lookup(doc)
            self._current(state, revision)
            pending = self._object_previews.get(doc)
            if pending is None or pending[:2] != (revision, token):
                raise RevisionConflict('点选预览已过期，请重新生成。')
            if state['cursor'] >= 2000:
                raise ValueError('编辑记录已超过上限。')
            if selection is None:
                edit = {'tool': 'sam_object', 'token': token}
            else:
                if set(selection) - {'stage', 'tool', 'module', 'strength', 'allow_protected', 'protected_parts'}:
                    raise ValueError('SAM2 选区操作字段无效。')
                _stage_tool(selection)
                if selection['tool'] in ('reset', 'invert', 'resolve_unknown'):
                    raise ValueError('此操作不能使用 SAM2 选区。')
                edit = {**selection, 'shape': 'sam', 'token': token}
            path = self.root / doc / token / 'object_add_mask.png'
            _write_atomic(path, pending[2])
            edits = [*state['edits'][:state['cursor']], edit]
            updated = {**state, 'revision': revision + 1, 'edits': edits, 'cursor': len(edits),
                       'selection_revision': state['selection_revision'] + 1,
                       'status': 'queued', 'phase': 'queued', 'error': None, '_updated': time()}
            self._store(updated)
            self._queue(updated)
            return self._public(updated)

    def _saved_object_mask(self, doc, token, shape):
        if not isinstance(token, str) or not _CORRECTION_TOKEN.fullmatch(token):
            raise ValueError('保存的外物记录无效。')
        with Image.open(self.root / doc / token / 'object_add_mask.png') as image:
            if image.size != (shape[1], shape[0]):
                raise ValueError('保存的外物蒙版尺寸与原图不一致。')
            return np.asarray(image.convert('L')) > 0

    def _stage(self, doc, revision, phase):
        with self._condition:
            state = self._documents[doc]
            if self._closed or state['revision'] != revision:
                return False
            state.update(status='running', phase=phase)
            return True

    def _run(self):
        while True:
            with self._condition:
                # Keep one prepared image waiting for the encoder. A newer edit
                # replaces its document's queued result before rendering again.
                self._condition.wait_for(lambda: self._closed or (
                    (bool(self._pending) or self._prepare_request is not None) and not self._pending_results))
                if self._closed:
                    return
                preparing = not self._pending
                if preparing:
                    doc, revision = self._prepare_request
                    self._prepare_request = None
                else:
                    doc, revision = self._pending.popitem(last=False)
                    if self._prepare_request is not None and self._prepare_request[0] == doc:
                        self._prepare_request = None
                state = deepcopy(self._documents[doc])
                if preparing and (state['revision'] != revision or state['status'] != 'ready'):
                    continue
            try:
                with self._inference, self.cache.processing():
                    if preparing:
                        self._warm_edit_cache(doc, revision, state)
                        continue
                    prepared = self._render(doc, revision, state)
                if prepared is not None:
                    with self._condition:
                        current = self._documents[doc]
                        if not self._closed and current['revision'] == revision:
                            current['phase'] = 'encoding'
                            self._pending_results[doc] = prepared
                            self._condition.notify_all()
                del prepared
            except Exception as error:
                if preparing:
                    # Preparation is optional. A later edit still uses the normal render path.
                    traceback.print_exc()
                else:
                    self._render_failed(doc, revision, error)

    def _run_encoder(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or bool(self._pending_results))
                if self._closed:
                    return
                doc, job = self._pending_results.popitem(last=False)
                self._condition.notify_all()
            try:
                if job.image.png is None:
                    job.image.png = _png(job.image.rgb, job.alpha, job.icc)
                self._publish_render(doc, job.revision, {'candidate.png': job.candidate_png, 'result.png': job.image.png},
                                     job.publication)
            except Exception as error:
                self._render_failed(doc, job.revision, error)
            finally:
                del job

    def _render_failed(self, doc, revision, error):
        with self._condition:
            current = self._documents[doc]
            if not self._closed and current['revision'] == revision:
                failed = {**current, 'status': 'error', 'phase': 'error', 'failed_phase': current['phase'],
                          'error': str(error) or type(error).__name__}
                try:
                    self._store(failed)
                except OSError:
                    current.update(failed)

    @staticmethod
    def _selection_cache_key(state):
        settings = state['settings']
        return (state['selection_revision'], settings['strength'], settings['structure_protection'])

    def _make_weight_cache(self, doc, editor):
        from .masks import WorkbenchWeightCache

        shape = editor.labels.shape
        return WorkbenchWeightCache(editor, object_mask=lambda token: self._saved_object_mask(doc, token, shape))

    def _drop_resident(self):
        self._resident_id = self._resident_epoch = self._resident = None
        self._has_suspected_objects = False
        self._weight_cache = None
        self._selection_key = self._selection_value = None
        self._restored_key = self._restored = None
        self._result_cache = self._candidate_png = None

    def _install_resident(self, doc, epoch, session):
        suspected = getattr(session, 'suspected_objects', None)
        self._has_suspected_objects = suspected is not None and suspected.any()
        if self._has_suspected_objects:
            overlay_path = self.root / doc / _SUSPECTED_ASSET
            if not overlay_path.is_file():
                editor = session.editor
                rgb = editor.rgb.copy()
                rgb[suspected] = _overlay(rgb[suspected], np.array([243, 194, 87], np.uint8), .56)
                _write_atomic(overlay_path, _png(rgb, editor.alpha, session.icc))
        self._weight_cache = self._make_weight_cache(doc, session.editor)
        self._resident = session
        self._resident_id, self._resident_epoch = doc, epoch

    def _prepare_selection(self, doc, state, session):
        selection_key = self._selection_cache_key(state)
        if selection_key != self._selection_key:
            mask = self._replay_selection(doc, state, session.editor, self._weight_cache)
            self._selection_value = mask, _selection_images(session, mask, previous=self._selection_value)
            self._selection_key = selection_key
        return self._selection_value

    def _warm_edit_cache(self, doc, revision, state):
        with self._condition:
            if self._closed or self._pending or self._prepare_request not in (None, (doc, revision)):
                return
        epoch = state.get('_source_epoch', 0)
        if (self._resident_id, self._resident_epoch) != (doc, epoch):
            self._drop_resident()
            source = self._source_path(doc, state['_suffix'])
            with self.cache.using() as root:
                session = self._saved_session(source, root, state)
            if session is None:
                return
            with self._condition:
                current = self._documents[doc]
                if (self._closed or current.get('_source_epoch', 0) != epoch
                        or (self._prepare_request is not None and self._prepare_request[0] != doc)
                        or (self._pending and doc not in self._pending)):
                    return
            self._install_resident(doc, epoch, session)
        with self._condition:
            if (self._closed or self._pending or self._documents[doc]['revision'] != revision
                    or self._prepare_request not in (None, (doc, revision))):
                return
        self._prepare_selection(doc, state, self._resident)

    def _replay_selection(self, doc, state, editor, cache):
        token, active = _active_edit_layers(state)
        correction = (load_edits(self.root / doc / token / 'corrections.json', editor.labels.shape, doc)
                      if token is not None else None)
        return _selection(editor, state['settings'], active, len(active), correction=correction,
                          object_mask=cache.object_mask, weight_cache=cache)

    def _blend_result(self, original, weight):
        from ..filters import blend

        if self._result_cache is not None:
            previous_weight, previous = self._result_cache
            changed = weight != previous_weight
            rows = np.flatnonzero(changed.any(axis=1))
            if not rows.size:
                return previous
            columns = np.flatnonzero(changed.any(axis=0))
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(columns[0]), int(columns[-1]) + 1
            if (y1 - y0) * (x1 - x0) <= weight.size // 4:
                region = (slice(y0, y1), slice(x0, x1))
                rgb = previous.rgb.copy()
                rgb[region] = blend(original[region], self._restored[region], weight[region])
                return _ResultImage(rgb)
        return _ResultImage(blend(original, self._restored, weight))

    def _render_whole(self, doc, revision, state):
        from ..filters import blend, whole_weight
        from .session import ImageCandidates

        settings = state['settings']
        epoch = state.get('_source_epoch', 0)
        if (self._resident_id, self._resident_epoch) != (doc, epoch):
            if not self._stage(doc, revision, 'loading'):
                return
            self._drop_resident()
            source = self._source_path(doc, state['_suffix']).read_bytes()
            if hashlib.sha256(source).hexdigest() != doc:
                raise ValueError('本地原图副本已更改，请重新打开源图。')
            self._resident = ImageCandidates(*_read_source(BytesIO(source)), device='auto')
            self._resident_id, self._resident_epoch = doc, epoch
        session = self._resident
        if not self._prepare_candidate(doc, revision, session, session.rgb, session.alpha, settings):
            return
        weight, _ = whole_weight(session.rgb, session.alpha, strength=settings['strength'],
                                 structure_protection=settings['structure_protection'])
        result = _ResultImage(blend(session.rgb, self._restored, weight))
        publication = {'b': {'revision': revision, 'profile': 'whole', 'method': settings['method'],
                             'selection_revision': state['selection_revision'], 'images': {}},
                       'selection_preview': None, 'status': 'ready', 'phase': 'done', 'error': None,
                       'edit_feedback': None, 'correction_error': None,
                       'active_percent': round(float((weight > 0).mean() * 100), 2)}
        return _ResultJob(revision, result, session.alpha, session.icc, self._candidate_png, publication)

    def _prepare_candidate(self, doc, revision, session, rgb, alpha, settings):
        key = _restoration_key(settings)
        if key == self._restored_key:
            return self._stage(doc, revision, 'compose')
        if not self._stage(doc, revision, 'candidate'):
            return False
        self._result_cache = None
        candidate = session.candidate(settings['method'], lam=settings['l0'], downscale=settings['downscale'])
        if not self._stage(doc, revision, 'compose'):
            return False
        self._restored = _restore_candidate(rgb, candidate, settings)
        self._candidate_png = _png(np.rint(candidate * 255).astype(np.uint8), alpha, session.icc)
        self._restored_key = key
        return True

    def _render(self, doc, revision, state):
        if state.get('profile') == 'whole':
            return self._render_whole(doc, revision, state)
        epoch = state.get('_source_epoch', 0)
        needs_analysis = self._resident_id != doc or self._resident_epoch != epoch
        settings = state['settings']
        restore_key = _restoration_key(settings)
        needs_candidate = needs_analysis or restore_key != self._restored_key
        phase = 'analysis' if needs_analysis else 'candidate' if needs_candidate else 'compose'
        if not self._stage(doc, revision, phase):
            return
        if needs_analysis:
            self._drop_resident()
            source = self._source_path(doc, state['_suffix'])
            if _sha(source) != doc:
                raise ValueError('本地原图副本已更改，请重新打开源图。')
            opened = self._open_session(source, state, on_stage=lambda name: self._stage(doc, revision, name))
            self._install_resident(doc, epoch, opened)
        session = self._resident
        if not self._stage(doc, revision, 'compose'):
            return
        mask, images = self._prepare_selection(doc, state, session)
        image_views = {name: True for name, available in (
            ('suspected_objects', self._has_suspected_objects), ('unknown', 'unknown.png' in images),
            ('parts_overlay', 'parts.png' not in images), ('weight_overlay', 'weight_overlay.png' in images)) if available}
        rendered = {'revision': revision, 'method': settings['method'], 'selection_revision': state['selection_revision'],
                    'cursor': state['cursor'], 'images': image_views}
        preview = {**rendered, 'analysis_version': _ANALYSIS_VERSION}
        if not self._publish_render(doc, revision, images, {'selection_preview': preview}):
            return
        if not self._prepare_candidate(doc, revision, session, session.editor.rgb, session.editor.alpha, settings):
            return
        result = self._blend_result(session.editor.rgb, mask.weight)
        if result.rgb.nbytes <= _MAX_RETAINED_RESULT_BYTES:
            self._result_cache = (mask.weight, result)
        publication = {'b': rendered, 'selection_preview': None, '_analysis_version': _ANALYSIS_VERSION,
                       '_range_edge_version': _RANGE_EDGE_VERSION,
                       'status': 'ready', 'phase': 'done', 'error': None,
                       'edit_feedback': mask.edit_feedback, 'correction_error': None,
                       'active_percent': round(float((mask.weight > 0).mean() * 100), 2)}
        return _ResultJob(revision, result, session.editor.alpha, session.icc, self._candidate_png, publication)

    def _publish_render(self, doc, revision, data, publication):
        # Publish metadata only after the matching files are available.
        with self._condition:
            current = self._documents[doc]
            if self._closed or current['revision'] != revision:
                return False
            folder = self._render_path(doc, revision)
            folder.mkdir(exist_ok=True)
            for name, content in data.items():
                (folder / name).write_bytes(content)
            updated = {**current, **publication}
            self._store(updated)
            self._prune(current, updated)
            return True

    @staticmethod
    def _encode_corrections(data, state, editor, mask):
        from .masks import binary_corrections

        try:
            masks = binary_corrections(editor, mask, state['settings'])
        except ValueError as error:
            return '当前设置不能导出为 CLI 二值校正。' + str(error)
        record = {'version': 2, 'source_sha256': state['id'], 'size': [state['width'], state['height']]}
        for key, value in masks.items():
            if key == 'module_overrides':
                if value:
                    record[key] = {}
                for part, array in value.items():
                    name = 'module-' + part + '.png'
                    data[name] = _png(array.astype(np.uint8) * 255, None, None)
                    record[key][part] = name
            else:
                name = key + '.png'
                data[name] = _png(value.astype(np.uint8) * 255, None, None)
                record[key] = name
        data['corrections.json'] = json.dumps(record, ensure_ascii=False, indent=2).encode('utf-8')

    def _prune(self, previous, current):
        keys = ('a', 'b', 'selection_preview')
        keep = {current[key]['revision'] for key in keys if current.get(key)}
        generated = ('result.png', 'parts.png', 'weight.png', 'weight_overlay.png', 'unknown.png', 'candidate.png', 'labels.png', 'corrections.json',
                     *(key + '.png' for key in MASK_KEYS),
                     *('module-' + part + '.png' for part in MODULES if part != 'background'))
        for revision in {previous[key]['revision'] for key in keys if previous.get(key)} - keep:
            folder = self._render_path(current['id'], revision)
            if folder.is_symlink() or folder.is_junction():
                continue
            try:
                for filename in generated:
                    (folder / filename).unlink(missing_ok=True)
                folder.rmdir()
            except OSError:
                pass  # Output already published; stale files are harmless after failed cleanup.
