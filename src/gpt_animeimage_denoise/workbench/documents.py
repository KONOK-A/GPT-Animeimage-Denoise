"""Persist workbench drafts and publish only the newest completed image revision."""
from base64 import b64encode
from collections import OrderedDict
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path, PureWindowsPath
import re
from threading import Condition, Lock, Thread
from time import time
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..corrections import MASK_KEYS, load_edits
from ..denoisers import resolve_denoiser
from ..editor import DEFAULT_MODULES, MODULES
from ..filters import _radius, blend, gradient_preserving_candidate, validate_unit_interval
from ..pipeline import _json, _read_source, _sha


DEFAULTS = {'method': 'l0', 'l0': .0005, 'downscale': .6, 'strength': 1,
            'restoration_gain': .25, 'sigma': 12, 'structure_protection': .2,
            'modules': list(DEFAULT_MODULES), 'part_strengths': {}}
WORKBENCH_METHODS = {'l0': '纯 L0', 'esrgan': 'ESRGAN', 'esrgan-l0': 'ESRGAN + L0', 'apisr': 'APISR'}
# Recognize old drafts so they can resume without the retired model.
_RETIRED_METHODS = {'cugan', 'cugan-l0', 'cugan60', 'cugan60_l0'}
_FORMATS = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp', 'BMP': '.bmp', 'TIFF': '.tif'}
_DOCUMENT_ID = re.compile(r'[0-9a-f]{64}')
_CORRECTION_TOKEN = re.compile(r'correction-[0-9a-f]{32}')
_ANALYSIS_VERSION = 4
_SUSPECTED_ASSET = f'suspected-objects-{_ANALYSIS_VERSION}.png'
_PALETTE = {'skin': (219, 137, 108), 'face': (225, 180, 80), 'hair': (101, 135, 223),
            'eyes': (54, 165, 160), 'mouth': (217, 120, 164), 'outfit': (139, 120, 199)}
_CHANGE_KINDS = ('added', 'removed', 'stronger', 'weaker', 'reclassified')
_CHANGE_COLORS = np.array([(0, 0, 0, 0), (56, 189, 248, 180), (244, 119, 118, 180),
                           (81, 223, 156, 180), (248, 191, 86, 180), (190, 154, 255, 180)], np.uint8)


class RevisionConflict(ValueError):
    """An update or export refers to a revision that is no longer current."""


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


def _restoration_key(settings):
    return (settings['method'], settings['l0'], settings['downscale'],
            settings['sigma'], settings['restoration_gain'])


def _restore_candidate(rgb, candidate, settings):
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


def _unknown_overlay(unknown, alpha):
    visible = unknown if alpha is None else unknown & (alpha != 0)
    if not visible.any():
        return None
    overlay = np.zeros((*unknown.shape, 4), np.uint8)
    yy, xx = np.nonzero(visible)
    stripe_width = max(4, round(min(unknown.shape) / 192))
    value = np.where(((xx + yy) // stripe_width) % 2 == 0, 245, 22)
    overlay[yy, xx, :3] = value[:, None]
    overlay[yy, xx, 3] = 158 if alpha is None else (alpha[yy, xx].astype(np.uint16) * 158 + 127) // 255
    return _png(overlay, None, None)


def _changes_overlay(editor, auto_weight, mask):
    weight = mask.weight
    classes = np.zeros(weight.shape, np.uint8)
    stronger = weight > auto_weight
    weaker = weight < auto_weight
    classes[stronger] = 3
    classes[stronger & (auto_weight == 0)] = 1
    classes[weaker] = 4
    classes[weaker & (weight == 0)] = 2
    reclassified = (mask.labels != editor.labels) | (mask.objects != editor.objects) | (mask.unknown != editor.unknown)
    classes[(classes == 0) & reclassified] = 5
    if editor.alpha is not None:
        classes[editor.alpha == 0] = 0
    sizes = np.bincount(classes.ravel(), minlength=len(_CHANGE_COLORS))
    counts = dict(zip(_CHANGE_KINDS, map(int, sizes[1:])))
    if not sizes[1:].any():
        return counts, None
    overlay = _CHANGE_COLORS[classes]
    if editor.alpha is not None:
        overlay[..., 3] = (overlay[..., 3].astype(np.uint16) * editor.alpha + 127) // 255
    return counts, 'data:image/png;base64,' + b64encode(_png(overlay, None, None)).decode('ascii')


def _selection_images(session, mask, *, previous=None):
    editor = session.editor
    old_mask, old_images = previous or (None, {})
    images = {}
    same_unknown = old_mask is not None and np.array_equal(old_mask.unknown, mask.unknown)
    if same_unknown and all(np.array_equal(getattr(old_mask, key), getattr(mask, key))
                            for key in ('labels', 'objects', 'soft_protection')):
        images['parts.png'] = old_images['parts.png']
    else:
        labels = mask.labels
        colors = np.zeros((256, 3), np.uint8)
        for module, color in _PALETTE.items():
            colors[MODULES[module]] = color
        tint = colors[labels]
        tint[mask.objects] = (224, 136, 68)
        opacity = np.where(mask.objects, np.float32(.56),
                           np.where(mask.unknown | (labels == 0) | (labels == 255),
                                    np.float32(0), np.float32(.42)))
        parts = _overlay(editor.rgb, tint, opacity)
        protected = mask.soft_protection > 0
        if protected.any():
            parts[protected] = _overlay(parts[protected], np.array([220, 76, 106], np.uint8),
                                        mask.soft_protection[protected] * .55)
        images['parts.png'] = _png(parts, editor.alpha, session.icc)
    if same_unknown:
        unknown = old_images.get('unknown.png')
    else:
        unknown = _unknown_overlay(mask.unknown, editor.alpha)
    if unknown is not None:
        images['unknown.png'] = unknown
    if old_mask is not None and np.array_equal(old_mask.weight, mask.weight):
        images['weight.png'] = old_images['weight.png']
    else:
        weights = _overlay(editor.rgb, np.array([79, 159, 232], np.uint8), mask.weight * .55)
        images['weight.png'] = _png(weights, editor.alpha, session.icc)
    return images


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
    result['part_strengths'] = {name: validate_unit_interval(value, name) for name, value in part.items()}
    if isinstance(result['method'], str) and result['method'] in _RETIRED_METHODS:
        raise ValueError('项目已移除 CUGAN，请选择其他降噪方法。')
    result['method'], result['l0'] = resolve_denoiser(result['method'], _radius(result['l0'], 'L0', True))
    if result['method'] not in WORKBENCH_METHODS:
        raise ValueError('请选择工作台支持的降噪方法。')
    for name in ('strength', 'structure_protection', 'restoration_gain', 'downscale'):
        result[name] = validate_unit_interval(result[name], name)
    result['sigma'] = _radius(result['sigma'], 'sigma', True)
    if not result['downscale'] or result['sigma'] > 256:
        raise ValueError('缩图比例须大于零，sigma 不能超过 256。')
    return result


def _history(edits, cursor, width, height, stored_edits):
    if (not isinstance(edits, list) or len(edits) > 2000 or type(cursor) is not int
            or not 0 <= cursor <= len(edits)):
        raise ValueError('编辑记录无效或已超过上限。')
    count = 0
    for index, edit in enumerate(edits):
        if isinstance(edit, dict) and edit.get('tool') in ('import', 'sam_object'):
            token = edit.get('token')
            if (set(edit) != {'tool', 'token'} or not isinstance(token, str)
                    or (edit['tool'], token) not in stored_edits):
                raise ValueError('蒙版编辑记录必须来自当前文档。')
            continue
        if not isinstance(edit, dict) or set(edit) - {'tool', 'shape', 'points', 'radius', 'strength', 'module', 'allow_protected'}:
            raise ValueError('编辑记录无效。')
        tool = edit.get('tool')
        if tool == 'resolve_unknown':
            from .masks import _resolution_module

            _resolution_module(edit, edits[index - 1] if index else None)
            continue
        if 'allow_protected' in edit and (tool != 'include' or type(edit['allow_protected']) is not bool):
            raise ValueError('只有补选可明确允许使用自动保护的区域。')
        if tool in ('invert', 'reset'):
            continue
        shape = edit.get('shape', 'brush')
        points = edit.get('points')
        if (tool not in ('include', 'exclude', 'protect', 'mix', 'erase', 'override') or shape not in ('brush', 'polygon')
                or not isinstance(points, list) or len(points) < (3 if shape == 'polygon' else 1)):
            raise ValueError('笔画或套索记录无效。')
        count += len(points)
        if count > 50000:
            raise ValueError('笔画采样点超过上限。')
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError('笔画坐标无效。')
            if _radius(point[0], 'x') >= width or _radius(point[1], 'y') >= height:
                raise ValueError('笔画坐标超出原图。')
        if shape == 'brush' and _radius(edit.get('radius', 24), 'radius', True) > 256:
            raise ValueError('笔刷半径不能超过 256。')
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


def _write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.write-' + uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
    """Source-bound drafts, with one resident analysis and one serialized render worker."""

    def __init__(self, workspace, opener):
        self.workspace = Path(workspace).resolve()
        self.opener = opener
        self.root = self.workspace / 'documents'
        self._condition = Condition()
        self._inference = Lock()
        self._magnetic = Lock()
        self._magnetic_cache = None
        self._changes_lock = Lock()
        self._changes_cache = None
        self._changes_baseline = None
        self._thumbnail_lock = Lock()
        self._thumbnails = OrderedDict()
        self._documents = {}
        self._pending = OrderedDict()
        self._object_previews = OrderedDict()
        self._closed = False
        self._resident_id = None
        self._resident_epoch = None
        self._resident = None
        self._has_suspected_objects = False
        self._weight_cache = None
        self._selection_key = None
        self._selection_value = None
        self._restored_key = None
        self._restored = None
        self._restore()
        self._worker = Thread(target=self._run, name='gaid-workbench', daemon=True)
        self._worker.start()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=2)

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
        _json(self.root / state['id'] / 'draft.json', state)
        self._documents[state['id']] = state
        preview = self._object_previews.get(state['id'])
        if preview is not None and preview[0] != state['revision']:
            self._object_previews.pop(state['id'])

    def _queue(self, state):
        self._pending[state['id']] = state['revision']
        self._condition.notify()

    def _lookup(self, doc, *, upgrade=True):
        if not isinstance(doc, str) or not _DOCUMENT_ID.fullmatch(doc) or doc not in self._documents:
            raise KeyError('找不到此文档。')
        state = self._documents[doc]
        if upgrade and self._analysis_outdated(state):
            return self._restart_analysis(state, upgrade=True)
        return state

    @staticmethod
    def _analysis_outdated(state):
        return state['status'] == 'ready' and state.get('_analysis_version') != _ANALYSIS_VERSION

    def _restart_analysis(self, state, *, upgrade):
        # Leave earlier previews available while the new revision is running.
        updated = {**state, 'revision': state['revision'] + 1, 'status': 'queued',
                   'phase': 'queued', 'error': None, '_source_epoch': state.get('_source_epoch', 0) + 1,
                   'selection_revision': state['selection_revision'] + int(upgrade)}
        self._store(updated)
        self._queue(updated)
        return updated

    @staticmethod
    def _public(state):
        return deepcopy({key: value for key, value in state.items() if not key.startswith('_')})

    def get(self, doc):
        with self._condition:
            return self._public(self._lookup(doc))

    def recent(self):
        with self._condition:
            states = sorted(self._documents.values(), key=lambda state: state['_updated'], reverse=True)
            return [{key: state[key] for key in ('id', 'name', 'width', 'height', 'status')}
                    for state in states[:20]]

    def thumbnail(self, doc):
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
        upgrade = self._analysis_outdated(state)
        if repaired or state['status'] == 'error' or upgrade:
            updated = self._restart_analysis(updated, upgrade=upgrade)
        else:
            self._store(updated)
        return self._public(updated)

    def open(self, content, name):
        if not isinstance(content, bytes) or not content or len(content) > 64 * 1024 * 1024:
            raise ValueError('请选择不超过 64 MiB 的图片。')
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
            state = {'id': doc, 'name': (name.replace('\\', '/').split('/')[-1] or 'image' + suffix)[:160],
                     'width': width, 'height': height, 'revision': 0, 'settings': deepcopy(DEFAULTS),
                     'edits': [], 'cursor': 0, 'status': 'queued', 'phase': 'queued', 'error': None,
                     'b': None, 'a': None, 'selection_revision': 0,
                     '_schema': 1, '_updated': time(), '_suffix': suffix, '_correction': None, '_source_epoch': 0}
            self._store(state)
            self._queue(state)
            return self._public(state)

    @staticmethod
    def _current(state, revision):
        if type(revision) is not int or revision != state['revision']:
            raise RevisionConflict('文档已在另一页面更新。请重新载入后继续。')

    def update(self, doc, payload):
        if not isinstance(payload, dict) or set(payload) != {'revision', 'settings', 'edits', 'cursor'}:
            raise ValueError('更新必须包含修订号、设置和编辑记录。')
        with self._condition:
            state = self._lookup(doc)
            self._current(state, payload['revision'])
            settings = _settings(payload['settings'])
            stored_edits = {(edit['tool'], edit['token']) for edit in state['edits']
                            if edit['tool'] in ('import', 'sam_object')}
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
            state = self._lookup(doc)
            if type(revision) is not int or not state['b'] or state['b']['revision'] != revision:
                raise RevisionConflict('可固定的结果已更新。请重新选择。')
            pinned = {**state, 'a': deepcopy(state['b']), '_updated': time()}
            self._store(pinned)
            self._prune(state, pinned)
            return self._public(pinned)

    def _render_path(self, doc, revision):
        return self.root / doc / f'render-{revision}'

    def _changes_state(self, doc, revision):
        state = self._lookup(doc, upgrade=False)
        self._current(state, revision)
        if (state['status'] != 'ready' or not state['b'] or state['b']['revision'] != revision
                or self._analysis_outdated(state)):
            raise RevisionConflict('当前成品尚未更新，请稍后查看手动改动。')
        return state

    def changes(self, doc, revision):
        with self._condition:
            state = self._changes_state(doc, revision)
            key = (doc, revision)
            if self._changes_cache is not None and self._changes_cache[0] == key:
                return self._changes_cache[1]
            if not state['cursor'] and not state.get('_correction'):
                return {'revision': revision, 'counts': dict.fromkeys(_CHANGE_KINDS, 0), 'overlay': None}

        with self._changes_lock:
            with self._condition:
                self._changes_state(doc, revision)
                if self._changes_cache is not None and self._changes_cache[0] == key:
                    return self._changes_cache[1]
            editor, mask, lines = self._changes_selection(doc, revision, state)
            settings = state['settings']
            baseline_key = (doc, state.get('_source_epoch', 0), tuple(settings['modules']),
                            tuple(sorted(settings['part_strengths'].items())), settings['strength'],
                            settings['structure_protection'])
            if self._changes_baseline is None or self._changes_baseline[0] != baseline_key:
                from .masks import WorkbenchWeightCache

                cache = WorkbenchWeightCache(editor)
                cache.lines = lines
                weight = _selection(editor, settings, [], 0, weight_cache=cache).weight
                self._changes_baseline = (baseline_key, weight)
            counts, overlay = _changes_overlay(editor, self._changes_baseline[1], mask)
            result = {'revision': revision, 'counts': counts, 'overlay': overlay}
            with self._condition:
                self._changes_state(doc, revision)
                self._changes_cache = (key, result)
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

        from .masks import WorkbenchWeightCache
        from .session import _cached_editor

        workspace = self.workspace / 'cases'
        try:
            editor = _cached_editor(workspace, doc, [state['width'], state['height']])
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            # Custom openers and missing caches still use the verified session
            # path. Only the analysis, if needed, holds the inference lock.
            if not self._inference.acquire(blocking=False):
                raise RevisionConflict('降噪工作台正在处理图片，请稍后查看。')
            try:
                with self._condition:
                    self._changes_state(doc, revision)
                source = self._source_path(doc, state['_suffix'])
                if _sha(source) != doc:
                    raise ValueError('本地原图副本已更改，请重新打开源图。')
                editor = self.opener(source, workspace).editor
            finally:
                self._inference.release()
        cache = WorkbenchWeightCache(editor)
        return editor, self._replay_selection(doc, state, editor, cache), cache.lines

    def asset(self, doc, view, *, download=False, revision=None):
        if view == 'magnetic_source' and not download:
            return self._magnetic_source(doc, revision)
        with self._condition:
            state = self._lookup(doc)
            if view == 'original' and not download:
                return (self.root / doc / 'original.png').read_bytes()
            if view not in ('a', 'b', 'parts', 'weight', 'unknown', 'suspected_objects') or (download and view not in ('a', 'b')):
                raise KeyError('找不到此视图。')
            rendered = state['a' if view == 'a' else 'b']
            if (not rendered or (revision is not None and rendered['revision'] != revision)
                    or (download and view == 'b' and rendered['revision'] != state['revision'])):
                raise RevisionConflict('请求的结果尚未完成或已更新。')
            if view == 'suspected_objects':
                if not rendered.get('images', {}).get(view):
                    raise KeyError('此版本没有疑似外物图层。')
                version = state.get('_analysis_version')
                filename = f'suspected-objects-{version}.png' if type(version) is int else _SUSPECTED_ASSET
                return (self.root / doc / filename).read_bytes()
            if view == 'unknown' and not rendered.get('images', {}).get(view):
                raise KeyError('此版本没有待确认图层。')
            filename = view + '.png' if view in ('parts', 'weight', 'unknown') else 'result.png'
            return (self._render_path(doc, rendered['revision']) / filename).read_bytes()

    def _magnetic_state(self, doc, revision):
        state = self._lookup(doc)
        self._current(state, revision)
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
                rgb, alpha, icc, candidate = _cached_l0_baseline(
                    self.workspace / 'cases', doc, [state['width'], state['height']], settings['l0'])
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
                session = self.opener(source, self.workspace / 'cases')
            candidate = session.candidate(settings['method'], lam=settings['l0'], downscale=settings['downscale'])
        finally:
            self._inference.release()
        return _restore_candidate(session.editor.rgb, candidate, settings), session.editor.alpha, session.icc

    def corrections(self, doc, revision):
        with self._condition:
            state = self._lookup(doc)
            self._current(state, revision)
            if not state['b'] or state['b']['revision'] != revision:
                raise RevisionConflict('请等待当前选区完成。')
            path = self._render_path(doc, revision) / 'corrections.json'
            if not path.is_file():
                raise ValueError('当前软保护或部位强度不能导出为旧版二值校正。已保存在工作台文档中。')
            record = json.loads(path.read_text(encoding='utf-8'))
            filenames = [record[key] for key in MASK_KEYS if key in record]
            filenames.extend(record.get('module_overrides', {}).values())
            folder = self.workspace / 'corrections' / doc / f'revision-{revision}'
            folder.mkdir(parents=True, exist_ok=True)
            for name in [*filenames, path.name]:
                target = folder / name
                content = (path.parent / name).read_bytes()
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
            width, height = state['width'], state['height']
            included = _sam_points(positive, width, height)
            excluded = _sam_points(negative, width, height)
            if not included or included & excluded:
                raise ValueError('请在外物上点至少一个加入点，加入点和排除点不能重叠。')
            if state['status'] != 'ready':
                raise RevisionConflict('请等待当前结果更新后再点选外物。')
        # The render worker and HTTP requests share the GPU. State queries and
        # edits remain available while inference is in progress.
        with self._inference:
            with self._condition:
                state = self._lookup(doc)
                self._current(state, revision)
                self._object_previews.pop(doc, None)
                source = self._source_path(doc, state['_suffix'])
            if _sha(source) != doc:
                raise ValueError('本地原图副本已更改，请重新打开源图。')
            session = self._resident
            if self._resident_id != doc or self._resident_epoch != state.get('_source_epoch', 0):
                session = self.opener(source, self.workspace / 'cases')
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

    def apply_object(self, doc, revision, token):
        with self._condition:
            state = self._lookup(doc)
            self._current(state, revision)
            pending = self._object_previews.get(doc)
            if pending is None or pending[:2] != (revision, token):
                raise RevisionConflict('点选预览已过期，请重新生成。')
            if state['cursor'] >= 2000:
                raise ValueError('编辑记录已超过上限。')
            path = self.root / doc / token / 'object_add_mask.png'
            _write_atomic(path, pending[2])
            edits = [*state['edits'][:state['cursor']], {'tool': 'sam_object', 'token': token}]
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
                self._condition.wait_for(lambda: self._closed or bool(self._pending))
                if self._closed:
                    return
                doc, revision = self._pending.popitem(last=False)
                state = deepcopy(self._documents[doc])
            try:
                with self._inference:
                    self._render(doc, revision, state)
            except Exception as error:
                with self._condition:
                    current = self._documents[doc]
                    if not self._closed and current['revision'] == revision:
                        failed = {**current, 'status': 'error', 'phase': 'error',
                                  'error': str(error) or type(error).__name__}
                        try:
                            self._store(failed)
                        except OSError:
                            current.update(failed)

    @staticmethod
    def _selection_cache_key(state):
        settings = state['settings']
        return (state['selection_revision'], settings['strength'], settings['structure_protection'])

    def _replay_selection(self, doc, state, editor, cache):
        token, active = _active_edit_layers(state)
        correction = (load_edits(self.root / doc / token / 'corrections.json', editor.labels.shape, doc)
                      if token is not None else None)
        return _selection(editor, state['settings'], active, len(active), correction=correction,
                          object_mask=lambda token: self._saved_object_mask(doc, token, editor.labels.shape),
                          weight_cache=cache)

    def _render(self, doc, revision, state):
        epoch = state.get('_source_epoch', 0)
        needs_analysis = self._resident_id != doc or self._resident_epoch != epoch
        settings = state['settings']
        restore_key = _restoration_key(settings)
        needs_candidate = needs_analysis or restore_key != self._restored_key
        phase = 'analysis' if needs_analysis else 'candidate' if needs_candidate else 'compose'
        if not self._stage(doc, revision, phase):
            return
        if needs_analysis:
            from .masks import WorkbenchWeightCache

            self._resident_id = None
            self._resident = None
            self._weight_cache = None
            self._selection_key = self._selection_value = None
            self._restored_key = self._restored = None
            source = self._source_path(doc, state['_suffix'])
            if _sha(source) != doc:
                raise ValueError('本地原图副本已更改，请重新打开源图。')
            self._resident = self.opener(source, self.workspace / 'cases')
            suspected = getattr(self._resident, 'suspected_objects', None)
            self._has_suspected_objects = suspected is not None and suspected.any()
            if self._has_suspected_objects:
                overlay_path = self.root / doc / _SUSPECTED_ASSET
                if not overlay_path.is_file():
                    editor = self._resident.editor
                    rgb = editor.rgb.copy()
                    rgb[suspected] = _overlay(rgb[suspected], np.array([243, 194, 87], np.uint8), .56)
                    _write_atomic(overlay_path, _png(rgb, editor.alpha, self._resident.icc))
            self._weight_cache = WorkbenchWeightCache(self._resident.editor)
            self._resident_id = doc
            self._resident_epoch = epoch
        session = self._resident
        if needs_analysis and not self._stage(doc, revision, 'candidate'):
            return
        if needs_candidate:
            candidate = session.candidate(settings['method'], lam=settings['l0'], downscale=settings['downscale'])
            if not self._stage(doc, revision, 'compose'):
                return
            self._restored = _restore_candidate(session.editor.rgb, candidate, settings)
            self._restored_key = restore_key
        selection_key = self._selection_cache_key(state)
        if selection_key != self._selection_key:
            mask = self._replay_selection(doc, state, session.editor, self._weight_cache)
            self._selection_value = mask, _selection_images(session, mask, previous=self._selection_value)
            self._selection_key = selection_key
        mask, images = self._selection_value
        result = blend(session.editor.rgb, self._restored, mask.weight)
        if not self._stage(doc, revision, 'encoding'):
            return
        data = {**images, 'result.png': _png(result, session.editor.alpha, session.icc)}
        self._encode_corrections(data, state, session.editor, mask)
        with self._condition:
            current = self._documents[doc]
            if self._closed or current['revision'] != revision:
                return
            folder = self._render_path(doc, revision)
            folder.mkdir(exist_ok=True)
            for name, content in data.items():
                (folder / name).write_bytes(content)
            rendered = {'revision': revision, 'method': settings['method'],
                        'selection_revision': state['selection_revision']}
            image_views = {name: True for name, available in (
                ('suspected_objects', self._has_suspected_objects), ('unknown', 'unknown.png' in images)) if available}
            if image_views:
                rendered['images'] = image_views
            ready = {**current, 'b': rendered, '_analysis_version': _ANALYSIS_VERSION,
                     'status': 'ready', 'phase': 'done', 'error': None,
                     'edit_feedback': mask.edit_feedback,
                     'active_percent': round(float((mask.weight > 0).mean() * 100), 2)}
            self._store(ready)
            self._prune(current, ready)

    @staticmethod
    def _encode_corrections(data, state, editor, mask):
        from .masks import binary_corrections

        try:
            masks = binary_corrections(editor, mask, state['settings'])
        except ValueError:
            return
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
        keep = {item['revision'] for item in (current['a'], current['b']) if item}
        generated = ('result.png', 'parts.png', 'weight.png', 'unknown.png', 'corrections.json',
                     *(key + '.png' for key in MASK_KEYS),
                     *('module-' + part + '.png' for part in MODULES if part != 'background'))
        for revision in {item['revision'] for item in (previous['a'], previous['b']) if item} - keep:
            folder = self._render_path(current['id'], revision)
            if folder.is_symlink() or folder.is_junction():
                continue
            try:
                for filename in generated:
                    (folder / filename).unlink(missing_ok=True)
                folder.rmdir()
            except OSError:
                pass  # Output already published; stale files are harmless after failed cleanup.
