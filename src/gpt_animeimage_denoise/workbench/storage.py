"""Machine-local preferences and the workspace's rebuildable analysis files."""
from contextlib import contextmanager
from copy import deepcopy
import errno
import hashlib
import json
from numbers import Real
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
from tempfile import NamedTemporaryFile
from time import sleep


class StorageBusy(ValueError):
    """Another workbench is using this workspace's analysis cache."""


@contextmanager
def _staged_file(directory, prefix, write):
    stream = NamedTemporaryFile(dir=directory, prefix=prefix, suffix='.tmp', delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _staged_file(path.parent, '.gaid-write-', lambda stream: stream.write(content)) as temporary:
        temporary.replace(path)


def _replace_json(path, value):
    write_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


@contextmanager
def _file_lock(path, *, wait=True):
    """OS releases the byte lock on exit, including after a process crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            def acquire():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            def acquire():
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                acquire()
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if not wait:
                    raise StorageBusy('缓存正在使用，请等待工作台空闲后再试。') from error
                sleep(.05)
        yield


def _folder(value):
    if value == '':
        return ''
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError('请选择本机的绝对目录。')
    folder = Path(value).resolve()
    if not folder.is_dir():
        raise ValueError('所选目录不存在或不是文件夹。')
    return str(folder)


class Preferences:
    def __init__(self, defaults, validate):
        self.validate = validate
        self.initial = {'defaults': defaults, 'export_dir': '', 'cache_dir': '', 'theme': 'system',
                        'remember_brush': False, 'brush_radius': 24, 'brush_feather': 0,
                        'exit_on_close': False}
        configured = os.environ.get('GAID_WORKBENCH_SETTINGS')
        local = os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config'
        self.path = Path(configured).resolve() if configured else Path(local) / 'GAID' / 'workbench' / 'settings.json'

    def read(self):
        try:
            value = json.loads(self.path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return deepcopy(self.initial)
        if not isinstance(value, dict):
            raise ValueError('工作台设置文件格式无效。')
        result = {**deepcopy(self.initial), **value}
        result['defaults'] = self.validate(result['defaults'])
        return result

    def change(self, patch, cache, maintenance):
        if not isinstance(patch, dict) or patch.keys() - self.initial.keys():
            raise ValueError('工作台设置字段无效。')
        with _file_lock(self.path.with_name(self.path.name + '.lock')):
            before = self.read()
            result = {**before, **patch}
            if 'defaults' in patch:
                if not isinstance(patch['defaults'], dict):
                    raise ValueError('新图默认设置必须是对象。')
                result['defaults'] = self.validate({**before['defaults'], **patch['defaults']})
            if result['theme'] not in ('system', 'light', 'dark') or type(result['remember_brush']) is not bool:
                raise ValueError('外观或笔刷记忆设置无效。')
            if type(result['exit_on_close']) is not bool:
                raise ValueError('关闭页面时的退出设置无效。')
            radius, feather = result['brush_radius'], result['brush_feather']
            if (type(radius) is not int or not 1 <= radius <= 256 or not isinstance(feather, Real)
                    or isinstance(feather, bool) or not 0 <= feather <= radius):
                raise ValueError('笔刷半径须为 1 至 256 的整数，羽化不能超过半径。')
            for key in ('export_dir', 'cache_dir'):
                if key in patch:
                    result[key] = _folder(patch[key])
                    if result[key]:
                        cache.validate_base(Path(result[key]))
            if result['cache_dir'] != before['cache_dir']:
                with maintenance(), cache.lock(wait=False):
                    cache.switch(result['cache_dir'], lambda: _replace_json(self.path, result))
            else:
                _replace_json(self.path, result)
            return result


def _linked(path):
    return path.is_symlink() or path.is_junction()


def _check_tree(path, root):
    """Refuse redirected children even when their target is another managed file."""
    current = path
    while current != root:
        if _linked(current) or current == current.parent:
            raise ValueError('缓存目录中存在链接或目录越界。')
        current = current.parent
    if _linked(root):
        raise ValueError('缓存目录不能是链接。')


_DIGEST = re.compile(r'[a-f0-9]{64}')
_FLOAT_CACHE = re.compile(r'(?:candidate|l0|evidence|sam|sam-roi|sr)-[a-f0-9]{64}\.(?:npz|tmp)')
_OWNER = '.gaid-workbench-case'


class CaseCache:
    def __init__(self, workspace, preferences):
        self.workspace = workspace
        self.preferences = preferences
        self.pointer = workspace / '.gaid-cache-location.json'
        self.namespace = hashlib.sha256(os.path.normcase(str(workspace)).encode()).hexdigest()[:24]

    def lock(self, *, wait=True):
        return _file_lock(self.workspace / '.gaid-cache.lock', wait=wait)

    def processing(self, *, wait=True):
        return _file_lock(self.workspace / '.gaid-analysis.lock', wait=wait)

    def validate_base(self, path):
        managed = [self.workspace / key for key in ('documents', 'inputs', 'corrections', 'cases')]
        if any(path.is_relative_to(root) for root in managed) or 'gaid-workbench-cache' in map(str.lower, path.parts):
            raise ValueError('请选择工作台草稿、原图和分析缓存之外的目录。')

    def _path(self, base):
        if not base:
            path = self.workspace / 'cases'
            _check_tree(path, self.workspace)
            return path
        parent = Path(base)
        if not parent.is_absolute() or parent.resolve() != parent:
            raise ValueError('缓存设置中的目录无效或已重定向。')
        self.validate_base(parent)
        path = parent / 'GAID-workbench-cache' / self.namespace / 'cases'
        _check_tree(path, parent)
        return path

    def _location(self):
        try:
            return json.loads(self.pointer.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return {'base': ''}

    def claim(self, root, digest, *, rendering=False):
        case = root / digest
        _check_tree(case, root)
        marker = case / _OWNER
        if case.exists() and any(case.iterdir()) and not self._owned(case):
            if not rendering or not self._interrupted_legacy_case(case):
                raise ValueError('分析缓存目标含有其他文件。请选择其他缓存目录。')
        if rendering and ((case / 'analysis').exists() or (case / '.cache').exists()
                          or _linked(case / 'analysis') or _linked(case / '.cache')):
            for path in self._case_files(case, root):
                if path.stat().st_nlink > 1:
                    raise ValueError('分析缓存文件与其他文件共用硬链接，请选择新的缓存目录。')
        if not marker.exists():
            case.mkdir(parents=True, exist_ok=True)
            _replace_json(marker, self._owner(digest))

    def _interrupted_legacy_case(self, case):
        # Rendering has verified the source digest. Older workbenches could
        # leave generated files in the default case before publishing a report.
        marker = case / _OWNER
        if (case.parent != self.workspace / 'cases' or marker.exists() or _linked(marker)
                or (case / 'analysis' / 'report.json').exists()):
            return False
        draft = self.workspace / 'documents' / case.name / 'draft.json'
        _check_tree(draft, self.workspace)
        try:
            state = json.loads(draft.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False
        if not isinstance(state, dict) or state.get('id') != case.name or state.get('_schema') != 1:
            return False
        if any(self._case_files(case, case.parent)):
            return True
        return any(folder.is_dir() and not any(folder.iterdir())
                   for folder in (case / '.cache', case / 'analysis'))

    def _owner(self, digest):
        return {'workspace': self.namespace, 'source': digest}

    def _owned(self, case):
        if not _DIGEST.fullmatch(case.name) or not case.is_dir() or _linked(case):
            return False
        marker = case / _OWNER
        if marker.exists():
            try:
                return not _linked(marker) and json.loads(marker.read_text(encoding='utf-8')) == self._owner(case.name)
            except (OSError, ValueError):
                return False
        # Older workbenches wrote no marker. Require a matching local draft and fusion report.
        report = case / 'analysis' / 'report.json'
        _check_tree(report, case)
        if not (self.workspace / 'documents' / case.name / 'draft.json').is_file() or not report.is_file():
            return False
        try:
            record = json.loads(report.read_text(encoding='utf-8'))
            return (record.get('source_sha256') == case.name and record.get('profile') == 'fusion'
                    and record.get('folder') == 'analysis' and not record.get('edits'))
        except (OSError, ValueError, AttributeError):
            return False

    def _files(self, root):
        for _, files in self._cases(root):
            yield from files

    def _cases(self, root):
        if not root.is_dir():
            return
        for case in root.iterdir():
            if self._owned(case):
                yield case, set(self._case_files(case, root))

    @staticmethod
    def _case_files(case, root):
        from ..fusion_pipeline import OUTPUT_FILES

        for name in (*OUTPUT_FILES, 'report.json', 'report.json.tmp', 'regions.json.tmp'):
            path = case / 'analysis' / name
            _check_tree(path, root)
            if path.is_file():
                yield path
        floats = case / '.cache'
        _check_tree(floats, root)
        if floats.is_dir():
            for path in floats.iterdir():
                if _FLOAT_CACHE.fullmatch(path.name):
                    _check_tree(path, root)
                    if path.is_file():
                        yield path

    @staticmethod
    def _intermediates(case, files, *, verify=False):
        from ..fusion_pipeline import WORKBENCH_FILES
        from ..pipeline import _digest, _sha

        analysis = case / 'analysis'
        report_path = analysis / 'report.json'
        if report_path not in files:
            return set()
        try:
            report = json.loads(report_path.read_text(encoding='utf-8'))
            hashes = report['output_hashes']
            identity = report['candidate_cache_identity']
            stage = report['candidate_cache_stage']
            if (report['status'] != 'complete' or report['profile'] != 'fusion'
                    or report['folder'] != 'analysis' or report['source_sha256'] != case.name or report['edits']
                    or stage not in ('l0', 'candidate') or identity['source_sha256'] != case.name):
                return set()
            required = {'person_labels.png', 'unknown.png', 'objects.png'}
            grid = {analysis / name: hashes[name] for name in (*WORKBENCH_FILES, 'original.png')
                    if name in required or name in hashes}
            candidate = case / '.cache' / (stage + '-' + _digest(identity) + '.npz')
            keep = {report_path, candidate, *grid}
            if not keep <= files:
                return set()
            obsolete = files - keep
            if obsolete and verify and any(_sha(path) != digest for path, digest in grid.items()):
                return set()
            return obsolete
        except (OSError, KeyError, ValueError, TypeError):
            return set()

    def info(self):
        root = self._path(self._location()['base'])
        size = reclaimable = 0
        for case, files in self._cases(root):
            obsolete = self._intermediates(case, files)
            for path in files:
                try:
                    length = path.stat().st_size
                except FileNotFoundError:
                    continue
                size += length
                if path in obsolete:
                    reclaimable += length
        return {'path': str(root), 'bytes': size, 'reclaimable_bytes': reclaimable}

    def _remove_files(self, root, files):
        # Never recurse into a chosen directory. Only remove known generated files.
        parents = set()
        released = 0
        for digest in {path.relative_to(root).parts[0] for path in files}:
            self.claim(root, digest)
        for path in files:
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
            released += size
            parents.add(path.parent)
        for folder in sorted(parents, key=lambda path: len(path.parts), reverse=True):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        return released

    def _remove(self, root):
        return self._remove_files(root, list(self._files(root)))

    def clear(self, *, mode='all'):
        if mode not in ('compact', 'all'):
            raise ValueError('缓存清理模式无效。')
        with self.lock(wait=False):
            root = self.switch(self.preferences.read()['cache_dir'])
            if mode == 'compact':
                files = [path for case, files in self._cases(root) for path in self._intermediates(case, files, verify=True)]
                released = self._remove_files(root, files)
            else:
                released = self._remove(root)
            return {**self.info(), 'released_bytes': released}

    def switch(self, base, commit=None):
        current = self._location()
        previous = current.get('previous')
        if previous is not None:
            try:
                self._remove(self._path(previous))
            except OSError:
                if current['base'] != base:
                    raise
                # A locked obsolete file must not prevent use of the completed copy.
            else:
                _replace_json(self.pointer, {'base': current['base']})
        source = self._path(current['base'])
        target = self._path(base)
        if source != target:
            if base and not Path(base).is_dir():
                raise ValueError('缓存目录不可用，请重新选择。')
            files = list(self._files(source))
            for case in {path.relative_to(source).parts[0] for path in files}:
                self.claim(target, case)
            for path in files:
                dest = target / path.relative_to(source)
                _check_tree(dest, target)
                if dest.is_file():
                    from ..pipeline import _sha
                    if _sha(dest) == _sha(path):
                        continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as source_stream:
                    def copy(stream):
                        shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
                    with _staged_file(dest.parent, '.gaid-copy-', copy) as temporary:
                        # Replace only generated paths in the verified, owned case.
                        # Retrying remains valid even if the source has since been re-rendered.
                        temporary.replace(dest)
        if commit is not None:
            commit()
        if source != target:
            # If interrupted here, retrying the copy compares existing files byte-for-byte.
            _replace_json(self.pointer, {'base': base, 'previous': current['base']})
            try:
                self._remove(source)
            except OSError:
                pass  # Retry cleanup at next access with the saved source pointer.
            else:
                _replace_json(self.pointer, {'base': base})
        return target

    @contextmanager
    def using(self, digest=None):
        with self.lock():
            root = self.switch(self.preferences.read()['cache_dir'])
            if digest is not None:
                self.claim(root, digest, rendering=True)
            yield root


def _publish(temporary, target):
    if os.name == 'nt':
        # On Windows rename fails if the destination exists, including on FAT/exFAT.
        temporary.rename(target)
    else:
        os.link(temporary, target)


def export_png(folder, name, data):
    folder = Path(folder)
    if folder.resolve() != folder or not folder.is_dir():
        raise ValueError('导出目录已移动或重定向，请重新选择。')
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', PureWindowsPath(name).stem).strip(' .')[:120] or 'image'
    if PureWindowsPath(stem).is_reserved():
        stem = '_' + stem
    with _staged_file(folder, '.gaid-export-', lambda stream: stream.write(data)) as temporary:
        number = 1
        while True:
            suffix = '' if number == 1 else '-' + str(number)
            target = folder / f'{stem}-gaid{suffix}.png'
            if target.is_dir():
                number += 1
                continue
            try:
                _publish(temporary, target)
                return target
            except FileExistsError:
                number += 1
