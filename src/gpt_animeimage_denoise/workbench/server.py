"""Loopback HTTP transport for the image workbench."""
from base64 import b64decode
from contextlib import closing
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from subprocess import SubprocessError
from threading import Lock, Thread, Timer, current_thread
from time import monotonic, sleep
import traceback
from urllib.parse import parse_qs, quote, unquote, urlsplit
from uuid import uuid4

from ..denoisers import uses_l0
from ..editor import DEFAULT_MODULES, MODULES
from .documents import PART_PALETTE, WORKBENCH_METHODS
from .file_manager import reveal_file
from .storage import StorageBusy, _file_lock, _replace_json


_WEB = Path(__file__).with_name('web')
_STATIC = {'/': ('index.html', 'text/html'), '/style.css': ('style.css', 'text/css'),
           '/app.js': ('app.js', 'text/javascript'),
           '/tools.js': ('tools.js', 'text/javascript'),
           '/settings.js': ('settings.js', 'text/javascript'), '/settings.css': ('settings.css', 'text/css'),
           '/magnetic-lasso-worker.js': ('magnetic-lasso-worker.js', 'text/javascript')}
_PART_LABELS = {'skin': '皮肤', 'face': '面部', 'hair': '头发', 'eyes': '眼睛',
                'mouth': '嘴巴', 'outfit': '服饰', 'background': '背景'}
_DEFAULTS = {'method': 'l0', 'l0': .0005, 'downscale': .6, 'strength': 1,
             'restoration_gain': .25, 'sigma': 12, 'structure_protection': .2}
_DOCUMENT = re.compile(r'/api/documents/([0-9a-f]{64})(?:/(.*))?')
_EXPORT_ID = re.compile(r'[0-9a-f]{32}')
_EXPORT_HISTORY_LIMIT = 128
_PAGE_ID = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')
_PAGE_CLOSE_GRACE = 4
_FOLDER_DIALOG = Lock()


class _RequestError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class _PagePresence:
    def __init__(self, server):
        self._server = server
        self._lock = Lock()
        self._active = set()
        self._departed = set()
        self._timer = None

    def update(self, identifier, opened):
        with self._lock:
            if identifier in self._departed:
                return  # A delayed open cannot revive a page that already departed.
            if opened:
                self._active.add(identifier)
                self._cancel()
                return
            self._active.discard(identifier)
            self._departed.add(identifier)
            if not self._active and self._timer is None and self._enabled():
                self._timer = Timer(_PAGE_CLOSE_GRACE, self._exit_if_empty)
                self._timer.daemon = True
                self._timer.start()

    def _enabled(self):
        return self._server.documents.preferences.read()['exit_on_close'] is True

    def _exit_if_empty(self):
        with self._lock:
            if self._timer is not current_thread():
                return
            self._timer = None
            if self._active or not self._enabled():
                return
        self._server.shutdown()

    def _cancel(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def close(self):
        with self._lock:
            self._cancel()


def _choose_folder(initial):
    if not isinstance(initial, str) or (initial and not Path(initial).is_absolute()):
        raise ValueError('请选择有效的起始目录。')
    import tkinter
    from tkinter.filedialog import askdirectory

    with _FOLDER_DIALOG:
        root = tkinter.Tk()
        try:
            root.withdraw()
            root.attributes('-topmost', True)
            selected = askdirectory(parent=root, title='选择工作台目录', mustexist=True,
                                    initialdir=initial if initial and Path(initial).is_dir() else str(Path.home()))
            return str(Path(selected).resolve()) if selected else None
        finally:
            root.destroy()


def _options():
    return {'modules': [{'id': name, 'label': _PART_LABELS[name], 'color': '#' + bytes(PART_PALETTE[name]).hex()}
                        for name in MODULES],
            'default_modules': list(DEFAULT_MODULES), 'defaults': dict(_DEFAULTS),
            'methods': [{'id': name, 'label': label, 'uses_l0': uses_l0(name),
                         'uses_downscale': name.startswith('esrgan')}
                        for name, label in WORKBENCH_METHODS.items()]}


def _revision(value):
    revision = value.get('revision')
    if type(revision) is not int or revision < 0:
        raise ValueError('请求缺少有效的编辑版本。')
    return revision


def _correction_files(value):
    files = value.get('files')
    record = value.get('record')
    if not isinstance(record, dict) or not isinstance(files, dict) or len(files) > 10:
        raise ValueError('请选择校正 JSON 及配套的 PNG 蒙版。')
    decoded = {}
    for name, data in files.items():
        if (not name or '/' in name or '\\' in name or ':' in name or not name.lower().endswith('.png')
                or not isinstance(data, str)):
            raise ValueError('校正蒙版必须为同目录下的 PNG 文件。')
        decoded[name] = b64decode(data, validate=True)
    return record, decoded


def create_server(workspace, *, port=0):
    """Bind the workbench to IPv4 loopback only."""
    from .documents import Documents, RevisionConflict

    # Keep the frontend paired with the Python code loaded by this process.
    assets = {route: ((_WEB / name).read_bytes(), mime) for route, (name, mime) in _STATIC.items()}
    exports = {}
    export_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_request(self, code='-', size='-'):
            if code in (200, 304) and self.command == 'GET' and _DOCUMENT.fullmatch(self.path):
                return
            super().log_request(code, size)

        def _send(self, status, data, mime, *, download=None, etag=None):
            self.send_response(status)
            if data is not None:
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(data)))
            if etag is not None:
                self.send_header('ETag', etag)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            if download is not None:
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(download, safe=''))
            self.end_headers()
            if data is not None:
                self.wfile.write(data)

        def _json(self, data, status=200, *, etag=None):
            self._send(status, json.dumps(data, ensure_ascii=False).encode('utf-8'),
                       'application/json; charset=utf-8', etag=etag)

        def _body(self, limit):
            length = self.headers.get('Content-Length', '')
            if not length.isdecimal() or not 0 < int(length) <= limit:
                raise _RequestError(413, '文件或请求过大，或者内容为空。')
            data = self.rfile.read(int(length))
            if len(data) != int(length):
                raise ValueError('未接收到完整的内容。')
            return data

        def _read_json(self, limit=2 * 1024 * 1024):
            if self.headers.get_content_type() != 'application/json':
                raise _RequestError(415, '设置必须使用 JSON。')
            value = json.loads(self._body(limit))
            if not isinstance(value, dict):
                raise ValueError('设置必须是 JSON 对象。')
            return value

        def _check_access(self, *, post):
            port_number = self.server.server_address[1]
            hosts = (f'127.0.0.1:{port_number}', f'localhost:{port_number}')
            host = self.headers.get('Host')
            if host not in hosts:
                raise _RequestError(403, '请使用工作台启动时显示的本机地址。')
            if post:
                origin = self.headers.get('Origin')
                if self.headers.get('X-GAID-Request') != '1' or (origin and origin != 'http://' + host):
                    raise _RequestError(403, '此请求必须来自本机工作台页面。')

        def _document(self, path, post):
            documents = self.server.documents
            if path.path == '/api/settings':
                return self._json(documents.change_preferences(self._read_json()) if post else documents.preferences.read())
            if not post and path.path == '/api/cache':
                return self._json(documents.cache_info())
            if post and path.path == '/api/cache/clear':
                value = self._read_json()
                if set(value) - {'mode'}:
                    raise ValueError('清理缓存不能指定路径。')
                return self._json(documents.clear_cache(mode=value.get('mode', 'all')))
            if post and path.path == '/api/folder':
                value = self._read_json()
                if set(value) - {'initial'}:
                    raise ValueError('目录选择字段无效。')
                return self._json({'path': _choose_folder(value.get('initial', ''))})
            if path.path == '/api/documents':
                if not post:
                    return self._json(documents.recent())
                if self.headers.get_content_type() != 'application/octet-stream':
                    raise _RequestError(415, '请选择一个图片文件。')
                name = unquote(self.headers.get('X-GAID-Filename', ''))
                return self._json(documents.open(self._body(64 * 1024 * 1024), name))

            match = _DOCUMENT.fullmatch(path.path)
            if not match:
                raise _RequestError(404, '找不到此页面。')
            identifier, action = match.groups()
            if not post and action is None:
                tag, snapshot = documents.document_snapshot(identifier, self.server.instance, self.headers.get('If-None-Match'))
                if snapshot is None:
                    return self._send(304, None, None, etag=tag)
                return self._json(snapshot, etag=tag)
            if post and action == 'prepare-edit':
                documents.prepare_edit(identifier, _revision(self._read_json()))
                return self._json({})
            if not post and action == 'thumbnail':
                query = parse_qs(path.query, keep_blank_values=True)
                view = query.get('view', ['original'])[0]
                value = query.get('revision', [None])[0]
                if (value is not None and (not value.isdecimal() or len(value) > 16)) or (view != 'original' and value is None):
                    raise ValueError('请指定节点缩略图的修订号。')
                return self._send(200, documents.thumbnail(identifier, view, None if value is None else int(value)), 'image/png')
            if not post and action == 'changes':
                query = parse_qs(path.query, keep_blank_values=True)
                values = query.get('revision', [])
                if len(values) != 1 or not values[0].isdecimal() or len(values[0]) > 16:
                    raise ValueError('请指定要核对的成品版本。')
                stages = query.get('stage', [None])
                if len(stages) != 1:
                    raise ValueError('请选择要核对的编辑阶段。')
                return self._json(documents.changes(identifier, int(values[0]), stage=stages[0]))
            if not post and action in ('image', 'download'):
                query = parse_qs(path.query)
                download = action == 'download'
                view = query.get('view', ['b' if download else 'original'])[0]
                value = query.get('revision', [None])[0]
                if download and (view not in ('a', 'b') or value is None):
                    raise ValueError('请指定要导出的结果及版本。')
                if value is not None and (not value.isdecimal() or len(value) > 16):
                    raise ValueError('图像版本无效。')
                hidden = query.get('hide', [])
                if len(hidden) > 1:
                    raise ValueError('请选择要隐藏的分区。')
                data = documents.asset(identifier, view, download=download,
                                       revision=int(value) if value is not None else None, module=query.get('module', [None])[0],
                                       hidden_parts=hidden[0].split(',') if hidden else ())
                name = Path(documents.get(identifier)['name']).stem + '-gaid.png' if download else None
                return self._send(200, data, 'image/png', download=name)
            if post and action == 'edit':
                return self._json(documents.update(identifier, self._read_json()))
            if post and action == 'pin':
                return self._json(documents.pin(identifier, _revision(self._read_json())))
            if post and action == 'export':
                value = self._read_json()
                if set(value) != {'revision', 'view'}:
                    raise ValueError('本机导出只接受版本和视图。')
                target = documents.export(identifier, value['view'], _revision(value))
                export_id = uuid4().hex
                with export_lock:
                    exports[identifier, export_id] = target
                    if len(exports) > _EXPORT_HISTORY_LIMIT:
                        exports.pop(next(iter(exports)))
                return self._json({'path': str(target), 'export_id': export_id})
            if post and action == 'export/reveal':
                value = self._read_json(512)
                export_id = value.get('export_id')
                if set(value) != {'export_id'} or not isinstance(export_id, str) or not _EXPORT_ID.fullmatch(export_id):
                    raise ValueError('请选择本工作台导出的 PNG。')
                with export_lock:
                    target = exports.get((identifier, export_id))
                if target is None or not target.is_file() or target.resolve() != target:
                    raise _RequestError(404, '找不到此导出的 PNG。')
                try:
                    reveal_file(target)
                except (OSError, SubprocessError) as error:
                    raise _RequestError(500, '无法打开文件管理器。') from error
                return self._json({})
            if post and action in ('sam-object', 'sam-selection'):
                value = self._read_json()
                return self._json(documents.preview_object(identifier, _revision(value),
                    value.get('positive'), value.get('negative', [])))
            if post and action == 'sam-object/apply':
                value = self._read_json()
                return self._json(documents.apply_object(identifier, _revision(value), value.get('token')))
            if post and action == 'sam-selection/apply':
                value = self._read_json()
                revision = _revision(value)
                value.pop('revision')
                token = value.pop('token', None)
                return self._json(documents.apply_object(identifier, revision, token, selection=value))
            if post and action == 'corrections/export':
                return self._json({'path': str(documents.corrections(identifier, _revision(self._read_json())))})
            if post and action == 'corrections/import':
                value = self._read_json(64 * 1024 * 1024)
                record, files = _correction_files(value)
                return self._json(documents.import_correction(identifier, _revision(value), record, files))
            raise _RequestError(404, '找不到此操作。')

        def _run(self, *, post):
            try:
                self._check_access(post=post)
                path = urlsplit(self.path)
                if not post and path.path in assets:
                    data, mime = assets[path.path]
                    return self._send(200, data, mime + '; charset=utf-8')
                if not post and path.path == '/api/options':
                    return self._json(_options())
                if not post and path.path == '/api/instance':
                    return self._json({'instance': self.server.instance})
                if post and path.path == '/api/prepare-models':
                    if self._read_json():
                        raise ValueError('预热模型的请求须为空。')
                    self.server.documents.prepare_models()
                    return self._json({})
                if post and path.path == '/api/page':
                    value = self._read_json(512)
                    identifier, opened = value.get('id'), value.get('open')
                    if (set(value) != {'id', 'open'} or not isinstance(identifier, str)
                            or not _PAGE_ID.fullmatch(identifier) or type(opened) is not bool):
                        raise ValueError('工作台页面状态无效。')
                    self.server.pages.update(identifier, opened)
                    return self._json({'open': opened})
                if post and path.path == '/api/exit':
                    if self._read_json():
                        raise ValueError('退出工作台的请求须为空。')
                    self._json({'closing': True})
                    Thread(target=self.server.shutdown, name='gaid-exit', daemon=True).start()
                    return
                return self._document(path, post)
            except (BrokenPipeError, ConnectionResetError):
                return
            except _RequestError as error:
                self._json({'error': str(error)}, error.status)
            except (RevisionConflict, StorageBusy) as error:
                self._json({'error': str(error)}, 409)
            except (FileNotFoundError, KeyError) as error:
                self._json({'error': str(error)}, 404)
            except (ValueError, TypeError) as error:
                self._json({'error': str(error)}, 400)
            except Exception as error:
                traceback.print_exc()
                self._json({'error': str(error) or type(error).__name__}, 500)

        def do_GET(self):
            self._run(post=False)

        def do_POST(self):
            self._run(post=True)

    class LocalServer(ThreadingHTTPServer):
        def server_close(self):
            if hasattr(self, 'pages'):
                self.pages.close()
            super().server_close()
            if hasattr(self, 'documents'):
                self.documents.close()

    server = LocalServer(('127.0.0.1', port), Handler)
    server.instance = uuid4().hex
    try:
        server.documents = Documents(workspace)
        server.pages = _PagePresence(server)
    except Exception:
        server.server_close()
        raise
    return server


def _open_browser(address):
    import webbrowser

    try:
        if webbrowser.open(address):
            return True
    except (OSError, webbrowser.Error):
        pass
    print(f'未能自动打开浏览器，请访问 {address}', flush=True)
    return False


def _running_port(marker):
    try:
        record = json.loads(marker.read_text(encoding='utf-8'))
        if not isinstance(record, dict):
            return None
        port, instance = record.get('port'), record.get('instance')
        if type(port) is not int or not 0 < port <= 65535 or not isinstance(instance, str):
            return None
        with closing(HTTPConnection('127.0.0.1', port, timeout=.4)) as connection:
            connection.request('GET', '/api/instance')
            response = connection.getresponse()
            if response.status == 200 and json.loads(response.read(4096)) == {'instance': instance}:
                return port
    except (OSError, ValueError, HTTPException):
        pass  # The process may still be publishing its address.
    return None


def launch_workbench(*, port=0, open_browser=True, workspace=None, browser=None):
    workspace = Path(workspace or 'outputs/workbench').resolve()
    marker = workspace / '.gaid-instance.json'
    browser = browser or _open_browser
    deadline = monotonic() + 5
    while True:
        try:
            with _file_lock(workspace / '.gaid-instance.lock', wait=False), create_server(workspace, port=port) as server:
                _replace_json(marker, {'port': server.server_port, 'instance': server.instance})
                address = f'http://127.0.0.1:{server.server_port}/'
                print(f'GAID 工作台：{address}', flush=True)
                if open_browser:
                    Thread(target=browser, args=(address,), name='gaid-browser', daemon=True).start()
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
                finally:
                    marker.unlink(missing_ok=True)
            return 0
        except StorageBusy:
            existing = _running_port(marker)
            if existing is None:
                if monotonic() >= deadline:
                    raise StorageBusy('工作台正在启动或退出，请稍后再打开。') from None
                sleep(.1)
                continue
            address = f'http://127.0.0.1:{existing}/'
            if port and port != existing:
                raise StorageBusy(f'该工作区的工作台已在 {address} 运行。') from None
            print(f'GAID 工作台：{address}', flush=True)
            if open_browser:
                browser(address)
            return 0

