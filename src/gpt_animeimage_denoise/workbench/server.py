"""Loopback HTTP transport for the image workbench."""
from base64 import b64decode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import traceback
from urllib.parse import parse_qs, quote, unquote, urlsplit

from ..denoisers import uses_l0
from ..editor import DEFAULT_MODULES, MODULES
from .documents import WORKBENCH_METHODS


_WEB = Path(__file__).with_name('web')
_STATIC = {'/': ('index.html', 'text/html'), '/style.css': ('style.css', 'text/css'),
           '/app.js': ('app.js', 'text/javascript'),
           '/magnetic-lasso-worker.js': ('magnetic-lasso-worker.js', 'text/javascript')}
_PARTS = {'skin': ('皮肤', '#db896c'), 'face': ('面部', '#e1b450'), 'hair': ('头发', '#6587df'),
          'eyes': ('眼睛', '#36a5a0'), 'mouth': ('嘴巴', '#d978a4'), 'outfit': ('服饰', '#8b78c7'),
          'background': ('背景', '#93a6bc')}
_DEFAULTS = {'method': 'l0', 'l0': .0005, 'downscale': .6, 'strength': 1,
             'restoration_gain': .25, 'sigma': 12, 'structure_protection': .2}
_DOCUMENT = re.compile(r'/api/documents/([0-9a-f]{64})(?:/(.*))?')


class _RequestError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _open_session(source, workspace):
    from .session import WorkbenchSession

    return WorkbenchSession.open(source, workspace)


def _options():
    return {'modules': [{'id': name, 'label': _PARTS[name][0], 'color': _PARTS[name][1]} for name in MODULES],
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

    class Handler(BaseHTTPRequestHandler):
        def log_request(self, code='-', size='-'):
            if code == 200 and self.command == 'GET' and _DOCUMENT.fullmatch(self.path):
                return
            super().log_request(code, size)

        def _send(self, status, data, mime, *, download=None):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            if download is not None:
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(download, safe=''))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, data, status=200):
            self._send(status, json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

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
                return self._json(documents.get(identifier))
            if not post and action == 'changes':
                values = parse_qs(path.query, keep_blank_values=True).get('revision', [])
                if len(values) != 1 or not values[0].isdecimal() or len(values[0]) > 16:
                    raise ValueError('请指定要核对的成品版本。')
                return self._json(documents.changes(identifier, int(values[0])))
            if not post and action in ('image', 'download'):
                query = parse_qs(path.query)
                download = action == 'download'
                view = query.get('view', ['b' if download else 'original'])[0]
                value = query.get('revision', [None])[0]
                if download and (view not in ('a', 'b') or value is None):
                    raise ValueError('请指定要导出的结果及版本。')
                if value is not None and (not value.isdecimal() or len(value) > 16):
                    raise ValueError('图像版本无效。')
                data = documents.asset(identifier, view, download=download,
                                       revision=int(value) if value is not None else None)
                name = Path(documents.get(identifier)['name']).stem + '-gaid.png' if download else None
                return self._send(200, data, 'image/png', download=name)
            if post and action == 'edit':
                return self._json(documents.update(identifier, self._read_json()))
            if post and action == 'pin':
                return self._json(documents.pin(identifier, _revision(self._read_json())))
            if post and action == 'sam-object':
                value = self._read_json()
                return self._json(documents.preview_object(identifier, _revision(value),
                    value.get('positive'), value.get('negative', [])))
            if post and action == 'sam-object/apply':
                value = self._read_json()
                return self._json(documents.apply_object(identifier, _revision(value), value.get('token')))
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
                if not post and path.path in _STATIC:
                    name, mime = _STATIC[path.path]
                    return self._send(200, (_WEB / name).read_bytes(), mime + '; charset=utf-8')
                if not post and path.path == '/api/options':
                    return self._json(_options())
                return self._document(path, post)
            except (BrokenPipeError, ConnectionResetError):
                return
            except _RequestError as error:
                self._json({'error': str(error)}, error.status)
            except RevisionConflict as error:
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
            super().server_close()
            if hasattr(self, 'documents'):
                self.documents.close()

    server = LocalServer(('127.0.0.1', port), Handler)
    try:
        server.documents = Documents(workspace, _open_session)
    except Exception:
        server.server_close()
        raise
    return server


def launch_workbench(*, port=0, open_browser=True):
    import webbrowser

    with create_server(Path('outputs/workbench'), port=port) as server:
        address = f'http://127.0.0.1:{server.server_address[1]}/'
        print(f'GAID 工作台：{address}', flush=True)
        if open_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0

