from base64 import b64decode
from contextlib import closing
from http.client import HTTPConnection
from io import BytesIO
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import tempfile
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


class WorkbenchStartupTest(unittest.TestCase):
    def test_running_server_keeps_frontend_assets_from_startup(self):
        from gpt_animeimage_denoise.workbench import server as transport

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / 'app.js'
            script.write_bytes(b'old page')
            with patch.object(transport, '_WEB', root), \
                 patch.object(transport, '_STATIC', {'/app.js': ('app.js', 'text/javascript')}), \
                 transport.create_server(root / 'workbench') as server:
                script.write_bytes(b'new page')
                service = Thread(target=server.serve_forever, daemon=True)
                service.start()
                try:
                    with closing(HTTPConnection('127.0.0.1', server.server_port, timeout=5)) as conn:
                        conn.request('GET', '/app.js')
                        response = conn.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.read(), b'old page')
                finally:
                    server.shutdown()
                    service.join(timeout=5)

    def test_startup_serves_options_and_saved_settings_before_importing_scipy(self):
        script = """\
from contextlib import closing
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import sys
from threading import Thread

from gpt_animeimage_denoise.workbench import create_server

preferences = Path(os.environ['GAID_WORKBENCH_SETTINGS'])
preferences.write_text(json.dumps({
    'theme': 'dark',
    'defaults': {'method': 'esrgan', 'modules': ['outfit'], 'strength': .4},
}), encoding='utf-8')
with create_server(Path.cwd()) as server:
    service = Thread(target=server.serve_forever, daemon=True)
    service.start()
    try:
        for path in ('/', '/api/options', '/api/settings'):
            with closing(HTTPConnection('127.0.0.1', server.server_port, timeout=5)) as conn:
                conn.request('GET', path)
                response = conn.getresponse()
                payload = response.read()
                assert response.status == 200, path
                if path == '/api/settings':
                    settings = json.loads(payload)
                    assert settings['theme'] == 'dark'
                    assert settings['defaults']['method'] == 'esrgan'
                    assert settings['defaults']['strength'] == .4
        assert 'scipy' not in sys.modules, 'image-processing dependencies loaded at startup'
    finally:
        server.shutdown()
        service.join(timeout=5)
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, '-B', '-c', script], cwd=temporary, timeout=15,
                                    capture_output=True, text=True, env={
                                        **os.environ, 'GAID_WORKBENCH_SETTINGS': str(Path(temporary) / 'settings.json'),
                                    })
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_slow_browser_launch_does_not_block_http(self):
        from gpt_animeimage_denoise.workbench import create_server, launch_workbench

        with tempfile.TemporaryDirectory() as temporary:
            server = create_server(Path(temporary))
            browser_called, browser_returned, serving, open_finished = Event(), Event(), Event(), Event()
            results = Queue()
            original_serve = server.serve_forever

            def open_browser(address):
                browser_called.set()
                try:
                    return browser_returned.wait(5)
                finally:
                    open_finished.set()

            def serve():
                serving.set()
                return original_serve()

            with patch('gpt_animeimage_denoise.workbench.server.create_server', return_value=server), \
                 patch.object(server, 'serve_forever', side_effect=serve), \
                 patch('webbrowser.open', side_effect=open_browser):
                launcher = Thread(target=lambda: results.put(launch_workbench(workspace=Path(temporary))), daemon=True)
                launcher.start()
                try:
                    self.assertTrue(browser_called.wait(5))
                    self.assertTrue(serving.wait(1), 'HTTP is waiting for the browser to return')
                    with closing(HTTPConnection('127.0.0.1', server.server_port, timeout=2)) as conn:
                        conn.request('GET', '/')
                        response = conn.getresponse()
                        self.assertEqual(response.status, 200)
                        response.read()
                    self.assertFalse(open_finished.is_set())
                finally:
                    browser_returned.set()
                    if serving.wait(5):
                        server.shutdown()
                    launcher.join(5)
                    server.server_close()
            self.assertEqual(results.get(timeout=1), 0)


class WorkbenchHTTPTest(unittest.TestCase):
    def setUp(self):
        from gpt_animeimage_denoise.workbench import create_server

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        settings = patch.dict(os.environ, GAID_WORKBENCH_SETTINGS=str(Path(temp.name) / 'preferences.json'))
        settings.start()
        self.addCleanup(settings.stop)
        self.server = create_server(Path(temp.name), port=0)
        self.addCleanup(self.server.server_close)
        thread = Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def request(self, method, path, body=None, **headers):
        connection = HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_rejects_other_origins_before_reading_upload(self):
        status, _ = self.request('POST', '/api/documents', b'image', Origin='http://example.test',
                                 **{'X-GAID-Request': '1', 'Content-Type': 'application/octet-stream'})
        self.assertEqual(status, 403)

    def test_recent_documents_starts_empty(self):
        status, body = self.request('GET', '/api/documents')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), [])

    def test_running_workbench_is_reused_for_the_same_workspace(self):
        from gpt_animeimage_denoise.workbench import launch_workbench
        from gpt_animeimage_denoise.workbench.storage import _file_lock

        code, body = self.request('GET', '/api/instance')
        self.assertEqual(code, 200)
        workspace = self.server.documents.workspace
        marker = workspace / '.gaid-instance.json'
        marker.write_text(json.dumps({'port': self.port, **json.loads(body)}), encoding='utf-8')
        opened = []
        with _file_lock(workspace / '.gaid-instance.lock'):
            self.assertEqual(launch_workbench(workspace=workspace, browser=opened.append), 0)
        self.assertEqual(opened, [f'http://127.0.0.1:{self.port}/'])

    def test_workbench_exit_requires_a_local_post_and_stops_the_server(self):
        self.assertEqual(self.request('GET', '/api/exit')[0], 404)
        self.assertEqual(self.request('POST', '/api/exit', '{}', **{
            'Origin': 'http://example.test', 'X-GAID-Request': '1', 'Content-Type': 'application/json',
        })[0], 403)
        self.assertEqual(self.post('/api/exit', {'unrecognized': True})[0], 400)
        with patch.object(self.server, 'shutdown', wraps=self.server.shutdown) as shutdown:
            self.assertEqual(self.post('/api/exit', {}), (200, {'closing': True}))
            deadline = monotonic() + 3
            while not shutdown.called and monotonic() < deadline:
                sleep(.01)
            shutdown.assert_called_once()

    def post(self, route, value):
        status, body = self.request('POST', route, json.dumps(value), **{
            'X-GAID-Request': '1', 'Content-Type': 'application/json'})
        return status, json.loads(body)

    def test_settings_are_machine_local_and_apply_only_to_new_images(self):
        from gpt_animeimage_denoise.editor import SelectionEditor
        from gpt_animeimage_denoise.workbench import create_server

        docs = self.server.documents
        self.assertEqual(json.loads(self.request('GET', '/api/settings')[1])['theme'], 'system')
        status, settings = self.post('/api/settings', {'defaults': {'method': 'esrgan', 'strength': .4,
                                    'modules': ['outfit'], 'part_strengths': {'outfit': .7}},
                                'theme': 'dark', 'remember_brush': True, 'brush_radius': 18, 'brush_feather': 7})
        self.assertEqual(status, 200)
        self.assertEqual(settings['defaults']['sigma'], 12)
        self.assertEqual(self.post('/api/settings', {'brush_radius': 3})[0], 400)
        self.assertEqual(self.post('/api/cache/clear', {'path': str(docs.root)})[0], 400)
        self.assertEqual(self.post('/api/settings', {'export_dir': 'not-absolute'})[0], 400)
        with patch('gpt_animeimage_denoise.workbench.server._choose_folder', return_value=None):
            self.assertEqual(self.post('/api/folder', {'initial': ''}), (200, {'path': None}))
        for url in ('/settings.css', '/settings.js'):
            self.assertEqual(self.request('GET', url)[0], 200)
        with create_server(docs.workspace / 'second-workspace') as second:
            self.assertEqual(second.documents.preferences.read(), settings)
        rgb = np.full((22, 32, 3), 167, np.uint8)
        labels = np.full(rgb.shape[:2], 6, np.uint8)
        empty = np.zeros(labels.shape, bool)
        editor = SelectionEditor(rgb, None, labels, empty, empty)
        session = SimpleNamespace(editor=editor, icc=None, candidate=Mock(return_value=rgb.astype(np.float32) / 255))
        def upload():
            content = BytesIO()
            Image.fromarray(rgb).save(content, 'PNG')
            code, body = self.request('POST', '/api/documents', content.getvalue(),
                                    **{'X-GAID-Request': '1', 'Content-Type': 'application/octet-stream',
                                       'X-GAID-Filename': 'image.png'})
            self.assertEqual(code, 200)
            return '/api/documents/' + json.loads(body)['id']
        with patch.object(docs, 'opener', return_value=session):
            first = upload()
            self.assertEqual(self.ready(first)['settings'], settings['defaults'])
            self.assertEqual(self.post('/api/settings', {'defaults': {'method': 'l0', 'strength': .8}})[0], 200)
            self.assertEqual(self.ready(upload())['settings'], settings['defaults'])
            rgb[0, 0] = 166
            second = self.ready(upload())
            self.assertEqual((second['settings']['method'], second['settings']['strength']), ('l0', .8))

    def test_generic_sam_highlight_candidate_thumbnails_and_revision_export(self):
        from gpt_animeimage_denoise.editor import SelectionEditor

        docs = self.server.documents
        rgb = np.full((96, 420, 3), 183, np.uint8)
        alpha = np.full(rgb.shape[:2], 255, np.uint8)
        alpha[:, 0] = 0
        labels = np.ones(alpha.shape, np.uint8)
        labels[:, 100:] = 0
        empty = np.zeros(alpha.shape, bool)
        chosen = empty.copy()
        chosen[13:29, 57:201] = True
        editor = SelectionEditor(rgb, alpha, labels, empty, empty)
        session = SimpleNamespace(editor=editor, icc=b'a-profile', candidate=Mock(return_value=np.full(rgb.shape, .2, np.float32)),
                                  predict_object=Mock(return_value=chosen))
        content = BytesIO()
        Image.fromarray(np.dstack((rgb, alpha))).save(content, 'PNG', icc_profile=b'a-profile')
        with patch.object(docs, 'opener', return_value=session):
            code, body = self.request('POST', '/api/documents', content.getvalue(), **{
                'X-GAID-Request': '1', 'Content-Type': 'application/octet-stream', 'X-GAID-Filename': 'sample.png'})
            self.assertEqual(code, 200)
            route = '/api/documents/' + json.loads(body)['id']
            self.ready(route)
            self.assertEqual(self.post(route + '/pin', {'revision': 0})[0], 200)
            saved = self.request('GET', route + '/download?view=a&revision=0')[1]
            code, body = self.request('GET', route + '/image?view=candidate&revision=0')
            self.assertEqual(code, 200)
            with Image.open(BytesIO(body)) as image:
                self.assertEqual(image.info['icc_profile'], b'a-profile')
                np.testing.assert_array_equal(np.array(image)[..., :3], 51)
                np.testing.assert_array_equal(np.array(image)[..., 3], alpha)
            code, preview = self.post(route + '/sam-selection', {'revision': 0, 'positive': [[70, 20]], 'negative': []})
            self.assertEqual(code, 200)
            self.assertEqual(self.ready(route)['cursor'], 0)
            action = {'revision': 0, 'stage': 'parts', 'tool': 'classify', 'module': 'mouth', 'token': preview['token'],
                      'protected_parts': ['skin']}
            self.assertEqual(self.post(route + '/sam-selection/apply', {**action, 'module': 'missing'})[0], 400)
            self.assertEqual(self.post(route + '/sam-selection/apply', {**action, 'protected_parts': ['missing']})[0], 400)
            self.assertEqual(self.post(route + '/sam-selection/apply', {**action, 'protected_parts': ['skin', 'skin']})[0], 400)
            self.assertEqual(self.post(route + '/sam-selection/apply', action)[0], 200)
            self.assertEqual(self.post(route + '/sam-selection/apply', action)[0], 409)
            state = self.ready(route)
            expected = chosen & (labels != 1)
            self.assertEqual(state['edit_feedback']['skipped_protected_pixels'], int((chosen & (labels == 1)).sum()))
            for view in ('parts', 'weight', 'candidate', 'b'):
                code, image = self.request('GET', route + '/thumbnail?view=' + view + '&revision=1')
                self.assertEqual(code, 200)
                with Image.open(BytesIO(image)) as thumb:
                    self.assertEqual(thumb.size, (320, 73))
            code, image = self.request('GET', route + '/image?view=labels&revision=1')
            self.assertEqual(code, 200)
            with Image.open(BytesIO(image)) as ids:
                self.assertEqual(ids.mode, 'L')
                self.assertEqual(ids.getpixel((0, 20)), 254)
                np.testing.assert_array_equal(np.asarray(ids) == 5, expected)
            code, image = self.request('GET', route + '/image?view=highlight&module=mouth&revision=1')
            self.assertEqual(code, 200)
            with Image.open(BytesIO(image)) as highlight:
                self.assertEqual(highlight.mode, 'P')
                np.testing.assert_array_equal(np.asarray(highlight.convert('RGBA'))[..., 3] > 0, expected)
            self.assertEqual(self.request('GET', route + '/thumbnail?view=parts&revision=0')[0], 409)
            self.assertEqual(self.request('GET', route + '/image?view=candidate&revision=0')[0], 409)
            update = {key: state[key] for key in ('revision', 'settings', 'edits', 'cursor')}
            forged = [{**state['edits'][0], 'module': 'hair'}]
            self.assertEqual(self.post(route + '/edit', {**update, 'edits': forged})[0], 400)
            self.assertEqual(self.post(route + '/edit', {**update, 'cursor': 0})[0], 200)
            self.assertEqual(self.ready(route)['cursor'], 0)
            self.assertEqual(self.post(route + '/edit', {**update, 'revision': 2})[0], 200)
            redone = self.ready(route)
            self.assertEqual(redone['edits'], state['edits'])
            export = docs.workspace / 'exports'
            export.mkdir()
            self.assertEqual(self.post('/api/settings', {'export_dir': str(export)})[0], 200)
            self.assertEqual(self.post(route + '/export', {'revision': 0, 'view': 'b'})[0], 409)
            self.assertEqual(self.post(route + '/export', {'revision': 0, 'view': 'a', 'path': '..'})[0], 400)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(2) as pool:
                outputs = list(pool.map(lambda _: self.post(route + '/export', {'revision': 0, 'view': 'a'}), range(2)))
            self.assertEqual([code for code, _ in outputs], [200, 200])
            paths = [Path(result['path']) for _, result in outputs]
            self.assertEqual({p.name for p in paths}, {'sample-gaid.png', 'sample-gaid-2.png'})
            self.assertTrue(all(path.parent == export and path.read_bytes() == saved for path in paths))

    def test_recent_thumbnail_uses_normalized_original_during_analysis(self):
        from gpt_animeimage_denoise.editor import SelectionEditor

        pixels = np.full((216, 384, 4), (80, 120, 180, 192), np.uint8)
        pixels[:, :96, 3] = 0
        icc = b'workbench-thumbnail-test-profile'
        content = BytesIO()
        Image.fromarray(pixels).save(content, 'PNG', icc_profile=icc)
        labels = np.ones(pixels.shape[:2], np.uint8)
        empty = np.zeros_like(labels, dtype=bool)
        editor = SelectionEditor(pixels[..., :3], pixels[..., 3], labels, empty, empty)
        candidate = Mock(return_value=np.full(pixels[..., :3].shape, .3, np.float32))
        session = SimpleNamespace(editor=editor, icc=icc, candidate=candidate)
        started, release = Event(), Event()

        def open_session(*_):
            started.set()
            self.assertTrue(release.wait(8))
            return session

        with patch.object(self.server.documents, 'opener', side_effect=open_session) as analyze:
            try:
                status, body = self.request('POST', '/api/documents', content.getvalue(),
                                            **{'X-GAID-Request': '1', 'Content-Type': 'application/octet-stream',
                                               'X-GAID-Filename': 'sample.png'})
                self.assertEqual(status, 200)
                doc = json.loads(body)['id']
                self.assertTrue(started.wait(5))
                route = '/api/documents/' + doc
                status, png = self.request('GET', route + '/thumbnail')
                self.assertEqual(status, 200)
                with Image.open(BytesIO(png)) as thumbnail:
                    self.assertEqual(thumbnail.format, 'PNG')
                    self.assertEqual(thumbnail.size, (192, 108))
                    self.assertEqual(thumbnail.info['icc_profile'], icc)
                    self.assertEqual(thumbnail.getpixel((10, 60))[3], 0)
                    self.assertEqual(thumbnail.getpixel((128, 60))[3], 192)
                analyze.assert_called_once()
                candidate.assert_not_called()
            finally:
                release.set()
            self.assertEqual(self.ready(route)['revision'], 0)
            (self.server.documents.root / doc / 'original.png').unlink()
            self.assertEqual(self.request('GET', route + '/thumbnail')[0], 404)
        self.assertEqual(self.request('GET', '/api/documents/' + '0' * 64 + '/thumbnail')[0], 404)

    def test_magnetic_lasso_worker_is_served_as_local_javascript(self):
        connection = HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.request('GET', '/magnetic-lasso-worker.js')
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader('Content-Type'), 'text/javascript; charset=utf-8')
            self.assertTrue(response.read())
        finally:
            connection.close()

    def test_cugan_is_not_offered_in_the_workbench(self):
        status, body = self.request('GET', '/api/options')
        self.assertEqual(status, 200)
        self.assertEqual([item['id'] for item in json.loads(body)['methods']],
                         ['l0', 'esrgan', 'esrgan-l0', 'apisr'])

    def ready(self, route):
        deadline = monotonic() + 8
        while monotonic() < deadline:
            status, body = self.request('GET', route)
            self.assertEqual(status, 200)
            state = json.loads(body)
            if state['status'] == 'ready':
                return state
            self.assertNotEqual(state['status'], 'error', state.get('error'))
            sleep(.02)
        self.fail('Document did not finish')

    def test_sam_object_endpoints_preview_then_commit_only_confirmed_mask(self):
        from gpt_animeimage_denoise.editor import SelectionEditor

        rgb = np.full((32, 40, 3), 150, np.uint8)
        labels = np.ones(rgb.shape[:2], np.uint8)
        empty = np.zeros(labels.shape, bool)
        cup = empty.copy()
        cup[10:18, 17:23] = True
        editor = SelectionEditor(rgb, None, labels, empty, empty)
        candidate = Mock(return_value=np.full(rgb.shape, .3, np.float32))
        predict = Mock(return_value=cup)
        session = SimpleNamespace(editor=editor, icc=None, candidate=candidate, predict_object=predict)
        content = BytesIO()
        Image.fromarray(rgb).save(content, 'PNG')
        headers = {'X-GAID-Request': '1', 'Content-Type': 'application/octet-stream',
                   'X-GAID-Filename': 'sample.png'}

        def post(route, value):
            return self.request('POST', route, json.dumps(value),
                                **{'X-GAID-Request': '1', 'Content-Type': 'application/json'})

        with patch.object(self.server.documents, 'opener', return_value=session):
            status, body = self.request('POST', '/api/documents', content.getvalue(), **headers)
            self.assertEqual(status, 200)
            route = '/api/documents/' + json.loads(body)['id']
            state = self.ready(route)
            self.assertEqual(self.request('GET', route + '/changes')[0], 400)
            self.assertEqual(self.request('GET', route + '/changes?revision=oops')[0], 400)
            status, body = self.request('GET', route + f"/changes?revision={state['revision']}")
            self.assertEqual(status, 200)
            self.assertIsNone(json.loads(body)['overlay'])
            prompt = {'revision': state['revision'], 'positive': [[19, 13]], 'negative': [[25, 13]]}
            self.assertEqual(post(route + '/sam-object', {**prompt, 'positive': [[40, 13]]})[0], 400)
            status, body = post(route + '/sam-object', prompt)
            self.assertEqual(status, 200)
            prediction = json.loads(body)
            predict.assert_called_once_with(prompt['positive'], prompt['negative'])
            self.assertEqual(prediction['pixels'], int(cup.sum()))
            self.assertEqual(prediction['revision'], state['revision'])
            prefix, png = prediction['preview'].split(',', 1)
            self.assertEqual(prefix, 'data:image/png;base64')
            with Image.open(BytesIO(b64decode(png))) as overlay:
                np.testing.assert_array_equal(np.asarray(overlay)[..., 3] > 0, cup)
            self.assertEqual(self.ready(route), state)
            apply = {'revision': state['revision'], 'token': prediction['token']}
            self.assertEqual(post(route + '/sam-object/apply', {**apply, 'token': 'invalid'})[0], 409)
            self.assertEqual(post(route + '/sam-object/apply', apply)[0], 200)
            self.assertEqual(post(route + '/sam-object/apply', apply)[0], 409)
            state = self.ready(route)
            self.assertEqual(state['edits'], [{'tool': 'sam_object', 'token': prediction['token']}])
            self.assertEqual(self.request('GET', route + f"/changes?revision={prompt['revision']}")[0], 409)
            status, body = self.request('GET', route + f"/changes?revision={state['revision']}")
            self.assertEqual(status, 200)
            changes = json.loads(body)
            self.assertEqual(changes['revision'], state['revision'])
            self.assertEqual(changes['counts']['removed'], int(cup.sum()))
            with Image.open(BytesIO(b64decode(changes['overlay'].split(',', 1)[1]))) as overlay:
                np.testing.assert_array_equal(np.asarray(overlay)[..., 3] > 0, cup)
            status, png = self.request('GET', route + f"/download?view=b&revision={state['revision']}")
            self.assertEqual(status, 200)
            with Image.open(BytesIO(png)) as result:
                output = np.asarray(result)
                np.testing.assert_array_equal(output[cup], rgb[cup])
                self.assertFalse(np.array_equal(output[13, 25], rgb[13, 25]))

    def test_switching_methods_reuses_analysis_and_downloads_latest_preview(self):
        from gpt_animeimage_denoise.editor import SelectionEditor

        rgb = np.full((32, 40, 3), 150, np.uint8)
        labels = np.ones(rgb.shape[:2], np.uint8)
        labels[:, :4] = 0
        objects = np.zeros_like(labels, dtype=bool)
        objects[:, :1] = True
        suspected = np.zeros_like(objects)
        suspected[8:16, 15:23] = True
        alpha = np.full(labels.shape, 128, np.uint8)
        alpha[:, -2:] = 0
        editor = SelectionEditor(rgb, alpha, labels, objects, np.zeros_like(objects))
        candidate = Mock(return_value=np.full(rgb.shape, .3, np.float32))
        icc = b'workbench-test-profile'
        session = SimpleNamespace(editor=editor, icc=icc, width=40, height=32, candidate=candidate,
                                  suspected_objects=suspected)
        content = BytesIO()
        Image.fromarray(np.dstack((rgb, alpha))).save(content, 'PNG', icc_profile=icc)
        headers = {'X-GAID-Request': '1', 'Content-Type': 'application/octet-stream',
                   'X-GAID-Filename': 'sample.png'}
        gate = Event()
        def open_session(*_):
            self.assertTrue(gate.wait(8))
            return session

        def post(route, value):
            return self.request('POST', route, json.dumps(value),
                                **{'X-GAID-Request': '1', 'Content-Type': 'application/json'})

        with patch.object(self.server.documents, 'opener', side_effect=open_session) as analyze:
            try:
                status, body = self.request('POST', '/api/documents', content.getvalue(), **headers)
                self.assertEqual(status, 200)
                route = '/api/documents/' + json.loads(body)['id']
                self.assertIsNone(json.loads(body)['b'])
                self.assertEqual(self.request('GET', route + '/image?view=original')[0], 200)
                self.assertEqual(self.request('GET', route + '/image?view=magnetic_source&revision=0')[0], 409)
            finally:
                gate.set()
            initial = self.ready(route)
            revision = initial['revision']
            self.assertTrue(initial['b']['images']['suspected_objects'])
            status, png = self.request('GET', route + f'/image?view=magnetic_source&revision={revision}')
            self.assertEqual(status, 200)
            with Image.open(BytesIO(png)) as edge_source:
                self.assertEqual(edge_source.size, (40, 32))
                self.assertEqual(edge_source.info['icc_profile'], icc)
                np.testing.assert_array_equal(np.asarray(edge_source)[..., 3], alpha)
                np.testing.assert_array_equal(np.asarray(edge_source)[..., :3], 95)
            status, png = self.request('GET', route + f'/image?view=suspected_objects&revision={revision}')
            self.assertEqual(status, 200)
            with Image.open(BytesIO(png)) as overlay:
                self.assertEqual(overlay.size, (40, 32))
                self.assertFalse(np.array_equal(np.array(overlay)[12, 18, :3], rgb[12, 18]))
            self.assertEqual(post(route + '/pin', {'revision': revision})[0], 200)
            settings = {**initial['settings'], 'method': 'esrgan-l0', 'modules': ['skin', 'background'],
                        'restoration_gain': 0}
            edit = {'tool': 'include', 'shape': 'brush', 'points': [[18, 12]], 'radius': 2}
            update = {'revision': revision, 'settings': settings, 'edits': [edit], 'cursor': 1}
            self.assertEqual(post(route + '/edit', update)[0], 200)
            self.assertEqual(post(route + '/edit', update)[0], 409)
            self.assertEqual(self.request('GET', route + f'/download?view=b&revision={revision}')[0], 409)
            self.assertEqual(self.request('GET', route + f'/download?view=a&revision={revision}')[0], 200)
            current = self.ready(route)
            self.assertEqual(self.request('GET', route + f'/image?view=magnetic_source&revision={revision}')[0], 409)
            self.assertEqual(current['edit_feedback'], {'tool': 'include', 'changed_pixels': 0,
                                                        'reason': 'already_selected'})
            analyze.assert_called_once()
        self.assertEqual([call.args[0] for call in candidate.call_args_list], ['l0', 'esrgan-l0'])
        status, png = self.request('GET', route + f"/download?view=b&revision={current['revision']}")
        self.assertEqual(status, 200)
        with Image.open(BytesIO(png)) as im:
            result = np.array(im)
            self.assertEqual(im.info['icc_profile'], icc)
        np.testing.assert_array_equal(result[..., 3], alpha)
        protected = objects | (alpha == 0)
        np.testing.assert_array_equal(result[..., :3][protected], rgb[protected])
        self.assertTrue(np.any(result[..., :3][suspected] != rgb[suspected]))
        self.assertTrue(np.any(result[..., :3][~protected] != rgb[~protected]))
