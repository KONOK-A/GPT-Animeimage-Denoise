from base64 import b64decode
from http.client import HTTPConnection
from io import BytesIO
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


class WorkbenchHTTPTest(unittest.TestCase):
    def setUp(self):
        from gpt_animeimage_denoise.workbench import create_server

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
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
