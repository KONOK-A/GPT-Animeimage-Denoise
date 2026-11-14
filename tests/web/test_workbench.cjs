const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const web = path.resolve(__dirname, '../../src/gpt_animeimage_denoise/workbench/web');
const html = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
const script = fs.readFileSync(path.join(web, 'app.js'), 'utf8');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.handlers = new Map();
    this.style = {}; this.dataset = {}; this.value = ''; this.width = this.clientWidth = 800;
    this.height = this.clientHeight = 640; this.classList = { add() {}, remove() {} };
  }
  append(...children) {
    for (const child of children) {
      if (typeof child === 'object') child.parent = this;
      this.children.push(child);
    }
  }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  get nextElementSibling() {
    return this.parent.children.slice(this.parent.children.indexOf(this) + 1).find(child => typeof child === 'object');
  }
  addEventListener(type, listener) { this.handlers.set(type, listener); }
  setAttribute(key, value) { this[key] = value; }
  click() { return this.fire('click'); }
  remove() {}
  checkValidity() { return true; }
  reportValidity() {}
  setPointerCapture() {}
  focus() { this.focused = true; }
  fire(type, event = {}) { return this.handlers.get(type)?.({ target: this, type, ...event }); }
  getBoundingClientRect() { return { left: 100, top: 50, width: this.clientWidth, height: this.clientHeight }; }
  getContext() {
    const operations = {
      drawImage: () => { this.imagePaints = (this.imagePaints || 0) + 1; },
      lineTo: (...point) => { (this.lines ||= []).push(point); },
      clearRect: () => { this.lines = []; },
    };
    return new Proxy(operations, { get: (object, key) => object[key] ?? (() => {}) });
  }
}

const options = {
  modules: ['skin', 'face', 'hair', 'eyes', 'mouth', 'outfit', 'background'].map(id => ({ id, label: id })),
  default_modules: ['skin', 'face', 'hair'],
  methods: [{ id: 'l0', label: '纯 L0', uses_l0: true, uses_downscale: false },
    { id: 'esrgan-l0', label: 'ESRGAN + L0', uses_l0: true, uses_downscale: true }],
  defaults: { method: 'l0', l0: .0005, downscale: .6, strength: 1, restoration_gain: .25, sigma: 12, structure_protection: .2 },
};
const file = { name: '插画.png', size: 23, type: 'image/png' };
const tick = () => new Promise(setImmediate);
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}
function harness({ theme, prefersDark = false } = {}) {
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(([, id]) => [id, new Element()]));
  const nodes = {};
  for (const [key, values] of [
    ['view', [...html.matchAll(/<button data-view="([^"]+)"/g)].map(([, view]) => view)],
    ['tool', [...html.matchAll(/<button data-tool="([^"]+)"/g)].map(([, tool]) => tool)],
    ['preset', ['standard', 'soft', 'strong']], ['mode', ['review', 'range', 'effect']],
    ['scope', ['auto', 'empty', 'all']], ['theme-option', ['system', 'light', 'dark']],
  ]) {
    nodes[key] = values.map(value => {
      const button = new Element('button');
      const datasetKey = key === 'theme-option' ? 'themeOption' : key;
      button.dataset[datasetKey] = value;
      return button;
    });
  }
  for (const name of ['strength', 'restoration_gain', 'structure_protection', 'radius', 'protection']) elements.get(name).type = 'range';
  elements.get('radius').value = '24';
  elements.get('magnetic-radius').value = '10';
  elements.get('protection').value = '.9';
  elements.get('lasso-action').value = 'include';
  elements.get('stage').dataset.tool = 'pan';
  let doc = {
    id: 'source', name: file.name, width: 640, height: 480, revision: 0,
    settings: { ...options.defaults, modules: options.default_modules, part_strengths: {} },
    edits: [], cursor: 0, status: 'queued', phase: 'analysis', error: null, b: null, a: null,
  };
  const frames = [];
  const api = { requests: [], workers: [], beforeOpen: null, beforeEdit: null, beforeDecode: null, beforeSam: null,
    beforeChanges: null, changeCounts: { added: 0, removed: 0, stronger: 0, weaker: 0, reclassified: 0 },
    manualFrames: false, beforeMagnetic: null, magneticStatus: 200,
    frame: () => { for (const callback of frames.splice(0)) callback(); },
    mutate: change => { doc = { ...doc, ...change }; },
    ready: () => { doc = { ...doc, status: 'ready', b: { revision: doc.revision, method: doc.settings.method } }; },
  };
  const body = new Element('body'), win = new Element();
  const root = new Element('html'), scheme = new Element();
  if (theme) root.dataset.themePreference = theme;
  scheme.matches = prefersDark;
  const storage = new Map();
  const preferences = new Map(theme ? [['gaid-theme', theme]] : []);
  const sandbox = vm.createContext({
    document: {
      getElementById: id => elements.get(id), createElement: tag => new Element(tag), body, documentElement: root,
      querySelectorAll(selector) {
        const match = /^(button)?\[data-(.+)\]$/.exec(selector);
        if (match) return match[2] === 'tool' && !match[1]
          ? [...nodes.tool, elements.get('stage')] : nodes[match[2]];
        const groups = selector.startsWith('#modules') ? ['modules', ...(selector.includes('#part-strengths') ? ['part-strengths'] : [])]
          : ['part-strengths'];
        const inputs = groups.flatMap(group => elements.get(group).children.flatMap(label => label.children).filter(node => node?.tagName === 'INPUT'));
        return selector.endsWith(':checked') ? inputs.filter(input => input.checked) : inputs;
      },
    },
    window: {
      devicePixelRatio: 1, addEventListener: (...args) => win.addEventListener(...args),
      matchMedia: () => scheme,
      localStorage: { setItem: (key, value) => preferences.set(key, value), removeItem: key => preferences.delete(key) },
      sessionStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) },
    }, file,
    Image: class {
      constructor() { this.naturalWidth = 640; this.naturalHeight = 480; }
      async decode() { await api.beforeDecode?.(this.src); }
    },
    ResizeObserver: class { observe() {} },
    AbortController,
    Worker: class {
      constructor(url) { this.url = url; this.messages = []; api.workers.push(this); }
      postMessage(message) { this.messages.push(message); }
      terminate() { this.terminated = true; }
      reply(data) { this.onmessage({ data }); }
    },
    createImageBitmap: async image => ({ width: image.naturalWidth, height: image.naturalHeight, source: image.src, close() {} }),
    requestAnimationFrame: callback => api.manualFrames ? frames.push(callback) : setImmediate(callback),
    URL: { createObjectURL: blob => 'blob:' + (typeof blob === 'string' ? blob : 'file'), revokeObjectURL() {} },
    setTimeout: () => 1, clearTimeout() {},
    fetch: async (route, args) => {
      api.requests.push({ route, args });
      const response = (data, status = 200) => ({
        ok: status === 200, status, json: async () => JSON.parse(JSON.stringify(data)), blob: async () => route,
      });
      if (route === '/api/options') return response(options);
      if (route.includes('/changes?revision=')) {
        const revision = Number(route.split('=').at(-1)), counts = { ...api.changeCounts };
        await api.beforeChanges;
        return response({ revision, counts, overlay: Object.values(counts).some(Boolean) ? 'data:image/png;base64,AAAA' : null });
      }
      if (route.includes('view=magnetic_source')) {
        await api.beforeMagnetic;
        return response({}, api.magneticStatus);
      }
      if (route === '/api/documents') {
        if (args.method === 'POST') { if (api.beforeOpen) await api.beforeOpen; return response(doc); }
        return response([]);
      }
      if (route.endsWith('/edit')) {
        if (api.beforeEdit) await api.beforeEdit;
        if (api.editError) return response({ error: api.editError }, 500);
        const { revision, ...draft } = JSON.parse(args.body);
        if (revision !== doc.revision) return response({ error: 'Conflict' }, 409);
        doc = { ...doc, ...draft, revision: revision + 1, status: 'queued' };
        return response(doc);
      }
      if (route.endsWith('/sam-object')) {
        if (api.beforeSam) await api.beforeSam;
        const { revision } = JSON.parse(args.body);
        if (revision !== doc.revision) return response({ error: 'Conflict' }, 409);
        return response({ revision, token: 'object-preview', preview: 'data:image/png;base64,AAAA', pixels: 100 });
      }
      if (route.endsWith('/sam-object/apply')) {
        const { revision, token } = JSON.parse(args.body);
        if (revision !== doc.revision || token !== 'object-preview') return response({ error: 'Conflict' }, 409);
        const edits = [...doc.edits.slice(0, doc.cursor), { tool: 'sam_object', token }];
        doc = { ...doc, edits, cursor: edits.length, revision: revision + 1, status: 'queued' };
        return response(doc);
      }
      if (route.endsWith('/pin')) { doc = { ...doc, a: doc.b }; return response(doc); }
      return response(doc);
    },
  });
  const run = code => vm.runInContext(code, sandbox);
  run(script);
  return { elements, nodes, run, api, window: win, storage, root, scheme, preferences };
}

test('appearance follows the system, supports manual overrides and can return to following the system', () => {
  const h = harness({ prefersDark: true });
  const [system, light, dark] = h.nodes['theme-option'];
  assert.equal(h.root.dataset.theme, 'dark');
  assert.equal(system['aria-pressed'], 'true');
  h.scheme.matches = false;
  h.scheme.fire('change');
  assert.equal(h.root.dataset.theme, 'light');
  dark.click();
  assert.equal(h.root.dataset.theme, 'dark');
  assert.equal(h.preferences.get('gaid-theme'), 'dark');
  h.scheme.fire('change');
  assert.equal(h.root.dataset.theme, 'dark');
  system.click();
  assert.equal(h.root.dataset.theme, 'light');
  assert.equal(h.preferences.has('gaid-theme'), false);
  h.scheme.matches = true;
  h.scheme.fire('change');
  assert.equal(h.root.dataset.theme, 'dark');
  light.click();
  assert.equal(h.root.dataset.theme, 'light');
  assert.equal(h.preferences.get('gaid-theme'), 'light');
  const reopened = harness({ theme: 'light', prefersDark: true });
  assert.equal(reopened.root.dataset.theme, 'light');
  assert.equal(reopened.nodes['theme-option'][1]['aria-pressed'], 'true');
});

test('Space on a disclosure does not trigger the canvas pan shortcut', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  let prevented = false;
  h.window.fire('keydown', { key: ' ', target: new Element('summary'), preventDefault() { prevented = true; } });
  assert.equal(prevented, false);
  assert.equal(h.run('state.space'), false);
});

test('canvas click bubbling does not reset an in-progress polygon', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('lasso'); state.polygon = [[15,16]]");
  h.elements.get('stage').fire('click');
  assert.equal(h.run('state.polygon.length'), 1);
});

test('manual changes are lazy, describe actual counts, and keep the selected tool', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('range'); selectTool('include')");
  assert.equal(h.api.requests.filter(x => x.route.includes('/changes?')).length, 0);
  h.elements.get('show-manual-changes').click();
  await tick();
  assert.equal(h.run('state.view'), 'changes');
  assert.equal(h.run('state.tool'), 'include');
  assert.match(h.elements.get('changes-status').textContent, /无手动改动/);
  h.run("addEdit({tool:'include',points:[[10,10]],radius:5})");
  assert.match(h.elements.get('change-history-list').children[0].children[1].textContent, /待更新/);
  assert.equal(h.api.requests.filter(x => x.route.includes('/changes?')).length, 1);
  await h.run('saveDraft()');
  h.api.changeCounts.added = 32;
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.run("state.images.get('changes').revision"), 1);
  assert.match(h.elements.get('changes-added').textContent, /32/);
  assert.match(h.elements.get('change-history-list').children[0].children[1].textContent, /B 已包含/);
});

test('outdated change responses never install over a newer result', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('range')");
  const pending = deferred();
  h.api.beforeChanges = pending.promise;
  h.api.changeCounts.added = 50;
  h.run("selectView('changes')");
  h.run("addEdit({tool:'include',points:[[10,10]],radius:5})");
  await h.run('saveDraft()');
  h.api.beforeChanges = null;
  h.api.changeCounts.added = 3;
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  pending.resolve();
  await tick();
  assert.equal(h.run("state.images.get('changes').revision"), 1);
  assert.equal(h.run("state.images.get('changes').counts.added"), 3);
});

test('a saved no-op can be located without changing tool, view or edit history', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.mutate({ edits: [{ tool: 'exclude', shape: 'brush', points: [[20, 30]], radius: 10 }], cursor: 1 });
  h.api.ready();
  // Reopen so local history matches the server fixture.
  await h.run("openRecent('source');");
  h.run("selectMode('range'); selectTool('exclude'); selectView('changes')");
  await tick();
  assert.match(h.elements.get('changes-status').textContent, /没有净变化/);
  assert.equal(h.elements.get('change-history-note').hidden, false);
  h.elements.get('change-history-list').children[0].children[2].click();
  assert.equal(h.run('state.historyLocation'), 0);
  assert.equal(h.elements.get('history-location-label').hidden, false);
  assert.equal(h.run('state.tool'), 'exclude');
  assert.equal(h.run('state.view'), 'changes');
  assert.equal(h.run('state.draft.cursor'), 1);
  assert.equal(h.api.requests.filter(x => x.route.endsWith('/edit')).length, 0);
  h.elements.get('change-history-list').children[0].children[2].click();
  assert.equal(h.elements.get('history-location-label').hidden, true);
});

test('the history label records the value and target used by an edit', async () => {
  const h = harness();
  await tick();
  h.run("state.options.modules.find(item => item.id === 'skin').label = '皮肤'");
  assert.match(h.run("editHistoryLabel({tool:'protect',strength:.665})"), /66\.5%/);
  assert.match(h.run("editHistoryLabel({tool:'mix',strength:.3,shape:'polygon'})"), /30%.*圈选/);
  assert.match(h.run("editHistoryLabel({tool:'override',module:'skin'})"), /皮肤/);
});

test('full result stays visible while choosing a tool and quick strength keeps the algorithm', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  h.run("selectMode('range'); selectView('result'); selectTool('include')");
  assert.equal(h.run('state.view'), 'result');
  assert.equal(h.run("visibleImage('result') === visibleImage('b')"), true);
  h.run("state.draft.settings.method = 'esrgan-l0'; state.draft.settings.l0 = .0007; choosePreset('soft')");
  assert.equal(h.run('state.draft.settings.strength'), .65);
  assert.equal(h.run('state.draft.settings.method'), 'esrgan-l0');
  assert.equal(h.run('state.draft.settings.l0'), .0007);
});

test('brush mode has explicit controls and only fetches the visible mask after edits', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  h.run("selectTool('include')");
  assert.equal(h.run('state.mode'), 'review');
  assert.equal(h.run('state.tool'), 'pan');
  h.run("selectMode('range'); selectTool('exclude')");
  assert.match(h.elements.get('tool-description').textContent, /保留原像素/);
  stroke(h, [[200,120]]);
  await h.run('saveDraft()');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.ok(h.api.requests.some(item => item.route.endsWith('/image?view=weight&revision=1')));
  assert.equal(h.api.requests.some(item => item.route.endsWith('/image?view=b&revision=1')), false);
  assert.match(h.elements.get('status').textContent, /范围已更新/);
  h.run("selectMode('review')");
  await tick();
  assert.equal(h.run('state.tool'), 'pan');
  assert.ok(h.api.requests.some(item => item.route.endsWith('/image?view=b&revision=1')));
});

test('suspected objects are an optional diagnostic view that leaves brush edits available', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('range'); selectTool('include')");
  const tab = h.nodes.view.find(button => button.dataset.view === 'suspected_objects');
  assert.ok(tab, 'the range toolbar includes a suspected-object view');
  assert.equal(tab.hidden, true);
  h.run("selectView('suspected_objects')");
  assert.equal(h.run('state.view'), 'weight');

  h.api.mutate({ b: { revision: 0, method: 'l0', images: { suspected_objects: true } } });
  await h.run('pollDocument()');
  assert.equal(tab.hidden, false);
  assert.equal(h.api.requests.some(item => item.route.includes('view=suspected_objects')), false);
  const decode = deferred();
  h.api.beforeDecode = url => url.includes('view=suspected_objects') ? decode.promise : undefined;
  tab.click();
  assert.equal(h.run('state.view'), 'suspected_objects');
  assert.equal(h.run('state.draft.cursor'), 0);
  assert.equal(h.elements.get('suspected-objects-legend').hidden, true);
  decode.resolve();
  await tick();
  assert.ok(h.api.requests.some(item => item.route.endsWith('/image?view=suspected_objects&revision=0')));
  assert.equal(h.elements.get('suspected-objects-legend').hidden, false);
  assert.equal(h.elements.get('overlay-legend').hidden, true);
  assert.match(h.elements.get('status').textContent, /疑似外物已显示/);
  stroke(h, [[200, 120]]);
  assert.equal(h.run('state.draft.cursor'), 1);
  assert.equal(h.run('state.view'), 'suspected_objects');
  await h.run('saveDraft()');
  const edit = h.api.requests.find(item => item.route.endsWith('/edit'));
  assert.deepEqual(JSON.parse(edit.args.body).edits, [{ tool:'include', shape:'brush', points:[[200,120]], radius:24 }]);

  h.api.ready();
  await h.run('pollDocument()');
  assert.equal(tab.hidden, true);
  assert.equal(h.run('state.view'), 'weight');
});

function stroke(h, points) {
  const rect = h.run('layout()');
  const canvas = h.elements.get('canvas');
  const event = ([x, y]) => ({ pointerId: 1, button: 0, clientX: 100 + rect.x + x * rect.scale,
    clientY: 50 + rect.y + y * rect.scale, preventDefault() {} });
  canvas.fire('pointerdown', event(points[0]));
  for (const point of points.slice(1)) canvas.fire('pointermove', event(point));
  canvas.fire('pointerup', event(points.at(-1)));
}

test('SAM2 point selection previews original-grid prompts and applies one undoable edit', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('range'); selectTool('sam_object'); zoomTo(2)");
  assert.equal(h.run('state.view'), 'original');
  assert.equal(h.elements.get('sam-panel').hidden, false);
  assert.equal(h.elements.get('sam-apply').disabled, true);
  stroke(h, [[285, 215]]);
  h.elements.get('sam-negative').click();
  stroke(h, [[310, 235]]);
  assert.equal(h.api.requests.some(item => item.route.endsWith('/sam-object')), false);
  await h.elements.get('sam-preview').click();
  const preview = h.api.requests.find(item => item.route.endsWith('/sam-object'));
  assert.deepEqual(JSON.parse(preview.args.body), { revision: 0, positive: [[285, 215]], negative: [[310, 235]] });
  assert.equal(h.elements.get('sam-apply').disabled, false);
  assert.equal(h.run('state.draft.cursor'), 0);
  await h.elements.get('sam-apply').click();
  assert.equal(h.run("state.draft.edits[0].tool"), 'sam_object');
  assert.equal(h.run('state.draft.cursor'), 1);
  assert.equal(h.run('state.view'), 'parts');
  h.elements.get('undo').click();
  await h.run('flushChanges()');
  assert.equal(h.run('state.draft.cursor'), 0);
});

test('SAM2 previews cannot be applied after cancellation or a remote revision change', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('range'); selectTool('sam_object')");
  stroke(h, [[295, 215]]);
  const gate = deferred();
  h.api.beforeSam = gate.promise;
  const pending = h.elements.get('sam-preview').click();
  h.elements.get('sam-cancel').click();
  gate.resolve();
  await pending;
  assert.equal(h.elements.get('sam-panel').hidden, true);
  assert.equal(h.elements.get('sam-apply').disabled, true);
  h.run("selectTool('sam_object')");
  stroke(h, [[295, 215]]);
  await h.elements.get('sam-preview').click();
  assert.equal(h.elements.get('sam-apply').disabled, false);
  h.api.mutate({ revision: 1 });
  h.api.ready();
  await h.run('pollDocument()');
  assert.equal(h.elements.get('sam-apply').disabled, true);
  assert.equal(h.api.requests.some(item => item.route.endsWith('/sam-object/apply')), false);
});

test('shows local original during upload and keeps editing during analysis, then gates export by revision', async () => {
  const h = harness();
  await tick();
  const opened = deferred();
  h.api.beforeOpen = opened.promise;
  const opening = h.run('openFile(file)');
  await tick();
  assert.equal(h.run("Boolean(visibleImage('original'))"), true);
  assert.equal(h.run('state.doc'), null);
  assert.equal(h.elements.get('zoom-in').disabled, false);
  opened.resolve();
  await opening;
  const upload = h.api.requests.find(item => item.route === '/api/documents' && item.args.method === 'POST');
  assert.equal(upload.args.body, file);
  assert.equal(upload.args.headers['X-GAID-Filename'], encodeURIComponent(file.name));
  assert.equal(upload.args.headers['X-GAID-Request'], '1');
  assert.equal(h.elements.get('download').disabled, true);
  assert.equal(h.run('state.doc.status'), 'queued');

  h.run("zoomTo(2); selectMode('range'); selectTool('protect')");
  stroke(h, [[200,120],[207,122]]);
  assert.equal(h.run('state.draft.cursor'), 1);
  await h.run('saveDraft()');
  const edit = h.api.requests.find(item => item.route.endsWith('/edit'));
  assert.deepEqual(JSON.parse(edit.args.body).edits, [{ tool:'protect', shape:'brush', points:[[200,120],[207,122]], radius:24, strength:.9 }]);
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  h.run("selectMode('review'); selectView('compare')");
  await tick();
  await h.run('pin()');
  await tick();
  assert.equal(h.run('comparing()'), true);
  assert.equal(h.elements.get('download').disabled, false);
  await h.run('download()');
  assert.equal(h.api.requests.filter(item => item.route.endsWith('/download?view=b&revision=1')).length, 1);

  h.run("choosePreset('strong'); undo()");
  assert.equal(h.run('state.draft.cursor'), 0);
  assert.equal(h.run('state.draft.settings.l0'), .001);
  assert.equal(h.elements.get('download').disabled, true);
  await h.run('download()');
  await h.run("download('a')");
  assert.equal(h.api.requests.filter(item => item.route.endsWith('/download?view=b&revision=1')).length, 1);
  assert.equal(h.api.requests.filter(item => item.route.endsWith('/download?view=a&revision=1')).length, 1);
  h.run('redo()');
  assert.equal(h.run('state.draft.cursor'), 1);
});

test('a concurrent edit is submitted after the prior save, and a conflicting tab never erases the draft', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  const delay = deferred();
  h.api.beforeEdit = delay.promise;
  h.run("choosePreset('soft')");
  const saving = h.run('saveDraft()');
  h.run("choosePreset('strong')");
  delay.resolve();
  await saving;
  h.api.beforeEdit = null;
  assert.equal(h.run('state.draft.settings.l0'), .001);
  assert.equal(h.run('state.dirty'), true);
  await h.run('saveDraft()');
  const edits = h.api.requests.filter(item => item.route.endsWith('/edit')).map(item => JSON.parse(item.args.body));
  assert.deepEqual(edits.map(data => [data.revision, data.settings.strength, data.settings.l0]), [[0,.65,.0005],[1,.65,.001]]);

  const strength = h.elements.get('strength');
  strength.value = '.55';
  strength.fire('input');
  h.api.mutate({ revision: 4 });
  await h.run('saveDraft()');
  assert.equal(h.run('state.conflict.revision'), 4);
  assert.equal(h.run('state.draft.settings.strength'), .55);
  assert.equal(h.elements.get('conflict-local').hidden, false);
  h.elements.get('conflict-local').click();
  await h.run('saveDraft()');
  const last = h.api.requests.filter(item => item.route.endsWith('/edit')).at(-1);
  assert.equal(JSON.parse(last.args.body).revision, 4);
  assert.equal(JSON.parse(last.args.body).settings.strength, .55);
  assert.equal(h.run('state.conflict'), null);
});

test('lasso closes explicitly, keeps source coordinates, and remains undoable during work', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("zoomTo(2); selectMode('range'); selectTool('lasso')");
  h.elements.get('lasso-action').value = 'override';
  h.elements.get('override-module').value = 'hair';
  const rect = h.run('layout()');
  const canvas = h.elements.get('canvas');
  for (const [x, y] of [[125, 105], [140, 115], [132, 120]]) {
    canvas.fire('pointerdown', { button: 0, clientX: 100 + rect.x + x * rect.scale,
      clientY: 50 + rect.y + y * rect.scale, preventDefault() {} });
  }
  assert.equal(h.elements.get('stage').focused, true);
  assert.equal(h.run('state.draft.cursor'), 0);
  h.window.fire('keydown', { key: 'Enter', preventDefault() {} });
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits)')), [{
    tool: 'override', shape: 'polygon', points: [[125,105],[140,115],[132,120]], module: 'hair',
  }]);
  h.elements.get('invert').click();
  h.run('undo()');
  await h.run('saveDraft()');
  const payload = JSON.parse(h.api.requests.find(item => item.route.endsWith('/edit')).args.body);
  assert.equal(payload.cursor, 1);
  assert.deepEqual(payload.edits[1], { tool: 'invert' });
  h.run('redo()');
  assert.equal(h.run('state.draft.cursor'), 2);
});

function magneticPointer(h, event, [x, y], extras = {}) {
  const box = h.run('layout()');
  h.elements.get('canvas').fire(event, { button: 0, clientX: 100 + box.x + x * box.scale,
    clientY: 50 + box.y + y * box.scale, preventDefault() {}, ...extras });
}
function traceReply(worker, points, snapped = true) {
  const { id } = worker.messages.at(-1);
  worker.reply({ type: 'trace', id, points, snapped });
}
test('magnetic lasso traces its closing edge and saves a single undoable polygon', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); zoomTo(2); selectTool('magnetic')");
  await tick();
  assert.equal(h.elements.get('range-scope').hidden, true);
  assert.equal(h.elements.get('brush-panel').hidden, false);
  const worker = h.api.workers[0];
  assert.equal(worker.url, '/magnetic-lasso-worker.js');
  assert.equal(worker.messages[0].type, 'init');
  worker.reply({ type: 'ready', width: 640, height: 480 });
  h.elements.get('lasso-action').value = 'exclude';
  magneticPointer(h, 'pointerdown', [315, 215]);
  assert.equal(h.elements.get('stage').focused, true);
  traceReply(worker, [[316, 214]]);
  magneticPointer(h, 'pointerdown', [360, 235]);
  traceReply(worker, [[316, 214], [335, 220], [360, 235]]);
  magneticPointer(h, 'pointerdown', [320, 270]);
  traceReply(worker, [[360, 235], [320, 270]]);
  assert.equal(h.run('state.draft.cursor'), 0);
  h.window.fire('keydown', { key: 'Enter', preventDefault() {} });
  assert.equal(worker.messages.at(-1).exact, true);
  assert.deepEqual(JSON.parse(JSON.stringify(worker.messages.at(-1).point)), [316, 214]);
  traceReply(worker, [[320, 270], [313, 242], [316, 214]]);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits)')), [{
    tool: 'exclude', shape: 'polygon', points: [[316, 214], [335, 220], [360, 235], [320, 270], [313, 242]],
  }]);
  h.run('undo()');
  assert.equal(h.run('state.draft.cursor'), 0);
});

test('magnetic lasso coalesces hover, supports straight anchors and ignores canceled work', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  magneticPointer(h, 'pointerdown', [315, 215], { altKey: true });
  assert.equal(worker.messages.length, 1);
  magneticPointer(h, 'pointermove', [350, 250]);
  await tick();
  magneticPointer(h, 'pointermove', [360, 250]);
  magneticPointer(h, 'pointermove', [370, 250]);
  assert.equal(worker.messages.length, 2);
  traceReply(worker, [[315, 215], [350, 250]]);
  await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(worker.messages.at(-1).point)), [370, 250]);
  h.window.fire('keydown', { key: 'Escape', preventDefault() {} });
  traceReply(worker, [[315, 215], [370, 250]]);
  assert.equal(h.run('state.draft.cursor'), 0);
  assert.equal(h.elements.get('close-lasso').disabled, true);
  h.run("selectTool('pan')");
  assert.equal(worker.terminated, true);
});

test('magnetic hover traces once per frame, retains its curve, and does not repaint the photo', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.manualFrames = true;
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  magneticPointer(h, 'pointerdown', [300, 200], { altKey: true });
  const canvas = h.elements.get('canvas');
  const paints = canvas.imagePaints;
  for (const point of [[308, 208], [315, 214], [320, 220]]) magneticPointer(h, 'pointermove', point);
  assert.equal(worker.messages.length, 1);
  h.api.frame();
  assert.deepEqual(JSON.parse(JSON.stringify(worker.messages.at(-1).point)), [320, 220]);
  traceReply(worker, [[300, 200], [308, 215], [320, 220]]);
  h.api.frame();
  const overlay = h.elements.get('magnetic-canvas');
  const lines = overlay.lines.slice();
  assert.ok(lines.length > 1);
  magneticPointer(h, 'pointermove', [360, 238]);
  h.api.frame();
  assert.ok(lines.every(point => overlay.lines.some(other => JSON.stringify(point) === JSON.stringify(other))));
  assert.equal(canvas.imagePaints, paints);
});

test('magnetic lasso floors source pixels and draws their centers without changing ordinary lasso', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); zoomTo(2); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  magneticPointer(h, 'pointerdown', [300.8, 200.9]);
  assert.deepEqual(JSON.parse(JSON.stringify(worker.messages.at(-1).point)), [300, 200]);
  traceReply(worker, [[300, 200]]);
  magneticPointer(h, 'pointermove', [350.9, 235.8]);
  await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(worker.messages.at(-1).point)), [350, 235]);
  traceReply(worker, [[300, 200], [350, 235]]);
  await tick();
  const box = h.run('layout()');
  assert.ok(h.elements.get('magnetic-canvas').lines.some(([x, y]) =>
    x === box.x + 350.5 * box.scale && y === box.y + 235.5 * box.scale));
  h.run("selectTool('lasso')");
  magneticPointer(h, 'pointerdown', [300.8, 200.9]);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.polygon)')), [[301, 201]]);
});

test('magnetic lasso caches the current denoised candidate and freezes its source', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  const candidate = deferred();
  h.api.beforeMagnetic = candidate.promise;
  h.run("selectMode('range'); selectTool('magnetic')");
  assert.match(h.elements.get('magnetic-status').textContent, /降噪候选/);
  candidate.resolve();
  await tick();
  assert.match(h.api.workers[0].messages[0].bitmap.source, /view=magnetic_source&revision=0/);
  h.run("selectTool('pan'); selectTool('magnetic')");
  await tick();
  assert.match(h.api.workers[1].messages[0].bitmap.source, /view=magnetic_source&revision=0/);
  h.api.mutate({ revision: 1 });
  h.api.ready();
  await h.run('pollDocument()');
  assert.equal(h.api.requests.filter(item => item.route.includes('view=magnetic_source')).length, 1);
  assert.equal(h.api.workers[1].messages.filter(message => message.type === 'init').length, 1);
  h.run("selectTool('pan'); selectTool('magnetic')");
  await tick();
  assert.match(h.api.workers[2].messages[0].bitmap.source, /view=magnetic_source&revision=0/);
  assert.equal(h.api.requests.filter(item => item.route.includes('view=magnetic_source')).length, 1);
  h.api.mutate({ revision: 2, settings: { ...JSON.parse(h.run('JSON.stringify(state.doc.settings)')), sigma: 14 } });
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectTool('pan'); selectTool('magnetic')");
  await tick();
  assert.match(h.api.workers[3].messages[0].bitmap.source, /view=magnetic_source&revision=2/);
});

test('magnetic lasso falls back to the normalized original when the candidate is busy', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  h.api.magneticStatus = 409;
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  assert.match(worker.messages[0].bitmap.source, /view=original/);
  worker.reply({ type: 'ready', width: 640, height: 480 });
  assert.match(h.elements.get('magnetic-status').textContent, /原图/);
});

test('magnetic reentry reuses a decoded candidate during a brush-only update', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  assert.match(h.api.workers[0].messages[0].bitmap.source, /view=magnetic_source/);
  h.run("state.draft.settings.modules = ['hair']; state.dirty = true; selectTool('pan'); selectTool('magnetic')");
  await tick();
  assert.match(h.api.workers[1].messages[0].bitmap.source, /view=magnetic_source/);
  h.run("state.draft.settings.sigma = 14; selectTool('pan'); selectTool('magnetic')");
  await tick();
  assert.match(h.api.workers[2].messages[0].bitmap.source, /view=original/);
  assert.equal(h.api.requests.filter(item => item.route.includes('view=magnetic_source')).length, 1);
});

test('an unavailable magnetic worker requires Alt to commit deliberate straight anchors', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("Worker = undefined; selectMode('range'); selectTool('magnetic')");
  assert.match(h.elements.get('magnetic-status').textContent, /重新启动工作台/);
  magneticPointer(h, 'pointerdown', [315, 215]);
  assert.equal(h.run('state.magnetic.segments.length'), 0);
  for (const point of [[315, 215], [370, 230], [360, 280]]) magneticPointer(h, 'pointerdown', point, { altKey: true });
  h.window.fire('keydown', { key: 'Backspace', preventDefault() {} });
  assert.equal(h.elements.get('close-lasso').disabled, true);
  magneticPointer(h, 'pointerdown', [310, 280], { altKey: true });
  magneticPointer(h, 'pointerdown', [315, 215], { altKey: true });
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits[0].points)')), [[315, 215], [370, 230], [310, 280]]);
});

test('a normal magnetic click waits for worker readiness and retains its original coordinates', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  magneticPointer(h, 'pointerdown', [315, 215]);
  assert.equal(worker.messages.length, 1);
  assert.equal(h.run('state.magnetic.segments.length'), 0);
  assert.match(h.elements.get('magnetic-status').textContent, /准备吸边/);
  worker.reply({ type: 'ready', width: 640, height: 480 });
  assert.equal(worker.messages.at(-1).anchor, null);
  assert.deepEqual(JSON.parse(JSON.stringify(worker.messages.at(-1).point)), [315, 215]);
  traceReply(worker, [[316, 214]]);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.magnetic.segments)')), [[[316, 214]]]);
});

test('a worker error does not turn its pending magnetic anchor into a straight segment', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  magneticPointer(h, 'pointerdown', [315, 215]);
  traceReply(worker, [[316, 214]]);
  magneticPointer(h, 'pointerdown', [340, 245]);
  worker.reply({ type: 'error', id: worker.messages.at(-1).id, message: 'worker read failed' });
  magneticPointer(h, 'pointerdown', [355, 250]);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.magnetic.segments)')), [[[316, 214]]]);
  assert.match(h.elements.get('magnetic-status').textContent, /重新启动工作台/);
  assert.equal(worker.terminated, true);
});

test('magnetic lasso labels an unsupported edge separately from the Alt straight preview', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  magneticPointer(h, 'pointerdown', [315, 215]);
  traceReply(worker, [[315, 215]]);
  magneticPointer(h, 'pointermove', [350, 245]);
  await tick();
  traceReply(worker, [[315, 215], [350, 245]], false);
  await tick();
  assert.match(h.elements.get('magnetic-status').textContent, /没有找到连续边缘/);
  magneticPointer(h, 'pointermove', [350, 245], { altKey: true });
  await tick();
  assert.match(h.elements.get('magnetic-status').textContent, /Alt.*直线/);
  assert.equal(h.run('state.magnetic.segments.length'), 1);
});

test('magnetic lasso draws a nearby completed hover but commits only the clicked point', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  magneticPointer(h, 'pointerdown', [315, 215], { altKey: true });
  magneticPointer(h, 'pointermove', [350, 250]);
  await tick();
  magneticPointer(h, 'pointermove', [354, 250]);
  traceReply(worker, [[315, 215], [349, 251]]);
  assert.equal(h.run('state.magnetic.preview?.points.at(-1)[0]'), 349);
  await tick();
  magneticPointer(h, 'pointerdown', [354, 250]);
  traceReply(worker, [[315, 215], [353, 250]]);
  assert.equal(h.run('state.magnetic.segments.length'), 1);
  traceReply(worker, [[315, 215], [354, 251]]);
  assert.equal(h.run('state.magnetic.segments.at(-1).at(-1)[0]'), 354);
});

test('undoing a pending magnetic anchor preserves the previous confirmed anchor', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  magneticPointer(h, 'pointerdown', [315, 215], { altKey: true });
  magneticPointer(h, 'pointerdown', [350, 250]);
  h.window.fire('keydown', { key: 'Backspace', preventDefault() {} });
  traceReply(worker, [[315, 215], [350, 250]]);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.magnetic.segments)')), [[[315, 215]]]);
});

test('magnetic lasso keeps the turns of a continuous trace and saves only at closure', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  magneticPointer(h, 'pointerdown', [300, 200]);
  traceReply(worker, [[300, 200]]);
  const hover = async (point, snapped = true) => {
    magneticPointer(h, 'pointermove', point);
    await tick();
    const { anchor } = worker.messages.at(-1);
    traceReply(worker, [anchor, point], snapped);
  };
  for (const point of [[300, 218], [300, 236], [300, 254]]) await hover(point);
  assert.equal(h.run('state.magnetic.segments.length'), 2);
  for (const point of [[318, 254], [336, 254], [354, 254]]) await hover(point);
  assert.deepEqual(JSON.parse(JSON.stringify(worker.messages.at(-1).anchor)), [300, 254]);
  assert.equal(h.run('state.magnetic.segments.length'), 3);
  h.window.fire('keydown', { key: 'Backspace', preventDefault() {} });
  assert.equal(h.run('state.magnetic.segments.length'), 2);
  for (const point of [[300, 254], [318, 254], [336, 254], [354, 254], [354, 236], [354, 218], [354, 200]])
    await hover(point);
  assert.equal(h.run('state.draft.cursor'), 0);
  h.window.fire('keydown', { key: 'Enter', preventDefault() {} });
  assert.equal(worker.messages.at(-1).exact, true);
  traceReply(worker, [[354, 200], [300, 200]]);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits[0].points)')),
    [[300, 200], [300, 254], [354, 254], [354, 200]]);
  h.run('undo()');
  assert.equal(h.run('state.draft.cursor'), 0);
});

test('magnetic auto pins ignore unsupported previews, Alt, jumps, and retracing an auto edge', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  magneticPointer(h, 'pointerdown', [300, 200]);
  traceReply(worker, [[300, 200]]);
  const hover = async (point, snapped) => {
    magneticPointer(h, 'pointermove', point);
    await tick();
    traceReply(worker, [worker.messages.at(-1).anchor, point], snapped);
  };
  for (const point of [[318, 200], [336, 200], [354, 200]]) await hover(point, false);
  assert.equal(h.run('state.magnetic.segments.length'), 1);
  for (const point of [[372, 200], [390, 200], [408, 200]]) magneticPointer(h, 'pointermove', point, { altKey: true });
  assert.equal(h.run('state.magnetic.segments.length'), 1);
  for (const point of [[450, 200], [468, 200], [486, 200], [504, 200]]) await hover(point, true);
  assert.equal(h.run('state.magnetic.segments.length'), 1);
  await hover([300, 200], false);
  for (const point of [[300, 240], [320, 240], [340, 240]]) await hover(point, true);
  assert.equal(h.run('state.magnetic.segments.length'), 1, 'a shortcut across the turn does not become an anchor');
  await hover([300, 200], false);
  for (const point of [[318, 200], [336, 200], [354, 200]]) await hover(point, true);
  assert.equal(h.run('state.magnetic.segments.length'), 2);
  for (const point of [[336, 200], [318, 200], [300, 200]]) await hover(point, true);
  assert.equal(h.run('state.magnetic.segments.length'), 2);
  h.window.fire('keydown', { key: 'Escape', preventDefault() {} });
  assert.equal(h.run('state.magnetic.segments.length'), 0);
  assert.equal(h.run('state.draft.cursor'), 0);
});

test('magnetic lasso does not auto-pin after retreating over more than one confirmed edge', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  for (const point of [[300, 200], [360, 200], [384, 200]]) {
    magneticPointer(h, 'pointerdown', point);
    const { anchor } = worker.messages.at(-1);
    traceReply(worker, anchor ? [anchor, point] : [point]);
  }
  for (const x of [376, 368, 360, 352, 344, 336, 328]) {
    magneticPointer(h, 'pointermove', [x, 200]);
    await tick();
    traceReply(worker, [worker.messages.at(-1).anchor, [x, 200]]);
  }
  assert.equal(h.run('state.magnetic.segments.length'), 3);
});

test('magnetic closure keeps the last exact snapped preview with Enter or the close button', async () => {
  for (const finish of ['enter', 'button']) {
    const h = harness();
    await tick();
    await h.run('openFile(file)');
    h.run("selectMode('range'); selectTool('magnetic')");
    await tick();
    const worker = h.api.workers[0];
    worker.reply({ type: 'ready', width: 640, height: 480 });
    magneticPointer(h, 'pointerdown', [300, 200]);
    traceReply(worker, [[300, 200]]);
    magneticPointer(h, 'pointerdown', [320, 255]);
    traceReply(worker, [[300, 200], [320, 255]]);
    magneticPointer(h, 'pointermove', [343, 263]);
    await tick();
    traceReply(worker, [[320, 255], [330, 272], [343, 263]]);
    await tick();
    assert.equal(h.elements.get('close-lasso').disabled, false);
    if (finish === 'enter') h.window.fire('keydown', { key: 'Enter', preventDefault() {} });
    else {
      h.elements.get('canvas').fire('pointerleave');
      h.elements.get('close-lasso').click();
    }
    assert.deepEqual(JSON.parse(JSON.stringify(worker.messages.at(-1).anchor)), [343, 263]);
    assert.equal(worker.messages.at(-1).exact, true);
    traceReply(worker, [[343, 263], [300, 200]]);
    assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits[0].points)')),
      [[300, 200], [320, 255], [330, 272], [343, 263]]);
  }
});

test('clicking the starting anchor closes directly without appending the last hover', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('magnetic')");
  await tick();
  const worker = h.api.workers[0];
  worker.reply({ type: 'ready', width: 640, height: 480 });
  for (const point of [[300, 200], [320, 255], [355, 262]]) {
    magneticPointer(h, 'pointerdown', point);
    const { anchor } = worker.messages.at(-1);
    traceReply(worker, anchor ? [anchor, point] : [point]);
  }
  magneticPointer(h, 'pointermove', [370, 270]);
  await tick();
  traceReply(worker, [[355, 262], [370, 270]]);
  magneticPointer(h, 'pointerdown', [300, 200]);
  assert.deepEqual(JSON.parse(JSON.stringify(worker.messages.at(-1).anchor)), [355, 262]);
  traceReply(worker, [[355, 262], [300, 200]]);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits[0].points)')),
    [[300, 200], [320, 255], [355, 262]]);
});

test('new B cannot be exported or pinned until its decoded frame replaces the old preview', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.run("state.images.get('b').revision"), 0);

  h.run("choosePreset('soft')");
  await h.run('saveDraft()');
  const delay = deferred();
  h.api.beforeDecode = url => url.endsWith('view=b&revision=1') ? delay.promise : undefined;
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.run('state.doc.b.revision'), 1);
  assert.equal(h.run("state.images.get('b').revision"), 0);
  assert.equal(h.elements.get('download').disabled, true);
  assert.equal(h.elements.get('pin').disabled, true);
  assert.equal(h.elements.get('status').dataset.tone, 'pending');
  assert.match(h.elements.get('right-badge').textContent, /待更新/);
  await h.run('download()');
  await h.run('pin()');
  assert.equal(h.api.requests.some(item => /\/download\?|\/pin$/.test(item.route)), false);

  delay.resolve();
  await tick();
  assert.equal(h.run("state.images.get('b').revision"), 1);
  assert.equal(h.elements.get('download').disabled, false);
  assert.equal(h.elements.get('pin').disabled, false);
  assert.equal(h.elements.get('status').dataset.tone, 'ready');
  await h.run('download()');
  await h.run('pin()');
  assert.equal(h.api.requests.some(item => item.route.endsWith('/download?view=b&revision=1')), true);
  assert.equal(h.api.requests.some(item => item.route.endsWith('/pin')), true);
});

test('command-line correction export can run once the server revision is ready even in a mask view', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range')");
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.run("state.images.has('b')"), false);
  assert.equal(h.elements.get('correction-export').disabled, false);
  await h.run('exportCorrection()');
  assert.equal(h.api.requests.some(item => item.route.endsWith('/corrections/export')), true);
});

test('a result loaded while viewing the original must be shown before it can be pinned or exported', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectView('original')");
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.run("state.images.get('b').revision"), 0);
  assert.equal(h.elements.get('pin').disabled, true);
  assert.equal(h.elements.get('download').disabled, true);
  h.run("selectView('result')");
  assert.equal(h.elements.get('pin').disabled, false);
  assert.equal(h.elements.get('download').disabled, false);
  h.run("selectView('original')");
  assert.equal(h.elements.get('download').disabled, false);
});

test('comparison dragging and clipping map to the visible image at fit and after panning', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  h.run('zoomTo(.5); state.panX = -70; state.panY = 35; draw()');
  const box = h.run('layout()');
  const slider = h.elements.get('divider');
  const drag = x => slider.fire('pointermove', { pointerId: 1, clientX: 100 + x });
  slider.fire('pointerdown', { pointerId: 1, button: 0, clientX: 100 + box.x + box.w / 2 });
  drag(box.x);
  assert.equal(slider['aria-valuenow'], '0');
  assert.equal(slider.style.left, box.x + 'px');
  drag(box.x + box.w / 2);
  assert.equal(slider['aria-valuenow'], '50');
  assert.equal(slider.style.top, box.y + 'px');
  assert.equal(slider.style.height, box.h + 'px');
  drag(box.x + box.w);
  assert.equal(slider['aria-valuenow'], '100');
  assert.equal(slider.style.left, box.x + box.w + 'px');
});

test('only a failed save exposes retry and the draft survives retry', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("choosePreset('soft')");
  assert.equal(h.elements.get('retry').hidden, true);
  h.api.editError = '保存失败';
  await h.run('saveDraft()');
  assert.equal(h.elements.get('status').dataset.tone, 'error');
  assert.equal(h.elements.get('retry').hidden, false);
  assert.equal(h.run('state.draft.settings.strength'), .65);
  h.api.editError = null;
  await h.run('saveDraft()');
  assert.equal(h.elements.get('retry').hidden, true);
  assert.equal(h.run('state.doc.settings.strength'), .65);

  h.run("choosePreset('strong')");
  h.api.editError = '保存失败';
  await h.run('saveDraft()');
  h.api.mutate({ revision: 2 });
  await h.run('pollDocument()');
  h.elements.get('conflict-remote').click();
  assert.equal(h.elements.get('retry').hidden, true);
  assert.equal(h.elements.get('status').dataset.tone, 'pending');
  assert.equal(h.run('state.draft.settings.l0'), .0005);
});

test('imported correction markers can be undone and redone without drawing a stroke', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  const marker = { tool: 'import', token: 'binary-selection' };
  h.api.mutate({ revision: 1, edits: [marker], cursor: 1 });
  await h.run('pollDocument()');
  h.run("selectView('weight'); draw(); undo(); draw()");
  assert.equal(h.run('state.draft.cursor'), 0);
  await h.run('saveDraft()');
  const sent = JSON.parse(h.api.requests.find(item => item.route.endsWith('/edit')).args.body);
  assert.deepEqual(sent.edits, [marker]);
  assert.equal(sent.cursor, 0);
  h.run('redo(); draw()');
  assert.equal(h.run('state.draft.cursor'), 1);
});

test('explains brush behavior and only shows feedback for the latest completed stroke', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('include')");
  assert.match(h.elements.get('tool-description').textContent, /未选区域/);
  assert.equal(h.elements.get('stroke-feedback').hidden, true);

  stroke(h, [[200, 120]]);
  await h.run('saveDraft()');
  h.api.mutate({ edit_feedback: { tool: 'include', changed_pixels: 0, reason: 'already_selected' } });
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.elements.get('stroke-feedback').hidden, false);
  assert.match(h.elements.get('stroke-feedback').textContent, /已选中/);
  stroke(h, [[210, 120]]);
  assert.equal(h.elements.get('stroke-feedback').hidden, true);

  h.run("selectTool('erase')");
  assert.match(h.elements.get('tool-description').textContent, /移出/);
  stroke(h, [[220, 120]]);
  await h.run('saveDraft()');
  h.api.mutate({ edit_feedback: { tool: 'erase', changed_pixels: 0, reason: 'no_marks' } });
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.elements.get('stroke-feedback').hidden, false);
  assert.match(h.elements.get('stroke-feedback').textContent, /没有手动标记/);
});

test('待确认图层按需加载，并在两个范围视图之间复用', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.mutate({ status: 'ready', b: { revision: 0, images: { unknown: true } } });
  await h.run('pollDocument()');
  await tick();
  const toggle = h.elements.get('show-unknown');
  assert.ok(toggle, '待确认图层应有独立开关');
  const requested = () => h.api.requests.filter(item => item.route.includes('view=unknown')).length;
  assert.equal(toggle.hidden, true);
  assert.equal(requested(), 0);
  h.run("selectMode('range')");
  await tick();
  assert.equal(toggle.hidden, false);
  assert.equal(toggle['aria-pressed'], 'true');
  assert.equal(requested(), 1);
  assert.equal(h.elements.get('unknown-legend').hidden, false);
  h.run("selectView('parts')");
  await tick();
  assert.equal(requested(), 1);
  toggle.click();
  assert.equal(h.elements.get('unknown-legend').hidden, true);
  toggle.click();
  assert.equal(h.elements.get('unknown-legend').hidden, false);
  assert.equal(requested(), 1);
  h.run("selectView('original')");
  assert.equal(toggle.hidden, true);
  assert.equal(h.elements.get('unknown-legend').hidden, true);
});

test('可修正最近一次被跳过的待确认笔画，且可撤销', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('include')");
  stroke(h, [[200, 120]]);
  await h.run('saveDraft()');
  h.api.mutate({ edit_feedback: {
    tool: 'include', changed_pixels: 0, reason: 'protected', skipped_unknown_pixels: 320,
  } });
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  const actions = h.elements.get('unknown-stroke-actions');
  assert.ok(actions, '被跳过的待确认区域应有快捷修正');
  assert.equal(actions.hidden, false);
  assert.match(h.elements.get('stroke-feedback').textContent, /跳过 320 个待确认像素/);
  h.elements.get('include-unknown-stroke').click();
  h.elements.get('include-unknown-stroke').click();
  assert.equal(h.run('state.draft.cursor'), 2);
  assert.equal(h.run('JSON.stringify(state.draft.edits[1])'), '{"tool":"resolve_unknown"}');
  assert.equal(actions.hidden, true);
  await h.run('saveDraft()');
  h.elements.get('undo').click();
  await h.run('saveDraft()');
  h.api.ready();
  await h.run('pollDocument()');
  h.elements.get('unknown-stroke-module').value = 'skin';
  h.elements.get('assign-unknown-stroke').click();
  assert.equal(h.run('state.draft.cursor'), 2);
  assert.equal(h.run('JSON.stringify(state.draft.edits[1])'), '{"tool":"resolve_unknown","module":"skin"}');
});

test('快捷修正在另一窗口更新后不会随本地草稿重试', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('include')");
  stroke(h, [[200, 120]]);
  await h.run('saveDraft()');
  h.api.mutate({ edit_feedback: {
    tool: 'include', changed_pixels: 0, reason: 'protected', skipped_unknown_pixels: 15,
  } });
  h.api.ready();
  await h.run('pollDocument()');
  const gate = deferred();
  h.api.beforeEdit = gate.promise;
  h.elements.get('include-unknown-stroke').click();
  assert.equal(h.run('state.draft.cursor'), 2);
  h.api.mutate({ revision: 2 });
  h.api.ready();
  gate.resolve();
  assert.equal(await h.run('saveDraft()'), false);
  assert.equal(h.run('state.conflict.revision'), 2);
  assert.equal(h.run('state.draft.cursor'), 1);
  assert.equal(h.elements.get('unknown-stroke-actions').hidden, true);
  h.elements.get('conflict-local').click();
  await h.run('saveDraft()');
  const payload = JSON.parse(h.api.requests.filter(item => item.route.endsWith('/edit')).at(-1).args.body);
  assert.equal(payload.revision, 2);
  assert.deepEqual(payload.edits.map(edit => edit.tool), ['include']);
});

test('恢复草稿时会丢弃尚未确认且版本已过期的待确认快捷操作', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('include')");
  stroke(h, [[200, 120]]);
  await h.run('saveDraft()');
  const draft = JSON.parse(h.run('JSON.stringify(state.draft)'));
  draft.edits.push({ tool: 'resolve_unknown' });
  draft.cursor++;
  h.storage.set('gaid-pending-source', JSON.stringify({ revision: 1, draft, resolutionIndex: 1 }));
  h.api.mutate({ revision: 2 });
  h.api.ready();
  await h.run("openRecent('source')");
  assert.equal(h.run('state.conflict.revision'), 2);
  assert.equal(h.run('state.draft.cursor'), 1);
  assert.equal(h.run('state.draft.edits.length'), 1);
});

test('manual scope, deliberate override and typed local retention are separate edits', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); chooseScope('empty'); selectTool('include')");
  assert.equal(h.run('state.draft.settings.modules.length'), 0);
  assert.equal(h.run('state.view'), 'weight');
  assert.equal(h.elements.get('review-inspector').hidden, true);
  assert.equal(h.elements.get('range-inspector').hidden, false);
  h.elements.get('allow-protected').checked = true;
  stroke(h, [[190, 125]]);
  h.run("selectTool('mix')");
  const ratio = h.elements.get('retention-percent');
  ratio.value = '67.5';
  ratio.fire('change');
  assert.match(h.elements.get('tool-description').textContent, /67\.5% 原图/);
  stroke(h, [[190, 125]]);
  h.run("selectTool('exclude')");
  stroke(h, [[210, 130]]);
  await h.run('saveDraft()');
  const payload = JSON.parse(h.api.requests.find(item => item.route.endsWith('/edit')).args.body);
  assert.deepEqual(payload.settings.modules, []);
  assert.equal(payload.edits[0].allow_protected, true);
  assert.equal(payload.edits[1].tool, 'mix');
  assert.equal(payload.edits[1].strength, .675);
  assert.equal(payload.edits[2].tool, 'exclude');
});

