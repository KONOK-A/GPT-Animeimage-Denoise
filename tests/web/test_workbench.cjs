const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const web = path.resolve(__dirname, '../../src/gpt_animeimage_denoise/workbench/web');
const html = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
const script = ['settings.js', 'tools.js', 'app.js'].map(name => fs.readFileSync(path.join(web, name), 'utf8')).join('\n');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.handlers = new Map();
    this.style = { setProperty(key, value) { this[key] = value; } }; this.dataset = {}; this.value = ''; this.width = this.clientWidth = 800;
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
  removeAttribute(key) { delete this[key]; }
  showModal() { this.open = true; }
  close() { this.open = false; this.fire('close'); }
  click() { return this.fire('click'); }
  remove() {}
  checkValidity() { return true; }
  reportValidity() { return this.checkValidity(); }
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
const defaultProtectedParts = [...options.modules.map(item => item.id), 'object'];
const file = { name: '插画.png', size: 23, type: 'image/png' };
const tick = () => new Promise(setImmediate);
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}
function harness({ theme, prefersDark = false, draftLink } = {}) {
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(([, id]) => [id, new Element()]));
  const nodes = {};
  for (const [key, values] of [
    ['view', [...html.matchAll(/<button data-view="([^"]+)"/g)].map(([, view]) => view)],
    ['tool', [...html.matchAll(/<button data-tool="([^"]+)"/g)].map(([, tool]) => tool)],
    ['mode', ['review', 'flow']],
    ['scope', ['auto', 'empty', 'all']], ['theme-option', ['system', 'light', 'dark']],
  ]) {
    nodes[key] = values.map(value => {
      const button = new Element('button');
      const datasetKey = key === 'theme-option' ? 'themeOption' : key;
      button.dataset[datasetKey] = value;
      return button;
    });
  }
  for (const name of ['strength', 'restoration_gain', 'structure_protection', 'radius', 'feather', 'protection']) elements.get(name).type = 'range';
  Object.assign(elements.get('radius'), { min: '1', max: '256', value: '24' });
  elements.get('feather').value = '0';
  elements.get('magnetic-radius').value = '10';
  elements.get('protection').value = '.9';
  elements.get('lasso-action').value = 'include';
  elements.get('stage').dataset.tool = 'pan';
  elements.get('tools').dataset.placement = /id="tools"[^>]*data-placement="([^"]+)"/.exec(html)?.[1];
  let doc = {
    id: 'source', name: file.name, width: 640, height: 480, revision: 0,
    settings: { ...options.defaults, modules: options.default_modules, part_strengths: {} },
    edits: [], cursor: 0, status: 'queued', phase: 'analysis', error: null, b: null, a: null,
  };
  let localSettings = { defaults: doc.settings, export_dir: '', cache_dir: '', theme: theme || 'system',
    remember_brush: false, brush_radius: 24, brush_feather: 0, exit_on_close: false };
  const frames = [];
  const api = { requests: [], workers: [], beforeOpen: null, beforeEdit: null, beforeDecode: null, beforeSam: null,
    beforeChanges: null, beforeSettings: null, changeCounts: { added: 0, removed: 0, stronger: 0, weaker: 0, reclassified: 0 },
    changeCountsByStage: {}, changedParts: [],
    manualFrames: false, beforeMagnetic: null, magneticStatus: 200,
    frame: () => { for (const callback of frames.splice(0)) callback(); },
    mutate: change => { doc = { ...doc, ...change }; },
    ready: () => { doc = { ...doc, status: 'ready', b: { revision: doc.revision, method: doc.settings.method } }; },
  };
  const body = new Element('body'), win = new Element();
  const root = new Element('html'), scheme = new Element(), compact = new Element();
  compact.matches = false;
  if (theme) root.dataset.themePreference = theme;
  scheme.matches = prefersDark;
  const storage = new Map();
  const preferences = new Map(theme ? [['gaid-theme', theme]] : []);
  const sandbox = vm.createContext({
    document: {
      getElementById: id => elements.get(id), createElement: tag => new Element(tag),
      createElementNS: (namespace, tag) => new Element(tag), body, documentElement: root,
      querySelectorAll(selector) {
        const match = /^(button)?\[data-(.+)\]$/.exec(selector);
        if (match) return match[2] === 'tool' && !match[1]
          ? [...nodes.tool, elements.get('stage')] : nodes[match[2]];
        return [];
      },
    },
    window: {
      devicePixelRatio: 1, addEventListener: (...args) => win.addEventListener(...args),
      location: { origin: 'http://127.0.0.1:1790', pathname: '/', search: '', hash: draftLink ? '#draft=' + draftLink : '' },
      history: { replaceState: (...args) => { api.lastHistoryReplace = args; } },
      close: () => { win.closed = true; },
      matchMedia: query => query.includes('color-scheme') ? scheme : compact,
      localStorage: { setItem: (key, value) => preferences.set(key, value), removeItem: key => preferences.delete(key) },
      sessionStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) },
    }, file,
    Image: class {
      constructor() { this.naturalWidth = 640; this.naturalHeight = 480; }
      async decode() { await api.beforeDecode?.(this.src); }
    },
    ResizeObserver: class { observe() {} },
    AbortController, Blob, crypto: require('node:crypto').webcrypto,
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
      if (route === '/api/exit') return response({ closing: true });
      if (route === '/api/settings') {
        if (args.method === 'POST') {
          await api.beforeSettings;
          localSettings = { ...localSettings, ...JSON.parse(args.body) };
        }
        return response(localSettings);
      }
      if (route === '/api/cache') return response({ bytes: 0, path: 'G:/cache' });
      if (route.endsWith('/export')) return response({ path: 'G:/exports/export.png' });
      if (route.includes('/changes?revision=')) {
        const query = new URLSearchParams(route.split('?')[1]);
        const revision = Number(query.get('revision')), stage = query.get('stage');
        const counts = { ...(api.changeCountsByStage[stage] || api.changeCounts) };
        await (typeof api.beforeChanges === 'function' ? api.beforeChanges(stage) : api.beforeChanges);
        return response({ revision, stage, counts, overlay: Object.values(counts).some(Boolean) ? 'data:image/png;base64,AAAA' : null,
          ...(stage === 'parts' ? { parts: api.changedParts } : {}) });
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
      if (route.endsWith('/sam-selection')) {
        if (api.beforeSam) await api.beforeSam;
        const { revision } = JSON.parse(args.body);
        if (revision !== doc.revision) return response({ error: 'Conflict' }, 409);
        return response({ revision, token: 'object-preview', preview: 'data:image/png;base64,AAAA', pixels: 100 });
      }
      if (route.endsWith('/sam-selection/apply')) {
        const { revision, token, ...selection } = JSON.parse(args.body);
        if (revision !== doc.revision || token !== 'object-preview') return response({ error: 'Conflict' }, 409);
        const edits = [...doc.edits.slice(0, doc.cursor), { ...selection, shape: 'sam', token }];
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

test('the tool handle detaches, clamps, snaps and restores the active tool without editing', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range')");
  const toolbar = h.elements.get('tools'), grip = h.elements.get('tools-handle');
  assert.ok(grip, 'the tools have a draggable handle');
  const box = (left, top, width, height) => ({ left, top, width, height });
  h.elements.get('workspace').getBoundingClientRect = () => box(0, 50, 960, 640);
  h.elements.get('stage').getBoundingClientRect = () => box(100, 130, 600, 440);
  toolbar.getBoundingClientRect = () => box(20, 100, 50, 220);
  grip.getBoundingClientRect = () => box(20, 100, 50, 24);
  const pointer = (type, clientX, clientY) => grip.fire(type, {
    pointerId: 4, button: 0, clientX, clientY, preventDefault() {},
  });

  pointer('pointerdown', 45, 112);
  pointer('pointermove', 47, 113);
  assert.equal(toolbar.dataset.placement, 'docked');
  pointer('pointermove', 380, 290);
  assert.equal(toolbar.dataset.placement, 'floating');
  assert.equal(toolbar.style.transform, 'translate(355px, 228px)');
  pointer('pointermove', 900, 700);
  assert.equal(toolbar.style.transform, 'translate(642px, 292px)');
  pointer('pointerup', 900, 700);
  grip.fire('click', { detail: 1 });
  assert.equal(toolbar.dataset.placement, 'floating');
  h.nodes.tool.find(button => button.dataset.tool === 'lasso').click();
  assert.equal(h.run('state.tool'), 'lasso');
  assert.equal(h.run('state.draft.cursor'), 0);

  grip.fire('keydown', { key: 'ArrowLeft', preventDefault() {} });
  assert.equal(toolbar.style.transform, 'translate(626px, 292px)');
  pointer('pointerdown', 45, 112);
  pointer('pointermove', 108, 180);
  assert.equal(h.elements.get('stage').dataset.toolsDock, 'true');
  pointer('pointerup', 108, 180);
  assert.equal(toolbar.dataset.placement, 'docked');
  assert.equal(h.run('state.tool'), 'lasso');
});

test('a focused disclosure handles Space locally and still allows flow navigation', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  let prevented = false;
  const summary = new Element('summary');
  h.window.fire('keydown', { key: ' ', target: summary, preventDefault() { prevented = true; } });
  assert.equal(prevented, false);
  assert.equal(h.run('state.space'), false);
  h.window.fire('keydown', { key: 'f', target: summary, preventDefault() {} });
  assert.equal(h.run('state.mode'), 'flow');
});

test('canvas click bubbling does not reset an in-progress polygon', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('lasso'); state.polygon = [[15,16]]");
  h.elements.get('stage').fire('click');
  assert.equal(h.run('state.polygon.length'), 1);
});

test('shortcut tools match the toolbar and keep the current brush action', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('exclude'); selectTool('lasso')");
  const key = (value, extra = {}) => h.window.fire('keydown', { key: value, preventDefault() {}, ...extra });
  key('b');
  assert.equal(h.run('state.tool'), 'exclude');
  key('s');
  assert.equal(h.run('state.tool'), 'sam_object');
  key('h');
  assert.equal(h.run('state.tool'), 'pan');
  key('i');
  assert.equal(h.run('state.tool'), 'include');
  key('d', { isComposing: true });
  assert.equal(h.run('state.tool'), 'include');
  key('d', { target: new Element('input') });
  assert.equal(h.run('state.tool'), 'include');
  h.run("selectMode('parts')");
  key('d');
  assert.equal(h.run('state.tool'), 'remove_part');
  key('c');
  assert.equal(h.run('state.tool'), 'classify');
});

test('keyboard export uses the current displayed B and ignores held-key repeats', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  const key = repeat => h.window.fire('keydown', { key: 's', ctrlKey: true, repeat, preventDefault() {} });
  const exports = () => h.api.requests.filter(item => item.route.includes('/download?view=b'));
  key(false);
  await tick();
  assert.equal(exports().length, 0);
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  key(false);
  key(true);
  await tick();
  assert.deepEqual(exports().map(item => item.route), ['/api/documents/source/download?view=b&revision=0']);
});

test('keyboard view controls share fit, actual scale and comparison state', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  await h.run('pin()');
  await tick();
  const key = (value, target = h.window) => h.window.fire('keydown', { key: value, target, preventDefault() {} });
  const original = h.elements.get('compare-reference-original'), pinned = h.elements.get('compare-reference-a');
  original.click();
  key('a', original);
  assert.equal(pinned['aria-pressed'], 'true');
  assert.equal(pinned.focused, true);
  key('a', h.elements.get('stage'));
  assert.equal(original.focused, undefined);
  assert.equal(h.elements.get('left-badge').textContent, '原图');
  key('1');
  assert.equal(h.run('layout().scale'), 1);
  assert.equal(h.run('state.fit'), false);
  key('+');
  assert.equal(h.run('layout().scale'), 1.25);
  key('0');
  assert.equal(h.run('state.fit'), true);
  key('f');
  assert.equal(h.run('state.mode'), 'flow');
  assert.equal(h.elements.get('node-parts').focused, true);
  key('r');
  assert.equal(h.run('state.mode'), 'review');
  assert.equal(h.elements.get('stage').focused, true);
  assert.equal(h.elements.get('left-badge').textContent, '原图');
});

test('clicking the comparison canvas takes keyboard focus', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.run('comparing()'), true);
  stroke(h, [[200, 120]]);
  assert.equal(h.elements.get('stage').focused, true);
});

test('the shortcut guide opens from question mark and takes precedence over workbench commands', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  const dialog = h.elements.get('shortcuts-dialog');
  assert.ok(dialog);
  h.window.fire('keydown', { key: '?', shiftKey: true, preventDefault() {} });
  assert.equal(dialog.open, true);
  h.window.fire('keydown', { key: 'f', preventDefault() {} });
  assert.equal(h.run('state.mode'), 'review');
  h.elements.get('shortcuts-close').click();
  assert.equal(dialog.open, false);
  h.elements.get('open-shortcuts').click();
  assert.equal(dialog.open, true);
});

test('Enter previews and applies a SAM2 selection in separate presses', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('range'); selectTool('sam_object')");
  stroke(h, [[300, 220]]);
  const enter = repeat => h.window.fire('keydown', { key: 'Enter', repeat, preventDefault() {} });
  enter(false);
  await tick();
  assert.equal(h.api.requests.filter(item => item.route.endsWith('/sam-selection')).length, 1);
  assert.equal(h.run('state.draft.cursor'), 0);
  enter(true);
  await tick();
  assert.equal(h.api.requests.filter(item => item.route.endsWith('/sam-selection/apply')).length, 0);
  enter(false);
  await tick();
  assert.equal(h.run('state.draft.edits[0].shape'), 'sam');
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
  const legend = h.elements.get('range-changes-legend');
  assert.ok(legend, 'the range audit has a canvas legend');
  assert.equal(legend.hidden, true);
  h.run("selectView('weight'); addEdit({tool:'include',points:[[10,10]],radius:5})");
  assert.match(h.elements.get('change-history-list').children[0].children[1].textContent, /待更新/);
  assert.equal(h.api.requests.filter(x => x.route.includes('/changes?')).length, 1);
  await h.run('saveDraft()');
  h.api.changeCounts.added = 32;
  h.api.changeCounts.stronger = 12;
  h.api.ready();
  await h.run('pollDocument()');
  h.elements.get('show-manual-changes').click();
  await tick();
  assert.equal(h.run("state.images.get('changes:range').revision"), 1);
  assert.match(h.elements.get('changes-added').textContent, /32/);
  assert.equal(legend.hidden, false);
  assert.equal(h.elements.get('range-legend-added').hidden, false);
  assert.equal(h.elements.get('range-legend-stronger').hidden, false);
  assert.equal(h.elements.get('range-legend-removed').hidden, true);
  assert.equal(h.elements.get('range-legend-weaker').hidden, true);
  h.window.fire('keydown', { key: 'x' });
  assert.equal(legend.hidden, true);
  h.window.fire('keyup', { key: 'x' });
  assert.equal(legend.hidden, false);
  h.run("selectView('compare')");
  assert.equal(legend.hidden, true);
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
  h.run("selectView('weight'); addEdit({tool:'include',points:[[10,10]],radius:5})");
  await h.run('saveDraft()');
  h.api.beforeChanges = null;
  h.api.changeCounts.added = 0;
  h.api.changeCounts.weaker = 3;
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectView('changes')");
  await tick();
  pending.resolve();
  await tick();
  assert.equal(h.run("state.images.get('changes:range').revision"), 1);
  assert.equal(h.run("state.images.get('changes:range').counts.weaker"), 3);
  assert.equal(h.elements.get('range-legend-added').hidden, true);
  assert.equal(h.elements.get('range-legend-weaker').hidden, false);
});

test('parts and range show their own edit rows and keep concurrent change previews separate', async () => {
  const h = harness();
  h.api.mutate({ edits: [
    { stage: 'parts', tool: 'classify', module: 'outfit', points: [[20, 30]], radius: 3 },
    { stage: 'range', tool: 'exclude', points: [[80, 90]], radius: 3 },
  ], cursor: 2 });
  h.api.ready();
  await tick();
  await h.run('openFile(file)');
  const history = () => h.elements.get('change-history-list').children.map(row => row.children[0].textContent);
  const pending = deferred();
  h.api.beforeChanges = stage => stage === 'parts' ? pending.promise : undefined;
  h.api.changeCountsByStage = {
    parts: { reclassified: 13 },
    range: { added: 0, removed: 29, stronger: 0, weaker: 0 },
  };
  h.api.changedParts = ['hair', 'outfit', 'unknown'];
  h.run("selectMode('parts'); selectView('changes')");
  assert.deepEqual(history(), ['1  归为outfit']);
  h.run("selectMode('range'); selectView('changes')");
  await tick();
  assert.deepEqual(history(), ['2  移出处理范围']);
  assert.match(h.elements.get('changes-removed').textContent, /29/);
  assert.equal(h.elements.get('change-count-reclassified').hidden, true);
  pending.resolve();
  await tick();
  assert.match(h.elements.get('changes-removed').textContent, /29/);
  h.run("selectMode('parts'); selectView('changes')");
  await tick();
  assert.match(h.elements.get('changes-reclassified').textContent, /13/);
  assert.equal(h.elements.get('change-count-removed').hidden, true);
  const legend = h.elements.get('part-changes-legend');
  assert.ok(legend, 'the parts audit has a destination legend');
  assert.equal(legend.hidden, false);
  assert.deepEqual(legend.children.map(item => item.children[1]), ['hair', 'outfit', '待确认']);
  assert.equal(legend.children.at(-1).children[0].className, 'hatch-key');
  h.window.fire('keydown', { key: 'x' });
  assert.equal(legend.hidden, true);
  h.window.fire('keyup', { key: 'x' });
  assert.equal(legend.hidden, false);
  h.run("selectMode('range'); selectView('changes')");
  assert.equal(legend.hidden, true);
  assert.deepEqual(h.api.requests.filter(x => x.route.includes('/changes?')).map(x => x.route.split('&stage=')[1]),
    ['parts', 'range']);
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
  assert.equal(h.elements.get('change-history-note').hidden, true);
  assert.equal(h.elements.get('show-manual-changes').hidden, true);
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

test('quick strength in the effect editor keeps the algorithm and displays the result', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  h.run("selectMode('effect')");
  assert.equal(h.run('state.view'), 'result');
  assert.equal(h.run("visibleImage('result') === visibleImage('b')"), true);
  h.run("state.draft.settings.method = 'esrgan-l0'; state.draft.settings.l0 = .0007; refresh()");
  const shortcut = h.elements.get('set-strength-65');
  assert.ok(shortcut, 'overall strength exposes its own shortcut');
  shortcut.click();
  assert.equal(h.run('state.draft.settings.strength'), .65);
  assert.equal(h.run('state.draft.settings.method'), 'esrgan-l0');
  assert.equal(h.run('state.draft.settings.l0'), .0007);
  assert.equal(shortcut.disabled, true);
  assert.equal(shortcut['aria-pressed'], undefined);
});

test('L0 quick adjustment and the effect reset are scoped separately', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('effect'); state.draft.settings.modules = ['hair']; state.draft.settings.part_strengths = {hair:.6}; refresh()");
  const reset = h.elements.get('reset-effect-defaults'), l0 = h.elements.get('set-l0-001');
  assert.ok(reset && l0, 'the effect reset and L0 shortcut are separate actions');
  assert.equal(reset.disabled, true);
  l0.click();
  assert.equal(h.run('state.draft.settings.l0'), .001);
  assert.equal(l0.disabled, true);
  assert.equal(reset.disabled, false);
  reset.click();
  assert.equal(h.run('state.draft.settings.l0'), .0005);
  assert.equal(h.run('state.draft.settings.strength'), 1);
  assert.equal(h.run("state.draft.settings.modules.join(',')"), 'hair');
  assert.equal(h.run('state.draft.settings.part_strengths.hair'), .6);
  assert.equal(reset.disabled, true);
});

test('effect parameter help opens one explanation at a time without adjusting the image', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('effect')");
  const help = name => h.elements.get(name + '-help');
  const toggle = name => h.elements.get(name + '-help-toggle');
  assert.ok(toggle('restoration_gain') && toggle('structure_protection') && toggle('sigma'));
  const settings = h.run('JSON.stringify(state.draft.settings)');
  toggle('restoration_gain').click();
  assert.equal(help('restoration_gain').hidden, false);
  assert.equal(toggle('restoration_gain')['aria-expanded'], 'true');
  assert.equal(help('structure_protection').hidden, true);
  toggle('sigma').click();
  assert.equal(help('restoration_gain').hidden, true);
  assert.equal(help('sigma').hidden, false);
  toggle('sigma').click();
  assert.equal(help('sigma').hidden, true);
  assert.equal(toggle('sigma')['aria-expanded'], 'false');
  assert.equal(h.run('JSON.stringify(state.draft.settings)'), settings);
  assert.equal(h.run('state.dirty'), false);
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
  const legacyLegend = h.elements.get('weight-legacy-legend');
  assert.ok(legacyLegend, 'a composite weight preview has its own legend');
  assert.equal(legacyLegend.hidden, false);
  assert.equal(h.elements.get('weight-ramp').hidden, true);
  assert.match(h.elements.get('status').textContent, /范围已更新/);
  h.run("selectMode('review')");
  await tick();
  assert.equal(h.run('state.tool'), 'pan');
  assert.ok(h.api.requests.some(item => item.route.endsWith('/image?view=b&revision=1')));
});

test('suspected objects load on demand and keep part edits in the overlay view', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('parts'); selectTool('classify')");
  const tab = h.nodes.view.find(button => button.dataset.view === 'suspected_objects');
  assert.ok(tab, 'the range toolbar includes a suspected-object view');
  assert.equal(tab.hidden, true);
  h.run("selectView('suspected_objects')");
  assert.equal(h.run('state.view'), 'parts');

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
  assert.equal(h.run('state.draft.cursor'), 0);
  assert.equal(h.run('state.view'), 'suspected_objects');

  h.api.ready();
  await h.run('pollDocument()');
  assert.equal(tab.hidden, true);
  assert.equal(h.run('state.view'), 'parts');
  stroke(h, [[200, 120]]);
  await h.run('saveDraft()');
  const edit = h.api.requests.find(item => item.route.endsWith('/edit'));
  assert.deepEqual(JSON.parse(edit.args.body).edits, [{ tool:'classify', stage:'parts', module:'hair', shape:'brush', geometry:'continuous',
    points:[[200,120]], radius:24, feather:0, protected_parts: defaultProtectedParts }]);
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

test('prepares a finished draft on opening without repeating preparation on polls', async () => {
  const h = harness();
  await tick();
  h.api.ready();
  await h.run('openFile(file)');
  await h.run('pollDocument()');
  await tick();
  const prepared = h.api.requests.filter(item => item.route.endsWith('/prepare-edit'));
  assert.equal(prepared.length, 1);
  assert.deepEqual(JSON.parse(prepared[0].args.body), { revision: 0 });
});

test('brush captures subpixel samples and reaches the release point', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); zoomTo(2)");
  const rect = h.run('layout()'), canvas = h.elements.get('canvas');
  const event = (x, y) => ({ pointerId: 1, button: 0, clientX: 100 + rect.x + x * rect.scale,
    clientY: 50 + rect.y + y * rect.scale, preventDefault() {} });
  canvas.fire('pointerdown', event(200.25, 120.25));
  canvas.fire('pointermove', { ...event(220.25, 120.25),
    getCoalescedEvents: () => [event(210.25, 121.25)] });
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.gesture.edit.points)')),
    [[200.25, 120.25], [210.25, 120.75], [220.25, 120.25]]);
  canvas.fire('pointerup', event(228.75, 120.875));
  await h.run('saveDraft()');
  const edit = h.api.requests.find(item => item.route.endsWith('/edit'));
  const [brush] = JSON.parse(edit.args.body).edits;
  assert.equal(brush.geometry, 'continuous');
  assert.equal(brush.adaptive_edge, true);
  assert.deepEqual(brush.points, [[200.25, 120.25], [210.25, 120.75], [228.75, 120.875]]);
});

test('dense samples along a straight brush stroke are compacted when saved', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range')");
  const points = Array.from({ length: 401 }, (_, i) => [100 + i, 200]);
  stroke(h, points);
  const stored = JSON.parse(h.run('JSON.stringify(state.draft.edits[0].points)'));
  assert.deepEqual(stored[0], [100, 200]);
  assert.deepEqual(stored.at(-1), [500, 200]);
  assert.ok(stored.length <= 8, stored.length);
  assert.ok(stored.every(([, y]) => y === 200));
});

test('exiting through settings waits for pending brush edits to save', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('parts')");
  stroke(h, [[200, 120]]);
  await h.elements.get('open-settings').click();

  const saved = deferred();
  h.api.beforeEdit = saved.promise;
  const exit = h.elements.get('exit-workbench').click();
  await tick();
  assert.ok(h.api.requests.some(item => item.route.endsWith('/edit')));
  assert.ok(!h.api.requests.some(item => item.route === '/api/exit'));
  assert.equal(h.window.closed, undefined);
  saved.resolve();

  await exit;
  assert.deepEqual(h.api.requests.filter(item => item.args?.method === 'POST').slice(-2).map(item => item.route),
    ['/api/documents/source/edit', '/api/exit']);
  assert.equal(h.elements.get('settings-dialog').open, false);
  assert.equal(h.elements.get('empty-title').textContent, '工作台已退出');
  assert.equal(h.window.closed, true);
});

test('closing a browser page reports its departure and sends a pending stroke', async () => {
  const h = harness();
  const reports = () => h.api.requests.filter(item => item.route === '/api/page');
  await tick();
  h.window.fire('blur');
  assert.equal(reports().length, 1);
  const joined = JSON.parse(reports()[0].args.body);
  assert.equal(joined.open, true);

  await h.run('openFile(file)');
  h.run("selectMode('parts')");
  stroke(h, [[200, 120]]);
  h.window.fire('pagehide');
  await tick();
  assert.deepEqual(JSON.parse(reports()[1].args.body), { id: joined.id, open: false });
  assert.equal(reports()[1].args.keepalive, true);
  const edit = h.api.requests.find(item => item.route.endsWith('/edit'));
  assert.equal(edit.args.keepalive, true);
  assert.equal(JSON.parse(edit.args.body).cursor, 1);

  h.window.fire('pageshow');
  const restored = JSON.parse(reports()[2].args.body);
  assert.equal(restored.open, true);
  assert.notEqual(restored.id, joined.id);
});

test('settings save and reload the preference to exit with the last page', async () => {
  const h = harness();
  await tick();
  await h.elements.get('open-settings').click();
  const field = h.elements.get('pref-exit-on-close');
  assert.equal(field.checked, false);
  field.checked = true;
  await h.elements.get('settings-form').fire('submit', { preventDefault() {} });
  const update = h.api.requests.find(item => item.route === '/api/settings' && item.args?.method === 'POST');
  assert.deepEqual(JSON.parse(update.args.body), { exit_on_close: true });
  await h.elements.get('open-settings').click();
  assert.equal(field.checked, true);
});

test('the comparison can switch between original and pinned A and remember the choice', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  const group = h.elements.get('compare-reference');
  assert.ok(group, 'the comparison offers a left reference selector');
  const original = h.elements.get('compare-reference-original');
  const pinned = h.elements.get('compare-reference-a');
  assert.equal(group.hidden, true);
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  await h.run('pin()');
  await tick();
  assert.equal(group.hidden, false);
  assert.equal(pinned['aria-pressed'], 'true');
  assert.match(h.elements.get('left-badge').textContent, /^A  /);
  h.run('zoomTo(1.6); state.split = .37; draw()');
  const viewport = h.run('JSON.stringify([state.zoom, state.fit, state.panX, state.panY, state.split])');
  original.click();
  assert.equal(original['aria-pressed'], 'true');
  assert.equal(h.elements.get('left-badge').textContent, '原图');
  assert.equal(h.run('state.doc.a.revision'), 0);
  assert.equal(h.run('state.images.get("b").revision'), 0);
  assert.equal(h.run('JSON.stringify([state.zoom, state.fit, state.panX, state.panY, state.split])'), viewport);
  await h.run("openRecent('source')");
  await tick();
  assert.equal(h.elements.get('left-badge').textContent, '原图');
  pinned.click();
  await tick();
  assert.match(h.elements.get('left-badge').textContent, /^A  /);
  assert.equal(h.api.requests.filter(item => item.route.endsWith('/pin')).length, 1);
});

test('a delayed A frame cannot replace the original after switching references', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  const pending = deferred();
  h.api.beforeDecode = url => url.includes('view=a') ? pending.promise : undefined;
  h.api.mutate({ a: { revision: 0, method: 'l0' } });
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.elements.get('compare-reference').hidden, false);
  h.elements.get('compare-reference-original').click();
  pending.resolve();
  await tick();
  assert.equal(h.run("state.images.has('a')"), true);
  assert.equal(h.elements.get('left-badge').textContent, '原图');
  assert.equal(h.elements.get('compare-reference-original')['aria-pressed'], 'true');
});

for (const [mode, editable] of [['parts', 'parts'], ['range', 'weight']]) {
  test(mode + ' comparison combines the original and B even when a pinned A exists', async () => {
    const h = harness();
    await tick();
    await h.run('openFile(file)');
    h.run(`selectMode('${mode}')`);
    h.api.mutate({ a: { revision: 0, method: 'l0' } });
    h.api.ready();
    await h.run('pollDocument()');
    const view = id => h.nodes.view.find(button => button.dataset.view === id);
    assert.equal(view('original').hidden, true);
    assert.equal(view('result').hidden, true);
    assert.equal(view('compare').hidden, false);
    assert.equal(view('compare').textContent, '原图 / 成品');
    const viewport = h.run('JSON.stringify([state.fit, state.zoom, state.panX, state.panY])');
    view('compare').click();
    await tick();
    assert.equal(h.elements.get('divider').hidden, false);
    assert.equal(h.elements.get('compare-reference').hidden, true);
    assert.equal(h.elements.get('left-badge').textContent, '原图');
    assert.equal(h.run('JSON.stringify([state.fit, state.zoom, state.panX, state.panY])'), viewport);
    assert.equal(h.api.requests.some(item => item.route.includes('view=a')), false);

    h.run("selectMode('review')");
    await tick();
    assert.match(h.elements.get('left-badge').textContent, /^A  /);
    h.run(`selectMode('${mode}'); selectView('compare')`);
    assert.equal(h.elements.get('left-badge').textContent, '原图');
  });

  test(mode + ' inspection views pan without editing by pointer, tools, or shortcuts', async () => {
    const h = harness();
    await tick();
    await h.run('openFile(file)');
    h.api.ready();
    await h.run('pollDocument()');
    h.run(`selectMode('${mode}')`);
    const brush = h.nodes.tool.find(button => button.dataset.tool === 'brush');
    for (const view of ['compare', 'changes']) {
      h.run("selectView('" + view + "')");
      assert.equal(brush.disabled, true);
      assert.equal(h.elements.get('stage').dataset.tool, 'pan');
      assert.equal(h.elements.get(mode + '-inspector').hidden, true);
      assert.equal(h.elements.get('edit-inspector').hidden, true);
      assert.equal(h.elements.get('feather-row').hidden, true);
      const before = h.run('state.panX');
      h.window.fire('keydown', {key: 'b', preventDefault() {}});
      stroke(h, [[200, 120], [210, 124]]);
      assert.ok(h.run('state.panX') > before);
      assert.equal(h.run('state.draft.cursor'), 0);
    }
    h.run(`selectView('${editable}')`);
    assert.equal(brush.disabled, false);
    h.window.fire('keydown', {key: 'x'});
    assert.equal(brush.disabled, true);
    stroke(h, [[200, 120]]);
    assert.equal(h.run('state.draft.cursor'), 0);
    h.window.fire('keyup', {key: 'x'});
    assert.equal(brush.disabled, false);
    stroke(h, [[200, 120]]);
    assert.equal(h.run('state.draft.cursor'), 1);
    h.run("selectView('compare')");
    h.window.fire('keydown', {key: 'z', ctrlKey: true, preventDefault() {}});
    assert.equal(h.run('state.draft.cursor'), 1);
  });

  test(mode + ' switching to comparison discards an unfinished lasso', async () => {
    const h = harness();
    await tick();
    await h.run('openFile(file)');
    h.run(`selectMode('${mode}'); selectTool('lasso')`);
    for (const p of [[125,105],[140,115],[132,120]]) stroke(h, [p]);
    assert.equal(h.run('state.polygon.length'), 3);
    h.run("selectView('compare')");
    h.window.fire('keydown', {key: 'Enter', preventDefault() {}});
    assert.equal(h.run('state.draft.cursor'), 0);
    h.run(`selectView('${editable}')`);
    assert.equal(h.run('state.polygon.length'), 0);
  });

  test(mode + ' SAM2 previews on the editable view and applies one undoable edit', async () => {
    const h = harness();
    await tick();
    await h.run('openFile(file)');
    h.api.ready();
    await h.run('pollDocument()');
    h.run(`selectMode('${mode}'); selectTool('sam_object'); zoomTo(2)`);
    assert.equal(h.run('state.view'), editable);
    assert.equal(h.elements.get('sam-panel').hidden, false);
    assert.equal(h.elements.get('sam-apply').disabled, true);
    stroke(h, [[285, 215]]);
    h.elements.get('sam-negative').click();
    stroke(h, [[310, 235]]);
    assert.equal(h.api.requests.some(item => item.route.endsWith('/sam-selection')), false);
    await h.elements.get('sam-preview').click();
    const preview = h.api.requests.find(item => item.route.endsWith('/sam-selection'));
    assert.deepEqual(JSON.parse(preview.args.body), { revision: 0, positive: [[285, 215]], negative: [[310, 235]] });
    assert.equal(h.elements.get('sam-apply').disabled, false);
    assert.equal(h.run('state.draft.cursor'), 0);
    await h.elements.get('sam-apply').click();
    assert.equal(h.run("state.draft.edits[0].tool"), mode === 'parts' ? 'classify' : 'include');
    assert.equal(h.run("state.draft.edits[0].shape"), 'sam');
    assert.equal(h.run('state.draft.cursor'), 1);
    assert.equal(h.run('state.view'), editable);
    h.elements.get('undo').click();
    await h.run('flushChanges()');
    assert.equal(h.run('state.draft.cursor'), 0);
  });
}

test('SAM2 previews cannot be applied after cancellation or a remote revision change', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('parts'); selectTool('sam_object')");
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
  assert.equal(h.api.requests.some(item => item.route.endsWith('/sam-selection/apply')), false);
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
  assert.deepEqual(JSON.parse(edit.args.body).edits, [{ tool:'protect', stage:'range', shape:'brush', geometry:'continuous',
    points:[[200,120],[207,122]], radius:24, strength:.9, feather:0 }]);
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
  h.run("zoomTo(2); selectMode('parts'); selectTool('lasso')");
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
    tool: 'classify', stage: 'parts', shape: 'polygon', points: [[125,105],[140,115],[132,120]], module: 'hair',
    protected_parts: defaultProtectedParts,
  }]);
  h.run("selectMode('range')");
  h.elements.get('invert').click();
  h.run('undo()');
  await h.run('saveDraft()');
  const payload = JSON.parse(h.api.requests.find(item => item.route.endsWith('/edit')).args.body);
  assert.equal(payload.cursor, 1);
  assert.deepEqual(payload.edits[1], { tool: 'invert', stage: 'range' });
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
  assert.notEqual(h.elements.get('range-scope').hidden, true);
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
    tool: 'exclude', stage: 'range', shape: 'polygon', points: [[316, 214], [335, 220], [360, 235], [320, 270], [313, 242]],
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

test('selection previews advance during encoding and never unlock a result that has not been shown', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('parts')");
  await tick();
  const encoding = revision => h.api.mutate({ status: 'running', phase: 'encoding',
    selection_preview: { revision, cursor: revision, images: { parts_overlay: true, weight_overlay: true } } });
  const old = deferred(), result = deferred();
  h.api.beforeDecode = url => url.endsWith('view=parts_overlay&revision=1') ? old.promise
    : url.endsWith('view=b&revision=2') ? result.promise : undefined;

  stroke(h, [[200, 120]]);
  await h.run('saveDraft()');
  encoding(1);
  await h.run('pollDocument()');
  await tick();
  assert.ok(h.api.requests.some(item => item.route.endsWith('view=parts_overlay&revision=1')));
  assert.equal(h.run("viewAsset('parts').revision"), 0);

  stroke(h, [[240, 130]]);
  await h.run('saveDraft()');
  encoding(2);
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.run("viewAsset('parts').revision"), 2);
  assert.equal(h.run("viewAsset('parts').cursor"), 2);
  assert.match(h.elements.get('status').textContent, /分区已更新.*成品 B/);
  assert.equal(h.run('state.doc.b.revision'), 0);
  old.resolve();
  await tick();
  assert.equal(h.run("viewAsset('parts').revision"), 2);
  h.run("selectMode('range')");
  await tick();
  assert.equal(h.run("viewAsset('weight').revision"), 2);
  assert.match(h.elements.get('status').textContent, /范围已更新.*成品 B/);

  h.run("selectMode('review')");
  await h.run('download()');
  await h.run('pin()');
  assert.equal(h.api.requests.some(item => /\/download\?|\/pin$/.test(item.route)), false);
  h.api.ready();
  h.api.mutate({ selection_preview: null });
  await h.run('pollDocument()');
  await tick();
  assert.equal(h.run('state.doc.b.revision'), 2);
  assert.equal(h.elements.get('pin').disabled, true);
  assert.equal(h.elements.get('download').disabled, true);
  result.resolve();
  await tick();
  assert.equal(h.run('state.displayedResult'), 2);
  assert.equal(h.elements.get('pin').disabled, false);
  assert.equal(h.elements.get('download').disabled, false);
});

test('more export options are only shown when the menu has actions', async () => {
  const h = harness();
  const menu = h.elements.get('export-panel');
  const visibleItems = () => ['download-a', 'correction-export', 'correction-import']
    .filter(id => !h.elements.get(id).hidden);
  await tick();
  assert.equal(menu.hidden, true);
  await h.run('openFile(file)');
  assert.equal(menu.hidden, true);

  for (const mode of ['parts', 'range']) {
    h.run("selectMode('" + mode + "')");
    assert.equal(menu.hidden, false);
    assert.deepEqual(visibleItems(), ['correction-export', 'correction-import']);
    menu.open = true;
    h.run("selectMode('review')");
    assert.equal(menu.hidden, true);
    assert.equal(menu.open, false);
  }

  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  await h.run('pin()');
  assert.equal(menu.hidden, false);
  assert.deepEqual(visibleItems(), ['download-a']);
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
  h.run("selectMode('parts')");
  await tick();
  assert.equal(requested(), 1);
  const unknownEye = h.elements.get('part-targets').children.flatMap(row => row.children)
    .find(button => button.dataset.showPart === 'unknown');
  assert.equal(toggle.hidden, true);
  unknownEye.click();
  assert.equal(h.elements.get('unknown-legend').hidden, true);
  unknownEye.click();
  await tick();
  assert.equal(h.elements.get('unknown-legend').hidden, false);
  assert.equal(requested(), 1);
  h.run("selectView('compare')");
  assert.equal(toggle.hidden, true);
  assert.equal(h.elements.get('unknown-legend').hidden, true);
});

test('新范围笔画补选待确认时保留分类，且可撤销', async () => {
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
  assert.equal(h.run('JSON.stringify(state.draft.edits[1])'), '{"tool":"resolve_unknown","stage":"range"}');
  assert.equal(actions.hidden, true);
  await h.run('saveDraft()');
  h.elements.get('undo').click();
  await h.run('saveDraft()');
  h.api.ready();
  await h.run('pollDocument()');
  assert.equal(h.run('state.draft.cursor'), 1);
  assert.equal(h.elements.get('unknown-assignment').hidden, true);
  h.elements.get('open-unknown-parts').click();
  assert.equal(h.run('state.mode'), 'parts');
});

test('旧范围笔画仍可快捷归为指定部位', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); addEdit({tool:'include',shape:'brush',points:[[200,120]],radius:24})");
  await h.run('saveDraft()');
  h.api.mutate({ edit_feedback: {
    tool: 'include', changed_pixels: 0, reason: 'protected', skipped_unknown_pixels: 15,
  } });
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



test('flow uses the published thumbnails and keeps the canvas position across nodes', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  h.run('zoomTo(2, 350, 275)');
  const viewport = h.run('JSON.stringify([state.zoom, state.panX, state.panY])');
  h.nodes.mode.find(button => button.dataset.mode === 'flow').click();
  assert.equal(h.elements.get('flow-board').hidden, false);
  assert.equal(h.elements.get('stage').hidden, true);
  assert.match(h.elements.get('thumb-parts').src, /thumbnail\?view=parts&revision=0$/);
  assert.match(h.elements.get('thumb-effect').src, /thumbnail\?view=candidate&revision=0$/);
  h.elements.get('node-effect').click();
  assert.equal(h.run('state.view'), 'result');
  assert.equal(h.elements.get('tools').hidden, true);
  h.nodes.view.find(button => button.dataset.view === 'candidate').click();
  await tick();
  assert.ok(h.api.requests.some(item => item.route.endsWith('/image?view=candidate&revision=0')));
  h.run("selectMode('flow')");
  h.elements.get('node-result').click();
  assert.equal(h.run('state.view'), 'compare');
  assert.equal(h.run('JSON.stringify([state.zoom, state.panX, state.panY])'), viewport);
});

test('node navigation makes the current editor clear and returns to its flow card', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  const [review, flow] = h.nodes.mode;
  const heading = h.elements.get('node-title'), back = h.elements.get('back-to-flow');
  flow.click();
  assert.equal(flow['aria-pressed'], 'true');
  for (const [mode, name] of [['parts', '部位分区'], ['range', '处理范围'], ['effect', '降噪效果']]) {
    const card = h.elements.get('node-' + mode);
    card.click();
    assert.equal(flow['aria-pressed'], 'false');
    assert.equal(flow.hidden, true);
    assert.equal(review['aria-pressed'], 'false');
    assert.equal(heading.hidden, false);
    assert.equal(heading.textContent, name);
    assert.equal(heading.focused, true);
    assert.equal(h.elements.get('stage').focused, undefined);
    assert.equal(back.hidden, false);
    back.click();
    assert.equal(h.elements.get('flow-board').hidden, false);
    assert.equal(flow['aria-pressed'], 'true');
    assert.equal(heading.hidden, true);
    assert.equal(back.hidden, true);
    assert.equal(card.focused, true);
  }
  h.elements.get('node-parts').click();
  review.click();
  assert.equal(h.run('state.mode'), 'review');
  assert.equal(review['aria-pressed'], 'true');
  assert.equal(heading.hidden, true);
});

test('brush geometry stays in original-image pixels and applies feather only in the range stage', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('flow')");
  h.elements.get('node-parts').click();
  h.run("setPart('outfit')");
  h.elements.get('radius').value = '18';
  h.elements.get('feather').value = '12';
  h.elements.get('radius').fire('input');
  assert.equal(h.elements.get('radius-value').textContent, '36 px');
  assert.equal(h.elements.get('feather-row').hidden, true);
  stroke(h, [[210, 150]]);
  h.run("selectMode('range'); selectTool('exclude'); zoomTo(2)");
  assert.equal(h.elements.get('feather-row').hidden, false);
  stroke(h, [[210, 150]]);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits)')), [
    { tool: 'classify', stage: 'parts', module: 'outfit', shape: 'brush', geometry: 'continuous', points: [[210, 150]], radius: 18, feather: 0,
      protected_parts: defaultProtectedParts },
    { tool: 'exclude', stage: 'range', shape: 'brush', geometry: 'continuous', points: [[210, 150]], radius: 18, feather: 12 },
  ]);
});

test('brackets resize the active brush in image pixels and clamp its feather', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  const radius = h.elements.get('radius'), feather = h.elements.get('feather');
  const press = (key, extra = {}) => {
    let handled = false;
    h.window.fire('keydown', { key, preventDefault() { handled = true; }, ...extra });
    return handled;
  };
  assert.equal(press(']'), false);
  h.run("selectMode('range'); selectTool('exclude'); zoomTo(2)");
  radius.value = '10';
  feather.value = '10';
  radius.fire('input');
  assert.equal(press('[', { target: radius }), true);
  assert.equal(h.elements.get('radius-value').textContent, '18 px');
  assert.equal(h.elements.get('feather-value').textContent, '9 px');
  assert.equal(press(']', { target: new Element('input') }), false);
  assert.equal(press(']', { ctrlKey: true }), false);
  assert.equal(h.run('state.draft.cursor'), 0);
  stroke(h, [[210, 150]]);
  assert.equal(h.run('state.draft.edits[0].radius'), 9);
  assert.equal(h.run('state.draft.edits[0].feather'), 9);
  h.run("selectMode('parts')");
  radius.value = '255';
  assert.equal(press('】'), true);
  assert.equal(h.elements.get('radius-value').textContent, '512 px');
  radius.value = '1';
  press('[');
  assert.equal(Number(radius.value), 1);
  h.nodes.view.find(button => button.dataset.view === 'compare').click();
  assert.equal(press(']'), false);
  assert.equal(Number(radius.value), 1);
});

test('holding a brush-size shortcut saves once and slow settings replies preserve a newer size', async () => {
  const h = harness();
  await tick();
  await h.run("preferences.update({ remember_brush: true, brush_radius: 24, brush_feather: 12 })");
  await h.run('openFile(file)');
  h.run("selectMode('range')");
  const settingsWrites = () => h.api.requests
    .filter(item => item.route === '/api/settings' && item.args.method === 'POST')
    .map(item => JSON.parse(item.args.body)).filter(patch => !Object.hasOwn(patch, 'remember_brush'));
  const key = (event, key, repeat = false) => h.window.fire(event, { key, repeat, preventDefault() {} });
  key('keydown', ']');
  key('keydown', ']', true);
  await tick();
  assert.equal(h.elements.get('radius-value').textContent, '52 px');
  assert.deepEqual(settingsWrites(), []);

  const firstWrite = deferred();
  h.api.beforeSettings = firstWrite.promise;
  key('keyup', ']');
  await tick();
  assert.deepEqual(settingsWrites(), [{ brush_radius: 26, brush_feather: 12 }]);
  key('keydown', '[');
  firstWrite.resolve();
  await tick();
  assert.equal(h.elements.get('radius-value').textContent, '50 px');
  key('keyup', '[');
  await tick();
  assert.deepEqual(settingsWrites().at(-1), { brush_radius: 25, brush_feather: 12 });
  assert.equal(settingsWrites().length, 2);
});

test('Shift-wheel resizes the brush without zooming and saves once on release', async () => {
  const h = harness();
  await tick();
  await h.run("preferences.update({ remember_brush: true, brush_radius: 24, brush_feather: 24 })");
  await h.run('openFile(file)');
  h.run("selectMode('range'); zoomTo(2)");
  const canvas = h.elements.get('canvas');
  const wheel = (deltaY, extra = {}) => canvas.fire('wheel', {
    shiftKey: true, deltaMode: 0, deltaX: 0, deltaY, clientX: 410, clientY: 290, preventDefault() {}, ...extra,
  });
  const diameter = () => h.elements.get('radius-value').textContent;
  const viewport = () => h.run('JSON.stringify([state.fit, state.zoom, state.panX, state.panY])');
  const initialView = viewport();
  const writes = () => h.api.requests.filter(item => item.route === '/api/settings' && item.args.method === 'POST').slice(1);
  wheel(100);
  assert.equal(diameter(), '46 px');
  assert.equal(h.elements.get('feather-value').textContent, '23 px');
  wheel(0, { deltaX: -100 });
  assert.equal(diameter(), '48 px');
  for (let index = 0; index < 3; index++) wheel(-10);
  assert.equal(diameter(), '48 px');
  wheel(-10);
  assert.equal(diameter(), '50 px');
  wheel(-3, { deltaMode: 1 });
  assert.equal(diameter(), '52 px');
  assert.equal(viewport(), initialView);
  assert.equal(h.run('state.draft.cursor'), 0);
  await tick();
  assert.equal(writes().length, 0);
  h.window.fire('keyup', { key: 'Shift' });
  await tick();
  assert.deepEqual(writes().map(item => JSON.parse(item.args.body)), [{ brush_radius: 26, brush_feather: 23 }]);

  wheel(-100, { shiftKey: false });
  assert.equal(diameter(), '52 px');
  assert.ok(h.run('state.zoom') > 2);
  h.run("selectMode('parts'); selectTool('lasso')");
  wheel(100);
  assert.equal(diameter(), '52 px');
});

test('changing only the saved size in settings updates the brush controls', async () => {
  const h = harness();
  await tick();
  await h.run("preferences.update({ remember_brush: true })");
  await h.elements.get('open-settings').click();
  h.elements.get('pref-brush-size').value = '60';
  await h.elements.get('settings-form').fire('submit', { preventDefault() {} });
  assert.equal(h.elements.get('radius-value').textContent, '60 px');
});

test('host export uses the exact displayed revision and reports its local path', async () => {
  const h = harness();
  await tick();
  await h.run("preferences.update({ export_dir: 'G:/exports' })");
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  await tick();
  await h.run('download()');
  const written = h.api.requests.filter(item => item.route.endsWith('/export'));
  assert.equal(written.length, 1);
  assert.deepEqual(JSON.parse(written[0].args.body), { revision: 0, view: 'b' });
  assert.match(h.elements.get('notice-text').textContent, /G:\/exports\/export.png/);
  h.run("selectMode('range'); addEdit({tool:'include',stage:'range',shape:'brush',points:[[20,30]],radius:5,feather:3})");
  await h.run('download()');
  assert.equal(h.api.requests.filter(item => item.route.endsWith('/export')).length, 1);
});

test('classification feedback names the applied part even after selecting another target', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('parts'); setPart('outfit')");
  stroke(h, [[200, 120]]);
  await h.run('saveDraft()');
  h.api.mutate({ edit_feedback: { tool: 'classify', changed_pixels: 40, skipped_protected_pixels: 12, reason: 'changed' } });
  h.api.ready();
  await h.run('pollDocument()');
  h.run("setPart('hair')");
  assert.match(h.elements.get('stroke-feedback').textContent, /40 个像素归入outfit/);
  assert.match(h.elements.get('stroke-feedback').textContent, /跳过 12 个受保护像素/);
});

test('part protection defaults to known parts and snapshots the switches for each stroke', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('parts'); setPart('outfit')");
  const lock = id => h.elements.get('part-targets').children.flatMap(row => row.children)
    .find(button => button.dataset?.protectPart === id);
  const skin = lock('skin'), unknown = lock('unknown');
  assert.ok(skin && unknown);
  assert.equal(skin['aria-pressed'], 'true');
  assert.equal(unknown['aria-pressed'], 'false');

  stroke(h, [[200, 120]]);
  const first = () => JSON.parse(h.run('JSON.stringify(state.draft.edits[0])'));
  assert.deepEqual(first().protected_parts, defaultProtectedParts);
  skin.click();
  unknown.click();
  assert.equal(h.run('state.draft.cursor'), 1);
  assert.equal(skin['aria-pressed'], 'false');
  assert.equal(unknown['aria-pressed'], 'true');
  stroke(h, [[210, 130]]);
  const second = JSON.parse(h.run('JSON.stringify(state.draft.edits[1])'));
  assert.equal(second.protected_parts.includes('skin'), false);
  assert.equal(second.protected_parts.includes('unknown'), true);
  assert.equal(first().protected_parts.includes('skin'), true);

  await h.run("openRecent('source')");
  h.run("selectMode('parts')");
  assert.equal(skin['aria-pressed'], 'false');
  assert.equal(unknown['aria-pressed'], 'true');
});

test('each range part row keeps its selection, strength and lock independently editable', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range')");
  const list = h.elements.get('range-part-list');
  assert.ok(list, 'the range controls share one part list');
  const distinctParts = list.children.map(row => row.dataset.rangePart);
  assert.equal(new Set(distinctParts).size, 9);
  const hair = list.children.find(row => row.dataset.rangePart === 'hair');
  const [member, strength, lock] = hair.children;
  const [enabled] = member.children, [slider, output] = strength.children;
  assert.equal(enabled.checked, true);
  assert.equal(output.textContent, '100%');
  lock.click();
  slider.value = '.64';
  slider.fire('input');
  assert.equal(output.textContent, '64%');
  assert.equal(h.run("state.draft.settings.part_strengths.hair"), .64);
  enabled.checked = false;
  enabled.fire('change');
  assert.equal(h.run("state.draft.settings.modules.includes('hair')"), false);
  assert.equal(lock['aria-pressed'], 'true');
  const object = list.children.find(row => row.dataset.rangePart === 'object');
  assert.equal(object.children[0].children.some(child => child.tagName === 'INPUT'), false);
  assert.equal(object.children[2].dataset.rangeProtectPart, 'object');
});

test('range locks are independent of classification and snapshot local and bulk edits', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range')");
  const locks = h.elements.get('range-part-list');
  assert.ok(locks, 'range editing exposes its own locks');
  const skin = locks.children.find(row => row.dataset.rangePart === 'skin').children[2];
  const all = h.elements.get('range-protection-all');
  assert.equal(skin['aria-pressed'], 'false');
  skin.click();
  assert.equal(skin['aria-pressed'], 'true');
  assert.equal(all['aria-pressed'], 'mixed');
  assert.equal(h.run('state.draft.cursor'), 0);
  stroke(h, [[200, 120]]);
  h.elements.get('invert').click();
  h.run("selectMode('parts')");
  const semantic = h.elements.get('part-targets').children.flatMap(row => row.children)
    .find(button => button.dataset.protectPart === 'skin');
  semantic.click();
  assert.equal(semantic['aria-pressed'], 'false');
  h.run("selectMode('range')");
  assert.equal(skin['aria-pressed'], 'true');
  all.click();
  assert.equal(all['aria-pressed'], 'true');
  h.elements.get('reset').click();
  all.click();
  assert.equal(all['aria-pressed'], 'false');
  skin.click();
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits.map(edit => edit.protected_parts))')),
    [['skin'], ['skin'], [...defaultProtectedParts, 'unknown']]);
  await h.run("openRecent('source')");
  h.run("selectMode('range')");
  assert.equal(skin['aria-pressed'], 'true');
  assert.equal(h.run('state.draft.cursor'), 3);
});

test('part eyes hide independently of classification and ignore outdated filtered previews', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.ready();
  await h.run('pollDocument()');
  h.run("selectMode('parts')");
  await tick();
  const control = (action, part) => h.elements.get('part-targets').children.flatMap(row => row.children)
    .find(button => button.dataset[action] === part);
  const hair = control('showPart', 'hair'), skin = control('showPart', 'skin');
  assert.ok(hair && skin);
  assert.equal(hair['aria-pressed'], 'true');
  const displayed = h.run("visibleImage('parts').src");
  const pending = deferred();
  h.api.beforeDecode = url => url.endsWith('hide=hair') ? pending.promise : undefined;
  hair.click();
  await tick();
  assert.equal(h.run("visibleImage('parts')?.src"), displayed);
  assert.match(h.elements.get('status').textContent, /正在更新分区显示/);
  skin.click();
  await tick();
  assert.equal(hair['aria-pressed'], 'false');
  assert.equal(skin['aria-pressed'], 'false');
  assert.equal(control('protectPart', 'hair')['aria-pressed'], 'true');
  assert.equal(h.run('state.part'), 'hair');
  assert.equal(h.run('state.draft.cursor'), 0);
  assert.match(h.run("visibleImage('parts').src"), /hide=hair%2Cskin$/);
  pending.resolve();
  await tick();
  assert.match(h.run("visibleImage('parts').src"), /hide=hair%2Cskin$/);
  await h.run("openRecent('source')");
  h.run("selectMode('parts')");
  await tick();
  assert.equal(hair['aria-pressed'], 'false');
  assert.equal(skin['aria-pressed'], 'false');
  hair.click();
  await tick();
  assert.equal(hair['aria-pressed'], 'true');
  assert.equal(skin['aria-pressed'], 'false');
  assert.match(h.run("visibleImage('parts').src"), /hide=skin$/);
  assert.equal(h.api.requests.some(item => item.route.endsWith('/edit')), false);
});

test('Ctrl-click isolates a part and restores the previous visibility on a second click', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.mutate({ status: 'ready', b: { revision: 0, images: { parts_overlay: true, unknown: true } } });
  await h.run('pollDocument()');
  h.run("selectMode('parts')");
  await tick();
  const rows = h.elements.get('part-targets').children;
  const rowFor = id => rows.find(row => row.children[0].dataset.part === id).children;
  const visible = () => rows.filter(row => row.children[1]['aria-pressed'] === 'true').map(row => row.children[0].dataset.part);
  const [hair, hairEye] = rowFor('hair'), [outfit] = rowFor('outfit'), [unknown] = rowFor('unknown');
  hairEye.click();
  outfit.click();
  const previous = visible();

  hair.fire('click', { ctrlKey: true });
  await tick();
  assert.equal(hair['aria-pressed'], 'true');
  assert.deepEqual(visible(), ['hair']);
  assert.match(h.run("visibleImage('parts').src"), /hide=background%2Ceyes%2Cface%2Cmouth%2Cobject%2Coutfit%2Cskin%2Cunknown$/);
  assert.equal(h.elements.get('unknown-legend').hidden, true);

  unknown.fire('click', { ctrlKey: true });
  await tick();
  assert.deepEqual(visible(), ['unknown']);
  assert.equal(h.elements.get('unknown-legend').hidden, false);
  unknown.fire('click', { ctrlKey: true });
  await tick();
  assert.deepEqual(visible(), previous);
  assert.match(h.run("visibleImage('parts').src"), /hide=hair$/);
  assert.equal(h.api.requests.some(item => item.route.endsWith('/edit')), false);
});

test('the master eye shows and hides every overlay and clears Ctrl-click isolation', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.mutate({ status: 'ready', b: { revision: 0, images: { parts_overlay: true, unknown: true } } });
  await h.run('pollDocument()');
  h.run("selectMode('parts')");
  await tick();
  const all = h.elements.get('part-visibility-all');
  assert.ok(all, 'the part list has a master eye');
  const rows = h.elements.get('part-targets').children;
  const eyes = rows.map(row => row.children[1]);
  const hair = rows.find(row => row.children[0].dataset.part === 'hair').children[0];
  assert.equal(all['aria-pressed'], 'true');
  hair.fire('click', { ctrlKey: true });
  assert.equal(all['aria-pressed'], 'mixed');
  all.click();
  assert.equal(all['aria-pressed'], 'true');
  assert.ok(eyes.every(eye => eye['aria-pressed'] === 'true'));
  hair.fire('click', { ctrlKey: true });
  assert.equal(all['aria-pressed'], 'mixed');
  all.click();

  all.click();
  await tick();
  assert.equal(all['aria-pressed'], 'false');
  assert.ok(eyes.every(eye => eye['aria-pressed'] === 'false'));
  assert.match(h.run("visibleImage('parts').src"), /hide=background%2Ceyes%2Cface%2Chair%2Cmouth%2Cobject%2Coutfit%2Cskin%2Cunknown$/);
  assert.equal(h.elements.get('unknown-legend').hidden, true);
  assert.equal(h.elements.get('overlay-legend').hidden, true);
  assert.equal(h.api.requests.some(item => item.route.endsWith('/edit')), false);

  await h.run("openRecent('source')");
  h.run("selectMode('parts')");
  assert.equal(all['aria-pressed'], 'false');
  eyes[0].click();
  assert.equal(all['aria-pressed'], 'mixed');
});

test('the master lock changes protection for all parts on subsequent strokes', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('parts')");
  const all = h.elements.get('part-protection-all');
  assert.ok(all, 'the part list has a master lock');
  const locks = h.elements.get('part-targets').children.map(row => row.children[2]);
  assert.equal(all['aria-pressed'], 'mixed');
  all.click();
  assert.equal(all['aria-pressed'], 'true');
  assert.ok(locks.every(lock => lock['aria-pressed'] === 'true'));
  stroke(h, [[200, 120]]);
  all.click();
  assert.equal(all['aria-pressed'], 'false');
  assert.ok(locks.every(lock => lock['aria-pressed'] === 'false'));
  stroke(h, [[210, 130]]);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits.map(edit => edit.protected_parts))')),
    [[...defaultProtectedParts, 'unknown'], []]);
  await h.run("openRecent('source')");
  h.run("selectMode('parts')");
  assert.equal(all['aria-pressed'], 'false');
  locks[0].click();
  assert.equal(all['aria-pressed'], 'mixed');
});

test('uses lightweight overlays for the active editor when the result advertises them', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.mutate({ status: 'ready', b: { revision: 0, images: { parts_overlay: true, weight_overlay: true } } });
  await h.run('pollDocument()');
  h.run("selectMode('parts')");
  await tick();
  const requests = () => h.api.requests.map(item => item.route);
  assert.ok(requests().some(route => route.includes('view=parts_overlay&revision=0')));
  assert.ok(!requests().some(route => route.includes('view=parts&') || route.includes('view=background_overlay')));
  assert.equal(h.run("viewAsset('parts').overlay"), true);
  const hair = h.elements.get('part-targets').children.flatMap(row => row.children)
    .find(button => button.dataset.showPart === 'hair');
  hair.click();
  await tick();
  assert.ok(requests().some(route => route.endsWith('view=parts_overlay&revision=0&hide=hair')));
  h.run("selectMode('range')");
  await tick();
  assert.ok(requests().some(route => route.includes('view=weight_overlay&revision=0')));
  assert.ok(!requests().some(route => route.includes('view=weight&')));
  assert.equal(h.run("viewAsset('weight').overlay"), true);
  const ramp = h.elements.get('weight-ramp');
  assert.ok(ramp, 'an indexed weight preview has a gradient legend');
  assert.equal(ramp.hidden, false);
  assert.equal(h.elements.get('weight-legacy-legend').hidden, true);
});

test('remove current part uses the selected source for brush and lasso', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('parts'); setPart('hair')");
  const menu = h.elements.get('lasso-action');
  const option = menu.children.find(item => item.value === 'remove_part');
  assert.equal(option?.textContent, '移出当前部位');
  menu.value = option.value;
  menu.fire('change');
  stroke(h, [[200, 120]]);
  h.run("selectTool('lasso')");
  const box = h.run('layout()');
  const points = [[180, 130], [200, 130], [195, 160]];
  for (const [x, y] of points) {
    h.elements.get('canvas').fire('pointerdown', { button: 0, clientX: 100 + box.x + x * box.scale,
      clientY: 50 + box.y + y * box.scale, preventDefault() {} });
  }
  h.window.fire('keydown', { key: 'Enter', preventDefault() {} });
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(state.draft.edits)')), [
    { stage: 'parts', tool: 'remove_part', module: 'hair', shape: 'brush', geometry: 'continuous', points: [[200, 120]], radius: 24, feather: 0 },
    { stage: 'parts', tool: 'remove_part', module: 'hair', shape: 'polygon', points },
  ]);
  h.run("setPart('unknown')");
  assert.equal(menu.value, 'classify');
  assert.equal(option.disabled, true);
});


test('a processing failure links to its stage and offers a retry without enabling export', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.api.mutate({ status: 'error', phase: 'error', failed_phase: 'candidate', error: '模型计算失败' });
  await h.run('pollDocument()');
  assert.match(h.elements.get('status').textContent, /降噪效果失败/);
  assert.equal(h.elements.get('retry').hidden, false);
  assert.equal(h.elements.get('download').disabled, true);
  h.elements.get('error-node').click();
  assert.equal(h.run('state.mode'), 'effect');
  h.elements.get('retry').click();
  await tick();
  assert.equal(h.run('state.doc.status'), 'queued');
  assert.ok(h.api.requests.some(item => item.route.endsWith('/edit')));
});

test('a recovery link opens the exact saved draft and removes its startup fragment', async () => {
  const id = 'a'.repeat(64);
  const h = harness({ draftLink: id });
  h.api.mutate({ id, name: '雪巷红伞.png', width: 3456, height: 2304 });
  await tick();
  assert.equal(h.run('state.doc?.id'), id);
  assert.ok(h.api.requests.some(item => item.route === '/api/documents/' + id));
  assert.deepEqual(h.api.lastHistoryReplace, [null, '', '/']);
});

test('dropping a workbench thumbnail preserves the active draft', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('flow')");
  const fileDrop = url => ({
    dataTransfer: { files: [{ ...file, name: 'thumbnail.png' }], types: ['Files', 'text/uri-list'],
      getData: type => type === 'text/uri-list' ? url : '' },
    preventDefault() {},
  });
  const drop = event => h.elements.get('viewer').fire('drop', event);
  const uploads = () => h.api.requests.filter(item => item.route === '/api/documents' && item.args.method === 'POST');
  drop(fileDrop('http://127.0.0.1:1790/api/documents/' + 'a'.repeat(64) + '/thumbnail?view=parts&revision=0'));
  await tick();
  assert.equal(uploads().length, 1);
  assert.equal(h.run('state.mode'), 'flow');
  assert.equal(h.run('state.doc.name'), file.name);

  drop(fileDrop('file:///C:/Pictures/thumbnail.png'));
  await tick();
  assert.equal(uploads().length, 2);
  assert.equal(h.run('state.mode'), 'review');
});

test('dropping into the flow opens the image in review', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('flow')");
  let intercepted = false;
  h.elements.get('viewer').fire('drop', {
    dataTransfer: { files: [file] },
    preventDefault() { intercepted = true; },
  });
  await tick();
  assert.equal(intercepted, true);
  assert.equal(h.run('state.mode'), 'review');
  assert.equal(h.api.requests.filter(item => item.route === '/api/documents' && item.args.method === 'POST').length, 2);
});

test('feathered exclusion describes fully removed and partially reduced pixels separately', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('exclude')");
  stroke(h, [[200, 120]]);
  await h.run('saveDraft()');
  h.api.mutate({ edit_feedback: {
    tool: 'exclude', reason: 'changed', changed_pixels: 437, partially_reduced_pixels: 436,
  } });
  h.api.ready();
  await h.run('pollDocument()');
  const message = h.elements.get('stroke-feedback').textContent;
  assert.match(message, /移出 1 个像素，保留原图/);
  assert.match(message, /减弱 436 个像素的处理/);
});

test('a saved feathered stroke with an older count only describes the affected area', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  h.run("selectMode('range'); selectTool('exclude')");
  h.elements.get('feather').value = '12';
  stroke(h, [[200, 120]]);
  await h.run('saveDraft()');
  h.api.mutate({ edit_feedback: { tool: 'exclude', reason: 'changed', changed_pixels: 437 } });
  h.api.ready();
  await h.run('pollDocument()');
  assert.equal(h.elements.get('stroke-feedback').textContent, '已调整 437 个像素的处理范围。');
});

test('key-activated original peek leaves all three editing views intact', async () => {
  const h = harness();
  await tick();
  await h.run('openFile(file)');
  const peek = h.elements.get('peek');
  for (const [mode, view] of [['parts', 'parts'], ['range', 'weight'], ['effect', 'candidate']]) {
    h.run("selectMode('" + mode + "'); selectView('" + view + "')");
    peek.fire('click', { detail: 0 });
    assert.equal(h.run('state.view'), view);
    assert.equal(peek['aria-pressed'], 'true');
    if (mode === 'parts') {
      stroke(h, [[200, 120]]);
      assert.equal(h.run('state.draft.cursor'), 0);
      assert.equal(peek['aria-pressed'], 'true');
    }
    peek.fire('click', { detail: 0 });
    assert.equal(h.run('state.view'), view);
    assert.equal(peek['aria-pressed'], 'false');
  }
});
