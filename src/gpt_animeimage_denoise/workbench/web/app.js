'use strict';

const byId = id => document.getElementById(id);
const themeToggle = byId('theme-toggle');
const themeRoot = document.documentElement;
const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
function applyTheme() {
  const theme = themeRoot.dataset.themePreference || (systemTheme.matches ? 'dark' : 'light');
  themeRoot.dataset.theme = theme;
  themeToggle.setAttribute('aria-pressed', String(theme === 'dark'));
  themeToggle.title = theme === 'dark' ? '切换到浅色外观' : '切换到深色外观';
}
applyTheme();
systemTheme.addEventListener('change', applyTheme);
themeToggle.addEventListener('click', () => {
  const theme = themeRoot.dataset.theme === 'dark' ? 'light' : 'dark';
  themeRoot.dataset.themePreference = theme;
  applyTheme();
  try { window.localStorage.setItem('gaid-theme', theme); }
  catch { /* Keep the selection for this page. */ }
});

const canvas = byId('canvas'), stage = byId('stage'), ctx = canvas.getContext('2d');
const magneticCanvas = byId('magnetic-canvas'), magneticCtx = magneticCanvas.getContext('2d');
const viewButtons = [...document.querySelectorAll('[data-view]')];
const toolButtons = [...document.querySelectorAll('button[data-tool]')];
const presetButtons = [...document.querySelectorAll('[data-preset]')];
const modeButtons = [...document.querySelectorAll('button[data-mode]')];
const scopeButtons = [...document.querySelectorAll('[data-scope]')];
const numberFields = ['strength', 'l0', 'downscale', 'sigma', 'restoration_gain', 'structure_protection'];
const MASK_VIEWS = ['weight', 'parts', 'suspected_objects'];
const CHANGE_CATEGORIES = ['added', 'removed', 'stronger', 'weaker', 'reclassified'];
const RANGE_VIEWS = [...MASK_VIEWS, 'changes'];
const MAGNETIC_PIN_DISTANCE = 48;
const MODE_VIEWS = { review: ['compare', 'original', 'result'], range: [...RANGE_VIEWS, 'original', 'result'],
  effect: ['compare', 'original', 'result'] };
const state = {
  options: null, doc: null, file: null, draft: null, dirty: false, save: null, saveFailed: false, conflict: null,
  epoch: 0, opening: false, images: new Map(), loading: new Map(), saveTimer: 0, saveDelay: 180, pollTimer: 0,
  zoom: 1, fit: true, panX: 0, panY: 0, split: .5, view: 'compare',
  mode: 'review', tool: 'pan', brushSized: false, peek: false, space: false, gesture: null, cursor: null, polygon: [],
  sam: null, magnetic: null, showUnknown: true, pendingUnknownResolution: null,
  changesError: null, appliedHistory: null, historyLocation: null, historyDisplayKey: '',
  displayedResult: null,
};
const isLasso = () => ['lasso', 'magnetic'].includes(state.tool);
const clone = value => JSON.parse(JSON.stringify(value));
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const route = (suffix = '') => '/api/documents/' + encodeURIComponent(state.doc.id) + suffix;
const visibleImage = view => state.images.get(view === 'result' ? 'b' : view)?.image;
const viewAvailable = view => (view !== 'suspected_objects' && view !== 'unknown') || Boolean(state.doc?.b?.images?.[view]);
const unknownOverlayAvailable = () => state.mode === 'range' && (state.view === 'parts' || state.view === 'weight')
  && viewAvailable('unknown');
const pendingKey = id => 'gaid-pending-' + id;
function status(message, tone = 'neutral') {
  const label = byId('status');
  if (label.textContent !== message) label.textContent = message;
  if (label.dataset.tone !== tone) label.dataset.tone = tone;
}
function showError(error) {
  byId('notice').dataset.tone = 'error';
  byId('notice-text').textContent = error.message || String(error);
  byId('notice').hidden = false;
}

async function request(url, options = {}) {
  const reply = await fetch(url, { cache: 'no-store', ...options });
  if (!reply.ok) {
    const body = await reply.json().catch(() => ({}));
    const error = new Error(body.error || '本机服务请求失败（' + reply.status + '）。');
    error.status = reply.status;
    throw error;
  }
  return reply;
}
async function post(url, payload) {
  return (await request(url, { method: 'POST', headers: {
    'Content-Type': 'application/json', 'X-GAID-Request': '1',
  }, body: JSON.stringify(payload) })).json();
}
function storage(method, ...args) {
  try { return window.sessionStorage[method](...args); } catch { return undefined; }
}
function persistPending() {
  if (!state.doc) return;
  const key = pendingKey(state.doc.id);
  if (state.dirty || state.save || state.conflict) {
    const resolutionIndex = state.draft.edits.indexOf(state.pendingUnknownResolution);
    const pending = { revision: state.doc.revision, draft: state.draft };
    if (resolutionIndex >= 0) pending.resolutionIndex = resolutionIndex;
    storage('setItem', key, JSON.stringify(pending));
  } else storage('removeItem', key);
}
function discardUnknownResolution(draft, index) {
  if (draft.edits[index]?.tool !== 'resolve_unknown') return false;
  draft.edits.splice(index, 1);
  if (index < draft.cursor) draft.cursor--;
  return true;
}
function discardPendingUnknownResolution() {
  const discarded = discardUnknownResolution(state.draft, state.draft.edits.indexOf(state.pendingUnknownResolution));
  state.pendingUnknownResolution = null;
  return discarded;
}
function methodLabel(name) { return state.options?.methods.find(entry => entry.id === name)?.label || name || '—'; }
function presetMatches(name) {
  if (!state.draft) return name === 'standard';
  const settings = state.draft.settings;
  if (name === 'soft') return settings.strength === .65;
  if (name === 'strong') return state.options.methods.find(item => item.id === settings.method)?.uses_l0 && settings.l0 === .001;
  return ['method', ...numberFields].every(key => settings[key] === state.options.defaults[key]);
}
function scopeModules(scope) {
  if (scope === 'empty') return [];
  return scope === 'all' ? state.options.modules.map(item => item.id) : [...state.options.default_modules];
}
function renderIsCurrent() {
  return Boolean(state.doc?.b && state.doc.b.revision === state.doc.revision
    && !state.dirty && !state.save && !state.conflict && state.doc.status === 'ready');
}
function resultIsCurrent() {
  return renderIsCurrent() && state.displayedResult === state.doc.b.revision;
}
function comparing() {
  return state.view === 'compare' && !state.peek && Boolean(visibleImage('b') && (visibleImage('a') || visibleImage('original')));
}
function skippedUnknownCount(current) {
  if (!current || state.mode !== 'range' || state.tool !== 'include' || state.gesture?.edit) return 0;
  const last = state.draft.edits[state.draft.cursor - 1], feedback = state.doc.edit_feedback;
  const count = feedback?.skipped_unknown_pixels;
  return last?.tool === 'include' && (last.shape ?? 'brush') === 'brush' && !last.allow_protected
    && feedback?.tool === 'include' && Number.isInteger(count) && count > 0 ? count : 0;
}
function updateToolGuidance(current) {
  const tool = isLasso() ? byId('lasso-action').value : state.tool;
  const region = isLasso() ? '圈住的区域' : '画过的区域';
  const descriptions = {
    pan: '展开「自动选区起点」选部位，或用左侧工具直接在图上修。拖动画布只改变查看位置。',
    include: byId('allow-protected').checked
      ? '把' + region + '加入处理范围，包括模型建议保留的像素。透明和手动保护仍生效。'
      : '把未选区域加入处理范围。可在下方允许手动补选模型建议保留的像素。',
    exclude: '把' + region + '移出处理范围，保留原像素。需要重新处理时可用补选。',
    protect: '保留部分原图，让这里少降噪。设为 100% 时，' + (isLasso() ? '圈内' : '笔画中心') + '完全保留。',
    mix: (isLasso() ? '圈内的已选区域' : '笔刷中心的已选区域') + '混合 ' + Number(byId('retention-percent').value)
      + '% 原图。' + (isLasso() ? '' : '边缘柔和过渡。'),
    erase: '擦掉手动画过的标记，恢复初始范围。想保留原像素可用「移出」。',
    override: '圈出误判的外物或待确认区域，再指定它实际属于哪个部位。',
  };
  const description = byId('tool-description');
  const help = descriptions[tool] || '';
  description.hidden = !help;
  if (description.textContent !== help) description.textContent = help;

  const output = byId('stroke-feedback');
  const feedback = state.doc?.edit_feedback;
  let message = '';
  if (current && !state.gesture?.edit && feedback?.tool === tool) {
    if (feedback.reason === 'changed' && Number.isInteger(feedback.changed_pixels) && feedback.changed_pixels > 0) {
      const count = feedback.changed_pixels.toLocaleString('zh-CN');
      message = tool === 'include' ? '已补选 ' + count + ' 个像素。'
        : tool === 'exclude' ? '已移出 ' + count + ' 个像素，保留原图。' : '已擦掉 ' + count + ' 个像素的手动标记。';
    } else if (feedback.reason === 'already_selected') message = '未新增：涂到的可编辑区域已选中。';
    else if (feedback.reason === 'protected') message = '未新增：笔画落在受保护区域。';
    else if (feedback.reason === 'no_marks') message = '这里没有手动标记，自动选区照常生效。';
    else if (feedback.reason === 'already_excluded') message = '这里原本就没有选中，不会处理。';
  }
  const skipped = skippedUnknownCount(current);
  byId('unknown-stroke-actions').hidden = !skipped;
  if (skipped) {
    const skippedMessage = '这笔跳过 ' + skipped.toLocaleString('zh-CN') + ' 个待确认像素。';
    message = feedback.changed_pixels > 0 ? message + skippedMessage : skippedMessage;
  }
  output.hidden = !message;
  if (output.textContent !== message) output.textContent = message;
}
function rangeToolName(tool) {
  return { pan: '移动画布', include: '画入处理范围', exclude: '移出处理范围', protect: '局部减弱',
    mix: '指定原图比例', erase: '擦除手动改动', override: '纠正部位', sam_object: '点选外物',
    invert: '反选处理范围', reset: '按部位恢复', import: '导入校正' }[tool] || '编辑范围';
}
function suggestedRangeView() {
  return isLasso() && byId('lasso-action').value === 'override' ? 'parts' : 'weight';
}
function editHistoryLabel(edit) {
  const module = state.options?.modules.find(item => item.id === edit.module)?.label || edit.module;
  if (edit.tool === 'resolve_unknown') return edit.module
    ? '待确认归为' + module
    : '加入本笔的待确认';
  let label = edit.tool === 'override' && module ? '纠正为' + module : rangeToolName(edit.tool);
  if (['protect', 'mix'].includes(edit.tool) && Number.isFinite(edit.strength)) {
    label += ' ' + (edit.strength * 100).toLocaleString('zh-CN', { maximumFractionDigits: 1 }) + '%';
  }
  return label + (edit.shape === 'polygon' ? ' · 圈选' : '');
}
function updateRangeHistory(canEdit) {
  if (state.mode !== 'range') return;
  const history = state.draft?.edits || [], cursor = state.draft?.cursor || 0;
  const audit = state.images.get('changes'), inAudit = state.view === 'changes';
  const current = renderIsCurrent(), now = state.doc?.revision;
  const auditCurrent = Boolean(current && audit?.revision === now);
  const error = current && state.changesError?.revision === now;
  const counts = audit?.counts;
  const showCounts = Boolean(counts && CHANGE_CATEGORIES.some(key => counts[key] > 0));
  const button = byId('show-manual-changes');
  button.disabled = !canEdit || inAudit;
  button.textContent = inAudit ? '正在查看' : '查看手动改动';
  byId('changes-details').hidden = !inAudit;
  byId('changes-counts').hidden = !showCounts;
  byId('retry-changes').hidden = !inAudit || !error;
  if (inAudit) {
    byId('changes-status').textContent = auditCurrent
      ? showCounts ? '显示当前成品 B 的净改动。' : cursor ? '手动操作已保存，当前范围和强度没有净变化。' : '当前无手动改动。'
      : error ? (audit ? '读取当前改动失败，仍显示上次的成品 B。' : '读取手动改动失败。')
        : !current ? (audit ? '显示上次成品 B，新改动待更新。' : '等成品 B 更新后即可核对。')
          : (audit ? '正在读取当前改动，仍显示上次的成品 B。' : '正在读取手动改动…');
    if (showCounts) for (const key of CHANGE_CATEGORIES) {
      byId('change-count-' + key).hidden = counts[key] === 0;
      byId('changes-' + key).textContent = counts[key].toLocaleString('zh-CN') + ' px';
    }
  }
  const recent = history.slice(Math.max(0, cursor - 4), cursor);
  byId('change-history-empty').hidden = Boolean(recent.length);
  byId('change-history-note').hidden = !recent.length;
  const applied = state.appliedHistory;
  const rows = recent.map((edit, i) => {
    const index = Math.max(0, cursor - recent.length) + i;
    const included = index < (applied?.cursor ?? 0) && JSON.stringify(applied.edits[index]) === JSON.stringify(edit);
    const points = edit.tool === 'resolve_unknown' ? history[index - 1]?.points : edit.points;
    return { index, label: editHistoryLabel(edit), included, locatable: Boolean(points?.length),
      focused: state.historyLocation === index };
  }).reverse();
  const key = JSON.stringify(rows);
  if (state.historyDisplayKey === key) return;
  state.historyDisplayKey = key;
  const list = byId('change-history-list');
  list.replaceChildren(...rows.map(row => {
    const li = document.createElement('li'), label = document.createElement('span'), status = document.createElement('small');
    label.textContent = (row.index + 1) + '  ' + row.label;
    status.textContent = row.included ? 'B 已包含' : '待更新';
    li.append(label, status);
    if (row.locatable) {
      const locate = document.createElement('button');
      locate.type = 'button';
      locate.classList.add('text-button');
      locate.textContent = row.focused ? '取消定位' : '定位';
      locate.setAttribute('aria-label', locate.textContent + '第 ' + (row.index + 1) + ' 条操作');
      locate.addEventListener('click', () => locateHistory(row.index));
      li.append(locate);
    }
    return li;
  }));
}
function updateStatus(current) {
  if (!state.doc || state.opening) return;
  const rangeMode = state.mode === 'range';
  const editing = rangeMode ? '，可继续编辑' : '';
  const phases = { queued: '等待处理', loading: '正在读取图片', analyzing: '正在识别分区', analysis: '正在识别分区',
    candidate: '正在生成降噪候选', rendering: '正在合成预览', compose: '正在合成预览',
    encoding: '正在保存结果', saving: '正在保存结果' };
  if (state.conflict) status('另一页面已更新，请选择保留哪份修改。', 'error');
  else if (state.save) status('正在提交修改' + editing, 'pending');
  else if (state.saveFailed) status('修改保存失败，可重试。', 'error');
  else if (state.dirty) status('修改待更新' + editing, 'pending');
  else if (state.doc.status === 'error') status(state.doc.error || '处理失败，请重试。', 'error');
  else if (state.doc.status !== 'ready') status((phases[state.doc.phase] || phases[state.doc.status] || '正在处理') + editing, 'pending');
  else if (rangeMode && RANGE_VIEWS.includes(state.view)) {
    const revision = state.doc.revision;
    if (state.view === 'changes' && state.changesError?.revision === revision) status('改动图读取失败，可在「核对改动」重试。', 'error');
    else if (state.images.get(state.view)?.revision === revision) {
      const label = { changes: '改动图已更新', suspected_objects: '疑似外物已显示' };
      status(label[state.view] || '范围已更新', 'ready');
    } else status('正在显示当前图层' + editing, 'pending');
  } else if (current) status('当前成品已就绪', 'ready');
  else if (state.view === 'original' && visibleImage('original')) status('原图已显示，可切换到「成品」检查。', 'neutral');
  else status('正在显示当前成品' + editing, 'pending');
}
function updateSamControls() {
  const sam = state.sam;
  const positive = sam?.points.filter(item => !item.negative).length ?? 0;
  const canPreview = Boolean(sam && !sam.busy && positive && renderIsCurrent());
  const text = !sam ? ''
    : sam.busy === 'preview' ? '正在生成预览，可继续缩放或移动画布。'
    : sam.busy === 'apply' ? '正在标记外物…'
    : sam.preview ? '橙色为待确认范围。若盖到手套或袖口，在那里加排除点，再生成预览。'
    : sam.error || (!positive ? '先点一下杯身等道具内部。'
      : !renderIsCurrent() ? '当前选区更新后即可预览。'
      : positive + ' 个目标点，' + (sam.points.length - positive) + ' 个排除点。');
  byId('sam-panel').hidden = !sam;
  for (const id of ['sam-positive', 'sam-negative', 'sam-undo-point']) byId(id).disabled = !sam || Boolean(sam.busy);
  byId('sam-positive').setAttribute('aria-pressed', String(!sam?.negative));
  byId('sam-negative').setAttribute('aria-pressed', String(Boolean(sam?.negative)));
  byId('sam-undo-point').disabled = !sam?.points.length || Boolean(sam.busy);
  byId('sam-cancel').disabled = sam?.busy === 'apply';
  byId('sam-preview').disabled = !canPreview;
  byId('sam-apply').disabled = !canPreview || sam.preview?.revision !== state.doc.revision;
  if (byId('sam-status').textContent !== text) byId('sam-status').textContent = text;
}
function updateControls() {
  const canEdit = Boolean(state.doc && state.draft && !state.opening && state.sam?.busy !== 'apply');
  const hasImage = Boolean(state.file), current = resultIsCurrent();
  const method = state.options?.methods.find(item => item.id === state.draft?.settings.method);
  byId('open').disabled = byId('empty-open').disabled = !state.options;
  const rangeMode = state.mode === 'range';
  const reviewMode = state.mode === 'review';
  document.body.dataset.mode = state.mode;
  byId('viewbar').hidden = byId('viewer-footer').hidden = byId('zoom-tools').hidden = byId('file-meta').hidden = !hasImage;
  const viewHint = comparing() ? '拖动分割线比较 · 空格拖动画布' : '滚轮缩放 · 空格拖动画布';
  if (byId('viewer-hint').textContent !== viewHint) byId('viewer-hint').textContent = viewHint;
  byId('inspector').hidden = !state.doc || state.opening;
  byId('empty-title').textContent = state.opening ? '正在打开图片…' : '拖入图片开始精修';
  byId('empty-help').hidden = state.opening;
  byId('empty-open').hidden = state.opening;
  const drawingMagnetic = Boolean(state.magnetic?.segments.length || state.magnetic?.intent);
  byId('undo').disabled = !canEdit || !rangeMode || Boolean(state.sam) || drawingMagnetic || state.draft.cursor === 0;
  byId('redo').disabled = !canEdit || !rangeMode || Boolean(state.sam) || drawingMagnetic || state.draft.cursor === state.draft.edits.length;
  byId('download-label').textContent = reviewMode ? '导出 PNG' : '去看成品';
  byId('download').disabled = reviewMode ? !current : !canEdit;
  byId('pin').disabled = !current;
  byId('export-panel').hidden = !state.doc || (!reviewMode && !rangeMode);
  byId('more-actions').textContent = rangeMode ? '校正文件' : '⌄';
  byId('more-actions').setAttribute('aria-label', rangeMode ? '校正文件' : '更多导出选项');
  byId('correction-export').hidden = byId('correction-import').hidden = !rangeMode;
  byId('download-a').hidden = !reviewMode;
  byId('correction-export').disabled = !renderIsCurrent();
  byId('download-a').disabled = !state.doc?.a;
  byId('correction-import').disabled = !canEdit || state.dirty || Boolean(state.save || state.conflict);
  byId('export-hint').textContent = reviewMode && !current ? '当前成品显示后，即可固定或导出。' : '';
  const a = state.doc?.a, b = state.images.get('b')?.meta;
  byId('comparison-a').textContent = !a ? '原图' : state.images.get('a')?.revision === a.revision
    ? '固定对照 · ' + methodLabel(a.method) : '正在载入固定对照';
  byId('comparison-b').textContent = !b ? '等待第一份成品' : methodLabel(b.method)
    + (current ? ' · 当前成品' : b.revision === state.doc?.revision && renderIsCurrent() ? ' · 待查看' : ' · 上一版');
  for (const id of ['zoom-out', 'zoom-in', 'fit', 'actual', 'peek']) byId(id).disabled = !hasImage;
  for (const [buttons, key, value, disabled] of [
    [viewButtons, 'view', state.view, !hasImage], [toolButtons, 'tool', state.tool, !canEdit],
    [modeButtons, 'mode', state.mode, !canEdit],
  ]) for (const button of buttons) {
    if (key === 'view') button.hidden = !MODE_VIEWS[state.mode].includes(button.dataset.view) || !viewAvailable(button.dataset.view);
    button.disabled = key === 'tool' ? !rangeMode || (button.dataset.tool === 'pan' ? !hasImage : disabled)
      : disabled || (button.dataset.view === 'result' && !state.doc?.b);
    button.setAttribute('aria-pressed', String(button.dataset[key] === value));
  }
  const unknownToggle = byId('show-unknown');
  unknownToggle.hidden = !unknownOverlayAvailable();
  unknownToggle.setAttribute('aria-pressed', String(state.showUnknown));
  for (const mode of ['review', 'range', 'effect']) byId(mode + '-inspector').hidden = state.mode !== mode;
  for (const button of presetButtons) {
    button.disabled = !canEdit || (button.dataset.preset === 'strong' && !method?.uses_l0);
    button.setAttribute('aria-pressed', String(Boolean(presetMatches(button.dataset.preset))));
  }
  for (const button of scopeButtons) {
    button.disabled = !canEdit;
    const selected = state.draft?.settings.modules || [];
    const expected = state.options ? scopeModules(button.dataset.scope) : [];
    button.setAttribute('aria-pressed', String(canEdit && expected.length === selected.length
      && expected.every(name => selected.includes(name))));
  }
  for (const field of document.querySelectorAll('#modules input, #part-strengths input')) field.disabled = !canEdit;
  for (const id of ['method', ...numberFields, 'radius', 'protection', 'retention-percent', 'allow-protected', 'lasso-action', 'override-module', 'invert', 'reset']) {
    byId(id).disabled = !canEdit || (id === 'l0' && !method?.uses_l0) || (id === 'downscale' && !method?.uses_downscale);
  }
  byId('lasso-options').hidden = !isLasso();
  byId('brush-radius-row').hidden = isLasso() || state.tool === 'pan';
  const operation = isLasso() ? byId('lasso-action').value : state.tool;
  byId('protect-row').hidden = !['protect', 'mix'].includes(operation);
  byId('protection-label').textContent = operation === 'mix' ? '原图比例' : '保护比例';
  byId('allow-protected-row').hidden = operation !== 'include';
  byId('override-label').hidden = byId('lasso-action').value !== 'override';
  const mag = state.magnetic;
  byId('range-scope').hidden = Boolean(state.sam || mag);
  byId('brush-panel').hidden = Boolean(state.sam);
  byId('current-tool-name').textContent = isLasso()
    ? (mag ? '磁性套索' : '套索') + ' · ' + rangeToolName(operation) : rangeToolName(operation);
  const suggested = suggestedRangeView();
  const jump = byId('suggested-range-view');
  jump.hidden = !canEdit || state.tool === 'pan' || state.view === suggested;
  jump.textContent = suggested === 'parts' ? '查看识别部位' : '查看实际处理';
  const modules = state.draft?.settings.modules || [];
  byId('scope-summary').textContent = !modules.length ? '当前从空范围开始'
    : modules.length === state.options?.modules.length ? '当前从整图开始'
      : '当前部位 · ' + modules.map(name => state.options?.modules.find(item => item.id === name)?.label || name).join('、');
  if (mag) updateMagneticCloseButton(mag, canEdit);
  else byId('close-lasso').disabled = !canEdit || state.polygon.length < 3;
  byId('magnetic-options').hidden = byId('magnetic-undo').hidden = !mag;
  byId('magnetic-undo').disabled = !canEdit || (!mag?.segments.length && mag?.intent?.kind !== 'anchor');
  byId('magnetic-radius').disabled = !canEdit || Boolean(mag?.intent);
  if (mag) {
    updateMagneticStatus(mag);
    byId('magnetic-radius-value').textContent = byId('magnetic-radius').value + ' 屏幕 px';
  }
  byId('tool-hint').textContent = mag ? (mag.segments.length ? mag.segments.length + ' 个锚点 · Enter 闭合' : '自动贴边')
    : state.tool === 'lasso'
    ? (state.polygon.length ? state.polygon.length + ' 个顶点，Enter 闭合' : '单击添加顶点') : '空格临时移动';
  updateToolGuidance(renderIsCurrent());
  updateSamControls();
  updateRangeHistory(canEdit);
  stage.dataset.tool = state.space ? 'pan' : state.tool;
  byId('conflict-local').hidden = byId('conflict-remote').hidden = !state.conflict;
  byId('retry').hidden = !state.doc || Boolean(state.conflict || state.save)
    || (!state.saveFailed && (state.doc.status !== 'error' || state.dirty));
  updateStatus(current);
}

function layout() {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (!state.file?.width || !state.file.height || !width || !height) return null;
  const fit = Math.min(Math.max(1, width - 38) / state.file.width, Math.max(1, height - 52) / state.file.height, 1);
  const scale = state.fit ? fit : state.zoom;
  const w = state.file.width * scale, h = state.file.height * scale;
  return { width, height, scale, w, h, x: (width - w) / 2 + state.panX, y: (height - h) / 2 + state.panY };
}
function paintImage(image, box) { if (image) ctx.drawImage(image, box.x, box.y, box.w, box.h); }
function drawSam(box) {
  paintImage(state.sam.preview?.image, box);
  for (const { point: [x, y], negative } of state.sam.points) {
    const cx = box.x + x * box.scale, cy = box.y + y * box.scale;
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fillStyle = negative ? '#f5a6ad' : '#8ce4c1';
    ctx.fill();
    ctx.strokeStyle = '#192320'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#182521';
    ctx.fillText(negative ? '−' : '+', cx, cy + .5);
  }
}
function strokePath(edit, box, outline = false) {
  if (!edit?.points?.length) return;
  const colors = { include: '#75badd', exclude: '#f3c87a', protect: '#ec879b', mix: '#62c8aa', erase: '#d5ded8', override: '#add4b7' };
  const [first, ...rest] = edit.points;
  const x = box.x + first[0] * box.scale, y = box.y + first[1] * box.scale;
  const radius = Math.max(1, (edit.radius || 24) * box.scale);
  ctx.save();
  ctx.globalAlpha = outline ? .9 : ['protect', 'mix'].includes(edit.tool) ? .2 + .38 * (edit.strength ?? .9) : .58;
  ctx.strokeStyle = ctx.fillStyle = colors[edit.tool] || '#add4b7';
  ctx.lineJoin = ctx.lineCap = 'round';
  ctx.beginPath();
  if (edit.shape === 'polygon') {
    ctx.moveTo(x, y);
    for (const [px, py] of rest) ctx.lineTo(box.x + px * box.scale, box.y + py * box.scale);
    if (outline) {
      if (state.cursor) ctx.lineTo(state.cursor.x, state.cursor.y);
      ctx.strokeStyle = '#13191d'; ctx.lineWidth = 4; ctx.stroke();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
      for (const [px, py] of edit.points) {
        const x = box.x + px * box.scale, y = box.y + py * box.scale;
        ctx.fillStyle = '#13191d'; ctx.fillRect(x - 3, y - 3, 6, 6);
        ctx.fillStyle = '#fff'; ctx.fillRect(x - 2, y - 2, 4, 4);
      }
    } else { ctx.closePath(); ctx.fill(); }
  } else if (!rest.length) { ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill(); }
  else {
    ctx.moveTo(x, y);
    for (const [px, py] of rest) ctx.lineTo(box.x + px * box.scale, box.y + py * box.scale);
    ctx.lineWidth = radius * 2; ctx.stroke();
  }
  ctx.restore();
}
function historyGeometry(index) {
  const history = state.draft?.edits;
  const edit = history?.[index];
  return edit?.tool === 'resolve_unknown' ? history[index - 1] : edit;
}
function locateHistory(index) {
  const box = layout(), edit = historyGeometry(index);
  if (!box || !edit?.points?.length || index >= state.draft.cursor) return;
  if (state.historyLocation === index) {
    state.historyLocation = null;
  } else {
    const xs = edit.points.map(p => p[0]), ys = edit.points.map(p => p[1]);
    const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
    const padding = 90 + 2 * (edit.radius || 0);
    const next = Math.min(1.5, box.width / (right - left + padding), box.height / (bottom - top + padding));
    state.fit = false;
    state.zoom = Math.max(box.scale, next);
    state.panX = (state.file.width - left - right) / 2 * state.zoom;
    state.panY = (state.file.height - top - bottom) / 2 * state.zoom;
    state.historyLocation = index;
  }
  stage.focus({ preventScroll: true });
  refresh();
}
function drawHistoryLocation(box) {
  if (state.historyLocation === null || state.mode !== 'range' || state.peek || state.sam) return;
  const edit = historyGeometry(state.historyLocation);
  if (!edit?.points?.length) return;
  const screen = ([x, y]) => [box.x + x * box.scale, box.y + y * box.scale];
  const [first, ...rest] = edit.points;
  const [x, y] = screen(first);
  ctx.save();
  ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
  ctx.beginPath();
  if (rest.length) {
    ctx.moveTo(x, y);
    for (const point of rest) ctx.lineTo(...screen(point));
    if (edit.shape === 'polygon') ctx.closePath();
  } else ctx.arc(x, y, Math.max(4, (edit.radius || 3) * box.scale), 0, Math.PI * 2);
  ctx.lineWidth = 3; ctx.strokeStyle = '#171c25'; ctx.stroke();
  ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.strokeStyle = '#f5f0fa'; ctx.stroke();
  ctx.restore();
}
function draw() {
  const box = layout(), showDivider = comparing();
  byId('history-location-label').hidden = !box || state.historyLocation === null || state.mode !== 'range' || state.peek || Boolean(state.sam);
  const maskView = MASK_VIEWS.includes(state.view), suspectedView = state.view === 'suspected_objects';
  const diffView = state.view === 'changes';
  byId('overlay-legend').hidden = !box || state.peek || !maskView || suspectedView;
  byId('objects-legend').hidden = byId('protection-legend').hidden = state.view !== 'parts';
  byId('weight-legend').hidden = state.view !== 'weight';
  byId('suspected-objects-legend').hidden = state.peek || !suspectedView || !visibleImage('suspected_objects');
  const unknown = state.images.get('unknown'), base = state.images.get(state.view);
  const showUnknown = Boolean(box && !state.peek && state.showUnknown && unknownOverlayAvailable()
    && unknown && unknown.revision === base?.revision);
  byId('unknown-legend').hidden = !showUnknown;
  const underlay = byId('art-underlay');
  underlay.hidden = !box;
  const divider = byId('divider'), badges = byId('compare-badges');
  const visible = box && box.x + box.w > 0 && box.x < box.width && box.y + box.h > 0 && box.y < box.height;
  divider.hidden = badges.hidden = !showDivider || !visible;
  if (!box) { drawMagneticLayer(null); return; }
  const cut = clamp(box.x + box.w * state.split, 0, box.width);
  const artStyle = underlay.style;
  artStyle.left = box.x + 'px'; artStyle.top = box.y + 'px';
  artStyle.width = box.w + 'px'; artStyle.height = box.h + 'px';
  if (showDivider && visible) {
    const top = Math.max(0, box.y);
    divider.style.left = clamp(cut, 15, box.width - 15) + 'px';
    divider.style.top = top + 'px';
    divider.style.height = Math.min(box.height, box.y + box.h) - top + 'px';
    divider.setAttribute('aria-valuenow', String(Math.round(state.split * 100)));
    const badgeInset = Math.max(9, box.width / 2 - 80);
    badges.style.left = clamp(box.x, 9, badgeInset) + 'px';
    badges.style.right = clamp(box.width - box.x - box.w, 9, badgeInset) + 'px';
    badges.style.top = Math.max(7, box.y - 27) + 'px';
  }
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(box.width * dpr)) canvas.width = Math.round(box.width * dpr);
  if (canvas.height !== Math.round(box.height * dpr)) canvas.height = Math.round(box.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  ctx.imageSmoothingEnabled = box.scale < 1.75;
  const original = visibleImage('original');
  const b = state.images.get('b');
  const showsResult = !state.peek && b && (state.view === 'result' || showDivider)
    && Math.max(box.x, showDivider ? cut : 0) < Math.min(box.x + box.w, box.width)
    && Math.max(box.y, 0) < Math.min(box.y + box.h, box.height);
  const firstResultFrame = showsResult && state.displayedResult !== b.revision;
  if (firstResultFrame) state.displayedResult = b.revision;
  if (showDivider) {
    const a = visibleImage('a');
    paintImage(a || original, box);
    ctx.save();
    ctx.beginPath(); ctx.rect(cut, 0, box.width - cut, box.height); ctx.clip();
    // Clear the right half so transparent pixels do not composite over the left reference.
    ctx.clearRect(cut, 0, box.width - cut, box.height);
    paintImage(visibleImage('b'), box);
    ctx.restore();
    const aMeta = state.images.get('a')?.meta, bMeta = state.images.get('b')?.meta;
    byId('left-badge').textContent = a && aMeta ? 'A  ' + methodLabel(aMeta.method) + ' / 选区 ' + (aMeta.selection_revision ?? aMeta.revision) : '原图';
    byId('right-badge').textContent = 'B  ' + methodLabel(bMeta?.method)
      + (bMeta ? ' / 选区 ' + (bMeta.selection_revision ?? bMeta.revision) : '')
      + (resultIsCurrent() && bMeta?.revision === state.doc.revision ? '' : ' · 待更新');
  } else if (state.peek || state.view === 'original') paintImage(original, box);
  else if (diffView) { paintImage(original, box); paintImage(visibleImage('changes'), box); }
  else paintImage(visibleImage(state.view) || original, box);
  if (showUnknown) paintImage(unknown.image, box);

  if (!state.peek && state.sam) {
    ctx.save(); drawSam(box); ctx.restore();
  } else if (!state.peek && (maskView || state.tool !== 'pan')) {
    ctx.save();
    ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
    // The brush gesture is drawn on the current frame, independently of backend jobs.
    const overlay = state.images.get(maskView ? state.view : 'b');
    if (!diffView && state.draft && (state.dirty || state.save || overlay?.revision !== state.doc?.revision)) {
      const start = Math.max(overlay?.cursor ?? 0, state.draft.cursor - 32);
      for (const edit of state.draft.edits.slice(start, state.draft.cursor)) strokePath(edit, box);
    }
    strokePath(state.gesture?.edit, box);
    if (state.polygon.length) strokePath({ tool: byId('lasso-action').value, shape: 'polygon', points: state.polygon }, box, true);
    if (state.cursor && !state.space && ['include', 'exclude', 'protect', 'mix', 'erase'].includes(state.tool)) {
      ctx.beginPath();
      ctx.arc(state.cursor.x, state.cursor.y, Math.max(1.5, Number(byId('radius').value) * box.scale), 0, Math.PI * 2);
      ctx.strokeStyle = '#1e2020'; ctx.lineWidth = 3; ctx.stroke();
      ctx.strokeStyle = '#f4f6f1'; ctx.lineWidth = 1.2; ctx.stroke();
    }
    ctx.restore();
  }
  drawHistoryLocation(box);
  const zoom = Math.round(box.scale * 100) + '%';
  if (byId('zoom-level').textContent !== zoom) byId('zoom-level').textContent = zoom;
  if (!isLasso()) updateBrushReadout(box);
  drawMagneticLayer(box);
  if (firstResultFrame && resultIsCurrent()) updateControls();
}
function zoomTo(scale, x, y) {
  const box = layout();
  if (!box) return;
  const next = clamp(scale, .015, 16);
  const ax = x ?? box.width / 2, ay = y ?? box.height / 2;
  state.zoom = next; state.fit = false;
  state.panX = ax - (ax - box.x) / box.scale * next - (box.width - state.file.width * next) / 2;
  state.panY = ay - (ay - box.y) / box.scale * next - (box.height - state.file.height * next) / 2;
  if (state.magnetic) hoverMagnetic(state.cursor, state.magnetic.straight);
  draw();
}
function position(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}
function sourcePoint(point, allowOutside = false, toPixel = Math.round) {
  const box = layout();
  if (!box) return null;
  const x = (point.x - box.x) / box.scale, y = (point.y - box.y) / box.scale;
  if (!allowOutside && (x < 0 || y < 0 || x >= state.file.width || y >= state.file.height)) return null;
  return [clamp(toPixel(x), 0, state.file.width - 1), clamp(toPixel(y), 0, state.file.height - 1)];
}
function refresh() { updateControls(); draw(); }


async function decodeImage(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } finally { URL.revokeObjectURL(url); }
}
function loadAsset(view, revision, meta = null) {
  if (state.images.get(view)?.revision === revision) return;
  const key = view + ':' + revision;
  if (state.loading.has(key)) return;
  const epoch = state.epoch;
  const cursor = state.doc.revision === revision ? state.doc.cursor : 0;
  const query = typeof revision === 'number' ? '&revision=' + revision : '';
  const promise = request(route('/image?view=' + view + query))
    .then(reply => reply.blob()).then(decodeImage).then(image => {
      if (epoch !== state.epoch) return;
      const expected = view === 'original' ? revision : view === 'a' ? state.doc.a?.revision : state.doc.b?.revision;
      if (revision !== expected || !viewAvailable(view)) return;
      state.images.set(view, { image, revision, meta, cursor });
      if (view === 'original') prepareMagnetic();
      refresh();
    }).catch(error => {
      if (epoch === state.epoch && ![404, 409].includes(error.status)) showError(error);
    }).finally(() => {
      if (state.loading.get(key) === promise) state.loading.delete(key);
    });
  state.loading.set(key, promise);
}
function loadChanges() {
  if (state.view !== 'changes' || !renderIsCurrent()) return;
  const revision = state.doc.revision, epoch = state.epoch, key = 'changes:' + revision;
  if (state.images.get('changes')?.revision === revision || state.loading.has(key)
      || state.changesError?.revision === revision) return;
  const cursor = state.doc.cursor;
  const current = () => epoch === state.epoch && state.view === 'changes'
    && state.doc.revision === revision && renderIsCurrent();
  const promise = request(route('/changes?revision=' + revision)).then(reply => reply.json()).then(async value => {
    if (!current()) return;
    const { counts, overlay } = value;
    if (value.revision !== revision || !counts || CHANGE_CATEGORIES.some(name => !Number.isSafeInteger(counts[name]) || counts[name] < 0)
        || (overlay !== null && (typeof overlay !== 'string' || !overlay.startsWith('data:image/png;base64,')))) {
      throw new Error('手动改动数据无效。');
    }
    let image = null;
    if (overlay) {
      image = new Image();
      image.src = overlay;
      await image.decode();
      if (!current()) return;
      if (image.naturalWidth !== state.file.width || image.naturalHeight !== state.file.height) {
        throw new Error('手动改动图与原图尺寸不一致。');
      }
    }
    state.images.set('changes', { image, counts, revision, cursor });
    refresh();
  }).catch(() => {
    if (current()) {
      state.changesError = { revision };
      refresh();
    }
  }).finally(() => {
    if (state.loading.get(key) === promise) state.loading.delete(key);
  });
  state.loading.set(key, promise);
}
function ensureAssets() {
  if (!state.doc) return;
  loadAsset('original', 'normalized');
  if (state.doc.b && (state.mode === 'review' || ['compare', 'result'].includes(state.view))) {
    loadAsset('b', state.doc.b.revision, state.doc.b);
  }
  if (state.doc.a && state.view === 'compare') loadAsset('a', state.doc.a.revision, state.doc.a);
  else if (!state.doc.a) state.images.delete('a');
  if (state.doc.b && MASK_VIEWS.includes(state.view) && viewAvailable(state.view)) loadAsset(state.view, state.doc.b.revision);
  if (state.showUnknown && unknownOverlayAvailable()) loadAsset('unknown', state.doc.b.revision);
  if (state.view === 'changes') loadChanges();
}
function selectView(view) {
  if (!state.file || !MODE_VIEWS[state.mode].includes(view) || !viewAvailable(view)) return;
  if (state.sam && view !== 'original') {
    if (state.sam.busy === 'apply') return;
    leaveSam();
    state.tool = 'pan';
  }
  state.view = view;
  ensureAssets();
  refresh();
}
function selectMode(mode) {
  if (!state.doc || !['review', 'range', 'effect'].includes(mode)) return;
  if (state.sam) {
    if (state.sam.busy === 'apply' || mode === 'range') return;
    leaveSam();
  }
  if (state.magnetic && mode === 'range') return;
  leaveMagnetic();
  state.mode = mode;
  state.polygon = [];
  if (mode !== 'range') { state.tool = 'pan'; state.historyLocation = null; }
  if (mode === 'range' && !RANGE_VIEWS.includes(state.view)) selectView('weight');
  else if (mode !== 'range' && RANGE_VIEWS.includes(state.view)) selectView('compare');
  else { ensureAssets(); refresh(); }
}
function selectTool(tool) {
  if (tool !== 'pan' && (!state.doc || state.mode !== 'range')) return;
  if (state.sam?.busy === 'apply') return;
  if (state.magnetic && tool === 'magnetic') return;
  leaveMagnetic();
  if (tool === 'sam_object') {
    if (state.sam) return;
    state.sam = { points: [], negative: false, preview: null, busy: null, error: '', previousView: state.view };
    state.tool = tool;
    selectView('original');
    return;
  }
  leaveSam();
  state.polygon = [];
  state.tool = tool;
  if (tool === 'magnetic') {
    const sourceRevision = renderIsCurrent() ? state.doc.b.revision : null;
    state.magnetic = { segments: [], autoPins: new Set(), pointCount: 0, worker: null, ready: false, failed: false,
      sequence: 0, generation: 0, sourceRevision,
      sourceKey: magneticSourceKey(state.draft.settings) };
    prepareMagnetic();
  } else if (tool !== 'pan' && tool !== 'lasso') {
    const box = layout();
    const radius = byId('radius');
    if (!state.brushSized && box && Number(radius.value) * box.scale < 6) {
      radius.value = clamp(Math.round(9 / box.scale), 24, 256);
    }
    state.brushSized = true;
  }
  ensureAssets(); refresh();
}

function leaveSam() {
  if (!state.sam) return;
  state.view = viewAvailable(state.sam.previousView) ? state.sam.previousView : 'weight';
  state.sam = null;
}

function installDraft(doc) {
  state.historyLocation = null;
  state.draft = { settings: clone(doc.settings), edits: clone(doc.edits), cursor: doc.cursor };
  syncSettings();
}
function applyMetadata(doc) {
  if (state.sam && state.doc?.revision !== doc.revision) state.sam.preview = null;
  state.doc = doc;
  if (doc.status === 'ready' && doc.b?.revision === doc.revision && state.appliedHistory?.revision !== doc.revision) {
    state.appliedHistory = { revision: doc.revision, cursor: doc.cursor, edits: doc.edits };
  }
  if (!viewAvailable('unknown')) state.images.delete('unknown');
  if (!viewAvailable('suspected_objects')) {
    state.images.delete('suspected_objects');
    if (state.view === 'suspected_objects') state.view = 'weight';
  }
  state.file = { name: doc.name, width: doc.width, height: doc.height };
  byId('image-name').textContent = state.file.name;
  byId('image-name').title = state.file.name;
  byId('image-size').textContent = doc.width + ' × ' + doc.height;
  byId('coverage').textContent = Number.isFinite(doc.active_percent) ? 'B 处理 ' + doc.active_percent.toFixed(1) + '%' : '';
  byId('empty').hidden = true;
  ensureAssets();
  refresh();
}
function startDocument(doc) {
  state.dirty = state.saveFailed = false;
  state.conflict = null;
  installDraft(doc);
  applyMetadata(doc);
  storage('setItem', 'gaid-active-document', doc.id);
  const pending = storage('getItem', pendingKey(doc.id));
  if (pending) {
    try {
      const saved = JSON.parse(pending);
      if (JSON.stringify(saved.draft) === JSON.stringify(state.draft)) storage('removeItem', pendingKey(doc.id));
      else {
        state.draft = saved.draft;
        state.dirty = true;
        if (saved.revision !== doc.revision) {
          discardUnknownResolution(state.draft, saved.resolutionIndex);
          state.conflict = doc;
          persistPending();
        } else {
          state.pendingUnknownResolution = state.draft.edits[saved.resolutionIndex] ?? null;
        }
        syncSettings();
        scheduleSave();
        refresh();
      }
    } catch { storage('removeItem', pendingKey(doc.id)); }
  }
  schedulePoll();
}
function beginDocumentChange() {
  clearTimeout(state.pollTimer);
  state.epoch += 1;
  state.opening = true;
  state.doc = state.file = state.draft = null;
  state.conflict = null;
  state.dirty = state.saveFailed = false;
  state.images.clear();
  state.loading.clear();
  state.displayedResult = null;
  state.polygon = [];
  state.sam = null;
  leaveMagnetic();
  state.gesture = state.cursor = null;
  state.showUnknown = true;
  state.changesError = state.appliedHistory = state.historyLocation = null;
  state.historyDisplayKey = '';
  state.pendingUnknownResolution = null;
  state.fit = true; state.panX = state.panY = 0;
  state.view = 'compare'; state.mode = 'review'; state.tool = 'pan'; state.brushSized = false; state.peek = false;
  byId('allow-protected').checked = false;
  byId('notice').hidden = true;
  byId('empty').hidden = false;
  byId('image-name').textContent = '';
  byId('image-name').title = '';
  byId('image-size').textContent = '';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  refresh();
  return state.epoch;
}

async function openFile(file) {
  if (!file || !state.options || state.opening) return;
  if (file.size > 64 * 1024 * 1024) { showError(new Error('请选择不超过 64 MiB 的图片。')); return; }
  if (!(await flushChanges()) || state.opening) return;
  const epoch = beginDocumentChange();
  status('正在打开图片…', 'pending');
  // Decode and transfer in parallel, so the view can appear before any model analysis.
  decodeImage(file).then(image => {
    if (epoch !== state.epoch || state.images.get('original')?.revision === 'normalized') return;
    if (!state.doc) state.file = { name: file.name, width: image.naturalWidth, height: image.naturalHeight };
    state.images.set('original', { image, revision: 'local' });
    byId('image-name').textContent = state.file.name;
    byId('image-name').title = state.file.name;
    byId('image-size').textContent = state.file.width + ' × ' + state.file.height;
    byId('empty').hidden = true;
    refresh();
  }).catch(() => { /* TIFF and other browser-undecodable sources use the server PNG. */ });
  try {
    const response = await request('/api/documents', { method: 'POST', body: file, headers: {
      'Content-Type': 'application/octet-stream', 'X-GAID-Request': '1', 'X-GAID-Filename': encodeURIComponent(file.name),
    } });
    const doc = await response.json();
    if (epoch === state.epoch) {
      state.opening = false;
      startDocument(doc);
      loadRecent();
    }
  } catch (error) {
    if (epoch === state.epoch) { state.opening = false; showError(error); status('无法打开图片，请重新选择。', 'error'); refresh(); }
  }
}
async function openRecent(id) {
  if (state.opening || !(await flushChanges()) || state.opening) return;
  byId('recent-panel').open = false;
  const epoch = beginDocumentChange();
  status('正在载入已保存的图片…', 'pending');
  try {
    const doc = await (await request('/api/documents/' + encodeURIComponent(id))).json();
    if (epoch === state.epoch) { state.opening = false; startDocument(doc); }
  } catch (error) {
    if (epoch === state.epoch) { state.opening = false; showError(error); status('无法恢复草稿，请从「近期」重试。', 'error'); refresh(); }
  }
}
async function loadRecent() {
  try {
    const payload = await (await request('/api/documents')).json();
    const list = Array.isArray(payload) ? payload : payload.documents;
    const panel = byId('recent');
    panel.replaceChildren();
    if (!list?.length) {
      const text = document.createElement('p');
      text.textContent = '还没有近期图片';
      panel.append(text);
    }
    for (const entry of list || []) {
      const button = document.createElement('button');
      button.type = 'button'; button.textContent = entry.name; button.title = entry.name;
      button.addEventListener('click', () => openRecent(entry.id));
      panel.append(button);
    }
  } catch (error) { showError(error); }
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  if (state.doc) state.pollTimer = setTimeout(pollDocument, ['queued', 'running'].includes(state.doc.status) ? 140 : 1250);
}
async function pollDocument() {
  const epoch = state.epoch, observed = state.doc;
  if (!state.doc) return;
  try {
    const doc = await (await request(route())).json();
    if (epoch !== state.epoch || state.save || state.doc !== observed || doc.revision < state.doc.revision) return;
    if (doc.revision > state.doc.revision) {
      if (state.dirty || state.conflict) {
        discardPendingUnknownResolution();
        state.conflict = doc;
        persistPending();
        refresh();
        return;
      }
      installDraft(doc);
    }
    applyMetadata(doc);
  } catch (error) {
    if (epoch === state.epoch) showError(error);
  } finally {
    if (epoch === state.epoch) schedulePoll();
  }
}
function scheduleSave() {
  clearTimeout(state.saveTimer);
  if (state.doc && state.dirty && !state.save && !state.conflict) state.saveTimer = setTimeout(saveDraft, state.saveDelay);
}
function edited(delay = 180) {
  state.dirty = true;
  state.saveFailed = false;
  state.historyLocation = null;
  state.saveDelay = delay;
  byId('notice').hidden = true;
  persistPending();
  scheduleSave();
  refresh();
}
async function saveDraft() {
  if (state.save) return state.save;
  if (!state.doc || !state.dirty || state.conflict) return !state.conflict;
  const epoch = state.epoch, endpoint = route('/edit');
  const payload = { revision: state.doc.revision, ...clone(state.draft) };
  state.dirty = state.saveFailed = false;
  state.save = (async () => {
    try {
      const doc = await post(endpoint, payload);
      if (epoch !== state.epoch) return false;
      state.pendingUnknownResolution = null;
      applyMetadata(doc);
      return true;
    } catch (error) {
      if (epoch !== state.epoch) return false;
      state.dirty = true;
      if (error.status === 409) {
        const discarded = discardPendingUnknownResolution();
        try { state.conflict = await (await request(route())).json(); }
        catch (readError) { showError(readError); }
        if (discarded) showError(new Error('这笔待确认结果已更新，快捷修正未提交。'));
      } else { state.saveFailed = true; showError(error); }
      return false;
    }
  })();
  updateStatus(false);
  byId('retry').hidden = true;
  const result = await state.save;
  if (epoch === state.epoch) {
    state.save = null;
    persistPending();
    ensureAssets();
    refresh();
    if (result) scheduleSave();
    schedulePoll();
  }
  return result;
}
async function flushChanges() {
  clearTimeout(state.saveTimer);
  if (state.conflict) { showError(new Error('请先决定如何处理此图片的版本冲突。')); return false; }
  while (state.save || state.dirty) {
    if (!(await (state.save || saveDraft()))) return false;
  }
  return !state.conflict;
}

function updateBrushReadout(box = layout()) {
  const value = Number(byId('radius').value);
  byId('radius-value').textContent = value + ' px' + (box ? ' · 屏幕 ' + Math.max(1, Math.round(value * box.scale)) + ' px' : '');
}
function updateReadouts() {
  for (const key of ['strength', 'restoration_gain', 'structure_protection', 'protection']) {
    byId(key + '-value').textContent = Math.round(Number(byId(key).value) * 100) + '%';
  }
  byId('retention-percent').value = Math.round(Number(byId('protection').value) * 1000) / 10;
  updateBrushReadout();
}
function syncSettings() {
  const settings = state.draft.settings;
  for (const key of ['method', ...numberFields]) byId(key).value = settings[key];
  for (const node of document.querySelectorAll('#modules input')) node.checked = settings.modules.includes(node.value);
  for (const node of document.querySelectorAll('#part-strengths input')) {
    node.value = settings.part_strengths?.[node.dataset.module] ?? 1;
    node.nextElementSibling.textContent = Math.round(Number(node.value) * 100) + '%';
  }
  updateReadouts();
}
function installOptions(options) {
  state.options = options;
  const order = ['skin', 'face', 'hair', 'eyes', 'mouth', 'outfit', 'background'];
  const modules = [...options.modules].sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  for (const item of modules) {
    const label = document.createElement('label'), check = document.createElement('input');
    const swatch = document.createElement('i');
    swatch.className = 'dot';
    if (/^#[0-9a-f]{6}$/i.test(item.color || '')) swatch.style.backgroundColor = item.color;
    label.className = 'module-chip'; check.type = 'checkbox'; check.value = item.id;
    check.checked = options.default_modules.includes(item.id); check.disabled = true;
    check.addEventListener('change', () => {
      if (!state.draft) return;
      state.draft.settings.modules = [...document.querySelectorAll('#modules input:checked')].map(input => input.value);
      edited();
    });
    label.append(check, swatch, item.label);
    byId('modules').append(label);

    const row = document.createElement('label'), output = document.createElement('output'), range = document.createElement('input');
    row.className = 'range-row';
    range.type = 'range'; range.min = '0'; range.max = '1'; range.step = '.01'; range.value = '1';
    range.dataset.module = item.id; range.disabled = true;
    output.textContent = '100%'; output.style.gridColumn = '2'; output.style.gridRow = '1';
    range.addEventListener('input', () => {
      if (!state.draft) return;
      state.draft.settings.part_strengths ||= {};
      state.draft.settings.part_strengths[item.id] = Number(range.value);
      output.textContent = Math.round(Number(range.value) * 100) + '%';
      edited();
    });
    row.append(item.label, range, output);
    byId('part-strengths').append(row);
    if (item.id !== 'background') {
      const option = document.createElement('option');
      option.value = item.id; option.textContent = item.label;
      byId('override-module').append(option);
      const assignment = document.createElement('option');
      assignment.value = item.id; assignment.textContent = item.label;
      byId('unknown-stroke-module').append(assignment);
    }
  }
  byId('override-module').value = 'hair';
  byId('unknown-stroke-module').value = 'skin';
  for (const item of options.methods) {
    const option = document.createElement('option');
    option.value = item.id; option.textContent = item.label;
    byId('method').append(option);
  }
  for (const key of ['method', ...numberFields]) byId(key).value = options.defaults[key];
  updateReadouts();
  status('本机服务就绪');
  refresh();
}
function choosePreset(name) {
  if (!state.draft) return;
  const settings = state.draft.settings;
  if (name === 'soft') settings.strength = .65;
  else if (name === 'strong') {
    if (!state.options.methods.find(item => item.id === settings.method)?.uses_l0) return;
    settings.l0 = .001;
  } else Object.assign(settings, state.options.defaults);
  syncSettings();
  edited();
}

function chooseScope(scope) {
  if (!state.draft) return;
  state.draft.settings.modules = scopeModules(scope);
  syncSettings();
  edited();
}


function addEdit(edit) {
  if (!state.draft || !state.doc) return;
  const history = state.draft.edits.slice(0, state.draft.cursor);
  const points = history.reduce((sum, item) => sum + (item.points?.length || 0), 0);
  if (history.length >= 2000 || points + (edit.points?.length || 0) > 50000) {
    showError(new Error('本次编辑记录已达上限。请先导出当前结果。'));
    return;
  }
  state.draft.edits = [...history, edit];
  state.draft.cursor = state.draft.edits.length;
  edited(0);
}
function resolveUnknownStroke(module) {
  if (!skippedUnknownCount(renderIsCurrent())) return;
  if (module !== undefined && (module === 'background' || !state.options.modules.some(item => item.id === module))) return;
  const edit = module === undefined ? { tool: 'resolve_unknown' } : { tool: 'resolve_unknown', module };
  addEdit(edit);
  if (state.draft.edits[state.draft.cursor - 1] !== edit) return;
  state.pendingUnknownResolution = edit;
  persistPending();
  stage.focus({ preventScroll: true });
  saveDraft();
}
function undo() { if (state.draft?.cursor > 0) { state.draft.cursor--; edited(); } }
function redo() {
  if (state.draft && state.draft.cursor < state.draft.edits.length) { state.draft.cursor++; edited(); }
}
function closeLasso(includePreview = true) {
  const mag = state.magnetic;
  if (mag) {
    const tail = includePreview ? magneticClosingTail(mag) : null;
    if (canCloseMagnetic(mag, tail)) queueMagnetic(mag.segments[0][0], 'close', mag.straight, true, tail);
    return;
  }
  if (state.tool !== 'lasso' || state.polygon.length < 3) return;
  const points = state.polygon;
  state.polygon = [];
  addEdit({ ...lassoAction(), points });
}
function lassoAction() {
  const tool = byId('lasso-action').value;
  const edit = { tool, shape: 'polygon' };
  if (tool === 'protect' || tool === 'mix') edit.strength = Number(byId('protection').value);
  if (tool === 'include' && byId('allow-protected').checked) edit.allow_protected = true;
  if (tool === 'override') edit.module = byId('override-module').value;
  return edit;
}

function leaveMagnetic() {
  state.magnetic?.worker?.terminate();
  state.magnetic = null;
  magneticCanvas.hidden = true;
}
function magneticSourceKey(settings) {
  return JSON.stringify(['method', 'l0', 'downscale', 'sigma', 'restoration_gain'].map(key => settings[key]));
}
function magneticSource(revision, sourceKey) {
  const key = 'magnetic_source:' + revision;
  if (state.loading.has(key)) return state.loading.get(key);
  const epoch = state.epoch, controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2500);
  const promise = request(route('/image?view=magnetic_source&revision=' + revision), { signal: controller.signal })
    .then(reply => reply.blob()).then(decodeImage).then(image => {
      if (epoch !== state.epoch || !renderIsCurrent() || state.doc.b.revision !== revision ||
          image.naturalWidth !== state.file.width || image.naturalHeight !== state.file.height) return null;
      state.images.set('magnetic_source', { image, sourceKey });
      return image;
    }).catch(() => null).finally(() => {
      clearTimeout(timer);
      if (state.loading.get(key) === promise) state.loading.delete(key);
    });
  state.loading.set(key, promise);
  return promise;
}
async function prepareMagnetic() {
  const mag = state.magnetic, original = state.images.get('original');
  if (!mag || mag.initializing || mag.failed) return;
  if (typeof Worker !== 'function' || typeof createImageBitmap !== 'function') { failMagnetic(mag); return; }
  if (original?.revision !== 'normalized') return;
  let bitmap;
  mag.initializing = true;
  try {
    const cached = state.images.get('magnetic_source');
    let image = cached?.sourceKey === mag.sourceKey ? cached.image : original.image;
    if (image === original.image && mag.sourceRevision !== null) {
      mag.edgeSource = 'loading';
      updateMagneticStatus(mag);
      const candidate = await magneticSource(mag.sourceRevision, mag.sourceKey);
      if (state.magnetic !== mag) return;
      if (candidate) image = candidate;
    }
    mag.edgeSource = image === original.image ? 'original' : 'candidate';
    const worker = new Worker('/magnetic-lasso-worker.js');
    mag.worker = worker;
    worker.onmessage = ({ data }) => {
      if (state.magnetic !== mag || mag.worker !== worker) return;
      if (data.type === 'ready') {
        if (data.width !== state.file.width || data.height !== state.file.height) { failMagnetic(mag); return; }
        mag.ready = true;
        if (mag.intent) dispatchMagnetic(mag);
        else if (mag.pending) drawMagneticFrame(mag);
        if (!mag.intent && state.cursor) hoverMagnetic(state.cursor, mag.straight);
        refresh();
      } else if (data.type === 'trace' && mag.active && data.id === mag.active.id) {
        const request = mag.active;
        mag.active = null;
        acceptMagnetic(mag, request, data.points, data.snapped);
        if (mag.intent) dispatchMagnetic(mag);
        else if (mag.pending) drawMagneticFrame(mag);
      } else if (data.type === 'error') failMagnetic(mag);
    };
    worker.onerror = () => failMagnetic(mag);
    worker.onmessageerror = () => failMagnetic(mag);
    bitmap = await createImageBitmap(image);
    if (state.magnetic === mag && mag.worker === worker) {
      worker.postMessage({ type: 'init', bitmap }, [bitmap]);
      bitmap = null;
    }
  } catch {
    failMagnetic(mag);
  } finally {
    bitmap?.close();
  }
}
function failMagnetic(mag) {
  if (state.magnetic !== mag) return;
  mag.worker?.terminate();
  mag.worker = null;
  mag.ready = false; mag.failed = true;
  mag.active = mag.pending = mag.intent = mag.latest = mag.preview = null;
  refresh();
}
function updateMagneticStatus(mag) {
  const currentHover = mag.latest?.kind === 'preview' && mag.preview?.request === mag.latest;
  const missingEdge = currentHover ? !mag.preview.snapped : !mag.latest && mag.lastTrace === 'missing';
  const detail = mag.failed ? '吸边加载失败。关闭并重新启动工作台，或切换「套索」。按 Alt 可使用直线。'
    : mag.error || (mag.straight ? 'Alt 直线模式，松开后恢复吸边。'
      : !mag.ready ? (mag.edgeSource === 'loading' ? '正在加载降噪候选用于吸边。'
        : mag.intent ? '正在准备吸边，已记住这个锚点。' : '正在准备吸边，请稍等。')
      : mag.intent?.kind === 'close' ? '正在贴合闭合边线…'
      : mag.intent ? '正在固定锚点…'
      : missingEdge ? '没有找到连续边缘，这一段为直线。可贴近轮廓或加密锚点。'
      : !mag.latest && mag.lastTrace === 'closed-straight' ? '闭合处没有找到连续边缘，该段使用直线。'
      : mag.segments.length ? '沿轮廓移动，边线会自动固定。转角也可单击固定。' : '单击轮廓开始。');
  const text = mag.ready ? (mag.edgeSource === 'candidate' ? '降噪候选边缘 · ' : '原图边缘 · ') + detail : detail;
  const status = byId('magnetic-status');
  if (status.textContent !== text) status.textContent = text;
}
function magneticAnchor(mag) { return mag.segments.at(-1)?.at(-1) || null; }
function magneticClosingTail(mag) {
  const preview = mag?.preview, request = preview?.request;
  if (!preview?.snapped || mag.straight || mag.latest !== request || request.exact || request.straight ||
      !request.anchor || samePoint(request.anchor, preview.points.at(-1))) return null;
  const scale = layout()?.scale;
  return request.scale === scale && request.radius === Number(byId('magnetic-radius').value) / scale ? preview.points : null;
}
function canCloseMagnetic(mag, tail = null) {
  return Boolean(mag && !mag.intent && (mag.segments.length > 1 || tail) &&
    mag.pointCount + (tail ? tail.length - 1 : 0) >= 3);
}
function updateMagneticCloseButton(mag, canEdit = Boolean(state.doc && state.draft && !state.opening)) {
  const button = byId('close-lasso');
  const disabled = !canEdit || !canCloseMagnetic(mag, magneticClosingTail(mag));
  if (button.disabled !== disabled) button.disabled = disabled;
}
function samePoint(a, b) { return Boolean(a && b && a[0] === b[0] && a[1] === b[1]); }
function magneticDistance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
function distanceToMagneticPath(point, path) {
  let best = magneticDistance(point, path[0]);
  for (let i = 1; i < path.length; i++) {
    const [x, y] = path[i - 1];
    const dx = path[i][0] - x, dy = path[i][1] - y;
    const length = dx * dx + dy * dy;
    if (!length) continue;
    const t = clamp(((point[0] - x) * dx + (point[1] - y) * dy) / length, 0, 1);
    best = Math.min(best, Math.hypot(point[0] - x - t * dx, point[1] - y - t * dy));
  }
  return best;
}
function trackMagneticMotion(mag, point, scale) {
  const anchor = magneticAnchor(mag);
  if (!anchor || !mag.ready || mag.intent) return;
  if (!mag.motion || mag.motion.scale !== scale)
    mag.motion = { scale, points: [mag.motionOrigin || anchor], distance: 0, continuous: true };
  const motion = mag.motion;
  const distance = magneticDistance(motion.points.at(-1), point) * scale;
  if (distance > 40 || !motion.continuous) {
    // A pointer jump doesn't prove that the cursor passed along the intervening edge.
    mag.motion = { scale, points: [point], distance: 0, continuous: magneticDistance(point, anchor) * scale <= 14 };
  } else if (distance >= 3) {
    motion.distance += distance;
    if (motion.points.length >= 16) motion.points.splice(1, 1);
    motion.points.push(point);
  }
}
function magneticAutoPin(mag, request, points, snapped) {
  const motion = mag.motion;
  if (!snapped || request.straight || request.exact || !request.anchor || !motion?.continuous ||
      motion.scale !== request.scale || motion.points.length < 4 || motion.distance < MAGNETIC_PIN_DISTANCE ||
      points.length < 2) return false;
  const endpoint = points.at(-1);
  const span = magneticDistance(endpoint, request.anchor) * request.scale;
  if (span < MAGNETIC_PIN_DISTANCE || span > MAGNETIC_PIN_DISTANCE * 2.5 ||
      magneticDistance(endpoint, request.point) * request.scale > 10) return false;
  if (mag.segments.slice(-4).some(segment => segment.length > 1 &&
        distanceToMagneticPath(endpoint, segment) * request.scale <= 3)) return false;
  const corridor = Math.min(16, request.radius * request.scale + 3);
  return motion.points.every(point => distanceToMagneticPath(point, points) * request.scale <= corridor);
}
function sameMagneticPreviewScope(latest, request) {
  return latest && request.kind === 'preview' && !latest.straight && !request.straight &&
    latest.generation === request.generation && latest.scale === request.scale &&
    latest.radius === request.radius && latest.exact === request.exact;
}
function straightMagnetic(request) {
  return request.anchor && !samePoint(request.anchor, request.point) ? [request.anchor, request.point] : [request.point];
}
function queueMagnetic(point, kind, straight = false, exact = kind === 'close', tail = null) {
  const mag = state.magnetic, box = layout();
  if (!mag || !box || mag.intent) return;
  if (mag.failed && !straight) { mag.latest = mag.preview = null; return; }
  const request = { type: 'trace', id: ++mag.sequence, generation: mag.generation, kind, point,
    anchor: tail?.at(-1) || magneticAnchor(mag),
    scale: box.scale, radius: Number(byId('magnetic-radius').value) / box.scale, exact };
  if (kind === 'close') { request.action = lassoAction(); request.tail = tail; }
  const previous = mag.latest;
  if (kind === 'preview' && !straight && mag.ready && previous?.kind === 'preview' && !previous.straight &&
      samePoint(previous.point, point) && previous.radius === request.radius && previous.scale === request.scale &&
      previous.exact === exact) return;
  request.straight = straight;
  mag.latest = request;
  if (kind !== 'preview') mag.intent = request;
  if (request.straight) {
    mag.pending = null;
    acceptMagnetic(mag, request, straightMagnetic(request), false);
  } else if (kind !== 'preview' && mag.preview && mag.preview.request === previous && !previous.straight &&
      samePoint(previous.point, point) && previous.radius === request.radius && previous.scale === request.scale &&
      previous.exact === exact) {
    mag.pending = null;
    acceptMagnetic(mag, request, mag.preview.points, mag.preview.snapped);
  } else {
    mag.pending = request;
    if (mag.intent) {
      dispatchMagnetic(mag);
      updateControls();
      drawMagneticLayer();
    } else drawMagneticFrame(mag);
  }
}
function dispatchMagnetic(mag) {
  if (!mag.ready || mag.active || !mag.pending) return;
  mag.active = mag.pending; mag.pending = null;
  const { type, id, anchor, point, radius, scale, exact } = mag.active;
  try { mag.worker.postMessage({ type, id, anchor, point, radius, scale, exact }); }
  catch { failMagnetic(mag); }
}
function acceptMagnetic(mag, request, points, snapped) {
  const current = mag.latest === request;
  const compatible = !current && snapped && mag.latest?.kind === 'preview' && sameMagneticPreviewScope(mag.latest, request);
  if (!current && !compatible) return;
  if (!points?.length) points = straightMagnetic(request);
  if (request.kind === 'preview') {
    const nearby = current || magneticDistance(mag.latest.point, request.point) * request.scale <= 8;
    if (nearby && magneticAutoPin(mag, request, points, snapped) && addMagneticSegment(mag, request, points, true)) {
      mag.generation++;
      mag.pending = mag.latest = mag.preview = null;
      updateControls();
      drawMagneticFrame(mag);
      return;
    }
    mag.preview = { request, points, snapped };
    drawMagneticFrame(mag);
    return;
  }
  mag.intent = mag.latest = mag.preview = null;
  mag.generation++;
  if (request.kind === 'anchor') {
    if (addMagneticSegment(mag, request, points)) mag.lastTrace = !request.straight && !snapped ? 'missing' : null;
    updateControls();
    drawMagneticLayer();
    return;
  }
  const path = [...mag.segments, ...(request.tail ? [request.tail] : []), points]
    .flatMap((segment, i) => i === 0 ? segment : segment.slice(1));
  if (samePoint(path[0], path.at(-1))) path.pop();
  if (path.length > 5000) {
    mag.error = '选区边线过长。可撤回几个锚点，分成两次圈选。';
    refresh();
    return;
  }
  if (path.length >= 3) {
    resetMagnetic();
    if (!request.straight && !snapped) mag.lastTrace = 'closed-straight';
    addEdit({ ...request.action, points: path });
  } else refresh();
}
function addMagneticSegment(mag, request, points, automatic = false) {
  if (samePoint(request.anchor, points.at(-1))) return false;
  const added = points.length - (request.anchor ? 1 : 0);
  if (mag.pointCount + added > 5000) {
    if (!automatic) mag.error = '选区边线过长。请先闭合当前范围，再圈选下一段。';
    return false;
  }
  mag.segments.push(points);
  if (automatic) mag.autoPins.add(points);
  mag.pointCount += added;
  mag.error = mag.lastTrace = null;
  mag.motion = null;
  mag.motionOrigin = request.point;
  return true;
}
function resetMagnetic(undo = false) {
  const mag = state.magnetic;
  if (!mag) return;
  const pendingAnchor = mag.intent?.kind === 'anchor';
  // Active remains occupied until the worker replies; stale results are discarded.
  mag.generation++;
  mag.pending = mag.intent = mag.latest = mag.preview = null;
  mag.error = '';
  mag.lastTrace = null;
  if (undo && !pendingAnchor && mag.segments.length) {
    const removed = mag.segments.pop();
    mag.autoPins.delete(removed);
    mag.pointCount -= removed.length - (mag.segments.length ? 1 : 0);
  } else if (!undo) {
    mag.segments.length = 0;
    mag.autoPins.clear();
    mag.pointCount = 0;
  }
  mag.motion = mag.motionOrigin = null;
  refresh();
}
function hoverMagnetic(pos, straight = false, pointerMove = false) {
  const mag = state.magnetic;
  if (!mag) return;
  mag.straight = straight;
  if (straight) mag.motion = null;
  const point = pos && sourcePoint(pos, false, Math.floor);
  if (point && !state.space && !state.peek) {
    const scale = layout().scale, start = mag.segments[0]?.[0];
    if (pointerMove && !straight) trackMagneticMotion(mag, point, scale);
    if (canCloseMagnetic(mag) && magneticDistance(point, start) * scale <= 9)
      queueMagnetic(start, 'preview', straight, true);
    else queueMagnetic(point, 'preview', straight);
  }
  drawMagneticFrame(mag);
}
function drawMagneticFrame(mag) {
  if (mag.frame) return;
  mag.frame = requestAnimationFrame(() => {
    mag.frame = 0;
    if (state.magnetic === mag) {
      dispatchMagnetic(mag);
      updateMagneticStatus(mag);
      updateMagneticCloseButton(mag);
      drawMagneticLayer();
    }
  });
}
function drawMagneticLayer(box = layout()) {
  const mag = state.magnetic;
  const hidden = !mag || !box || state.peek;
  if (magneticCanvas.hidden !== hidden) magneticCanvas.hidden = hidden;
  if (hidden) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(box.width * dpr), height = Math.round(box.height * dpr);
  if (magneticCanvas.width !== width) magneticCanvas.width = width;
  if (magneticCanvas.height !== height) magneticCanvas.height = height;
  magneticCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  magneticCtx.clearRect(0, 0, box.width, box.height);
  magneticCtx.save();
  magneticCtx.beginPath();
  magneticCtx.rect(box.x, box.y, box.w, box.h);
  magneticCtx.clip();
  drawMagnetic(mag, box, magneticCtx);
  magneticCtx.restore();
}
function drawMagnetic(mag, box, ctx) {
  function stroke() {
    ctx.strokeStyle = '#13191d'; ctx.lineWidth = 3.5; ctx.stroke();
    ctx.strokeStyle = '#95e7c6'; ctx.lineWidth = 1.5; ctx.stroke();
  }
  let lastX, lastY;
  function addPath(points) {
    for (const [i, [x, y]] of points.entries()) {
      const px = box.x + (x + .5) * box.scale, py = box.y + (y + .5) * box.scale;
      if (i !== points.length - 1 && Math.hypot(px - lastX, py - lastY) < 1.5) continue;
      ctx.lineTo(px, py);
      lastX = px; lastY = py;
    }
  }
  if (mag.segments.length > 1) {
    ctx.beginPath();
    for (const segment of mag.segments) addPath(segment);
    stroke();
  }
  const request = mag.latest, preview = mag.preview;
  const retain = preview?.snapped && request?.scale === box.scale && sameMagneticPreviewScope(request, preview.request);
  const exactReply = preview && request === preview.request && request.scale === box.scale;
  const path = mag.intent?.tail || (!state.space && !state.gesture && state.cursor && (retain || exactReply) ? preview.points : null);
  if (path) {
    if (!preview?.snapped && !mag.intent?.tail) ctx.setLineDash([4, 4]);
    ctx.beginPath();
    lastX = lastY = undefined;
    addPath(path);
    stroke();
    ctx.setLineDash([]);
    const [x, y] = path.at(-1);
    ctx.strokeStyle = '#13191d'; ctx.lineWidth = 2.5; ctx.beginPath();
    ctx.arc(box.x + (x + .5) * box.scale, box.y + (y + .5) * box.scale, 3, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = '#95e7c6'; ctx.lineWidth = 1; ctx.stroke();
  }
  for (const [i, segment] of mag.segments.entries()) {
    const [x, y] = segment.at(-1), px = box.x + (x + .5) * box.scale, py = box.y + (y + .5) * box.scale;
    if (mag.autoPins.has(segment)) {
      ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI * 2);
      ctx.fillStyle = '#95e7c6'; ctx.fill();
      ctx.strokeStyle = '#13191d'; ctx.lineWidth = 1; ctx.stroke();
      continue;
    }
    const radius = i === 0 ? 4 : 3;
    ctx.fillStyle = '#13191d'; ctx.fillRect(px - radius - 1, py - radius - 1, radius * 2 + 2, radius * 2 + 2);
    ctx.fillStyle = i === 0 ? '#95e7c6' : '#fff'; ctx.fillRect(px - radius, py - radius, radius * 2, radius * 2);
  }
}
function setSamPointType(negative) {
  if (!state.sam || state.sam.busy) return;
  state.sam.negative = negative;
  refresh();
  stage.focus({ preventScroll: true });
}
function addSamPoint(point) {
  const sam = state.sam;
  if (!sam || sam.busy) return;
  if (sam.points.some(item => item.point[0] === point[0] && item.point[1] === point[1])) return;
  if (sam.points.filter(item => item.negative === sam.negative).length >= 32) {
    sam.error = '每类点最多 32 个。可撤回上一点。';
  } else {
    sam.points.push({ point, negative: sam.negative });
    sam.preview = null;
    sam.error = '';
  }
  refresh();
}
function undoSamPoint() {
  const sam = state.sam;
  if (!sam || sam.busy || !sam.points.length) return;
  sam.points.pop();
  sam.preview = null;
  sam.error = '';
  refresh();
}
async function previewSamObject() {
  const sam = state.sam;
  if (!sam || sam.busy || !renderIsCurrent()) return;
  const positive = sam.points.filter(item => !item.negative).map(item => item.point);
  const negative = sam.points.filter(item => item.negative).map(item => item.point);
  if (!positive.length) return;
  const epoch = state.epoch, revision = state.doc.revision;
  const current = () => state.sam === sam && state.epoch === epoch && state.doc?.revision === revision && renderIsCurrent();
  sam.busy = 'preview'; sam.preview = null; sam.error = '';
  refresh();
  try {
    const result = await post(route('/sam-object'), { revision, positive, negative });
    if (!current() || result.revision !== revision) return;
    const image = new Image();
    image.src = result.preview;
    await image.decode();
    if (current()) sam.preview = { image, token: result.token, revision };
  } catch (error) {
    if (current()) {
      sam.error = error.status === 409 ? '选区已更新，请重新生成预览。' : error.message;
      if (error.status === 409) schedulePoll();
    }
  } finally {
    if (state.sam === sam) { sam.busy = null; refresh(); }
  }
}
async function applySamObject() {
  const sam = state.sam;
  const preview = sam?.preview;
  if (!preview || sam.busy || !renderIsCurrent() || preview.revision !== state.doc.revision) return;
  const epoch = state.epoch;
  sam.busy = 'apply'; sam.error = '';
  refresh();
  try {
    const doc = await post(route('/sam-object/apply'), { revision: preview.revision, token: preview.token });
    if (state.epoch !== epoch || doc.revision < state.doc.revision) return;
    if (state.dirty || state.save) {
      state.conflict = doc;
    } else {
      state.sam = null;
      state.tool = 'pan';
      state.view = 'parts';
      installDraft(doc);
      applyMetadata(doc);
    }
    persistPending(); schedulePoll(); refresh();
  } catch (error) {
    if (state.sam === sam && state.epoch === epoch) {
      sam.preview = null;
      sam.error = error.status === 409 ? '选区已更新，请重新生成预览。' : error.message;
      if (error.status === 409) schedulePoll();
    }
  } finally {
    if (state.sam === sam) { sam.busy = null; refresh(); }
  }
}
function startGesture(event) {
  if (!state.file || state.gesture || ![0, 1].includes(event.button)) return;
  const pos = position(event);
  if (state.tool === 'pan' || state.space || event.button === 1) {
    if (state.magnetic) state.magnetic.motion = null;
    state.gesture = { id: event.pointerId, start: pos, panX: state.panX, panY: state.panY };
  } else if (state.doc) {
    const point = sourcePoint(pos, false, state.magnetic ? Math.floor : Math.round);
    if (!point) return;
    if (state.historyLocation !== null) {
      state.historyLocation = null;
      updateRangeHistory(true);
    }
    if (state.tool === 'sam_object') {
      event.preventDefault();
      addSamPoint(point);
      return;
    }
    if (state.magnetic) {
      const mag = state.magnetic, start = mag.segments[0]?.[0];
      stage.focus({ preventScroll: true });
      mag.straight = Boolean(event.altKey);
      if (canCloseMagnetic(mag) && magneticDistance(point, start) * layout().scale <= 9)
        closeLasso(false);
      else queueMagnetic(point, 'anchor', mag.straight);
      event.preventDefault();
      return;
    }
    if (state.tool === 'lasso') {
      const first = state.polygon[0], last = state.polygon.at(-1);
      stage.focus({ preventScroll: true });
      if (state.polygon.length >= 3 && Math.hypot(point[0] - first[0], point[1] - first[1]) * layout().scale <= 9) closeLasso();
      else if ((!last || point[0] !== last[0] || point[1] !== last[1]) && state.polygon.length < 5000) state.polygon.push(point);
      event.preventDefault();
      refresh();
      return;
    }
    const edit = { tool: state.tool, shape: 'brush', points: [point], radius: Number(byId('radius').value) };
    if (state.tool === 'protect' || state.tool === 'mix') edit.strength = Number(byId('protection').value);
    if (state.tool === 'include' && byId('allow-protected').checked) edit.allow_protected = true;
    state.gesture = { id: event.pointerId, edit };
    byId('stroke-feedback').hidden = true;
    byId('unknown-stroke-actions').hidden = true;
  }
  if (!state.gesture) return;
  canvas.setPointerCapture(event.pointerId);
  event.preventDefault();
  draw();
}
function moveGesture(event) {
  const pos = position(event);
  state.cursor = sourcePoint(pos) ? pos : null;
  const gesture = state.gesture;
  if (state.magnetic && !gesture) {
    hoverMagnetic(state.cursor, Boolean(event.altKey), true);
    return;
  }
  if (gesture?.id === event.pointerId) {
    if (gesture.edit) {
      const point = sourcePoint(pos, true), points = gesture.edit.points, last = points.at(-1);
      if ((point[0] !== last[0] || point[1] !== last[1]) && points.length < 5000) points.push(point);
    } else {
      state.panX = gesture.panX + pos.x - gesture.start.x;
      state.panY = gesture.panY + pos.y - gesture.start.y;
    }
  }
  draw();
}
function endGesture(event) {
  if (state.gesture?.id !== event.pointerId) return;
  const { edit } = state.gesture;
  state.gesture = null;
  if (edit && event.type !== 'pointercancel') addEdit(edit);
  else {
    updateToolGuidance(renderIsCurrent());
    draw();
  }
}

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function download(view = 'b') {
  const target = state.doc?.[view];
  if (!target || (view === 'b' && !resultIsCurrent())) return;
  const epoch = state.epoch, name = state.file.name.replace(/\.[^.]*$/, '');
  try {
    const result = await request(route('/download?view=' + view + '&revision=' + target.revision));
    const blob = await result.blob();
    if (epoch !== state.epoch) return;
    if (view === 'b' && (!resultIsCurrent() || state.doc.b.revision !== target.revision)) {
      throw new Error('方案 B 已更新，请重新导出。');
    }
    downloadBlob(blob, name + (view === 'a' ? '-gaid-A.png' : '-gaid.png'));
    status('PNG 已发送到浏览器下载。', 'ready');
  } catch (error) { if (epoch === state.epoch) showError(error); }
}
async function pin() {
  if (!resultIsCurrent()) return;
  const epoch = state.epoch, revision = state.doc.revision;
  try {
    const doc = await post(route('/pin'), { revision });
    if (state.epoch === epoch && doc.revision >= state.doc.revision) {
      applyMetadata(doc);
      selectView('compare');
    }
  } catch (error) { if (epoch === state.epoch) showError(error); }
}
async function exportCorrection() {
  if (!renderIsCurrent()) return;
  const epoch = state.epoch;
  try {
    const info = await post(route('/corrections/export'), { revision: state.doc.revision });
    if (epoch !== state.epoch) return;
    byId('notice').dataset.tone = 'ready';
    byId('notice-text').textContent = '命令行校正已保存到 ' + info.path;
    byId('notice').hidden = false;
  } catch (error) { if (epoch === state.epoch) showError(error); }
}
async function importCorrection(files) {
  if (!state.doc || !(await flushChanges())) return;
  const epoch = state.epoch, revision = state.doc.revision;
  try {
    const selected = Array.from(files), records = selected.filter(file => /\.json$/i.test(file.name));
    if (records.length !== 1) throw new Error('请同时选择一份校正 JSON 和其中引用的 PNG 蒙版。');
    const record = JSON.parse(await records[0].text()), masks = Object.create(null);
    for (const file of selected.filter(item => /\.png$/i.test(item.name))) {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let binary = '';
      for (let offset = 0; offset < bytes.length; offset += 16384) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 16384));
      }
      masks[file.name] = btoa(binary);
    }
    if (epoch !== state.epoch) return;
    if (revision !== state.doc.revision || state.dirty || state.save) throw new Error('选区已修改，请重新导入校正。');
    const doc = await post(route('/corrections/import'), { revision, record, files: masks });
    if (epoch !== state.epoch) return;
    if (state.dirty || state.save) {
      state.conflict = doc;
      persistPending();
      refresh();
      return;
    }
    installDraft(doc);
    applyMetadata(doc);
    persistPending();
    schedulePoll();
    selectMode('range');
    selectView('weight');
  } catch (error) { if (epoch === state.epoch) showError(error); }
}

byId('file').addEventListener('change', event => {
  const file = event.target.files[0];
  event.target.value = '';
  openFile(file);
});
for (const id of ['open', 'empty-open']) byId(id).addEventListener('click', () => byId('file').click());
byId('correction-files').addEventListener('change', event => {
  const files = Array.from(event.target.files);
  event.target.value = '';
  importCorrection(files);
});
byId('correction-import').addEventListener('click', () => byId('correction-files').click());
byId('correction-export').addEventListener('click', exportCorrection);
byId('recent-panel').addEventListener('toggle', event => { if (event.target.open) loadRecent(); });
byId('undo').addEventListener('click', undo);
byId('redo').addEventListener('click', redo);
byId('download').addEventListener('click', () => state.mode === 'review' ? download() : selectMode('review'));
byId('download-a').addEventListener('click', () => download('a'));
byId('pin').addEventListener('click', pin);
byId('notice-close').addEventListener('click', () => { byId('notice').hidden = true; });
byId('retry').addEventListener('click', () => { if (state.doc) { state.dirty = true; saveDraft(); } });
byId('conflict-local').addEventListener('click', () => {
  if (!state.conflict) return;
  applyMetadata(state.conflict);
  state.conflict = null;
  edited();
});
byId('conflict-remote').addEventListener('click', () => {
  if (!state.conflict) return;
  installDraft(state.conflict);
  applyMetadata(state.conflict);
  state.conflict = null;
  state.dirty = state.saveFailed = false;
  byId('notice').hidden = true;
  persistPending(); refresh();
});
for (const button of viewButtons) button.addEventListener('click', () => selectView(button.dataset.view));
byId('show-manual-changes').addEventListener('click', () => selectView('changes'));
byId('suggested-range-view').addEventListener('click', () => selectView(suggestedRangeView()));
byId('retry-changes').addEventListener('click', () => {
  state.changesError = null;
  loadChanges();
  refresh();
});
byId('show-unknown').addEventListener('click', () => {
  if (!unknownOverlayAvailable()) return;
  state.showUnknown = !state.showUnknown;
  ensureAssets();
  refresh();
});
byId('include-unknown-stroke').addEventListener('click', () => resolveUnknownStroke());
byId('assign-unknown-stroke').addEventListener('click', () => resolveUnknownStroke(byId('unknown-stroke-module').value));
for (const button of modeButtons) button.addEventListener('click', () => selectMode(button.dataset.mode));
for (const button of scopeButtons) button.addEventListener('click', () => chooseScope(button.dataset.scope));
for (const button of toolButtons) button.addEventListener('click', () => {
  selectTool(button.dataset.tool);
  stage.focus({ preventScroll: true });
});
for (const button of presetButtons) button.addEventListener('click', () => choosePreset(button.dataset.preset));
for (const key of ['method', ...numberFields]) {
  const field = byId(key);
  field.addEventListener(field.type === 'range' ? 'input' : 'change', () => {
    if (!state.draft) return;
    if (key !== 'method' && (!field.value || !field.checkValidity())) { field.reportValidity(); return; }
    state.draft.settings[key] = key === 'method' ? field.value : Number(field.value);
    updateReadouts();
    edited();
  });
}
for (const id of ['radius', 'protection']) byId(id).addEventListener('input', () => {
  if (id === 'radius') state.brushSized = true;
  updateReadouts(); refresh();
});
byId('retention-percent').addEventListener('change', event => {
  const input = event.target;
  if (!input.value || !input.checkValidity()) { input.reportValidity(); return; }
  byId('protection').value = Number(input.value) / 100;
  updateReadouts();
  refresh();
});
byId('allow-protected').addEventListener('change', refresh);
byId('lasso-action').addEventListener('change', refresh);
byId('close-lasso').addEventListener('click', () => closeLasso());
byId('magnetic-undo').addEventListener('click', () => { resetMagnetic(true); stage.focus({ preventScroll: true }); });
byId('magnetic-radius').addEventListener('input', () => {
  hoverMagnetic(state.cursor, state.magnetic?.straight);
  refresh();
});
byId('sam-positive').addEventListener('click', () => setSamPointType(false));
byId('sam-negative').addEventListener('click', () => setSamPointType(true));
byId('sam-undo-point').addEventListener('click', undoSamPoint);
byId('sam-cancel').addEventListener('click', () => selectTool('pan'));
byId('sam-preview').addEventListener('click', previewSamObject);
byId('sam-apply').addEventListener('click', applySamObject);
byId('invert').addEventListener('click', () => { addEdit({ tool: 'invert' }); selectView('weight'); });
byId('reset').addEventListener('click', () => { addEdit({ tool: 'reset' }); selectView('weight'); });
byId('zoom-in').addEventListener('click', () => { const box = layout(); if (box) zoomTo(box.scale * 1.25); });
byId('zoom-out').addEventListener('click', () => { const box = layout(); if (box) zoomTo(box.scale / 1.25); });
byId('fit').addEventListener('click', () => { state.fit = true; state.panX = state.panY = 0; draw(); });
byId('actual').addEventListener('click', () => zoomTo(1));
const peekButton = byId('peek');
peekButton.addEventListener('pointerdown', event => { state.peek = true; peekButton.setPointerCapture(event.pointerId); draw(); });
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) peekButton.addEventListener(type, () => { state.peek = false; draw(); });
peekButton.addEventListener('click', event => { if (event.detail === 0) selectView(state.view === 'original' ? 'compare' : 'original'); });
canvas.addEventListener('wheel', event => {
  const box = layout();
  if (!box) return;
  event.preventDefault();
  const point = position(event);
  zoomTo(box.scale * (event.deltaY < 0 ? 1.15 : 1 / 1.15), point.x, point.y);
}, { passive: false });
canvas.addEventListener('pointerdown', startGesture);
canvas.addEventListener('pointermove', moveGesture);
canvas.addEventListener('pointerup', endGesture);
canvas.addEventListener('pointercancel', endGesture);
canvas.addEventListener('pointerleave', () => {
  state.cursor = null;
  if (state.magnetic) {
    state.magnetic.motion = null;
    drawMagneticFrame(state.magnetic);
  } else draw();
});
canvas.addEventListener('dblclick', () => { if (state.tool === 'lasso') closeLasso(); });
const divider = byId('divider');
let dividerPointer = null;
function moveDivider(event) {
  const box = layout();
  if (!box) return;
  state.split = clamp((position(event).x - box.x) / box.w, 0, 1);
  draw();
}
divider.addEventListener('pointerdown', event => {
  if (event.button !== 0 || dividerPointer !== null) return;
  dividerPointer = event.pointerId;
  divider.setPointerCapture(event.pointerId);
  moveDivider(event);
});
divider.addEventListener('pointermove', event => { if (event.pointerId === dividerPointer) moveDivider(event); });
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) divider.addEventListener(type, () => { dividerPointer = null; });
divider.addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  state.split = event.key === 'Home' ? 0 : event.key === 'End' ? 1 : clamp(state.split + (event.key === 'ArrowLeft' ? -.025 : .025), 0, 1);
  draw();
});
stage.addEventListener('dragover', event => {
  if (!event.dataTransfer.types.includes('Files')) return;
  event.preventDefault();
  stage.classList.add('dragging');
});
stage.addEventListener('dragleave', event => {
  if (!stage.contains(event.relatedTarget)) stage.classList.remove('dragging');
});
stage.addEventListener('drop', event => { event.preventDefault(); stage.classList.remove('dragging'); openFile(event.dataTransfer.files[0]); });
window.addEventListener('paste', event => {
  const file = event.clipboardData?.files[0];
  if (file?.type.startsWith('image/')) { event.preventDefault(); openFile(file); }
});
window.addEventListener('keydown', event => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName) || event.target.isContentEditable) return;
  const key = event.key.toLowerCase();
  if (state.mode === 'range' && (event.ctrlKey || event.metaKey) && key === 'z') {
    event.preventDefault();
    if (state.sam) { if (!event.shiftKey) undoSamPoint(); }
    else if (state.magnetic?.segments.length || state.magnetic?.intent) { if (!event.shiftKey) resetMagnetic(true); }
    else (event.shiftKey ? redo : undo)();
    return;
  }
  if (state.mode === 'range' && (event.ctrlKey || event.metaKey) && key === 'y') {
    event.preventDefault();
    if (!state.sam && !state.magnetic?.segments.length && !state.magnetic?.intent) redo();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && key === 'o') { event.preventDefault(); byId('file').click(); return; }
  if (key === 'alt' && state.magnetic) {
    event.preventDefault(); hoverMagnetic(state.cursor, true); return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey || !state.file) return;
  if (key === ' ' && event.target.tagName !== 'BUTTON') { event.preventDefault(); state.space = true; refresh(); }
  if (key === 'x') { state.peek = true; draw(); }
  if (key === 'escape') {
    if (state.sam) selectTool('pan');
    else if (state.magnetic) { event.preventDefault(); resetMagnetic(); }
    else { state.polygon = []; refresh(); }
  }
  if (key === 'backspace' && state.magnetic) { event.preventDefault(); resetMagnetic(true); }
  if (key === 'enter' && event.target.tagName !== 'BUTTON' &&
      (state.polygon.length >= 3 || canCloseMagnetic(state.magnetic, magneticClosingTail(state.magnetic)))) {
    event.preventDefault(); closeLasso();
  }
  const tool = { b: 'include', d: 'exclude', p: 'protect', m: 'mix', e: 'erase', l: 'lasso', g: 'magnetic' }[key];
  if (tool) selectTool(tool);
});
window.addEventListener('keyup', event => {
  if (event.key === ' ') { state.space = false; refresh(); }
  if (event.key === 'Alt' && state.magnetic) hoverMagnetic(state.cursor, false);
  if (event.key.toLowerCase() === 'x') { state.peek = false; draw(); }
});
window.addEventListener('blur', () => {
  state.space = state.peek = false;
  if (state.magnetic) hoverMagnetic(null, false);
  refresh();
});
window.addEventListener('pagehide', persistPending);
new ResizeObserver(draw).observe(stage);

request('/api/options').then(reply => reply.json()).then(options => {
  installOptions(options);
  loadRecent();
  const previous = storage('getItem', 'gaid-active-document');
  if (previous) openRecent(previous);
}).catch(error => {
  status('无法连接本机服务，请确认工作台已启动后刷新页面。', 'error');
  showError(error);
});

