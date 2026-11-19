'use strict';

const byId = id => document.getElementById(id);
const themeToggle = byId('theme-toggle');
const themeRoot = document.documentElement;
const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
const compactInspector = window.matchMedia('(max-width: 760px)');
const themeOptions = [...document.querySelectorAll('button[data-theme-option]')];
const themeLabels = { system: '跟随系统', light: '浅色', dark: '深色' };
const themeIcons = { system: 'system', light: 'sun', dark: 'moon' };
function applyTheme() {
  const preference = themeRoot.dataset.themePreference || 'system';
  const theme = preference === 'system' ? (systemTheme.matches ? 'dark' : 'light') : preference;
  themeRoot.dataset.theme = theme;
  byId('theme-icon').setAttribute('href', '#icon-' + themeIcons[preference]);
  themeToggle.title = '外观 · ' + themeLabels[preference];
  themeToggle.setAttribute('aria-label', '外观，' + themeLabels[preference]);
  for (const button of themeOptions) {
    button.setAttribute('aria-pressed', String(button.dataset.themeOption === preference));
  }
}
function setThemePreference(preference) {
  if (preference === 'system') delete themeRoot.dataset.themePreference;
  else themeRoot.dataset.themePreference = preference;
  applyTheme();
  try {
    if (preference === 'system') window.localStorage.removeItem('gaid-theme');
    else window.localStorage.setItem('gaid-theme', preference);
  } catch { /* The host preference still applies to this page. */ }
}
applyTheme();
systemTheme.addEventListener('change', applyTheme);
for (const button of themeOptions) button.addEventListener('click', () => {
  const preference = button.dataset.themeOption;
  setThemePreference(preference);
  preferences.update({ theme: preference }).catch(showError);
  byId('theme-panel').open = false;
  themeToggle.focus({ preventScroll: true });
});

const canvas = byId('canvas'), stage = byId('stage'), ctx = canvas.getContext('2d');
const magneticCanvas = byId('magnetic-canvas'), magneticCtx = magneticCanvas.getContext('2d');
const viewButtons = [...document.querySelectorAll('[data-view]')];
const comparisonButtons = [
  ['original', byId('compare-reference-original')], ['a', byId('compare-reference-a')],
];
const toolButtons = [...document.querySelectorAll('button[data-tool]')];
const effectActions = [
  ['soft', byId('set-strength-65')], ['strong', byId('set-l0-001')], ['standard', byId('reset-effect-defaults')],
];
const modeButtons = [...document.querySelectorAll('button[data-mode]')];
const scopeButtons = [...document.querySelectorAll('[data-scope]')];
const rangeRows = [];
const numberFields = ['strength', 'l0', 'downscale', 'sigma', 'restoration_gain', 'structure_protection'];
const MASK_VIEWS = ['weight', 'parts', 'suspected_objects'];
const SELECTION_ASSETS = [...MASK_VIEWS, 'highlight', 'background_overlay', 'unknown'];
const CHANGE_CATEGORIES = { parts: ['reclassified'], range: ['added', 'removed', 'stronger', 'weaker'] };
const RANGE_VIEWS = [...MASK_VIEWS, 'changes'];
const MAGNETIC_PIN_DISTANCE = 48;
const MODE_VIEWS = { review: ['compare', 'original', 'result'], flow: [],
  parts: ['compare', 'parts', 'changes', 'suspected_objects'],
  range: ['compare', 'weight', 'changes'], effect: ['result', 'candidate', 'compare', 'original'] };
const nodeNames = { parts: '部位分区', range: '处理范围', effect: '降噪效果' };
const actions = {
  parts: { classify: '归入目标部位', remove_part: '移出当前部位', erase: '擦除本步修改' },
  range: { include: '加入处理范围', exclude: '移出处理范围', protect: '局部减弱', mix: '指定原图比例', erase: '擦除本步修改' },
};
const toolShortcuts = {
  h: ['pan', '移动画布'], b: ['brush', '画笔（沿用当前动作）'], l: ['lasso', '套索'],
  g: ['magnetic', '磁性套索'], s: ['sam_object', 'SAM2 点选'],
};
const actionShortcuts = {
  parts: { c: 'classify', d: 'remove_part', e: 'erase' },
  range: { i: 'include', d: 'exclude', p: 'protect', m: 'mix', e: 'erase' },
};
const zoomShortcuts = { '0': 'fit', '1': 'actual', '+': 'zoom-in', '=': 'zoom-in', '-': 'zoom-out' };
const state = {
  options: null, doc: null, file: null, draft: null, dirty: false, save: null, saveFailed: false, conflict: null,
  epoch: 0, opening: false, exited: false, images: new Map(), loading: new Map(), saveTimer: 0, saveDelay: 180, pollTimer: 0, pollEtag: null,
  zoom: 1, fit: true, panX: 0, panY: 0, split: .5, view: 'compare',
  leftReference: 'a',
  mode: 'review', tool: 'pan', brushSized: false, peek: false, keyboardPeek: false, space: false, gesture: null, cursor: null, polygon: [],
  sam: null, magnetic: null, showUnknown: true, pendingUnknownResolution: null,
  changesErrors: new Map(), appliedHistory: null, historyLocation: null, historyDisplayKey: '', partFrame: null,
  displayedResult: null, part: 'hair', protectedParts: new Set(), rangeProtectedParts: new Set(),
  hiddenParts: new Set(), partIsolation: null,
  actions: { parts: 'classify', range: 'include' },
};
const isLasso = () => ['lasso', 'magnetic'].includes(state.tool);
const editingMask = () => state.mode === 'parts' || state.mode === 'range';
const isBrush = () => editingMask() && !['pan', 'sam_object', 'lasso', 'magnetic'].includes(state.tool);
const activeAction = () => isBrush() ? state.tool : byId('lasso-action').value;
const originalPreviewActive = () => state.peek || state.keyboardPeek;
const editableMaskView = () => state.mode === 'parts' ? 'parts' : 'weight';
const maskInspectionView = () => editingMask() && state.view !== editableMaskView();
const maskEditsBlocked = () => maskInspectionView() || (editingMask() && originalPreviewActive());
const clone = value => JSON.parse(JSON.stringify(value));
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const route = (suffix = '') => '/api/documents/' + encodeURIComponent(state.doc.id) + suffix;
const changesKey = stage => 'changes:' + stage;
const hiddenPartsQuery = () => [...state.hiddenParts].sort().join(',');
function viewAsset(view) {
  if (view === 'parts') {
    const next = state.images.get(state.hiddenParts.size ? 'filtered-parts' : 'parts');
    if (next?.hiddenParts === hiddenPartsQuery() && (!state.partFrame || next.revision >= state.partFrame.revision)) {
      return next;
    }
    return state.partFrame;
  }
  return state.images.get(view === 'result' ? 'b' : view === 'changes' ? changesKey(state.mode) : view);
}
const visibleImage = view => viewAsset(view)?.image;
const canChooseComparisonReference = () => state.view === 'compare' && !editingMask() && Boolean(state.doc?.a);
const usesPinnedComparison = () => state.leftReference === 'a' && canChooseComparisonReference();
function pinnedComparisonAsset() {
  const a = state.images.get('a');
  return usesPinnedComparison() && a?.revision === state.doc.a.revision ? a : null;
}
const selectionPreview = () => state.doc?.selection_preview ?? state.doc?.b;
function assetSnapshot(view) {
  if (SELECTION_ASSETS.includes(view)) return selectionPreview();
  return view === 'a' ? state.doc?.a : state.doc?.b;
}
const viewAvailable = view => (view !== 'suspected_objects' && view !== 'unknown') || Boolean(selectionPreview()?.images?.[view]);
const unknownOverlayAvailable = () => editingMask() && (state.view === 'parts' || state.view === 'weight')
  && viewAvailable('unknown');
const unknownOverlayVisible = () => unknownOverlayAvailable()
  && (state.view === 'parts' ? !state.hiddenParts.has('unknown') : state.showUnknown);
const pendingKey = id => 'gaid-pending-' + id;
const protectionKey = id => 'gaid-part-protection-' + id;
const rangeProtectionKey = id => 'gaid-range-protection-' + id;
const visibilityKey = id => 'gaid-hidden-parts-' + id;
const comparisonKey = id => 'gaid-left-reference-' + id;
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

async function request(url, options = {}, acceptNotModified = false) {
  const reply = await fetch(url, { cache: 'no-store', ...options });
  if (!reply.ok && !(acceptNotModified && reply.status === 304)) {
    const body = await reply.json().catch(() => ({}));
    const error = new Error(body.error || '本机服务请求失败（' + reply.status + '）。');
    error.status = reply.status;
    throw error;
  }
  return reply;
}
async function post(url, payload, { keepalive = false } = {}) {
  const body = JSON.stringify(payload);
  return (await request(url, { method: 'POST', headers: {
    'Content-Type': 'application/json', 'X-GAID-Request': '1',
  }, body, keepalive: keepalive && new Blob([body]).size <= 60 * 1024 })).json();
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
  return scope === 'all' ? state.options.modules.map(item => item.id)
    : [...(preferences.value?.defaults?.modules || state.options.default_modules)];
}
function renderIsCurrent() {
  return Boolean(state.doc?.b && state.doc.b.revision === state.doc.revision
    && !state.dirty && !state.save && !state.conflict && state.doc.status === 'ready');
}
function resultIsCurrent() {
  return renderIsCurrent() && state.displayedResult === state.doc.b.revision;
}
function comparing() {
  const reference = pinnedComparisonAsset()?.image || visibleImage('original');
  return state.view === 'compare' && !originalPreviewActive() && Boolean(visibleImage('b') && reference);
}
function skippedUnknownCount(current) {
  if (!current || state.mode !== 'range' || state.tool !== 'include' || state.gesture?.edit) return 0;
  const last = state.draft.edits[state.draft.cursor - 1], feedback = state.doc.edit_feedback;
  const count = feedback?.skipped_unknown_pixels;
  return last?.tool === 'include' && (last.shape ?? 'brush') === 'brush' && !last.allow_protected
    && feedback?.tool === 'include' && Number.isInteger(count) && count > 0 ? count : 0;
}
function updateToolGuidance(current) {
  const tool = state.tool === 'pan' ? 'pan' : activeAction();
  const region = isLasso() ? '圈住的区域' : '画过的区域';
  const descriptions = {
    pan: state.mode === 'range' ? '' : '拖动画布查看细节，选择左侧工具开始修改。',
    classify: '把' + region + '中可编辑的像素归入目标部位。',
    remove_part: '把' + region + '的' + partLabel(state.part) + '像素移到待确认。',
    include: byId('allow-protected').checked ? '补入' + region + '，含自动保留区。' : '把未选区域加入处理范围。',
    exclude: '把' + region + '移出处理范围，保留原像素。',
    protect: '减弱 100% 时，' + (isLasso() ? '圈内' : '笔心') + '保留原图。',
    mix: (isLasso() ? '圈内的已选区域' : '笔刷中心的已选区域') + '混合 ' + Number(byId('retention-percent').value)
      + '% 原图。',
    erase: state.mode === 'parts' ? '擦掉这一步的分区修正，恢复原有归属。' : '擦掉本步手动标记。想保留原像素可用「移出」。',
  };
  const description = byId('tool-description');
  const help = descriptions[tool] || '';
  description.hidden = !help;
  if (description.textContent !== help) description.textContent = help;

  const output = byId('stroke-feedback');
  const feedback = state.doc?.edit_feedback;
  const latest = state.doc?.edits?.[state.doc.cursor - 1];
  const editedPart = partLabel(latest?.module || state.part);
  const feedbackTool = tool === 'pan' && latest?.shape === 'sam' && latest.stage === state.mode ? latest.tool : tool;
  let message = '';
  if (current && !state.gesture?.edit && feedback?.tool === feedbackTool && (!latest?.stage || latest.stage === state.mode)) {
    if (feedback.reason === 'changed' && Number.isInteger(feedback.changed_pixels) && feedback.changed_pixels > 0) {
      const count = feedback.changed_pixels.toLocaleString('zh-CN');
      const pixels = count + ' 个像素';
      message = {
        include: '已补选 ' + pixels + '。',
        erase: '已擦掉 ' + pixels + '的手动标记。',
        classify: '已将 ' + pixels + '归入' + editedPart + '。',
        remove_part: '已将 ' + pixels + '从' + editedPart + '移到待确认。',
        protect: '已减弱 ' + pixels + '的处理。',
        mix: '已更新 ' + pixels + '的原图比例。',
      }[feedbackTool] || '';
      if (feedbackTool === 'exclude') {
        if (latest?.feather > 0 && feedback.partially_reduced_pixels === undefined) {
          message = '已调整 ' + pixels + '的处理范围。';
        } else {
          const partial = feedback.partially_reduced_pixels ?? 0;
          const removed = feedback.changed_pixels - partial;
          if (removed) message = '已移出 ' + removed.toLocaleString('zh-CN') + ' 个像素，保留原图。';
          if (partial) message += '羽化减弱 ' + partial.toLocaleString('zh-CN') + ' 个像素的处理。';
        }
      }
    } else if (feedback.reason === 'already_selected') message = '未新增：涂到的可编辑区域已选中。';
    else if (feedback.reason === 'protected' && feedbackTool !== 'classify') message = '未新增：笔画落在受保护区域。';
    else if (feedback.reason === 'no_marks') message = {
      classify: '这笔未改变分区。', remove_part: '本笔移出 0 个' + editedPart + '像素。',
    }[feedbackTool] || '这里没有手动标记，自动选区照常生效。';
    else if (feedback.reason === 'already_excluded') message = '这里原本就没有选中，不会处理。';
    const protectedCount = feedback.skipped_protected_pixels;
    if (protectedCount > 0) {
      const count = protectedCount.toLocaleString('zh-CN');
      if (latest?.stage === 'range') {
        message = (feedback.changed_pixels > 0 ? message : '') + '已跳过 ' + count + ' 个锁定像素。';
      } else message += '已跳过 ' + count + ' 个受保护像素。';
    }
  }
  const skipped = skippedUnknownCount(current);
  byId('unknown-stroke-actions').hidden = !skipped;
  const unknownLocked = state.mode === 'range' && state.rangeProtectedParts.has('unknown');
  byId('include-unknown-stroke').disabled = byId('assign-unknown-stroke').disabled = unknownLocked;
  byId('unknown-assignment').hidden = latest?.stage === 'range';
  byId('open-unknown-parts').hidden = latest?.stage !== 'range';
  if (skipped) {
    const skippedMessage = '这笔跳过 ' + skipped.toLocaleString('zh-CN') + ' 个待确认像素。';
    message = feedback.changed_pixels > 0 || feedback.skipped_protected_pixels > 0 ? message + skippedMessage : skippedMessage;
  }
  output.hidden = !message;
  if (output.textContent !== message) output.textContent = message;
}
function rangeToolName(tool) {
  return { pan: '移动画布', include: '画入处理范围', exclude: '移出处理范围', protect: '局部减弱',
    mix: '指定原图比例', erase: '擦除手动改动', override: '纠正部位', sam_object: '点选外物',
    classify: '归入目标部位', remove_part: '移出当前部位',
    invert: '反选处理范围', reset: '按部位恢复', import: '导入校正' }[tool] || '编辑范围';
}
function editHistoryLabel(edit) {
  const module = partLabel(edit.module);
  if (edit.tool === 'resolve_unknown') return edit.module
    ? '待确认归为' + module
    : '加入本笔的待确认';
  let label = ['override', 'classify'].includes(edit.tool) && module ? '归为' + module : rangeToolName(edit.tool);
  if (edit.tool === 'remove_part') label = '从' + module + '移到待确认';
  if (['protect', 'mix'].includes(edit.tool) && Number.isFinite(edit.strength)) {
    label += ' ' + (edit.strength * 100).toLocaleString('zh-CN', { maximumFractionDigits: 1 }) + '%';
  }
  return label + (edit.shape === 'polygon' ? ' · 圈选' : edit.shape === 'sam' ? ' · SAM2' : '');
}
function editBelongsToStage(edit, stage) {
  if (edit.stage) return edit.stage === stage;
  if (edit.tool === 'import' || edit.tool === 'erase') return true;
  const parts = edit.tool === 'override' || edit.tool === 'sam_object'
    || (edit.tool === 'resolve_unknown' && edit.module);
  return stage === (parts ? 'parts' : 'range');
}
function updateEditHistory(canEdit) {
  if (!editingMask()) return;
  const stage = state.mode, parts = stage === 'parts';
  const subject = parts ? '部位归属' : '处理权重';
  const history = state.draft?.edits || [], cursor = state.draft?.cursor || 0;
  const applied = state.appliedHistory, rows = [];
  for (let index = cursor - 1; index >= 0 && rows.length < 4; index--) {
    const edit = history[index];
    if (!editBelongsToStage(edit, stage)) continue;
    const included = index < (applied?.cursor ?? 0) && JSON.stringify(applied.edits[index]) === JSON.stringify(edit);
    const points = edit.tool === 'resolve_unknown' ? history[index - 1]?.points : edit.points;
    rows.push({ index, label: editHistoryLabel(edit), included, locatable: Boolean(points?.length),
      focused: state.historyLocation === index });
  }
  const audit = viewAsset('changes'), inAudit = state.view === 'changes';
  const current = renderIsCurrent(), now = state.doc?.revision;
  const auditCurrent = Boolean(current && audit?.revision === now);
  const error = current && state.changesErrors.get(stage) === now;
  const counts = audit?.counts, categories = CHANGE_CATEGORIES[stage];
  const showCounts = Boolean(counts && categories.some(key => counts[key] > 0));
  byId('change-history-title').textContent = parts ? '分区修改记录' : '范围修改记录';
  const button = byId('show-manual-changes');
  byId('change-history-toolbar').hidden = !parts && inAudit;
  byId('change-history-audit-heading').hidden = !parts;
  button.hidden = !parts && inAudit;
  button.disabled = !canEdit || inAudit;
  button.textContent = inAudit ? '正在查看' : parts ? '查看手动改动' : '查看范围改动';
  byId('changes-details').hidden = !inAudit;
  byId('changes-counts').hidden = !showCounts;
  byId('retry-changes').hidden = !inAudit || !error;
  if (inAudit) {
    if (parts) updatePartChangesLegend(audit?.parts ?? []);
    else for (const key of categories) byId('range-legend-' + key).hidden = !counts?.[key];
    byId('changes-baseline').textContent = parts ? '与自动分区对比，颜色表示改动后的部位。' : '与按当前分区和设置生成的自动范围对比。';
    byId('changes-status').textContent = auditCurrent
      ? showCounts ? '显示' + subject + '的净改动。'
        : rows.length ? '手动操作已保存，' + subject + '没有净变化。' : '当前无手动改动。'
      : error ? (audit ? '读取当前改动失败，仍显示上次的成品 B。' : '读取手动改动失败。')
        : !current ? (audit ? '显示上次成品 B，新改动待更新。' : '等成品 B 更新后即可核对。')
          : (audit ? '正在读取当前改动，仍显示上次的成品 B。' : '正在读取手动改动…');
    if (showCounts) for (const key of [...CHANGE_CATEGORIES.parts, ...CHANGE_CATEGORIES.range]) {
      const count = categories.includes(key) ? counts[key] : 0;
      byId('change-count-' + key).hidden = count === 0;
      byId('changes-' + key).textContent = count.toLocaleString('zh-CN') + ' px';
    }
  }
  byId('change-history-empty').hidden = Boolean(rows.length);
  byId('change-history-note').hidden = !parts || !rows.length;
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
  const rangeMode = editingMask();
  const editing = rangeMode && !maskEditsBlocked() ? '，可继续编辑' : '';
  const selectionView = state.view === 'parts' || state.view === 'weight';
  const layer = selectionView ? viewAsset(state.view) : null;
  const updatedSelection = selectionView && !originalPreviewActive() && layer?.revision === state.doc.revision
    && (!layer.overlay || visibleImage('original'))
    && (state.view !== 'parts' || layer.hiddenParts === hiddenPartsQuery());
  const phases = { queued: '等待处理', loading: '正在读取图片', analyzing: '正在识别分区', analysis: '正在识别分区',
    candidate: '正在生成降噪候选', rendering: '正在合成预览', compose: '正在合成预览',
    encoding: '正在保存结果', saving: '正在保存结果' };
  if (state.conflict) status('另一页面已更新，请选择保留哪份修改。', 'error');
  else if (state.save) status('正在提交修改' + editing, 'pending');
  else if (state.saveFailed) status('修改保存失败，可重试。', 'error');
  else if (state.dirty) status('修改待更新' + editing, 'pending');
  else if (state.doc.status === 'error') status(nodeNames[failedNode()] + '失败。' + (state.doc.error || '可重试此步。'), 'error');
  else if (state.doc.status !== 'ready' && updatedSelection) {
    status((state.view === 'parts' ? '分区已更新' : '范围已更新') + ' · 正在生成成品 B' + editing, 'pending');
  } else if (state.doc.status !== 'ready') status((state.doc.b ? '正在更新 · ' : '') + (phases[state.doc.phase] || phases[state.doc.status] || '正在处理') + editing, 'pending');
  else if (state.mode === 'flow') status('当前流程已完成', 'ready');
  else if (state.view === 'candidate') status(state.images.get('candidate')?.revision === state.doc.revision ? '全图降噪预览已就绪' : '正在载入全图降噪预览', state.images.get('candidate')?.revision === state.doc.revision ? 'ready' : 'pending');
  else if (rangeMode && RANGE_VIEWS.includes(state.view)) {
    const revision = state.doc.revision;
    if (state.view === 'changes' && state.changesErrors.get(state.mode) === revision) status('改动图读取失败，可在「核对改动」重试。', 'error');
    else if (state.view === 'parts' && viewAsset('parts')?.hiddenParts !== hiddenPartsQuery()) {
      status('正在更新分区显示' + editing, 'pending');
    } else if (viewAsset(state.view)?.revision === revision) {
      const label = { changes: '改动图已更新', parts: '分区已更新', suspected_objects: '疑似外物已显示' };
      status(label[state.view] || '范围已更新', 'ready');
    } else status('正在显示当前图层' + editing, 'pending');
  } else if (current) status('当前成品已就绪', 'ready');
  else if (state.view === 'original' && visibleImage('original')) status('原图已显示，可切换到「成品」检查。', 'neutral');
  else status('正在显示当前成品' + editing, 'pending');
}
function updateSamControls() {
  const sam = state.sam;
  const positive = sam?.points.filter(item => !item.negative).length ?? 0;
  const canPreview = Boolean(sam && !sam.busy && positive && !maskEditsBlocked() && renderIsCurrent());
  const action = state.mode === 'parts' ? activeAction() : '';
  const classifying = action === 'classify', removingPart = action === 'remove_part';
  const part = partLabel(state.part);
  const hasLocks = state.mode === 'range' ? state.rangeProtectedParts.size : classifying && state.protectedParts.size;
  const previewHint = removingPart ? '确认后仅移出轮廓内的' + part + '像素。'
    : hasLocks ? '核对高亮的轮廓。确认时会跳过受保护的分区。'
      : '核对高亮的轮廓，确认后应用到当前步骤。';
  const text = !sam ? ''
    : sam.busy === 'preview' ? '正在生成预览，可继续缩放或移动画布。'
    : sam.busy === 'apply' ? '正在应用选区…'
    : sam.preview ? previewHint
    : sam.error || (!positive ? '点一下想选中的区域。'
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
  byId('sam-apply').textContent = removingPart ? '移出' + part : classifying ? '归为' + part : '应用选区';
  if (byId('sam-status').textContent !== text) byId('sam-status').textContent = text;
}
function updateControls() {
  const canEdit = Boolean(state.doc && state.draft && !state.opening && state.sam?.busy !== 'apply');
  const inspecting = maskInspectionView(), blocked = maskEditsBlocked();
  const canDraw = canEdit && !blocked;
  const hasImage = Boolean(state.file), current = resultIsCurrent();
  const method = state.options?.methods.find(item => item.id === state.draft?.settings.method);
  byId('open').disabled = byId('empty-open').disabled = byId('open-settings').disabled = !state.options;
  const rangeMode = editingMask();
  const graph = state.mode === 'flow';
  const reviewMode = state.mode === 'review';
  const nodeName = nodeNames[state.mode] || '';
  document.body.dataset.mode = state.mode;
  byId('viewbar').hidden = byId('viewer-footer').hidden = byId('zoom-tools').hidden = !hasImage || graph;
  byId('file-meta').hidden = !hasImage;
  byId('flow-board').hidden = !graph;
  stage.hidden = graph;
  byId('tools').hidden = !rangeMode;
  byId('edit-bar').hidden = !rangeMode;
  const nodeTitle = byId('node-title');
  byId('back-to-flow').hidden = nodeTitle.hidden = !nodeName;
  byId('back-to-flow').disabled = !canEdit;
  if (nodeTitle.textContent !== nodeName) nodeTitle.textContent = nodeName;
  byId('drawer-title').textContent = nodeName || '成品检查';
  byId('inspector-toggle').hidden = !state.doc || graph;
  if (graph) updateFlow();
  const viewHint = comparing() ? '拖动分割线比较 · 空格拖动画布' : '滚轮缩放 · 空格拖动画布';
  const hint = byId('viewer-hint');
  hint.hidden = !hasImage || graph;
  if (hint.textContent !== viewHint) hint.textContent = viewHint;
  byId('inspector').hidden = !state.doc || state.opening || graph;
  byId('empty-title').textContent = state.exited ? '工作台已退出' : state.opening ? '正在打开图片…' : '拖入图片开始精修';
  byId('empty-help').hidden = state.opening;
  byId('empty-open').hidden = state.opening || state.exited;
  byId('recent-panel').hidden = state.exited;
  const drawingMagnetic = Boolean(state.magnetic?.segments.length || state.magnetic?.intent);
  byId('undo').disabled = !canDraw || !rangeMode || Boolean(state.sam) || drawingMagnetic || state.draft.cursor === 0;
  byId('redo').disabled = !canDraw || !rangeMode || Boolean(state.sam) || drawingMagnetic || state.draft.cursor === state.draft.edits.length;
  byId('download').disabled = !current;
  byId('download').title = current ? '导出当前成品 B（Ctrl+S）' : '当前成品显示后可导出（Ctrl+S）';
  byId('pin').disabled = !current;
  const a = state.doc?.a, showCorrections = rangeMode && Boolean(state.doc);
  const exportPanel = byId('export-panel');
  exportPanel.hidden = !a && !showCorrections;
  if (exportPanel.hidden) exportPanel.open = false;
  byId('correction-export').hidden = byId('correction-import').hidden = !showCorrections;
  byId('download-a').hidden = byId('download-a').disabled = !a;
  byId('correction-export').disabled = !renderIsCurrent();
  byId('correction-import').disabled = !canDraw || state.dirty || Boolean(state.save || state.conflict);
  byId('export-hint').textContent = reviewMode && !current ? '当前成品显示后，即可固定或导出。' : '';
  const b = state.images.get('b')?.meta;
  const waitingForA = usesPinnedComparison() && !pinnedComparisonAsset();
  byId('comparison-a').textContent = !a ? '原图'
    : waitingForA ? '正在载入固定对照' : '固定对照 · ' + methodLabel(a.method);
  byId('compare-reference').hidden = !canChooseComparisonReference();
  for (const [reference, button] of comparisonButtons) {
    button.setAttribute('aria-pressed', String(reference === state.leftReference));
  }
  byId('comparison-b').textContent = !b ? '等待第一份成品' : methodLabel(b.method)
    + (current ? ' · 当前成品' : b.revision === state.doc?.revision && renderIsCurrent() ? ' · 待查看' : ' · 上一版');
  for (const id of ['zoom-out', 'zoom-in', 'fit', 'actual', 'peek']) byId(id).disabled = !hasImage;
  for (const [buttons, key, value, disabled] of [
    [viewButtons, 'view', state.view, !hasImage],
    [toolButtons, 'tool', blocked ? 'pan' : isBrush() ? 'brush' : state.tool, !canDraw],
    [modeButtons, 'mode', state.mode, !canEdit],
  ]) for (const button of buttons) {
    if (key === 'view') {
      button.hidden = !MODE_VIEWS[state.mode].includes(button.dataset.view) || !viewAvailable(button.dataset.view);
      if (button.dataset.view === 'compare') {
        const label = rangeMode ? '原图 / 成品' : '对比';
        if (button.textContent !== label) button.textContent = label;
      } else if (button.dataset.view === 'changes') {
        const label = state.mode === 'parts' ? '分区改动' : '范围改动';
        if (button.textContent !== label) button.textContent = label;
      }
    } else if (key === 'mode' && button.dataset.mode === 'flow') {
      button.hidden = Boolean(nodeName);
    }
    button.disabled = key === 'tool' ? !rangeMode || (button.dataset.tool === 'pan' ? !hasImage : disabled)
      : disabled || (button.dataset.view === 'result' && !state.doc?.b);
    button.setAttribute('aria-pressed', String(button.dataset[key] === value));
  }
  const unknownToggle = byId('show-unknown');
  unknownToggle.hidden = state.view !== 'weight' || !unknownOverlayAvailable();
  unknownToggle.setAttribute('aria-pressed', String(state.showUnknown));
  byId('edit-inspector').hidden = !rangeMode || inspecting;
  byId('range-history').hidden = !rangeMode;
  for (const mode of ['review', 'parts', 'range', 'effect']) byId(mode + '-inspector').hidden = state.mode !== mode || inspecting;
  for (const [name, button] of effectActions) {
    button.disabled = !canEdit || (name === 'strong' && !method?.uses_l0) || Boolean(presetMatches(name));
  }
  for (const button of scopeButtons) {
    button.disabled = !canDraw;
    const selected = state.draft?.settings.modules || [];
    const expected = state.options ? scopeModules(button.dataset.scope) : [];
    button.setAttribute('aria-pressed', String(canEdit && expected.length === selected.length
      && expected.every(name => selected.includes(name))));
  }
  byId('range-protection-all').disabled = !canDraw;
  for (const { check, strength, lock } of rangeRows) {
    lock.disabled = !canDraw;
    if (check) check.disabled = strength.disabled = !canDraw;
  }
  for (const id of ['method', ...numberFields, 'radius', 'feather', 'protection', 'retention-percent', 'allow-protected', 'lasso-action', 'invert', 'reset']) {
    byId(id).disabled = !canDraw || (id === 'l0' && !method?.uses_l0) || (id === 'downscale' && !method?.uses_downscale);
  }
  byId('lasso-options').hidden = !isLasso() || inspecting;
  byId('brush-radius-row').hidden = !isBrush() || inspecting;
  byId('feather-row').hidden = !isBrush() || inspecting || state.mode !== 'range';
  const operation = inspecting || state.tool === 'pan' ? 'pan' : activeAction();
  byId('action-field').hidden = operation === 'pan';
  byId('pan-hint').hidden = operation !== 'pan';
  const editHint = state.mode === 'parts' ? '在「分区叠加」中修改部位。' : '在「实际处理」中修改范围。';
  byId('pan-hint').textContent = inspecting ? editHint : '拖动画布，滚轮缩放';
  byId('protect-row').hidden = !['protect', 'mix'].includes(operation);
  const target = byId('target-shortcut');
  const removingPart = operation === 'remove_part';
  target.hidden = state.mode !== 'parts' || !['classify', 'remove_part'].includes(operation);
  target.style.setProperty('--part-color', partColor(state.part));
  if (state.mode === 'parts') {
    const selectLabel = removingPart ? '选择要移出的部位' : '选择目标部位';
    target.title = selectLabel;
    target.setAttribute('aria-label', selectLabel + '，当前' + partLabel(state.part));
    byId('part-heading').textContent = removingPart ? '要移出的部位' : '目标部位';
    byId('part-hint').textContent = removingPart ? '点选来源部位。移出的像素归入待确认，Ctrl+点击独显。'
      : '点部位选目标，Ctrl+点击独显。眼睛控制叠加，锁控制保护。';
    const removeOption = [...byId('lasso-action').children].find(option => option.value === 'remove_part');
    if (removeOption) removeOption.disabled = state.part === 'unknown';
  }
  byId('protection-label').textContent = operation === 'mix' ? '原图比例' : '减弱比例';
  byId('allow-protected-row').hidden = state.mode !== 'range' || operation !== 'include';
  byId('l0-field').hidden = !method?.uses_l0;
  byId('downscale-field').hidden = !method?.uses_downscale;
  const mag = state.magnetic;
  byId('current-tool-name').textContent = state.tool === 'pan' ? '移动画布'
    : state.sam ? 'SAM2 点选' : mag ? '磁性套索' : isLasso() ? '套索' : '画笔';
  if (mag) updateMagneticCloseButton(mag, canDraw);
  else byId('close-lasso').disabled = !canDraw || state.polygon.length < 3;
  byId('magnetic-options').hidden = byId('magnetic-undo').hidden = !mag;
  byId('magnetic-undo').disabled = !canEdit || (!mag?.segments.length && mag?.intent?.kind !== 'anchor');
  byId('magnetic-radius').disabled = !canEdit || Boolean(mag?.intent);
  if (mag) {
    updateMagneticStatus(mag);
    byId('magnetic-radius-value').textContent = byId('magnetic-radius').value + ' 屏幕 px';
  }
  byId('tool-hint').hidden = state.mode === 'range' && !isLasso();
  byId('tool-hint').textContent = mag ? (mag.segments.length ? mag.segments.length + ' 个锚点 · Enter 闭合' : '自动贴边')
    : state.tool === 'lasso'
    ? (state.polygon.length ? state.polygon.length + ' 个顶点，Enter 闭合' : '单击添加顶点') : '空格临时移动';
  updateToolGuidance(renderIsCurrent());
  byId('brush-panel').hidden = Boolean(state.sam)
    || (state.mode === 'range' && state.tool === 'pan' && byId('stroke-feedback').hidden);
  updateSamControls();
  updateEditHistory(canEdit);
  stage.dataset.tool = state.space || blocked ? 'pan' : state.tool;
  byId('conflict-local').hidden = byId('conflict-remote').hidden = !state.conflict;
  byId('retry').hidden = !state.doc || Boolean(state.conflict || state.save)
    || (!state.saveFailed && (state.doc.status !== 'error' || state.dirty));
  const failed = state.doc?.status === 'error' && !state.dirty && !state.conflict;
  byId('error-node').hidden = !failed;
  if (failed) byId('error-node').textContent = '打开' + nodeNames[failedNode()];
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
function paintOverlay(image, box) {
  ctx.save();
  ctx.globalCompositeOperation = 'source-atop';
  paintImage(image, box);
  ctx.restore();
}
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
  const colors = { classify: partColor(edit.module), remove_part: partColor('unknown'), include: '#75badd',
    exclude: '#f3c87a', protect: '#ec879b', mix: '#62c8aa', erase: '#d5ded8', override: '#add4b7' };
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
  if (state.historyLocation === null || !editingMask() || originalPreviewActive() || state.sam) return;
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
  if (state.mode === 'flow') return;
  const box = layout(), showDivider = comparing();
  const peek = originalPreviewActive();
  byId('history-location-label').hidden = !box || state.historyLocation === null || !editingMask() || peek || Boolean(state.sam);
  const maskView = MASK_VIEWS.includes(state.view), suspectedView = state.view === 'suspected_objects';
  const diffView = state.view === 'changes';
  const unknown = state.images.get('unknown'), base = viewAsset(state.view);
  const allPartsHidden = state.view === 'parts' && state.hiddenParts.size === state.options?.modules.length + 2;
  byId('overlay-legend').hidden = !box || peek || !maskView || suspectedView || allPartsHidden
    || (state.view === 'weight' && !base?.image);
  byId('objects-legend').hidden = state.view !== 'parts' || state.hiddenParts.has('object');
  byId('protection-legend').hidden = state.view !== 'parts';
  byId('weight-legend').hidden = state.view !== 'weight';
  if (state.view === 'weight') {
    byId('weight-ramp').hidden = !base?.overlay;
    byId('weight-legacy-legend').hidden = Boolean(base?.overlay);
  }
  byId('suspected-objects-legend').hidden = peek || !suspectedView || !visibleImage('suspected_objects');
  byId('part-changes-legend').hidden = !box || peek || !diffView || state.mode !== 'parts' || !base?.parts?.length;
  byId('range-changes-legend').hidden = !box || peek || !diffView || state.mode !== 'range' || !base?.image;
  const showUnknown = Boolean(box && !peek && unknownOverlayVisible()
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
  const showsResult = !peek && b && (state.view === 'result' || showDivider)
    && Math.max(box.x, showDivider ? cut : 0) < Math.min(box.x + box.w, box.width)
    && Math.max(box.y, 0) < Math.min(box.y + box.h, box.height);
  const firstResultFrame = showsResult && state.displayedResult !== b.revision;
  if (firstResultFrame) state.displayedResult = b.revision;
  if (showDivider) {
    const a = pinnedComparisonAsset();
    paintImage(a?.image || original, box);
    ctx.save();
    ctx.beginPath(); ctx.rect(cut, 0, box.width - cut, box.height); ctx.clip();
    // Clear the right half so transparent pixels do not composite over the left reference.
    ctx.clearRect(cut, 0, box.width - cut, box.height);
    paintImage(visibleImage('b'), box);
    ctx.restore();
    const aMeta = a?.meta, bMeta = state.images.get('b')?.meta;
    byId('left-badge').textContent = aMeta ? 'A  ' + methodLabel(aMeta.method) + ' / 选区 ' + (aMeta.selection_revision ?? aMeta.revision) : '原图';
    byId('right-badge').textContent = 'B  ' + methodLabel(bMeta?.method)
      + (bMeta ? ' / 选区 ' + (bMeta.selection_revision ?? bMeta.revision) : '')
      + (resultIsCurrent() && bMeta?.revision === state.doc.revision ? '' : ' · 待更新');
  } else if (peek || state.view === 'original' || allPartsHidden) paintImage(original, box);
  else if (diffView) { paintImage(original, box); paintImage(visibleImage('changes'), box); }
  else if (base?.overlay) { paintImage(original, box); paintOverlay(base.image, box); }
  else paintImage(base?.image || original, box);
  if (!peek && state.mode === 'parts' && state.view === 'parts') {
    if (base) state.partFrame = base;
    const background = state.images.get('background_overlay');
    if (!base?.overlay && background && background.revision === base?.revision && !state.hiddenParts.has('background')) {
      paintOverlay(background.image, box);
    }
    const highlight = state.images.get('highlight');
    if (!state.hiddenParts.has(state.part) && highlight?.part === state.part && highlight.revision === base?.revision) {
      const paintHighlight = state.part === 'background' ? paintOverlay : paintImage;
      paintHighlight(highlight.image, box);
    }
  }
  if (showUnknown) paintImage(unknown.image, box);

  if (!peek && state.sam) {
    ctx.save(); drawSam(box); ctx.restore();
  } else if (!peek && !maskInspectionView() && (maskView || state.tool !== 'pan')) {
    ctx.save();
    ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
    // The brush gesture is drawn on the current frame, independently of backend jobs.
    const overlay = maskView ? base : state.images.get('b');
    if (!diffView && state.draft && (state.dirty || state.save || overlay?.revision !== state.doc?.revision)) {
      const publishedCursor = state.view === 'parts' ? state.appliedHistory?.cursor : 0;
      const start = Math.max(overlay?.cursor ?? publishedCursor ?? 0, state.draft.cursor - 32);
      for (const edit of state.draft.edits.slice(start, state.draft.cursor)) {
        const hiddenPart = edit.tool === 'remove_part' ? 'unknown' : edit.module;
        if (state.view === 'parts' && state.hiddenParts.has(hiddenPart)) continue;
        if (!edit.stage || edit.stage === state.mode) strokePath(edit, box);
      }
    }
    strokePath(state.gesture?.edit, box);
    if (state.polygon.length) strokePath({ tool: byId('lasso-action').value, shape: 'polygon', points: state.polygon }, box, true);
    if (state.cursor && !state.space && isBrush()) {
      const radius = Number(byId('radius').value);
      const feather = state.mode === 'range' ? Math.min(Number(byId('feather').value), radius) : 0;
      const ring = size => {
        ctx.beginPath();
        ctx.arc(state.cursor.x, state.cursor.y, Math.max(1, size * box.scale), 0, Math.PI * 2);
        ctx.strokeStyle = '#18232c'; ctx.lineWidth = 2.7; ctx.stroke();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.1; ctx.stroke();
      };
      ring(radius);
      if (feather > 0 && feather * box.scale >= 1) {
        ctx.setLineDash([3, 3]);
        ring(radius - feather);
      }
    }
    ctx.restore();
  }
  drawHistoryLocation(box);
  const zoom = Math.round(box.scale * 100) + '%';
  if (byId('zoom-level').textContent !== zoom) byId('zoom-level').textContent = zoom;
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
function position(event, rect = canvas.getBoundingClientRect()) {
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}
function sourcePoint(point, allowOutside = false, toPixel = Math.round, box = layout()) {
  if (!box) return null;
  const x = (point.x - box.x) / box.scale, y = (point.y - box.y) / box.scale;
  if (!allowOutside && (x < 0 || y < 0 || x >= state.file.width || y >= state.file.height)) return null;
  return [clamp(toPixel(x), 0, state.file.width - 1), clamp(toPixel(y), 0, state.file.height - 1)];
}
const brushCoordinate = value => Math.round(value * 64) / 64;
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
  const part = view === 'highlight' ? state.part : undefined;
  const hiddenParts = view === 'parts' ? hiddenPartsQuery() : '';
  const snapshot = assetSnapshot(view);
  const overlay = (view === 'parts' || view === 'weight') && Boolean(snapshot?.images?.[view + '_overlay']);
  const resource = overlay ? view + '_overlay' : view;
  const asset = hiddenParts ? 'filtered-parts' : view;
  const cached = state.images.get(asset);
  if (cached?.revision === revision && cached?.part === part && (cached?.hiddenParts || '') === hiddenParts) return;
  const key = asset + ':' + revision + (part ? ':' + part : '') + (hiddenParts ? ':' + hiddenParts : '');
  if (state.loading.has(key)) return;
  const epoch = state.epoch;
  const cursor = snapshot?.cursor ?? (state.doc.revision === revision ? state.doc.cursor : 0);
  const query = (typeof revision === 'number' ? '&revision=' + revision : '') + (part ? '&module=' + encodeURIComponent(part) : '')
    + (hiddenParts ? '&hide=' + encodeURIComponent(hiddenParts) : '');
  const wanted = () => epoch === state.epoch && (view === 'original' || revision === assetSnapshot(view)?.revision)
    && (!hiddenParts || hiddenParts === hiddenPartsQuery()) && (!part || part === state.part);
  const promise = request(route('/image?view=' + resource + query))
    .then(reply => reply.blob()).then(decodeImage).then(image => {
      if (!wanted() || !viewAvailable(view)) return;
      state.images.set(asset, { image, revision, meta, cursor, part, hiddenParts, overlay });
      if (view === 'original') prepareMagnetic();
      refresh();
    }).catch(error => {
      if (wanted() && ![404, 409].includes(error.status)) showError(error);
    }).finally(() => {
      if (state.loading.get(key) === promise) state.loading.delete(key);
    });
  state.loading.set(key, promise);
}
function loadChanges() {
  if (state.view !== 'changes' || !renderIsCurrent()) return;
  const stage = state.mode, revision = state.doc.revision, epoch = state.epoch;
  const asset = changesKey(stage), key = asset + ':' + revision, categories = CHANGE_CATEGORIES[stage];
  if (state.images.get(asset)?.revision === revision || state.loading.has(key)
      || state.changesErrors.get(stage) === revision) return;
  const current = () => epoch === state.epoch && state.doc?.revision === revision && renderIsCurrent();
  const visible = () => state.mode === stage && state.view === 'changes';
  const promise = request(route('/changes?revision=' + revision + '&stage=' + stage)).then(reply => reply.json()).then(async value => {
    if (!current()) return;
    const { counts, overlay } = value;
    const parts = stage === 'parts' ? (value.parts ?? []) : [];
    if (value.stage !== stage || value.revision !== revision || !counts
        || categories.some(name => !Number.isSafeInteger(counts[name]) || counts[name] < 0)
        || !Array.isArray(parts) || parts.some(part => !partCategories().includes(part))
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
    state.images.set(asset, { image, counts, revision, parts });
    if (visible()) refresh();
  }).catch(() => {
    if (current()) {
      state.changesErrors.set(stage, revision);
      if (visible()) refresh();
    }
  }).finally(() => {
    if (state.loading.get(key) === promise) state.loading.delete(key);
  });
  state.loading.set(key, promise);
}
function ensureAssets() {
  if (!state.doc || state.mode === 'flow') return;
  loadAsset('original', 'normalized');
  if (state.doc.b && (state.mode === 'review' || ['compare', 'result'].includes(state.view))) {
    loadAsset('b', state.doc.b.revision, state.doc.b);
  }
  if (usesPinnedComparison()) loadAsset('a', state.doc.a.revision, state.doc.a);
  else if (!state.doc.a) state.images.delete('a');
  const preview = selectionPreview();
  if (preview && MASK_VIEWS.includes(state.view) && viewAvailable(state.view)) loadAsset(state.view, preview.revision);
  if (state.doc.b && state.view === 'candidate') loadAsset('candidate', state.doc.b.revision);
  if (preview && state.mode === 'parts' && state.view === 'parts') {
    if (!state.hiddenParts.has(state.part)) loadAsset('highlight', preview.revision);
    if (!preview.images?.parts_overlay && !state.hiddenParts.has('background')) {
      loadAsset('background_overlay', preview.revision);
    }
  }
  if (unknownOverlayVisible()) loadAsset('unknown', preview.revision);
  if (state.view === 'changes') loadChanges();
}
const graphViews = { source: 'original', parts: 'parts', range: 'weight', effect: 'candidate', result: 'b' };
const graphEdges = [['source', 'parts'], ['parts', 'range'], ['source', 'effect'], ['range', 'result'], ['effect', 'result']];

function failedNode() {
  const phase = state.doc?.failed_phase || state.doc?.phase;
  return { analysis: 'parts', analyzing: 'parts', candidate: 'effect', compose: 'range', encoding: 'effect' }[phase] || 'effect';
}
function updateFlow() {
  if (!state.doc || !state.draft) return;
  const doc = state.doc, settings = state.draft.settings, revision = doc.b?.revision;
  const ready = renderIsCurrent();
  const hint = doc.b && !ready ? '正在显示上一版预览。选择节点可继续调整。' : '选择节点，查看或修改这一步的结果。';
  if (byId('flow-hint').textContent !== hint) byId('flow-hint').textContent = hint;
  const captions = {
    source: doc.width + ' × ' + doc.height,
    parts: '部位、外物与待确认区域',
    range: settings.modules.map(partLabel).join('、') || '从空范围开始',
    effect: methodLabel(settings.method) + ' · 强度 ' + Math.round(settings.strength * 100) + '%',
    result: ready ? '已完成 · 检查成品' : doc.status === 'error' ? '等待重试' : doc.b ? '上一版 · 正在更新' : '正在生成',
  };
  for (const [node, view] of Object.entries(graphViews)) {
    const image = byId('thumb-' + node), wait = byId('wait-' + node);
    const summary = byId('summary-' + node);
    if (summary.textContent !== captions[node]) summary.textContent = summary.title = captions[node];
    const url = node === 'source' ? route('/thumbnail') : revision == null ? ''
      : route('/thumbnail?view=' + view + '&revision=' + revision);
    const button = byId('node-' + node);
    button.dataset.state = doc.status === 'error' && node === failedNode() ? 'error' : 'normal';
    if (image.dataset.preview === url) continue;
    image.dataset.preview = url;
    image.hidden = true; wait.hidden = false;
    wait.textContent = url ? '载入预览' : '等待计算';
    if (!url) { image.removeAttribute('src'); continue; }
    image.onload = () => {
      if (image.dataset.preview !== url) return;
      image.hidden = false; wait.hidden = true;
    };
    image.onerror = () => {
      if (image.dataset.preview === url) { image.hidden = true; wait.hidden = false; wait.textContent = '预览暂不可用'; }
    };
    image.src = url;
  }
}
function drawFlowWires() {
  if (state.mode !== 'flow') return;
  const map = byId('flow-map').getBoundingClientRect();
  if (!map.width) return;
  const vertical = compactInspector.matches;
  for (const [from, to] of graphEdges) {
    const a = byId('node-' + from).getBoundingClientRect(), b = byId('node-' + to).getBoundingClientRect();
    let path;
    if (vertical) {
      const x = a.left - map.left, y = a.top - map.top + a.height / 2;
      const next = b.top - map.top + b.height / 2;
      const trail = from === 'effect' || to === 'effect' ? 4 : 16;
      path = 'M' + x + ' ' + y + ' H' + trail + ' V' + next + ' H' + x;
    } else {
      const x = a.right - map.left, y = a.top - map.top + a.height / 2;
      const end = b.left - map.left, next = b.top - map.top + b.height / 2;
      const bend = (x + end) / 2;
      path = 'M' + x + ' ' + y + ' C' + bend + ' ' + y + ' ' + bend + ' ' + next + ' ' + end + ' ' + next;
    }
    byId('wire-' + from + '-' + to).setAttribute('d', path);
  }
}
for (const node of Object.keys(graphViews)) byId('node-' + node).addEventListener('click', () => {
  if (node === 'source' || node === 'result') {
    selectMode('review');
    selectView(node === 'source' ? 'original' : 'compare');
    stage.focus({ preventScroll: true });
  } else selectMode(node);
});
function selectView(view) {
  if (!state.file || !MODE_VIEWS[state.mode].includes(view) || !viewAvailable(view)) return;
  const leavingMaskView = editingMask() && !maskInspectionView() && view !== state.view;
  if (leavingMaskView && state.sam?.busy === 'apply') return;
  setKeyboardPeek(false);
  state.view = view;
  if (leavingMaskView) {
    if (state.sam) {
      state.sam = null;
      state.tool = 'pan';
    }
    state.polygon = [];
    state.cursor = null;
    if (state.gesture?.edit) state.gesture = null;
    if (state.magnetic) resetMagnetic();
  }
  ensureAssets();
  refresh();
}
function setDrawer(open, restoreFocus = false, pointer = false) {
  document.body.dataset.drawerMotion = String(open && pointer);
  document.body.dataset.drawer = open ? 'open' : 'closed';
  byId('inspector-toggle').setAttribute('aria-expanded', String(open));
  byId('drawer-scrim').hidden = !open;
  if (open) byId('drawer-close').focus({ preventScroll: true });
  else if (restoreFocus) byId('inspector-toggle').focus({ preventScroll: true });
}
function selectMode(mode) {
  if (!state.doc || !Object.hasOwn(MODE_VIEWS, mode) || state.sam?.busy === 'apply') return;
  if (mode === state.mode) return;
  const previousMode = state.mode;
  state.sam = null;
  leaveMagnetic();
  setKeyboardPeek(false);
  state.mode = mode;
  state.polygon = [];
  state.gesture = state.cursor = state.historyLocation = null;
  state.tool = 'pan';
  setDrawer(false);
  if (editingMask()) {
    const list = byId('lasso-action');
    list.replaceChildren(...Object.entries(actions[mode]).map(([value, label]) => {
      const option = document.createElement('option');
      option.value = value; option.textContent = label;
      return option;
    }));
    list.value = state.actions[mode];
    state.tool = state.actions[mode];
    state.view = editableMaskView();
  } else {
    state.view = { review: 'compare', effect: 'result', flow: 'flow' }[mode];
  }
  ensureAssets();
  refresh();
  if (nodeNames[mode]) byId('node-title').focus({ preventScroll: true });
  if (mode === 'flow') {
    byId('node-' + (nodeNames[previousMode] ? previousMode : 'parts')).focus();
    requestAnimationFrame(drawFlowWires);
  }
}
function setEditAction(action) {
  state.actions[state.mode] = action;
  byId('lasso-action').value = action;
  if (isBrush()) state.tool = action;
}
function selectTool(tool) {
  if (tool !== 'pan' && (!state.doc || !editingMask() || maskEditsBlocked())) return;
  if (tool === 'brush') tool = state.actions[state.mode];
  if (state.sam?.busy === 'apply') return;
  if (editingMask() && Object.hasOwn(actions[state.mode], tool)) {
    setEditAction(tool);
  }
  if (state.magnetic && tool === 'magnetic') return;
  leaveMagnetic();
  state.polygon = [];
  if (tool === 'sam_object') {
    if (state.sam) return;
    state.sam = { points: [], negative: false, preview: null, busy: null, error: '' };
    state.tool = tool;
    refresh();
    return;
  }
  state.sam = null;
  state.tool = tool;
  if (tool === 'magnetic') {
    const sourceRevision = renderIsCurrent() ? state.doc.b.revision : null;
    state.magnetic = { segments: [], autoPins: new Set(), pointCount: 0, worker: null, ready: false, failed: false,
      sequence: 0, generation: 0, sourceRevision,
      sourceKey: magneticSourceKey(state.draft.settings) };
    prepareMagnetic();
  } else if (isBrush()) {
    const box = layout();
    const radius = byId('radius');
    if (!preferences.value?.remember_brush && !state.brushSized && box && Number(radius.value) * box.scale < 6) {
      radius.value = clamp(Math.round(9 / box.scale), 24, 256);
      updateBrushReadout();
    }
    state.brushSized = true;
  }
  ensureAssets(); refresh();
}

function installDraft(doc) {
  state.historyLocation = null;
  state.draft = { settings: clone(doc.settings), edits: clone(doc.edits), cursor: doc.cursor };
  syncSettings();
}
function applyMetadata(doc) {
  if (state.sam && state.doc?.revision !== doc.revision) state.sam.preview = null;
  state.doc = doc;
  state.pollEtag = null;
  if (doc.status === 'ready' && doc.b?.revision === doc.revision && state.appliedHistory?.revision !== doc.revision) {
    state.appliedHistory = { revision: doc.revision, cursor: doc.cursor, edits: doc.edits };
  }
  if (!viewAvailable('unknown')) state.images.delete('unknown');
  if (!viewAvailable('suspected_objects')) {
    state.images.delete('suspected_objects');
    if (state.view === 'suspected_objects') state.view = editableMaskView();
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
  state.leftReference = storage('getItem', comparisonKey(doc.id)) === 'original' ? 'original' : 'a';
  state.protectedParts = loadPartSwitches(protectionKey(doc.id), partCategories().filter(part => part !== 'unknown'));
  state.rangeProtectedParts = loadPartSwitches(rangeProtectionKey(doc.id));
  state.hiddenParts = loadPartSwitches(visibilityKey(doc.id));
  updatePartTargets();
  updateRangeProtection();
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
  if (doc.status === 'ready' && !state.dirty && !state.conflict) {
    post(route('/prepare-edit'), { revision: doc.revision }).catch(() => {});
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
  state.partFrame = null;
  state.displayedResult = null;
  state.polygon = [];
  state.sam = null;
  leaveMagnetic();
  state.gesture = state.cursor = null;
  state.showUnknown = true;
  state.hiddenParts = new Set();
  state.partIsolation = null;
  state.changesErrors.clear();
  state.appliedHistory = state.historyLocation = null;
  state.historyDisplayKey = '';
  state.part = 'hair';
  state.actions = { parts: 'classify', range: 'include' };
  updatePartTargets();
  setDrawer(false);
  state.pendingUnknownResolution = null;
  state.fit = true; state.panX = state.panY = 0;
  state.view = 'compare'; state.mode = 'review'; state.tool = 'pan'; state.brushSized = false; state.peek = false;
  setKeyboardPeek(false);
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
      if (byId('recent-panel').open) loadRecent();
    }
  } catch (error) {
    if (epoch === state.epoch) { state.opening = false; showError(error); status('无法打开图片，请重新选择。', 'error'); refresh(); }
  }
}
async function openRecent(id) {
  if (state.opening || !(await flushChanges()) || state.opening) return;
  if (byId('recent-panel').open) {
    byId('recent-panel').open = false;
    byId('recent-toggle').focus({ preventScroll: true });
  }
  const epoch = beginDocumentChange();
  status('正在载入已保存的图片…', 'pending');
  try {
    const doc = await (await request('/api/documents/' + encodeURIComponent(id))).json();
    if (epoch === state.epoch) { state.opening = false; startDocument(doc); }
  } catch (error) {
    if (epoch === state.epoch) { state.opening = false; showError(error); status('无法恢复草稿，请从「近期」重试。', 'error'); refresh(); }
  }
}
let recentRequest = 0;
function recentMessage(message) {
  const text = document.createElement('p');
  text.classList.add('recent-message');
  text.textContent = message;
  byId('recent').replaceChildren(text);
}
function recentButton(entry) {
  const button = document.createElement('button');
  const preview = document.createElement('span'), image = document.createElement('img');
  const fallback = document.createElement('span');
  const details = document.createElement('span'), name = document.createElement('strong');
  const metadata = document.createElement('small');
  const current = entry.id === state.doc?.id;
  const status = { queued: '等待处理', running: '处理中', error: '处理失败' }[entry.status];
  button.type = 'button';
  button.classList.add('recent-item');
  button.title = entry.name;
  if (current) button.setAttribute('aria-current', 'true');
  preview.classList.add('recent-preview');
  fallback.classList.add('recent-fallback');
  fallback.textContent = '无预览';
  fallback.hidden = true;
  fallback.setAttribute('aria-hidden', 'true');
  image.classList.add('recent-thumb');
  image.alt = '';
  image.draggable = false;
  image.loading = 'lazy';
  image.decoding = 'async';
  image.addEventListener('error', () => { image.hidden = true; fallback.hidden = false; });
  image.src = '/api/documents/' + encodeURIComponent(entry.id) + '/thumbnail';
  preview.append(image, fallback);
  if (status || current) {
    const indicator = document.createElement('span');
    indicator.classList.add('recent-indicator');
    indicator.dataset.tone = entry.status === 'error' ? 'error' : status ? 'pending' : 'current';
    indicator.textContent = status || '当前';
    preview.append(indicator);
  }
  details.classList.add('recent-details');
  name.textContent = entry.name;
  metadata.textContent = entry.width + ' × ' + entry.height;
  details.append(name, metadata);
  button.append(preview, details);
  button.addEventListener('click', () => openRecent(entry.id));
  return button;
}
async function loadRecent() {
  const requestId = ++recentRequest;
  recentMessage('正在读取近期图片…');
  try {
    const payload = await (await request('/api/documents')).json();
    if (requestId !== recentRequest || !byId('recent-panel').open) return;
    const list = Array.isArray(payload) ? payload : payload.documents;
    if (list?.length) byId('recent').replaceChildren(...list.map(recentButton));
    else recentMessage('还没有近期图片');
  } catch {
    if (requestId === recentRequest && byId('recent-panel').open) {
      recentMessage('无法读取近期图片，请关闭后重试。');
    }
  }
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  if (state.doc) state.pollTimer = setTimeout(pollDocument, ['queued', 'running'].includes(state.doc.status) ? 140 : 1250);
}
async function pollDocument() {
  const epoch = state.epoch, observed = state.doc;
  if (!state.doc) return;
  try {
    const headers = state.pollEtag ? { 'If-None-Match': state.pollEtag } : {};
    const reply = await request(route(), { headers }, true);
    const doc = reply.status === 304 ? null : await reply.json();
    if (epoch !== state.epoch || state.save || state.doc !== observed) return;
    if (doc === null) {
      ensureAssets();
      return;
    }
    if (doc.revision < observed.revision) return;
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
    state.pollEtag = reply.headers.get('ETag');
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
      const doc = await post(endpoint, payload, { keepalive: true });
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

const brushRadiusKeys = new Map([['[', -1], [']', 1], ['【', -1], ['】', 1]]);
const brushWheelStepPixels = 40;
let brushWheelPixels = 0;
let brushResizePending = false;
function updateBrushReadout() {
  const radius = Number(byId('radius').value), feather = byId('feather');
  byId('radius-value').textContent = radius * 2 + ' px';
  byId('radius').setAttribute('aria-valuetext', '直径 ' + radius * 2 + ' 像素');
  feather.max = radius;
  if (Number(feather.value) > radius) feather.value = radius;
  byId('feather-value').textContent = Number(feather.value) + ' px';
}
function resizeBrush(direction) {
  const field = byId('radius'), radius = Number(field.value);
  const next = clamp(radius + direction, Number(field.min), Number(field.max));
  if (next === radius) return;
  field.value = String(next);
  state.brushSized = true;
  brushResizePending = true;
  updateBrushReadout();
  draw();
}
function rememberBrush() {
  brushResizePending = false;
  if (preferences.value?.remember_brush) preferences.update({
    brush_radius: Number(byId('radius').value), brush_feather: Number(byId('feather').value),
  }, { syncControls: false }).catch(showError);
}
function updateReadouts() {
  for (const key of ['strength', 'restoration_gain', 'structure_protection', 'protection']) {
    byId(key + '-value').textContent = Math.round(Number(byId(key).value) * 100) + '%';
  }
  byId('retention-percent').value = Math.round(Number(byId('protection').value) * 1000) / 10;
  updateBrushReadout();
}
function updatePartStrength(input) {
  const percent = Math.round(Number(input.value) * 100) + '%';
  input.nextElementSibling.textContent = percent;
  input.setAttribute('aria-valuetext', percent);
  input.style.setProperty('--strength-level', percent);
}
function syncSettings() {
  const settings = state.draft.settings;
  for (const key of ['method', ...numberFields]) byId(key).value = settings[key];
  for (const { id, check, strength } of rangeRows) {
    if (!check) continue;
    check.checked = settings.modules.includes(id);
    strength.value = settings.part_strengths?.[id] ?? 1;
    updatePartStrength(strength);
  }
  updateReadouts();
}
function partLabel(id) {
  return state.options?.modules.find(item => item.id === id)?.label || { object: '外物', unknown: '待确认' }[id] || id;
}
function partColor(id) {
  return state.options?.modules.find(item => item.id === id)?.color || { object: '#e08844', unknown: '#89909d' }[id] || '#759bd9';
}
function partSwatch(id) {
  const swatch = document.createElement('i');
  swatch.className = id === 'unknown' ? 'hatch-key' : 'dot';
  if (id !== 'unknown') swatch.style.backgroundColor = partColor(id);
  swatch.setAttribute('aria-hidden', 'true');
  return swatch;
}
function updatePartChangesLegend(parts) {
  const legend = byId('part-changes-legend'), key = parts.join(',');
  if (legend.dataset.parts === key) return;
  legend.dataset.parts = key;
  legend.replaceChildren(...parts.map(id => {
    const label = document.createElement('span');
    label.append(partSwatch(id), partLabel(id));
    return label;
  }));
}
function partCategories() {
  return [...state.options.modules.map(item => item.id), 'object', 'unknown'];
}
function loadPartSwitches(key, defaults = []) {
  const parts = partCategories();
  try {
    const saved = JSON.parse(storage('getItem', key) ?? 'null');
    if (Array.isArray(saved) && saved.every(part => parts.includes(part))) return new Set(saved);
  } catch { /* Use the defaults if the remembered switches cannot be read. */ }
  return new Set(defaults);
}
function togglePartProtection(id, mode = 'parts') {
  if (!state.doc || state.mode !== mode || maskEditsBlocked() || state.sam?.busy === 'apply') return;
  const range = mode === 'range';
  const protectedParts = range ? state.rangeProtectedParts : state.protectedParts;
  if (id === undefined) {
    const parts = partCategories();
    if (protectedParts.size === parts.length) protectedParts.clear();
    else for (const part of parts) protectedParts.add(part);
  } else if (!protectedParts.delete(id)) {
    protectedParts.add(id);
  }
  const key = range ? rangeProtectionKey(state.doc.id) : protectionKey(state.doc.id);
  storage('setItem', key, JSON.stringify([...protectedParts]));
  if (range) {
    updateRangeProtection();
    updateToolGuidance(renderIsCurrent());
  } else updatePartTargets();
  if (state.sam) updateSamControls();
}
function togglePartVisibility(id) {
  if (!state.doc || state.mode !== 'parts' || state.view !== 'parts') return;
  state.partIsolation = null;
  if (id === undefined) {
    state.hiddenParts = new Set(state.hiddenParts.size ? [] : partCategories());
  } else if (!state.hiddenParts.delete(id)) {
    state.hiddenParts.add(id);
  }
  storage('setItem', visibilityKey(state.doc.id), JSON.stringify([...state.hiddenParts]));
  updatePartTargets();
  ensureAssets();
  refresh();
}
function updateProtectionButton(button, id, locked) {
  button.setAttribute('aria-pressed', String(locked));
  button.title = (locked ? '取消保护' : '保护') + partLabel(id);
  button.children[0].children[0].setAttribute('href', locked ? '#icon-lock' : '#icon-unlock');
}
function updateRangeProtection() {
  const count = state.rangeProtectedParts.size, all = byId('range-protection-all');
  const allLocked = count === rangeRows.length;
  all.title = allLocked ? '解锁全部分区' : '锁定全部分区';
  all.setAttribute('aria-pressed', allLocked ? 'true' : count ? 'mixed' : 'false');
  byId('range-protection-icon').setAttribute('href', count ? '#icon-lock' : '#icon-unlock');
  for (const { id, lock } of rangeRows) updateProtectionButton(lock, id, state.rangeProtectedParts.has(id));
}
function protectionButton(id, mode) {
  const button = document.createElement('button');
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('aria-hidden', 'true');
  svg.append(document.createElementNS('http://www.w3.org/2000/svg', 'use'));
  button.type = 'button';
  button.append(svg);
  if (mode === 'range') {
    button.className = 'range-part-lock';
    button.dataset.rangeProtectPart = id;
    button.setAttribute('aria-label', partLabel(id) + '范围保护');
  } else {
    button.className = 'part-lock';
    button.dataset.protectPart = id;
    button.setAttribute('aria-label', partLabel(id) + '分区保护');
  }
  button.addEventListener('click', () => togglePartProtection(id, mode));
  return button;
}
function updatePartTargets() {
  const rows = byId('part-targets').children;
  const hidden = state.hiddenParts.size, locked = state.protectedParts.size;
  const allHidden = hidden === rows.length, allLocked = locked === rows.length;
  const masterEye = byId('part-visibility-all'), masterLock = byId('part-protection-all');
  masterEye.setAttribute('aria-pressed', allHidden ? 'false' : hidden ? 'mixed' : 'true');
  masterEye.title = hidden ? '显示全部分区叠加' : '隐藏全部分区叠加';
  byId('part-visibility-all-icon').setAttribute('href', allHidden ? '#icon-eye-off' : '#icon-eye');
  masterLock.setAttribute('aria-pressed', allLocked ? 'true' : locked ? 'mixed' : 'false');
  masterLock.title = allLocked ? '解锁全部分区' : '锁定全部分区';
  byId('part-protection-all-icon').setAttribute('href', locked ? '#icon-lock' : '#icon-unlock');
  for (const row of rows) {
    const [button, eye, lock] = row.children;
    const id = button.dataset.part;
    const shown = !state.hiddenParts.has(id);
    button.setAttribute('aria-pressed', String(id === state.part));
    button.title = state.partIsolation?.part === id ? '再次 Ctrl+点击，恢复先前显隐' : 'Ctrl+点击，只显示' + partLabel(id);
    eye.setAttribute('aria-pressed', String(shown));
    eye.title = (shown ? '隐藏' : '显示') + partLabel(id) + '叠加';
    eye.children[0].children[0].setAttribute('href', shown ? '#icon-eye' : '#icon-eye-off');
    updateProtectionButton(lock, id, state.protectedParts.has(id));
  }
  byId('active-part-name').textContent = partLabel(state.part);
}
function setPart(id, isolate = false) {
  if (state.mode !== 'parts' || maskEditsBlocked() || state.sam?.busy === 'apply') return;
  if (isolate) {
    const previous = state.partIsolation;
    if (previous?.part === id) {
      state.hiddenParts = previous.hiddenParts;
      state.partIsolation = null;
    } else {
      state.partIsolation = { part: id, hiddenParts: previous?.hiddenParts ?? state.hiddenParts };
      state.hiddenParts = new Set(partCategories().filter(part => part !== id));
    }
  }
  state.part = id;
  if (id === 'unknown' && state.actions.parts === 'remove_part') setEditAction('classify');
  updatePartTargets();
  ensureAssets();
  refresh();
}
function installRangeRow(id, automatic, defaults) {
  const row = document.createElement('div'), member = document.createElement(automatic ? 'label' : 'span');
  const cell = document.createElement('div'), lock = protectionButton(id, 'range');
  row.className = automatic ? 'range-part-row' : 'range-part-row range-part-special';
  row.dataset.rangePart = id;
  member.className = 'range-member'; cell.className = 'range-strength';
  let check = null, strength = null;
  if (automatic) {
    check = document.createElement('input');
    check.type = 'checkbox'; check.value = id;
    check.checked = defaults.includes(id); check.disabled = true;
    check.addEventListener('change', () => {
      if (!state.draft || maskEditsBlocked()) return;
      state.draft.settings.modules = rangeRows.filter(row => row.check?.checked).map(row => row.id);
      edited();
    });
    member.append(check);
    const output = document.createElement('output');
    strength = document.createElement('input');
    strength.id = 'range-strength-' + id;
    strength.type = 'range'; strength.min = '0'; strength.max = '1'; strength.step = '.01'; strength.value = '1';
    strength.disabled = true;
    strength.setAttribute('aria-label', partLabel(id) + '强度');
    output.setAttribute('for', strength.id);
    cell.append(strength, output);
    updatePartStrength(strength);
    strength.addEventListener('input', () => {
      if (!state.draft || maskEditsBlocked()) return;
      state.draft.settings.part_strengths ||= {};
      state.draft.settings.part_strengths[id] = Number(strength.value);
      updatePartStrength(strength);
      edited();
    });
    if (id !== 'background') {
      const assignment = document.createElement('option');
      assignment.value = id; assignment.textContent = partLabel(id);
      byId('unknown-stroke-module').append(assignment);
    }
  }
  const label = document.createElement('span');
  label.textContent = partLabel(id); label.title = label.textContent;
  member.append(partSwatch(id), label);
  row.append(member, cell, lock);
  byId('range-part-list').append(row);
  rangeRows.push({ id, check, strength, lock });
}
function installOptions(options) {
  state.options = options;
  const order = ['skin', 'face', 'hair', 'eyes', 'mouth', 'outfit', 'background'];
  const modules = [...options.modules].sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  for (const id of [...modules.map(module => module.id), 'object', 'unknown']) {
    const row = document.createElement('div'), button = document.createElement('button');
    const eye = document.createElement('button');
    row.className = 'part-target';
    button.className = 'part-choice';
    button.type = 'button'; button.dataset.part = id;
    button.append(partSwatch(id), partLabel(id));
    button.addEventListener('click', event => setPart(id, event.ctrlKey));
    eye.type = 'button'; eye.className = 'part-eye'; eye.dataset.showPart = id;
    eye.setAttribute('aria-label', '显示' + partLabel(id) + '叠加');
    const eyeIcon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    eyeIcon.setAttribute('aria-hidden', 'true');
    eyeIcon.append(document.createElementNS('http://www.w3.org/2000/svg', 'use'));
    eye.append(eyeIcon);
    eye.addEventListener('click', () => togglePartVisibility(id));
    row.append(button, eye, protectionButton(id, 'parts'));
    byId('part-targets').append(row);
    installRangeRow(id, id !== 'object' && id !== 'unknown', options.default_modules);
  }
  updatePartTargets();
  updateRangeProtection();
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
  if (!state.draft || maskEditsBlocked()) return;
  state.draft.settings.modules = scopeModules(scope);
  syncSettings();
  edited();
}


function addEdit(edit) {
  if (!state.draft || !state.doc || maskEditsBlocked()) return;
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
  if (state.rangeProtectedParts.has('unknown') || !skippedUnknownCount(renderIsCurrent())) return;
  if (module !== undefined && (module === 'background' || !state.options.modules.some(item => item.id === module))) return;
  const last = state.draft.edits[state.draft.cursor - 1];
  if (last.stage === 'range' && module !== undefined) return;
  const edit = last.stage === 'range' ? rangeOperation('resolve_unknown') : { tool: 'resolve_unknown' };
  if (module !== undefined) edit.module = module;
  addEdit(edit);
  if (state.draft.edits[state.draft.cursor - 1] !== edit) return;
  state.pendingUnknownResolution = edit;
  persistPending();
  stage.focus({ preventScroll: true });
  saveDraft();
}
function undo() {
  if (!maskEditsBlocked() && state.draft?.cursor > 0) { state.draft.cursor--; edited(); }
}
function redo() {
  if (!maskEditsBlocked() && state.draft && state.draft.cursor < state.draft.edits.length) { state.draft.cursor++; edited(); }
}
function closeLasso(includePreview = true) {
  if (maskEditsBlocked()) return;
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
function rangeOperation(tool) {
  const edit = { tool, stage: 'range' };
  if (state.rangeProtectedParts.size) edit.protected_parts = [...state.rangeProtectedParts];
  return edit;
}
function editOperation() {
  const tool = activeAction();
  const edit = state.mode === 'range' ? rangeOperation(tool) : { tool, stage: state.mode };
  if (tool === 'protect' || tool === 'mix') edit.strength = Number(byId('protection').value);
  if (tool === 'include' && byId('allow-protected').checked) edit.allow_protected = true;
  if (tool === 'classify' || tool === 'remove_part') edit.module = state.part;
  if (tool === 'classify') edit.protected_parts = [...state.protectedParts];
  return edit;
}
function lassoAction() { return { ...editOperation(), shape: 'polygon' }; }

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
  const disabled = !canEdit || maskEditsBlocked() || !canCloseMagnetic(mag, magneticClosingTail(mag));
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
  if (!mag || maskInspectionView()) return;
  mag.straight = straight;
  if (straight) mag.motion = null;
  const point = pos && sourcePoint(pos, false, Math.floor);
  if (point && !state.space && !originalPreviewActive()) {
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
  const hidden = !mag || !box || maskEditsBlocked();
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
  if (!sam || sam.busy || maskEditsBlocked() || !renderIsCurrent()) return;
  const positive = sam.points.filter(item => !item.negative).map(item => item.point);
  const negative = sam.points.filter(item => item.negative).map(item => item.point);
  if (!positive.length) return;
  const epoch = state.epoch, revision = state.doc.revision;
  const current = () => state.sam === sam && state.epoch === epoch && state.doc?.revision === revision && renderIsCurrent();
  sam.busy = 'preview'; sam.preview = null; sam.error = '';
  refresh();
  try {
    const result = await post(route('/sam-selection'), { revision, positive, negative });
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
  if (!preview || sam.busy || maskEditsBlocked() || !renderIsCurrent() || preview.revision !== state.doc.revision) return;
  const epoch = state.epoch;
  sam.busy = 'apply'; sam.error = '';
  refresh();
  try {
    const doc = await post(route('/sam-selection/apply'), { revision: preview.revision, token: preview.token, ...editOperation() });
    if (state.epoch !== epoch || doc.revision < state.doc.revision) return;
    if (state.dirty || state.save) {
      state.conflict = doc;
    } else {
      state.sam = null;
      state.tool = 'pan';
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
  if (event.button === 0) stage.focus({ preventScroll: true, focusVisible: false });
  if (state.keyboardPeek && !editingMask()) {
    setKeyboardPeek(false);
    draw();
  }
  const pos = position(event);
  if (state.tool === 'pan' || state.space || event.button === 1 || maskEditsBlocked()) {
    if (state.magnetic) state.magnetic.motion = null;
    state.gesture = { id: event.pointerId, start: pos, panX: state.panX, panY: state.panY };
  } else if (state.doc) {
    const toPixel = state.magnetic ? Math.floor : isBrush() ? brushCoordinate : Math.round;
    const point = sourcePoint(pos, false, toPixel);
    if (!point) return;
    if (state.historyLocation !== null) {
      state.historyLocation = null;
      updateEditHistory(true);
    }
    if (state.tool === 'sam_object') {
      event.preventDefault();
      addSamPoint(point);
      return;
    }
    if (state.magnetic) {
      const mag = state.magnetic, start = mag.segments[0]?.[0];
      mag.straight = Boolean(event.altKey);
      if (canCloseMagnetic(mag) && magneticDistance(point, start) * layout().scale <= 9)
        closeLasso(false);
      else queueMagnetic(point, 'anchor', mag.straight);
      event.preventDefault();
      return;
    }
    if (state.tool === 'lasso') {
      const first = state.polygon[0], last = state.polygon.at(-1);
      if (state.polygon.length >= 3 && Math.hypot(point[0] - first[0], point[1] - first[1]) * layout().scale <= 9) closeLasso();
      else if ((!last || point[0] !== last[0] || point[1] !== last[1]) && state.polygon.length < 5000) state.polygon.push(point);
      event.preventDefault();
      refresh();
      return;
    }
    const radius = Number(byId('radius').value);
    const edit = { ...editOperation(), shape: 'brush', geometry: 'continuous', points: [point], radius,
      feather: state.mode === 'parts' ? 0 : Math.min(radius, Number(byId('feather').value)) };
    if (edit.stage === 'range' && edit.tool === 'include') edit.adaptive_edge = true;
    state.gesture = { id: event.pointerId, edit, previousPoint: null, lastPoint: point };
    byId('stroke-feedback').hidden = true;
    byId('unknown-stroke-actions').hidden = true;
  }
  if (!state.gesture) return;
  canvas.setPointerCapture(event.pointerId);
  event.preventDefault();
  draw();
}
function appendBrushSamples(gesture, event, bounds = canvas.getBoundingClientRect(), box = layout()) {
  if (!box) return;
  const { points, radius } = gesture.edit;
  const maxShift = Math.min(radius / 4, 2 / box.scale);
  for (const sample of [...(event.getCoalescedEvents?.() || []), event]) {
    if (points.length >= 5000) break;
    const point = sourcePoint(position(sample, bounds), true, brushCoordinate, box);
    const { previousPoint, lastPoint } = gesture;
    if (samePoint(point, lastPoint)) continue;
    if (previousPoint) {
      // Soften the interior point while keeping the brush tip at the pointer.
      const dx = (previousPoint[0] + point[0] - 2 * lastPoint[0]) / 4;
      const dy = (previousPoint[1] + point[1] - 2 * lastPoint[1]) / 4;
      const amount = Math.min(1, maxShift / Math.hypot(dx, dy));
      points[points.length - 1] = [brushCoordinate(lastPoint[0] + amount * dx),
        brushCoordinate(lastPoint[1] + amount * dy)];
    }
    points.push(point);
    gesture.previousPoint = lastPoint;
    gesture.lastPoint = point;
  }
}
function moveGesture(event) {
  const bounds = canvas.getBoundingClientRect(), box = layout();
  const pos = position(event, bounds);
  state.cursor = sourcePoint(pos, false, Math.round, box) ? pos : null;
  const gesture = state.gesture;
  if (state.magnetic && !gesture) {
    hoverMagnetic(state.cursor, Boolean(event.altKey), true);
    return;
  }
  if (gesture?.id === event.pointerId) {
    if (gesture.edit) {
      if (maskEditsBlocked()) {
        state.gesture = null;
        draw();
        return;
      }
      appendBrushSamples(gesture, event, bounds, box);
    } else {
      state.panX = gesture.panX + pos.x - gesture.start.x;
      state.panY = gesture.panY + pos.y - gesture.start.y;
    }
  }
  draw();
}
function simplifyBrushPath(points, tolerance) {
  if (points.length < 3) return points;
  const keep = new Uint8Array(points.length);
  const threshold = tolerance * tolerance;
  // Bound the work on very long strokes while retaining each block endpoint.
  for (let start = 0; start < points.length - 1; start += 64) {
    const end = Math.min(start + 64, points.length - 1);
    keep[start] = keep[end] = 1;
    const pending = [[start, end]];
    while (pending.length) {
      const [first, last] = pending.pop();
      const [x, y] = points[first];
      const dx = points[last][0] - x, dy = points[last][1] - y;
      const length2 = dx * dx + dy * dy;
      let farthest = 0, distance2 = threshold;
      for (let i = first + 1; i < last; i++) {
        const px = points[i][0] - x, py = points[i][1] - y;
        const along = length2 ? clamp((px * dx + py * dy) / length2, 0, 1) : 0;
        const offsetX = px - along * dx, offsetY = py - along * dy;
        const offset2 = offsetX * offsetX + offsetY * offsetY;
        if (offset2 > distance2) {
          farthest = i;
          distance2 = offset2;
        }
      }
      if (farthest) {
        keep[farthest] = 1;
        pending.push([first, farthest], [farthest, last]);
      }
    }
  }
  return points.filter((_, index) => keep[index]);
}
function endGesture(event) {
  const gesture = state.gesture;
  if (gesture?.id !== event.pointerId) return;
  state.gesture = null;
  if (gesture.edit && event.type !== 'pointercancel' && !maskEditsBlocked()) {
    const box = layout();
    appendBrushSamples(gesture, event, canvas.getBoundingClientRect(), box);
    if (box) gesture.edit.points = simplifyBrushPath(gesture.edit.points, Math.min(.5, .35 / box.scale));
    addEdit(gesture.edit);
  } else {
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
    if (preferences.value?.export_dir) {
      const result = await post(route('/export'), { revision: target.revision, view });
      if (epoch === state.epoch) {
        byId('notice').dataset.tone = 'ready';
        byId('notice-text').textContent = '已导出到 ' + result.path;
        byId('notice').hidden = false;
      }
      return;
    }
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
function setComparisonReference(reference) {
  state.leftReference = reference;
  storage('setItem', comparisonKey(state.doc.id), reference);
}
function chooseComparisonReference(reference) {
  if (!canChooseComparisonReference()) return;
  setComparisonReference(reference);
  setKeyboardPeek(false);
  ensureAssets();
  refresh();
}
async function pin() {
  if (!resultIsCurrent()) return;
  const epoch = state.epoch, revision = state.doc.revision;
  try {
    const doc = await post(route('/pin'), { revision });
    if (state.epoch === epoch && doc.revision >= state.doc.revision) {
      setComparisonReference('a');
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
  if (!state.doc || maskEditsBlocked() || !(await flushChanges())) return;
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
    if (epoch !== state.epoch || maskEditsBlocked()) return;
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
const popovers = [byId('recent-panel'), byId('theme-panel'), byId('export-panel')];
for (const popover of popovers) popover.addEventListener('toggle', () => {
  if (!popover.open) return;
  for (const other of popovers) if (other !== popover) other.open = false;
  if (popover.id === 'recent-panel') loadRecent();
});
window.addEventListener('click', event => {
  for (const popover of popovers) if (popover.open && !popover.contains(event.target)) popover.open = false;
});
byId('undo').addEventListener('click', undo);
byId('redo').addEventListener('click', redo);
byId('download').addEventListener('click', () => download());
byId('download-a').addEventListener('click', () => download('a'));
byId('pin').addEventListener('click', pin);
for (const [reference, button] of comparisonButtons) {
  button.addEventListener('click', () => chooseComparisonReference(reference));
}
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
byId('retry-changes').addEventListener('click', () => {
  state.changesErrors.delete(state.mode);
  loadChanges();
  refresh();
});
byId('show-unknown').addEventListener('click', () => {
  if (state.view !== 'weight' || !unknownOverlayAvailable()) return;
  state.showUnknown = !state.showUnknown;
  ensureAssets();
  refresh();
});
byId('include-unknown-stroke').addEventListener('click', () => resolveUnknownStroke());
byId('assign-unknown-stroke').addEventListener('click', () => resolveUnknownStroke(byId('unknown-stroke-module').value));
byId('open-unknown-parts').addEventListener('click', () => selectMode('parts'));
byId('back-to-flow').addEventListener('click', () => selectMode('flow'));
for (const button of modeButtons) button.addEventListener('click', () => selectMode(button.dataset.mode));
for (const button of scopeButtons) button.addEventListener('click', () => chooseScope(button.dataset.scope));
for (const button of toolButtons) button.addEventListener('click', () => {
  selectTool(button.dataset.tool);
  stage.focus({ preventScroll: true });
});
for (const [name, button] of effectActions) button.addEventListener('click', () => choosePreset(name));
const effectHelp = ['restoration_gain', 'structure_protection', 'sigma']
  .map(name => ({ button: byId(name + '-help-toggle'), description: byId(name + '-help') }));
for (const { button, description } of effectHelp) {
  button.title = description.textContent;
  description.hidden = true;
  button.addEventListener('click', () => {
    const opening = description.hidden;
    for (const item of effectHelp) {
      const open = opening && item.description === description;
      item.description.hidden = !open;
      item.button.setAttribute('aria-expanded', String(open));
    }
  });
}
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
for (const id of ['radius', 'feather', 'protection']) byId(id).addEventListener('input', () => {
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
byId('lasso-action').addEventListener('change', event => {
  const tool = event.target.value;
  if (maskEditsBlocked() || !Object.hasOwn(actions[state.mode] || {}, tool)) return;
  setEditAction(tool);
  refresh();
});
for (const id of ['radius', 'feather']) byId(id).addEventListener('change', rememberBrush);
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
byId('invert').addEventListener('click', () => { addEdit(rangeOperation('invert')); selectView('weight'); });
byId('reset').addEventListener('click', () => { addEdit(rangeOperation('reset')); selectView('weight'); });
byId('zoom-in').addEventListener('click', () => { const box = layout(); if (box) zoomTo(box.scale * 1.25); });
byId('zoom-out').addEventListener('click', () => { const box = layout(); if (box) zoomTo(box.scale / 1.25); });
byId('fit').addEventListener('click', () => { state.fit = true; state.panX = state.panY = 0; draw(); });
byId('actual').addEventListener('click', () => zoomTo(1));
const peekButton = byId('peek');
function setKeyboardPeek(active) {
  state.keyboardPeek = active;
  peekButton.setAttribute('aria-pressed', String(active));
}
function redrawOriginalPreview() {
  if (editingMask()) refresh();
  else draw();
}
peekButton.addEventListener('pointerdown', event => {
  state.peek = true;
  peekButton.setPointerCapture(event.pointerId);
  redrawOriginalPreview();
});
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) peekButton.addEventListener(type, () => {
  if (!state.peek) return;
  state.peek = false;
  redrawOriginalPreview();
});
peekButton.addEventListener('click', event => {
  if (event.detail !== 0) return;
  setKeyboardPeek(!state.keyboardPeek);
  redrawOriginalPreview();
});
canvas.addEventListener('wheel', event => {
  const box = layout();
  if (!box) return;
  const { deltaMode, deltaY, deltaX } = event;
  const resize = event.shiftKey && !event.altKey && !event.ctrlKey && !event.metaKey && isBrush() && !maskEditsBlocked();
  const delta = resize ? deltaY || deltaX : deltaY;
  if (!delta) return;
  event.preventDefault();
  if (resize) {
    const direction = Math.sign(delta);
    // Accumulate small trackpad deltas and limit each wheel event to one size step.
    const pixels = deltaMode === 0 ? Math.min(Math.abs(delta), brushWheelStepPixels) : brushWheelStepPixels;
    if (brushWheelPixels * direction < 0) brushWheelPixels = 0;
    brushWheelPixels += direction * pixels;
    if (Math.abs(brushWheelPixels) >= brushWheelStepPixels) {
      brushWheelPixels -= direction * brushWheelStepPixels;
      const point = position(event);
      state.cursor = sourcePoint(point, false, Math.round, box) ? point : null;
      resizeBrush(-direction);
    }
    return;
  }
  brushWheelPixels = 0;
  const point = position(event);
  zoomTo(box.scale * (delta < 0 ? 1.15 : 1 / 1.15), point.x, point.y);
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
const viewer = byId('viewer');
function indicateDrop(active) {
  for (const area of [stage, byId('flow-board')]) {
    if (active) area.classList.add('dragging');
    else area.classList.remove('dragging');
  }
}
viewer.addEventListener('dragover', event => {
  if (!event.dataTransfer.types.includes('Files')) return;
  event.preventDefault();
  indicateDrop(true);
});
viewer.addEventListener('dragleave', event => {
  if (!viewer.contains(event.relatedTarget)) indicateDrop(false);
});
viewer.addEventListener('drop', event => {
  event.preventDefault();
  indicateDrop(false);
  const transfer = event.dataTransfer;
  if (transfer.types?.includes('text/uri-list')) {
    const origin = window.location.origin;
    const hasWorkbenchPreview = transfer.getData('text/uri-list').split(/\r?\n/).some(uri =>
      uri.startsWith(origin) && /^\/api\/documents\/[0-9a-f]{64}\/thumbnail(?:[?#]|$)/.test(uri.slice(origin.length)));
    if (hasWorkbenchPreview) return;
  }
  openFile(transfer.files[0]);
});
const shortcutsDialog = byId('shortcuts-dialog');
function showShortcuts() {
  if (shortcutsDialog.open || byId('settings-dialog').open) return;
  const list = byId('shortcuts-list');
  if (!list.children.length) {
    const groups = [
      ['文件与对比', [
        ['打开图片', 'Ctrl+O'], ['粘贴图片', 'Ctrl+V'], ['导出已显示的成品 B', 'Ctrl+S'],
        ['切换左侧原图 / 固定 A', 'A'], ['按住暂看整张原图', 'X'],
        ['移动对比分割线（聚焦手柄）', '←', '→'],
      ]],
      ['画布与流程', [
        ['适合窗口', '0'], ['按原像素显示', '1'], ['放大', '+', '='], ['缩小', '-'],
        ['暂时移动画布', '空格 + 拖动'], ['打开编辑流程', 'F'], ['回到成品检查', 'R'],
      ]],
      ['绘制工具', [
        ...Object.entries(toolShortcuts).map(([key, [, label]]) => [label, key.toUpperCase()]),
        ['调画笔直径', '[', ']', 'Shift + 滚轮'],
      ]],
      ...Object.entries(actionShortcuts).map(([mode, bindings]) => [
        nodeNames[mode], [
          ...Object.entries(bindings).map(([key, action]) => [actions[mode][action], key.toUpperCase()]),
          ...(mode === 'parts' ? [['独显部位 / 恢复显隐', 'Ctrl + 点击']] : []),
        ],
      ]),
      ['选区与历史', [
        ['在画布上闭合套索，或预览 / 应用 SAM2', 'Enter'],
        ['放弃当前绘制或选区', 'Esc'], ['磁性套索临时拉直线', 'Alt'], ['撤回磁性锚点', 'Backspace'],
        ['撤销操作', 'Ctrl+Z'], ['重做操作', 'Ctrl+Shift+Z', 'Ctrl+Y'],
      ]],
    ];
    for (const [title, rows] of groups) {
      const section = document.createElement('section'), heading = document.createElement('h2');
      const definitions = document.createElement('dl');
      heading.textContent = title;
      for (const [action, ...keys] of rows) {
        const row = document.createElement('div'), term = document.createElement('dt'), values = document.createElement('dd');
        term.textContent = action;
        for (const key of keys) {
          const cap = document.createElement('kbd');
          cap.textContent = key;
          values.append(cap);
        }
        row.append(term, values);
        definitions.append(row);
      }
      section.append(heading, definitions);
      list.append(section);
    }
  }
  shortcutsDialog.showModal();
}
byId('open-shortcuts').addEventListener('click', showShortcuts);
byId('shortcuts-close').addEventListener('click', () => shortcutsDialog.close());

window.addEventListener('paste', event => {
  const file = event.clipboardData?.files[0];
  if (!byId('settings-dialog').open && !shortcutsDialog.open && file?.type.startsWith('image/')) {
    event.preventDefault();
    openFile(file);
  }
});
window.addEventListener('keydown', event => {
  if (event.defaultPrevented || event.isComposing || byId('settings-dialog').open || shortcutsDialog.open) return;
  if (event.key === 'Escape' && document.body.dataset.drawer === 'open') {
    event.preventDefault(); setDrawer(false, true); return;
  }
  const open = popovers.find(popover => popover.open);
  if (event.key === 'Escape') {
    if (open) {
      event.preventDefault();
      open.open = false;
      open.querySelector('summary').focus({ preventScroll: true });
      return;
    }
    if (state.keyboardPeek) {
      event.preventDefault();
      setKeyboardPeek(false);
      redrawOriginalPreview();
      return;
    }
  }
  if (open?.contains(event.target)) return;
  const key = event.key.toLowerCase();
  const radiusDirection = brushRadiusKeys.get(key);
  const brushSlider = radiusDirection && (event.target === byId('radius') || event.target === byId('feather'));
  if ((/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName) && !brushSlider) || event.target.isContentEditable) return;
  if (event.ctrlKey || event.metaKey) {
    if (event.altKey) return;
    if (!event.shiftKey && (key === 'o' || key === 's')) {
      event.preventDefault();
      if (!event.repeat) {
        if (key === 'o') byId('file').click();
        else download();
      }
    } else if (editingMask() && !maskEditsBlocked() && (key === 'z' || (key === 'y' && !event.shiftKey))) {
      event.preventDefault();
      const redoing = key === 'y' || event.shiftKey;
      if (state.sam) { if (!redoing) undoSamPoint(); }
      else if (state.magnetic?.segments.length || state.magnetic?.intent) { if (!redoing) resetMagnetic(true); }
      else (redoing ? redo : undo)();
    }
    return;
  }
  if (key === 'alt' && state.magnetic && !maskEditsBlocked()) {
    event.preventDefault(); hoverMagnetic(state.cursor, true); return;
  }
  if (event.altKey) return;
  if (key === '?') {
    event.preventDefault();
    if (!event.repeat) showShortcuts();
    return;
  }
  if (!state.file) return;
  if (Object.hasOwn(zoomShortcuts, key) && state.mode !== 'flow' && (key === '+' || !event.shiftKey)) {
    event.preventDefault();
    byId(zoomShortcuts[key]).click();
    return;
  }
  if (event.shiftKey) return;
  if (radiusDirection && isBrush() && !maskEditsBlocked()) {
    event.preventDefault();
    resizeBrush(radiusDirection);
    return;
  }
  const focusedControl = /^(BUTTON|SUMMARY|A)$/.test(event.target.tagName);
  if (key === ' ' && !focusedControl) { event.preventDefault(); state.space = true; refresh(); }
  if (key === 'x' && state.mode !== 'flow' && !state.peek) { state.peek = true; redrawOriginalPreview(); }
  if (key === 'a' && canChooseComparisonReference() && !event.repeat) {
    event.preventDefault();
    const next = state.leftReference === 'a' ? 'original' : 'a';
    chooseComparisonReference(next);
    if (comparisonButtons.some(([, button]) => button === event.target)) {
      const [, selected] = comparisonButtons.find(([reference]) => reference === next);
      selected.focus({ preventScroll: true });
    }
    return;
  }
  if ((key === 'f' || key === 'r') && state.doc && !event.repeat) {
    event.preventDefault();
    selectMode(key === 'f' ? 'flow' : 'review');
    if (key === 'r' && state.mode === 'review') stage.focus({ preventScroll: true });
    return;
  }
  if (key === 'escape') {
    if (state.gesture?.edit) { state.gesture = null; refresh(); }
    else if (state.sam) selectTool('pan');
    else if (state.magnetic) resetMagnetic();
    else { state.polygon = []; refresh(); }
    return;
  }
  if (key === 'backspace' && state.magnetic && !maskEditsBlocked()) { event.preventDefault(); resetMagnetic(true); }
  if (key === 'enter' && !focusedControl && !maskEditsBlocked()) {
    if (state.sam || state.polygon.length >= 3 || canCloseMagnetic(state.magnetic, magneticClosingTail(state.magnetic))) {
      event.preventDefault();
      if (event.repeat) return;
      if (state.sam?.preview) applySamObject();
      else if (state.sam) previewSamObject();
      else closeLasso();
    }
    return;
  }
  if (editingMask() && !maskEditsBlocked() && !event.repeat) {
    const tool = toolShortcuts[key]?.[0] || actionShortcuts[state.mode][key];
    if (tool) {
      event.preventDefault();
      selectTool(tool);
      stage.focus({ preventScroll: true });
    }
  }
});
window.addEventListener('keyup', event => {
  const shiftReleased = event.key === 'Shift' && !event.shiftKey;
  if (shiftReleased) brushWheelPixels = 0;
  if (brushResizePending && (shiftReleased || brushRadiusKeys.has(event.key) || event.code === 'BracketLeft' || event.code === 'BracketRight')) rememberBrush();
  if (event.key === ' ') { state.space = false; refresh(); }
  if (event.key === 'Alt' && state.magnetic) hoverMagnetic(state.cursor, false);
  if (event.key.toLowerCase() === 'x' && state.peek) { state.peek = false; redrawOriginalPreview(); }
});
window.addEventListener('blur', () => {
  brushWheelPixels = 0;
  if (brushResizePending) rememberBrush();
  state.space = state.peek = false;
  if (state.magnetic) hoverMagnetic(null, false);
  refresh();
});
let activePageId = null;
function registerPage() {
  if (activePageId || state.exited) return;
  activePageId = crypto.randomUUID();
  post('/api/page', { id: activePageId, open: true }, { keepalive: true }).catch(() => {});
}
window.addEventListener('pageshow', registerPage);
window.addEventListener('pagehide', () => {
  persistPending();
  if (state.dirty) saveDraft();
  if (!activePageId || state.exited) return;
  const id = activePageId;
  activePageId = null;
  post('/api/page', { id, open: false }, { keepalive: true }).catch(() => {});
});
registerPage();
installDockableTools({ workspace: byId('workspace'), stage, toolbar: byId('tools'),
  grip: byId('tools-handle'), compact: compactInspector });
new ResizeObserver(draw).observe(stage);
new ResizeObserver(drawFlowWires).observe(byId('flow-map'));
byId('error-node').addEventListener('click', () => selectMode(failedNode()));
byId('inspector-toggle').addEventListener('click', event => {
  setDrawer(document.body.dataset.drawer !== 'open', false, event.detail > 0);
});
byId('part-visibility-all').addEventListener('click', () => togglePartVisibility());
byId('part-protection-all').addEventListener('click', () => togglePartProtection());
byId('range-protection-all').addEventListener('click', () => togglePartProtection(undefined, 'range'));
byId('target-shortcut').addEventListener('click', event => {
  if (compactInspector.matches) setDrawer(true, false, event.detail > 0);
  else [...byId('part-targets').children].map(row => row.children[0]).find(button => button.dataset.part === state.part)?.focus();
});
compactInspector.addEventListener('change', () => { setDrawer(false); drawFlowWires(); });
byId('drawer-close').addEventListener('click', () => setDrawer(false, true));
byId('drawer-scrim').addEventListener('click', () => setDrawer(false, true));

async function exitWorkbench() {
  if (state.opening || state.sam?.busy === 'apply') throw new Error('请等待当前操作完成后再退出。');
  if (!(await flushChanges())) throw new Error('请先处理页面上的保存提示，再退出。');
  await post('/api/exit', {});
  state.options = null;
  beginDocumentChange();
  state.opening = false;
  state.exited = true;
  byId('empty-help').textContent = '可以关闭此标签页。双击启动脚本可重新打开工作台。';
  refresh();
  status('已退出工作台');
  window.close();
}

const preferences = createWorkbenchSettings({
  byId, request, post, options: () => state.options, draft: () => state.draft?.settings,
  onExit: exitWorkbench,
  onChange(value, fields) {
    if (!fields || 'theme' in fields) setThemePreference(value.theme);
    if (value.remember_brush && (!fields || 'remember_brush' in fields || 'brush_radius' in fields || 'brush_feather' in fields)) {
      byId('radius').value = value.brush_radius;
      byId('feather').max = value.brush_radius;
      byId('feather').value = value.brush_feather;
      state.brushSized = true;
      updateBrushReadout();
    }
  },
});
const preferencesReady = preferences.load().catch(showError);
request('/api/options').then(reply => reply.json()).then(options => {
  installOptions(options);
  const linkedDraft = /^#draft=([0-9a-f]{64})$/.exec(window.location.hash)?.[1];
  if (linkedDraft) window.history.replaceState(null, '', window.location.pathname + window.location.search);
  const previous = linkedDraft || storage('getItem', 'gaid-active-document');
  if (previous) {
    openRecent(previous);
    return;
  }
  preferencesReady.then(() => {
    setTimeout(() => {
      if (!state.doc && !state.opening && !state.exited) post('/api/prepare-models', {}).catch(() => {});
    }, 800);
  });
}).catch(error => {
  status('无法连接本机服务，请确认工作台已启动后刷新页面。', 'error');
  showError(error);
});

