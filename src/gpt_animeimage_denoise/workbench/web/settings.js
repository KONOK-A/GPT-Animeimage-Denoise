'use strict';

function createWorkbenchSettings({ byId, request, post, options, draft, onChange, onExit }) {
  const dialog = byId('settings-dialog'), form = byId('settings-form');
  const tabs = ['defaults', 'storage', 'appearance'];
  const themeNames = ['system', 'light', 'dark'];
  const profileField = byId('default-profile');
  const profileNotes = {
    auto: '有分区模型时使用按部位精修，否则使用整图降噪。',
    whole: '直接处理整张图。纯 L0 无需下载模型。',
    fusion: '优先识别人物部位。缺少分区模型时使用整图降噪。',
    ask: '导入后先显示图片，选择模式后开始处理。',
  };
  function describeProfile() {
    byId('default-profile-note').textContent = profileNotes[profileField.value] || '';
  }
  profileField.addEventListener('change', describeProfile);
  const fields = new Map();
  let value, read, queue = Promise.resolve(), defaults, saved, busy = false, visit = 0;
  let cacheInfo, cacheRead = 0, diagnosing = false;
  const runDoctor = byId('run-doctor');

  function install(next, changed, syncControls = true) {
    value = next;
    if (syncControls) onChange(next, changed);
    return next;
  }
  function load() {
    if (!read) read = request('/api/settings').then(response => response.json())
      .then(next => install(next)).catch(error => { read = null; throw error; });
    return read;
  }
  function update(patch, { syncControls = true } = {}) {
    const task = queue.then(() => load()).then(() => post('/api/settings', patch))
      .then(next => install(next, patch, syncControls));
    queue = task.catch(() => {});
    return task;
  }
  function message(text, error = false) {
    const label = byId('pref-message');
    label.textContent = text;
    label.dataset.tone = error ? 'error' : 'neutral';
  }
  function setBusy(next) {
    busy = next;
    form.setAttribute('aria-busy', String(next));
    for (const id of ['prefs-save', 'settings-close', 'prefs-cancel', 'exit-workbench',
      'confirm-cache', 'cancel-cache', 'choose-export-dir', 'choose-cache-dir']) {
      byId(id).disabled = next;
    }
    updateCacheActions();
    runDoctor.disabled = busy || diagnosing;
  }
  function selectTab(name) {
    for (const key of tabs) {
      const active = key === name, tab = byId('tab-' + key);
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      byId('prefs-' + key).hidden = !active;
    }
    if (name === 'storage') cacheUsage(visit);
  }
  function formatBytes(bytes) {
    const units = [[1024 ** 3, 'GiB'], [1024 ** 2, 'MiB'], [1024, 'KiB'], [1, 'B']];
    const [scale, name] = units.find(([scale]) => bytes >= scale) || units.at(-1);
    return (bytes / scale).toLocaleString('zh-CN', { maximumFractionDigits: 1 }) + ' ' + name;
  }
  function updateCacheActions() {
    byId('compact-cache').disabled = busy || !cacheInfo?.reclaimable_bytes;
    byId('clear-cache').disabled = busy || !cacheInfo?.bytes;
  }
  function showCacheUsage(info) {
    cacheInfo = info;
    byId('cache-size').textContent = '缓存占用 ' + formatBytes(info.bytes);
    byId('cache-location').textContent = info.path;
    const reclaimable = byId('cache-reclaimable');
    reclaimable.hidden = !info.reclaimable_bytes;
    reclaimable.textContent = info.reclaimable_bytes
      ? '可释放 ' + formatBytes(info.reclaimable_bytes) + '，保留编辑所需的分区与降噪缓存。' : '';
    updateCacheActions();
  }
  async function cacheUsage(generation) {
    const reading = ++cacheRead;
    try {
      const info = await (await request('/api/cache')).json();
      if (dialog.open && generation === visit && reading === cacheRead) showCacheUsage(info);
    } catch (error) { if (generation === visit && reading === cacheRead) message(error.message, true); }
  }
  function displayDefaults(source) {
    defaults = JSON.parse(JSON.stringify(source));
    byId('default-method').value = defaults.method;
    byId('default-strength').value = Math.round(defaults.strength * 1000) / 10;
    for (const [id, { checked, strength }] of fields) {
      checked.checked = defaults.modules.includes(id);
      strength.value = Math.round((defaults.part_strengths?.[id] ?? 1) * 1000) / 10;
    }
  }
  function buildOptions() {
    if (fields.size) return;
    for (const item of options().methods) {
      const option = document.createElement('option');
      option.value = item.id; option.textContent = item.label;
      byId('default-method').append(option);
    }
    const ordered = ['skin', 'face', 'hair', 'eyes', 'mouth', 'outfit', 'background'];
    for (const id of ordered) {
      const item = options().modules.find(part => part.id === id);
      if (!item) continue;
      const row = document.createElement('div'), label = document.createElement('label');
      const checked = document.createElement('input'), swatch = document.createElement('i');
      checked.type = 'checkbox'; swatch.className = 'dot'; swatch.style.backgroundColor = item.color;
      label.append(checked, swatch, item.label);
      const amount = document.createElement('label'), strength = document.createElement('input');
      amount.className = 'pref-percentage';
      strength.type = 'number'; strength.min = '0'; strength.max = '100'; strength.step = '0.1';
      strength.setAttribute('aria-label', item.label + '默认强度'); strength.required = true;
      amount.append(strength, '%'); row.append(label, amount);
      byId('default-modules').append(row); fields.set(id, { checked, strength });
    }
  }
  function brushState() {
    const remember = byId('pref-remember-brush').checked;
    byId('pref-brush-size').disabled = byId('pref-brush-feather').disabled = !remember;
    const limit = Math.floor(Number(byId('pref-brush-size').value) / 2);
    byId('pref-brush-feather').max = Math.max(0, limit);
    if (Number(byId('pref-brush-feather').value) > limit) byId('pref-brush-feather').value = limit;
  }
  async function open(event) {
    if (!options() || dialog.open) return;
    const generation = ++visit;
    dialog.dataset.openPointer = String(event.detail > 0);
    dialog.showModal();
    cacheInfo = null;
    byId('cache-size').textContent = '正在读取占用…';
    byId('cache-location').textContent = '';
    byId('cache-reclaimable').hidden = true;
    setBusy(true);
    message('正在读取设置…');
    try {
      await load();
      await queue;
      if (generation !== visit || !dialog.open) return;
      // Refresh path and defaults if another tab changed the host preferences.
      saved = await (await request('/api/settings')).json();
      if (generation !== visit || !dialog.open) return;
      install(saved);
      buildOptions();
      displayDefaults(saved.defaults);
      profileField.value = saved.default_profile || 'auto';
      describeProfile();
      byId('pref-export-dir').value = saved.export_dir;
      byId('pref-cache-dir').value = saved.cache_dir;
      byId('pref-theme-' + saved.theme).checked = true;
      byId('pref-remember-brush').checked = saved.remember_brush;
      byId('pref-exit-on-close').checked = saved.exit_on_close;
      byId('pref-brush-size').value = saved.brush_radius * 2;
      byId('pref-brush-feather').value = saved.brush_feather;
      brushState();
      byId('default-from-image').disabled = !draft();
      byId('cache-confirm').hidden = true;
      selectTab('defaults'); message('');
      byId('tab-defaults').focus();
    } catch (error) { saved = null; message(error.message, true); }
    finally { setBusy(false); byId('prefs-save').disabled = !saved; }
  }
  function close() {
    if (busy) return;
    dialog.close();
    byId('open-settings').focus({ preventScroll: true });
  }
  async function save(event) {
    event.preventDefault();
    if (busy || !saved || !form.reportValidity()) return;
    const next = { ...defaults, method: byId('default-method').value, strength: Number(byId('default-strength').value) / 100,
      modules: [], part_strengths: {} };
    for (const [id, { checked, strength }] of fields) {
      if (checked.checked) next.modules.push(id);
      const level = Number(strength.value) / 100;
      if (level !== 1) next.part_strengths[id] = level;
    }
    const values = {
      defaults: next, default_profile: profileField.value,
      export_dir: byId('pref-export-dir').value.trim(), cache_dir: byId('pref-cache-dir').value.trim(),
      theme: themeNames.find(name => byId('pref-theme-' + name).checked) || 'system',
      remember_brush: byId('pref-remember-brush').checked, brush_radius: Number(byId('pref-brush-size').value) / 2,
      brush_feather: Number(byId('pref-brush-feather').value), exit_on_close: byId('pref-exit-on-close').checked,
    };
    const patch = Object.fromEntries(Object.entries(values).filter(([key, setting]) => JSON.stringify(setting) !== JSON.stringify(saved[key])));
    setBusy(true); message('正在保存…');
    try {
      if (Object.keys(patch).length) await update(patch);
      setBusy(false); close();
    } catch (error) { setBusy(false); message(error.message, true); }
  }
  byId('open-settings').addEventListener('click', open);
  runDoctor.addEventListener('click', async () => {
    if (busy || diagnosing) return;
    const status = byId('doctor-status'), output = byId('doctor-report');
    diagnosing = runDoctor.disabled = true;
    output.hidden = true;
    status.textContent = '正在检查环境…';
    status.dataset.tone = 'neutral';
    status.hidden = false;
    try {
      const { report, text } = await post('/api/doctor', {
        whole: Boolean(byId('doctor-whole').checked),
        verify_models: Boolean(byId('doctor-verify').checked),
      });
      status.textContent = report.status === 'ok' ? '检查完成' : '请查看报告中的修复方法';
      status.dataset.tone = report.status === 'ok' ? 'neutral' : 'error';
      output.textContent = text;
      output.hidden = false;
    } catch (error) {
      status.textContent = error.message;
      status.dataset.tone = 'error';
    } finally {
      diagnosing = false;
      runDoctor.disabled = busy;
    }
  });
  byId('exit-workbench').addEventListener('click', async () => {
    if (busy) return;
    setBusy(true);
    message('正在退出工作台…');
    try {
      await onExit();
      setBusy(false);
      close();
    } catch (error) { setBusy(false); message(error.message, true); }
  });
  for (const id of ['settings-close', 'prefs-cancel']) byId(id).addEventListener('click', close);
  dialog.addEventListener('cancel', event => { if (busy) event.preventDefault(); });
  dialog.addEventListener('close', () => { visit++; });
  form.addEventListener('submit', save);
  for (const [index, name] of tabs.entries()) {
    byId('tab-' + name).addEventListener('click', () => selectTab(name));
    byId('tab-' + name).addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
        : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      selectTab(tabs[next]); byId('tab-' + tabs[next]).focus();
    });
  }
  byId('default-from-image').addEventListener('click', () => {
    if (draft()) { displayDefaults(draft()); message('已填入当前图的参数，保存后用于新图。'); }
  });
  byId('pref-remember-brush').addEventListener('change', brushState);
  byId('pref-brush-size').addEventListener('change', brushState);
  for (const kind of ['export', 'cache']) byId('choose-' + kind + '-dir').addEventListener('click', async () => {
    const input = byId('pref-' + kind + '-dir');
    setBusy(true); message('请选择本机文件夹…');
    try {
      const reply = await post('/api/folder', { initial: input.value.trim() });
      if (reply.path) input.value = reply.path;
      message('');
    } catch (error) { message(error.message, true); }
    finally { setBusy(false); }
  });
  byId('clear-cache').addEventListener('click', () => { byId('cache-confirm').hidden = false; });
  byId('cancel-cache').addEventListener('click', () => { byId('cache-confirm').hidden = true; });
  async function cleanCache(mode) {
    if (busy) return;
    setBusy(true); message('正在清理缓存…');
    try {
      const info = await post('/api/cache/clear', { mode });
      cacheRead++;
      showCacheUsage(info);
      message(info.released_bytes ? '已释放 ' + formatBytes(info.released_bytes) + '。' : '当前没有可清理的缓存。');
    }
    catch (error) { message(error.message, true); }
    finally { setBusy(false); byId('cache-confirm').hidden = true; }
  }
  byId('compact-cache').addEventListener('click', () => cleanCache('compact'));
  byId('confirm-cache').addEventListener('click', () => cleanCache('all'));
  return { get value() { return value; }, update, load };
}
