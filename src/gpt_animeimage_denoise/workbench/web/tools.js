'use strict';

function installDockableTools({ workspace, stage, toolbar, grip, compact }) {
  const inset = 8, snapDistance = 26, dragThreshold = 5;
  let position = null, drag = null, suppressClick = false;
  const floating = () => toolbar.dataset.placement === 'floating';
  const clamp = (value, min, max) => Math.min(Math.max(value, min), Math.max(min, max));

  function setPlacement(placement) {
    toolbar.dataset.placement = placement;
    const attached = placement === 'docked';
    grip.setAttribute('aria-label', attached ? '浮动工具栏' : '停靠工具栏');
    grip.title = attached ? '拖出工具栏，或点击浮动' : '拖动或按方向键移动，点击停靠';
    if (attached) toolbar.style.transform = '';
  }

  function bounds() {
    const area = stage.getBoundingClientRect(), origin = workspace.getBoundingClientRect();
    if (!area.width || !area.height) return null;
    toolbar.style.setProperty('--tools-max-height', Math.max(28, area.height - 2 * inset) + 'px');
    const palette = toolbar.getBoundingClientRect();
    const left = area.left - origin.left, top = area.top - origin.top;
    return { originX: origin.left, originY: origin.top, left, top,
      maxX: left + area.width - inset - palette.width, maxY: top + area.height - inset - palette.height };
  }

  function place(point, area) {
    const x = clamp(Math.round(point.x), area.left + inset, area.maxX);
    const y = clamp(Math.round(point.y), area.top + inset, area.maxY);
    position = { x, y };
    toolbar.style.transform = 'translate(' + x + 'px, ' + y + 'px)';
  }

  function refit() {
    if (!floating() || toolbar.hidden) return;
    const area = bounds();
    if (!area) return;
    place(position || { x: area.left + 16, y: area.top + 16 }, area);
    if (drag?.active) drag.area = area;
  }

  grip.addEventListener('pointerdown', event => {
    if (event.button !== 0 || drag) return;
    const rect = grip.getBoundingClientRect();
    suppressClick = false;
    drag = { id: event.pointerId, startX: event.clientX, startY: event.clientY,
      relativeX: (event.clientX - rect.left) / Math.max(1, rect.width),
      relativeY: (event.clientY - rect.top) / Math.max(1, rect.height),
      placement: toolbar.dataset.placement, position };
    grip.setPointerCapture(event.pointerId);
  });

  function move(event) {
    if (drag?.id !== event.pointerId) return;
    if (!drag.active) {
      if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < dragThreshold) return;
      setPlacement('floating');
      drag.area = bounds();
      if (!drag.area) { setPlacement(drag.placement); return; }
      const rect = grip.getBoundingClientRect(), palette = toolbar.getBoundingClientRect();
      drag.offsetX = rect.left - palette.left + rect.width * drag.relativeX;
      drag.offsetY = rect.top - palette.top + rect.height * drag.relativeY;
      drag.active = true;
      toolbar.dataset.dragging = 'true';
    }
    const area = drag.area;
    const x = event.clientX - area.originX, y = event.clientY - area.originY;
    const snap = compact.matches ? y <= area.top + snapDistance : x <= area.left + snapDistance;
    if (snap !== drag.snap) {
      drag.snap = snap;
      if (snap) stage.dataset.toolsDock = 'true';
      else delete stage.dataset.toolsDock;
    }
    place({ x: x - drag.offsetX, y: y - drag.offsetY }, area);
  }
  grip.addEventListener('pointermove', move);

  function finish(event) {
    if (drag?.id !== event.pointerId) return;
    if (event.type === 'pointerup' && drag.active) move(event);
    const completed = drag;
    drag = null;
    if (!completed.active) return;
    delete toolbar.dataset.dragging;
    delete stage.dataset.toolsDock;
    suppressClick = true;
    if (event.type !== 'pointerup') {
      position = completed.position;
      setPlacement(completed.placement);
      refit();
    } else if (completed.snap) {
      setPlacement('docked');
    }
  }
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) grip.addEventListener(type, finish);

  grip.addEventListener('click', event => {
    if (event.detail > 0 && suppressClick) { suppressClick = false; return; }
    if (drag?.active || toolbar.hidden) return;
    setPlacement(floating() ? 'docked' : 'floating');
    refit();
  });

  const arrows = { ArrowLeft: [-16, 0], ArrowRight: [16, 0], ArrowUp: [0, -16], ArrowDown: [0, 16] };
  grip.addEventListener('keydown', event => {
    if (!floating() || !Object.hasOwn(arrows, event.key) || drag?.active) return;
    const area = bounds();
    if (!area || !position) return;
    event.preventDefault();
    const [dx, dy] = arrows[event.key];
    place({ x: position.x + dx, y: position.y + dy }, area);
  });

  const resize = new ResizeObserver(refit);
  resize.observe(stage);
  resize.observe(toolbar);
}
