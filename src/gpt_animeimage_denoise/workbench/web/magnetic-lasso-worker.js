'use strict';

// Adapted from CXP-2024/gpt-anime-denoise v1.2.1, app/index.html:1241–1358.
// Copyright (c) 2026 CXP-2024. MIT license, see the project LICENSE.
const EDGE_TILE = 256;
const EDGE_CACHE_LIMIT = 96;
const SEARCH_MARGIN = 24;
const MAX_SEARCH_PIXELS = 1_500_000;

function blur3(values, width, height) {
  const rowSum = new Uint16Array(values.length), blurred = new Uint8Array(values.length);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const i = row + x;
      rowSum[i] = values[i] + values[x > 0 ? i - 1 : i] + values[x < width - 1 ? i + 1 : i];
    }
  }
  for (let y = 0; y < height; y++) {
    const above = y > 0 ? -width : 0, below = y < height - 1 ? width : 0, row = y * width;
    for (let x = 0; x < width; x++) {
      const i = row + x;
      blurred[i] = (rowSum[i] + rowSum[i + above] + rowSum[i + below]) / 9;
    }
  }
  return blurred;
}

function buildEdgeMap(rgba, width, height) {
  const size = width * height;
  const lum = new Uint8Array(size), alpha = new Uint8Array(size);
  for (let i = 0, j = 0; i < size; i++, j += 4) {
    alpha[i] = rgba[j + 3];
    lum[i] = ((rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8) * alpha[i] / 255;
  }
  const blurred = blur3(lum, width, height), edges = new Uint8Array(size);
  function sobel(values, i) {
    const above = i - width, below = i + width;
    const gx = values[above + 1] + 2 * values[i + 1] + values[below + 1]
      - values[above - 1] - 2 * values[i - 1] - values[below - 1];
    const gy = values[below - 1] + 2 * values[below] + values[below + 1]
      - values[above - 1] - 2 * values[above] - values[above + 1];
    return Math.abs(gx) + Math.abs(gy);
  }
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1, i = y * width + 1; x < width - 1; x++, i++) {
      edges[i] = Math.min(255, Math.max(sobel(blurred, i), sobel(alpha, i)) / 4);
    }
  }
  return edges;
}

function createEdgeReader(width, height, readPixels) {
  const cache = new Map();
  const columns = Math.ceil(width / EDGE_TILE);
  function tile(tx, ty) {
    const key = ty * columns + tx;
    const existing = cache.get(key);
    if (existing) {
      cache.delete(key);
      cache.set(key, existing);
      return existing;
    }
    const x = tx * EDGE_TILE, y = ty * EDGE_TILE;
    const tw = Math.min(EDGE_TILE, width - x), th = Math.min(EDGE_TILE, height - y);
    // The luma box blur followed by Sobel has a two-pixel halo. Cropping
    // that halo away preserves the full-frame byte values at tile seams.
    const left = Math.max(0, x - 2), top = Math.max(0, y - 2);
    const rw = Math.min(width, x + tw + 2) - left, rh = Math.min(height, y + th + 2) - top;
    const source = buildEdgeMap(readPixels({ x: left, y: top, width: rw, height: rh, step: 1 }), rw, rh);
    const edges = new Uint8Array(tw * th);
    for (let row = 0; row < th; row++) {
      const from = (y - top + row) * rw + x - left;
      edges.set(source.subarray(from, from + tw), row * tw);
    }
    const result = { x, y, width: tw, height: th, edges };
    cache.set(key, result);
    if (cache.size > EDGE_CACHE_LIMIT) cache.delete(cache.keys().next().value);
    return result;
  }

  return (left, top, rw, rh) => {
    const output = new Uint8Array(rw * rh);
    const right = left + rw, bottom = top + rh;
    for (let ty = Math.floor(top / EDGE_TILE); ty * EDGE_TILE < bottom; ty++) {
      for (let tx = Math.floor(left / EDGE_TILE); tx * EDGE_TILE < right; tx++) {
        const part = tile(tx, ty);
        const start = Math.max(left, part.x), stop = Math.min(right, part.x + part.width);
        for (let y = Math.max(top, part.y); y < Math.min(bottom, part.y + part.height); y++) {
          const from = (y - part.y) * part.width + start - part.x;
          output.set(part.edges.subarray(from, from + stop - start), (y - top) * rw + start - left);
        }
      }
    }
    return output;
  };
}

function snapPoint(readEdges, width, height, point, radius) {
  const [px, py] = point;
  const left = Math.max(0, px - radius), top = Math.max(0, py - radius);
  const right = Math.min(width - 1, px + radius), bottom = Math.min(height - 1, py + radius);
  const rw = right - left + 1, rh = bottom - top + 1;
  const edges = readEdges(left, top, rw, rh);
  let best = -1, bx = px, by = py;
  for (let y = top; y <= bottom; y++) {
    for (let x = left; x <= right; x++) {
      const d2 = (x - px) ** 2 + (y - py) ** 2;
      if (d2 > radius * radius) continue;
      const score = edges[(y - top) * rw + x - left] - 16 * Math.sqrt(d2) / radius;
      if (score > best) { best = score; bx = x; by = y; }
    }
  }
  return best >= 20 ? { point: [bx, by], snapped: true } : { point, snapped: false };
}

function liveWire(edges, width, height, start, target, left, top) {
  const size = width * height;
  let maxEdge = 1;
  for (const value of edges) if (value > maxEdge) maxEdge = value;
  const cost = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const f = 1 - edges[i] / maxEdge;
    cost[i] = .1 + f * f;
  }
  const distance = new Float32Array(size).fill(Infinity);
  const previous = new Int32Array(size).fill(-1);
  const done = new Uint8Array(size);
  let keys = new Float32Array(1024), nodes = new Int32Array(1024), count = 0;
  // Match upstream insertion and extraction ties, including duplicate entries.
  function push(node, key) {
    if (count === keys.length) {
      const newKeys = new Float32Array(count * 2), newNodes = new Int32Array(count * 2);
      newKeys.set(keys); newNodes.set(nodes);
      keys = newKeys; nodes = newNodes;
    }
    let i = count++;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (keys[parent] <= key) break;
      keys[i] = keys[parent]; nodes[i] = nodes[parent]; i = parent;
    }
    keys[i] = key; nodes[i] = node;
  }
  function pop() {
    const first = nodes[0], key = keys[--count], node = nodes[count];
    let i = 0;
    for (;;) {
      let child = 2 * i + 1;
      if (child >= count) break;
      if (child + 1 < count && keys[child + 1] < keys[child]) child++;
      if (keys[child] >= key) break;
      keys[i] = keys[child]; nodes[i] = nodes[child]; i = child;
    }
    keys[i] = key; nodes[i] = node;
    return first;
  }

  distance[start] = 0;
  push(start, 0);
  while (count) {
    const index = pop();
    if (done[index]) continue;
    done[index] = 1;
    if (index === target) break;
    const x = index % width, y = (index - x) / width;
    const here = distance[index];
    for (let dy = -1; dy <= 1; dy++) {
      const ny = y + dy;
      if (ny < 0 || ny >= height) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx;
        if ((!dx && !dy) || nx < 0 || nx >= width) continue;
        const next = ny * width + nx;
        if (done[next]) continue;
        const d = here + cost[next] * (dx && dy ? 1.4142 : 1);
        if (d < distance[next]) {
          distance[next] = d;
          previous[next] = index;
          push(next, d);
        }
      }
    }
  }

  const points = [];
  let traversed = 0, supported = 0;
  for (let index = target; index !== -1; index = previous[index]) {
    traversed++;
    if (edges[index] >= 4) supported++;
    const x = index % width, y = (index - x) / width;
    const point = [x + left, y + top];
    const n = points.length;
    if (n >= 2) {
      const [ax, ay] = points[n - 2], [bx, by] = points[n - 1];
      const ux = bx - ax, uy = by - ay, vx = point[0] - bx, vy = point[1] - by;
      if (ux * vy === uy * vx && ux * vx + uy * vy > 0) {
        points[n - 1] = point;
        continue;
      }
    }
    points.push(point);
  }
  // Unrelated gradients in the search box do not justify an auto anchor.
  // Count support before lossless point reduction, without changing the route.
  return { points: points.reverse(), snapped: supported >= Math.max(3, traversed * .5) };
}

function createTracer(width, height, readPixels) {
  const readEdges = createEdgeReader(width, height, readPixels);
  const clampPoint = ([x, y]) => [
    Math.max(0, Math.min(width - 1, Math.floor(x))),
    Math.max(0, Math.min(height - 1, Math.floor(y))),
  ];
  let last = null;
  return ({ anchor, point, radius, scale, exact = false }) => {
    const cursor = clampPoint(point);
    const seed = anchor ? clampPoint(anchor) : null;
    const snapRadius = Math.min(24, Math.max(2, Math.round(radius ?? 10 / scale)));
    const destination = exact ? { point: cursor, snapped: false } : snapPoint(readEdges, width, height, cursor, snapRadius);
    const end = destination.point;
    if (!seed) return { points: [end], snapped: destination.snapped };
    if (seed[0] === end[0] && seed[1] === end[1]) return { points: [seed], snapped: false };
    const left = Math.max(0, Math.min(seed[0], end[0]) - SEARCH_MARGIN);
    const top = Math.max(0, Math.min(seed[1], end[1]) - SEARCH_MARGIN);
    const right = Math.min(width - 1, Math.max(seed[0], end[0]) + SEARCH_MARGIN);
    const bottom = Math.min(height - 1, Math.max(seed[1], end[1]) + SEARCH_MARGIN);
    const rw = right - left + 1, rh = bottom - top + 1;
    if (rw * rh > MAX_SEARCH_PIXELS) return { points: [seed, end], snapped: false };
    const key = [seed[0], seed[1], end[0], end[1]].join(',');
    if (last?.key !== key) {
      const start = (seed[1] - top) * rw + seed[0] - left;
      const target = (end[1] - top) * rw + end[0] - left;
      last = { key, result: liveWire(readEdges(left, top, rw, rh), rw, rh, start, target, left, top) };
    }
    const result = last.result;
    return destination.snapped && !result.snapped ? { points: result.points, snapped: true } : result;
  };
}

if (typeof module !== 'undefined') module.exports = { buildEdgeMap, createEdgeReader, createTracer };

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  let bitmap = null, trace = null;
  self.onmessage = ({ data }) => {
    try {
      if (data.type === 'init') {
        trace = null;
        bitmap?.close();
        bitmap = data.bitmap;
        const canvas = new OffscreenCanvas(1, 1);
        const context = canvas.getContext('2d', { willReadFrequently: true });
        if (!context) throw new Error('Could not read source pixels.');
        trace = createTracer(bitmap.width, bitmap.height, ({ x, y, width, height }) => {
          if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
          }
          context.clearRect(0, 0, width, height);
          context.drawImage(bitmap, x, y, width, height, 0, 0, width, height);
          return context.getImageData(0, 0, width, height).data;
        });
        self.postMessage({ type: 'ready', width: bitmap.width, height: bitmap.height });
      } else if (data.type === 'trace' && trace) {
        self.postMessage({ type: 'trace', id: data.id, ...trace(data) });
      }
    } catch (error) {
      self.postMessage({ type: 'error', id: data.id, message: error.message });
    }
  };
}
