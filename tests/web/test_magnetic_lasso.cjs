const assert = require('node:assert/strict');
const test = require('node:test');
const {
  buildEdgeMap, createEdgeReader, createTracer,
} = require('../../src/gpt_animeimage_denoise/workbench/web/magnetic-lasso-worker.js');

function pixels(x0, y0, width, height, pixel) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) rgba.set(pixel(x0 + x, y0 + y), 4 * (y * width + x));
  }
  return rgba;
}

function fixture(width, height, pixel) {
  let reads = 0;
  const read = ({ x, y, width, height, step }) => {
    assert.equal(step, 1);
    reads++;
    return pixels(x, y, width, height, pixel);
  };
  return { trace: createTracer(width, height, read), read, reads: () => reads };
}

// Numeric fixtures captured from app/index.html at CXP-2024/gpt-anime-denoise v1.2.1.
test('matches the upstream blurred luminance and transparency Sobel bytes', () => {
  const rgba = pixels(0, 0, 7, 6, (x, y) => [20 + x * 26, 40 + y * 27, 225 - x * 15, x >= 4 ? 255 : y * 31]);
  assert.deepEqual(Array.from(buildEdgeMap(rgba, 7, 6)), [
    0, 0, 0, 0, 0, 0, 0,
    0, 62, 62, 255, 239, 62, 0,
    0, 62, 62, 239, 208, 67, 0,
    0, 62, 69, 208, 177, 67, 0,
    0, 62, 66, 177, 146, 60, 0,
    0, 0, 0, 0, 0, 0, 0,
  ]);
});

test('matches full-map edge bytes across tile seams and true image borders', () => {
  const width = 520, height = 271;
  const pixel = (x, y) => [(x * 5 + y) % 256, x % 256, y % 256, (x + 3 * y) % 256];
  const { read, reads } = fixture(width, height, pixel);
  const edge = createEdgeReader(width, height, read);
  const full = buildEdgeMap(pixels(0, 0, width, height, pixel), width, height);
  const region = edge(250, 247, 270, 24);
  const expected = new Uint8Array(region.length);
  for (let y = 0; y < 24; y++) {
    const from = (247 + y) * width + 250;
    expected.set(full.subarray(from, from + 270), y * 270);
  }
  assert.deepEqual(region, expected);
  const before = reads();
  edge(250, 247, 270, 24);
  assert.equal(reads(), before);
});

test('uses the upstream radius, score threshold and transparency attraction', () => {
  const { trace } = fixture(18, 13, (x, y) => [
    ...(x >= 8 ? [200, 146, 55] : [29, 61, 220]), y >= 9 ? 0 : 255,
  ]);
  assert.deepEqual(trace({ anchor: null, point: [6, 3], scale: 1 }), { points: [[6, 8]], snapped: true });
  assert.deepEqual(trace({ anchor: null, point: [13, 8], scale: 1 }), { points: [[13, 8]], snapped: true });
  const { trace: faint } = fixture(84, 64, (x, y) => {
    const value = y < 25 + .008 * (x - 42) ** 2 ? 110 : 121;
    return [value, value, value, 255];
  });
  assert.deepEqual(faint({ anchor: null, point: [45, 25], scale: 1 }), { points: [[45, 25]], snapped: false });
  const { points, snapped } = faint({ anchor: [12, 32], point: [70, 31], scale: 1, exact: true });
  assert.ok(snapped);
  assert.deepEqual(points[0], [12, 32]);
  assert.deepEqual(points.at(-1), [70, 31]);
  assert.ok(points.some(([x, y]) => x > 25 && x < 58 && y === 26));
});

test('retains the native Dijkstra tie order with lossless collinear compression', () => {
  const { trace } = fixture(12, 9, () => [120, 120, 120, 255]);
  const result = trace({ anchor: [1, 1], point: [10, 6], scale: 1, exact: true });
  // Full upstream path: 1,1 2,2 3,3 4,4 5,4 6,4 7,5 8,6 9,6 10,6.
  assert.deepEqual(result, { points: [[1, 1], [4, 4], [6, 4], [8, 6], [10, 6]], snapped: false });
});

test('does not mark a straight flat route as snapped for a nearby isolated texture', () => {
  const { trace } = fixture(100, 64, (x, y) => {
    const value = x === 45 && y === 42 ? 250 : 120;
    return [value, value, value, 255];
  });
  assert.deepEqual(trace({ anchor: [5, 24], point: [85, 24], scale: 1 }),
    { points: [[5, 24], [85, 24]], snapped: false });
});

test('keeps endpoint attraction separate from cached path support', () => {
  const { trace } = fixture(100, 64, (x, y) => {
    const value = x < 84 ? 120 : 240;
    return [value, value, value, 255];
  });
  const point = [83, 24];
  const manual = trace({ anchor: [5, 24], point, scale: 1 });
  assert.deepEqual(manual, { points: [[5, 24], point], snapped: true });
  const exact = trace({ anchor: [5, 24], point, scale: 1, exact: true });
  assert.deepEqual(exact, { points: manual.points, snapped: false });
});

test('preserves exact closing endpoint and the upstream route at alpha edges', () => {
  const { trace } = fixture(18, 13, (x, y) => [
    ...(x >= 8 ? [200, 146, 55] : [29, 61, 220]), y >= 9 ? 0 : 255,
  ]);
  const result = trace({ anchor: [8, 2], point: [15, 8], scale: 1, exact: true });
  assert.deepEqual(result, { points: [[8, 2], [8, 7], [9, 8], [15, 8]], snapped: true });
});

test('does not read an oversized search box or allocate a full-resolution working map', () => {
  let total = 0;
  const trace = createTracer(16_384, 8_192, ({ width, height }) => {
    total += width * height;
    return new Uint8ClampedArray(width * height * 4);
  });
  const anchor = [0, 0], point = [16383, 8191];
  assert.deepEqual(trace({ anchor, point, radius: 100, scale: .1 }),
    { points: [anchor, point], snapped: false });
  assert.ok(total < 90_000, String(total));
  const tiny = fixture(1, 1, () => [10, 10, 10, 255]);
  assert.deepEqual(tiny.trace({ anchor: [0, 0], point: [0, 0], radius: 5, scale: 3 }),
    { points: [[0, 0]], snapped: false });
});
