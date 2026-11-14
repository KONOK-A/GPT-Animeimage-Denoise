"""Aligned tiled super-resolution shared by the CLI and Codex skill."""

import numpy as np
import torch
from PIL import Image


def _axis_weights(length, overlap, before, after):
    weights = np.ones(length, dtype=np.float32)
    n = min(overlap, length)
    if n:
        ramp = np.sin((np.arange(n, dtype=np.float32) + 0.5) * (np.pi / (2 * n))) ** 2
        if before:
            weights[:n] *= ramp
        if after:
            weights[-n:] *= ramp[::-1]
    return weights


def tiled_upscale(model, image, device, tile=256, padding=64):
    """Blend context-cropped predictions in float32, with 4-pixel aligned origins.

    The 64-pixel context is discarded; a separate 64-pixel core overlap is
    feathered. This reduces local border errors and smooths models whose global
    statistics depend on each tile. Such models are not whole-image equivalent.
    Accumulation stays on CPU, and normalization is striped to bound RAM use.
    """
    if tile <= 0 or padding < 0:
        raise ValueError('tile must be positive and padding must be nonnegative')
    # APISR uses pixel unshuffle. Keeping origins aligned avoids changing its
    # sampling phase between tiles.
    tile = max(4, ((tile + 3) // 4) * 4)
    padding = ((padding + 3) // 4) * 4
    overlap = (tile // 4 // 4) * 4
    stride = tile - overlap
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    height, width = pixels.shape[:2]
    result_sum = weight_sum = None
    scale = None
    for y in range(0, max(height - tile, 0) + stride, stride):
        for x in range(0, max(width - tile, 0) + stride, stride):
            y1, x1 = min(y + tile, height), min(x + tile, width)
            ya, xa = max(y - padding, 0), max(x - padding, 0)
            yb, xb = min(y1 + padding, height), min(x1 + padding, width)
            block = torch.from_numpy(pixels[ya:yb, xa:xb].copy()).permute(2, 0, 1)[None].to(device)
            with torch.inference_mode():
                out = model(block).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            bh, bw = yb - ya, xb - xa
            if out.shape[0] % bh or out.shape[1] % bw:
                raise ValueError(f'Unexpected model output shape {out.shape} for tile {(bh, bw)}')
            block_scale = out.shape[0] // bh
            if block_scale < 1 or out.shape[1] // bw != block_scale:
                raise ValueError('Model produced unequal or invalid upscale factors')
            if result_sum is None:
                scale = block_scale
                result_sum = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
                weight_sum = np.zeros((height * scale, width * scale), dtype=np.float32)
            elif block_scale != scale:
                raise ValueError('Model upscale factor changed between tiles')
            core = out[(y - ya) * scale:(y1 - ya) * scale,
                       (x - xa) * scale:(x1 - xa) * scale]
            wy = _axis_weights(core.shape[0], overlap * scale, y > 0, y1 < height)
            wx = _axis_weights(core.shape[1], overlap * scale, x > 0, x1 < width)
            weights = wy[:, None] * wx[None, :]
            result_sum[y * scale:y1 * scale, x * scale:x1 * scale] += core * weights[..., None]
            weight_sum[y * scale:y1 * scale, x * scale:x1 * scale] += weights
    result = np.empty(result_sum.shape, dtype=np.uint8)
    for row in range(0, result.shape[0], 256):
        values = result_sum[row:row + 256]
        values /= weight_sum[row:row + 256, :, None]
        result[row:row + 256] = (values.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(result)

