"""Aligned tiled super-resolution shared by the CLI and Codex skill."""

from functools import partial

import numpy as np
import psutil
import torch
from PIL import Image


_RRDB_TAIL_HALO = 2
_MIN_TAIL_ROWS = 16
_HOST_RESERVE = 512 * 2**20


def _rrdb_band_rows(height, width, channels, element_size, scale):
    available = psutil.virtual_memory().available
    budget = min(available * 3 // 5, max(0, available - _HOST_RESERVE))
    # Reserve room for the live feature maps and convolution workspace at
    # the highest resolution. The remaining RAM also serves the trunk and stitcher.
    bytes_per_row = width * channels * element_size * scale**2 * 6
    if height * bytes_per_row <= budget:
        return height
    for rows in (256, 128, 64, 32):
        if (rows + 2 * _RRDB_TAIL_HALO) * bytes_per_row <= budget:
            return min(height, rows)
    return min(height, _MIN_TAIL_ROWS)


def _rrdb_forward(model, head, tail, block):
    input_height, input_width = block.shape[-2:]
    shuffle = model.shuffle_factor or 1
    height = (input_height + shuffle - 1) // shuffle
    width = (input_width + shuffle - 1) // shuffle
    scale = model.scale
    rows = _rrdb_band_rows(height, width, tail[1].in_channels, block.element_size(), scale)
    if rows >= height:
        return model(block)

    if shuffle != 1:
        from spandrel.architectures.__arch_helpers.padding import pad_to_multiple

        block = torch.pixel_unshuffle(pad_to_multiple(block, shuffle, mode='reflect'), shuffle)
    features = head(block)
    output = features.new_empty((features.shape[0], tail[-1].out_channels, height * scale, width * scale))
    for top in range(0, height, rows):
        bottom = min(top + rows, height)
        start = max(0, top - _RRDB_TAIL_HALO)
        stop = min(height, bottom + _RRDB_TAIL_HALO)
        band = tail(features[..., start:stop, :])
        output[..., top * scale:bottom * scale, :] = band[..., (top - start) * scale:(bottom - start) * scale, :]
        del band
    output_scale = scale // shuffle
    return output[..., :input_height * output_scale, :input_width * output_scale]


def _cpu_forward(model):
    layers = getattr(model, 'model', None)
    if not isinstance(layers, torch.nn.Sequential):
        return model

    from spandrel.architectures.ESRGAN.__arch.RRDB import RRDBNet

    if type(model) is not RRDBNet or model.scale != 4 or model.shuffle_factor not in (None, 2):
        return model
    nn = torch.nn
    tail = layers[2:]
    # Both shipped RRDB networks have two nearest-neighbor 2x resize stages,
    # then local 3x3 convolutions with a two-row halo in the feature grid.
    expected = (nn.Upsample, nn.Conv2d, nn.LeakyReLU, nn.Upsample, nn.Conv2d,
                nn.LeakyReLU, nn.Conv2d, nn.LeakyReLU, nn.Conv2d)
    if tuple(map(type, tail)) != expected:
        return model
    for layer in tail:
        if isinstance(layer, nn.Upsample) and (layer.mode, layer.scale_factor) != ('nearest', 2):
            return model
        if isinstance(layer, nn.Conv2d) and (layer.kernel_size, layer.stride, layer.padding,
                                             layer.dilation, layer.padding_mode) != ((3, 3), (1, 1), (1, 1), (1, 1), 'zeros'):
            return model
    return partial(_rrdb_forward, model, layers[:2], tail)


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
    Accumulation keeps one tile row on CPU. Completed rows are converted in
    small stripes and written directly into the RGB result image.
    The CPU can also run the RRDB upsampling layers in memory-sized row bands.
    """
    if tile <= 0 or padding < 0:
        raise ValueError('tile must be positive and padding must be nonnegative')
    # APISR uses pixel unshuffle. Keeping origins aligned avoids changing its
    # sampling phase between tiles.
    tile = max(4, ((tile + 3) // 4) * 4)
    padding = ((padding + 3) // 4) * 4
    overlap = (tile // 4 // 4) * 4
    stride = tile - overlap
    infer = _cpu_forward(model) if torch.device(device).type == 'cpu' else model
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    height, width = pixels.shape[:2]
    row_origins = range(0, max(height - tile, 0) + stride, stride)
    column_origins = range(0, max(width - tile, 0) + stride, stride)
    result = None
    result_sum = weight_sum = None
    scale = None
    for row_index, y in enumerate(row_origins):
        y1 = min(y + tile, height)
        for x in column_origins:
            x1 = min(x + tile, width)
            ya, xa = max(y - padding, 0), max(x - padding, 0)
            yb, xb = min(y1 + padding, height), min(x1 + padding, width)
            block = torch.from_numpy(pixels[ya:yb, xa:xb].copy()).permute(2, 0, 1)[None].to(device)
            with torch.inference_mode():
                out = infer(block).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            bh, bw = yb - ya, xb - xa
            if out.shape[0] % bh or out.shape[1] % bw:
                raise ValueError(f'Unexpected model output shape {out.shape} for tile {(bh, bw)}')
            block_scale = out.shape[0] // bh
            if block_scale < 1 or out.shape[1] // bw != block_scale:
                raise ValueError('Model produced unequal or invalid upscale factors')
            if result is None:
                scale = block_scale
                band_shape = (min(tile, height) * scale, width * scale)
                result_sum = np.zeros((*band_shape, 3), dtype=np.float32)
                weight_sum = np.zeros(band_shape, dtype=np.float32)
                result = Image.new('RGB', (width * scale, height * scale))
            elif block_scale != scale:
                raise ValueError('Model upscale factor changed between tiles')
            core = out[(y - ya) * scale:(y1 - ya) * scale,
                       (x - xa) * scale:(x1 - xa) * scale]
            wy = _axis_weights(core.shape[0], overlap * scale, y > 0, y1 < height)
            wx = _axis_weights(core.shape[1], overlap * scale, x > 0, x1 < width)
            weights = wy[:, None] * wx[None, :]
            columns = slice(x * scale, x1 * scale)
            result_sum[:core.shape[0], columns] += core * weights[..., None]
            weight_sum[:core.shape[0], columns] += weights
            del block, out, core, weights

        # Subsequent tile rows can only contribute from the next origin onward.
        next_y = y + stride if row_index + 1 < len(row_origins) else height
        completed = (next_y - y) * scale
        for row in range(0, completed, 256):
            end = min(row + 256, completed)
            values = result_sum[row:end]
            values /= weight_sum[row:end, :, None]
            np.clip(values, 0, 1, out=values)
            values *= 255
            values += 0.5
            result.paste(Image.fromarray(values.astype(np.uint8)), (0, y * scale + row))

        if next_y < height:
            retained = (y1 - next_y) * scale
            result_sum[:retained] = result_sum[completed:completed + retained]
            weight_sum[:retained] = weight_sum[completed:completed + retained]
            result_sum[retained:] = 0
            weight_sum[retained:] = 0
    return result

