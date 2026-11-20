"""Torch implementation of the whole-image L0 solve on CUDA, ROCm and MPS."""

import numpy as np
import torch


@torch.inference_mode()
def l0_smooth_cuda(img, lam, kappa, beta_max, device):
    dtype = torch.float64
    if torch.device(device).type == 'mps':
        # Convert on the host before transferring to Metal.
        img = np.ascontiguousarray(img, dtype=np.float32)
        dtype = torch.float32
    value = torch.as_tensor(img, dtype=dtype, device=device).permute(2, 0, 1).contiguous()
    height, width = value.shape[-2:]
    fx = value.new_zeros((height, width))
    fy = torch.zeros_like(fx)
    fx[0, 0], fx[0, -1] = -1, 1
    fy[0, 0], fy[-1, 0] = -1, 1
    otf = torch.fft.rfft2(fx).abs().square() + torch.fft.rfft2(fy).abs().square()
    initial = torch.fft.rfft2(value)
    beta = 2 * lam
    while beta < beta_max:
        h = torch.roll(value, -1, 2) - value
        v = torch.roll(value, -1, 1) - value
        remove = (h.square() + v.square()).sum(dim=0) < lam / beta
        h.masked_fill_(remove[None], 0)
        v.masked_fill_(remove[None], 0)
        div = (torch.roll(h, 1, 2) - h) + (torch.roll(v, 1, 1) - v)
        spectrum = (initial + beta * torch.fft.rfft2(div)) / (1 + beta * otf)[None]
        # Explicit shape is necessary for odd input widths to round-trip.
        value = torch.fft.irfft2(spectrum, s=(height, width))
        beta *= kappa
    return value.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
