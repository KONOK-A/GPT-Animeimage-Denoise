"""Double-precision CUDA implementation of the existing whole-image L0 solve."""

import torch


@torch.inference_mode()
def l0_smooth_cuda(img, lam, kappa, beta_max, device):
    # Channels form the FFT batch. No resizing, independent tiles, or reduced
    # precision: the periodic gradient objective matches the NumPy CPU path.
    value = torch.as_tensor(img, dtype=torch.float64, device=device).permute(2, 0, 1).contiguous()
    height, width = value.shape[-2:]
    fx = torch.zeros((height, width), dtype=value.dtype, device=device)
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
