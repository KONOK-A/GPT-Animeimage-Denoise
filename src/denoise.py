"""Remove the fish-scale texture that GPT image models (gpt-image-2 / 2.5, ChatGPT, Codex image_gen) leave in
anime and TV-anime style images.

Default method (recommended): downscale to 60% -> Real-ESRGAN x4plus anime 6B -> resize back -> L0 smoothing 0.001.

  python denoise.py image.png [more.png ...]            # default method only
  python denoise.py image.png --all                      # all 5 candidates + a 3x2 comparison sheet
  python denoise.py ./folder --out ./cleaned             # every image in a folder
  python denoise.py image.png --l0 0.002                 # stronger L0 (0.001 is only the recommended value)

Super-resolution uses overlapping tiled inference. L0 uses float64 on CUDA/ROCm and float32 on Apple MPS.
Outputs keep the input size. Models are downloaded on first use into --models
(default: GPT_ANIMEIMAGE_DENOISE_MODEL_DIR or ./models).
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path
from warnings import warn

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gpt_animeimage_denoise.devices import DEVICE_HELP, clear_accelerator_cache, resolve_device
from gpt_animeimage_denoise.model_paths import model_root
from l0_cuda import l0_smooth_cuda
from tiled_inference import tiled_upscale

MODELS = {
    "esrgan": ("RealESRGAN_x4plus_anime_6B.pth",
               "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
    "apisr": ("2x_APISR_RRDB_GAN_generator.pth",
              "https://huggingface.co/camenduru/APISR/resolve/main/2x_APISR_RRDB_GAN_generator.pth"),
}
SHA256 = {
    "RealESRGAN_x4plus_anime_6B.pth": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    "2x_APISR_RRDB_GAN_generator.pth": "3b0d2b3a3c0461ac17d00f4f32240666fb832b738ea5a48449b1acf07fbb07e5",
}
DOWN = 0.6
L0 = 0.001  # recommended default; tune with --l0 (larger = flatter, too large wipes soft blush/gradients)
METHODS = ["esrgan60_l0", "esrgan60", "apisr", "l0_0.001", "l0_0.0005"]
LABELS = {"original": "original", "l0_0.0005": "L0 0.0005", "l0_0.001": "L0 0.001", "apisr": "APISR",
          "esrgan60": "ESRGAN @60%", "esrgan60_l0": "ESRGAN @60% + L0 {lam:g} (default)"}
SHEET_ORDER = ["original", "l0_0.0005", "l0_0.001", "apisr", "esrgan60", "esrgan60_l0"]


def fetch(key, folder):
    name, url = MODELS[key]
    path = folder / name
    if path.exists():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...", flush=True)
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "gpt-animeimage-denoise"})).read()
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        raise SystemExit(f"checksum mismatch for {name} from {url}; not saved")
    path.write_bytes(data)
    return path


def load_model(key, folder, device):
    import spandrel
    return spandrel.ModelLoader().load_from_file(str(fetch(key, folder))).model.to(device).eval()


def _validate_l0_strength(lam):
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("L0 strength must be a finite positive value")


def _l0_divergence_spectrum(value, threshold, spectrum):
    height = value.shape[0]
    rows_per_band = 64
    for top in range(0, height, rows_per_band):
        bottom = min(top + rows_per_band, height)
        # The first row supplies the preceding vertical gradient, including
        # the wrap from the bottom of the image into the top row.
        rows = np.arange(top - 1, bottom)
        band = value[rows]
        h = np.roll(band, -1, axis=1)
        h -= band
        v = value[(rows + 1) % height]
        v -= band
        np.square(h, out=band)
        band += np.square(v)
        remove = band.sum(axis=2) < threshold
        h[remove] = 0
        v[remove] = 0
        div = np.roll(h[1:], 1, axis=1)
        div -= h[1:]
        div += v[:-1] - v[1:]
        np.fft.rfft(div, axis=1, out=spectrum[top:bottom])
    np.fft.fft(spectrum, axis=0, out=spectrum)


def l0_smooth(img, lam, kappa=2.0, beta_max=1e5, *, device="cpu"):
    """L0 gradient minimization (Xu et al. 2011) on an HxWx3 float image in [0, 1].

    Flattens small ripples inside flat colour areas while keeping lines and colour edges; invents no new detail.
    The whole-image real FFT runs in float32 on MPS and float64 elsewhere.
    """
    import torch

    _validate_l0_strength(lam)
    if not np.isfinite(kappa) or kappa <= 1:
        raise ValueError("L0 kappa must be finite and greater than 1")
    if not np.isfinite(beta_max) or beta_max <= 0:
        raise ValueError("L0 beta_max must be a finite positive value")
    kind = torch.device(device).type
    if kind == "cuda":
        return l0_smooth_cuda(img, lam, kappa, beta_max, device)
    if kind == "mps":
        try:
            result = l0_smooth_cuda(img, lam, kappa, beta_max, device)
        except RuntimeError as error:
            if not isinstance(error, torch.OutOfMemoryError) and "out of memory" not in str(error).lower():
                raise
            result = None
            warn("MPS memory exhausted. Running L0 on CPU.", RuntimeWarning, stacklevel=2)
        clear_accelerator_cache(device)
        if result is not None:
            return result
    value = img.astype(np.float64, order="C")
    height, width, channels = value.shape
    fx = np.zeros(width)
    fy = np.zeros(height)
    fx[0], fx[-1] = -1, 1
    fy[0], fy[-1] = -1, 1
    # The horizontal and vertical kernels each vary along just one axis.
    otf = np.abs(np.fft.rfft(fx))[None, :] ** 2 + np.abs(np.fft.fft(fy))[:, None] ** 2
    initial = np.empty((height, width // 2 + 1, channels), dtype=np.complex128)
    np.fft.rfft2(value, axes=(0, 1), out=initial)
    spectrum = np.empty_like(initial)
    denominator = np.empty_like(otf)
    beta = 2 * lam
    while beta < beta_max:
        _l0_divergence_spectrum(value, lam / beta, spectrum)
        spectrum *= beta
        np.add(initial, spectrum, out=spectrum)
        np.multiply(otf, beta, out=denominator)
        denominator += 1
        spectrum /= denominator[..., None]
        # Invert the vertical transform in place, then restore the exact width.
        np.fft.ifft(spectrum, axis=0, out=spectrum)
        np.fft.irfft(spectrum, n=width, axis=1, out=value)
        beta *= kappa
    return value.clip(0, 1, out=value)


def to_arr(im):
    return np.asarray(im.convert("RGB"), np.float64) / 255


def to_img(a):
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def upscale(model, im, device):
    return tiled_upscale(model, im, device)


def sr_at(model, src, device):
    """Downscale (box filter averages the fine texture away), upscale with the model, resize back to the input size."""
    W, H = src.size
    small = src.resize((round(W * DOWN), round(H * DOWN)), Image.BOX)
    return upscale(model, small, device).resize((W, H), Image.LANCZOS)


def run(src, methods, models, device, lam=L0):
    W, H = src.size
    out, cache = {}, {}
    for m in methods:
        if m.startswith("l0_"):
            out[m] = to_img(l0_smooth(to_arr(src), float(m[3:]), device=device))
        elif m == "apisr":
            out[m] = upscale(models["apisr"], src, device).resize((W, H), Image.LANCZOS)
        else:
            key = m.split("60")[0]
            if key not in MODELS:
                raise ValueError(f"Unknown denoiser: {m}")
            if key not in cache:
                cache[key] = sr_at(models[key], src, device)
            out[m] = to_img(l0_smooth(to_arr(cache[key]), lam, device=device)) if m.endswith("_l0") else cache[key]
    return out


def sheet(src, res, path, lam=L0):
    items = [("original", src)] + [(k, res[k]) for k in SHEET_ORDER[1:]]
    w, h = src.size
    g, top, cols = 20, max(40, w // 18), 3
    rows = (len(items) + cols - 1) // cols
    s = Image.new("RGB", (cols * (w + g) - g, rows * (h + top + g)), (60, 60, 60))
    d = ImageDraw.Draw(s)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", top // 2)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", top // 2)
        except OSError:
            font = ImageFont.load_default()
    for i, (k, im) in enumerate(items):
        x, y = (i % cols) * (w + g), (i // cols) * (h + top + g)
        s.paste(im, (x, y + top))
        d.text((x + 12, y + top // 5), LABELS[k].format(lam=lam), fill="white", font=font)
    s.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="images or folders")
    ap.add_argument("--out", help="output folder (default: <image folder>/<stem>_denoise)")
    ap.add_argument("--all", action="store_true", help="write all 5 candidates and a comparison sheet")
    ap.add_argument("--method", default="esrgan60_l0", choices=METHODS)
    ap.add_argument("--l0", type=float, default=L0,
                    help="Finite positive L0 strength for *_l0 methods (default 0.001, try 0.0005-0.002)")
    ap.add_argument("--models", default=str(model_root()))
    ap.add_argument("--device", default="auto", help=DEVICE_HELP)
    a = ap.parse_args()
    try:
        _validate_l0_strength(a.l0)
    except ValueError:
        ap.error("--l0 must be a finite positive value")
    try:
        device = resolve_device(a.device)
    except (ValueError, RuntimeError) as error:
        ap.error(str(error))

    files = []
    for p in map(Path, a.inputs):
        files += sorted(q for q in p.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) if p.is_dir() else [p]
    methods = METHODS if a.all else [a.method]
    need = {m.split("60")[0] if "60" in m else m for m in methods if not m.startswith("l0_")}
    models = {k: load_model(k, Path(a.models), device) for k in need}

    for p in files:
        src = Image.open(p).convert("RGB")
        out = Path(a.out) if a.out else p.parent / f"{p.stem}_denoise"
        out.mkdir(parents=True, exist_ok=True)
        res = run(src, methods, models, device, a.l0)
        for k, im in res.items():
            im.save(out / f"{p.stem}_{k}.png")
        if a.all:
            sheet(src, res, out / f"{p.stem}_sheet.png", a.l0)
        print(f"{p} -> {out}", flush=True)


if __name__ == "__main__":
    main()
