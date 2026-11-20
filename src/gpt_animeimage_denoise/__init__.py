"""GPT AnimeImage Denoise, a local region-aware denoising pipeline."""
import os


# Set before importing PyTorch so MIOpen can skip convolution autotuning.
os.environ.setdefault('MIOPEN_FIND_MODE', 'FAST')

__version__ = '2.1.0'
