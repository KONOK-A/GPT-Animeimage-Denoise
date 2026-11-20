"""Resolve external model assets for both source and installed entry points."""
import os
from pathlib import Path


MODEL_ENV = 'GPT_ANIMEIMAGE_DENOISE_MODEL_DIR'


def model_root():
    """Read the configured asset directory at invocation time."""
    configured = os.environ.get(MODEL_ENV)
    return Path(configured or 'models').expanduser().resolve()


def part_models_present():
    """Check for the main segmentation files without importing the inference runtime."""
    root = model_root()
    return all((root / name).is_file() for name in (
        'v2/asset_hashes.json', 'v2/skytnt/isnet.py', 'v2/skytnt/model.safetensors',
        'v2/see_through/checkpoint-18000.pt', 'v2/sam2/sam2.1_hiera_large.pt',
        'AnimeSeg/base-configs/large/config.json', 'AnimeSeg/models/anime_seg_mask2former_v4.safetensors',
    ))
