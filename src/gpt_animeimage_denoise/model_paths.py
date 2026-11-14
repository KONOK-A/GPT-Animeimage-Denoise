"""Resolve external model assets for both source and installed entry points."""
import os
from pathlib import Path


def model_root():
    """Read the configured asset directory at invocation time."""
    configured = os.environ.get('GPT_ANIMEIMAGE_DENOISE_MODEL_DIR')
    return Path(configured or 'models').expanduser().resolve()
