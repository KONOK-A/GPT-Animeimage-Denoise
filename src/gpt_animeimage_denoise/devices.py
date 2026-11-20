"""PyTorch device selection shared by commands and the workbench."""
from contextlib import contextmanager
import re


DEVICE_HELP = 'auto, cpu, cuda, cuda:N or mps (ROCm uses cuda)'
_DEVICE_NAME = re.compile(r'(?:cpu|mps|cuda(?::(?:0|[1-9][0-9]*))?)')


def is_device_name(device):
    """Check a saved device without requiring that backend on this machine."""
    return isinstance(device, str) and _DEVICE_NAME.fullmatch(device) is not None


def resolve_device(device='auto'):
    if device != 'auto' and not is_device_name(device):
        raise ValueError('device must be ' + DEVICE_HELP)
    if device == 'cpu':
        return device

    import torch

    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable. Select --device cpu.')
        return device

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/ROCm is unavailable. Select --device cpu.')
    _, _, index_text = device.partition(':')
    if index_text:
        index = int(index_text)
        # PyTorch can wrap large indices while parsing the device string.
        if index >= torch.cuda.device_count() or torch.device(device).index != index:
            raise ValueError(f'CUDA/ROCm device {index} is unavailable')
    return device


def uses_rocm(device):
    if str(device).partition(':')[0] != 'cuda':
        return False

    import torch

    return bool(torch.version.hip)


def clear_accelerator_cache(device):
    kind = str(device).partition(':')[0]
    if kind in ('cuda', 'mps'):
        import torch

        getattr(torch, kind).empty_cache()


@contextmanager
def fp32_context(device):
    """Disable autocast and preserve the existing CUDA precision controls."""
    import torch

    kind = torch.device(device).type
    with torch.autocast(kind, enabled=False):
        if kind == 'mps' or uses_rocm(device):
            yield
            return
        matmul = torch.backends.cuda.matmul.allow_tf32
        cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
