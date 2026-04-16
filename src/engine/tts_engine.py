from contextlib import nullcontext
from typing import Optional

import torch

import config


def rocm_sdpa_context():
    """on rocm, select sdp backends explicitly with MATH as fallback."""
    if not config.is_rocm():
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ])
    except ImportError:
        return nullcontext()


def inference_context():
    """combined inference context: inference_mode + rocm sdpa kernel selection."""
    return torch.inference_mode()


class TTSEngine:
    def __init__(self):
        raise NotImplementedError("Subclasses must implement this method")

    def generate_audio(
        self,
        text: str,
        voice: str,
        output_format: str,
        speed: float = 1.0,
        chunk_size: int = 250,
        seed: int = 0,
        params: dict = {},
    ) -> Optional[bytes]:
        raise NotImplementedError("Subclasses must implement this method")
