from typing import Optional


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
