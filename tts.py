from typing import Optional
from engine import index

_cached_tts_model = None
_current_model_name = None


def generate_audio(
    text: str,
    voice: str,
    output_format: str,
    model: str,
    speed: float = 1.0,
    chunk_size: int = 250,
    seed: int = 0,
    params: dict = {},
) -> Optional[bytes]:
    global _cached_tts_model, _current_model_name

    if _current_model_name != model:
        available_engines = index.get_available_model_engines()

        if model in available_engines:
            print(f"Loading model: {model}")
            loaded_engine = available_engines[model]()
        else:
            raise ValueError(f"Unknown model: {model}")

        _cached_tts_model = loaded_engine
        _current_model_name = model
    else:
        print(f"Using cached model: {model}")

    tts_model = _cached_tts_model

    return tts_model.generate_audio(
        text, voice, output_format, speed, chunk_size, seed, params
    )
