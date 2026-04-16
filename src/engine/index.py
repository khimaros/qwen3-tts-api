import config
from engine.tts_engine import TTSEngine
from engine.qwen3.tts_engine_17_voice_design import Qwen317VoiceDesignTTSEngine
from engine.qwen3.tts_engine_17_base import Qwen317BaseTTSEngine
from engine.qwen3.tts_engine_06_base import Qwen306BaseTTSEngine


def attn_implementation() -> str:
    """use sdpa on rocm, flash_attention_2 on cuda when available, sdpa otherwise."""
    if config.is_rocm():
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def get_available_model_engines() -> dict[TTSEngine]:
    """
    Returns a list of available model engines.
    """
    return {
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign": Qwen317VoiceDesignTTSEngine,
        "Qwen3-TTS-12Hz-1.7B-Base": Qwen317BaseTTSEngine,
        "Qwen3-TTS-12Hz-0.6B-Base": Qwen306BaseTTSEngine,
    }
