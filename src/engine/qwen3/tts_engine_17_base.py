import utils
import config
import numpy as np
from typing import Optional
from engine.tts_engine import TTSEngine


class Qwen317BaseTTSEngine(TTSEngine):
    def __init__(self):
        from qwen_tts import Qwen3TTSModel
        import torch

        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

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
        if seed != 0:
            utils.set_seed(seed)  # For reproducibility

        voice_file = config.VOICES_DIR + f"{voice}.wav"

        all_audio_data = []

        chunks = utils.chunk_text_by_sentences(text, chunk_size)

        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=voice_file,
            # ref_text=ref_text,
            x_vector_only_mode=True,
        )

        # Generate the waveform
        wavs, sample_rate = self.model.generate_voice_clone(
            text=chunks,
            # language="English",
            voice_clone_prompt=prompt_items,
        )

        all_audio_data = np.concatenate(wavs)
        bytes_object = utils.encode_audio(all_audio_data, sample_rate, output_format)

        return bytes_object
