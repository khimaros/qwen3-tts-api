import utils
import config
import numpy as np
from typing import Optional
from engine.tts_engine import TTSEngine


class Qwen317VoiceDesignTTSEngine(TTSEngine):
    def __init__(self):
        from qwen_tts import Qwen3TTSModel
        import torch

        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
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

        # Generate the waveform
        wavs, sample_rate = self.model.generate_voice_design(
            text=chunks,
            audio_prompt_path=voice_file,
            # language="English",
            instruct=params.get("instruct", "Speak in a plain voice."),
        )

        all_audio_data = np.concatenate(wavs)
        bytes_object = utils.encode_audio(all_audio_data, sample_rate, output_format)

        return bytes_object


# single inference
# wavs, sr = model.generate_voice_design(
#     text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
#     language="Chinese",
#     instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
# )
# sf.write("output_voice_design.wav", wavs[0], sr)

# # batch inference
# wavs, sr = model.generate_voice_design(
#     text=[
#       "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
#       "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
#     ],
#     language=["Chinese", "English"],
#     instruct=[
#       "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
#       "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
#     ]
# )
# sf.write("output_voice_design_1.wav", wavs[0], sr)
# sf.write("output_voice_design_2.wav", wavs[1], sr)
