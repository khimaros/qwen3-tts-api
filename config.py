import os
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()


def is_rocm() -> bool:
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def setup_rocm_env():
    """set environment variables for optimal rocm performance."""
    if not is_rocm():
        return
    env = {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
        "MIOPEN_WORKSPACE_MAX": "256000000",
        "MIOPEN_FIND_MODE": "FAST",
        "MIOPEN_USER_DB_PATH": os.path.expanduser("~/.cache/miopen"),
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)


setup_rocm_env()

VOICES_DIR = os.getenv("VOICES_DIR", "./voices")

# check if path exists
if not os.path.exists(VOICES_DIR):
    raise ValueError(f"Path {VOICES_DIR} does not exist")

if VOICES_DIR[-1] != "/":
    VOICES_DIR += "/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_PORT = os.getenv("API_PORT", "5001")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
MODEL = os.getenv("MODEL", "Qwen3-TTS-12Hz-1.7B-Base")
SUPPORTED_VOICES = os.getenv("SUPPORTED_VOICES", "").split(",")
SUPPORTED_RESPONSE_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
CORS_ALLOWED_ORIGIN = os.getenv("CORS_ALLOWED_ORIGIN", "*")
SEED = int(os.getenv("SEED", 0))

# if SUPPORTED_VOICES is empty, then we will use all voices in the VOICES_DIR directory
if SUPPORTED_VOICES == [""]:
    print("No voices specified, using all voices in the VOICES_DIR directory")

    SUPPORTED_VOICES = [f[:-4] for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]

    print(f"Found {len(SUPPORTED_VOICES)} voices in the VOICES_DIR directory")

print(f"🚀 Running on device: {DEVICE}")

if SEED != 0:
    import utils

    utils.set_seed(SEED)  # For reproducibility
