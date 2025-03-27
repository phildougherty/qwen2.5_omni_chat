from typing import ClassVar, Optional, Dict, Any
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    # Model settings
    MODEL_PATH: str = "Qwen/Qwen2.5-Omni-7B"
    DEVICE_MAP: str = "auto"
    TORCH_DTYPE: ClassVar = torch.float16 if torch.cuda.is_available() else torch.float32
    ATTN_IMPLEMENTATION: str = "sdpa"
    DEFAULT_VOICE: str = "Chelsie"
    SAMPLE_RATE: int = 24000
    ENABLE_AUDIO_OUTPUT: bool = True
    USE_AUDIO_IN_VIDEO: bool = True
    MODEL_MAX_LENGTH: int = 8192
    DEBUG: bool = True
    
    # Memory management settings
    MAX_CONVERSATION_TURNS: int = 10
    CLEANUP_AFTER_RESPONSE: bool = True
    PYTORCH_CUDA_ALLOC_CONF: str = "expandable_segments:True" 
    
    class Config:
        env_file = ".env"
        arbitrary_types_allowed = True

settings = Settings()