from typing import ClassVar, Optional, Dict, Any, Literal
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    # Model settings
    MODEL_PATH: str = "Qwen/Qwen2.5-Omni-7B"
    MODEL_SIZE: Literal["3B", "7B"] = "7B"
    DEVICE_MAP: str = "auto"  # Make sure this is "auto" for GPU
    TORCH_DTYPE: ClassVar = torch.float16 if torch.cuda.is_available() else torch.float32
    ATTN_IMPLEMENTATION: str = "flash_attention_2" if torch.cuda.is_available() else "sdpa"  # Use Flash Attention if GPU available
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
        
    def get_model_path(self) -> str:
        """Get the full model path based on selected size"""
        if self.MODEL_SIZE == "3B" and "Qwen2.5-Omni" in self.MODEL_PATH:
            # Replace 7B with 3B in the model path if using the default path
            return self.MODEL_PATH.replace("7B", "3B")
        return self.MODEL_PATH

settings = Settings()