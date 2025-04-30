import logging
import time
import torch
import os
import sys
from app.model import model
from app.config import settings
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
import argparse
import huggingface_hub
# Import from the new module
from app.model_utils import check_model_download_progress, get_model_path_for_size, download_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_status():
    """Check GPU status and memory."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = []
        for i in range(device_count):
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                devices.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "free_memory_gb": f"{free_mem/(1024**3):.2f}",
                    "total_memory_gb": f"{total_mem/(1024**3):.2f}",
                    "memory_allocated_gb": f"{torch.cuda.memory_allocated(i)/(1024**3):.2f}",
                })
            except Exception as e:
                devices.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "error": str(e)
                })
        return {"available": True, "device_count": device_count, "devices": devices}
    else:
        return {"available": False, "reason": "CUDA not available"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model warmup and download utility")
    parser.add_argument("--download-only", action="store_true", help="Only download the models without loading")
    parser.add_argument("--download-all", action="store_true", help="Download both 3B and 7B models")
    parser.add_argument("--size", choices=["3B", "7B"], default=settings.MODEL_SIZE, 
                         help="Model size to download/load")
    args = parser.parse_args()
    
    logger.info("Starting model preparation...")
    
    # Check GPU status
    logger.info("Checking GPU status...")
    gpu_status = check_gpu_status()
    logger.info(f"GPU status: {gpu_status}")
    
    # Handle model downloading
    if args.download_all:
        logger.info("Downloading both 3B and 7B models...")
        download_model("Qwen/Qwen2.5-Omni-3B")
        download_model("Qwen/Qwen2.5-Omni-7B")
        if args.download_only:
            logger.info("Download complete. Exiting.")
            sys.exit(0)
    else:
        # If not download-only or downloading specific size
        model_path = get_model_path_for_size(args.size)
        logger.info(f"Working with model: {model_path}")
        
        # Check model download status
        logger.info("Checking model download status...")
        model_status = check_model_download_progress(model_path)
        logger.info(f"Model status: {model_status}")
        
        if model_status["status"] != "found":
            logger.info(f"Model {model_path} not fully downloaded. Starting download...")
            download_success = download_model(model_path)
            if not download_success and not args.download_only:
                logger.error(f"Failed to download model {model_path}. Cannot proceed with warmup.")
                sys.exit(1)
            elif args.download_only:
                logger.info("Download complete. Exiting.")
                sys.exit(0)
    
    # Only load the model if we're not in download-only mode
    if not args.download_only:
        # Set correct model size in settings
        settings.MODEL_SIZE = args.size
        # Get correct model path based on size
        model_path = settings.get_model_path()
        
        logger.info(f"Warming up model: {model_path}")
        logger.info(f"Device map: {settings.DEVICE_MAP}")
        logger.info(f"Torch dtype: {settings.TORCH_DTYPE}")
        logger.info(f"Attention implementation: {settings.ATTN_IMPLEMENTATION}")
        
        # Load model
        try:
            start_time = time.time()
            logger.info("Loading model...")
            model.load_model()
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
            # Check GPU status after loading
            logger.info("Checking GPU status after model loading...")
            gpu_status = check_gpu_status()
            logger.info(f"GPU status after loading: {gpu_status}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise