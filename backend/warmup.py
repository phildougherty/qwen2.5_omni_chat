import logging
import time
import torch
from app.model import model
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_download_progress():
    """Check download progress for model files."""
    import os
    import huggingface_hub
    
    try:
        # Get model cache directory
        cache_dir = huggingface_hub.constants.HF_HUB_CACHE
        model_id = settings.MODEL_PATH.split("/")[-1]
        snapshots_dir = os.path.join(cache_dir, "models--" + settings.MODEL_PATH.replace('/', '--'))
        
        if os.path.exists(snapshots_dir):
            refs_dir = os.path.join(snapshots_dir, "refs")
            if os.path.exists(refs_dir):
                refs = os.listdir(refs_dir)
                if refs:
                    # There's at least one downloaded version
                    logger.info(f"Found model snapshots: {refs}")
                    return {"status": "found", "refs": refs}
            
            # Snapshots directory exists but no refs
            logger.info(f"Model directory exists but no refs found: {snapshots_dir}")
            return {"status": "initializing"}
        else:
            # No snapshots directory
            logger.info(f"Model directory not found, will be downloaded: {snapshots_dir}")
            return {"status": "not_downloaded"}
            
    except Exception as e:
        logger.error(f"Error checking model progress: {e}")
        return {"status": "error", "error": str(e)}

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

logger.info("Starting model warmup...")
logger.info(f"Model path: {settings.MODEL_PATH}")
logger.info(f"Device map: {settings.DEVICE_MAP}")
logger.info(f"Torch dtype: {settings.TORCH_DTYPE}")
logger.info(f"Attention implementation: {settings.ATTN_IMPLEMENTATION}")

# Check GPU status
logger.info("Checking GPU status...")
gpu_status = check_gpu_status()
logger.info(f"GPU status: {gpu_status}")

# Check model download status
logger.info("Checking model download status...")
model_status = check_model_download_progress()
logger.info(f"Model status: {model_status}")

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
