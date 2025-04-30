import os
import logging
import huggingface_hub
import time
from huggingface_hub import snapshot_download, hf_hub_download

logger = logging.getLogger(__name__)

def get_model_path_for_size(size):
    """Get the model path for a specific size."""
    if size == "3B":
        return "Qwen/Qwen2.5-Omni-3B"
    else:
        return "Qwen/Qwen2.5-Omni-7B"

def check_model_download_progress(model_path):
    """Check download progress for model files."""
    try:
        # Get model cache directory
        cache_dir = huggingface_hub.constants.HF_HUB_CACHE
        model_id = model_path.split("/")[-1]
        snapshots_dir = os.path.join(cache_dir, "models--" + model_path.replace('/', '--'))
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

def download_model(model_name, revision=None):
    """Download model files from Hugging Face Hub."""
    try:
        logger.info(f"Downloading model: {model_name}, revision: {revision or 'default'}")
        cache_dir = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        
        # First download model configuration and tokenizer (which are smaller)
        hf_hub_download(model_name, "config.json", revision=revision)
        hf_hub_download(model_name, "tokenizer_config.json", revision=revision)
        
        # Then download the full model snapshot (can be large)
        start_time = time.time()
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_files_only=False,
            resume_download=True
        )
        download_time = time.time() - start_time
        logger.info(f"Successfully downloaded {model_name} in {download_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        return False
