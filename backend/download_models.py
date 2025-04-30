#!/usr/bin/env python3
"""
Script to download Qwen Omni models in the background.
This can be run as a separate process during container initialization.
"""

import logging
import os
import sys
import time
from huggingface_hub import snapshot_download, hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_downloader")

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

if __name__ == "__main__":
    # Allow specifying model size as command line argument
    model_size = "7B"  # Default
    if len(sys.argv) > 1 and sys.argv[1] in ["3B", "7B"]:
        model_size = sys.argv[1]
    
    # Allow downloading both models
    download_all = False
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        download_all = True
    
    if download_all:
        logger.info("Downloading both 3B and 7B models...")
        download_model("Qwen/Qwen2.5-Omni-3B")
        download_model("Qwen/Qwen2.5-Omni-7B")
    else:
        # Download specific model
        model_path = f"Qwen/Qwen2.5-Omni-{model_size}"
        logger.info(f"Downloading model: {model_path}")
        download_model(model_path)
