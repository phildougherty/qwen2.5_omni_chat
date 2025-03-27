import os
import logging
import numpy as np
import soundfile as sf
import tempfile
import shutil
from scipy import signal
import librosa

logger = logging.getLogger(__name__)

def resample_audio(audio_data, orig_sr, target_sr=24000):
    """Resample audio to target sample rate."""
    try:
        # First try using librosa which generally has better quality
        return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
    except:
        # Fall back to scipy if librosa fails
        return signal.resample_poly(
            audio_data, 
            target_sr, 
            orig_sr, 
            axis=0
        ).astype(np.float32)

def process_audio_data(conversation):
    """
    Process audio data from conversations in a way that works with Qwen Omni.
    This is a modified version that handles NumPy arrays directly.
    """
    audios = []
    
    # Create a temporary directory for audio files
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for audio files: {temp_dir}")
    
    try:
        # Iterate through conversation to find audio content
        for message in conversation:
            content = message.get("content", [])
            
            # Handle list content (multimodal)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "audio" and "audio" in item:
                        audio_data = item["audio"]
                        
                        # Handle NumPy array directly
                        if isinstance(audio_data, np.ndarray):
                            logger.info(f"Processing NumPy audio array: shape={audio_data.shape}, dtype={audio_data.dtype}")
                            
                            # Ensure audio is at 24kHz sample rate
                            # Assume audio is already at 24kHz as we validate this elsewhere
                            
                            # Save to temporary file
                            temp_path = os.path.join(temp_dir, f"audio_{len(audios)}.wav")
                            sf.write(temp_path, audio_data, 24000)  # Already at 24kHz
                            
                            # Add to audios list
                            audios.append(temp_path)
                            logger.info(f"Saved audio to temporary file: {temp_path}")
    
        return audios, temp_dir
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Clean up temp directory on error
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        return [], None

def cleanup_temp_files(temp_dir):
    """Clean up temporary files after processing."""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {e}")