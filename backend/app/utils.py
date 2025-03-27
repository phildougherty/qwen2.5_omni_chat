import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import logging
import librosa

logger = logging.getLogger(__name__)

def log_audio_info(audio_np):
    """Log detailed information about the audio array."""
    try:
        logger.info(f"Audio shape: {audio_np.shape}")
        logger.info(f"Audio dtype: {audio_np.dtype}")
        logger.info(f"Audio min/max: {np.min(audio_np)}/{np.max(audio_np)}")
        logger.info(f"Audio non-zero elements: {np.count_nonzero(audio_np)}/{audio_np.size}")
        if np.isnan(audio_np).any():
            logger.warning("Audio contains NaN values!")
        if np.isinf(audio_np).any():
            logger.warning("Audio contains infinity values!")
        first_values = audio_np[:10].tolist()
        logger.info(f"First 10 values: {first_values}")
    except Exception as e:
        logger.error(f"Error logging audio info: {e}")

def validate_audio(audio_bytes):
    """
    Validate and convert audio to the format expected by the model.
    """
    try:
        logger.info(f"Validating audio of size {len(audio_bytes)} bytes")
        
        # Try reading as WAV first
        audio_io = io.BytesIO(audio_bytes)
        audio_np, sr = sf.read(audio_io)
        logger.info(f"Read audio: shape={audio_np.shape}, sample_rate={sr}")
        
        # Convert to mono if stereo
        if len(audio_np.shape) > 1 and audio_np.shape[1] > 1:
            logger.info(f"Converting {audio_np.shape[1]} channels to mono")
            audio_np = np.mean(audio_np, axis=1)
        
        # Convert sample rate if needed
        if sr != 24000:
            logger.info(f"Converting sample rate from {sr} to 24000")
            try:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)
                logger.info(f"Resampled audio: shape={audio_np.shape}")
            except Exception as resample_error:
                logger.error(f"Error resampling with librosa: {resample_error}")
                audio_np = convert_audio_to_24khz(audio_bytes)
        
        # Log audio info for debugging
        log_audio_info(audio_np)
        
        return audio_np
    except Exception as e:
        logger.warning(f"Error with sf.read, trying pydub: {e}")
        
        # Try with pydub if sf.read fails (handles more formats)
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            # Convert to 24kHz mono
            audio_segment = audio_segment.set_channels(1).set_frame_rate(24000)
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            # Convert to float32 in [-1.0, 1.0]
            max_val = float(2**(8 * audio_segment.sample_width - 1))
            samples = samples.astype(np.float32) / max_val
            
            logger.info(f"Processed audio with pydub: shape={samples.shape}")
            
            # Log audio info for debugging
            log_audio_info(samples)
            
            return samples
        except Exception as nested_e:
            logger.error(f"Failed to process audio with pydub: {nested_e}")
            raise

def convert_audio_to_24khz(audio_bytes):
    """
    Convert audio to 24kHz sample rate, which is what Qwen2.5-Omni expects.
    """
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(24000)
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())
        # Convert to float32 in [-1.0, 1.0]
        max_val = float(2**(8 * audio_segment.sample_width - 1))
        samples = samples.astype(np.float32) / max_val
        return samples
    except Exception as e:
        logger.error(f"Error converting audio to 24kHz: {e}")
        raise