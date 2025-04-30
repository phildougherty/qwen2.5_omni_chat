import torch
import logging
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import numpy as np
import soundfile as sf
import io
import base64
from .config import settings
import time
import traceback
import os
import tempfile
import shutil
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenOmniModel:
    def __init__(self):
        model_path = settings.get_model_path()
        logger.info(f"Initializing Qwen2.5-Omni {settings.MODEL_SIZE} model from {model_path}...")
        logger.info(f"Using device map: {settings.DEVICE_MAP}")
        logger.info(f"Using torch dtype: {settings.TORCH_DTYPE}")
        logger.info(f"Using attention implementation: {settings.ATTN_IMPLEMENTATION}")
        
        # Check if model is downloaded
        from huggingface_hub import try_to_load_from_cache
        from os.path import isfile
        
        model_file = try_to_load_from_cache(model_path, "config.json")
        if not model_file or not isfile(model_file):
            logger.warning(f"Model {model_path} is not downloaded yet.")
            logger.info("Models will be downloaded on first use, which may cause a delay.")
        else:
            logger.info(f"Model {model_path} files found in cache.")
        
        self.model = None
        self.processor = None
        self.initialized = False

    def load_model(self):
        try:
            start_time = time.time()
            # Log the available GPU memory before loading
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info()
                logger.info(f"GPU memory before loading: {free_mem/(1024**3):.2f} GB free out of {total_mem/(1024**3):.2f} GB total")
            
            # Get the full model path based on size
            model_path = settings.get_model_path()
            logger.info(f"Loading model from: {model_path} (size: {settings.MODEL_SIZE})")
            
            # Load processor first
            logger.info("Loading processor...")
            self.processor = Qwen2_5OmniProcessor.from_pretrained(
                model_path,
                model_max_length=settings.MODEL_MAX_LENGTH,
            )
            logger.info(f"Processor loaded successfully in {time.time() - start_time:.2f} seconds")
            
            # Then load model
            load_start_time = time.time()
            logger.info("Loading model...")
            # Print what attention implementation we're using
            logger.info(f"Using attention implementation: {settings.ATTN_IMPLEMENTATION}")
            self.model = Qwen2_5OmniModel.from_pretrained(
                model_path,
                torch_dtype=settings.TORCH_DTYPE,
                device_map=settings.DEVICE_MAP,
                attn_implementation=settings.ATTN_IMPLEMENTATION,
                enable_audio_output=settings.ENABLE_AUDIO_OUTPUT
            )
            logger.info(f"Model loaded successfully in {time.time() - load_start_time:.2f} seconds")
            
            # Log the available GPU memory after loading
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info()
                logger.info(f"GPU memory after loading: {free_mem/(1024**3):.2f} GB free out of {total_mem/(1024**3):.2f} GB total")
                logger.info(f"Model device: {next(self.model.parameters()).device}")
            
            if hasattr(self.model, 'config'):
                logger.info(f"Audio output enabled: {self.model.config.enable_audio_output}")
            
            self.initialized = True
            logger.info(f"Total initialization time: {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise

    def save_audio_to_temp_file(self, audio_data, sample_rate=24000):
        """Save numpy audio data to a temporary file."""
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_path = temp_file.name
            temp_file.close()
            
            # Save audio data to the file
            if audio_data.dtype == np.float16:
                # Convert to float32 for soundfile
                audio_data = audio_data.astype(np.float32)
                
            sf.write(temp_path, audio_data, sample_rate)
            
            logger.info(f"Saved audio to temporary file: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Error saving audio to temp file: {e}")
            return None

    def convert_conversation_for_processing(self, conversation):
        """Convert conversation format to be compatible with process_mm_info."""
        converted = []
        temp_files = []
        
        for message in conversation:
            new_message = {"role": message["role"]}
            content = message.get("content", "")
            
            # Handle multimodal content
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "audio" and "audio" in item:
                        audio_data = item["audio"]
                        if isinstance(audio_data, np.ndarray):
                            # Save audio data to temp file
                            temp_path = self.save_audio_to_temp_file(audio_data)
                            if temp_path:
                                temp_files.append(temp_path)
                                # Add path instead of numpy array
                                new_content.append({"type": "audio", "audio": temp_path})
                            else:
                                # If save failed, skip this item
                                logger.warning("Skipping audio due to save failure")
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                new_message["content"] = new_content
            else:
                new_message["content"] = content
                
            converted.append(new_message)
            
        return converted, temp_files

    def cleanup_temp_files(self, temp_files):
        """Clean up temporary files."""
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Error removing temp file {file_path}: {e}")

    async def generate_response(self, conversation):
        """
        Generate response from the model based on the conversation history.
        Following Qwen2.5-Omni's implementation for audio input/output.
        """
        if not self.initialized:
            logger.info("Model not initialized. Loading model...")
            self.load_model()
            logger.info("Model loaded successfully.")
        
        temp_files = []
        try:
            logger.info("Generating response...")
            gen_start_time = time.time()
            
            # Convert conversation to be compatible with process_mm_info
            logger.info("Converting conversation format...")
            converted_conversation, temp_files = self.convert_conversation_for_processing(conversation)
            
            # Prepare the conversation for the model
            logger.info("Applying chat template...")
            text = self.processor.apply_chat_template(
                converted_conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            logger.info(f"Chat template applied: length={len(text)}")
            
            # Process multimodal information
            logger.info("Processing multimodal information...")
            try:
                audios, images, videos = process_mm_info(
                    converted_conversation,
                    use_audio_in_video=settings.USE_AUDIO_IN_VIDEO
                )
                logger.info(f"Multimodal info processed: audios={len(audios) if audios else 0}, images={len(images) if images else 0}, videos={len(videos) if videos else 0}")
            except Exception as e:
                logger.error(f"Error processing multimodal info: {e}")
                logger.error(traceback.format_exc())
                audios, images, videos = [], [], []
            
            # Encode inputs
            logger.info("Encoding inputs...")
            inputs = self.processor(
                text=text,
                audios=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True
            )
            logger.info("Inputs encoded successfully")
            
            # IMPORTANT: Force conversion of inputs to float16 before moving to device
            logger.info("Converting inputs to float16")
            for k in inputs:
                if torch.is_tensor(inputs[k]) and inputs[k].dtype == torch.float32:
                    inputs[k] = inputs[k].to(dtype=torch.float16)
            
            # Move inputs to the appropriate device
            logger.info(f"Moving inputs to device: {self.model.device}")
            inputs = inputs.to(self.model.device)
            
            # Ensure audio output is enabled
            logger.info(f"Audio output enabled: {settings.ENABLE_AUDIO_OUTPUT}")
            
            # Generate response
            logger.info(f"Generating with model using voice: {settings.DEFAULT_VOICE}...")
            text_ids, audio_output = self.model.generate(
                **inputs,
                use_audio_in_video=settings.USE_AUDIO_IN_VIDEO,
                spk=settings.DEFAULT_VOICE,
            )
            logger.info("Model generation completed")
            
            # Decode text response
            logger.info("Decoding text response...")
            response_text = self.processor.batch_decode(
                text_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            logger.info(f"Text decoded, length={len(response_text)}")
            
            # Get the assistant's portion of the response (after the prompt)
            if "<|assistant|>" in response_text:
                response_text = response_text.split("<|assistant|>", 1)[1].strip()
                logger.info("Extracted assistant portion of response")
            
            # Process audio output
            audio_data = None
            if audio_output is not None:
                logger.info("Processing audio output...")
                # Convert tensor to numpy array
                audio_np = audio_output.reshape(-1).detach().cpu().numpy()
                logger.info(f"Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}")
                
                # Create WAV file in memory
                audio_bytes_io = io.BytesIO()
                sf.write(audio_bytes_io, audio_np, samplerate=settings.SAMPLE_RATE, format='WAV')
                audio_bytes_io.seek(0)
                audio_data = base64.b64encode(audio_bytes_io.read()).decode('utf-8')
                logger.info(f"Audio processed, size={len(audio_data)} bytes")
                
                # Free memory
                del audio_np
                del audio_output
            else:
                logger.warning("No audio output generated! Check model configuration.")
            
            logger.info(f"Response generated in {time.time() - gen_start_time:.2f} seconds")
            return {
                "text": response_text,
                "audio": audio_data
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
        finally:
            # Clean up temporary files
            if temp_files:
                self.cleanup_temp_files(temp_files)
                
# Singleton instance
model = QwenOmniModel()