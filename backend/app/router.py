from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Body, HTTPException
from fastapi.responses import JSONResponse
import base64
import json
import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import time
import logging
import torch
import os
import traceback
import tempfile
import gc
import asyncio
from app.model_utils import check_model_download_progress, get_model_path_for_size
from .model import model
from .utils import validate_audio, convert_audio_to_24khz
from .config import settings
from huggingface_hub import snapshot_download, hf_hub_download

router = APIRouter()
logger = logging.getLogger(__name__)

# Configure CUDA memory allocation if the setting exists
if hasattr(settings, 'PYTORCH_CUDA_ALLOC_CONF'):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = settings.PYTORCH_CUDA_ALLOC_CONF
    logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF to {settings.PYTORCH_CUDA_ALLOC_CONF}")

# Store active websocket connections and conversation histories
active_connections = {}
conversations = {}

# Conversation history management function
def manage_conversation_history(conversation_history):
    """Limits conversation history to prevent OOM errors."""
    if len(conversation_history) > settings.MAX_CONVERSATION_TURNS + 1:  # +1 for system message
        # Keep system message and recent turns
        system_message = conversation_history[0]
        recent_turns = conversation_history[-(settings.MAX_CONVERSATION_TURNS):]
        logger.info(f"Trimmed conversation from {len(conversation_history)} to {len([system_message]) + len(recent_turns)} messages")
        return [system_message] + recent_turns
    return conversation_history

@router.get("/model/status")
async def model_status():
    """Get status of model downloads and current model."""
    # Check 3B model
    model_3b_path = get_model_path_for_size("3B")
    model_3b_status = check_model_download_progress(model_3b_path)
    
    # Check 7B model
    model_7b_path = get_model_path_for_size("7B")
    model_7b_status = check_model_download_progress(model_7b_path)
    
    # Get current model info
    current_model_info = {
        "path": settings.get_model_path(),
        "size": settings.MODEL_SIZE,
        "loaded": model.initialized
    }
    
    return {
        "current_model": current_model_info,
        "models": {
            "3B": model_3b_status,
            "7B": model_7b_status
        }
    }

@router.post("/model/download")
async def download_model_endpoint(size: str = "7B"):
    """Trigger download for a specific model size."""
    if size not in ["3B", "7B"]:
        raise HTTPException(status_code=400, detail="Invalid model size. Must be '3B' or '7B'")
    
    model_path = get_model_path_for_size(size)
    
    # Check if already downloaded
    status = check_model_download_progress(model_path)
    if status["status"] == "found":
        return {"status": "already_downloaded", "model": model_path}
    
    # Start download in background
    async def download_task():
        try:
            logger.info(f"Starting download for model: {model_path}")
            huggingface_hub.snapshot_download(
                repo_id=model_path,
                local_files_only=False,
                resume_download=True
            )
            logger.info(f"Download completed for model: {model_path}")
        except Exception as e:
            logger.error(f"Error downloading model {model_path}: {e}")
    
    # We use a background task here
    asyncio.create_task(download_task())
    
    return {"status": "download_started", "model": model_path}

@router.get("/health")
async def health_check():
    """Health check endpoint with detailed information."""
    status = {
        "status": "ok",
        "model_loaded": model.initialized,
        "model_size": settings.MODEL_SIZE,
        "timestamp": time.time(),
    }

    # Add model information if available
    if model.model is not None:
        try:
            status["model_info"] = {
                "model_id": getattr(model.model.config, "_name_or_path", "unknown"),
                "model_type": getattr(model.model.config, "model_type", "unknown"),
                "enable_audio_output": getattr(model.model.config, "enable_audio_output", "unknown"),
            }
        except Exception as e:
            status["model_info_error"] = str(e)

    # Add GPU info if available
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            status["gpu_info"] = {
                "device": torch.cuda.get_device_name(0),
                "free_memory_gb": f"{free_mem/(1024**3):.2f}",
                "total_memory_gb": f"{total_mem/(1024**3):.2f}",
                "memory_allocated_gb": f"{torch.cuda.memory_allocated()/(1024**3):.2f}",
                "max_memory_allocated_gb": f"{torch.cuda.max_memory_allocated()/(1024**3):.2f}",
            }
        except Exception as e:
            status["gpu_info_error"] = str(e)

    # Add active connections info
    status["active_connections"] = len(active_connections)
    return status

@router.post("/chat/completions")
async def chat_completions(request: Request):
    """
    REST API endpoint that mimics OpenAI's chat completions with speech support.
    Simplified version to demonstrate the concept.
    """
    try:
        data = await request.json()
        # Extract parameters
        messages = data.get("messages", [])
        voice = data.get("voice", "Chelsie")  # Default to Chelsie voice
        
        # Convert to the format expected by our model
        conversation = []
        
        # Add system message if it exists
        system_msg = next((msg for msg in messages if msg["role"] == "system"), None)
        if system_msg:
            conversation.append({
                "role": "system",
                "content": system_msg["content"]
            })
        else:
            # Default system message for audio output
            conversation.append({
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            })
            
        # Add other messages
        for msg in messages:
            if msg["role"] == "system":
                continue  # Already handled
                
            if msg["role"] in ["user", "assistant"]:
                content = msg["content"]
                
                # Handle audio content if present
                if isinstance(content, list):
                    processed_content = []
                    for item in content:
                        if item.get("type") == "audio" and "audio" in item:
                            # Base64 audio data
                            audio_data = item["audio"]
                            
                            # Make sure it's a string
                            if isinstance(audio_data, str):
                                # Handle data URI format
                                if audio_data.startswith("data:audio/"):
                                    # Extract base64 part from data URI
                                    audio_data = audio_data.split(",")[1]
                                    
                                try:
                                    # Decode and process audio
                                    audio_bytes = base64.b64decode(audio_data)
                                    audio_np = validate_audio(audio_bytes)
                                    # Convert to format expected by Qwen model
                                    processed_content.append({"type": "audio", "audio": audio_np})
                                except Exception as e:
                                    logger.error(f"Error processing audio in REST API: {e}")
                                    logger.error(traceback.format_exc())
                                    raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")
                            else:
                                logger.warning(f"Unexpected audio data type: {type(audio_data)}")
                                processed_content.append(item)
                        else:
                            processed_content.append(item)
                    conversation.append({"role": msg["role"], "content": processed_content})
                else:
                    conversation.append({"role": msg["role"], "content": content})
                    
        # Manage conversation history to prevent OOM
        conversation = manage_conversation_history(conversation)
        
        # Generate response
        logger.info("Generating response via REST API")
        response = await model.generate_response(conversation)
        
        if "error" in response:
            logger.error(f"Error in generation: {response['error']}")
            raise HTTPException(status_code=500, detail=response["error"])
            
        # Format response in OpenAI-like format
        completion_response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"qwen-2.5-omni-{settings.MODEL_SIZE.lower()}", 
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response["text"]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # We don't track token usage here
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        # Add audio if available
        if response.get("audio"):
            completion_response["choices"][0]["message"]["audio"] = response["audio"]
            
        return JSONResponse(content=completion_response)
    
    except HTTPException as he:
        logger.error(f"HTTP Exception in chat completions: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info(f"New WebSocket connection: {session_id}")
    
    # Initialize conversation if needed
    if session_id not in conversations:
        conversations[session_id] = [
            {
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }
        ]
        
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                logger.info(f"Received message from {session_id}: {data.get('type', 'unknown')}")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from {session_id}")
                await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                continue
                
            # Handle different message types
            if "type" in data:
                if data["type"] == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                    
                elif data["type"] == "reset":
                    # Check if model size is specified
                    new_model_size = data.get("modelSize")
                    if new_model_size in ["3B", "7B"] and new_model_size != settings.MODEL_SIZE:
                        logger.info(f"Changing model size from {settings.MODEL_SIZE} to {new_model_size}")
                        # Update settings
                        settings.MODEL_SIZE = new_model_size
                        # Unload current model to free memory
                        model.model = None
                        model.processor = None
                        model.initialized = False
                        # Force memory cleanup
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.info("Forced memory cleanup after model size change")
                        # Load new model (will happen on next request)

                    # Reset conversation
                    conversations[session_id] = [
                        {
                            "role": "system",
                            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                        }
                    ]
                    # Force memory cleanup
                    if hasattr(settings, 'CLEANUP_AFTER_RESPONSE') and settings.CLEANUP_AFTER_RESPONSE:
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.info("Forced memory cleanup after conversation reset")
                    
                    await websocket.send_json({"type": "reset_confirmed", "modelSize": settings.MODEL_SIZE})
                    logger.info(f"Reset conversation for {session_id} with model size {settings.MODEL_SIZE}")
                    continue
                    
                elif data["type"] == "message":
                    # Handle multi-modal message
                    try:
                        # Manage conversation history to prevent OOM
                        conversations[session_id] = manage_conversation_history(conversations[session_id])
                        
                        logger.info(f"Processing multi-modal message from {session_id}")
                        
                        # Process the content array
                        content_array = data.get("content", [])
                        if not content_array:
                            logger.warning(f"Empty content array in message from {session_id}")
                            await websocket.send_json({"type": "error", "message": "Empty message content"})
                            continue
                            
                        # Convert content array to the format expected by the model
                        processed_content = []
                        temp_files = []  # Track temporary files for cleanup
                        
                        for item in content_array:
                            item_type = item.get("type")
                            if item_type == "text":
                                # Add text directly
                                processed_content.append({"type": "text", "text": item.get("text", "")})
                                
                            elif item_type == "image" and "image" in item:
                                # Process image
                                try:
                                    image_data = base64.b64decode(item["image"])
                                    # Create a temporary file to save the image
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                                        temp_path = temp_file.name
                                        temp_file.write(image_data)
                                    logger.info(f"Saved image to temporary file: {temp_path}")
                                    temp_files.append(temp_path)  # Track the temp file
                                    # Add the file path instead of raw bytes
                                    processed_content.append({"type": "image", "image": temp_path})
                                except Exception as e:
                                    logger.error(f"Error processing image: {e}")
                                    logger.error(traceback.format_exc())
                                    await websocket.send_json({"type": "error", "message": f"Image processing error: {str(e)}"})
                                    continue
                                    
                            elif item_type == "audio" and "audio" in item:
                                # Process audio
                                try:
                                    audio_data = base64.b64decode(item["audio"])
                                    audio_np = validate_audio(audio_data)
                                    processed_content.append({"type": "audio", "audio": audio_np})
                                except Exception as e:
                                    logger.error(f"Error processing audio: {e}")
                                    logger.error(traceback.format_exc())
                                    await websocket.send_json({"type": "error", "message": f"Audio processing error: {str(e)}"})
                                    continue
                                    
                            elif item_type == "video" and "video" in item:
                                # Process video
                                try:
                                    video_data = base64.b64decode(item["video"])
                                    # Create a temporary file to save the video
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                                        temp_path = temp_file.name
                                        temp_file.write(video_data)
                                    logger.info(f"Saved video to temporary file: {temp_path}")
                                    temp_files.append(temp_path)  # Track the temp file
                                    # Add the file path instead of raw bytes
                                    processed_content.append({"type": "video", "video": temp_path})
                                except Exception as e:
                                    logger.error(f"Error processing video: {e}")
                                    logger.error(traceback.format_exc())
                                    await websocket.send_json({"type": "error", "message": f"Video processing error: {str(e)}"})
                                    continue
                                    
                            elif item_type == "file" and "file" in item:
                                # Process generic file (not directly supported by Qwen, convert to text)
                                try:
                                    file_info = item["file"]
                                    file_name = file_info.get("name", "unnamed_file")
                                    file_type = file_info.get("type", "unknown")
                                    file_size = file_info.get("size", 0)
                                    # Add as text description
                                    file_desc = f"[File attached: {file_name}, type: {file_type}, size: {file_size} bytes]"
                                    processed_content.append({"type": "text", "text": file_desc})
                                except Exception as e:
                                    logger.error(f"Error processing file: {e}")
                                    logger.error(traceback.format_exc())
                                    await websocket.send_json({"type": "error", "message": f"File processing error: {str(e)}"})
                                    continue
                                    
                        # If we have processed content, add to conversation
                        if processed_content:
                            conversations[session_id].append({
                                "role": "user",
                                "content": processed_content
                            })
                            logger.info(f"Added multi-modal message to conversation")
                        else:
                            logger.warning(f"No valid content found in message")
                            await websocket.send_json({"type": "error", "message": "No valid content found in message"})
                            # Clean up any temporary files
                            if temp_files:
                                cleanup_temp_files(temp_files)
                            continue
                            
                        # Send acknowledgement
                        await websocket.send_json({"type": "processing"})
                        logger.info("Sent processing acknowledgement")
                        
                        try:
                            # Generate response
                            logger.info("Generating response...")
                            response = await model.generate_response(conversations[session_id])
                            
                            if "error" in response:
                                logger.error(f"Error generating response: {response['error']}")
                                await websocket.send_json({"type": "error", "message": response["error"]})
                                continue
                                
                            if response.get("audio") is None:
                                logger.warning("No audio in response, model may not be properly configured for audio output")
                            else:
                                logger.info(f"Got audio response of size: {len(response.get('audio'))} bytes")
                                
                            logger.info("Response generated successfully")
                            
                            # Add assistant response to conversation
                            conversations[session_id].append({"role": "assistant", "content": response["text"]})
                            
                            # Send response to client
                            response_data = {
                                "type": "response",
                                "text": response["text"],
                                "audio": response.get("audio")
                            }
                            
                            # Add debugging info if in debug mode
                            if hasattr(settings, 'DEBUG') and settings.DEBUG:
                                response_data["debug"] = {
                                    "conversation_length": len(conversations[session_id]),
                                    "response_length": len(response["text"]),
                                    "has_audio": response.get("audio") is not None
                                }
                                
                            await websocket.send_json(response_data)
                            logger.info(f"Sent response to {session_id}")
                            
                        finally:
                            # Clean up temporary files
                            if temp_files:
                                cleanup_temp_files(temp_files)
                                
                    except Exception as e:
                        logger.error(f"Error processing multi-modal message from {session_id}: {e}")
                        logger.error(traceback.format_exc())
                        await websocket.send_json({"type": "error", "message": str(e)})
                        
                        # Check if it's an OOM error
                        if "CUDA out of memory" in str(e) or "OOM" in str(e):
                            await websocket.send_json({
                                "type": "error_oom",
                                "message": "Memory limit reached. Please reset the conversation to continue."
                            })
                            logger.warning("OOM error detected. Suggesting conversation reset.")
                            
                    continue
                    
            # Handle audio message (legacy format)
            if "audio" in data:
                try:
                    # Manage conversation history to prevent OOM
                    conversations[session_id] = manage_conversation_history(conversations[session_id])
                    
                    logger.info(f"Processing audio message from {session_id}")
                    
                    # Get audio data and convert if needed
                    audio_data = data["audio"]
                    
                    # Make sure audio_data is a string before decoding
                    if isinstance(audio_data, str):
                        # Handle data URI format if present
                        if audio_data.startswith('data:audio/'):
                            logger.info("Detected data URI format, extracting base64 content")
                            audio_data = audio_data.split(',')[1]
                            
                        # Decode base64
                        try:
                            audio_bytes = base64.b64decode(audio_data)
                            logger.info(f"Decoded audio: {len(audio_bytes)} bytes")
                        except Exception as e:
                            logger.error(f"Base64 decoding error: {e}")
                            await websocket.send_json({"type": "error", "message": f"Base64 decoding error: {str(e)}"})
                            continue
                            
                        # Validate and process audio
                        try:
                            audio_np = validate_audio(audio_bytes)
                            logger.info(f"Validated audio: shape={audio_np.shape}, dtype={audio_np.dtype}")
                            
                            # Add message to conversation
                            conversations[session_id].append({
                                "role": "user",
                                "content": [{"type": "audio", "audio": audio_np}]
                            })
                            logger.info(f"Added audio message to conversation")
                            
                        except Exception as e:
                            logger.error(f"Audio validation error: {e}")
                            logger.error(traceback.format_exc())
                            await websocket.send_json({"type": "error", "message": f"Audio processing error: {str(e)}"})
                            continue
                            
                    else:
                        logger.error(f"Invalid audio data type: {type(audio_data)}")
                        await websocket.send_json({"type": "error", "message": "Invalid audio data format"})
                        continue
                        
                    # Send acknowledgement
                    await websocket.send_json({"type": "processing"})
                    logger.info("Sent processing acknowledgement")
                    
                    # Generate response
                    logger.info("Generating response...")
                    response = await model.generate_response(conversations[session_id])
                    
                    if "error" in response:
                        logger.error(f"Error generating response: {response['error']}")
                        await websocket.send_json({"type": "error", "message": response["error"]})
                        continue
                        
                    if response.get("audio") is None:
                        logger.warning("No audio in response, model may not be properly configured for audio output")
                    else:
                        logger.info(f"Got audio response of size: {len(response.get('audio'))} bytes")
                        
                    logger.info("Response generated successfully")
                    
                    # Add assistant response to conversation
                    conversations[session_id].append({"role": "assistant", "content": response["text"]})
                    
                    # Send response to client
                    response_data = {
                        "type": "response",
                        "text": response["text"],
                        "audio": response.get("audio")
                    }
                    
                    # Add debugging info if in debug mode
                    if hasattr(settings, 'DEBUG') and settings.DEBUG:
                        response_data["debug"] = {
                            "conversation_length": len(conversations[session_id]),
                            "response_length": len(response["text"]),
                            "has_audio": response.get("audio") is not None
                        }
                        
                    await websocket.send_json(response_data)
                    logger.info(f"Sent response to {session_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing audio from {session_id}: {e}")
                    logger.error(traceback.format_exc())
                    await websocket.send_json({"type": "error", "message": str(e)})
                    
                    # Check if it's an OOM error
                    if "CUDA out of memory" in str(e) or "OOM" in str(e):
                        await websocket.send_json({
                            "type": "error_oom",
                            "message": "Memory limit reached. Please reset the conversation to continue."
                        })
                        logger.warning("OOM error detected. Suggesting conversation reset.")
                        
            # Handle text message (legacy format)
            elif "text" in data:
                try:
                    # Manage conversation history to prevent OOM
                    conversations[session_id] = manage_conversation_history(conversations[session_id])
                    
                    logger.info(f"Processing text message from {session_id}")
                    
                    # Validate text data
                    if not isinstance(data["text"], str):
                        logger.error(f"Invalid text data type: {type(data['text'])}")
                        await websocket.send_json({"type": "error", "message": "Text must be a string"})
                        continue
                        
                    # Add message to conversation
                    conversations[session_id].append({"role": "user", "content": data["text"]})
                    
                    # Send acknowledgement
                    await websocket.send_json({"type": "processing"})
                    logger.info("Sent processing acknowledgement")
                    
                    # Generate response
                    logger.info("Generating response...")
                    response = await model.generate_response(conversations[session_id])
                    
                    if "error" in response:
                        logger.error(f"Error generating response: {response['error']}")
                        await websocket.send_json({"type": "error", "message": response["error"]})
                        continue
                        
                    if response.get("audio") is None:
                        logger.warning("No audio in response, model may not be properly configured for audio output")
                    else:
                        logger.info(f"Got audio response of size: {len(response.get('audio'))} bytes")
                        
                    logger.info("Response generated successfully")
                    
                    # Add assistant response to conversation
                    conversations[session_id].append({"role": "assistant", "content": response["text"]})
                    
                    # Send response to client
                    response_data = {
                        "type": "response",
                        "text": response["text"],
                        "audio": response.get("audio")
                    }
                    
                    # Add debugging info if in debug mode
                    if hasattr(settings, 'DEBUG') and settings.DEBUG:
                        response_data["debug"] = {
                            "conversation_length": len(conversations[session_id]),
                            "response_length": len(response["text"]),
                            "has_audio": response.get("audio") is not None
                        }
                        
                    await websocket.send_json(response_data)
                    logger.info(f"Sent response to {session_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing text message from {session_id}: {e}")
                    logger.error(traceback.format_exc())
                    await websocket.send_json({"type": "error", "message": str(e)})
                    
                    # Check if it's an OOM error
                    if "CUDA out of memory" in str(e) or "OOM" in str(e):
                        await websocket.send_json({
                            "type": "error_oom",
                            "message": "Memory limit reached. Please reset the conversation to continue."
                        })
                        logger.warning("OOM error detected. Suggesting conversation reset.")
                        
            else:
                logger.warning(f"Unknown message format from {session_id}: {data}")
                await websocket.send_json({"type": "error", "message": "Unknown message format"})
                
    except WebSocketDisconnect:
        # Clean up on disconnect
        if session_id in active_connections:
            del active_connections[session_id]
        logger.info(f"Client disconnected: {session_id}")
        
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            logger.error(f"Could not send error to client {session_id}")
        finally:
            # Clean up on error
            if session_id in active_connections:
                del active_connections[session_id]

# Clean up temporary files helper function
def cleanup_temp_files(temp_files):
    """Clean up temporary files."""
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Error removing temp file {file_path}: {e}")

def get_model_path_for_size(size):
    """Get the model path for a specific size."""
    if size == "3B":
        return "Qwen/Qwen2.5-Omni-3B"
    else:
        return "Qwen/Qwen2.5-Omni-7B"