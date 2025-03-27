# Qwen2.5 Omni Chat

A full-stack application that leverages Alibaba's Qwen2.5-Omni model to create a multi-modal AI assistant capable of processing and generating text, audio, images, and video.

![Qwen Omni Chat Interface](https://i.imgur.com/placeholder.png)

## Features

- **Multi-modal Interactions**: Communicate with the AI using text, voice, images, and video
- **Voice Conversations**: Natural voice-based conversations with automatic speech detection
- **Real-time Visualizations**: Dynamic audio visualizations that respond to speech and AI processing
- **File Uploads**: Support for uploading and processing various file types
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Dark Mode Support**: Automatic theme switching based on system preferences

I HAVE NOT TESTED IMAGE/VIDEO YET, I DO NOT HAVE ENOUGH VRAM

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with at least 16GB VRAM (recommended: 24GB+)
- NVIDIA Container Toolkit (nvidia-docker)
- 50GB+ of free disk space (for model weights and Docker images)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/phildougherty/qwen2.5_omni_chat.git
cd qwen2.5_omni_chat
```

### 2. SSL Certificate Setup

The application requires SSL certificates for secure WebSocket connections. You have two options:

#### Option A: Generate Self-Signed Certificates (for development)

```bash
mkdir -p certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem
```

When prompted, use `localhost` as the Common Name if you're running the application locally.

#### Option B: Use Let's Encrypt Certificates (for production)

If you have a domain name and want to deploy the application publicly:

1. Install certbot:
   ```bash
   sudo apt-get update
   sudo apt-get install certbot
   ```

2. Generate certificates:
   ```bash
   sudo certbot certonly --standalone -d yourdomain.com
   ```

3. Copy the certificates to the project:
   ```bash
   sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem certs/cert.pem
   sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem certs/key.pem
   sudo chmod 644 certs/cert.pem certs/key.pem
   ```

### 3. Configuration

#### Backend Configuration

Edit `backend/app/config.py` to customize the model settings:

```python
# Model settings
MODEL_PATH: str = "Qwen/Qwen2.5-Omni-7B"  # Model ID from Hugging Face
DEVICE_MAP: str = "auto"                  # Device mapping strategy
ATTN_IMPLEMENTATION: str = "sdpa"         # Attention implementation (sdpa recommended)
DEFAULT_VOICE: str = "Chelsie"            # Default voice for audio responses
SAMPLE_RATE: int = 24000                  # Audio sample rate
ENABLE_AUDIO_OUTPUT: bool = True          # Enable audio generation
USE_AUDIO_IN_VIDEO: bool = True           # Extract audio from video inputs
MODEL_MAX_LENGTH: int = 8192              # Maximum context length
```

#### Nginx Configuration

Edit `nginx/nginx.conf` to customize the server settings:

- Port configuration (default: 80 for HTTP, 443 for HTTPS)
- SSL settings
- Proxy settings for the backend

#### Docker Compose Configuration

Edit `docker-compose.yml` to customize:

- Port mappings (default: 8888 for HTTP, 8443 for HTTPS)
- GPU allocation
- Volume mounts

### 4. Build and Run

```bash
docker-compose up -d
```

This will:
1. Build the Docker images for the frontend, backend, and nginx
2. Download the Qwen2.5-Omni model (this may take some time on first run)
3. Start all services

### 5. Access the Application

Open your browser and navigate to:

- `https://localhost:8443` (if using self-signed certificates)
- `https://yourdomain.com:8443` (if using Let's Encrypt certificates)

If you're using self-signed certificates, you'll need to accept the security warning in your browser.

## Usage

### Voice Conversations

1. Click the microphone button in the bottom right to start a voice call
2. Speak naturally - the system will automatically detect speech and silence
3. Use the mute button to temporarily disable the microphone
4. Click the end call button to terminate the voice session

### Text and File Input

1. Type your message in the text input field
2. Click the paperclip icon to attach files (images, videos, audio, documents)
3. Press Enter or click the send button to submit your message

### Keyboard Shortcuts

- **Space**: Toggle voice call (when not focused on text input)
- **Escape**: End voice call
- **Ctrl+R** or **Cmd+R**: Reset conversation
- **Enter**: Send message
- **Shift+Enter**: New line in text input

## Customization

### Model Selection

- `Qwen/Qwen2.5-Omni-7B`: 7 billion parameter model (default)

### Voice Options

The model supports multiple voice options. Change the `DEFAULT_VOICE` in `backend/app/config.py`:

- `Chelsie`: Default female voice
- `Ethan`: Male voice


### Port Configuration

By default, the application uses:
- Port 8888 for HTTP
- Port 8443 for HTTPS

To change these ports, edit the `docker-compose.yml` file:

```yaml
nginx:
  ports:
    - "8888:80"  # Change 8888 to your desired HTTP port
    - "8443:443" # Change 8443 to your desired HTTPS port
```

## Troubleshooting

### CUDA Out of Memory Errors

If you encounter "CUDA out of memory" errors:

1. Reduce the `MAX_CONVERSATION_TURNS` in `backend/app/config.py`
2. Use a smaller model variant
3. Increase the GPU memory allocation in `docker-compose.yml`

### WebSocket Connection Issues

If you have trouble with WebSocket connections:

1. Ensure your SSL certificates are properly configured
2. Check that ports 8888 and 8443 (or your custom ports) are open in your firewall
3. Verify that the nginx configuration is correctly routing WebSocket requests

### Audio Not Working

If audio input or output isn't working:

1. Ensure your browser has permission to access the microphone
2. Click anywhere on the page to enable audio (browser autoplay policy)
3. Check that `ENABLE_AUDIO_OUTPUT` is set to `True` in the backend configuration

## Advanced Configuration

### Memory Management

To optimize memory usage, adjust these settings in `backend/app/config.py`:

```python
# Memory management settings
MAX_CONVERSATION_TURNS: int = 3  # Maximum number of conversation turns to keep
CLEANUP_AFTER_RESPONSE: bool = True  # Force garbage collection after responses
PYTORCH_CUDA_ALLOC_CONF: str = "expandable_segments:True"  # PyTorch memory allocation
```

### NVIDIA Container Runtime

Ensure your `docker-compose.yml` is configured to use the NVIDIA runtime:

```yaml
backend:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Qwen Team at Alibaba](https://github.com/QwenLM/Qwen) for creating the Qwen2.5-Omni model
- [Hugging Face](https://huggingface.co/) for hosting the model and providing the transformers library
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [NVIDIA](https://developer.nvidia.com/) for GPU acceleration support

## Disclaimer

This project uses the Qwen2.5-Omni model which has its own license and usage restrictions. Please review the [Qwen2.5-Omni license](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) before using this application for any purpose.

---

For questions, issues, or feature requests, please [open an issue](https://github.com/phildougherty/qwen2.5_omni_chat/issues) on GitHub.
