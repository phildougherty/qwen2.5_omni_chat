FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ninja-build \
    curl \
    ffmpeg \
    libsndfile1 \
    python3-setuptools \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Update pip and install PyTorch with CUDA support
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir packaging wheel ninja setuptools && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir --upgrade accelerate

# Install Flash Attention 2 with specific configuration
RUN FLASH_ATTN_INSTALL_NO_OPENAI_TRITON=1 pip3 install --no-cache-dir flash-attn --no-build-isolation

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 uninstall -y transformers && \
    pip3 install --no-cache-dir git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356

# Create cache directory with appropriate permissions
RUN mkdir -p /app/.cache/huggingface && \
    chmod -R 777 /app/.cache

# Copy application code
COPY ./app /app/app
COPY main.py /app/
COPY warmup.py /app/

# Expose port
EXPOSE 8000

# Check GPU availability at startup and run the application with warmup
CMD echo "Checking GPU availability..." && \
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" && \
    python3 warmup.py && \
    uvicorn main:app --host 0.0.0.0 --port 8000