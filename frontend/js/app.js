// Main application code

// Request audio autoplay permissions early
function setupAudioPermissions() {
    // Create a temporary audio element
    const tempAudio = document.createElement('audio');
    tempAudio.volume = 0.01; // Very low volume
    // Try to play it to request permission
    const playPromise = tempAudio.play();
    if (playPromise !== undefined) {
        playPromise
            .then(() => {
                console.log("Autoplay permission granted");
                tempAudio.pause();
            })
            .catch(error => {
                console.warn("Autoplay not allowed:", error);
                // Show message to user
                document.getElementById('status').textContent = 
                    'Please click anywhere on the page to enable audio responses';
                // Add a click handler to the document to enable audio
                document.addEventListener('click', function enableAudio() {
                    console.log("User interaction detected, enabling audio");
                    tempAudio.play().then(() => {
                        tempAudio.pause();
                        document.getElementById('status').textContent = 'Click to start voice call';
                        // Remove the event listener after successful enabling
                        document.removeEventListener('click', enableAudio);
                    }).catch(e => {
                        console.error("Still couldn't enable audio:", e);
                    });
                }, { once: false });
            });
    }
}

// Audio Visualizer class
class AudioVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error('Canvas element not found');
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.rafId = null;
        this.isVisualizing = false;
        this.particles = [];
        this.particleCount = 100;
        this.mode = 'idle'; // 'idle', 'listening', 'processing', 'speaking'
        
        // Initialize particles
        this.initParticles();
    }
    
    initParticles() {
        this.particles = [];
        const radius = this.canvas.width / 2;
        
        for (let i = 0; i < this.particleCount; i++) {
            const angle = Math.random() * Math.PI * 2;
            const distance = Math.random() * radius * 0.8;
            
            this.particles.push({
                x: Math.cos(angle) * distance,
                y: Math.sin(angle) * distance,
                size: Math.random() * 3 + 1,
                speedX: (Math.random() - 0.5) * 0.5,
                speedY: (Math.random() - 0.5) * 0.5,
                color: this.getRandomColor(),
                opacity: Math.random() * 0.5 + 0.3
            });
        }
    }
    
    getRandomColor() {
        const colors = [
            '#10a37f', // Primary green
            '#1a73e8', // Blue
            '#7c4dff', // Purple
            '#00bcd4', // Cyan
            '#4285f4'  // Google blue
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    setupAudioAnalyser(stream) {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        const source = this.audioContext.createMediaStreamSource(stream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        source.connect(this.analyser);
        
        const bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Uint8Array(bufferLength);
    }
    
    startListeningVisualization(stream) {
        if (stream) {
            this.setupAudioAnalyser(stream);
        }
        this.mode = 'listening';
        this.start();
    }
    
    startProcessingVisualization() {
        this.mode = 'processing';
        this.start();
    }
    
    startSpeakingVisualization() {
        this.mode = 'speaking';
        this.start();
    }
    
    startIdleVisualization() {
        this.mode = 'idle';
        this.start();
    }
    
    start() {
        if (!this.isVisualizing) {
            this.isVisualizing = true;
            this.animate();
        }
    }
    
    stop() {
        this.isVisualizing = false;
        if (this.rafId) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }
        this.clearCanvas();
    }
    
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    animate() {
        if (!this.isVisualizing) return;
        
        this.rafId = requestAnimationFrame(() => this.animate());
        this.clearCanvas();
        
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        // Different visualization based on mode
        if (this.mode === 'listening' && this.analyser) {
            // Listening mode - audio reactive
            this.analyser.getByteFrequencyData(this.dataArray);
            this.drawListeningVisualization(centerX, centerY);
        } else if (this.mode === 'processing') {
            // Processing mode - AI-like pattern
            this.drawProcessingVisualization(centerX, centerY);
        } else if (this.mode === 'speaking') {
            // Speaking mode - voice waves
            this.drawSpeakingVisualization(centerX, centerY);
        } else {
            // Idle mode - gentle floating particles
            this.drawIdleVisualization(centerX, centerY);
        }
    }
    
    drawListeningVisualization(centerX, centerY) {
        const radius = Math.min(this.canvas.width, this.canvas.height) / 2 * 0.7;
        
        // Draw circular audio spectrum
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius * 0.9, 0, Math.PI * 2);
        this.ctx.strokeStyle = 'rgba(16, 163, 127, 0.2)';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        const barCount = 32;
        const angleStep = (Math.PI * 2) / barCount;
        
        for (let i = 0; i < barCount; i++) {
            const angle = i * angleStep;
            const value = this.dataArray[i % this.dataArray.length] / 255;
            const barHeight = value * radius * 0.7;
            
            const x1 = centerX + Math.cos(angle) * radius * 0.9;
            const y1 = centerY + Math.sin(angle) * radius * 0.9;
            const x2 = centerX + Math.cos(angle) * (radius * 0.9 + barHeight);
            const y2 = centerY + Math.sin(angle) * (radius * 0.9 + barHeight);
            
            this.ctx.beginPath();
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);
            this.ctx.strokeStyle = `rgba(16, 163, 127, ${0.5 + value * 0.5})`;
            this.ctx.lineWidth = 3;
            this.ctx.stroke();
        }
    }
    
    drawProcessingVisualization(centerX, centerY) {
        const radius = Math.min(this.canvas.width, this.canvas.height) / 2 * 0.7;
        const time = Date.now() / 1000;
        
        // Draw pulsing circle
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius * (0.8 + Math.sin(time * 2) * 0.05), 0, Math.PI * 2);
        this.ctx.strokeStyle = 'rgba(16, 163, 127, 0.3)';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Draw rotating dots
        const dotCount = 12;
        const angleStep = (Math.PI * 2) / dotCount;
        
        for (let i = 0; i < dotCount; i++) {
            const angle = i * angleStep + time;
            const x = centerX + Math.cos(angle) * radius * 0.8;
            const y = centerY + Math.sin(angle) * radius * 0.8;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3 + Math.sin(time * 3 + i) * 2, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(16, 163, 127, ${0.5 + Math.sin(time + i) * 0.3})`;
            this.ctx.fill();
        }
        
        // Draw connecting lines
        this.ctx.beginPath();
        for (let i = 0; i < dotCount; i++) {
            const angle = i * angleStep + time;
            const x = centerX + Math.cos(angle) * radius * 0.8;
            const y = centerY + Math.sin(angle) * radius * 0.8;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        this.ctx.closePath();
        this.ctx.strokeStyle = 'rgba(16, 163, 127, 0.2)';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }
    
    drawSpeakingVisualization(centerX, centerY) {
        const radius = Math.min(this.canvas.width, this.canvas.height) / 2 * 0.7;
        const time = Date.now() / 1000;
        
        // Draw concentric circles
        for (let i = 0; i < 3; i++) {
            const pulseRadius = radius * (0.6 + i * 0.2);
            const phase = i * Math.PI / 3;
            const scale = 0.05 + 0.03 * Math.sin(time * 2 + phase);
            
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, pulseRadius * (1 + scale), 0, Math.PI * 2);
            this.ctx.strokeStyle = `rgba(16, 163, 127, ${0.3 - i * 0.1})`;
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        }
        
        // Draw animated particles around the circle
        const particleCount = 20;
        const angleStep = (Math.PI * 2) / particleCount;
        
        for (let i = 0; i < particleCount; i++) {
            const angle = i * angleStep + time * 0.5;
            const waveOffset = Math.sin(time * 3 + i) * 0.1;
            const particleRadius = radius * (0.9 + waveOffset);
            const x = centerX + Math.cos(angle) * particleRadius;
            const y = centerY + Math.sin(angle) * particleRadius;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, 2 + Math.sin(time * 2 + i) * 1, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(16, 163, 127, ${0.6 + Math.sin(time + i) * 0.2})`;
            this.ctx.fill();
        }
    }
    
    drawIdleVisualization(centerX, centerY) {
        const radius = Math.min(this.canvas.width, this.canvas.height) / 2 * 0.7;
        
        // Draw base circle
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius * 0.8, 0, Math.PI * 2);
        this.ctx.strokeStyle = 'rgba(16, 163, 127, 0.1)';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
        
        // Update and draw particles
        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];
            
            // Update position
            p.x += p.speedX;
            p.y += p.speedY;
            
            // Contain particles within circle
            const distance = Math.sqrt(p.x * p.x + p.y * p.y);
            if (distance > radius * 0.7) {
                // Bounce back
                const angle = Math.atan2(p.y, p.x);
                p.x = Math.cos(angle) * radius * 0.7;
                p.y = Math.sin(angle) * radius * 0.7;
                p.speedX *= -0.5;
                p.speedY *= -0.5;
            }
            
            // Draw particle
            this.ctx.beginPath();
            this.ctx.arc(centerX + p.x, centerY + p.y, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color.replace(')', `, ${p.opacity})`).replace('rgb', 'rgba');
            this.ctx.fill();
        }
    }
}

// Enhanced AudioRecorder with continuous mode
class ContinuousAudioRecorder extends AudioRecorder {
    constructor() {
        super();
        this.continuousMode = false;
        this.silenceDetectionThreshold = 0.01;
        this.silenceTimeout = 2000; // 2 seconds of silence before stopping
        this.minRecordingTime = 1000; // Minimum recording time in ms
        this.silenceTimer = null;
        this.recordingStartTime = null;
        this.audioProcessor = null;
        this.isSpeaking = false;
        this.onSpeechStart = null;
        this.onSpeechEnd = null;
    }
    
    async startContinuous() {
        if (this.continuousMode) return;
        
        try {
            // Request microphone access if we don't already have it
            if (!this.stream) {
                console.log("Requesting microphone access for continuous mode...");
                this.stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                console.log("Microphone access granted for continuous mode!");
            }
            
            // Set up audio context for silence detection
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(this.stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            
            processor.onaudioprocess = (e) => this.detectSpeech(e);
            source.connect(processor);
            processor.connect(audioContext.destination);
            
            this.audioProcessor = processor;
            this.audioContext = audioContext;
            this.continuousMode = true;
            
            console.log("Continuous listening mode activated");
            return this.stream;
        } catch (error) {
            console.error('Error starting continuous recording:', error);
            throw error;
        }
    }

    muteDetection(muted) {
        if (!this.continuousMode) return;
        
        this.muted = muted; // Set the muted state
        
        if (muted) {
            // Disconnect the processor to stop audio processing
            if (this.audioProcessor) {
                this.audioProcessor.disconnect();
            }
            
            // Stop any active recording
            if (this.recording) {
                this.stop();
            }
        } else {
            // Reconnect the processor
            if (this.audioProcessor && this.audioContext && this.stream) {
                const source = this.audioContext.createMediaStreamSource(this.stream);
                source.connect(this.audioProcessor);
                this.audioProcessor.connect(this.audioContext.destination);
            }
        }
    }

    stopContinuous() {
        if (!this.continuousMode) return;
        
        // Clean up audio processing
        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor = null;
        }
        
        // Stop any active recording
        if (this.recording) {
            this.stop();
        }
        
        this.continuousMode = false;
        console.log("Continuous listening mode deactivated");
    }
    
    detectSpeech(e) {
        if (!this.continuousMode) return;
        
        // Skip processing if muted
        if (this.muted) return;
        
        // Get audio data
        const input = e.inputBuffer.getChannelData(0);
        
        // Calculate RMS (root mean square) as a measure of volume
        let sum = 0;
        for (let i = 0; i < input.length; i++) {
            sum += input[i] * input[i];
        }
        const rms = Math.sqrt(sum / input.length);
        
        // Check if speaking
        if (rms > this.silenceDetectionThreshold) {
            // If not already recording, start recording
            if (!this.recording) {
                console.log("Speech detected, starting recording...");
                this.start();
                this.recordingStartTime = Date.now();
                if (this.onSpeechStart) this.onSpeechStart();
            }
            
            // Reset silence timer
            if (this.silenceTimer) {
                clearTimeout(this.silenceTimer);
                this.silenceTimer = null;
            }
            
            this.isSpeaking = true;
        } else if (this.recording) {
            // Check if we've been recording for the minimum time
            const recordingTime = Date.now() - this.recordingStartTime;
            
            if (recordingTime >= this.minRecordingTime) {
                // If silence is detected and we're recording, start silence timer
                if (!this.silenceTimer) {
                    this.silenceTimer = setTimeout(() => {
                        console.log("Silence detected, stopping recording...");
                        this.stop().then(audioBlob => {
                            if (this.onSpeechEnd) this.onSpeechEnd(audioBlob);
                        });
                        this.silenceTimer = null;
                        this.isSpeaking = false;
                    }, this.silenceTimeout);
                }
            }
        }
    }
}

// File upload handling
let uploadedFiles = [];

function setupFileUpload() {
    const fileUploadInput = document.getElementById('file-upload');
    const clearUploadsButton = document.getElementById('clear-uploads');
    const uploadPreview = document.getElementById('upload-preview');
    const previewContainer = document.getElementById('preview-container');
    const textInput = document.getElementById('text-input');
    const sendButton = document.getElementById('send-button');
    
    // File upload change handler
    fileUploadInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;
        
        // Process each file
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            
            // Check file size (limit to 10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert(`File ${file.name} is too large. Maximum size is 10MB.`);
                continue;
            }
            
            // Add to uploaded files array
            uploadedFiles.push(file);
            
            // Create preview
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            
            // Different preview based on file type
            if (file.type.startsWith('image/')) {
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                previewItem.appendChild(img);
            } else if (file.type.startsWith('video/')) {
                const video = document.createElement('video');
                video.src = URL.createObjectURL(file);
                video.setAttribute('muted', 'true');
                previewItem.appendChild(video);
            } else {
                // Generic file preview
                const filePreview = document.createElement('div');
                filePreview.className = 'file-preview';
                
                // Icon based on file type
                const icon = document.createElement('i');
                if (file.type.startsWith('audio/')) {
                    icon.className = 'fas fa-music';
                } else if (file.type.includes('pdf')) {
                    icon.className = 'fas fa-file-pdf';
                } else if (file.type.includes('word') || file.type.includes('document')) {
                    icon.className = 'fas fa-file-word';
                } else {
                    icon.className = 'fas fa-file';
                }
                filePreview.appendChild(icon);
                
                // File extension
                const fileExt = document.createElement('div');
                fileExt.className = 'file-ext';
                fileExt.textContent = file.name.split('.').pop().toUpperCase();
                filePreview.appendChild(fileExt);
                
                previewItem.appendChild(filePreview);
            }
            
            // Remove button
            const removeButton = document.createElement('button');
            removeButton.className = 'remove-preview';
            removeButton.innerHTML = '<i class="fas fa-times"></i>';
            removeButton.addEventListener('click', () => {
                // Remove from array
                const index = uploadedFiles.indexOf(file);
                if (index > -1) {
                    uploadedFiles.splice(index, 1);
                }
                
                // Remove preview
                previewItem.remove();
                
                // Hide preview container if empty
                if (uploadedFiles.length === 0) {
                    uploadPreview.classList.add('hidden');
                }
                
                // Update send button state
                updateSendButtonState();
            });
            previewItem.appendChild(removeButton);
            
            // Add to preview container
            previewContainer.appendChild(previewItem);
        }
        
        // Show preview container
        uploadPreview.classList.remove('hidden');
        
        // Update send button state
        updateSendButtonState();
        
        // Reset file input
        fileUploadInput.value = '';
    });
    
    // Clear uploads button
    clearUploadsButton.addEventListener('click', () => {
        // Clear uploaded files array
        uploadedFiles = [];
        
        // Clear preview container
        previewContainer.innerHTML = '';
        
        // Hide preview container
        uploadPreview.classList.add('hidden');
        
        // Update send button state
        updateSendButtonState();
    });
    
    // Text input handler for send button state
    textInput.addEventListener('input', () => {
        // Auto-resize textarea
        textInput.style.height = 'auto';
        textInput.style.height = (textInput.scrollHeight) + 'px';
        
        // Update send button state
        updateSendButtonState();
    });
    
    // Function to update send button state
    function updateSendButtonState() {
        const hasText = textInput.value.trim().length > 0;
        const hasFiles = uploadedFiles.length > 0;
        
        sendButton.disabled = !hasText && !hasFiles;
    }
}

// Function to handle sending a message with text and/or files
async function sendMessage() {
    const textInput = document.getElementById('text-input');
    const text = textInput.value.trim();
    
    // Don't send if no content and no files
    if (!text && uploadedFiles.length === 0) return;
    
    // Prepare message content
    let messageContent = [];
    
    // Add text if present
    if (text) {
        messageContent.push({
            type: 'text',
            text: text
        });
    }
    
    // Process files
    for (const file of uploadedFiles) {
        try {
            // Convert file to base64
            const base64Data = await fileToBase64(file);
            
            if (file.type.startsWith('image/')) {
                messageContent.push({
                    type: 'image',
                    image: base64Data
                });
            } else if (file.type.startsWith('video/')) {
                messageContent.push({
                    type: 'video',
                    video: base64Data
                });
            } else if (file.type.startsWith('audio/')) {
                messageContent.push({
                    type: 'audio',
                    audio: base64Data
                });
            } else {
                // Generic file
                messageContent.push({
                    type: 'file',
                    file: {
                        name: file.name,
                        type: file.type,
                        size: file.size,
                        data: base64Data
                    }
                });
            }
        } catch (error) {
            console.error(`Error processing file ${file.name}:`, error);
            chatUI.addErrorMessage(`Failed to process file ${file.name}`);
        }
    }
    
    // Create message preview in chat
    const messagePreview = createMessagePreview(text, uploadedFiles);
    chatUI.addCustomMessage('user', messagePreview);
    
    // Clear input and uploads
    textInput.value = '';
    textInput.style.height = 'auto';
    uploadedFiles = [];
    document.getElementById('preview-container').innerHTML = '';
    document.getElementById('upload-preview').classList.add('hidden');
    document.getElementById('send-button').disabled = true;
    
    // Send to server
    if (ws.readyState === WebSocket.OPEN) {
        // Add typing indicator
        chatUI.addTypingIndicator();
        
        // Prepare message for server
        const serverMessage = {
            type: 'message',
            content: messageContent
        };
        
        console.log("Sending message to server:", serverMessage);
        ws.send(JSON.stringify(serverMessage));
    } else {
        chatUI.addErrorMessage("Connection lost. Please reload the page.");
    }
}

// Helper function to convert file to base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const base64String = reader.result.split(',')[1];
            resolve(base64String);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Function to create a message preview with files
function createMessagePreview(text, files) {
    const container = document.createElement('div');
    
    // Add text if present
    if (text) {
        const textElement = document.createElement('div');
        textElement.textContent = text;
        container.appendChild(textElement);
    }
    
    // Add files
    for (const file of files) {
        const mediaElement = document.createElement('div');
        mediaElement.className = 'message-media';
        
        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.className = 'message-image';
            img.src = URL.createObjectURL(file);
            mediaElement.appendChild(img);
        } else if (file.type.startsWith('video/')) {
            const video = document.createElement('video');
            video.className = 'message-video';
            video.src = URL.createObjectURL(file);
            video.controls = true;
            mediaElement.appendChild(video);
        } else if (file.type.startsWith('audio/')) {
            const audio = document.createElement('audio');
            audio.className = 'message-audio';
            audio.src = URL.createObjectURL(file);
            audio.controls = true;
            mediaElement.appendChild(audio);
        } else {
            // Generic file
            const fileElement = document.createElement('div');
            fileElement.className = 'message-file';
            
            const iconElement = document.createElement('div');
            iconElement.className = 'file-icon';
            
            // Icon based on file type
            const icon = document.createElement('i');
            if (file.type.includes('pdf')) {
                icon.className = 'fas fa-file-pdf';
            } else if (file.type.includes('word') || file.type.includes('document')) {
                icon.className = 'fas fa-file-word';
            } else {
                icon.className = 'fas fa-file';
            }
            iconElement.appendChild(icon);
            
            const infoElement = document.createElement('div');
            infoElement.className = 'file-info';
            
            const nameElement = document.createElement('div');
            nameElement.className = 'file-name';
            nameElement.textContent = file.name;
            
            const sizeElement = document.createElement('div');
            sizeElement.className = 'file-size';
            sizeElement.textContent = formatFileSize(file.size);
            
            infoElement.appendChild(nameElement);
            infoElement.appendChild(sizeElement);
            
            fileElement.appendChild(iconElement);
            fileElement.appendChild(infoElement);
            
            mediaElement.appendChild(fileElement);
        }
        
        container.appendChild(mediaElement);
    }
    
    return container;
}

// Helper function to format file size
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
}

window.addEventListener('DOMContentLoaded', async () => {
    try {
        // Check if browser supports necessary APIs
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Your browser doesn't support audio recording. Please use a modern browser like Chrome, Firefox, or Edge.");
        }
        
        // Try to get microphone permissions early
        await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log("Microphone access granted on page load");
        
        // Setup audio playback permissions
        setupAudioPermissions();
        
        // Initialize the rest of the application
        initializeApp();
    } catch (error) {
        console.error("Microphone access error:", error);
        document.getElementById('status').textContent = "Microphone access denied. Please enable microphone access in your browser settings and refresh the page.";
        document.getElementById('toggleVoiceButton').disabled = true;
    }
});

function initializeApp() {
    // Generate a unique session ID
    const sessionId = generateSessionId();
    
    // Debug logging for URL information
    console.log("Current URL:", window.location.href);
    console.log("Hostname:", window.location.hostname);
    console.log("Port:", window.location.port);
    console.log("Protocol:", window.location.protocol);
    
    // Initialize WebSocket connection
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Use the same host and port as the page
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/${sessionId}`;
    console.log("Connecting to WebSocket at:", wsUrl);
    window.ws = new WebSocket(wsUrl);
    let reconnectAttempts = 0;
    let reconnectTimer = null;

    // Initialize audio recorder with continuous mode
    const recorder = new ContinuousAudioRecorder();
    
    // Initialize audio visualizers
    const smallVisualizer = new AudioVisualizer('visualizer');
    smallVisualizer.startIdleVisualization();
    
    const callVisualizer = new AudioVisualizer('call-visualizer');
    callVisualizer.startIdleVisualization();
    
    // UI elements
    const toggleVoiceButton = document.getElementById('toggleVoiceButton');
    const voiceCallPanel = document.getElementById('voice-call-panel');
    const muteButton = document.getElementById('muteButton');
    const endCallButton = document.getElementById('endCallButton');
    const resetButton = document.getElementById('resetButton');
    const statusElement = document.getElementById('status');
    const messagesContainer = document.getElementById('messages');
    
    // Call state
    let callActive = false;
    let muted = false;
    let processingResponse = false;
    
    // Chat UI manager
    window.chatUI = new ChatUI(messagesContainer);
    
    // Setup file upload and text input
    setupFileUpload();

    function setupWebSocket() {
        ws.onopen = () => {
            console.log('WebSocket connection established');
            statusElement.textContent = 'Ready';
            toggleVoiceButton.disabled = false;
            resetButton.disabled = false;
            reconnectAttempts = 0; // Reset reconnect attempts on successful connection
            
            // Add welcome message if no messages
            if (messagesContainer.children.length === 0) {
                chatUI.addMessage('assistant', 'Hello! I\'m your Qwen Omni Assistant. I can help with text, voice, images, and more. How can I assist you today?');
            }
        };
        
        ws.onclose = (event) => {
            console.log('WebSocket connection closed', event);
            statusElement.textContent = 'Connection lost. Reconnecting...';
            toggleVoiceButton.disabled = true;
            endCallButton.disabled = true;
            muteButton.disabled = true;
            document.getElementById('send-button').disabled = true;
            
            // End call if active
            if (callActive) {
                endCall();
            }
            
            // Clear any existing reconnect timer
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
            }
            
            // Attempt to reconnect with exponential backoff
            if (reconnectAttempts < 5) {
                const timeout = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
                console.log(`Attempting to reconnect in ${timeout/1000} seconds...`);
                reconnectTimer = setTimeout(() => {
                    reconnectAttempts++;
                    console.log(`Reconnect attempt ${reconnectAttempts}`);
                    window.ws = new WebSocket(wsUrl);
                    setupWebSocket();
                }, timeout);
            } else {
                statusElement.textContent = 'Connection failed. Please reload the page.';
            }
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("Received message:", data.type);
                
                if (data.type === 'processing') {
                    statusElement.textContent = 'Processing...';
                    statusElement.classList.add('processing');
                    processingResponse = true;
                    
                    // Switch visualizer to processing mode
                    if (callActive) {
                        callVisualizer.startProcessingVisualization();
                    } else {
                        smallVisualizer.startProcessingVisualization();
                    }
                }
                else if (data.type === 'response') {
                    // Remove temporary processing message if exists
                    chatUI.removeTemporaryMessage();
                    processingResponse = false;
                    
                    // Add new response to chat
                    chatUI.addMessage('assistant', data.text);
                    
                    // Debug audio response
                    console.log("Response data:", {
                        text_length: data.text?.length || 0,
                        has_audio: !!data.audio,
                        audio_length: data.audio?.length || 0,
                        debug: data.debug || "No debug info"
                    });
                    
                    // Play audio if available
                    if (data.audio) {
                        console.log("Playing audio response, length:", data.audio.length);
                        if (callActive) {
                            callVisualizer.startSpeakingVisualization();
                        } else {
                            smallVisualizer.startSpeakingVisualization();
                        }
                        playResponseAudio(data.audio);
                    } else {
                        console.warn("No audio in response");
                        statusElement.textContent = callActive ? 'Listening...' : 'Ready';
                        statusElement.classList.remove('processing');
                        
                        // Return visualizer to appropriate mode
                        if (callActive) {
                            callVisualizer.startListeningVisualization(recorder.stream);
                        } else {
                            smallVisualizer.startIdleVisualization();
                        }
                    }
                }
                else if (data.type === 'error') {
                    statusElement.classList.remove('processing');
                    chatUI.removeTemporaryMessage();
                    processingResponse = false;
                    
                    // Return visualizer to appropriate mode
                    if (callActive) {
                        callVisualizer.startListeningVisualization(recorder.stream);
                    } else {
                        smallVisualizer.startIdleVisualization();
                    }
                    
                    // Check if it's an OOM error
                    if (data.message.includes('CUDA out of memory') || data.message.includes('OOM')) {
                        statusElement.textContent = 'Memory limit reached. Please reset the conversation to continue.';
                        // Make reset button pulse to draw attention
                        resetButton.classList.add('pulse-attention');
                        // Show a more user-friendly error message
                        chatUI.addErrorMessage("The AI's memory is full. Please click the reset button to start a new conversation.");
                    } else {
                        statusElement.textContent = 'Error: ' + data.message;
                        chatUI.addErrorMessage(data.message);
                    }
                }
                else if (data.type === 'reset_confirmed') {
                    chatUI.clearMessages();
                    statusElement.textContent = callActive ? 'Listening...' : 'Ready';
                    resetButton.classList.remove('pulse-attention');
                    
                    // Add welcome message
                    chatUI.addMessage('assistant', 'Hello! I\'m your Qwen Omni Assistant. I can help with text, voice, images, and more. How can I assist you today?');
                    
                    // Return visualizer to appropriate mode
                    if (callActive) {
                        callVisualizer.startListeningVisualization(recorder.stream);
                    } else {
                        smallVisualizer.startIdleVisualization();
                    }
                }
                else if (data.type === 'pong') {
                    console.log('Heartbeat received');
                }
            } catch (error) {
                console.error('Error processing message:', error);
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            statusElement.textContent = 'Connection error. Please check console for details.';
        };
    }
    
    setupWebSocket();
    
    // Set up speech detection callbacks
    recorder.onSpeechStart = () => {
        console.log("Speech started");
        toggleVoiceButton.classList.add('listening');
        statusElement.textContent = 'Listening...';
    };
    
    recorder.onSpeechEnd = (audioBlob) => {
        console.log("Speech ended, processing audio...");
        toggleVoiceButton.classList.remove('listening');
        statusElement.textContent = 'Processing...';
        
        // Check if recording is too short
        if (audioBlob.size < 1000) {
            console.log("Recording too short, continuing to listen...");
            statusElement.textContent = 'Listening...';
            return;
        }
        
        // Add user message to chat
        chatUI.addMessage('user', 'Voice message', true);
        
        // Switch visualizer to processing mode
        if (callActive) {
            callVisualizer.startProcessingVisualization();
        } else {
            smallVisualizer.startProcessingVisualization();
        }
        
        // Convert blob to base64 and send to server
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];
            if (ws.readyState === WebSocket.OPEN) {
                console.log("Sending audio to server, size:", base64Audio.length);
                ws.send(JSON.stringify({
                    audio: base64Audio
                }));
            } else {
                statusElement.textContent = 'Connection lost. Please reload the page.';
                if (callActive) {
                    callVisualizer.startIdleVisualization();
                } else {
                    smallVisualizer.startIdleVisualization();
                }
                endCall();
            }
        };
        reader.readAsDataURL(audioBlob);
    };
    
    // Toggle voice button handler
    toggleVoiceButton.addEventListener('click', () => {
        if (voiceCallPanel.classList.contains('hidden')) {
            // Show voice call panel
            voiceCallPanel.classList.remove('hidden');
            toggleVoiceButton.classList.add('active');
            
            // Start voice call
            startCall();
        } else {
            // Hide voice call panel
            voiceCallPanel.classList.add('hidden');
            toggleVoiceButton.classList.remove('active');
            
            // End voice call
            endCall();
        }
    });
    
    // Mute button handler
    muteButton.addEventListener('click', toggleMute);
    
    // End call button handler
    endCallButton.addEventListener('click', () => {
        endCall();
        voiceCallPanel.classList.add('hidden');
        toggleVoiceButton.classList.remove('active');
    });
    
    // Reset button handler
    resetButton.addEventListener('click', () => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'reset' }));
            statusElement.textContent = 'Resetting conversation...';
            // Remove the pulsing effect when clicked
            resetButton.classList.remove('pulse-attention');
        }
    });
    
    // Send button handler
    document.getElementById('send-button').addEventListener('click', sendMessage);
    
    // Text input enter key handler (Shift+Enter for new line)
    document.getElementById('text-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!document.getElementById('send-button').disabled) {
                sendMessage();
            }
        }
    });
    
    function startCall() {
        if (callActive) return;
        
        recorder.startContinuous().then((stream) => {
            callActive = true;
            toggleVoiceButton.classList.add('active');
            statusElement.textContent = 'Listening...';
            
            // Enable call control buttons
            muteButton.disabled = false;
            endCallButton.disabled = false;
            
            // Start visualizer in listening mode
            callVisualizer.startListeningVisualization(stream);
        }).catch(error => {
            statusElement.textContent = 'Could not start call: ' + error.message;
            console.error('Call start error:', error);
        });
    }
    
    function endCall() {
        if (!callActive) return;
        
        recorder.stopContinuous();
        callActive = false;
        muted = false;
        
        // Update UI
        toggleVoiceButton.classList.remove('active', 'listening');
        muteButton.classList.remove('active');
        muteButton.querySelector('i').className = 'fas fa-microphone-slash';
        statusElement.textContent = 'Call ended';
        
        // Disable call control buttons
        muteButton.disabled = true;
        endCallButton.disabled = true;
        
        // Switch visualizer back to idle
        callVisualizer.startIdleVisualization();
        smallVisualizer.startIdleVisualization();
    }

    function toggleMute() {
        if (!callActive) return;
        
        muted = !muted;
        
        if (muted) {
            // Update UI for muted state
            muteButton.classList.add('active');
            statusElement.textContent = 'Microphone muted';
            
            // Switch visualizer to idle
            callVisualizer.startIdleVisualization();
            
            // Temporarily disable speech detection without stopping continuous mode
            if (recorder.audioProcessor) {
                // Disconnect the processor to stop audio processing
                recorder.audioProcessor.disconnect();
            }
        } else {
            // Update UI for unmuted state
            muteButton.classList.remove('active');
            statusElement.textContent = 'Listening...';
            
            // Switch visualizer back to listening
            callVisualizer.startListeningVisualization(recorder.stream);
            
            // Re-enable speech detection
            if (recorder.audioProcessor && recorder.audioContext && recorder.stream) {
                // Reconnect the processor
                const source = recorder.audioContext.createMediaStreamSource(recorder.stream);
                source.connect(recorder.audioProcessor);
                recorder.audioProcessor.connect(recorder.audioContext.destination);
            }
        }
    }
    
    function playResponseAudio(base64Audio) {
        try {
            console.log(`Playing audio response (${base64Audio.substring(0, 20)}...)`);
            // Create audio element
            const audioSrc = 'data:audio/wav;base64,' + base64Audio;
            const audio = new Audio(audioSrc);
            
            // Set up debugging events
            audio.onerror = (e) => {
                console.error('Error playing audio:', e);
                statusElement.textContent = 'Error playing audio response';
                statusElement.classList.remove('processing');
                
                // Return visualizer to appropriate mode
                if (callActive) {
                    callVisualizer.startListeningVisualization(recorder.stream);
                } else {
                    smallVisualizer.startIdleVisualization();
                }
            };
            
            audio.oncanplaythrough = () => {
                console.log('Audio ready to play');
                statusElement.textContent = 'Playing response...';
                // Start playback when ready
                audio.play().catch(error => {
                    console.error('Error playing audio:', error);
                    statusElement.textContent = 'Could not play audio. Click to interact with the page first.';
                    statusElement.classList.remove('processing');
                    
                    // Return visualizer to appropriate mode
                    if (callActive) {
                        callVisualizer.startListeningVisualization(recorder.stream);
                    } else {
                        smallVisualizer.startIdleVisualization();
                    }
                });
            };
            
            audio.onended = () => {
                console.log('Audio playback finished');
                statusElement.textContent = callActive ? 'Listening...' : 'Ready';
                statusElement.classList.remove('processing');
                
                // Return visualizer to appropriate mode
                if (callActive) {
                    callVisualizer.startListeningVisualization(recorder.stream);
                } else {
                    smallVisualizer.startIdleVisualization();
                }
            };
            
            // Start loading the audio
            audio.load();
        } catch (error) {
            console.error('Error setting up audio playback:', error);
            statusElement.textContent = 'Error playing audio: ' + error.message;
            statusElement.classList.remove('processing');
            
            // Return visualizer to appropriate mode
            if (callActive) {
                callVisualizer.startListeningVisualization(recorder.stream);
            } else {
                smallVisualizer.startIdleVisualization();
            }
        }
    }
    
    function generateSessionId() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
    
    // Send heartbeat to keep the connection alive
    setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
            console.log("Sending heartbeat ping...");
            ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);
    
    // Cleanup when page is closed
    window.addEventListener('beforeunload', () => {
        if (callActive) {
            recorder.stopContinuous();
        }
        recorder.dispose();
        smallVisualizer.stop();
        callVisualizer.stop();
        if (ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    });
    
    // Handle visibility changes (tab switching)
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            // Page is now visible, check connection
            if (ws.readyState !== WebSocket.OPEN) {
                statusElement.textContent = 'Reconnecting...';
                window.ws = new WebSocket(wsUrl);
                setupWebSocket();
            }
        }
    });
    
    // Handle keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Spacebar to toggle call when focused on body
        if (e.code === 'Space' && !e.repeat && !e.target.matches('input, textarea, [contenteditable]')) {
            e.preventDefault();
            if (voiceCallPanel.classList.contains('hidden')) {
                // Show voice call panel and start call
                voiceCallPanel.classList.remove('hidden');
                toggleVoiceButton.classList.add('active');
                startCall();
            } else if (callActive) {
                // Toggle mute if call is active
                toggleMute();
            }
        }
        
        // Escape to end call
        if (e.code === 'Escape' && callActive) {
            e.preventDefault();
            endCall();
            voiceCallPanel.classList.add('hidden');
            toggleVoiceButton.classList.remove('active');
        }
        
        // 'R' to reset conversation
        if (e.code === 'KeyR' && !e.repeat && (e.ctrlKey || e.metaKey) && !e.target.matches('input, textarea, [contenteditable]')) {
            e.preventDefault();
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'reset' }));
                statusElement.textContent = 'Resetting conversation...';
                resetButton.classList.remove('pulse-attention');
            }
        }
    });
}