class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recording = false;
        this.stream = null;
    }
    
    async start() {
        if (this.recording) return;
        
        try {
            // Request microphone access if we don't already have it
            if (!this.stream) {
                console.log("Requesting microphone access...");
                this.stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                console.log("Microphone access granted!");
            }
            
            // Create media recorder with the best available type
            const mimeType = this.getSupportedMimeType();
            console.log(`Using MIME type: ${mimeType}`);
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: mimeType,
            });
            
            // Set up event handlers
            this.audioChunks = [];
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            // Start recording
            this.mediaRecorder.start();
            this.recording = true;
            console.log("Recording started");
            
            return this.stream;
        } catch (error) {
            console.error('Error starting recording:', error);
            alert("Could not access microphone: " + error.message);
            throw error;
        }
    }
    
    getSupportedMimeType() {
        const types = [
            'audio/webm',
            'audio/webm;codecs=opus',
            'audio/ogg;codecs=opus',
            'audio/wav',
            'audio/mp4'
        ];
        
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        
        return '';  // Empty string will use browser default
    }
    
    async stop() {
        if (!this.recording || !this.mediaRecorder) {
            return Promise.resolve(new Blob([]));
        }
        
        return new Promise((resolve) => {
            this.mediaRecorder.onstop = () => {
                console.log(`Recording stopped, got ${this.audioChunks.length} chunks`);
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                this.audioChunks = [];
                this.recording = false;
                resolve(audioBlob);
            };
            
            this.mediaRecorder.stop();
        });
    }
    
    isRecording() {
        return this.recording;
    }
    
    dispose() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }
}