class ChatUI {
    constructor(messagesContainer) {
        this.messagesContainer = messagesContainer;
        this.typingIndicator = null;
    }
    
    addMessage(role, content, isTemporary = false) {
        // Remove typing indicator if it exists
        this.removeTypingIndicator();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role} ${isTemporary ? 'temporary-message' : ''}`;
        
        // Message content
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Format the message with line breaks and preserve whitespace
        messageContent.style.whiteSpace = 'pre-wrap';
        messageContent.textContent = content;
        messageDiv.appendChild(messageContent);
        
        this.messagesContainer.appendChild(messageDiv);
        
        // Scroll to the latest message
        this.scrollToBottom();
        return messageDiv;
    }
    
    addErrorMessage(errorText) {
        // Remove typing indicator if it exists
        this.removeTypingIndicator();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message message-error';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content error-content';
        
        const errorIcon = document.createElement('div');
        errorIcon.className = 'error-icon';
        errorIcon.innerHTML = '⚠️';
        messageContent.appendChild(errorIcon);
        
        const errorTextElement = document.createElement('div');
        errorTextElement.className = 'error-text';
        errorTextElement.innerHTML = errorText;
        messageContent.appendChild(errorTextElement);
        
        messageDiv.appendChild(messageContent);
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addTypingIndicator() {
        // Only add if not already present
        if (this.typingIndicator) return;
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            indicator.appendChild(dot);
        }
        
        this.messagesContainer.appendChild(indicator);
        this.typingIndicator = indicator;
        this.scrollToBottom();
    }
    
    removeTypingIndicator() {
        if (this.typingIndicator) {
            this.typingIndicator.remove();
            this.typingIndicator = null;
        }
    }
    
    removeTemporaryMessage() {
        const tempMessage = this.messagesContainer.querySelector('.temporary-message');
        if (tempMessage) {
            tempMessage.remove();
        }
        this.removeTypingIndicator();
    }

    addCustomMessage(role, contentElement, isTemporary = false) {
        // Remove typing indicator if it exists
        this.removeTypingIndicator();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role} ${isTemporary ? 'temporary-message' : ''}`;
        
        // Message content
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Append the content element
        messageContent.appendChild(contentElement);
        messageDiv.appendChild(messageContent);
        
        this.messagesContainer.appendChild(messageDiv);
        
        // Scroll to the latest message
        this.scrollToBottom();
        return messageDiv;
    }
    
    clearMessages() {
        while (this.messagesContainer.firstChild) {
            this.messagesContainer.removeChild(this.messagesContainer.firstChild);
        }
        this.typingIndicator = null;
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 10);
    }
}