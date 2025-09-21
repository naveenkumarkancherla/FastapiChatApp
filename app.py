import base64
import mimetypes
import os
import io
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types
import uvicorn
from datetime import datetime
import json
import time

app = FastAPI()

# Configuration - Multiple API Keys for rotation
API_KEYS = [
    "AIzaSyA2NPFWjZNH-gxRvtEMNam0jaqEfnxb9MA",
    "AIzaSyBoE7fK9TBB5eztsrgQe61od0EUU23_-Hk",
    "AIzaSyBakEXLFZwWK1pCP6qfvRqiGsM1Ph6gKcM",
    "AIzaSyCvZRMnrmnWRAuiShDTqG9cuUKsriNhiHk",
    "AIzaSyBVnM9fb-TMDXhIHS3sjRbrFHnYmd9sZdw",
    "AIzaSyA-p2EZBRtK5bR5XMOCznjitUXti5n-9g8",
    "AIzaSyDSiTPN4dHwYQ_o2aPWn6GrfLI9xn3oWcw",
    "AIzaSyCmwfW_XEsM-a3yiEmDNlGO_l6hQFSUdh4",
    "AIzaSyC2Dfhpw5kDTdEsZA5O3FpbSCrkTD340J8"
]

# Available models
AVAILABLE_MODELS = [
    "gemini-2.5-flash-image-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp"
]

# Track current API key index and failed keys
current_key_index = 0
failed_keys = set()
last_key_rotation = time.time()

def get_working_client():
    """Get a working Gemini client, rotating through API keys if needed"""
    global current_key_index, failed_keys, last_key_rotation
    
    # Reset failed keys every hour
    if time.time() - last_key_rotation > 3600:
        failed_keys = set()
        last_key_rotation = time.time()
    
    # Try to find a working key
    attempts = 0
    while attempts < len(API_KEYS):
        if current_key_index not in failed_keys:
            try:
                client = genai.Client(api_key=API_KEYS[current_key_index])
                return client, current_key_index
            except Exception as e:
                print(f"API key {current_key_index} failed during initialization: {str(e)}")
                failed_keys.add(current_key_index)
        
        # Move to next key
        current_key_index = (current_key_index + 1) % len(API_KEYS)
        attempts += 1
    
    # If all keys failed, reset and try again
    failed_keys = set()
    current_key_index = 0
    return genai.Client(api_key=API_KEYS[0]), 0

# Initialize with first working client
client, _ = get_working_client()

# Request/Response models
class ImageData(BaseModel):
    data: str
    mime_type: str

class ChatMessage(BaseModel):
    message: str
    images: List[ImageData] = []
    model: str = "gemini-2.0-flash-exp"
    generate_image: bool = False

class ChatResponse(BaseModel):
    text: str
    images: List[dict] = []  # List of {"data": base64_string, "mime_type": str}

# Store conversation history in memory (consider using a database for production)
conversations = {}

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main chat interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini AI Multi-Model Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 900px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header-title {
            font-size: 24px;
            font-weight: bold;
        }
        
        .model-selector {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .model-selector label {
            font-size: 14px;
        }
        
        .model-dropdown {
            padding: 8px 12px;
            border-radius: 10px;
            border: none;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            font-size: 14px;
            cursor: pointer;
            outline: none;
            min-width: 200px;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.assistant .message-content {
            background: white;
            color: #333;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .message.error .message-content {
            background: #ffebee;
            color: #c62828;
            border: 1px solid #ef5350;
        }
        
        .message-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            margin-top: 10px;
            cursor: pointer;
            transition: transform 0.2s;
            display: block;
        }
        
        .message-image:hover {
            transform: scale(1.02);
        }
        
        .model-badge {
            display: inline-block;
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            margin-bottom: 5px;
        }
        
        .error-badge {
            display: inline-block;
            background: rgba(198, 40, 40, 0.1);
            color: #c62828;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            margin-bottom: 5px;
        }
        
        .retry-button {
            margin-top: 10px;
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: transform 0.2s;
            display: inline-block;
        }
        
        .retry-button:hover {
            transform: scale(1.05);
        }
        
        .retry-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .uploaded-images-preview {
            display: none;
            margin-bottom: 15px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 10px;
        }
        
        .uploaded-images-preview.active {
            display: block;
        }
        
        .preview-title {
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .images-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .image-preview-item {
            position: relative;
            width: 100px;
            height: 100px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .image-preview-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .remove-image-btn {
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(255, 0, 0, 0.8);
            color: white;
            border: none;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            line-height: 1;
            transition: background 0.2s;
        }
        
        .remove-image-btn:hover {
            background: rgba(255, 0, 0, 1);
        }
        
        .options-row {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            align-items: center;
            justify-content: space-between;
        }
        
        .left-options {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .right-options {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .checkbox-wrapper {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .checkbox-wrapper input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        
        .checkbox-wrapper label {
            cursor: pointer;
            font-size: 14px;
            color: #666;
        }
        
        .restore-button {
            padding: 6px 12px;
            background: #4caf50;
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.3s;
            display: none;
        }
        
        .restore-button.active {
            display: inline-block;
        }
        
        .restore-button:hover {
            background: #45a049;
            transform: scale(1.05);
        }
        
        .input-wrapper {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .chat-input:focus {
            border-color: #667eea;
        }
        
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
        }
        
        .file-input-wrapper input[type=file] {
            position: absolute;
            left: -9999px;
        }
        
        .file-input-label {
            display: inline-block;
            padding: 10px 15px;
            background: #f0f0f0;
            border-radius: 20px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .file-input-label:hover {
            background: #e0e0e0;
        }
        
        .image-count-badge {
            display: inline-block;
            background: #667eea;
            color: white;
            border-radius: 10px;
            padding: 2px 6px;
            font-size: 11px;
            margin-left: 5px;
        }
        
        .send-button {
            padding: 10px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .send-button:hover {
            transform: scale(1.05);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .typing-indicator {
            display: none;
            padding: 20px;
            text-align: center;
            color: #666;
        }
        
        .typing-indicator.active {
            display: block;
        }
        
        .dots {
            display: inline-block;
        }
        
        .dots::after {
            content: '...';
            animation: dots 1.5s steps(4, end) infinite;
        }
        
        @keyframes dots {
            0%, 20% {
                color: rgba(0, 0, 0, 0);
                text-shadow: .25em 0 0 rgba(0, 0, 0, 0), .5em 0 0 rgba(0, 0, 0, 0);
            }
            40% {
                color: #666;
                text-shadow: .25em 0 0 rgba(0, 0, 0, 0), .5em 0 0 rgba(0, 0, 0, 0);
            }
            60% {
                text-shadow: .25em 0 0 #666, .5em 0 0 rgba(0, 0, 0, 0);
            }
            80%, 100% {
                text-shadow: .25em 0 0 #666, .5em 0 0 #666;
            }
        }
        
        .error-message {
            background: #ff4444;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 20px;
            display: none;
        }
        
        .image-gallery {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        
        .image-gallery img {
            max-width: 200px;
            max-height: 200px;
            object-fit: cover;
            border-radius: 8px;
            cursor: pointer;
        }
        
        .clear-all-btn {
            background: #ff4444;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            cursor: pointer;
            margin-left: auto;
        }
        
        .clear-all-btn:hover {
            background: #ff6666;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="header-title">🤖 Gemini AI Multi-Model Chat</div>
            <div class="model-selector">
                <label for="modelSelect">Model:</label>
                <select id="modelSelect" class="model-dropdown">
                    <option value="gemini-2.0-flash-exp">Gemini 2.0 Flash Exp</option>
                    <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                    <option value="gemini-2.5-flash-image-preview">Gemini 2.5 Flash Image Preview</option>
                    <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                </select>
            </div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message assistant">
                <div class="message-content">
                    <div class="model-badge">System</div>
                    <div>Hello! I'm your AI assistant powered by Gemini. You can:
                    • Switch between different models using the dropdown above
                    • Upload multiple images for analysis
                    • Request image generation (with compatible models)
                    • Remove uploaded images before sending
                    • Retry failed messages without re-uploading
                    
                    How can I help you today?</div>
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <span class="dots">Thinking</span>
        </div>
        
        <div class="error-message" id="errorMessage"></div>
        
        <div class="chat-input-container">
            <div class="uploaded-images-preview" id="uploadedImagesPreview">
                <div style="display: flex; align-items: center;">
                    <span class="preview-title">Uploaded Images:</span>
                    <button class="clear-all-btn" onclick="clearAllImages()">Clear All</button>
                </div>
                <div class="images-grid" id="imagesGrid"></div>
            </div>
            
            <div class="options-row">
                <div class="left-options">
                    <div class="checkbox-wrapper">
                        <input type="checkbox" id="generateImageCheck">
                        <label for="generateImageCheck">Request image generation</label>
                    </div>
                </div>
                <div class="right-options">
                    <button id="restoreButton" class="restore-button" onclick="restoreLastMessage()">
                        ↻ Restore Last Message
                    </button>
                </div>
            </div>
            
            <div class="input-wrapper">
                <input 
                    type="text" 
                    class="chat-input" 
                    id="chatInput" 
                    placeholder="Type your message..."
                    onkeypress="handleKeyPress(event)"
                >
                
                <div class="file-input-wrapper">
                    <input type="file" id="fileInput" accept="image/*" multiple onchange="handleFileSelect(event)">
                    <label for="fileInput" class="file-input-label">
                        📷 Upload
                        <span id="imageCountBadge" class="image-count-badge" style="display: none;">0</span>
                    </label>
                </div>
                
                <button class="send-button" id="sendButton" onclick="sendMessage()">
                    Send
                </button>
            </div>
        </div>
    </div>
    
    <script>
        let uploadedImages = [];
        let lastMessageData = null;
        let retryCount = 0;
        
        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        function handleFileSelect(event) {
            const files = Array.from(event.target.files);
            
            files.forEach(file => {
                if (file && file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const imageData = {
                            data: e.target.result.split(',')[1], // Remove data:image/...;base64,
                            mimeType: file.type,
                            name: file.name,
                            preview: e.target.result,
                            id: Date.now() + Math.random() // Unique ID for each image
                        };
                        uploadedImages.push(imageData);
                        updateImagePreview();
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            // Clear the file input
            event.target.value = '';
        }
        
        function updateImagePreview() {
            const previewContainer = document.getElementById('uploadedImagesPreview');
            const imagesGrid = document.getElementById('imagesGrid');
            const imageCountBadge = document.getElementById('imageCountBadge');
            
            if (uploadedImages.length > 0) {
                previewContainer.classList.add('active');
                imageCountBadge.style.display = 'inline-block';
                imageCountBadge.textContent = uploadedImages.length;
                
                // Clear and rebuild the grid
                imagesGrid.innerHTML = '';
                
                uploadedImages.forEach((image, index) => {
                    const imageItem = document.createElement('div');
                    imageItem.className = 'image-preview-item';
                    imageItem.innerHTML = `
                        <img src="${image.preview}" alt="${image.name}" title="${image.name}">
                        <button class="remove-image-btn" onclick="removeImage(${index})" title="Remove image">×</button>
                    `;
                    imagesGrid.appendChild(imageItem);
                });
            } else {
                previewContainer.classList.remove('active');
                imageCountBadge.style.display = 'none';
                imagesGrid.innerHTML = '';
            }
        }
        
        function removeImage(index) {
            uploadedImages.splice(index, 1);
            updateImagePreview();
        }
        
        function clearAllImages() {
            uploadedImages = [];
            updateImagePreview();
        }
        
        function saveLastMessage(message, images, model, generateImage) {
            lastMessageData = {
                message: message,
                images: [...images], // Create a copy
                model: model,
                generateImage: generateImage,
                timestamp: Date.now()
            };
            
            // Show restore button
            document.getElementById('restoreButton').classList.add('active');
        }
        
        function restoreLastMessage() {
            if (!lastMessageData) return;
            
            // Restore text
            document.getElementById('chatInput').value = lastMessageData.message;
            
            // Restore images
            uploadedImages = [...lastMessageData.images];
            updateImagePreview();
            
            // Restore model
            document.getElementById('modelSelect').value = lastMessageData.model;
            
            // Restore generate image checkbox
            document.getElementById('generateImageCheck').checked = lastMessageData.generateImage;
            
            // Hide restore button after restoring
            document.getElementById('restoreButton').classList.remove('active');
        }
        
        async function sendMessage(isRetry = false) {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            const selectedModel = document.getElementById('modelSelect').value;
            const generateImage = document.getElementById('generateImageCheck').checked;
            
            if (!message && uploadedImages.length === 0) return;
            
            const sendButton = document.getElementById('sendButton');
            sendButton.disabled = true;
            
            // Save message data before sending (for retry functionality)
            if (!isRetry) {
                saveLastMessage(message, uploadedImages, selectedModel, generateImage);
                retryCount = 0;
                
                // Add user message to chat with all uploaded images
                const userImagePreviews = uploadedImages.map(img => img.preview);
                addMessage(message, 'user', userImagePreviews, null, selectedModel);
            } else {
                retryCount++;
            }
            
            // Prepare images for API
            const apiImages = uploadedImages.map(img => ({
                data: img.data,
                mime_type: img.mimeType
            }));
            
            // Clear input and images (only if not retrying)
            if (!isRetry) {
                input.value = '';
                clearAllImages();
                document.getElementById('generateImageCheck').checked = false;
            }
            
            // Show typing indicator
            document.getElementById('typingIndicator').classList.add('active');
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        images: apiImages,
                        model: selectedModel,
                        generate_image: generateImage
                    })
                });
                
                if (!response.ok) {
                    const error = await response.text();
                    throw new Error(error || 'Failed to get response');
                }
                
                const data = await response.json();
                
                // Add assistant response to chat
                const assistantImages = data.images.map(img => 
                    'data:' + img.mime_type + ';base64,' + img.data
                );
                
                addMessage(data.text, 'assistant', null, assistantImages, selectedModel);
                
                // Clear saved message data on success
                lastMessageData = null;
                document.getElementById('restoreButton').classList.remove('active');
                
                // Clear input if it was a retry
                if (isRetry) {
                    input.value = '';
                    clearAllImages();
                    document.getElementById('generateImageCheck').checked = false;
                }
                
            } catch (error) {
                console.error('Error:', error);
                
                // Add error message to chat with retry button
                addErrorMessage(error.message, selectedModel);
                
            } finally {
                document.getElementById('typingIndicator').classList.remove('active');
                sendButton.disabled = false;
            }
        }
        
        function addErrorMessage(errorText, model) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message error';
            
            let content = '<div class="message-content">';
            content += '<div class="error-badge">Error</div>';
            content += '<div>' + escapeHtml(errorText || 'Failed to send message. Please try again.') + '</div>';
            content += `<button class="retry-button" onclick="retryLastMessage()" ${retryCount >= 3 ? 'disabled' : ''}>
                        ${retryCount >= 3 ? 'Max retries reached' : '🔄 Try Again (Attempt ' + (retryCount + 1) + '/3)'}
                       </button>`;
            content += '</div>';
            
            messageDiv.innerHTML = content;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function retryLastMessage() {
            if (!lastMessageData || retryCount >= 3) return;
            
            // Remove the last error message
            const messages = document.getElementById('chatMessages');
            const errorMessages = messages.querySelectorAll('.message.error');
            if (errorMessages.length > 0) {
                errorMessages[errorMessages.length - 1].remove();
            }
            
            // Restore the message data
            document.getElementById('chatInput').value = lastMessageData.message;
            uploadedImages = [...lastMessageData.images];
            updateImagePreview();
            document.getElementById('modelSelect').value = lastMessageData.model;
            document.getElementById('generateImageCheck').checked = lastMessageData.generateImage;
            
            // Retry sending
            sendMessage(true);
        }
        
        function addMessage(text, sender, uploadedImages = null, generatedImages = null, model = null) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + sender;
            
            let content = '<div class="message-content">';
            
            // Add model badge for assistant messages
            if (sender === 'assistant' && model) {
                content += '<div class="model-badge">' + model + '</div>';
            }
            
            if (text) {
                content += '<div>' + escapeHtml(text) + '</div>';
            }
            
            // Display uploaded images
            if (uploadedImages && uploadedImages.length > 0) {
                if (uploadedImages.length === 1) {
                    content += '<img src="' + uploadedImages[0] + '" class="message-image" onclick="openImage(this.src)" />';
                } else {
                    content += '<div class="image-gallery">';
                    uploadedImages.forEach(img => {
                        content += '<img src="' + img + '" onclick="openImage(this.src)" />';
                    });
                    content += '</div>';
                }
            }
            
            // Display generated images
            if (generatedImages && generatedImages.length > 0) {
                if (generatedImages.length === 1) {
                    content += '<img src="' + generatedImages[0] + '" class="message-image" onclick="openImage(this.src)" />';
                } else {
                    content += '<div class="image-gallery">';
                    generatedImages.forEach(img => {
                        content += '<img src="' + img + '" onclick="openImage(this.src)" />';
                    });
                    content += '</div>';
                }
            }
            
            content += '</div>';
            
            messageDiv.innerHTML = content;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function openImage(src) {
            window.open(src, '_blank');
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages and generate responses with automatic API key rotation"""
    global current_key_index, failed_keys, client
    
    max_retries = len(API_KEYS)
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # Get current working client
            client, key_index = get_working_client()
            
            # Build content parts
            parts = []
            
            # Add text if present
            if message.message:
                # If requesting image generation, modify the prompt
                if message.generate_image:
                    parts.append(types.Part.from_text(
                        text=f"Generate an image of: {message.message}"
                    ))
                else:
                    parts.append(types.Part.from_text(text=message.message))
            
            # Add all uploaded images
            for image in message.images:
                image_bytes = base64.b64decode(image.data)
                parts.append(types.Part.from_bytes(
                    mime_type=image.mime_type,
                    data=image_bytes
                ))
            
            # If no content, return early
            if not parts:
                return ChatResponse(text="Please provide a message or upload images.", images=[])
            
            # Create content
            contents = [
                types.Content(
                    role="user",
                    parts=parts
                )
            ]
            
            # Configure generation based on model and request type
            generate_content_config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
            
            # Add response modalities for image-capable models when image generation is requested
            if message.generate_image and "image" in message.model.lower():
                generate_content_config = types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                    response_modalities=["IMAGE", "TEXT"]
                )
            
            # Generate response
            response_text = ""
            response_images = []
            
            try:
                # Use streaming for better handling of responses
                response_stream = client.models.generate_content_stream(
                    model=message.model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                for chunk in response_stream:
                    if chunk.candidates and chunk.candidates[0].content:
                        for part in chunk.candidates[0].content.parts:
                            # Handle text
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                            
                            # Handle inline data (images)
                            elif hasattr(part, 'inline_data') and part.inline_data:
                                if part.inline_data.data:
                                    response_images.append({
                                        "data": base64.b64encode(part.inline_data.data).decode('utf-8'),
                                        "mime_type": part.inline_data.mime_type
                                    })
                
                # Success - return the response
                print(f"Successfully used API key index {key_index}")
                
            except Exception as stream_error:
                # Fallback to non-streaming if streaming fails
                print(f"Streaming failed with key {key_index}, trying non-streaming: {str(stream_error)}")
                
                response = client.models.generate_content(
                    model=message.model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                if response.candidates and response.candidates[0].content:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                        elif hasattr(part, 'inline_data') and part.inline_data:
                            if part.inline_data.data:
                                response_images.append({
                                    "data": base64.b64encode(part.inline_data.data).decode('utf-8'),
                                    "mime_type": part.inline_data.mime_type
                                })
            
            # Ensure we have some response
            if not response_text and not response_images:
                if message.images:
                    response_text = f"I've analyzed the {len(message.images)} image(s) you uploaded. How can I help you with them?"
                else:
                    response_text = "I've processed your request. How else can I help you?"
            
            return ChatResponse(
                text=response_text,
                images=response_images
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            last_error = str(e)
            
            print(f"Error with API key {current_key_index}: {str(e)}")
            
            # Check if it's a quota/rate limit error
            if any(err in error_msg for err in ['quota', 'rate', 'limit', '429', 'resource', 'exhaust']):
                # Mark this key as failed and try next one
                failed_keys.add(current_key_index)
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                retry_count += 1
                
                # If we haven't tried all keys yet, continue
                if retry_count < max_retries:
                    print(f"Rate limit hit, rotating to API key index {current_key_index}")
                    time.sleep(0.5)  # Small delay before retry
                    continue
            
            # For non-quota errors, provide specific error messages
            if "model not found" in error_msg:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model '{message.model}' is not available. Please try a different model."
                )
            else:
                # Try next key for any other error too
                failed_keys.add(current_key_index)
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                retry_count += 1
                
                if retry_count < max_retries:
                    print(f"Error occurred, trying next API key index {current_key_index}")
                    continue
    
    # If all retries failed
    print(f"All API keys exhausted. Failed keys: {failed_keys}")
    
    # Reset failed keys for next attempt
    if len(failed_keys) >= len(API_KEYS):
        failed_keys = set()
        return ChatResponse(
            text="I'm experiencing high demand right now. Please try again in a moment.",
            images=[]
        )
    
    raise HTTPException(
        status_code=503,
        detail=f"Service temporarily unavailable. Please try again. Error: {last_error}"
    )

@app.get("/models")
async def get_models():
    """Get list of available models"""
    return {"models": AVAILABLE_MODELS}

@app.get("/health")
async def health_check():
    """Health check endpoint with API key status"""
    global failed_keys, current_key_index
    
    return {
        "status": "healthy", 
        "available_models": AVAILABLE_MODELS,
        "total_api_keys": len(API_KEYS),
        "working_keys": len(API_KEYS) - len(failed_keys),
        "current_key_index": current_key_index,
        "failed_keys": list(failed_keys)
    }

@app.get("/reset-keys")
async def reset_api_keys():
    """Reset failed API keys to retry them"""
    global failed_keys, current_key_index
    
    old_failed = len(failed_keys)
    failed_keys = set()
    current_key_index = 0
    
    return {
        "message": "API keys reset successfully",
        "previously_failed": old_failed,
        "now_available": len(API_KEYS)
    }

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    print("Starting Gemini Multi-Model Chat Server...")
    print("Available models:", ", ".join(AVAILABLE_MODELS))
    print("Open http://localhost:8000 in your browser")
    
    # Run the server without reload for direct execution
    uvicorn.run(app, host="0.0.0.0", port=8000)