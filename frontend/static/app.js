/**
 * RAG FastAPI - Frontend JavaScript
 * Handles file uploads, queries, and UI interactions
 */

// ============================================================================
// Constants & Configuration
// ============================================================================

const API_BASE = window.location.origin + '/api';
let selectedFiles = [];

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    // Upload
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    browseBtn: document.getElementById('browseBtn'),
    uploadBtn: document.getElementById('uploadBtn'),
    fileList: document.getElementById('fileList'),
    
    // Knowledge Base
    kbStats: document.getElementById('kbStats'),
    docCount: document.getElementById('docCount'),
    chunkCount: document.getElementById('chunkCount'),
    documentList: document.getElementById('documentList'),
    clearBtn: document.getElementById('clearBtn'),
    
    // Chat
    messagesContainer: document.getElementById('messagesContainer'),
    queryInput: document.getElementById('queryInput'),
    sendBtn: document.getElementById('sendBtn'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    charCounter: document.getElementById('charCounter'),
    
    // Status
    systemStatus: document.getElementById('systemStatus'),
    
    // Overlays
    loadingOverlay: document.getElementById('loadingOverlay'),
    toastContainer: document.getElementById('toastContainer')
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadSystemStatus();
    autoResizeTextarea();
});

function initializeEventListeners() {
    // File Upload
    elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.uploadBtn.addEventListener('click', uploadFiles);
    
    // Drag and Drop
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    
    // Chat
    elements.sendBtn.addEventListener('click', sendQuery);
    elements.queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });
    elements.queryInput.addEventListener('input', handleInputChange);
    elements.clearChatBtn.addEventListener('click', clearChat);
    
    // Knowledge Base
    elements.clearBtn.addEventListener('click', clearKnowledgeBase);
}

// ============================================================================
// File Upload Handling
// ============================================================================

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addFiles(files);
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
    if (files.length === 0) {
        showToast('Please drop PDF files only', 'warning');
        return;
    }
    addFiles(files);
}

function addFiles(files) {
    const pdfFiles = files.filter(f => f.type === 'application/pdf');
    if (pdfFiles.length !== files.length) {
        showToast('Only PDF files are allowed', 'warning');
    }
    
    selectedFiles = [...selectedFiles, ...pdfFiles];
    renderFileList();
    elements.uploadBtn.disabled = selectedFiles.length === 0;
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
    elements.uploadBtn.disabled = selectedFiles.length === 0;
}

function renderFileList() {
    if (selectedFiles.length === 0) {
        elements.fileList.innerHTML = '';
        return;
    }
    
    elements.fileList.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-item">
            <div class="file-info">
                <span class="file-name" title="${file.name}">📄 ${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
            </div>
            <button class="remove-file" onclick="removeFile(${index})">✕</button>
        </div>
    `).join('');
}

async function uploadFiles() {
    if (selectedFiles.length === 0) return;
    
    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));
    
    showLoading('Uploading and processing PDFs...');
    
    try {
        const response = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const result = await response.json();
        
        showToast(`✅ Uploaded ${result.files_processed} file(s), ${result.total_chunks} chunks created`, 'success');
        
        // Clear selected files
        selectedFiles = [];
        elements.fileInput.value = '';
        renderFileList();
        elements.uploadBtn.disabled = true;
        
        // Reload system status
        await loadSystemStatus();
        
    } catch (error) {
        console.error('Upload error:', error);
        showToast(`Upload failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// ============================================================================
// System Status & Knowledge Base
// ============================================================================

async function loadSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        
        // Update status indicator
        const statusDot = elements.systemStatus.querySelector('.status-dot');
        const statusText = elements.systemStatus.querySelector('.status-text');
        statusDot.style.background = '#10B981';
        statusText.textContent = 'Online';
        
        // Update KB stats
        elements.docCount.textContent = data.statistics.total_documents;
        elements.chunkCount.textContent = data.statistics.total_chunks;
        
        // Enable/disable send button based on KB status
        updateSendButtonState();
        
    } catch (error) {
        console.error('Status load error:', error);
        const statusDot = elements.systemStatus.querySelector('.status-dot');
        const statusText = elements.systemStatus.querySelector('.status-text');
        statusDot.style.background = '#EF4444';
        statusText.textContent = 'Offline';
    }
}

async function clearKnowledgeBase() {
    if (!confirm('Are you sure you want to clear all documents? This cannot be undone.')) {
        return;
    }
    
    showLoading('Clearing knowledge base...');
    
    try {
        const response = await fetch(`${API_BASE}/clear`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Failed to clear');
        
        showToast('Knowledge base cleared', 'success');
        await loadSystemStatus();
        
    } catch (error) {
        console.error('Clear error:', error);
        showToast(`Failed to clear: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// ============================================================================
// Chat & Query Handling
// ============================================================================

function handleInputChange() {
    const text = elements.queryInput.value;
    elements.charCounter.textContent = `${text.length}/1000`;
    updateSendButtonState();
    
    // Auto-resize textarea
    elements.queryInput.style.height = 'auto';
    elements.queryInput.style.height = elements.queryInput.scrollHeight + 'px';
}

function updateSendButtonState() {
    const hasText = elements.queryInput.value.trim().length > 0;
    const hasKB = parseInt(elements.chunkCount.textContent) > 0;
    elements.sendBtn.disabled = !hasText;
}

async function sendQuery() {
    const query = elements.queryInput.value.trim();
    if (!query) return;
    
    // Add user message
    addMessage('user', query);
    
    // Clear input
    elements.queryInput.value = '';
    elements.charCounter.textContent = '0/1000';
    elements.queryInput.style.height = 'auto';
    updateSendButtonState();
    
    // Show loading
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                top_k: 5,
                include_sources: true
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }
        
        const result = await response.json();
        
        // Remove loading message
        removeMessage(loadingId);
        
        // Add assistant message
        addMessage('assistant', result.answer, result);
        
    } catch (error) {
        console.error('Query error:', error);
        removeMessage(loadingId);
        addMessage('assistant', `❌ Error: ${error.message}`, { intent: 'error' });
    }
}

function addMessage(sender, text, metadata = null) {
    // Remove welcome message if exists
    const welcomeMsg = elements.messagesContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.setAttribute('data-id', Date.now());
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const avatar = sender === 'user' ? '👤' : '🤖';
    const senderName = sender === 'user' ? 'You' : 'Assistant';
    
    let html = `
        <div class="message-header">
            <span class="message-avatar">${avatar}</span>
            <span class="message-sender">${senderName}</span>
            <span class="message-time">${time}</span>
        </div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(text)}</div>
    `;
    
    // Add metadata for assistant messages
    if (sender === 'assistant' && metadata) {
        html += `
            <div class="message-metadata">
                ${metadata.intent ? `<span class="intent-badge">${metadata.intent}</span>` : ''}
                ${metadata.metadata ? `
                    | 🔍 ${metadata.metadata.search_time_ms.toFixed(0)}ms 
                    | 🤖 ${metadata.metadata.llm_time_ms.toFixed(0)}ms 
                    | ⏱️ ${metadata.metadata.total_time_ms.toFixed(0)}ms
                ` : ''}
            </div>
        `;
        
        // Add sources
        if (metadata.sources && metadata.sources.length > 0) {
            html += `
                <div class="sources-section">
                    <div class="sources-header">📚 Sources (${metadata.sources.length})</div>
                    ${metadata.sources.map((source, idx) => `
                        <div class="source-item">
                            <div class="source-header">
                                <span class="source-file">[${idx + 1}] ${source.source_file}</span>
                                <div class="source-meta">
                                    <span>Page ${source.page_number}</span>
                                    <span>Score: ${source.similarity_score.toFixed(3)}</span>
                                </div>
                            </div>
                            <div class="source-text">${escapeHtml(truncateText(source.text, 150))}</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }
    }
    
    html += '</div>';
    messageDiv.innerHTML = html;
    
    elements.messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv.getAttribute('data-id');
}

function addLoadingMessage() {
    const id = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.setAttribute('data-id', id);
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-avatar">🤖</span>
            <span class="message-sender">Assistant</span>
        </div>
        <div class="message-content">
            <div class="message-text">Thinking...</div>
        </div>
    `;
    elements.messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const message = elements.messagesContainer.querySelector(`[data-id="${id}"]`);
    if (message) message.remove();
}

function clearChat() {
    if (!confirm('Clear all messages?')) return;
    
    elements.messagesContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">👋</div>
            <h3>Chat cleared!</h3>
            <p>Ask a question about your documents to continue.</p>
        </div>
    `;
}

function scrollToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

// ============================================================================
// UI Helpers
// ============================================================================

function showLoading(text = 'Processing...') {
    elements.loadingOverlay.querySelector('.loading-text').textContent = text;
    elements.loadingOverlay.classList.add('active');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️'
    };
    
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
        <span class="toast-message">${message}</span>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'toastSlide 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function autoResizeTextarea() {
    elements.queryInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

// ============================================================================
// Utility Functions
// ============================================================================

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Make removeFile available globally (called from rendered HTML)
window.removeFile = removeFile;

console.log('🚀 RAG FastAPI Frontend initialized!');

