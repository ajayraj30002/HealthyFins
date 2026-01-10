// script.js - Dashboard functionality

// Global variables
let currentFile = null;
let currentResult = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    if (!checkAuth()) return;
    
    // Load initial data
    loadDashboardData();
    
    // Setup event listeners
    setupFileUpload();
    setupEventListeners();
});

// Load all dashboard data
async function loadDashboardData() {
    await Promise.all([
        loadDashboardStats(),
        loadRecentHistory(),
        loadPHData()
    ]);
}

// Setup file upload functionality
function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    
    if (!fileInput || !uploadArea) return;
    
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlightArea, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlightArea, false);
    });
    
    uploadArea.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlightArea() {
    document.getElementById('uploadArea').style.borderColor = '#1a5f6b';
    document.getElementById('uploadArea').style.background = '#e1f5fe';
}

function unhighlightArea() {
    document.getElementById('uploadArea').style.borderColor = '#2c8c99';
    document.getElementById('uploadArea').style.background = '#e9f7fe';
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    
    // Validate file
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, PNG, BMP)');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
        alert('File too large. Maximum size is 10MB.');
        return;
    }
    
    currentFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('previewSection').style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Analyze image with AI
async function analyzeImage() {
    if (!currentFile) {
        alert('Please select an image first!');
        return;
    }
    
    // Show loading
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('previewSection').style.display = 'none';
    
    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        
        const response = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            currentResult = result;
            displayResults(result);
            // Automatically save to history
            saveResult();
        } else {
            throw new Error(result.detail || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        
        // Fallback to mock data if backend fails
        const mockResult = {
            success: true,
            prediction: Math.random() > 0.5 ? 'Healthy Fish' : 'White Spot Disease',
            confidence: Math.floor(Math.random() * 30) + 70,
            timestamp: new Date().toISOString()
        };
        
        currentResult = mockResult;
        displayResults(mockResult);
        alert('Using mock data. Backend connection failed.');
    }
    
    // Hide loading
    document.getElementById('loadingSection').style.display = 'none';
}

// Display analysis results
function displayResults(result) {
    const disease = result.prediction;
    const confidence = result.confidence;
    
    // Update UI
    document.getElementById('resultDisease').textContent = disease;
    document.getElementById('confidenceValue').textContent = `${confidence}%`;
    document.getElementById('confidenceFill').style.width = `${confidence}%`;
    
    // Update badge
    const badge = document.getElementById('diseaseBadge');
    badge.textContent = disease.includes('Healthy') ? 'Healthy' : 'Disease';
    badge.className = 'badge ' + (disease.includes('Healthy') ? 'badge-success' : 'badge-danger');
    
    // Update treatment text
    updateTreatmentText(disease, confidence);
    
    // Show results
    document.getElementById('resultsSection').style.display = 'block';
}

// Update treatment recommendation
function updateTreatmentText(disease, confidence) {
    const treatments = {
        'healthy': '✅ Your fish appears healthy! Continue regular maintenance: weekly water changes (20-25%), quality fish food, and regular observation for any changes in behavior.',
        'white spot': '🚨 White Spot Disease detected! Immediate action required:\n1. Raise water temperature to 30°C gradually\n2. Add aquarium salt (1 tablespoon per 20 liters)\n3. Use anti-parasitic medication for 10-14 days\n4. Increase aeration during treatment',
        'fin rot': '⚠️ Fin Rot detected! Treatment steps:\n1. Improve water quality immediately (test parameters)\n2. Use antibacterial medication specifically for fin rot\n3. Remove any sharp decorations\n4. Add aquarium salt (1 teaspoon per 4 liters)\n5. Consider isolation if condition worsens',
        'fungal': '⚠️ Fungal Infection detected! Treatment:\n1. Use antifungal medication (methylene blue or similar)\n2. Improve water quality and filtration\n3. Consider salt bath treatment\n4. Remove affected fish if infection is severe',
        'parasite': '🚨 Parasitic Infection detected! Emergency treatment:\n1. Use anti-parasitic medication immediately\n2. Quarantine affected fish if possible\n3. Clean and disinfect tank thoroughly\n4. Treat all fish in the tank\n5. Improve water quality parameters'
    };
    
    let treatment = treatments['healthy'];
    disease = disease.toLowerCase();
    
    if (disease.includes('white spot')) treatment = treatments['white spot'];
    else if (disease.includes('fin rot')) treatment = treatments['fin rot'];
    else if (disease.includes('fungal')) treatment = treatments['fungal'];
    else if (disease.includes('parasite')) treatment = treatments['parasite'];
    
    // Add confidence warning if low
    if (confidence < 70) {
        treatment = '⚠️ Low confidence prediction. ' + treatment + '\n\n🔍 Recommendation: Take clearer photos from multiple angles or consult a veterinarian for confirmation.';
    }
    
    document.getElementById('treatmentText').textContent = treatment;
}

// Save result to history
async function saveResult() {
    if (!currentResult) return;
    
    // Result is already saved by backend during prediction
    showNotification('Result saved to history!', 'success');
    loadRecentHistory(); // Refresh history display
}

// Clear current image
function clearImage() {
    currentFile = null;
    currentResult = null;
    
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('fileInput').value = '';
}

// Start new analysis
function newAnalysis() {
    clearImage();
}

// Load dashboard statistics
async function loadDashboardStats() {
    try {
        const response = await fetch(`${BACKEND_URL}/history?limit=100`, {
            headers: getAuthHeaders()
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                updateDashboardStats(data.history);
            }
        }
    } catch (error) {
        console.error('Error loading stats:', error);
        // Use mock data
        updateDashboardStats([]);
    }
}

function updateDashboardStats(history) {
    const total = history.length;
    const healthy = history.filter(h => h.prediction.toLowerCase().includes('healthy')).length;
    const diseases = total - healthy;
    
    // Update counters
    document.getElementById('totalScans').textContent = total;
    document.getElementById('healthyCount').textContent = healthy;
    document.getElementById('diseaseCount').textContent = diseases;
    
    // Update quick stats
    document.getElementById('statHealthy').textContent = healthy;
    document.getElementById('statWarning').textContent = Math.floor(diseases * 0.7); // Mock
    document.getElementById('statCritical').textContent = Math.floor(diseases * 0.3); // Mock
    document.getElementById('statAccuracy').textContent = '95%'; // Mock
}

// Load recent history
async function loadRecentHistory() {
    try {
        const response = await fetch(`${BACKEND_URL}/history?limit=5`, {
            headers: getAuthHeaders()
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                displayRecentHistory(data.history);
            }
        }
    } catch (error) {
        console.error('Error loading history:', error);
        displayRecentHistory([]);
    }
}

function displayRecentHistory(history) {
    const container = document.getElementById('recentHistory');
    
    if (history.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-clock"></i>
                <p>No scans yet. Upload your first fish image!</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    history.forEach(item => {
        const date = new Date(item.timestamp).toLocaleDateString();
        const time = new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        const isHealthy = item.prediction.toLowerCase().includes('healthy');
        
        html += `
            <div class="history-item">
                <div class="history-icon ${isHealthy ? 'healthy' : 'disease'}">
                    <i class="fas ${isHealthy ? 'fa-check' : 'fa-exclamation'}"></i>
                </div>
                <div class="history-details">
                    <h4>${item.prediction}</h4>
                    <p>${date} at ${time}</p>
                    <span class="confidence-badge">${item.confidence}% confidence</span>
                </div>
                <button class="history-action" onclick="viewHistoryItem(${item.id})">
                    <i class="fas fa-chevron-right"></i>
                </button>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Load PH monitoring data
async function loadPHData(forceRefresh = false) {
    try {
        const response = await fetch(`${BACKEND_URL}/ph-monitoring`, {
            headers: getAuthHeaders()
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                displayPHData(data.data);
            }
        }
    } catch (error) {
        console.error('Error loading PH data:', error);
        displayMockPHData();
    }
}

function displayPHData(data) {
    document.getElementById('phReading').textContent = data.ph.toFixed(2);
    document.getElementById('tempValue').textContent = `${data.temperature}°C`;
    document.getElementById('turbidityValue').textContent = `${data.turbidity} NTU`;
    document.getElementById('phStatus').textContent = 'Connected';
    document.getElementById('phStatus').className = 'status-badge status-connected';
    
    // Update gauge
    const phValue = parseFloat(data.ph);
    let gaugePercent = (phValue / 14) * 100;
    gaugePercent = Math.min(Math.max(gaugePercent, 0), 100);
    document.getElementById('phGaugeFill').style.width = `${gaugePercent}%`;
    
    // Color based on PH value
    let gaugeColor = '#27ae60'; // Green for optimal
    if (phValue < 6.5 || phValue > 8.5) {
        gaugeColor = '#e74c3c'; // Red for dangerous
    } else if (phValue < 7.0 || phValue > 8.0) {
        gaugeColor = '#f39c12'; // Orange for warning
    }
    document.getElementById('phGaugeFill').style.background = gaugeColor;
}

function displayMockPHData() {
    // Mock data for demonstration
    const mockData = {
        ph: (Math.random() * 3) + 6.5, // 6.5-9.5
        temperature: (Math.random() * 5) + 24, // 24-29°C
        turbidity: Math.floor(Math.random() * 50), // 0-50 NTU
        status: 'Mock Data'
    };
    
    displayPHData(mockData);
    document.getElementById('phStatus').textContent = 'Mock Data';
    document.getElementById('phStatus').className = 'status-badge status-disconnected';
}

// Refresh PH data
function refreshPHData() {
    loadPHData(true);
    showNotification('Refreshing PH data...', 'info');
}

// Connect hardware
function connectHardware() {
    alert('Redirecting to hardware setup...');
    window.location.href = 'profile.html#hardware';
}

// Setup event listeners
function setupEventListeners() {
    // Export data
    window.exportData = function() {
        alert('Export feature coming soon!');
    };
    
    // Show tips
    window.showTips = function() {
        alert('Fish Care Tips:\n\n1. Maintain water temperature: 24-28°C\n2. PH level: 6.5-8.0\n3. Regular water changes: 20-25% weekly\n4. Test water parameters regularly\n5. Quarantine new fish for 2 weeks\n6. Avoid overfeeding');
    };
    
    // View history item
    window.viewHistoryItem = function(id) {
        window.location.href = `history.html#item-${id}`;
    };
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Show with animation
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add notification styles
const style = document.createElement('style');
style.textContent = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 9999;
        transform: translateX(150%);
        transition: transform 0.3s ease;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-success {
        border-left: 4px solid #27ae60;
    }
    
    .notification-info {
        border-left: 4px solid #3498db;
    }
    
    .notification-warning {
        border-left: 4px solid #f39c12;
    }
    
    .notification i {
        font-size: 1.2rem;
    }
    
    .notification-success i { color: #27ae60; }
    .notification-info i { color: #3498db; }
    .notification-warning i { color: #f39c12; }
`;
document.head.appendChild(style);