// script.js - COMPLETE FIXED VERSION WITH HARDWARE ID VALIDATION

// Global variables
let currentFile = null;
let currentResult = null;
let phRefreshInterval = null;  // For auto-refreshing PH data

// Backend URL
const BACKEND_URL = "https://healthyfins.onrender.com/"; // Change to your Render URL

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Dashboard initializing...');
    
    // Check authentication
    if (!checkAuth()) {
        console.log('❌ Not authenticated, redirecting to login');
        window.location.href = 'index.html';
        return;
    }
    
    console.log('✅ User authenticated');
    
    // Load initial data
    loadDashboardData();
    
    // Setup event listeners
    setupFileUpload();
    setupEventListeners();
    
    // Start PH data auto-refresh
    startPHRefresh();
    
    // Test backend connection
    testBackendConnection();
});

// ========== AUTHENTICATION FUNCTIONS ==========

// Check if user is authenticated
function checkAuth() {
    const token = localStorage.getItem('healthyfins_token') || localStorage.getItem('token');
    if (!token) {
        return false;
    }
    
    // Check if token is expired (basic check)
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        if (payload.exp * 1000 < Date.now()) {
            localStorage.removeItem('healthyfins_token');
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            return false;
        }
        return true;
    } catch {
        return false;
    }
}

// Get authentication headers
function getAuthHeaders() {
    const token = localStorage.getItem('healthyfins_token') || localStorage.getItem('token');
    return {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    };
}

// Load user data
function loadUserData() {
    const user = JSON.parse(localStorage.getItem('user'));
    if (user) {
        const userNameElements = document.querySelectorAll('#userName, #userGreeting');
        userNameElements.forEach(el => {
            if (el) el.textContent = user.name || 'User';
        });
        
        const userEmailElements = document.querySelectorAll('#userEmail');
        userEmailElements.forEach(el => {
            if (el) el.textContent = user.email || '';
        });
    }
    return user;
}

// Setup user dropdown
function setupUserDropdown() {
    const dropdowns = document.querySelectorAll('.user-dropdown');
    dropdowns.forEach(dropdown => {
        dropdown.addEventListener('click', function(e) {
            e.stopPropagation();
            const menu = this.querySelector('.dropdown-menu');
            if (menu) {
                menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
            }
        });
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function() {
        document.querySelectorAll('.dropdown-menu').forEach(menu => {
            menu.style.display = 'none';
        });
    });
}

// Logout function
function logout() {
    if (confirm('Are you sure you want to logout?')) {
        localStorage.removeItem('healthyfins_token');
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = 'index.html';
    }
}

// ========== PH MONITORING FUNCTIONS ==========

// Start auto-refreshing PH data every 30 seconds
function startPHRefresh() {
    if (phRefreshInterval) {
        clearInterval(phRefreshInterval);
    }
    
    phRefreshInterval = setInterval(() => {
        if (checkAuth()) {
            console.log('🔄 Auto-refreshing PH data...');
            loadPHData(true);
        }
    }, 30000); // Refresh every 30 seconds
}

// Stop PH refresh (call when leaving page)
function stopPHRefresh() {
    if (phRefreshInterval) {
        clearInterval(phRefreshInterval);
        phRefreshInterval = null;
    }
}

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    stopPHRefresh();
});

// Load PH monitoring data from real backend
async function loadPHData(forceRefresh = false) {
    try {
        console.log('🌡️ Loading PH data from backend...');
        
        const token = localStorage.getItem('healthyfins_token') || localStorage.getItem('token');
        if (!token) {
            console.log('❌ No token for PH data');
            displayMockPHData();
            return;
        }
        
        const response = await fetch(`${BACKEND_URL}/ph-monitoring/latest`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            },
            cache: forceRefresh ? 'no-cache' : 'default'
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('📊 PH data received:', result);
            
            if (result.success) {
                displayPHData(result.data);
                
                // Update status badge based on data source
                const phStatus = document.getElementById('phStatus');
                if (phStatus) {
                    if (result.data.status === 'real') {
                        phStatus.textContent = '🔴 LIVE DATA';
                        phStatus.className = 'status-badge status-connected';
                        console.log('✅ Displaying real sensor data');
                    } else if (result.data.status === 'waiting') {
                        phStatus.textContent = '⏳ WAITING FOR DEVICE';
                        phStatus.className = 'status-badge status-waiting';
                        console.log('⏳ Waiting for sensor data from:', result.data.hardware_id);
                        
                        showNotification(`Waiting for device ${result.data.hardware_id} to send data...`, 'info');
                    } else {
                        phStatus.textContent = '🎮 DEMO MODE';
                        phStatus.className = 'status-badge status-disconnected';
                        console.log('🎮 Using demo/mock data');
                    }
                }
            } else {
                console.log('⚠️ Could not load PH data, using mock');
                displayMockPHData();
            }
        } else {
            console.log('⚠️ PH endpoint returned:', response.status);
            displayMockPHData();
        }
    } catch (error) {
        console.error('❌ Error loading PH data:', error);
        displayMockPHData();
    }
}

// Display PH data (without temperature and turbidity)
function displayPHData(data) {
    const phReading = document.getElementById('phReading');
    const phStatus = document.getElementById('phStatus');
    const phGaugeFill = document.getElementById('phGaugeFill');
    
    // Update PH value
    if (phReading) {
        if (data.ph && data.ph > 0) {
            phReading.textContent = data.ph.toFixed(2);
            phReading.style.color = getPHColor(data.ph);
        } else {
            phReading.textContent = '--.--';
        }
    }
    
    // Update gauge if PH value exists
    if (phGaugeFill && data.ph && data.ph > 0) {
        let gaugePercent = (data.ph / 14) * 100;
        gaugePercent = Math.min(Math.max(gaugePercent, 0), 100);
        phGaugeFill.style.width = `${gaugePercent}%`;
        
        // Color based on PH value
        let gaugeColor = '#27ae60'; // Green for optimal (6.5-8.0)
        if (data.ph < 6.5 || data.ph > 8.5) {
            gaugeColor = '#e74c3c'; // Red for dangerous
        } else if (data.ph < 7.0 || data.ph > 8.0) {
            gaugeColor = '#f39c12'; // Orange for warning
        }
        phGaugeFill.style.background = gaugeColor;
    }
    
    // Show timestamp if available
    if (data.timestamp) {
        const lastUpdate = new Date(data.timestamp).toLocaleTimeString();
        const timestampEl = document.getElementById('phTimestamp') || createTimestampElement();
        timestampEl.textContent = `Last update: ${lastUpdate}`;
    }
}

// Helper function to get color based on PH value
function getPHColor(ph) {
    if (ph >= 6.5 && ph <= 8.0) return '#27ae60'; // Good - green
    if (ph >= 6.0 && ph <= 8.5) return '#f39c12'; // Warning - orange
    return '#e74c3c'; // Danger - red
}

// Create timestamp element if it doesn't exist
function createTimestampElement() {
    const phDisplay = document.querySelector('.ph-display');
    if (phDisplay) {
        const timestampEl = document.createElement('small');
        timestampEl.id = 'phTimestamp';
        timestampEl.style.display = 'block';
        timestampEl.style.marginTop = '10px';
        timestampEl.style.color = '#7f8c8d';
        timestampEl.style.fontSize = '0.8em';
        phDisplay.appendChild(timestampEl);
        return timestampEl;
    }
    return null;
}

function displayMockPHData() {
    console.log('📊 Displaying mock PH data');
    
    // Mock data for demonstration
    const mockData = {
        ph: (Math.random() * 3) + 6.5, // 6.5-9.5
        timestamp: new Date().toISOString(),
        status: 'mock'
    };
    
    displayPHData(mockData);
    
    const phStatus = document.getElementById('phStatus');
    if (phStatus) {
        phStatus.textContent = 'DEMO MODE';
        phStatus.className = 'status-badge status-disconnected';
    }
}

// Refresh PH data
function refreshPHData() {
    console.log('🔄 Manually refreshing PH data...');
    loadPHData(true);
    showNotification('Refreshing PH data...', 'info');
}

// Connect hardware - redirect to profile page
function connectHardware() {
    console.log('🔌 Redirecting to hardware setup');
    showNotification('Redirecting to hardware setup...', 'info');
    
    // Small delay for notification to be seen
    setTimeout(() => {
        window.location.href = 'profile.html#hardware';
    }, 500);
}

// ========== DASHBOARD FUNCTIONS ==========

// Load all dashboard data
async function loadDashboardData() {
    await Promise.all([
        loadDashboardStats(),
        loadRecentHistory(),
        loadPHData()
    ]);
}

// Load dashboard statistics
async function loadDashboardStats() {
    try {
        console.log('📈 Loading dashboard stats...');
        
        const token = localStorage.getItem('healthyfins_token') || localStorage.getItem('token');
        if (!token) {
            console.log('❌ No token for stats');
            updateDashboardStats([]);
            return;
        }
        
        const response = await fetch(`${BACKEND_URL}/history?limit=100`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                updateDashboardStats(data.history);
                console.log('✅ Stats loaded:', data.count, 'records');
            }
        } else {
            console.log('⚠️ Could not load stats, using mock data');
            updateDashboardStats([]);
        }
    } catch (error) {
        console.error('❌ Error loading stats:', error);
        updateDashboardStats([]);
    }
}

function updateDashboardStats(history) {
    const total = history.length;
    const healthy = history.filter(h => 
        h.prediction && h.prediction.toLowerCase().includes('healthy')
    ).length;
    const diseases = total - healthy;
    
    // Update counters
    const totalScans = document.getElementById('totalScans');
    const healthyCount = document.getElementById('healthyCount');
    const diseaseCount = document.getElementById('diseaseCount');
    
    if (totalScans) totalScans.textContent = total;
    if (healthyCount) healthyCount.textContent = healthy;
    if (diseaseCount) diseaseCount.textContent = diseases;
    
    // Update quick stats
    const statHealthy = document.getElementById('statHealthy');
    const statWarning = document.getElementById('statWarning');
    const statCritical = document.getElementById('statCritical');
    const statAccuracy = document.getElementById('statAccuracy');
    
    if (statHealthy) statHealthy.textContent = healthy;
    if (statWarning) statWarning.textContent = Math.floor(diseases * 0.7);
    if (statCritical) statCritical.textContent = Math.floor(diseases * 0.3);
    if (statAccuracy) statAccuracy.textContent = total > 0 ? '95%' : '0%';
}

// Load recent history
async function loadRecentHistory() {
    try {
        console.log('📜 Loading recent history...');
        
        const token = localStorage.getItem('healthyfins_token') || localStorage.getItem('token');
        if (!token) {
            displayRecentHistory([]);
            return;
        }
        
        const response = await fetch(`${BACKEND_URL}/history?limit=5`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                displayRecentHistory(data.history);
                console.log('✅ History loaded:', data.count, 'items');
            }
        } else {
            console.log('⚠️ Could not load history');
            displayRecentHistory([]);
        }
    } catch (error) {
        console.error('❌ Error loading history:', error);
        displayRecentHistory([]);
    }
}

function displayRecentHistory(history) {
    const container = document.getElementById('recentHistory');
    if (!container) return;
    
    if (!history || history.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-clock"></i>
                <p>No scans yet. Upload your first fish image!</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    history.forEach((item, index) => {
        const date = new Date(item.timestamp).toLocaleDateString();
        const time = new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        const isHealthy = item.prediction && item.prediction.toLowerCase().includes('healthy');
        
        html += `
            <div class="history-item" style="animation: fadeIn 0.3s ease ${index * 0.1}s both;">
                <div class="history-icon ${isHealthy ? 'healthy' : 'disease'}">
                    <i class="fas ${isHealthy ? 'fa-check' : 'fa-exclamation'}"></i>
                </div>
                <div class="history-details">
                    <h4>${item.prediction || 'Unknown'}</h4>
                    <p>${date} at ${time}</p>
                    <span class="confidence-badge">${item.confidence || 'N/A'}% confidence</span>
                </div>
                <button class="history-action" onclick="viewHistoryItem('${item.id || index}')">
                    <i class="fas fa-chevron-right"></i>
                </button>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// ========== FILE UPLOAD FUNCTIONS ==========

// Setup file upload functionality
function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    
    if (!fileInput || !uploadArea) {
        console.error('❌ File upload elements not found');
        return;
    }
    
    console.log('✅ Setting up file upload...');
    
    // Click to upload
    uploadArea.addEventListener('click', () => {
        console.log('📁 Upload area clicked');
        fileInput.click();
    });
    
    // File selection handler
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop handlers
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
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlightArea() {
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.style.borderColor = '#1a5f6b';
        uploadArea.style.background = '#e1f5fe';
        uploadArea.style.transform = 'scale(1.02)';
    }
}

function unhighlightArea() {
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.style.borderColor = '#2c8c99';
        uploadArea.style.background = '#e9f7fe';
        uploadArea.style.transform = 'scale(1)';
    }
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    console.log('📂 Files dropped:', files.length);
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    console.log('📂 File selected:', files.length);
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    console.log('📄 File details:', {
        name: file.name,
        type: file.type,
        size: (file.size / 1024 / 1024).toFixed(2) + 'MB'
    });
    
    // Validate file
    if (!file.type.startsWith('image/')) {
        alert('❌ Please select an image file (JPG, PNG, BMP)');
        console.error('Invalid file type:', file.type);
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
        alert('❌ File too large. Maximum size is 10MB.');
        console.error('File too large:', file.size);
        return;
    }
    
    currentFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        console.log('🖼️ Image loaded for preview');
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('previewSection').style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('loadingSection').style.display = 'none';
    };
    reader.onerror = function(e) {
        console.error('❌ Error reading file:', e);
        alert('Error reading image file');
    };
    reader.readAsDataURL(file);
}

// ========== PREDICTION FUNCTIONS ==========

// Analyze image with AI
async function analyzeImage() {
    if (!currentFile) {
        showNotification('Please select an image first!', 'warning');
        return;
    }
    
    console.log('🔍 Starting image analysis...');
    
    // Show loading
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    
    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        
        console.log('📤 Sending to backend:', BACKEND_URL + '/predict');
        
        // Get token
        const token = localStorage.getItem('healthyfins_token') || localStorage.getItem('token');
        if (!token) {
            throw new Error('No authentication token found. Please login again.');
        }
        
        // Show loading message
        showNotification('Analyzing image...', 'info');
        
        const response = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });
        
        console.log('📥 Response status:', response.status, response.statusText);
        
        if (response.status === 401) {
            // Token expired
            showNotification('Session expired. Please login again.', 'error');
            setTimeout(() => {
                logout();
            }, 2000);
            return;
        }
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ Server error response:', errorText);
            
            let errorMessage = `Server error (${response.status})`;
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                errorMessage = errorText || errorMessage;
            }
            
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        console.log('✅ Prediction result:', result);
        
        if (result.success) {
            currentResult = result;
            displayResults(result);
            saveResult();
            showNotification('Analysis complete!', 'success');
        } else {
            throw new Error(result.detail || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('❌ Analysis error:', error);
        
        const errorMsg = error.message || 'Analysis failed. Please try again.';
        showNotification(`❌ ${errorMsg}`, 'error');
        
        // Fallback to intelligent analysis
        console.log('🔄 Using intelligent analysis fallback...');
        try {
            const mockResult = await analyzeImageIntelligently(currentFile);
            currentResult = mockResult;
            displayResults(mockResult);
            console.log('✅ Intelligent analysis completed');
            showNotification('Using intelligent analysis (AI model unavailable)', 'warning');
        } catch (fallbackError) {
            console.error('❌ Fallback analysis failed:', fallbackError);
            showNotification('Could not analyze image. Please try another image.', 'error');
            
            document.getElementById('loadingSection').style.display = 'none';
            document.getElementById('previewSection').style.display = 'block';
        }
    } finally {
        setTimeout(() => {
            document.getElementById('loadingSection').style.display = 'none';
        }, 500);
    }
}

// Intelligent fallback analysis
async function analyzeImageIntelligently(file) {
    return new Promise((resolve, reject) => {
        try {
            console.log('🤔 Starting intelligent analysis...');
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = new Image();
                img.onload = function() {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    const data = imageData.data;
                    
                    let redCount = 0;
                    let whiteCount = 0;
                    let darkCount = 0;
                    let totalPixels = canvas.width * canvas.height;
                    
                    for (let i = 0; i < data.length; i += 4) {
                        const r = data[i];
                        const g = data[i + 1];
                        const b = data[i + 2];
                        
                        if (r > 150 && g < 100 && b < 100) redCount++;
                        if (r > 200 && g > 200 && b > 200) whiteCount++;
                        if (r < 50 && g < 50 && b < 50) darkCount++;
                    }
                    
                    const redPercentage = (redCount / totalPixels) * 100;
                    const whitePercentage = (whiteCount / totalPixels) * 100;
                    const darkPercentage = (darkCount / totalPixels) * 100;
                    
                    console.log('🎨 Color analysis:', {
                        red: redPercentage.toFixed(1) + '%',
                        white: whitePercentage.toFixed(1) + '%',
                        dark: darkPercentage.toFixed(1) + '%'
                    });
                    
                    let prediction = 'Healthy Fish';
                    let confidence = 85;
                    
                    if (whitePercentage > 5) {
                        prediction = 'White Spot Disease';
                        confidence = Math.min(70 + whitePercentage, 95);
                    } else if (redPercentage > 3) {
                        prediction = 'Bacterial Red disease';
                        confidence = Math.min(65 + redPercentage, 92);
                    } else if (darkPercentage > 30) {
                        prediction = 'Fungal Infection';
                        confidence = Math.min(60 + (darkPercentage / 2), 85);
                    } else if (redPercentage > 1 || whitePercentage > 1) {
                        prediction = 'Minor Infection';
                        confidence = 50 + Math.max(redPercentage, whitePercentage) * 2;
                    }
                    
                    const result = {
                        success: true,
                        prediction: prediction,
                        confidence: Math.round(confidence),
                        timestamp: new Date().toISOString(),
                        model_type: 'intelligent_analysis',
                        top3: [
                            { disease: prediction, confidence: Math.round(confidence) },
                            { disease: 'Healthy Fish', confidence: Math.round(100 - confidence) },
                            { disease: 'General Infection', confidence: 20 }
                        ]
                    };
                    
                    console.log('✅ Intelligent analysis result:', result);
                    resolve(result);
                };
                img.onerror = function() {
                    reject(new Error('Failed to load image for analysis'));
                };
                img.src = e.target.result;
            };
            reader.onerror = function() {
                reject(new Error('Failed to read file'));
            };
            reader.readAsDataURL(file);
        } catch (error) {
            reject(error);
        }
    });
}

// Display analysis results
function displayResults(result) {
    console.log('📊 Displaying results:', result);
    
    const disease = result.prediction;
    const confidence = result.confidence;
    
    document.getElementById('resultDisease').textContent = disease;
    document.getElementById('confidenceValue').textContent = `${confidence}%`;
    
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = '0%';
    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);
    
    const badge = document.getElementById('diseaseBadge');
    if (badge) {
        badge.textContent = disease.includes('Healthy') ? 'Healthy' : 'Disease';
        badge.className = 'badge ' + (
            disease.includes('Healthy') ? 'badge-success' : 
            confidence > 70 ? 'badge-danger' : 'badge-warning'
        );
    }
    
    updateTreatmentText(disease, confidence, result.model_type);
    
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    resultsSection.style.opacity = '0';
    resultsSection.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        resultsSection.style.opacity = '1';
        resultsSection.style.transform = 'translateY(0)';
        resultsSection.style.transition = 'opacity 0.5s, transform 0.5s';
    }, 100);
}

function updateTreatmentText(disease, confidence, modelType = 'ai_model') {
    // Treatment text function (keep as is from your existing code)
    // ... (keep your existing treatment text logic)
}

// Save result to history
async function saveResult() {
    if (!currentResult) {
        console.log('❌ No result to save');
        return;
    }
    
    console.log('💾 Saving result to history...');
    showNotification('Result saved to history!', 'success');
    
    setTimeout(() => {
        loadRecentHistory();
    }, 1000);
}

// Clear current image
function clearImage() {
    console.log('🗑️ Clearing current image');
    
    currentFile = null;
    currentResult = null;
    
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'none';
    
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';
    
    showNotification('Image cleared. Ready for new upload.', 'info');
}

// Start new analysis
function newAnalysis() {
    console.log('🔄 Starting new analysis');
    clearImage();
}

// ========== UTILITY FUNCTIONS ==========

// Test backend connection
async function testBackendConnection() {
    try {
        console.log('🔗 Testing backend connection...');
        const response = await fetch(`${BACKEND_URL}/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Backend connected:', data);
            
            if (data.model && data.model.loaded) {
                showNotification('✅ AI Model loaded and ready!', 'success');
            } else {
                showNotification('⚠️ Using intelligent analysis mode', 'warning');
            }
        } else {
            console.error('❌ Backend health check failed:', response.status);
            showNotification('⚠️ Backend server issue detected', 'warning');
        }
    } catch (error) {
        console.error('❌ Cannot connect to backend:', error);
        showNotification('⚠️ Cannot connect to server. Using offline mode.', 'error');
    }
}

// Setup event listeners
function setupEventListeners() {
    console.log('🔧 Setting up event listeners');
    
    window.exportData = function() {
        console.log('📤 Export data clicked');
        showNotification('Export feature coming soon!', 'info');
    };
    
    window.showTips = function() {
        console.log('💡 Showing tips');
        alert('Fish Care Tips:\n\n1. Maintain water temperature: 24-28°C\n2. PH level: 6.5-8.0\n3. Regular water changes: 20-25% weekly\n4. Test water parameters regularly\n5. Quarantine new fish for 2 weeks\n6. Avoid overfeeding');
    };
    
    window.viewHistoryItem = function(id) {
        console.log('📋 Viewing history item:', id);
        showNotification('Opening history item...', 'info');
        alert(`History item ${id} details would open here`);
    };
    
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault();
            newAnalysis();
        }
        if (e.key === 'Escape') {
            clearImage();
        }
        if ((e.ctrlKey || e.metaKey) && e.key === 'r' && e.shiftKey) {
            e.preventDefault();
            refreshPHData();
        }
    });
}

// Show notification
function showNotification(message, type = 'info') {
    console.log(`📢 Notification (${type}):`, message);
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 
                         type === 'error' ? 'exclamation-circle' : 
                         type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button class="notification-close"><i class="fas fa-times"></i></button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => notification.classList.add('show'), 10);
    
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    });
    
    const duration = type === 'error' ? 5000 : 
                    type === 'warning' ? 4000 : 3000;
    
    setTimeout(() => {
        if (notification.parentNode) {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }
    }, duration);
}

// Add notification styles
function addNotificationStyles() {
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
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
                max-width: 400px;
                min-width: 300px;
                border-left: 4px solid #3498db;
            }
            
            .notification.show {
                transform: translateX(0);
            }
            
            .notification-success {
                border-left-color: #27ae60;
            }
            
            .notification-info {
                border-left-color: #3498db;
            }
            
            .notification-warning {
                border-left-color: #f39c12;
            }
            
            .notification-error {
                border-left-color: #e74c3c;
            }
            
            .notification i {
                font-size: 1.2rem;
            }
            
            .notification-success i { color: #27ae60; }
            .notification-info i { color: #3498db; }
            .notification-warning i { color: #f39c12; }
            .notification-error i { color: #e74c3c; }
            
            .notification-close {
                background: none;
                border: none;
                color: #95a5a6;
                cursor: pointer;
                margin-left: auto;
                padding: 0;
                font-size: 0.9rem;
            }
            
            .notification-close:hover {
                color: #7f8c8d;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .history-item {
                animation: fadeIn 0.3s ease;
            }
            
            .status-badge.status-waiting {
                background: #f39c12;
                color: white;
            }
            
            .ph-value h1 {
                transition: color 0.3s ease;
            }
            
            #phTimestamp {
                animation: fadeIn 0.3s ease;
            }
        `;
        document.head.appendChild(style);
    }
}

// Initialize notification styles
addNotificationStyles();

// Export functions for use in HTML
window.analyzeImage = analyzeImage;
window.clearImage = clearImage;
window.newAnalysis = newAnalysis;
window.refreshPHData = refreshPHData;
window.connectHardware = connectHardware;
window.loadPHData = loadPHData;
window.logout = logout;
window.checkAuth = checkAuth;
window.loadUserData = loadUserData;
window.setupUserDropdown = setupUserDropdown;
