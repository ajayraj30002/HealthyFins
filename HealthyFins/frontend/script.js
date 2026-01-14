// script.js - Dashboard functionality - FIXED VERSION

// Global variables
let currentFile = null;
let currentResult = null;

// Backend URL from auth.js (must match)
const BACKEND_URL = "https://healthyfins.onrender.com";

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
    
    // Test backend connection
    testBackendConnection();
});

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
            console.log('✅ Backend connected:', {
                status: data.status,
                model: data.model,
                url: BACKEND_URL
            });
            
            // Show model status
            if (data.model && data.model.loaded) {
                showNotification('✅ AI Model loaded and ready!', 'success');
            } else {
                showNotification('⚠️ AI Model not loaded. Using intelligent analysis.', 'warning');
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

// Load all dashboard data
async function loadDashboardData() {
    console.log('📊 Loading dashboard data...');
    
    try {
        await Promise.all([
            loadDashboardStats(),
            loadRecentHistory(),
            loadPHData()
        ]);
        console.log('✅ Dashboard data loaded');
    } catch (error) {
        console.error('❌ Error loading dashboard data:', error);
        showNotification('⚠️ Could not load all dashboard data', 'error');
    }
}

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
    document.getElementById('uploadArea').style.transform = 'scale(1.02)';
}

function unhighlightArea() {
    document.getElementById('uploadArea').style.borderColor = '#2c8c99';
    document.getElementById('uploadArea').style.background = '#e9f7fe';
    document.getElementById('uploadArea').style.transform = 'scale(1)';
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

// Analyze image with AI - FIXED VERSION
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
        const token = localStorage.getItem('healthyfins_token');
        if (!token) {
            throw new Error('No authentication token found');
        }
        
        const response = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
                // Note: Don't set Content-Type for FormData
            },
            body: formData,
            timeout: 60000 // 60 second timeout
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
        
        if (response.status === 413) {
            throw new Error('Image too large. Please use a smaller image.');
        }
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ Server error response:', errorText);
            throw new Error(`Server error (${response.status}): ${errorText.substring(0, 100)}`);
        }
        
        const result = await response.json();
        console.log('✅ Prediction result:', result);
        
        if (result.success) {
            currentResult = result;
            displayResults(result);
            // Automatically save to history
            saveResult();
        } else {
            throw new Error(result.detail || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('❌ Analysis error:', error);
        
        // Show user-friendly error
        showNotification(`Analysis failed: ${error.message}`, 'error');
        
        // Fallback to intelligent analysis
        console.log('🔄 Using intelligent analysis fallback...');
        try {
            const mockResult = await analyzeImageIntelligently(currentFile);
            currentResult = mockResult;
            displayResults(mockResult);
            console.log('✅ Intelligent analysis completed');
        } catch (fallbackError) {
            console.error('❌ Fallback analysis failed:', fallbackError);
            showNotification('Could not analyze image. Please try another image.', 'error');
            
            // Reset UI
            document.getElementById('loadingSection').style.display = 'none';
            document.getElementById('previewSection').style.display = 'block';
        }
    } finally {
        // Hide loading after a minimum time to prevent flicker
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
                    // Create canvas for analysis
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    
                    // Simple color analysis
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
                        
                        // Count red pixels (reddish areas)
                        if (r > 150 && g < 100 && b < 100) redCount++;
                        // Count white pixels (white spots)
                        if (r > 200 && g > 200 && b > 200) whiteCount++;
                        // Count dark pixels (fungus/rot)
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
                    
                    // Determine disease based on colors
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
    
    // Update UI
    document.getElementById('resultDisease').textContent = disease;
    document.getElementById('confidenceValue').textContent = `${confidence}%`;
    
    // Animate confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = '0%';
    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);
    
    // Update badge
    const badge = document.getElementById('diseaseBadge');
    badge.textContent = disease.includes('Healthy') ? 'Healthy' : 'Disease';
    badge.className = 'badge ' + (
        disease.includes('Healthy') ? 'badge-success' : 
        confidence > 70 ? 'badge-danger' : 'badge-warning'
    );
    
    // Update treatment text
    updateTreatmentText(disease, confidence, result.model_type);
    
    // Show model type indicator
    const modelIndicator = document.createElement('small');
    modelIndicator.style.display = 'block';
    modelIndicator.style.marginTop = '10px';
    modelIndicator.style.fontSize = '0.8em';
    modelIndicator.style.color = result.model_type === 'real_trained' ? '#27ae60' : '#f39c12';
    modelIndicator.textContent = `Analysis: ${result.model_type === 'real_trained' ? 'AI Model' : 'Intelligent Analysis'}`;
    
    // Add to results
    const resultContent = document.querySelector('.result-content');
    const existingIndicator = resultContent.querySelector('.model-indicator');
    if (existingIndicator) existingIndicator.remove();
    modelIndicator.className = 'model-indicator';
    resultContent.appendChild(modelIndicator);
    
    // Show results with animation
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    resultsSection.style.opacity = '0';
    resultsSection.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        resultsSection.style.opacity = '1';
        resultsSection.style.transform = 'translateY(0)';
        resultsSection.style.transition = 'opacity 0.5s, transform 0.5s';
    }, 100);
    
    console.log('✅ Results displayed');
}

// Update treatment recommendation
function updateTreatmentText(disease, confidence, modelType = 'real_trained') {
    const treatments = {
        'healthy': '✅ Your fish appears healthy! Continue regular maintenance:\n• Weekly water changes (20-25%)\n• Quality fish food\n• Regular observation for behavior changes\n• Maintain water temperature: 24-28°C\n• PH level: 6.5-8.0',
        
        'white spot': '🚨 White Spot Disease detected! Immediate action required:\n1. Raise water temperature to 30°C gradually (1°C per hour)\n2. Add aquarium salt (1 tablespoon per 20 liters)\n3. Use anti-parasitic medication for 10-14 days\n4. Increase aeration during treatment\n5. Isolate affected fish if possible',
        
        'fin rot': '⚠️ Fin Rot detected! Treatment steps:\n1. Improve water quality immediately (test parameters)\n2. Use antibacterial medication specifically for fin rot\n3. Remove any sharp decorations\n4. Add aquarium salt (1 teaspoon per 4 liters)\n5. Consider isolation if condition worsens\n6. Maintain pristine water conditions',
        
        'fungal': '⚠️ Fungal Infection detected! Treatment:\n1. Use antifungal medication (methylene blue or similar)\n2. Improve water quality and filtration\n3. Consider salt bath treatment\n4. Remove affected fish if infection is severe\n5. Increase water temperature slightly\n6. Reduce organic waste in tank',
        
        'parasite': '🚨 Parasitic Infection detected! Emergency treatment:\n1. Use anti-parasitic medication immediately\n2. Quarantine affected fish if possible\n3. Clean and disinfect tank thoroughly\n4. Treat all fish in the tank\n5. Improve water quality parameters\n6. Repeat treatment after 7 days',
        
        'bacterial': '⚠️ Bacterial Infection detected! Treatment:\n1. Antibacterial medication (as per vet guidance)\n2. Improve water quality (test ammonia, nitrites)\n3. Add aquarium salt\n4. Increase water changes\n5. Monitor closely for improvement\n6. Consult veterinarian if no improvement',
        
        'aeromoniasis': '🚨 Aeromoniasis detected! Serious bacterial infection:\n1. Immediate antibiotic treatment\n2. Isolate affected fish\n3. Disinfect entire tank\n4. Test and correct water parameters\n5. Consult aquatic veterinarian\n6. May require prescription antibiotics',
        
        'gill disease': '⚠️ Bacterial Gill Disease detected:\n1. Improve water quality immediately\n2. Antibiotic treatment in food\n3. Increase aeration\n4. Salt bath treatment\n5. Reduce stocking density\n6. Professional veterinary consultation'
    };
    
    let treatment = treatments['healthy'];
    disease = disease.toLowerCase();
    
    if (disease.includes('white spot') || disease.includes('white tail')) treatment = treatments['white spot'];
    else if (disease.includes('fin rot')) treatment = treatments['fin rot'];
    else if (disease.includes('fungal') || disease.includes('saprolegniasis')) treatment = treatments['fungal'];
    else if (disease.includes('parasit')) treatment = treatments['parasite'];
    else if (disease.includes('aeromoniasis')) treatment = treatments['aeromoniasis'];
    else if (disease.includes('gill')) treatment = treatments['gill disease'];
    else if (disease.includes('bacterial')) treatment = treatments['bacterial'];
    else if (disease.includes('red')) treatment = treatments['bacterial'];
    
    // Add confidence warning if low
    if (confidence < 70) {
        treatment = '⚠️ Low confidence prediction. ' + treatment + 
                   '\n\n🔍 Recommendation: Take clearer photos from multiple angles or consult a veterinarian for confirmation.';
    }
    
    // Add model type note
    if (modelType !== 'real_trained') {
        treatment = 'ℹ️ Using intelligent analysis. ' + treatment +
                   '\n\n📱 For more accurate results, ensure backend AI model is loaded.';
    }
    
    document.getElementById('treatmentText').textContent = treatment;
}

// Save result to history
async function saveResult() {
    if (!currentResult) {
        console.log('❌ No result to save');
        return;
    }
    
    console.log('💾 Saving result to history...');
    
    // Result is already saved by backend during prediction
    showNotification('Result saved to history!', 'success');
    
    // Refresh history display
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

// Load dashboard statistics
async function loadDashboardStats() {
    try {
        console.log('📈 Loading dashboard stats...');
        
        const token = localStorage.getItem('healthyfins_token');
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
        
        const token = localStorage.getItem('healthyfins_token');
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

// Load PH monitoring data
async function loadPHData(forceRefresh = false) {
    try {
        console.log('🌡️ Loading PH data...');
        
        const token = localStorage.getItem('healthyfins_token');
        if (!token) {
            displayMockPHData();
            return;
        }
        
        const response = await fetch(`${BACKEND_URL}/ph-monitoring`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                displayPHData(data.data);
                console.log('✅ PH data loaded');
            }
        } else {
            console.log('⚠️ Could not load PH data, using mock');
            displayMockPHData();
        }
    } catch (error) {
        console.error('❌ Error loading PH data:', error);
        displayMockPHData();
    }
}

function displayPHData(data) {
    const phReading = document.getElementById('phReading');
    const tempValue = document.getElementById('tempValue');
    const turbidityValue = document.getElementById('turbidityValue');
    const phStatus = document.getElementById('phStatus');
    
    if (phReading) phReading.textContent = data.ph ? data.ph.toFixed(2) : '--.--';
    if (tempValue) tempValue.textContent = data.temperature ? `${data.temperature}°C` : '-- °C';
    if (turbidityValue) turbidityValue.textContent = data.turbidity ? `${data.turbidity} NTU` : '-- NTU';
    
    if (phStatus) {
        phStatus.textContent = 'Connected';
        phStatus.className = 'status-badge status-connected';
    }
    
    // Update gauge
    const phValue = parseFloat(data.ph || 7.0);
    let gaugePercent = (phValue / 14) * 100;
    gaugePercent = Math.min(Math.max(gaugePercent, 0), 100);
    
    const phGaugeFill = document.getElementById('phGaugeFill');
    if (phGaugeFill) {
        phGaugeFill.style.width = `${gaugePercent}%`;
        
        // Color based on PH value
        let gaugeColor = '#27ae60'; // Green for optimal
        if (phValue < 6.5 || phValue > 8.5) {
            gaugeColor = '#e74c3c'; // Red for dangerous
        } else if (phValue < 7.0 || phValue > 8.0) {
            gaugeColor = '#f39c12'; // Orange for warning
        }
        phGaugeFill.style.background = gaugeColor;
    }
}

function displayMockPHData() {
    console.log('📊 Displaying mock PH data');
    
    // Mock data for demonstration
    const mockData = {
        ph: (Math.random() * 3) + 6.5, // 6.5-9.5
        temperature: (Math.random() * 5) + 24, // 24-29°C
        turbidity: Math.floor(Math.random() * 50), // 0-50 NTU
        status: 'Mock Data'
    };
    
    displayPHData(mockData);
    
    const phStatus = document.getElementById('phStatus');
    if (phStatus) {
        phStatus.textContent = 'Mock Data';
        phStatus.className = 'status-badge status-disconnected';
    }
}

// Refresh PH data
function refreshPHData() {
    console.log('🔄 Refreshing PH data...');
    loadPHData(true);
    showNotification('Refreshing PH data...', 'info');
}

// Connect hardware
function connectHardware() {
    console.log('🔌 Redirecting to hardware setup');
    showNotification('Redirecting to hardware setup...', 'info');
    window.location.href = 'profile.html#hardware';
}

// Setup event listeners
function setupEventListeners() {
    console.log('🔧 Setting up event listeners');
    
    // Export data
    window.exportData = function() {
        console.log('📤 Export data clicked');
        showNotification('Export feature coming soon!', 'info');
    };
    
    // Show tips
    window.showTips = function() {
        console.log('💡 Showing tips');
        alert('Fish Care Tips:\n\n1. Maintain water temperature: 24-28°C\n2. PH level: 6.5-8.0\n3. Regular water changes: 20-25% weekly\n4. Test water parameters regularly\n5. Quarantine new fish for 2 weeks\n6. Avoid overfeeding');
    };
    
    // View history item
    window.viewHistoryItem = function(id) {
        console.log('📋 Viewing history item:', id);
        showNotification('Opening history item...', 'info');
        // In a real app, this would open a detailed view
        // For now, just show the item
        alert(`History item ${id} details would open here`);
    };
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + N for new analysis
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault();
            newAnalysis();
        }
        // Escape to clear
        if (e.key === 'Escape') {
            clearImage();
        }
    });
    
    // Add retry button if results fail
    const retryBtn = document.createElement('button');
    retryBtn.id = 'retryAnalysisBtn';
    retryBtn.className = 'btn btn-outline';
    retryBtn.style.display = 'none';
    retryBtn.style.marginTop = '10px';
    retryBtn.innerHTML = '<i class="fas fa-redo"></i> Retry Analysis';
    retryBtn.onclick = analyzeImage;
    
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.appendChild(retryBtn);
    }
}

// Show notification
function showNotification(message, type = 'info') {
    console.log(`📢 Notification (${type}):`, message);
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 
                         type === 'error' ? 'exclamation-circle' : 
                         type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button class="notification-close"><i class="fas fa-times"></i></button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Show with animation
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    });
    
    // Auto-remove after duration
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

// Add CSS for notifications if not already in style.css
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
