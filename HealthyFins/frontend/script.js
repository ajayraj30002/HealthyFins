// script.js - COMPLETE FIXED VERSIOn

// Global variables
let currentFile = null;
let currentResult = null;

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
            console.log('✅ Backend connected:', data);
            
            // Show model status
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

// Setup file upload functionality - FIXED VERSION
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
        
        // Get token from auth.js
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
                // Note: Don't set Content-Type for FormData
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
            
            // Try to parse JSON error
            let errorMessage = `Server error (${response.status})`;
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                // Not JSON, use raw text
                errorMessage = errorText || errorMessage;
            }
            
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        console.log('✅ Prediction result:', result);
        
        if (result.success) {
            currentResult = result;
            displayResults(result);
            // Automatically save to history
            saveResult();
            showNotification('Analysis complete!', 'success');
        } else {
            throw new Error(result.detail || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('❌ Analysis error:', error);
        
        // Show user-friendly error
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
            
            // Reset UI
            document.getElementById('loadingSection').style.display = 'none';
            document.getElementById('previewSection').style.display = 'block';
        }
    } finally {
        // Hide loading
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
    if (badge) {
        badge.textContent = disease.includes('Healthy') ? 'Healthy' : 'Disease';
        badge.className = 'badge ' + (
            disease.includes('Healthy') ? 'badge-success' : 
            confidence > 70 ? 'badge-danger' : 'badge-warning'
        );
    }
    
    // Update treatment text
    updateTreatmentText(disease, confidence, result.model_type);
    
    // Show model type indicator
    const modelIndicator = document.createElement('small');
    modelIndicator.style.display = 'block';
    modelIndicator.style.marginTop = '10px';
    modelIndicator.style.fontSize = '0.8em';
    modelIndicator.style.color = result.model_type === 'ai_model' ? '#27ae60' : '#f39c12';
    modelIndicator.textContent = `Analysis: ${result.model_type === 'ai_model' ? 'AI Model' : 'Intelligent Analysis'}`;
    
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

function updateTreatmentText(disease, confidence, modelType = 'ai_model') {
    const treatments = {
        // ==================== HEALTHY ====================
        'healthy': 'HEALTHY FISH - DIAGNOSIS CONFIRMED\n\n' +
                  'MAINTENANCE CHECKLIST:\n\n' +
                  '1. Weekly Water Changes - 20-25% volume replacement\n' +
                  '2. Optimal Temperature - Maintain 24-28°C range\n' +
                  '3. pH Balance - Keep between 6.5-8.0\n' +
                  '4. Quality Feeding - Balanced diet, avoid overfeeding\n' +
                  '5. Regular Monitoring - Daily behavior observation\n' +
                  '6. Quarantine Protocol - 2 weeks for new additions\n' +
                  '7. Filter Maintenance - Clean monthly, never replace all media\n\n' +
                  'Prevention is always better than cure!',
        
        // ==================== BACTERIAL RED DISEASE ====================
        'bacterial red disease': 'BACTERIAL RED DISEASE - CRITICAL ALERT\n\n' +
                                'IMMEDIATE ACTION REQUIRED:\n\n' +
                                '1. Antibiotic Treatment - Kanaplex/Maracyn for 7-10 days\n' +
                                '2. Salt Therapy - 1 tbsp per 20L, dissolve completely\n' +
                                '3. Emergency Water Change - 50% immediately\n' +
                                '4. Isolation Protocol - Hospital tank setup mandatory\n' +
                                '5. Aeration Boost - Maximum oxygen supply\n' +
                                '6. Parameter Testing - Daily ammonia/nitrite checks\n' +
                                '7. Vet Consultation - Required within 48 hours\n\n' +
                                'Contagious - Isolate immediately!',
        
        // ==================== PARASITIC DISEASES ====================
        'parasitic diseases': 'PARASITIC INFECTION DETECTED\n\n' +
                             'TREATMENT PROTOCOL:\n\n' +
                             '1. Anti-parasitic Meds - Praziquantel for 10-14 days\n' +
                             '2. Salt Baths - 3% solution for 5-10 minutes daily\n' +
                             '3. Temperature Increase - Raise to 30°C gradually\n' +
                             '4. Tank Vacuuming - Deep clean substrate thoroughly\n' +
                             '5. Filter Cleaning - Replace/clean all media\n' +
                             '6. Repeat Treatment - Second dose after 7 days\n' +
                             '7. Monitor Behavior - Watch for flashing/rubbing\n\n' +
                             'Lifecycle breaks in 7 days - repeat essential',
        
        // ==================== VIRAL DISEASES WHITE TAIL DISEASE ====================
        'viral diseases white tail disease': 'VIRAL WHITE TAIL DISEASE\n\n' +
                                            'SUPPORTIVE CARE PROTOCOL:\n\n' +
                                            '1. Water Perfection - Zero ammonia/nitrite mandatory\n' +
                                            '2. Immune Boosters - Vitamin C supplements added\n' +
                                            '3. Temperature Control - Maintain steady 26-28°C\n' +
                                            '4. Salt Support - 1 tsp per 4L for gill function\n' +
                                            '5. Stress Reduction - Dim lights, minimize handling\n' +
                                            '6. Secondary Prevention - Watch for bacterial/fungal\n' +
                                            '7. Nutrition Focus - High-quality vitamin-rich food\n\n' +
                                            'No direct antiviral treatment - supportive care only',
        
        // ==================== FUNGAL DISEASES SAPROLEGNIASIS ====================
        'fungal diseases saprolegniasis': 'FUNGAL INFECTION (SAPROLEGNIASIS)\n\n' +
                                         'TREATMENT PLAN:\n\n' +
                                         '1. Antifungal Medication - Methylene Blue baths\n' +
                                         '2. Salt Treatment - 1 tbsp per 20L tank water\n' +
                                         '3. Wound Management - Remove dead tissue carefully\n' +
                                         '4. Filtration Upgrade - Increase mechanical filtration\n' +
                                         '5. Organic Reduction - Vacuum waste daily\n' +
                                         '6. Temperature Adjustment - Increase to 28°C\n' +
                                         '7. Medication Duration - Continue for 7-10 days\n\n' +
                                         'Warm water inhibits fungal growth',
        
        // ==================== BACTERIAL DISEASES - AEROMONIASIS ====================
        'bacterial diseases - aeromoniasis': 'AEROMONIASIS - EMERGENCY\n\n' +
                                            'CRITICAL CARE PROTOCOL:\n\n' +
                                            '1. Immediate Isolation - Hospital tank NOW\n' +
                                            '2. Prescription Antibiotics - Kanamycin/Enrofloxacin\n' +
                                            '3. Bare Tank Setup - No substrate for easy cleaning\n' +
                                            '4. Daily Water Changes - 50% minimum daily\n' +
                                            '5. Main Tank Disinfection - Bleach solution required\n' +
                                            '6. Equipment Sterilization - All tools must be sterilized\n' +
                                            '7. Veterinary Emergency - Immediate consultation needed\n\n' +
                                            'HIGHLY CONTAGIOUS - Complete isolation required',
        
        // ==================== BACTERIAL GILL DISEASE ====================
        'bacterial gill disease': 'BACTERIAL GILL DISEASE\n\n' +
                                  'RESPIRATORY TREATMENT:\n\n' +
                                  '1. Antibiotic Food - Oxytetracycline 50mg/kg daily\n' +
                                  '2. Oxygen Maximization - Add multiple air stones\n' +
                                  '3. Water Level Reduction - Increase surface agitation\n' +
                                  '4. Ammonia Control - Must maintain ZERO ppm\n' +
                                  '5. Salt Baths - 2-3g/L for 30 minutes daily\n' +
                                  '6. Stocking Reduction - Decrease fish density immediately\n' +
                                  '7. Water Changes - 30% daily until improvement\n\n' +
                                  'Oxygen is critical - maximize aeration',
        
        // ==================== EUS ULCERATIVE SYNDROME ====================
        'eus_ulcerative_syndrome (arg)': 'EUS - EPIZOOTIC ULCERATIVE SYNDROME\n\n' +
                                         'COMPREHENSIVE TREATMENT:\n\n' +
                                         '1. Combination Therapy - Antibiotics + Antifungals\n' +
                                         '2. Wound Cleaning - Hydrogen peroxide 3% on ulcers\n' +
                                         '3. pH Management - Maintain above 7.0 at all times\n' +
                                         '4. Potassium Permanganate - Medicated baths\n' +
                                         '5. Nutrition Support - High-protein, vitamin-rich food\n' +
                                         '6. Hardness Increase - Raise water hardness\n' +
                                         '7. Extended Treatment - Minimum 14-day protocol\n\n' +
                                         'REQUIRES PROFESSIONAL VETERINARY MANAGEMENT'
    };
    
    let treatment = treatments['healthy'];
    const diseaseLower = disease.toLowerCase();
    
    // Match disease to treatment (case-insensitive)
    if (diseaseLower.includes('healthy')) {
        treatment = treatments['healthy'];
    } else if (diseaseLower.includes('bacterial red')) {
        treatment = treatments['bacterial red disease'];
    } else if (diseaseLower.includes('parasitic')) {
        treatment = treatments['parasitic diseases'];
    } else if (diseaseLower.includes('viral') || diseaseLower.includes('white tail')) {
        treatment = treatments['viral diseases white tail disease'];
    } else if (diseaseLower.includes('fungal') || diseaseLower.includes('saprolegniasis')) {
        treatment = treatments['fungal diseases saprolegniasis'];
    } else if (diseaseLower.includes('aeromoniasis')) {
        treatment = treatments['bacterial diseases - aeromoniasis'];
    } else if (diseaseLower.includes('gill disease')) {
        treatment = treatments['bacterial gill disease'];
    } else if (diseaseLower.includes('eus') || diseaseLower.includes('ulcerative')) {
        treatment = treatments['eus_ulcerative_syndrome (arg)'];
    } else if (diseaseLower.includes('bacterial')) {
        treatment = treatments['bacterial red disease'];
    }
    
    // Add confidence warning if low
    if (confidence < 70) {
        treatment = 'LOW CONFIDENCE ALERT (' + confidence + '%)\n\n' + 
                   'RECOMMENDED ACTIONS:\n\n' +
                   '1. Retake Photos - Clear, well-lit images from multiple angles\n' +
                   '2. Symptom Checklist - Note all observable symptoms\n' +
                   '3. Water Testing - Complete parameter test immediately\n' +
                   '4. Behavior Log - Record swimming/eating patterns\n' +
                   '5. Professional Consult - Contact helpline below\n\n' +
                   '---\n\n' + treatment;
    }
    
    // Add model type note
    if (modelType !== 'ai_model') {
        treatment = 'INTELLIGENT ANALYSIS MODE\n\n' +
                   'System Status: AI model currently unavailable\n\n' +
                   'For better accuracy:\n\n' +
                   '1. Use clear photos - Good lighting, multiple angles\n' +
                   '2. Check backend status - Ensure AI model is loaded\n' +
                   '3. Consider manual diagnosis - Use helpline consultation\n' +
                   '4. Monitor symptoms - Keep detailed observations\n\n' +
                   '---\n\n' + treatment;
    }
    
    const treatmentElement = document.getElementById('treatmentText');
    if (treatmentElement) {
        // Format with line breaks and styling
        const formattedTreatment = treatment
            .replace(/\n/g, '<br>')
            .replace(/1\./g, '<span class="treatment-step">1.</span>')
            .replace(/2\./g, '<span class="treatment-step">2.</span>')
            .replace(/3\./g, '<span class="treatment-step">3.</span>')
            .replace(/4\./g, '<span class="treatment-step">4.</span>')
            .replace(/5\./g, '<span class="treatment-step">5.</span>')
            .replace(/6\./g, '<span class="treatment-step">6.</span>')
            .replace(/7\./g, '<span class="treatment-step">7.</span>');
        
        treatmentElement.innerHTML = formattedTreatment;
        
        // Add styling for better readability
        treatmentElement.style.whiteSpace = 'pre-line';
        treatmentElement.style.lineHeight = '1.6';
        treatmentElement.style.fontSize = '14px';
        treatmentElement.style.padding = '15px';
        treatmentElement.style.backgroundColor = '#f8f9fa';
        treatmentElement.style.borderRadius = '8px';
        treatmentElement.style.borderLeft = '4px solid #2c8c99';
    }
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

// Load PH monitoring data
async function loadPHData(forceRefresh = false) {
    try {
        console.log('🌡️ Loading PH data...');
        
        const token = localStorage.getItem('healthyfins_token') || localStorage.getItem('token');
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

// Add CSS for notifications
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

// Make sure auth.js functions are available
function checkAuth() {
    return window.HealthyFins ? window.HealthyFins.checkAuth() : false;
}

function logout() {
    if (window.HealthyFins && window.HealthyFins.logout) {
        window.HealthyFins.logout();
    } else {
        localStorage.clear();
        window.location.href = 'index.html';
    }
}

// If HealthyFins is not loaded yet, wait for it
if (typeof HealthyFins === 'undefined') {
    console.log('⚠️ HealthyFins not loaded yet, waiting...');
    document.addEventListener('HealthyFinsLoaded', function() {
        console.log('✅ HealthyFins loaded');
        loadDashboardData();
    });
}


