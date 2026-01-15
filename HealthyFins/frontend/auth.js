// auth.js - COMPLETE UPDATED VERSION for Supabase

// ========== CONFIGURATION ==========
const BACKEND_URL = "https://healthyfins.onrender.com";
const FRONTEND_URL = "https://healthy-fins.vercel.app";

// Default demo user
const DEMO_USER = {
    id: "demo123",
    email: "demo@healthyfins.com",
    name: "Demo User",
    hardware_id: "HF-FDDS-001",
    created_at: new Date().toISOString()
};

// ========== AUTH FUNCTIONS ==========

function checkAuth() {
    console.log('🔐 Checking authentication...');
    
    const token = localStorage.getItem('healthyfins_token');
    if (!token) {
        console.log('❌ No authentication token found');
        return false;
    }
    
    // Check if token is expired
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        const isExpired = payload.exp * 1000 < Date.now();
        
        if (isExpired) {
            console.log('❌ Token expired, clearing storage');
            localStorage.removeItem('healthyfins_token');
            localStorage.removeItem('healthyfins_user');
            return false;
        }
        
        console.log('✅ User authenticated');
        return true;
    } catch (error) {
        console.error('❌ Error parsing token:', error);
        return false;
    }
}

function getAuthHeaders(contentType = 'application/json') {
    const token = localStorage.getItem('healthyfins_token');
    
    const headers = {
        'Accept': 'application/json'
    };
    
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    
    if (contentType && contentType !== 'multipart/form-data') {
        headers['Content-Type'] = contentType;
    }
    
    return headers;
}

function loadUserData() {
    console.log('👤 Loading user data...');
    
    try {
        const userStr = localStorage.getItem('healthyfins_user');
        if (!userStr) {
            console.log('❌ No user data found');
            return null;
        }
        
        const user = JSON.parse(userStr);
        console.log('✅ User data loaded:', user.email);
        
        // Update UI elements
        updateUserUI(user);
        
        return user;
    } catch (error) {
        console.error('❌ Error loading user data:', error);
        return null;
    }
}

function updateUserUI(user) {
    // Update username elements
    const userNameElements = document.querySelectorAll('#userName, #userGreeting, .user-name');
    userNameElements.forEach(el => {
        if (el) {
            el.textContent = user.name || user.email;
        }
    });
    
    // Update profile form if exists
    const profileName = document.getElementById('profileName');
    const profileEmail = document.getElementById('profileEmail');
    const profileHardware = document.getElementById('profileHardware');
    
    if (profileName) profileName.value = user.name || '';
    if (profileEmail) profileEmail.value = user.email || '';
    if (profileHardware) profileHardware.value = user.hardware_id || '';
}

function logout() {
    console.log('🚪 Logging out...');
    
    if (confirm('Are you sure you want to logout from HealthyFins?')) {
        // Clear all auth data
        localStorage.removeItem('healthyfins_token');
        localStorage.removeItem('healthyfins_user');
        
        // Clear session data
        sessionStorage.clear();
        
        console.log('✅ All auth data cleared');
        showNotification('Logged out successfully', 'success');
        
        // Redirect to login page
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1000);
    }
}

// ========== NETWORK FUNCTIONS ==========

async function testBackendConnection() {
    console.log("🔗 Testing connection to backend...");
    
    try {
        const response = await fetch(`${BACKEND_URL}/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log("✅ Backend connected");
            return { connected: true, data };
        } else {
            console.error("❌ Backend error:", response.status);
            return { connected: false, error: `HTTP ${response.status}` };
        }
    } catch (error) {
        console.error("❌ Cannot connect to backend:", error.message);
        return { connected: false, error: error.message };
    }
}

async function apiRequest(endpoint, options = {}) {
    console.log(`📤 API Request: ${endpoint}`);
    
    try {
        const headers = getAuthHeaders(options.headers?.['Content-Type']);
        
        const response = await fetch(`${BACKEND_URL}${endpoint}`, {
            ...options,
            headers: { ...headers, ...options.headers }
        });
        
        console.log(`📥 Response status: ${response.status}`);
        
        if (response.status === 401) {
            // Token expired
            showNotification('Session expired. Please login again.', 'error');
            setTimeout(logout, 2000);
            throw new Error('Session expired');
        }
        
        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = `Server error (${response.status})`;
            
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                errorMessage = errorText || errorMessage;
            }
            
            throw new Error(errorMessage);
        }
        
        const data = await response.json();
        return data;
        
    } catch (error) {
        console.error(`❌ API request failed:`, error);
        throw error;
    }
}

// ========== USER MANAGEMENT ==========

async function loginUser(email, password) {
    console.log(`🔐 Attempting login for: ${email}`);
    
    try {
        const formData = new FormData();
        formData.append('email', email);
        formData.append('password', password);
        
        const data = await apiRequest('/login', {
            method: 'POST',
            body: formData
        });
        
        if (data.success) {
            console.log('✅ Login successful for:', data.user.email);
            
            // Store auth data
            localStorage.setItem('healthyfins_token', data.access_token);
            localStorage.setItem('healthyfins_user', JSON.stringify(data.user));
            
            // Show welcome notification
            showNotification(`Welcome back, ${data.user.name}!`, 'success');
            
            return data;
        } else {
            throw new Error(data.detail || 'Login failed');
        }
    } catch (error) {
        console.error('❌ Login error:', error);
        
        // Fallback to demo mode
        if (error.message.includes('Failed to fetch') || error.message.includes('timeout')) {
            console.log('🔄 Backend unavailable, using demo mode');
            return loginDemoUser(email, password);
        }
        
        throw error;
    }
}

async function registerUser(name, email, password, hardwareId = null) {
    console.log(`📝 Attempting registration for: ${email}`);
    
    try {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('email', email);
        formData.append('password', password);
        
        if (hardwareId) {
            // Validate hardware ID format
            if (!validateHardwareId(hardwareId)) {
                throw new Error('Invalid hardware ID format. Use: HF-FDDS-XXX');
            }
            formData.append('hardware_id', hardwareId);
        }
        
        const data = await apiRequest('/register', {
            method: 'POST',
            body: formData
        });
        
        if (data.success) {
            console.log('✅ Registration successful for:', data.user.email);
            
            // Store auth data
            localStorage.setItem('healthyfins_token', data.access_token);
            localStorage.setItem('healthyfins_user', JSON.stringify(data.user));
            
            showNotification(`Welcome to HealthyFins, ${name}!`, 'success');
            
            return data;
        } else {
            throw new Error(data.detail || 'Registration failed');
        }
    } catch (error) {
        console.error('❌ Registration error:', error);
        
        // Fallback to demo mode
        if (error.message.includes('Failed to fetch') || error.message.includes('timeout')) {
            console.log('🔄 Backend unavailable, creating demo account');
            return registerDemoUser(name, email, password, hardwareId);
        }
        
        throw error;
    }
}

// Demo functions for when backend is down
function loginDemoUser(email, password) {
    console.log('🎮 Using demo login mode');
    
    const mockToken = 'demo_token_' + Date.now();
    const mockUser = {
        ...DEMO_USER,
        email: email,
        name: email.split('@')[0].charAt(0).toUpperCase() + email.split('@')[0].slice(1)
    };
    
    localStorage.setItem('healthyfins_token', mockToken);
    localStorage.setItem('healthyfins_user', JSON.stringify(mockUser));
    
    showNotification('Logged in with demo mode', 'warning');
    
    return {
        success: true,
        user: mockUser,
        access_token: mockToken,
        message: 'Demo mode active'
    };
}

function registerDemoUser(name, email, password, hardwareId) {
    console.log('🎮 Creating demo account');
    
    const mockToken = 'demo_token_' + Date.now();
    const mockUser = {
        ...DEMO_USER,
        email: email,
        name: name,
        hardware_id: hardwareId || 'HF-FDDS-001'
    };
    
    localStorage.setItem('healthyfins_token', mockToken);
    localStorage.setItem('healthyfins_user', JSON.stringify(mockUser));
    
    showNotification('Account created in demo mode', 'warning');
    
    return {
        success: true,
        user: mockUser,
        access_token: mockToken,
        message: 'Demo account created'
    };
}

// ========== HARDWARE MANAGEMENT ==========

function validateHardwareId(deviceId) {
    const pattern = /^HF-[A-Z]{2,4}-\d{3}$/;
    return pattern.test(deviceId);
}

async function registerHardware(deviceId) {
    try {
        if (!validateHardwareId(deviceId)) {
            throw new Error('Invalid hardware ID format. Use: HF-FDDS-XXX');
        }
        
        const formData = new FormData();
        formData.append('device_id', deviceId);
        
        const data = await apiRequest('/hardware/register', {
            method: 'POST',
            body: formData
        });
        
        if (data.success) {
            // Update local user data
            const user = JSON.parse(localStorage.getItem('healthyfins_user'));
            user.hardware_id = deviceId;
            localStorage.setItem('healthyfins_user', JSON.stringify(user));
            
            showNotification('Hardware registered successfully!', 'success');
            return data;
        } else {
            throw new Error(data.detail || 'Hardware registration failed');
        }
    } catch (error) {
        console.error('❌ Hardware registration error:', error);
        throw error;
    }
}

async function getHardwareStatus() {
    try {
        const data = await apiRequest('/hardware/status');
        return data;
    } catch (error) {
        console.error('❌ Get hardware status error:', error);
        return { success: false, devices: [] };
    }
}

// ========== USER DATA FUNCTIONS ==========

async function getUserStats() {
    try {
        const data = await apiRequest('/stats');
        return data.stats;
    } catch (error) {
        console.error('❌ Get stats error:', error);
        return {
            total_predictions: 0,
            healthy_count: 0,
            disease_count: 0,
            hardware_connected: false
        };
    }
}

async function getUserHistory(limit = 50, offset = 0) {
    try {
        const data = await apiRequest(`/history?limit=${limit}&offset=${offset}`);
        return data;
    } catch (error) {
        console.error('❌ Get history error:', error);
        return { success: false, history: [] };
    }
}

async function updateUserProfile(name, hardwareId) {
    try {
        const formData = new FormData();
        if (name) formData.append('name', name);
        if (hardwareId) {
            if (!validateHardwareId(hardwareId)) {
                throw new Error('Invalid hardware ID format. Use: HF-FDDS-XXX');
            }
            formData.append('hardware_id', hardwareId);
        }
        
        const data = await apiRequest('/profile', {
            method: 'PUT',
            body: formData
        });
        
        if (data.success) {
            // Update local user data
            const user = JSON.parse(localStorage.getItem('healthyfins_user'));
            if (name) user.name = name;
            if (hardwareId) user.hardware_id = hardwareId;
            localStorage.setItem('healthyfins_user', JSON.stringify(user));
            
            // Update UI
            updateUserUI(user);
            
            showNotification('Profile updated successfully!', 'success');
            return data;
        } else {
            throw new Error(data.detail || 'Profile update failed');
        }
    } catch (error) {
        console.error('❌ Update profile error:', error);
        throw error;
    }
}

async function getUserProfile() {
    try {
        const data = await apiRequest('/profile');
        return data.profile;
    } catch (error) {
        console.error('❌ Get profile error:', error);
        return null;
    }
}

// ========== UI FUNCTIONS ==========

function showNotification(message, type = 'info', duration = 5000) {
    console.log(`📢 Notification (${type}):`, message);
    
    // Remove existing notifications
    const existing = document.querySelectorAll('.healthyfins-notification');
    existing.forEach(n => n.remove());
    
    // Create notification
    const notification = document.createElement('div');
    notification.className = `healthyfins-notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-width: 400px;
        transform: translateX(120%);
        transition: transform 0.3s ease;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        display: flex;
        align-items: center;
        gap: 10px;
    `;
    
    // Set colors
    if (type === 'success') {
        notification.style.background = 'linear-gradient(135deg, #27ae60, #219653)';
    } else if (type === 'error') {
        notification.style.background = 'linear-gradient(135deg, #e74c3c, #c0392b)';
    } else if (type === 'warning') {
        notification.style.background = 'linear-gradient(135deg, #f39c12, #e67e22)';
    } else {
        notification.style.background = 'linear-gradient(135deg, #3498db, #2980b9)';
    }
    
    // Add icon
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';
    if (type === 'warning') icon = '⚠️';
    
    notification.innerHTML = `
        <span style="font-size: 1.2em;">${icon}</span>
        <span style="flex: 1;">${message}</span>
        <button class="notification-close" style="
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 1.1em;
            opacity: 0.8;
            padding: 0 0 0 10px;
        ">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 10);
    
    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.style.transform = 'translateX(120%)';
        setTimeout(() => notification.remove(), 300);
    });
    
    // Auto-remove
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.transform = 'translateX(120%)';
            setTimeout(() => notification.remove(), 300);
        }
    }, duration);
}

function showLoading(message = 'Loading...') {
    const loading = document.createElement('div');
    loading.id = 'healthyfins-loading';
    loading.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(44, 140, 153, 0.95);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 99999;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    `;
    
    loading.innerHTML = `
        <div style="text-align: center;">
            <div style="
                width: 60px;
                height: 60px;
                border: 4px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            "></div>
            <h3 style="margin-bottom: 10px; color: white;">HealthyFins</h3>
            <p style="opacity: 0.9;">${message}</p>
        </div>
    `;
    
    // Add animation style if not exists
    if (!document.querySelector('#healthyfins-spin-style')) {
        const style = document.createElement('style');
        style.id = 'healthyfins-spin-style';
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(loading);
    document.body.style.overflow = 'hidden';
}

function hideLoading() {
    const loading = document.getElementById('healthyfins-loading');
    if (loading) loading.remove();
    document.body.style.overflow = '';
}

function updateConnectionStatus(isConnected, data) {
    const connectionStatus = document.getElementById('connectionStatus');
    if (!connectionStatus) return;
    
    if (isConnected) {
        connectionStatus.className = 'connection-status connection-connected';
        connectionStatus.innerHTML = `
            <i class="fas fa-wifi"></i>
            <span>Connected to HealthyFins</span>
        `;
        connectionStatus.style.display = 'flex';
    } else {
        connectionStatus.className = 'connection-status connection-disconnected';
        connectionStatus.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>Backend unavailable - Using Demo Mode</span>
        `;
        connectionStatus.style.display = 'flex';
    }
}

// ========== INITIALIZATION ==========

function initializePage() {
    console.log('🚀 Initializing HealthyFins page...');
    
    // Test backend connection
    setTimeout(() => {
        testBackendConnection().then(result => {
            updateConnectionStatus(result.connected, result.data);
        });
    }, 1000);
    
    // Check authentication
    const isLoginPage = window.location.pathname.includes('index.html') || 
                       window.location.pathname === '/' ||
                       window.location.pathname.includes('login.html');
    
    if (!isLoginPage && !checkAuth()) {
        console.warn('❌ User not authenticated, redirecting to login');
        showNotification('Please login to continue', 'warning');
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1500);
        return;
    }
    
    // If already logged in and on login page, redirect
    if (isLoginPage && checkAuth()) {
        console.log('✅ User already logged in, redirecting to dashboard');
        window.location.href = 'dashboard.html';
        return;
    }
    
    // Load user data if authenticated
    if (checkAuth()) {
        loadUserData();
    }
    
    // Setup logout button
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
    }
    
    console.log('✅ Page initialization complete');
}

// Add global styles
function addGlobalStyles() {
    if (!document.getElementById('healthyfins-global-styles')) {
        const style = document.createElement('style');
        style.id = 'healthyfins-global-styles';
        style.textContent = `
            .connection-status {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 15px;
                border-radius: 8px;
                font-size: 14px;
                z-index: 1000;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                gap: 8px;
                animation: slideIn 0.3s ease;
            }
            
            .connection-connected {
                background: #27ae60;
                color: white;
            }
            
            .connection-disconnected {
                background: #e74c3c;
                color: white;
            }
            
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// ========== GLOBAL EXPORTS ==========

window.HealthyFins = {
    // Auth functions
    checkAuth,
    loginUser,
    registerUser,
    logout,
    
    // Hardware functions
    validateHardwareId,
    registerHardware,
    getHardwareStatus,
    
    // User data functions
    getUserStats,
    getUserHistory,
    updateUserProfile,
    getUserProfile,
    
    // UI functions
    showNotification,
    showLoading,
    hideLoading,
    
    // Network functions
    apiRequest,
    testBackendConnection,
    getAuthHeaders,
    
    // Configuration
    BACKEND_URL,
    FRONTEND_URL,
    
    // Version
    version: '5.0.0'
};

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        addGlobalStyles();
        initializePage();
    });
} else {
    addGlobalStyles();
    initializePage();
}

console.log('✅ HealthyFins API v5.0.0 loaded successfully');
