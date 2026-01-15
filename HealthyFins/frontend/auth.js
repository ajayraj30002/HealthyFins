// auth.js - HealthyFins Production Version - COMPLETE FIXED

// ========== CONFIGURATION ==========

// Backend URL - Your Render backend URL
const BACKEND_URL = "https://healthyfins.onrender.com";

// Frontend URL - Your Vercel frontend
const FRONTEND_URL = "https://healthy-fins.vercel.app";

// Default demo user (for testing when backend is down)
const DEMO_USER = {
    id: "demo123",
    email: "demo@healthyfins.com",
    name: "Demo User",
    hardware_id: "DEMO-001",
    created_at: new Date().toISOString()
};

// ========== AUTH FUNCTIONS ==========

// Check if user is authenticated
function checkAuth() {
    console.log('🔐 Checking authentication...');
    
    // Check for token with BOTH keys (for compatibility)
    const token1 = localStorage.getItem('healthyfins_token');
    const token2 = localStorage.getItem('token');
    const token = token1 || token2;
    
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
            localStorage.removeItem('token');
            localStorage.removeItem('healthyfins_user');
            localStorage.removeItem('user');
            return false;
        }
        
        console.log('✅ User authenticated, token valid until:', new Date(payload.exp * 1000).toLocaleString());
        return true;
    } catch (error) {
        console.error('❌ Error parsing token:', error);
        // If we can't parse the token, assume invalid
        return false;
    }
}

// Get authentication headers
function getAuthHeaders(contentType = 'application/json') {
    console.log('🔧 Getting auth headers...');
    
    // Get token with BOTH keys (for compatibility)
    const token1 = localStorage.getItem('healthyfins_token');
    const token2 = localStorage.getItem('token');
    const token = token1 || token2;
    
    if (!token) {
        console.warn('⚠️ No token found for auth headers');
        return {
            'Accept': 'application/json',
            'Content-Type': contentType
        };
    }
    
    const headers = {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/json'
    };
    
    // Only add Content-Type if not FormData (FormData sets its own)
    if (contentType && contentType !== 'multipart/form-data') {
        headers['Content-Type'] = contentType;
    }
    
    console.log('✅ Auth headers prepared');
    return headers;
}

// Load user data from localStorage
function loadUserData() {
    console.log('👤 Loading user data...');
    
    try {
        // Try both keys for compatibility
        const userStr1 = localStorage.getItem('healthyfins_user');
        const userStr2 = localStorage.getItem('user');
        const userStr = userStr1 || userStr2;
        
        if (!userStr) {
            console.log('❌ No user data found in localStorage');
            return null;
        }
        
        const user = JSON.parse(userStr);
        console.log('✅ User data loaded:', user.email);
        
        // Update all user elements on page
        const userNameElements = document.querySelectorAll('#userName, #userGreeting, .user-name');
        userNameElements.forEach(el => {
            if (el) {
                el.textContent = user.name || user.email;
                console.log('✅ Updated username element:', el.id);
            }
        });
        
        const userEmailElements = document.querySelectorAll('#userEmail, .user-email');
        userEmailElements.forEach(el => {
            if (el) el.textContent = user.email;
        });
        
        // Update profile form if exists
        const profileName = document.getElementById('profileName');
        const profileEmail = document.getElementById('profileEmail');
        if (profileName) profileName.value = user.name || '';
        if (profileEmail) profileEmail.value = user.email || '';
        
        return user;
    } catch (error) {
        console.error('❌ Error loading user data:', error);
        return null;
    }
}

// Setup user dropdown
function setupUserDropdown() {
    console.log('🔽 Setting up user dropdown...');
    
    const dropdowns = document.querySelectorAll('.user-dropdown');
    if (dropdowns.length === 0) {
        console.log('ℹ️ No user dropdown found on this page');
        return;
    }
    
    dropdowns.forEach(dropdown => {
        dropdown.addEventListener('click', function(e) {
            e.stopPropagation();
            const menu = this.querySelector('.dropdown-menu');
            if (menu) {
                const isVisible = menu.style.display === 'block';
                menu.style.display = isVisible ? 'none' : 'block';
                console.log('📋 Dropdown toggled:', !isVisible ? 'shown' : 'hidden');
            }
        });
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function() {
        document.querySelectorAll('.dropdown-menu').forEach(menu => {
            if (menu.style.display === 'block') {
                menu.style.display = 'none';
                console.log('📋 Dropdown closed (outside click)');
            }
        });
    });
}

// Logout function
function logout() {
    console.log('🚪 Logging out...');
    
    if (confirm('Are you sure you want to logout from HealthyFins?')) {
        // Clear ALL auth data (both key formats)
        localStorage.removeItem('healthyfins_token');
        localStorage.removeItem('token');
        localStorage.removeItem('healthyfins_user');
        localStorage.removeItem('user');
        
        // Clear any session data
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

// Test backend connection
async function testBackendConnection() {
    console.log("🔗 Testing connection to backend...");
    
    try {
        const startTime = Date.now();
        const response = await fetch(`${BACKEND_URL}/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            },
            // Add timeout using AbortController
            signal: AbortSignal.timeout(10000) // 10 second timeout
        });
        
        const responseTime = Date.now() - startTime;
        
        if (response.ok) {
            const data = await response.json();
            console.log("✅ Backend connected:", {
                status: data.status,
                responseTime: `${responseTime}ms`,
                model: data.model?.loaded ? 'Loaded' : 'Not loaded',
                classes: data.model?.classes_count || 0,
                url: BACKEND_URL
            });
            
            // Update connection status on login page
            updateConnectionStatus(true, data);
            
            return {
                connected: true,
                data: data,
                responseTime: responseTime
            };
        } else {
            console.error("❌ Backend error:", response.status, response.statusText);
            updateConnectionStatus(false, null);
            return {
                connected: false,
                error: `HTTP ${response.status}`,
                responseTime: responseTime
            };
        }
    } catch (error) {
        console.error("❌ Cannot connect to backend:", error.message);
        
        // Show user-friendly message on login page
        if (window.location.pathname.includes('index.html') || 
            window.location.pathname === '/' ||
            window.location.pathname.includes('login')) {
            showNotification(
                '⚠️ Cannot connect to HealthyFins server. The system will use demo mode.',
                'warning'
            );
        }
        
        updateConnectionStatus(false, error);
        
        return {
            connected: false,
            error: error.message,
            responseTime: null
        };
    }
}

// Enhanced fetch with timeout and error handling
async function apiRequest(endpoint, options = {}) {
    console.log(`📤 API Request: ${endpoint}`, options.method || 'GET');
    
    const controller = new AbortController();
    const timeout = 30000; // 30 seconds timeout
    
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
        // Add authorization header if token exists (for protected endpoints)
        const token = localStorage.getItem('healthyfins_token') || localStorage.getItem('token');
        const headers = {
            'Accept': 'application/json',
            ...options.headers
        };
        
        // Add auth header for protected endpoints
        if (token && !endpoint.includes('/login') && !endpoint.includes('/register') && !endpoint.includes('/health')) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        // Log request details (without sensitive data)
        console.log(`🔗 Making request to: ${BACKEND_URL}${endpoint}`);
        
        const response = await fetch(`${BACKEND_URL}${endpoint}`, {
            ...options,
            headers,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        // Log response status
        console.log(`📥 Response status: ${response.status} ${response.statusText}`);
        
        // Handle non-JSON responses
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            console.error('❌ Server returned non-JSON response:', contentType);
            throw new Error(`Server returned non-JSON response: ${contentType}`);
        }
        
        const data = await response.json();
        
        if (!response.ok) {
            console.error(`❌ API Error (${response.status}):`, data.detail || data.message);
            throw new Error(data.detail || data.message || `HTTP ${response.status}`);
        }
        
        console.log(`✅ API Request successful: ${endpoint}`);
        return data;
        
    } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
            console.error(`❌ Request timeout for ${endpoint}`);
            throw new Error('Request timeout. Please check your internet connection.');
        }
        
        console.error(`❌ API request failed for ${endpoint}:`, error);
        throw error;
    }
}

// Login function
async function loginUser(email, password) {
    console.log(`🔐 Attempting login for: ${email}`);
    
    try {
        const formData = new FormData();
        formData.append('email', email);
        formData.append('password', password);
        
        const data = await apiRequest('/login', {
            method: 'POST',
            body: formData
            // Note: Don't set Content-Type header for FormData
        });
        
        if (data.success) {
            console.log('✅ Login successful for:', data.user.email);
            
            // Store auth data with app-specific keys
            localStorage.setItem('healthyfins_token', data.access_token);
            localStorage.setItem('healthyfins_user', JSON.stringify(data.user));
            
            // Also store with generic keys for compatibility
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('user', JSON.stringify(data.user));
            
            // Show welcome notification
            showNotification(`Welcome back, ${data.user.name}!`, 'success');
            
            return data;
        } else {
            console.error('❌ Login failed:', data.detail);
            throw new Error(data.detail || 'Login failed');
        }
    } catch (error) {
        console.error('❌ Login error:', error);
        
        // If backend is down, use demo mode
        if (error.message.includes('timeout') || error.message.includes('Failed to fetch')) {
            console.log('🔄 Backend unavailable, using demo mode');
            return loginDemoUser(email, password);
        }
        
        throw error;
    }
}

// Demo login for when backend is unavailable
function loginDemoUser(email, password) {
    console.log('🎮 Using demo login mode');
    
    // Check if using demo credentials
    const isDemoEmail = email === 'demo@healthyfins.com' || email.includes('demo');
    const isDemoPassword = password === 'demo123' || password.includes('demo');
    
    if (!isDemoEmail && !isDemoPassword) {
        throw new Error('Cannot connect to server. Please try demo credentials: demo@healthyfins.com / demo123');
    }
    
    // Create mock response
    const mockToken = 'demo_token_' + Date.now();
    const mockUser = {
        ...DEMO_USER,
        email: email,
        name: email.split('@')[0].charAt(0).toUpperCase() + email.split('@')[0].slice(1)
    };
    
    // Store demo data
    localStorage.setItem('healthyfins_token', mockToken);
    localStorage.setItem('healthyfins_user', JSON.stringify(mockUser));
    localStorage.setItem('token', mockToken);
    localStorage.setItem('user', JSON.stringify(mockUser));
    
    showNotification('Logged in with demo mode (backend offline)', 'warning');
    
    return {
        success: true,
        user: mockUser,
        access_token: mockToken,
        message: 'Demo mode active - Backend unavailable'
    };
}

// Register function
async function registerUser(name, email, password, hardwareId = null) {
    console.log(`📝 Attempting registration for: ${email}`);
    
    try {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('email', email);
        formData.append('password', password);
        if (hardwareId) {
            formData.append('hardware_id', hardwareId);
        }
        
        const data = await apiRequest('/register', {
            method: 'POST',
            body: formData
        });
        
        if (data.success) {
            console.log('✅ Registration successful for:', data.user.email);
            
            // Store auth data with app-specific keys
            localStorage.setItem('healthyfins_token', data.access_token);
            localStorage.setItem('healthyfins_user', JSON.stringify(data.user));
            
            // Also store with generic keys for compatibility
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('user', JSON.stringify(data.user));
            
            showNotification(`Welcome to HealthyFins, ${name}!`, 'success');
            
            return data;
        } else {
            console.error('❌ Registration failed:', data.detail);
            throw new Error(data.detail || 'Registration failed');
        }
    } catch (error) {
        console.error('❌ Registration error:', error);
        
        // If backend is down, create local demo account
        if (error.message.includes('timeout') || error.message.includes('Failed to fetch')) {
            console.log('🔄 Backend unavailable, creating local demo account');
            return registerDemoUser(name, email, password, hardwareId);
        }
        
        throw error;
    }
}

// Demo registration for when backend is unavailable
function registerDemoUser(name, email, password, hardwareId) {
    console.log('🎮 Creating demo account');
    
    // Create mock response
    const mockToken = 'demo_token_' + Date.now();
    const mockUser = {
        ...DEMO_USER,
        email: email,
        name: name,
        hardware_id: hardwareId || 'DEMO-' + Date.now().toString().slice(-6)
    };
    
    // Store demo data
    localStorage.setItem('healthyfins_token', mockToken);
    localStorage.setItem('healthyfins_user', JSON.stringify(mockUser));
    localStorage.setItem('token', mockToken);
    localStorage.setItem('user', JSON.stringify(mockUser));
    
    showNotification('Account created in demo mode (backend offline)', 'warning');
    
    return {
        success: true,
        user: mockUser,
        access_token: mockToken,
        message: 'Demo account created - Backend unavailable'
    };
}

// ========== UI NOTIFICATION FUNCTIONS ==========

// Show notification
function showNotification(message, type = 'info', duration = 5000) {
    console.log(`📢 Notification (${type}):`, message);
    
    // Remove existing notifications of same type
    const existingNotifications = document.querySelectorAll(`.healthyfins-notification.notification-${type}`);
    existingNotifications.forEach(n => {
        if (n.parentNode) {
            n.parentNode.removeChild(n);
        }
    });
    
    // Create notification element
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
    
    // Set colors based on type
    if (type === 'success') {
        notification.style.background = 'linear-gradient(135deg, #27ae60, #219653)';
        notification.style.borderLeft = '4px solid #1e874b';
    } else if (type === 'error') {
        notification.style.background = 'linear-gradient(135deg, #e74c3c, #c0392b)';
        notification.style.borderLeft = '4px solid #a93226';
    } else if (type === 'warning') {
        notification.style.background = 'linear-gradient(135deg, #f39c12, #e67e22)';
        notification.style.borderLeft = '4px solid #d35400';
    } else {
        notification.style.background = 'linear-gradient(135deg, #3498db, #2980b9)';
        notification.style.borderLeft = '4px solid #21618c';
    }
    
    // Add icon based on type
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
    
    // Close button handler
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.style.transform = 'translateX(120%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    });
    
    // Auto-remove
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.transform = 'translateX(120%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }
    }, duration);
    
    // Click to dismiss
    notification.addEventListener('click', (e) => {
        if (!e.target.classList.contains('notification-close')) {
            notification.style.transform = 'translateX(120%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }
    });
}

// Show loading overlay
function showLoading(message = 'Loading...') {
    console.log('⏳ Showing loading:', message);
    
    // Remove existing loading
    hideLoading();
    
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
    
    // Add animation if not exists
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

// Hide loading overlay
function hideLoading() {
    const loading = document.getElementById('healthyfins-loading');
    if (loading) {
        console.log('✅ Hiding loading overlay');
        loading.remove();
    }
    document.body.style.overflow = '';
}

// Update connection status on login page
function updateConnectionStatus(isConnected, data) {
    const connectionStatus = document.getElementById('connectionStatus');
    if (!connectionStatus) return;
    
    if (isConnected) {
        connectionStatus.className = 'connection-status connection-connected';
        connectionStatus.innerHTML = `
            <i class="fas fa-wifi"></i>
            <span>Connected to HealthyFins</span>
            <small style="margin-left: 10px; opacity: 0.8;">
                Model: ${data?.model?.loaded ? '✅ Loaded' : '❌ Not loaded'}
            </small>
        `;
        connectionStatus.style.display = 'flex';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (connectionStatus) {
                connectionStatus.style.opacity = '0.5';
            }
        }, 5000);
    } else {
        connectionStatus.className = 'connection-status connection-disconnected';
        connectionStatus.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>Backend server unavailable - Using Demo Mode</span>
        `;
        connectionStatus.style.display = 'flex';
    }
}

// ========== PAGE INITIALIZATION ==========

// Initialize page with authentication check
function initializePage() {
    console.log('🚀 Initializing HealthyFins page...');
    console.log('📁 Current page:', window.location.pathname);
    
    // Test backend connection (non-blocking)
    setTimeout(() => {
        testBackendConnection().then(result => {
            if (result.connected) {
                console.log('✅ Backend connection established');
            } else {
                console.warn('⚠️ Backend connection failed:', result.error);
            }
        });
    }, 1000);
    
    // Check authentication for protected pages
    const isLoginPage = window.location.pathname.includes('index.html') || 
                       window.location.pathname === '/' ||
                       window.location.pathname.endsWith('login') ||
                       window.location.pathname.includes('login.html');
    
    console.log('🔐 Is login page:', isLoginPage);
    console.log('🔐 Is authenticated:', checkAuth());
    
    if (!isLoginPage && !checkAuth()) {
        console.warn('❌ User not authenticated, redirecting to login');
        showNotification('Please login to continue', 'warning');
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1500);
        return;
    }
    
    // If already logged in and on login page, redirect to dashboard
    if (isLoginPage && checkAuth()) {
        console.log('✅ User already logged in, redirecting to dashboard');
        window.location.href = 'dashboard.html';
        return;
    }
    
    // Load user data if authenticated
    if (checkAuth()) {
        const user = loadUserData();
        if (user) {
            console.log('👤 User loaded:', user.email);
        }
    }
    
    // Setup user dropdown if exists
    if (document.querySelector('.user-dropdown')) {
        setupUserDropdown();
    }
    
    // Add logout button listener
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
        console.log('✅ Logout button listener added');
    }
    
    // Add demo credentials helper on login page
    if (isLoginPage) {
        addDemoCredentialsHelper();
    }
    
    console.log('✅ Page initialization complete');
}

// Add demo credentials helper
function addDemoCredentialsHelper() {
    // Only add on localhost for testing
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        const demoBtn = document.createElement('button');
        demoBtn.type = 'button';
        demoBtn.innerHTML = '<i class="fas fa-vial"></i> Use Demo Credentials';
        demoBtn.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 20px;
            padding: 10px 15px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            z-index: 1000;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            display: flex;
            align-items: center;
            gap: 8px;
        `;
        demoBtn.onclick = function() {
            const emailField = document.getElementById('loginEmail');
            const passwordField = document.getElementById('loginPassword');
            if (emailField) emailField.value = 'demo@healthyfins.com';
            if (passwordField) passwordField.value = 'demo123';
            showNotification('Demo credentials loaded. Click Sign In to test.', 'info');
        };
        document.body.appendChild(demoBtn);
        console.log('✅ Demo credentials helper added');
    }
}

// ========== GLOBAL ERROR HANDLER ==========

// Handle uncaught errors
window.addEventListener('error', function(event) {
    console.error('❌ Global error:', event.error);
    showNotification('An unexpected error occurred', 'error');
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
    console.error('❌ Unhandled promise rejection:', event.reason);
    showNotification('Network error occurred', 'error');
});

// ========== INITIALIZE ON PAGE LOAD ==========

// Wait for DOM to be fully loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePage);
} else {
    // DOM already loaded
    initializePage();
}

// ========== EXPORT FUNCTIONS FOR GLOBAL USE ==========

// Create global HealthyFins object
window.HealthyFins = {
    // Auth functions
    checkAuth,
    loginUser,
    registerUser,
    logout,
    
    // UI functions
    showNotification,
    showLoading,
    hideLoading,
    
    // Network functions
    apiRequest,
    testBackendConnection,
    getAuthHeaders,
    
    // Data functions
    loadUserData,
    
    // Configuration
    BACKEND_URL,
    FRONTEND_URL,
    
    // Demo functions (for testing)
    loginDemoUser: function(email, password) {
        return loginDemoUser(email, password);
    },
    
    // Version
    version: '4.0.0'
};

console.log('✅ HealthyFins API v4.0.0 loaded successfully');
console.log('🌐 Backend URL:', BACKEND_URL);
console.log('🏠 Frontend URL:', FRONTEND_URL);

// Add CSS for notifications and connection status
function addGlobalStyles() {
    if (!document.getElementById('healthyfins-global-styles')) {
        const style = document.createElement('style');
        style.id = 'healthyfins-global-styles';
        style.textContent = `
            /* Connection Status */
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
            
            /* Notification close button hover */
            .notification-close:hover {
                opacity: 1 !important;
            }
            
            /* Loading overlay */
            #healthyfins-loading {
                backdrop-filter: blur(5px);
            }
        `;
        document.head.appendChild(style);
        console.log('✅ Global styles added');
    }
}

// Add global styles
addGlobalStyles();

