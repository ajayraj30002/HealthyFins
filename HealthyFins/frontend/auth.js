// auth.js - HealthyFins Production Version

// Backend URL - Your Render backend URL
const BACKEND_URL = "https://healthyfins.onrender.com"; // Update with your actual Render URL

// Frontend URL - Your Vercel frontend
const FRONTEND_URL = "https://healthy-fins.vercel.app";

// ========== AUTH FUNCTIONS ==========

// Check if user is authenticated
function checkAuth() {
    const token = localStorage.getItem('healthyfins_token');
    if (!token) {
        return false;
    }
    
    // Check if token is expired
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        if (payload.exp * 1000 < Date.now()) {
            localStorage.removeItem('healthyfins_token');
            localStorage.removeItem('healthyfins_user');
            return false;
        }
        return true;
    } catch {
        return false;
    }
}

// Get authentication headers
function getAuthHeaders() {
    const token = localStorage.getItem('healthyfins_token');
    if (!token) {
        console.error('No token found in localStorage');
        return {
            'Content-Type': 'application/json'
        };
    }
    
    return {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    };
}

// Load user data from localStorage
function loadUserData() {
    try {
        const userStr = localStorage.getItem('healthyfins_user');
        if (!userStr) return null;
        
        const user = JSON.parse(userStr);
        
        // Update all user elements on page
        const userNameElements = document.querySelectorAll('#userName, #userGreeting');
        userNameElements.forEach(el => {
            if (el) el.textContent = user.name || user.email;
        });
        
        const userEmailElements = document.querySelectorAll('#userEmail');
        userEmailElements.forEach(el => {
            if (el) el.textContent = user.email;
        });
        
        return user;
    } catch (error) {
        console.error('Error loading user data:', error);
        return null;
    }
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
    if (confirm('Are you sure you want to logout from HealthyFins?')) {
        localStorage.removeItem('healthyfins_token');
        localStorage.removeItem('healthyfins_user');
        window.location.href = 'index.html';
    }
}

// ========== NETWORK FUNCTIONS ==========

// Test backend connection
async function testBackendConnection() {
    try {
        console.log("🔗 Testing connection to HealthyFins backend...");
        const response = await fetch(`${BACKEND_URL}/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log("✅ HealthyFins backend connected:", {
                status: data.status,
                model: data.model?.loaded ? 'Loaded' : 'Not loaded',
                url: BACKEND_URL
            });
            return true;
        } else {
            console.error("❌ Backend error:", response.status, response.statusText);
            return false;
        }
    } catch (error) {
        console.error("❌ Cannot connect to HealthyFins backend:", error.message);
        
        // Show user-friendly message on login page
        if (window.location.pathname.includes('index.html') || 
            window.location.pathname === '/') {
            showNotification(
                '⚠️ Cannot connect to HealthyFins server. The system will use demo mode.',
                'warning'
            );
        }
        return false;
    }
}

// Enhanced fetch with timeout and error handling
async function apiRequest(endpoint, options = {}) {
    const controller = new AbortController();
    const timeout = 30000; // 30 seconds timeout
    
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
        // Add authorization header if token exists
        const token = localStorage.getItem('healthyfins_token');
        const headers = {
            'Accept': 'application/json',
            ...options.headers
        };
        
        if (token && !endpoint.includes('/login') && !endpoint.includes('/register')) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        const response = await fetch(`${BACKEND_URL}${endpoint}`, {
            ...options,
            headers,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        // Handle non-JSON responses
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error(`Server returned non-JSON response: ${contentType}`);
        }
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || data.message || `HTTP ${response.status}`);
        }
        
        return data;
        
    } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
            throw new Error('Request timeout. Please check your internet connection.');
        }
        
        console.error(`API request failed for ${endpoint}:`, error);
        throw error;
    }
}

// Login function
async function loginUser(email, password) {
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
            // Store auth data with app-specific keys
            localStorage.setItem('healthyfins_token', data.access_token);
            localStorage.setItem('healthyfins_user', JSON.stringify(data.user));
            console.log('✅ User logged in:', data.user.email);
        }
        
        return data;
    } catch (error) {
        console.error('Login error:', error);
        throw error;
    }
}

// Register function
async function registerUser(name, email, password, hardwareId = null) {
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
            // Store auth data with app-specific keys
            localStorage.setItem('healthyfins_token', data.access_token);
            localStorage.setItem('healthyfins_user', JSON.stringify(data.user));
            console.log('✅ User registered:', data.user.email);
        }
        
        return data;
    } catch (error) {
        console.error('Registration error:', error);
        throw error;
    }
}

// ========== UI NOTIFICATION FUNCTIONS ==========

// Show notification
function showNotification(message, type = 'info', duration = 5000) {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.healthyfins-notification');
    existingNotifications.forEach(n => n.remove());
    
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
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.2em;">${icon}</span>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 10);
    
    // Auto-remove
    setTimeout(() => {
        notification.style.transform = 'translateX(120%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, duration);
    
    // Click to dismiss
    notification.addEventListener('click', () => {
        notification.style.transform = 'translateX(120%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    });
}

// Show loading overlay
function showLoading(message = 'Loading...') {
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
        loading.remove();
    }
    document.body.style.overflow = '';
}

// ========== PAGE INITIALIZATION ==========

// Initialize page with authentication check
function initializePage() {
    // Test backend connection
    testBackendConnection().then(isConnected => {
        if (isConnected) {
            console.log('✅ HealthyFins backend is ready');
        } else {
            console.warn('⚠️ HealthyFins backend is not accessible');
        }
    });
    
    // Check authentication for protected pages
    const isLoginPage = window.location.pathname.includes('index.html') || 
                       window.location.pathname === '/' ||
                       window.location.pathname.includes('login');
    
    if (!isLoginPage && !checkAuth()) {
        console.warn('User not authenticated, redirecting to login');
        showNotification('Please login to continue', 'warning');
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1500);
        return;
    }
    
    // If already logged in and on login page, redirect to dashboard
    if (isLoginPage && checkAuth()) {
        console.log('User already logged in, redirecting to dashboard');
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
    }
}

// ========== GLOBAL ERROR HANDLER ==========

// Handle uncaught errors
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showNotification('An unexpected error occurred', 'error');
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showNotification('Network error occurred', 'error');
});

// ========== INITIALIZE ON PAGE LOAD ==========

// Wait for DOM to be fully loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePage);
} else {
    initializePage();
}

// Export functions for use in other scripts
window.HealthyFins = {
    checkAuth,
    loginUser,
    registerUser,
    logout,
    showNotification,
    showLoading,
    hideLoading,
    apiRequest,
    BACKEND_URL,
    FRONTEND_URL
};
