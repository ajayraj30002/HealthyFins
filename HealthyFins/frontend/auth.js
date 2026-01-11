// auth.js - Authentication helper functions

// Backend URL - UPDATE THIS AFTER DEPLOYMENT
const BACKEND_URL = "https://healthyfins.onrender.com"; // Change to your Render URL

// Check if user is authenticated
function checkAuth() {
    const token = localStorage.getItem('token');
    if (!token) {
        return false;
    }
    
    // Check if token is expired (basic check)
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        if (payload.exp * 1000 < Date.now()) {
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
    const token = localStorage.getItem('token');
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
            if (el) el.textContent = user.name;
        });
        
        const userEmailElements = document.querySelectorAll('#userEmail');
        userEmailElements.forEach(el => {
            if (el) el.textContent = user.email;
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
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = 'index.html';
    }
}

// Check authentication on page load
document.addEventListener('DOMContentLoaded', function() {
    // Exclude login page from auth check
    if (window.location.pathname.includes('index.html')) {
        // If user is already logged in, redirect to dashboard
        if (checkAuth()) {
            window.location.href = 'dashboard.html';
        }
        return;
    }
    
    // For all other pages, check authentication
    if (!checkAuth()) {
        window.location.href = 'index.html';
    }
});

// Test backend connection
async function testBackendConnection() {
    try {
        const response = await fetch(`${BACKEND_URL}/health`);
        return response.ok;
    } catch {
        return false;
    }
}

// Show error message
function showError(message) {
    alert('Error: ' + message);
}

// Show success message
function showSuccess(message) {
    alert('Success: ' + message);
}
