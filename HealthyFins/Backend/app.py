# app.py - COMPLETE VERSION with Supabase Integration

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import json
import os
import sys
from datetime import datetime, timedelta
import traceback
import warnings
from typing import Optional
import bcrypt
import jwt
from supabase import create_client, Client
import uuid

# ========== CONFIGURATION ==========
SUPABASE_URL = os.getenv("SUPABASE_URL", "your-supabase-url")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-supabase-anon-key")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "your-supabase-service-key")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "healthyfins-supabase-secret-2025-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== INITIALIZE APP ==========
app = FastAPI(
    title="HealthyFins API",
    description="AI Fish Disease Detection System with Supabase Database",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ========== CORS CONFIGURATION ==========
origins = [
    "https://healthy-fins.vercel.app",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ========== AUTH FUNCTIONS ==========
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: Optional[str] = Header(None)):
    """Get current user from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format. Use 'Bearer <token>'")
    
    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")
    
    payload = verify_token(token)
    
    # Verify user exists in Supabase
    try:
        response = supabase.table("users").select("*").eq("id", payload.get("user_id")).execute()
        
        if not response.data:
            raise HTTPException(status_code=401, detail="User not found")
        
        user = response.data[0]
        return {
            "user_id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "hardware_id": user.get("hardware_id", "")
        }
    except Exception as e:
        print(f"❌ Error fetching user: {str(e)}")
        raise HTTPException(status_code=401, detail="User verification failed")

# ========== DATABASE FUNCTIONS ==========
class SupabaseDatabase:
    def __init__(self):
        self.supabase = supabase
    
    async def create_user(self, email: str, password: str, name: str, hardware_id: str = None):
        """Create new user in Supabase"""
        try:
            # Check if user already exists
            existing = self.supabase.table("users").select("*").eq("email", email).execute()
            
            if existing.data:
                return False, "Email already registered"
            
            # Hash password
            password_hash = hash_password(password)
            
            # Insert user
            user_data = {
                "email": email,
                "password_hash": password_hash,
                "name": name,
                "hardware_id": hardware_id
            }
            
            response = self.supabase.table("users").insert(user_data).execute()
            
            if response.data:
                user = response.data[0]
                
                # If hardware_id provided, register device
                if hardware_id:
                    await self.register_hardware_device(user["id"], hardware_id)
                
                return True, {
                    "id": user["id"],
                    "email": user["email"],
                    "name": user["name"],
                    "hardware_id": user["hardware_id"],
                    "created_at": user["created_at"]
                }
            
            return False, "Registration failed"
            
        except Exception as e:
            print(f"❌ Create user error: {str(e)}")
            return False, str(e)
    
    async def authenticate_user(self, email: str, password: str):
        """Authenticate user against Supabase"""
        try:
            response = self.supabase.table("users").select("*").eq("email", email).execute()
            
            if not response.data:
                return False, "User not found"
            
            user = response.data[0]
            
            if not verify_password(password, user["password_hash"]):
                return False, "Invalid password"
            
            return True, {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "hardware_id": user["hardware_id"]
            }
            
        except Exception as e:
            print(f"❌ Authentication error: {str(e)}")
            return False, str(e)
    
    def _validate_device_id(self, device_id: str) -> bool:
        """Validate hardware device ID format"""
        import re
        pattern = r'^HF-[A-Z]{2,4}-\d{3}$'
        return bool(re.match(pattern, device_id))
    
    def _classify_disease_type(self, prediction: str) -> str:
        """Classify disease type from prediction"""
        prediction_lower = prediction.lower()
        
        if 'healthy' in prediction_lower:
            return 'healthy'
        elif 'bacterial' in prediction_lower:
            return 'bacterial'
        elif 'fungal' in prediction_lower:
            return 'fungal'
        elif 'parasitic' in prediction_lower:
            return 'parasitic'
        elif 'viral' in prediction_lower:
            return 'viral'
        else:
            return 'unknown'
    
    def _get_treatment_recommendation(self, prediction: str, confidence: float) -> str:
        """Get treatment recommendation based on prediction"""
        treatments = {
            "healthy": "Continue regular maintenance and monitoring.",
            "bacterial": "1. Antibiotic treatment (Kanaplex/Maracyn)\n2. Salt therapy\n3. Emergency water change\n4. Isolation protocol",
            "fungal": "1. Antifungal medication\n2. Salt treatment\n3. Wound management\n4. Temperature adjustment",
            "parasitic": "1. Anti-parasitic medication\n2. Salt baths\n3. Temperature increase\n4. Tank vacuuming",
            "viral": "1. Supportive care\n2. Immune boosters\n3. Stress reduction\n4. Nutrition focus"
        }
        
        disease_type = self._classify_disease_type(prediction)
        base_treatment = treatments.get(disease_type, "Consult with fish health professional.")
        
        if confidence < 70:
            base_treatment = f"⚠️ LOW CONFIDENCE ({confidence}%)\n\nRECOMMENDED ACTIONS:\n\n1. Retake clear photos\n2. Complete water parameter test\n3. Monitor symptoms closely\n4. Contact helpline\n\n---\n\n{base_treatment}"
        
        return base_treatment
    
    async def add_prediction(self, user_id: str, image_name: str, prediction: str, 
                           confidence: float, model_type: str = "ai_model", image_url: str = None):
        """Add prediction to history"""
        try:
            # Determine disease type
            disease_type = self._classify_disease_type(prediction)
            
            # Get treatment recommendation
            treatment = self._get_treatment_recommendation(prediction, confidence)
            
            prediction_data = {
                "user_id": user_id,
                "image_name": image_name,
                "image_url": image_url,
                "prediction": prediction,
                "confidence": float(confidence),
                "disease_type": disease_type,
                "model_type": model_type,
                "treatment_recommendation": treatment
            }
            
            response = self.supabase.table("predictions").insert(prediction_data).execute()
            
            if response.data:
                return True, response.data[0]["id"]
            return False, "Failed to save prediction"
            
        except Exception as e:
            print(f"❌ Add prediction error: {str(e)}")
            return False, str(e)
    
    async def get_user_predictions(self, user_id: str, limit: int = 50, offset: int = 0):
        """Get user's prediction history"""
        try:
            response = self.supabase.table("predictions") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .offset(offset) \
                .execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            print(f"❌ Get predictions error: {str(e)}")
            return []
    
    async def get_user_stats(self, user_id: str):
        """Get user statistics"""
        try:
            # Get total predictions
            predictions = self.supabase.table("predictions") \
                .select("*") \
                .eq("user_id", user_id) \
                .execute()
            
            total = len(predictions.data) if predictions.data else 0
            
            # Count healthy vs disease
            healthy_count = 0
            disease_count = 0
            
            if predictions.data:
                for pred in predictions.data:
                    if pred["disease_type"] == "healthy":
                        healthy_count += 1
                    else:
                        disease_count += 1
            
            # Get hardware status
            hardware = self.supabase.table("hardware_devices") \
                .select("*") \
                .eq("user_id", user_id) \
                .execute()
            
            hardware_connected = bool(hardware.data)
            
            return {
                "total_predictions": total,
                "healthy_count": healthy_count,
                "disease_count": disease_count,
                "hardware_connected": hardware_connected,
                "last_prediction": predictions.data[0]["created_at"] if predictions.data else None
            }
            
        except Exception as e:
            print(f"❌ Get stats error: {str(e)}")
            return {
                "total_predictions": 0,
                "healthy_count": 0,
                "disease_count": 0,
                "hardware_connected": False,
                "last_prediction": None
            }
    
    async def register_hardware_device(self, user_id: str, device_id: str):
        """Register hardware device"""
        try:
            # Validate device ID format (e.g., HF-FDDS-001)
            if not self._validate_device_id(device_id):
                return False, "Invalid device ID format. Use format: HF-FDDS-XXX"
            
            # Check if device already registered
            existing = self.supabase.table("hardware_devices") \
                .select("*") \
                .eq("device_id", device_id) \
                .execute()
            
            if existing.data:
                return False, "Device ID already registered"
            
            device_data = {
                "user_id": user_id,
                "device_id": device_id,
                "device_type": "PH_MONITOR",
                "status": "active",
                "last_connected": datetime.utcnow().isoformat()
            }
            
            response = self.supabase.table("hardware_devices").insert(device_data).execute()
            
            if not response.data:
                return False, "Failed to register device"
            
            # Update user's hardware_id
            self.supabase.table("users") \
                .update({"hardware_id": device_id}) \
                .eq("id", user_id) \
                .execute()
            
            return True, "Device registered successfully"
            
        except Exception as e:
            print(f"❌ Register hardware error: {str(e)}")
            return False, str(e)
    
    async def update_user_profile(self, user_id: str, name: str = None, hardware_id: str = None):
        """Update user profile"""
        try:
            update_data = {}
            if name:
                update_data["name"] = name
            if hardware_id:
                if not self._validate_device_id(hardware_id):
                    return False, "Invalid hardware ID format"
                update_data["hardware_id"] = hardware_id
            
            if update_data:
                response = self.supabase.table("users") \
                    .update(update_data) \
                    .eq("id", user_id) \
                    .execute()
                
                if response.data:
                    return True, "Profile updated"
                return False, "Update failed"
            
            return False, "No changes to update"
            
        except Exception as e:
            print(f"❌ Update profile error: {str(e)}")
            return False, str(e)

# Initialize database
db = SupabaseDatabase()

# ========== MODEL LOADING ==========
model = None
class_names = []
reverse_label_map = {}

def load_model():
    """Load AI model - Simplified for Supabase demo"""
    global model, class_names, reverse_label_map
    
    print("=" * 60)
    print("🐟 HEALTHYFINS - SIMPLIFIED MODEL LOADING")
    print("=" * 60)
    
    # Use default classes
    class_names = [
        "Bacterial Red disease",
        "Parasitic diseases", 
        "Viral diseases White tail disease",
        "Fungal diseases Saprolegniasis",
        "Bacterial diseases - Aeromoniasis",
        "Bacterial gill disease",
        "Healthy Fish",
        "EUS_Ulcerative_Syndrome"
    ]
    
    reverse_label_map = {str(i): class_names[i] for i in range(len(class_names))}
    
    print(f"✅ Classes loaded: {len(class_names)}")
    print("⚠️ Using intelligent analysis mode (Model not loaded)")
    print("=" * 60)
    
    return True

def preprocess_image(image_bytes):
    """Preprocess image for analysis"""
    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (224, 224))
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Expand dimensions
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def analyze_image_intelligently(image_array):
    """Intelligent image analysis"""
    try:
        # Convert to uint8 for OpenCV
        img_uint8 = (image_array[0] * 255).astype(np.uint8)
        
        # Color analysis
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        # Calculate color percentages
        total_pixels = 224 * 224
        
        # Red areas
        red_mask = ((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)) & (hsv[:,:,1] > 100)
        red_percent = np.sum(red_mask) / total_pixels * 100
        
        # White spots
        white_mask = (hsv[:,:,1] < 30) & (hsv[:,:,2] > 200)
        white_percent = np.sum(white_mask) / total_pixels * 100
        
        # Dark areas (fungus)
        dark_mask = hsv[:,:,2] < 50
        dark_percent = np.sum(dark_mask) / total_pixels * 100
        
        # Generate predictions based on analysis
        predictions = np.zeros(len(class_names))
        
        # Determine most likely disease
        if white_percent > 3:
            # White spot disease
            predictions[2] = 0.7  # Viral diseases
            confidence = min(70 + white_percent, 95)
        elif red_percent > 2:
            # Bacterial disease
            predictions[0] = 0.8  # Bacterial Red
            predictions[6] = 0.2  # Other bacterial
            confidence = min(65 + red_percent, 90)
        elif dark_percent > 20:
            # Fungal infection
            predictions[3] = 0.75  # Fungal
            confidence = min(60 + dark_percent/2, 85)
        elif red_percent > 0.5 or white_percent > 0.5:
            # Minor infection
            predictions[1] = 0.6  # Parasitic
            confidence = 50 + max(red_percent, white_percent) * 2
        else:
            # Healthy
            predictions[6] = 0.9  # Healthy
            confidence = 85
        
        # Normalize predictions
        if np.sum(predictions) > 0:
            predictions = predictions / np.sum(predictions)
        else:
            predictions = np.ones(len(class_names)) / len(class_names)
            confidence = 50
        
        # Get top 3 predictions
        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3 = []
        for idx in top3_idx:
            idx_int = int(idx)
            disease = class_names[idx_int] if idx_int < len(class_names) else "Unknown"
            top3.append({
                "disease": disease,
                "confidence": float(predictions[idx_int]) * 100
            })
        
        # Get best prediction
        best_idx = np.argmax(predictions)
        best_disease = class_names[best_idx] if best_idx < len(class_names) else "Unknown"
        
        return {
            "prediction": best_disease,
            "confidence": confidence,
            "top3": top3,
            "analysis": {
                "red_percent": red_percent,
                "white_percent": white_percent,
                "dark_percent": dark_percent
            }
        }
        
    except Exception as e:
        print(f"❌ Intelligent analysis error: {e}")
        return {
            "prediction": "Unknown",
            "confidence": 50,
            "top3": [{"disease": "Unknown", "confidence": 50}],
            "analysis": {}
        }

# Load model on startup
load_model()

# ========== API ENDPOINTS ==========
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🐟 HealthyFins API with Supabase",
        "status": "active",
        "version": "5.0.0",
        "database": "Supabase",
        "model": {
            "loaded": False,
            "mode": "intelligent_analysis",
            "classes": len(class_names)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Supabase connection
        supabase_test = supabase.table("users").select("count", count="exact").execute()
        
        return {
            "status": "healthy",
            "database": "connected" if supabase_test else "disconnected",
            "service": "HealthyFins Backend v5.0",
            "timestamp": datetime.now().isoformat(),
            "model": {
                "loaded": False,
                "classes": len(class_names),
                "mode": "intelligent_analysis"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/register")
async def register_user(
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    hardware_id: Optional[str] = Form(None)
):
    """Register new user"""
    try:
        print(f"📝 Registration attempt for: {email}")
        
        # Validate hardware ID if provided
        if hardware_id:
            if not db._validate_device_id(hardware_id):
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid hardware ID format. Use format: HF-FDDS-XXX"
                )
        
        success, result = await db.create_user(email, password, name, hardware_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=result)
        
        # Create access token
        access_token = create_access_token(
            data={"sub": email, "user_id": result["id"]}
        )
        
        print(f"✅ User registered: {email} (ID: {result['id']})")
        
        return {
            "success": True,
            "message": "Registration successful",
            "user": result,
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")

@app.post("/login")
async def login_user(
    email: str = Form(...),
    password: str = Form(...)
):
    """Login user"""
    try:
        print(f"🔐 Login attempt for: {email}")
        
        success, result = await db.authenticate_user(email, password)
        
        if not success:
            raise HTTPException(status_code=401, detail=result)
        
        # Create access token
        access_token = create_access_token(
            data={"sub": email, "user_id": result["id"]}
        )
        
        print(f"✅ User logged in: {email}")
        
        return {
            "success": True,
            "message": "Login successful",
            "user": result,
            "access_token": access_token,
            "token_type": "bearer",
            "requires_hardware": bool(result.get("hardware_id"))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Predict fish disease from image"""
    try:
        print(f"🔍 Prediction request from: {current_user['email']}")
        
        # Check if image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        
        # Check file size
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Get prediction using intelligent analysis
        analysis_result = analyze_image_intelligently(processed_image)
        
        # Save to Supabase
        success, prediction_id = await db.add_prediction(
            user_id=current_user["user_id"],
            image_name=file.filename[:50],
            prediction=analysis_result["prediction"],
            confidence=analysis_result["confidence"],
            model_type="intelligent_analysis"
        )
        
        if not success:
            print("⚠️ Failed to save prediction to database")
        
        print(f"✅ Prediction complete: {analysis_result['prediction']} ({analysis_result['confidence']:.1f}%)")
        
        return {
            "success": True,
            "prediction": analysis_result["prediction"],
            "confidence": round(analysis_result["confidence"], 2),
            "top3": analysis_result["top3"],
            "model_type": "intelligent_analysis",
            "model_available": False,
            "prediction_id": prediction_id if success else None,
            "user": current_user,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/history")
async def get_user_history(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get user's prediction history"""
    try:
        predictions = await db.get_user_predictions(current_user["user_id"], limit, offset)
        
        return {
            "success": True,
            "count": len(predictions),
            "history": predictions,
            "user": current_user
        }
        
    except Exception as e:
        print(f"❌ History error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")

@app.get("/stats")
async def get_user_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get user statistics"""
    try:
        stats = await db.get_user_stats(current_user["user_id"])
        
        return {
            "success": True,
            "stats": stats,
            "user": current_user
        }
        
    except Exception as e:
        print(f"❌ Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@app.post("/hardware/register")
async def register_hardware(
    device_id: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Register hardware device"""
    try:
        success, message = await db.register_hardware_device(current_user["user_id"], device_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        # Update current_user data
        current_user["hardware_id"] = device_id
        
        return {
            "success": True,
            "message": message,
            "device_id": device_id,
            "user": current_user
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Hardware registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hardware registration error: {str(e)}")

@app.get("/hardware/status")
async def get_hardware_status(
    current_user: dict = Depends(get_current_user)
):
    """Get hardware device status"""
    try:
        response = supabase.table("hardware_devices") \
            .select("*") \
            .eq("user_id", current_user["user_id"]) \
            .execute()
        
        devices = response.data if response.data else []
        
        return {
            "success": True,
            "devices": devices,
            "count": len(devices),
            "user": current_user
        }
        
    except Exception as e:
        print(f"❌ Hardware status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hardware status error: {str(e)}")

@app.put("/profile")
async def update_profile(
    name: Optional[str] = Form(None),
    hardware_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Update user profile"""
    try:
        success, message = await db.update_user_profile(
            current_user["user_id"], name, hardware_id
        )
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        # Update current_user data
        if name:
            current_user["name"] = name
        if hardware_id:
            current_user["hardware_id"] = hardware_id
        
        return {
            "success": True,
            "message": message,
            "user": current_user
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Profile update error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Profile update error: {str(e)}")

@app.get("/profile")
async def get_profile(
    current_user: dict = Depends(get_current_user)
):
    """Get user profile"""
    try:
        response = supabase.table("users") \
            .select("*") \
            .eq("id", current_user["user_id"]) \
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = response.data[0]
        
        return {
            "success": True,
            "profile": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "hardware_id": user["hardware_id"],
                "created_at": user["created_at"],
                "updated_at": user["updated_at"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Get profile error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Get profile error: {str(e)}")

# ========== STARTUP MESSAGE ==========
print("\n" + "=" * 60)
print("🐟 HEALTHYFINS API v5.0 - SUPABASE EDITION")
print("=" * 60)
print(f"📊 Database: Supabase")
print(f"🔗 Supabase URL: {SUPABASE_URL}")
print(f"🎯 Model Mode: Intelligent Analysis")
print(f"📈 Total Classes: {len(class_names)}")
print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
