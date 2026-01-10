from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from datetime import datetime
from typing import Optional

# Import our modules
from database import db
from auth import create_access_token, get_current_user

# Initialize app
app = FastAPI(
    title="Fish Disease Detection API",
    description="Complete system with authentication and history",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI model
model = None
class_names = []

@app.on_event("startup")
def load_model():
    """Load AI model on startup"""
    global model, class_names
    try:
        print("🔄 Loading AI model...")
        model = tf.keras.models.load_model('models/fish_disease_model_final.h5')
        
        with open('models/model_info_final.json', 'r') as f:
            data = json.load(f)
            class_names = data['class_names']
            
        print(f"✅ Model loaded successfully!")
        print(f"📊 Classes: {class_names}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Create dummy model for testing if real model fails
        model = None
        class_names = ["healthy", "white spot", "fin rot", "fungal", "parasite"]

# ========== PUBLIC ENDPOINTS ==========

@app.get("/")
def home():
    return {
        "message": "🐟 Fish Disease Detection API",
        "status": "active",
        "version": "2.0.0",
        "endpoints": {
            "public": ["/health", "/register", "/login"],
            "protected": ["/predict", "/profile", "/history"]
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "users_count": len(db.data["users"])
    }

@app.post("/register")
async def register(
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    hardware_id: Optional[str] = Form(None)
):
    """Register new user"""
    success, result = db.create_user(email, password, name, hardware_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=result)
    
    # Create access token
    access_token = create_access_token(data={"sub": email, "user_id": result["user_id"]})
    
    return {
        "success": True,
        "message": "Registration successful",
        "user": result,
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...)
):
    """Login user"""
    success, result = db.authenticate_user(email, password)
    
    if not success:
        raise HTTPException(status_code=401, detail=result)
    
    # Create access token
    access_token = create_access_token(data={"sub": email, "user_id": result["user_id"]})
    
    return {
        "success": True,
        "message": "Login successful",
        "user": result,
        "access_token": access_token,
        "token_type": "bearer"
    }

# ========== PROTECTED ENDPOINTS ==========

def preprocess_image(image_bytes):
    """Preprocess image for AI model"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Same preprocessing as training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    
    return np.expand_dims(img, axis=0)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Protected endpoint for disease prediction"""
    try:
        # Check if image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        
        # Check file size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Preprocess
        processed_image = preprocess_image(image_bytes)
        
        # Predict
        if model is None:
            # Mock prediction for testing
            predictions = np.random.rand(len(class_names))
            predictions = predictions / predictions.sum()
        else:
            predictions = model.predict(processed_image, verbose=0)[0]
        
        # Get results
        best_class_idx = np.argmax(predictions)
        confidence = float(predictions[best_class_idx]) * 100
        disease_name = class_names[best_class_idx]
        
        # Get top 3 predictions
        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3 = [
            {
                "disease": class_names[int(idx)],
                "confidence": float(predictions[int(idx)]) * 100
            }
            for idx in top3_idx
        ]
        
        # Save to history
        image_name = file.filename[:50]  # Truncate if too long
        db.add_prediction_history(
            user_id=current_user["user_id"],
            image_name=image_name,
            prediction=disease_name,
            confidence=confidence
        )
        
        # Return result
        return {
            "success": True,
            "prediction": disease_name,
            "confidence": round(confidence, 2),
            "top3": top3,
            "user": {
                "id": current_user["user_id"],
                "email": current_user["sub"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile"""
    email = current_user["sub"]
    user = db.data["users"].get(email)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Don't return password
    user_data = user.copy()
    user_data.pop("password", None)
    
    return {
        "success": True,
        "profile": user_data
    }

@app.put("/profile")
async def update_profile(
    name: Optional[str] = Form(None),
    hardware_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Update user profile"""
    email = current_user["sub"]
    success, message = db.update_user_profile(email, name, hardware_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return {
        "success": True,
        "message": message
    }

@app.get("/history")
async def get_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get user's prediction history"""
    user_id = current_user["user_id"]
    history = db.get_user_history(user_id, limit)
    
    return {
        "success": True,
        "count": len(history),
        "history": history
    }

@app.delete("/history/{history_id}")
async def delete_history_item(
    history_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete specific history item"""
    user_id = current_user["user_id"]
    
    if user_id not in db.data["history"]:
        raise HTTPException(status_code=404, detail="No history found")
    
    # Find and remove item
    history = db.data["history"][user_id]
    initial_count = len(history)
    
    db.data["history"][user_id] = [item for item in history if item["id"] != history_id]
    
    if len(db.data["history"][user_id]) == initial_count:
        raise HTTPException(status_code=404, detail="History item not found")
    
    db._save_data()
    
    return {
        "success": True,
        "message": "History item deleted"
    }

# PH Monitoring endpoint (placeholder for Blynk integration)
@app.get("/ph-monitoring")
async def get_ph_data(current_user: dict = Depends(get_current_user)):
    """Get pH monitoring data (placeholder for hardware integration)"""
    # This would connect to your Blynk/Arduino system
    # For now, return mock data
    return {
        "success": True,
        "data": {
            "ph": 7.2,
            "temperature": 26.5,
            "turbidity": 15,
            "timestamp": datetime.now().isoformat(),
            "status": "normal",
            "hardware_id": db.data["users"].get(current_user["sub"], {}).get("hardware_id", "Not set")
        },
        "message": "Connect hardware to get real-time data"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)