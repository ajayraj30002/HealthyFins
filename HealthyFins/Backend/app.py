from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import json
import os
import sys
from datetime import datetime
from typing import Optional
import traceback
import warnings

# Import our modules
sys.path.append('.')
from database import db
from auth import create_access_token, get_current_user

# Initialize app
app = FastAPI(
    title="HealthyFins API",
    description="AI Fish Disease Detection System",
    version="4.0.0"
)

# ========== CORS CONFIGURATION ==========
origins = [
    "https://healthy-fins.vercel.app",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
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

# ========== MODEL LOADING ==========
model = None
class_names = []
reverse_label_map = {}

@app.on_event("startup")
async def load_model():
    """Load AI model on startup - EXACT as Colab"""
    global model, class_names, reverse_label_map
    
    print("=" * 60)
    print("🐟 HEALTHYFINS - LOADING TRAINED MODEL FROM COLAB")
    print("=" * 60)
    
    model_path = 'models/fish_disease_model_final.h5'
    info_path = 'models/model_info_final.json'
    
    # Check if files exist
    print(f"🔍 Checking model file: {model_path}")
    print(f"✅ File exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        # Try alternative paths
        possible_paths = [
            'fish_disease_model_final.h5',
            './fish_disease_model_final.h5'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Found model at: {path}")
                break
    
    if not os.path.exists(model_path):
        print("❌ Model file not found anywhere!")
        model = None
        return
    
    # Load class info from JSON
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                data = json.load(f)
                class_names = data.get('class_names', [])
                reverse_label_map = data.get('reverse_label_map', {})
                print(f"📊 Classes from JSON: {len(class_names)}")
                print(f"📊 Reverse map: {reverse_label_map}")
        except Exception as e:
            print(f"⚠️ Error loading class info: {e}")
            class_names = []
            reverse_label_map = {}
    else:
        print(f"⚠️ Info file not found, trying to load from model")
        # Try to load from the model itself
        try:
            # This is the EXACT class order from your Colab training
            class_names = [
                "Bacterial Red disease",
                "Parasitic diseases", 
                "Viral diseases White tail disease",
                "Fungal diseases Saprolegniasis",
                "Bacterial diseases - Aeromoniasis",
                "Bacterial gill disease",
                "Healthy Fish",
                "EUS_Ulcerative_Syndrome (arg)"
            ]
            reverse_label_map = {str(i): class_names[i] for i in range(len(class_names))}
            print(f"📊 Using default classes from Colab training")
        except:
            class_names = []
            reverse_label_map = {}
    
    if not class_names:
        # Fallback classes
        class_names = [
            "Bacterial Red disease",
            "Parasitic diseases", 
            "Viral diseases White tail disease",
            "Fungal diseases Saprolegniasis",
            "Bacterial diseases - Aeromoniasis",
            "Bacterial gill disease",
            "Healthy Fish",
            "EUS_Ulcerative_Syndrome (arg)"
        ]
        reverse_label_map = {str(i): class_names[i] for i in range(len(class_names))}
    
    print(f"📊 Total classes: {len(class_names)}")
    print(f"📊 TensorFlow version: {tf.__version__}")
    print(f"📊 Keras version: {tf.keras.__version__}")
    
    try:
        print("\n🔄 Loading model with EXACT Colab preprocessing...")
        
        # IMPORTANT: Disable experimental features for compatibility
        tf.compat.v1.disable_eager_execution()
        
        # Load the EXACT model from your Colab training
        # Use custom_objects to handle any custom layers
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                'MobileNetV2': tf.keras.applications.MobileNetV2
            }
        )
        
        print("✅ Model loaded successfully!")
        
        # Verify the model structure
        print(f"\n📊 Model Structure:")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Layers: {len(model.layers)}")
        
        # Test the model with dummy data
        print("\n🧪 Testing model with dummy input...")
        dummy_input = np.random.randn(1, 224, 224, 3).astype('float32')
        
        try:
            predictions = model.predict(dummy_input, verbose=0)
            print(f"✅ Model test passed!")
            print(f"  Output shape: {predictions.shape}")
            print(f"  Predictions sum: {np.sum(predictions):.4f} (should be ~1.0)")
            
            # Check number of classes
            if predictions.shape[1] != len(class_names):
                print(f"⚠️ Warning: Model expects {predictions.shape[1]} classes but we have {len(class_names)}")
                print(f"⚠️ Adjusting class list...")
                # If model has different number of classes, use what the model expects
                class_names = [f"Class_{i}" for i in range(predictions.shape[1])]
                reverse_label_map = {str(i): class_names[i] for i in range(len(class_names))}
                
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            # Try loading weights only
            print("🔄 Trying alternative loading method...")
            model = None
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        traceback.print_exc()
        model = None
    
    # ============================================
    # FINAL STATUS
    # ============================================
    if model is not None:
        print(f"\n🎯 MODEL LOADED SUCCESSFULLY!")
        print(f"   Classes: {len(class_names)}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Ready for predictions!")
    else:
        print(f"\n⚠️ MODEL STATUS: Using Intelligent Analysis Mode")
        print(f"   Classes: {len(class_names)}")
    
    print("=" * 60)

# ========== HEALTH CHECK ==========
@app.get("/")
async def root():
    model_status = {
        "loaded": model is not None,
        "type": "real_trained" if model is not None else "analysis_mode",
        "layers": len(model.layers) if model is not None else 0,
        "classes": len(class_names),
        "architecture": "MobileNetV2-based"
    }
    
    return {
        "message": "🐟 HealthyFins API",
        "status": "active",
        "version": "4.0.0",
        "frontend": "https://healthy-fins.vercel.app",
        "model": model_status,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "public": ["/", "/health", "/register", "/login"],
            "protected": ["/predict", "/profile", "/history", "/ph-monitoring"]
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    users_count = len(db.data.get("users", {}))
    history_count = sum(len(v) for v in db.data.get("history", {}).values())
    
    model_info = {
        "loaded": model is not None,
        "type": "real_trained" if model is not None else "analysis_mode",
        "layers": len(model.layers) if model is not None else 0,
        "classes_count": len(class_names),
        "class_names": class_names if len(class_names) <= 10 else class_names[:5] + ["..."],
        "input_shape": str(model.input_shape) if model is not None else "N/A",
        "output_shape": str(model.output_shape) if model is not None else "N/A"
    }
    
    return {
        "status": "healthy",
        "service": "HealthyFins Backend v4.0",
        "timestamp": datetime.now().isoformat(),
        "model": model_info,
        "database": {
            "users": users_count,
            "history_entries": history_count,
            "file": "database.json"
        },
        "system": {
            "python_version": sys.version.split()[0],
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "environment": os.environ.get("RENDER", "development")
        }
    }

# ========== AUTH ENDPOINTS ==========
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
        
        success, result = db.create_user(email, password, name, hardware_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=result)
        
        # Create access token
        access_token = create_access_token(data={"sub": email, "user_id": result["user_id"]})
        
        print(f"✅ User registered: {email} (ID: {result['user_id']})")
        
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
        
        success, result = db.authenticate_user(email, password)
        
        if not success:
            raise HTTPException(status_code=401, detail=result)
        
        # Create access token
        access_token = create_access_token(data={"sub": email, "user_id": result["user_id"]})
        
        print(f"✅ User logged in: {email}")
        
        return {
            "success": True,
            "message": "Login successful",
            "user": result,
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

# ========== IMAGE PREPROCESSING ==========
def preprocess_image_exact_colab(image_bytes):
    """EXACT SAME preprocessing as your Colab training"""
    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # EXACT SAME as Colab:
        # 1. Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Resize to 224x224 (from your Colab code)
        img = cv2.resize(img, (224, 224))
        
        # 3. Normalize to [0, 1] - EXACT as Colab
        img = img.astype('float32') / 255.0
        
        # 4. Expand dimensions for batch
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# ========== ENHANCED ANALYSIS ==========
def analyze_image_features(image_array):
    """Enhanced image analysis when model isn't available"""
    # Simple feature extraction
    try:
        # Convert to uint8 for OpenCV
        img_uint8 = (image_array[0] * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        # Check for common disease indicators
        features = {
            'white_spots': np.mean(hsv[:,:,1] > 150),  # High saturation white spots
            'red_patches': np.mean((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)),  # Red areas
            'fuzzy_areas': np.std(hsv[:,:,2]),  # Texture variation (fungus)
            'overall_health': np.mean(hsv[:,:,1])  # Lower saturation = healthier
        }
        
        # Generate probabilities based on features
        predictions = np.zeros(len(class_names))
        
        # Map features to classes
        for i, disease in enumerate(class_names):
            disease_lower = disease.lower()
            
            if 'healthy' in disease_lower or 'healthy fish' in disease_lower:
                predictions[i] = 0.7 - features['white_spots'] * 0.3 - features['red_patches'] * 0.2
            
            elif 'white' in disease_lower:
                predictions[i] = features['white_spots'] * 0.8
            
            elif 'red' in disease_lower or 'bacterial' in disease_lower:
                predictions[i] = features['red_patches'] * 0.7
            
            elif 'fungal' in disease_lower or 'saprolegniasis' in disease_lower:
                predictions[i] = features['fuzzy_areas'] * 0.6
            
            elif 'parasit' in disease_lower:
                predictions[i] = (features['white_spots'] + features['red_patches']) * 0.4
            
            else:
                predictions[i] = 0.1
        
        # Normalize to sum to 1
        predictions = np.clip(predictions, 0, 1)
        if np.sum(predictions) > 0:
            predictions = predictions / np.sum(predictions)
        else:
            predictions = np.ones(len(class_names)) / len(class_names)
        
        return predictions
    except Exception as e:
        print(f"❌ Feature analysis error: {e}")
        # Return uniform distribution if analysis fails
        return np.ones(len(class_names)) / len(class_names)

# ========== PROTECTED ENDPOINTS ==========
@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Predict fish disease from image - EXACT as Colab"""
    try:
        print(f"🔍 Prediction request from: {current_user['sub']}")
        
        # Check if image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        
        # Check file size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Preprocess EXACT same as Colab
        processed_image = preprocess_image_exact_colab(image_bytes)
        
        print(f"📊 Preprocessed image shape: {processed_image.shape}")
        print(f"📊 Preprocessed image range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        
        # Get predictions
        if model is not None:
            print("🧠 Using REAL trained model from Colab")
            try:
                predictions = model.predict(processed_image, verbose=0)[0]
                model_type = "real_trained"
                print(f"📊 Raw predictions: {predictions}")
                print(f"📊 Predictions sum: {np.sum(predictions):.4f}")
            except Exception as e:
                print(f"⚠️ Model prediction failed: {e}, using enhanced analysis")
                predictions = analyze_image_features(processed_image)
                model_type = "enhanced_analysis"
        else:
            print("🔬 Using ENHANCED image analysis")
            predictions = analyze_image_features(processed_image)
            model_type = "enhanced_analysis"
        
        # Ensure predictions are valid
        if np.sum(predictions) < 0.9 or np.sum(predictions) > 1.1:
            print(f"⚠️ Normalizing predictions (sum was {np.sum(predictions):.4f})")
            predictions = np.clip(predictions, 0, 1)
            predictions = predictions / np.sum(predictions)
        
        # Get results
        best_class_idx = np.argmax(predictions)
        confidence = float(predictions[best_class_idx]) * 100
        
        # Get disease name from reverse map
        if reverse_label_map and str(best_class_idx) in reverse_label_map:
            disease_name = reverse_label_map[str(best_class_idx)]
        elif best_class_idx < len(class_names):
            disease_name = class_names[best_class_idx]
        else:
            disease_name = f"Class_{best_class_idx}"
        
        # Get top 3 predictions
        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3 = []
        for idx in top3_idx:
            idx_int = int(idx)
            if reverse_label_map and str(idx_int) in reverse_label_map:
                disease = reverse_label_map[str(idx_int)]
            elif idx_int < len(class_names):
                disease = class_names[idx_int]
            else:
                disease = f"Class_{idx_int}"
                
            top3.append({
                "disease": disease,
                "confidence": float(predictions[idx_int]) * 100
            })
        
        # Save to history
        image_name = file.filename[:50]
        db.add_prediction_history(
            user_id=current_user["user_id"],
            image_name=image_name,
            prediction=disease_name,
            confidence=confidence
        )
        
        print(f"✅ Prediction complete: {disease_name} ({confidence:.1f}%)")
        print(f"✅ Model type: {model_type}")
        
        # Return result
        return {
            "success": True,
            "prediction": disease_name,
            "confidence": round(confidence, 2),
            "top3": top3,
            "model_type": model_type,
            "model_available": model is not None,
            "user": {
                "id": current_user["user_id"],
                "email": current_user["sub"]
            },
            "timestamp": datetime.now().isoformat(),
            "debug_info": {
                "classes_loaded": len(class_names),
                "reverse_map_keys": len(reverse_label_map) if reverse_label_map else 0,
                "best_class_index": int(best_class_idx)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile"""
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile error: {str(e)}")

@app.put("/profile")
async def update_user_profile(
    name: Optional[str] = Form(None),
    hardware_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Update user profile"""
    try:
        email = current_user["sub"]
        success, message = db.update_user_profile(email, name, hardware_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return {
            "success": True,
            "message": message
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")

@app.get("/history")
async def get_user_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get user's prediction history"""
    try:
        user_id = current_user["user_id"]
        history = db.get_user_history(user_id, limit)
        
        return {
            "success": True,
            "count": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")

@app.delete("/history/{history_id}")
async def delete_history_item(
    history_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete specific history item"""
    try:
        user_id = current_user["user_id"]
        
        if user_id not in db.data["history"]:
            raise HTTPException(status_code=404, detail="No history found")
        
        # Find and remove item
        history = db.data["history"][user_id]
        initial_count = len(history)
        
        db.data["history"][user_id] = [item for item in history if item["id"] != history_id]
        
        if len(db.data["history"][user_id]) == initial_count:
            raise HTTPException(status_code=404, detail="History item not found")
        
        return {
            "success": True,
            "message": "History item deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")

# ========== PH MONITORING ==========
@app.get("/ph-monitoring")
async def get_ph_data(current_user: dict = Depends(get_current_user)):
    """Get PH monitoring data"""
    try:
        # Get user's hardware ID
        user_email = current_user["sub"]
        user_data = db.data["users"].get(user_email, {})
        hardware_id = user_data.get("hardware_id", "Not set")
        
        # Mock data (replace with actual hardware integration)
        import random
        mock_data = {
            "ph": round(random.uniform(6.5, 8.5), 2),
            "temperature": round(random.uniform(24, 30), 1),
            "turbidity": random.randint(5, 50),
            "status": "normal",
            "hardware_id": hardware_id,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "data": mock_data,
            "message": "Connect your Arduino/Raspberry Pi for real-time data"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PH monitoring error: {str(e)}")

# ========== DEBUG ENDPOINT ==========
@app.get("/debug/model")
async def debug_model():
    """Debug endpoint to check model status"""
    model_info = {
        "loaded": model is not None,
        "classes": class_names,
        "reverse_map": reverse_label_map,
        "num_classes": len(class_names),
        "model_paths_checked": [
            'models/fish_disease_model_final.h5',
            'fish_disease_model_final.h5',
            './fish_disease_model_final.h5'
        ],
        "files_exist": {
            'models/fish_disease_model_final.h5': os.path.exists('models/fish_disease_model_final.h5'),
            'fish_disease_model_final.h5': os.path.exists('fish_disease_model_final.h5'),
            'model_info_final.json': os.path.exists('model_info_final.json')
        }
    }
    
    return {
        "success": True,
        "debug": model_info,
        "timestamp": datetime.now().isoformat()
    }

# ========== ERROR HANDLERS ==========
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"❌ Unhandled exception: {str(exc)}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "Internal server error",
            "error": str(exc)
        }
    )

# ========== STARTUP MESSAGE ==========
print("\n" + "=" * 60)
print("🐟 HEALTHYFINS API v4.0 - DEBUGGED VERSION")
print("=" * 60)
print(f"📡 Backend URL: https://healthyfins.onrender.com")
print(f"🌐 Frontend URL: https://healthy-fins.vercel.app")
print(f"🤖 Model: Exact Colab Model Loading")
print("=" * 60)
print("\n✅ Backend initialized successfully!")

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
