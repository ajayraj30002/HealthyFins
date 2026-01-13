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
    version="3.1.0"
)

# ========== CORS CONFIGURATION ==========
origins = [
    "https://healthy-fins.vercel.app",  # Your Vercel frontend
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "*"  # Keep for testing
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

@app.on_event("startup")
async def load_model():
    """Load AI model on startup - FIXED VERSION"""
    global model, class_names
    
    print("=" * 60)
    print("🐟 HEALTHYFINS - FORCING MODEL LOAD")
    print("=" * 60)
    
    # Your actual classes
    ACTUAL_CLASSES = [
        "Bacterial Red disease",
        "Parasitic diseases", 
        "Viral diseases White tail disease",
        "Fungal diseases Saprolegniasis",
        "Bacterial diseases - Aeromoniasis",
        "Bacterial gill disease",
        "Healthy Fish",
        "EUS_Ulcerative_Syndrome (arg)"
    ]
    
    model_path = 'models/fish_disease_model_final.h5'
    info_path = 'models/model_info_final.json'
    
    # Check if files exist
    print(f"📂 Checking for model files...")
    print(f"🔍 Model path: {model_path}")
    print(f"🔍 Info path: {info_path}")
    
    # List current directory
    print(f"\n📂 Current directory: {os.getcwd()}")
    print("📂 Files in current directory:")
    for item in os.listdir('.'):
        item_path = os.path.join('.', item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
            # List files in subdirectories
            try:
                for subitem in os.listdir(item_path):
                    print(f"    - {subitem}")
            except:
                pass
        else:
            size = os.path.getsize(item_path) / 1024
            print(f"  📄 {item} ({size:.1f} KB)")
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model file not found at: {model_path}")
        # Check if it exists elsewhere
        print("🔍 Searching for model file in other locations...")
        for root, dirs, files in os.walk('.'):
            if 'fish_disease_model_final.h5' in files:
                alt_path = os.path.join(root, 'fish_disease_model_final.h5')
                print(f"✅ Found model at alternative location: {alt_path}")
                model_path = alt_path
                break
        
        if not os.path.exists(model_path):
            print("❌ Could not find model file anywhere!")
            model = None
            class_names = ACTUAL_CLASSES
            return
    else:
        print(f"✅ Model file found!")
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📊 Model size: {file_size:.2f} MB")
    
    print(f"\n📊 TensorFlow version: {tf.__version__}")
    
    try:
        # METHOD 1: Try standard load with warnings suppressed
        print("\n🔄 Attempting to load model (Method 1: Standard)...")
        
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings('ignore')
        
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False  # Disable safe mode
        )
        print("✅ Model loaded successfully with Method 1!")
        
    except Exception as e:
        print(f"⚠️ Method 1 failed: {str(e)[:100]}...")
        
        try:
            # METHOD 2: Load with custom_objects workaround
            print("\n🔄 Trying Method 2: Custom objects workaround...")
            
            # Define all possible layer types
            custom_objects = {
                # Input layers
                'InputLayer': tf.keras.layers.InputLayer,
                'Input': tf.keras.layers.Input,
                
                # Convolutional layers
                'Conv2D': tf.keras.layers.Conv2D,
                'Conv1D': tf.keras.layers.Conv1D,
                'Conv3D': tf.keras.layers.Conv3D,
                'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
                'SeparableConv2D': tf.keras.layers.SeparableConv2D,
                
                # Pooling layers
                'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                'AveragePooling2D': tf.keras.layers.AveragePooling2D,
                'MaxPooling1D': tf.keras.layers.MaxPooling1D,
                'AveragePooling1D': tf.keras.layers.AveragePooling1D,
                'GlobalMaxPooling2D': tf.keras.layers.GlobalMaxPooling2D,
                'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
                
                # Core layers
                'Dense': tf.keras.layers.Dense,
                'Flatten': tf.keras.layers.Flatten,
                'Dropout': tf.keras.layers.Dropout,
                'Reshape': tf.keras.layers.Reshape,
                'Permute': tf.keras.layers.Permute,
                
                # Normalization
                'BatchNormalization': tf.keras.layers.BatchNormalization,
                'LayerNormalization': tf.keras.layers.LayerNormalization,
                
                # Activation
                'Activation': tf.keras.layers.Activation,
                'ReLU': tf.keras.layers.ReLU,
                'LeakyReLU': tf.keras.layers.LeakyReLU,
                'PReLU': tf.keras.layers.PReLU,
                'ELU': tf.keras.layers.ELU,
                'Softmax': tf.keras.layers.Softmax,
                'Sigmoid': tf.keras.layers.Sigmoid,
                
                # Regularization
                'SpatialDropout2D': tf.keras.layers.SpatialDropout2D,
                'GaussianDropout': tf.keras.layers.GaussianDropout,
                'GaussianNoise': tf.keras.layers.GaussianNoise,
                
                # Advanced
                'Add': tf.keras.layers.Add,
                'Concatenate': tf.keras.layers.Concatenate,
                'Multiply': tf.keras.layers.Multiply,
                'Average': tf.keras.layers.Average,
                'Maximum': tf.keras.layers.Maximum,
                'Minimum': tf.keras.layers.Minimum,
                'Dot': tf.keras.layers.Dot,
                
                # Embedding
                'Embedding': tf.keras.layers.Embedding,
                
                # RNN layers (just in case)
                'LSTM': tf.keras.layers.LSTM,
                'GRU': tf.keras.layers.GRU,
                'SimpleRNN': tf.keras.layers.SimpleRNN,
                'Bidirectional': tf.keras.layers.Bidirectional,
                'TimeDistributed': tf.keras.layers.TimeDistributed,
                
                # Attention
                'Attention': tf.keras.layers.Attention,
                'MultiHeadAttention': tf.keras.layers.MultiHeadAttention,
            }
            
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects
            )
            print("✅ Model loaded successfully with Method 2!")
            
        except Exception as e2:
            print(f"⚠️ Method 2 failed: {str(e2)[:100]}...")
            
            try:
                # METHOD 3: Last resort - try to fix the model file
                print("\n🔄 Trying Method 3: Force load with legacy format...")
                
                # Try loading weights only
                from tensorflow.keras import layers, models
                
                # Create a dummy model with same architecture
                print("Building dummy model architecture...")
                dummy_model = models.Sequential([
                    layers.Input(shape=(224, 224, 3)),
                    layers.Conv2D(32, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(128, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(len(ACTUAL_CLASSES), activation='softmax')
                ])
                
                # Try to load weights
                dummy_model.load_weights(model_path)
                model = dummy_model
                print("✅ Model weights loaded with Method 3!")
                
            except Exception as e3:
                print(f"❌ All methods failed: {str(e3)[:100]}...")
                print("📋 Using mock mode")
                model = None
    
    # Load class names
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                data = json.load(f)
                class_names = data.get('class_names', ACTUAL_CLASSES)
            print(f"📊 Loaded {len(class_names)} classes from JSON")
        except Exception as e:
            print(f"⚠️ Error loading class info: {e}")
            class_names = ACTUAL_CLASSES
    else:
        print(f"⚠️ Info file not found, using default classes")
        class_names = ACTUAL_CLASSES
    
    # Test if model works
    if model is not None:
        try:
            print("\n🧪 Testing model with dummy input...")
            test_input = np.random.rand(1, 224, 224, 3).astype('float32')
            prediction = model.predict(test_input, verbose=0)
            print(f"✅ Model test passed! Output shape: {prediction.shape}")
            print(f"✅ Prediction sum: {np.sum(prediction):.4f}")
            
            # Test with actual image preprocessing
            print("\n🧪 Testing with image preprocessing...")
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            processed = cv2.resize(dummy_image, (224, 224))
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            processed = processed.astype('float32') / 255.0
            processed = np.expand_dims(processed, axis=0)
            
            test_pred = model.predict(processed, verbose=0)
            predicted_class = np.argmax(test_pred)
            confidence = test_pred[0][predicted_class] * 100
            print(f"✅ Image test passed! Predicted: {predicted_class} ({confidence:.1f}%)")
            
        except Exception as e:
            print(f"⚠️ Model test failed: {str(e)[:100]}...")
            print("📋 Switching to mock mode")
            model = None
    
    print(f"\n🎯 Final Status: {'REAL MODEL' if model is not None else 'MOCK MODE'}")
    print(f"🎯 Classes loaded: {len(class_names)}")
    if model is not None:
        print(f"🎯 Model input shape: {model.input_shape}")
        print(f"🎯 Model output shape: {model.output_shape}")
    print("=" * 60)

# ========== HEALTH CHECK ==========
@app.get("/")
async def root():
    return {
        "message": "🐟 HealthyFins API",
        "status": "active",
        "version": "3.1.0",
        "frontend": "https://healthy-fins.vercel.app",
        "model_loaded": model is not None,
        "model_type": "real" if model is not None else "mock",
        "num_classes": len(class_names),
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
    
    return {
        "status": "healthy",
        "service": "HealthyFins Backend",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": model is not None,
            "type": "real" if model is not None else "mock",
            "classes_count": len(class_names),
            "classes": class_names if len(class_names) <= 10 else f"{len(class_names)} classes"
        },
        "database": {
            "users": users_count,
            "history_entries": history_count,
            "file": "database.json"
        },
        "system": {
            "python_version": sys.version.split()[0],
            "tensorflow_version": tf.__version__,
            "environment": os.environ.get("RENDER", "development")
        },
        "links": {
            "frontend": "https://healthy-fins.vercel.app",
            "github": "https://github.com/yourusername/HealthyFins",
            "documentation": "https://healthyfins.onrender.com/docs"
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
def preprocess_image(image_bytes):
    """Preprocess image for AI model"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Same preprocessing as training
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        
        return np.expand_dims(img, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# ========== PROTECTED ENDPOINTS ==========
@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Predict fish disease from image"""
    try:
        print(f"🔍 Prediction request from: {current_user['sub']}")
        print(f"🎯 Model available: {'YES' if model is not None else 'NO (using mock)'}")
        
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
            print("⚠️ Using ENHANCED mock prediction (model not loaded)")
            # Enhanced mock that gives realistic probabilities
            predictions = np.random.rand(len(class_names))
            
            # Bias towards "Healthy Fish" (usually most common)
            if "Healthy Fish" in class_names:
                healthy_idx = class_names.index("Healthy Fish")
                predictions[healthy_idx] = predictions[healthy_idx] + 0.5
            
            # Normalize to sum to 1
            predictions = predictions / predictions.sum()
        else:
            print("✅ Using REAL model prediction")
            predictions = model.predict(processed_image, verbose=0)[0]
            print(f"📊 Raw predictions shape: {predictions.shape}")
            print(f"📊 Prediction sum: {np.sum(predictions):.4f}")
        
        # Get results
        best_class_idx = np.argmax(predictions)
        confidence = float(predictions[best_class_idx]) * 100
        disease_name = class_names[best_class_idx]
        
        # Get top 3 predictions
        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3 = []
        for idx in top3_idx:
            top3.append({
                "disease": class_names[int(idx)],
                "confidence": float(predictions[int(idx)]) * 100
            })
        
        # Save to history
        image_name = file.filename[:50]  # Truncate if too long
        db.add_prediction_history(
            user_id=current_user["user_id"],
            image_name=image_name,
            prediction=disease_name,
            confidence=confidence
        )
        
        print(f"✅ Prediction complete: {disease_name} ({confidence:.1f}%)")
        
        # Return result
        return {
            "success": True,
            "prediction": disease_name,
            "confidence": round(confidence, 2),
            "top3": top3,
            "model_type": "real" if model is not None else "mock",
            "model_available": model is not None,
            "user": {
                "id": current_user["user_id"],
                "email": current_user["sub"]
            },
            "timestamp": datetime.now().isoformat()
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
        
        db._save_data()
        
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
            "message": "Connect your Arduino/Raspberry Pi for real-time data",
            "integration_guide": "https://github.com/yourusername/HealthyFins"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PH monitoring error: {str(e)}")

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
print("🐟 HEALTHYFINS API - VERSION 3.1.0")
print("=" * 60)
print(f"📡 Backend URL: https://healthyfins.onrender.com")
print(f"🌐 Frontend URL: https://healthy-fins.vercel.app")
print(f"🤖 Model Loading: Attempting with multiple methods...")
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
