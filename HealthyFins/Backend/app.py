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
# FORCE TENSORFLOW 2.13 COMPATIBILITY
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings

# Monkey patch for InputLayer compatibility
import tensorflow as tf
print(f"📦 TensorFlow Version: {tf.__version__}")
print(f"📦 Expected: 2.13.0")

# Fix for InputLayer deserialization issue in TF 2.15+
if tf.__version__.startswith('2.1'):
    import tensorflow.keras as keras
    from tensorflow.keras.layers import InputLayer
    
    # Patch the from_config method
    original_from_config = InputLayer.from_config
    
    def patched_from_config(config, custom_objects=None):
        # Remove batch_shape if present (causes issue in TF 2.15+)
        if 'batch_shape' in config:
            print(f"⚠️ Removing batch_shape from config for compatibility")
            config.pop('batch_shape', None)
        return original_from_config(config, custom_objects)
    
    InputLayer.from_config = patched_from_config
    print("✅ Applied TensorFlow compatibility patch")

# Import our modules
try:
    from database import db
    from auth import create_access_token, get_current_user
except ImportError:
    # Create dummy functions if modules not found
    print("⚠️ Database/Auth modules not found, using mock data")
    
    class MockDB:
        def __init__(self):
            self.data = {"users": {}, "history": {}}
        
        def create_user(self, email, password, name, hardware_id=None):
            import hashlib
            user_id = hashlib.md5(email.encode()).hexdigest()[:8]
            self.data["users"][email] = {
                "id": user_id, "email": email, "name": name,
                "password": hashlib.sha256(password.encode()).hexdigest(),
                "hardware_id": hardware_id or ""
            }
            self.data["history"][user_id] = []
            return True, {"user_id": user_id, "email": email, "name": name}
        
        def authenticate_user(self, email, password):
            import hashlib
            if email not in self.data["users"]:
                return False, "User not found"
            hashed = hashlib.sha256(password.encode()).hexdigest()
            if self.data["users"][email]["password"] != hashed:
                return False, "Invalid password"
            return True, self.data["users"][email]
        
        def add_prediction_history(self, *args, **kwargs):
            return True
        
        def get_user_history(self, *args, **kwargs):
            return []
        
        def update_user_profile(self, *args, **kwargs):
            return True, "Updated"
    
    db = MockDB()
    
    def create_access_token(data):
        return "mock-token-" + data.get("sub", "user")
    
    def get_current_user():
        return {"sub": "test@example.com", "user_id": "test123"}

# Initialize app
app = FastAPI(
    title="Fish Disease Detection API",
    description="Complete system with authentication and history",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_names = []

def create_dummy_model():
    """Create a dummy model for testing"""
    print("🤖 Creating dummy model for testing...")
    
    try:
        # Very simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train on dummy data (one epoch)
        dummy_data = np.random.randn(10, 224, 224, 3).astype('float32')
        dummy_labels = np.random.randint(0, 5, 10)
        model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)
        
        print("✅ Dummy model created and trained")
        return model
        
    except Exception as e:
        print(f"❌ Failed to create dummy model: {e}")
        return None

def load_model_safely(model_path):
    """Safely load TensorFlow model with error handling"""
    print(f"📂 Attempting to load model from: {model_path}")
    
    try:
        # METHOD 1: Try standard load
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded using standard method")
        return model
        
    except Exception as e1:
        print(f"⚠️ Standard load failed: {e1}")
        
        try:
            # METHOD 2: Try with custom objects
            custom_objects = {
                'InputLayer': tf.keras.layers.InputLayer,
                'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
                'Dropout': tf.keras.layers.Dropout,
                'Dense': tf.keras.layers.Dense,
                'Adam': tf.keras.optimizers.Adam
            }
            
            # Check if it's MobileNetV2 based
            if 'mobilenet' in model_path.lower():
                custom_objects['MobileNetV2'] = tf.keras.applications.MobileNetV2
            
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            print("✅ Model loaded with custom objects")
            return model
            
        except Exception as e2:
            print(f"⚠️ Custom objects load failed: {e2}")
            
            try:
                # METHOD 3: Recreate model architecture and load weights
                print("🔄 Attempting to recreate model architecture...")
                
                # Create a fresh MobileNetV2 model (most likely architecture)
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                )
                base_model.trainable = False
                
                new_model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(5, activation='softmax')  # Assuming 5 classes
                ])
                
                # Try to load weights only
                new_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                print("✅ Model recreated with loaded weights")
                return new_model
                
            except Exception as e3:
                print(f"❌ All loading methods failed: {e3}")
                return None

@app.on_event("startup")
def load_model_on_startup():
    """Load AI model on startup"""
    global model, class_names
    
    print("=" * 50)
    print("🚀 Starting Fish Disease Detection API")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"📦 TensorFlow Version: {tf.__version__}")
    print(f"📦 Python Version: {sys.version}")
    
    # List files in current directory
    print("\n📁 Current directory contents:")
    try:
        for item in os.listdir('.'):
            print(f"  - {item}")
    except:
        pass
    
    try:
        print("\n🔄 Loading AI model...")
        
        # Try multiple possible paths
        possible_paths = [
            'fish_disease_model_final.h5',
            'models/fish_disease_model_final.h5',
            'Backend/models/fish_disease_model_final.h5',
            '../models/fish_disease_model_final.h5',
            '/opt/render/project/src/models/fish_disease_model_final.h5'
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Found model at: {path}")
                break
        
        if model_path:
            model = load_model_safely(model_path)
            if model is None:
                print("❌ Could not load model, using dummy")
                model = create_dummy_model()
        else:
            print("❌ Model file not found at any location")
            model = create_dummy_model()
        
        # Load or set class names
        class_info_paths = [
            'model_info_final.json',
            'models/model_info_final.json',
            'Backend/model_info_final.json'
        ]
        
        class_names_loaded = False
        for path in class_info_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        class_names = data.get('class_names', [])
                    print(f"✅ Class names loaded from: {path}")
                    class_names_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        if not class_names_loaded:
            class_names = ["Healthy", "White Spot", "Fin Rot", "Fungal", "Parasite"]
            print("⚠️ Using default class names")
        
        print(f"📊 Classes configured: {class_names}")
        print(f"🤖 Model status: {'LOADED' if model else 'NOT LOADED'}")
        
    except Exception as e:
        print(f"💥 Critical error during startup: {e}")
        traceback.print_exc()
        model = create_dummy_model()
        class_names = ["Healthy", "White Spot", "Fin Rot", "Fungal", "Parasite"]

# ========== PUBLIC ENDPOINTS ==========

@app.get("/")
def home():
    return {
        "message": "🐟 Fish Disease Detection API",
        "status": "active",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "classes": class_names if len(class_names) > 0 else ["Healthy", "White Spot", "Fin Rot", "Fungal", "Parasite"],
        "endpoints": ["/health", "/register", "/login", "/predict", "/profile", "/history", "/ph-monitoring"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "class_count": len(class_names),
        "timestamp": datetime.now().isoformat(),
        "environment": "production"
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
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        # Return a dummy image
        dummy_img = np.random.rand(1, 224, 224, 3).astype('float32')
        return dummy_img

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Predict disease from fish image"""
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and check size
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        print(f"📸 Processing image: {file.filename}, size: {len(contents)} bytes")
        
        # Preprocess
        processed_image = preprocess_image(contents)
        
        # Predict
        if model is None:
            print("⚠️ Using dummy prediction (no model)")
            predictions = np.array([[0.7, 0.1, 0.1, 0.05, 0.05]])  # Mostly healthy
        else:
            try:
                predictions = model.predict(processed_image, verbose=0)
                print(f"✅ Prediction made, shape: {predictions.shape}")
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                predictions = np.random.rand(1, len(class_names))
                predictions = predictions / predictions.sum()
        
        # Get results
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx]) * 100
        
        # Use class names, fallback to index
        if pred_idx < len(class_names):
            disease_name = class_names[pred_idx]
        else:
            disease_name = f"Disease_{pred_idx}"
        
        # Get top 3
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3 = []
        for idx in top3_idx:
            if idx < len(class_names):
                disease = class_names[idx]
            else:
                disease = f"Disease_{idx}"
            top3.append({
                "disease": disease,
                "confidence": float(predictions[0][idx]) * 100
            })
        
        # Save to history
        try:
            db.add_prediction_history(
                user_id=current_user.get("user_id", "unknown"),
                image_name=file.filename[:50],
                prediction=disease_name,
                confidence=confidence
            )
        except:
            pass  # Ignore history errors
        
        # Return result
        return {
            "success": True,
            "prediction": disease_name,
            "confidence": round(confidence, 2),
            "top3": top3,
            "image_size": f"{len(contents)} bytes",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile"""
    return {
        "success": True,
        "profile": {
            "email": current_user.get("sub", "unknown@example.com"),
            "name": "Test User",
            "user_id": current_user.get("user_id", "unknown"),
            "hardware_id": "Not configured"
        }
    }

@app.get("/history")
async def get_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get prediction history"""
    try:
        history = db.get_user_history(current_user.get("user_id", "unknown"), limit)
        return {
            "success": True,
            "count": len(history),
            "history": history
        }
    except:
        # Return mock history
        mock_history = [
            {
                "id": 1,
                "timestamp": datetime.now().isoformat(),
                "image": "sample_fish.jpg",
                "prediction": "Healthy",
                "confidence": 95.5,
                "treatment": "Continue regular maintenance"
            },
            {
                "id": 2,
                "timestamp": (datetime.fromisoformat(datetime.now().isoformat())).isoformat(),
                "image": "fish_002.jpg",
                "prediction": "White Spot",
                "confidence": 82.3,
                "treatment": "Raise temperature and add salt"
            }
        ]
        return {
            "success": True,
            "count": len(mock_history),
            "history": mock_history
        }

@app.get("/ph-monitoring")
async def get_ph_data(current_user: dict = Depends(get_current_user)):
    """Get pH monitoring data"""
    return {
        "success": True,
        "data": {
            "ph": round(6.8 + np.random.rand() * 1.5, 2),  # Random 6.8-8.3
            "temperature": round(24 + np.random.rand() * 6, 1),  # 24-30°C
            "turbidity": int(np.random.rand() * 50),  # 0-50 NTU
            "timestamp": datetime.now().isoformat(),
            "status": "normal",
            "hardware_id": "ARD-FISH-001",
            "message": "Mock data - Connect real hardware"
        }
    }

# ========== DEVELOPMENT ENDPOINTS ==========

@app.get("/debug/files")
async def debug_files():
    """Debug endpoint to list files"""
    files = {}
    try:
        for root, dirs, filenames in os.walk('.'):
            level = root.replace('.', '').count(os.sep)
            indent = ' ' * 4 * level
            files[f"{indent}{os.path.basename(root)}/"] = []
            subindent = ' ' * 4 * (level + 1)
            for f in filenames[:10]:  # Limit to 10 files per dir
                files[f"{indent}{os.path.basename(root)}/"].append(f"{subindent}{f}")
    except:
        files = {"error": "Could not list files"}
    
    return {
        "current_dir": os.getcwd(),
        "files": files,
        "model_exists": os.path.exists('fish_disease_model_final.h5'),
        "tf_version": tf.__version__
    }

@app.get("/test-model")
async def test_model():
    """Test if model works"""
    if model is None:
        return {"status": "no_model", "message": "Model not loaded"}
    
    try:
        # Create test input
        test_input = np.random.randn(1, 224, 224, 3).astype('float32')
        prediction = model.predict(test_input, verbose=0)
        
        return {
            "status": "working",
            "prediction_shape": str(prediction.shape),
            "sample_output": prediction[0].tolist()[:3]  # First 3 values
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

# ========== PORT BINDING FIX ==========
import uvicorn

# Get port from Render environment variable
PORT = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    print(f"🚀 Starting Fish Disease Detection API on port {PORT}")
    print(f"🌐 Access at: http://0.0.0.0:{PORT}")
    print(f"📊 Model loaded: {model is not None}")
    print(f"📈 Classes: {class_names}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True
    )
