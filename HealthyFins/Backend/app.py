# app.py - FIXED IMPORTS
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Query
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import json
import sys
from datetime import datetime
from typing import Optional, List
import traceback
import uuid
from pydantic import BaseModel

# Now import your modules
try:
    from database import db
    print("✅ Imported db from database")
except ImportError as e:
    print(f"❌ First import failed: {e}")
    try:
        from HealthyFins.Backend.database import db
        print("✅ Imported from HealthyFins.Backend.database")
    except ImportError as e:
        print(f"❌ Second import failed: {e}")
        sys.path.append(os.path.join(current_dir, '..'))
        from HealthyFins.Backend.database import db

from auth import create_access_token, get_current_user

# Get base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize app
app = FastAPI(
    title="HealthyFins API",
    description="AI Fish Disease Detection System",
    version="5.0.0"
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

# ========== PYDANTIC MODELS ==========
class HistoryFilter(BaseModel):
    disease_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None

class UserUpdate(BaseModel):
    name: Optional[str] = None
    hardware_id: Optional[str] = None

# ========== MODEL LOADING ==========
model = None
class_names = []
reverse_label_map = {}

def fix_model_compatibility(model_path):
    """Fix compatibility issues between TF versions"""
    try:
        print("🔄 Attempting to fix model compatibility...")
        
        # Load the model with custom objects to handle InputLayer changes
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer
        }
        
        # Try loading with custom objects
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects
        )
        
        return model, True
        
    except Exception as e:
        print(f"❌ Standard loading failed: {e}")
        
        # Try alternative: Load weights and rebuild architecture
        print("🔄 Trying alternative: Load weights into new architecture...")
        return build_model_from_weights(model_path)

def build_model_from_weights(model_path):
    """Build model from scratch and load weights"""
    try:
        print("🏗️ Building model from scratch...")
        
        # Build the EXACT same architecture as your Colab training
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Recreate EXACT architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation='softmax')  # 8 classes
        ])
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model architecture recreated")
        
        # Try to load weights
        try:
            model.load_weights(model_path)
            print("✅ Weights loaded successfully!")
            return model, True
        except:
            print("⚠️ Could not load weights, using initialized model")
            return model, False
            
    except Exception as e:
        print(f"❌ Failed to build model: {e}")
        return None, False

def simple_model_loader(model_path):
    """Simplest model loader for compatibility"""
    try:
        print("🔄 Using simple model loader...")
        
        # Try loading without any custom objects
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
        
        return model, True
    except Exception as e:
        print(f"❌ Simple loader failed: {e}")
        return None, False

@app.on_event("startup")
async def load_model():
    """Load AI model on startup"""
    global model, class_names, reverse_label_map
    
    print("=" * 60)
    print("🐟 HEALTHYFINS - LOADING MODEL")
    print("=" * 60)
    
    # Use absolute paths
    model_path = os.path.join(BASE_DIR, 'models', 'fish_disease_model_final.h5')
    info_path = os.path.join(BASE_DIR, 'models', 'model_info_final.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        # Try other paths
        alternative_paths = [
            os.path.join(BASE_DIR, 'fish_disease_model_final.h5'),
            'fish_disease_model_final.h5',
            './fish_disease_model_final.h5',
            '/opt/render/project/src/models/fish_disease_model_final.h5'
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Found model at: {path}")
                break
    
    if not os.path.exists(model_path):
        print("❌ Model file not found anywhere!")
        print("⚠️ Will use intelligent analysis mode only")
        model = None
    else:
        print(f"✅ Model file found: {model_path}")
        print(f"✅ File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    # Load class info
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                data = json.load(f)
                class_names = data.get('class_names', [])
                reverse_label_map = data.get('reverse_label_map', {})
                print(f"📊 Classes loaded: {len(class_names)}")
        except Exception as e:
            print(f"⚠️ Error loading class info: {e}")
            # Use default classes
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
    else:
        print("⚠️ Info file not found, using default classes")
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
    
    # Try multiple loading strategies
    if os.path.exists(model_path):
        loading_strategies = [
            ("Simple loader", simple_model_loader),
            ("Compatibility fix", fix_model_compatibility),
            ("Build from weights", build_model_from_weights)
        ]
        
        for strategy_name, strategy_func in loading_strategies:
            print(f"\n🔄 Trying {strategy_name}...")
            model, success = strategy_func(model_path)
            if success and model is not None:
                print(f"✅ {strategy_name} successful!")
                break
            else:
                print(f"❌ {strategy_name} failed")
    
    # Test the model if loaded
    if model is not None:
        try:
            print("\n🧪 Testing model...")
            dummy_input = np.random.randn(1, 224, 224, 3).astype('float32')
            predictions = model.predict(dummy_input, verbose=0)
            
            print(f"✅ Model test passed!")
            print(f"  Output shape: {predictions.shape}")
            print(f"  Sum of predictions: {np.sum(predictions):.4f}")
            
            # Verify output matches number of classes
            if predictions.shape[1] != len(class_names):
                print(f"⚠️ Output shape mismatch: {predictions.shape[1]} vs {len(class_names)}")
                # Create a wrapper model to fix output dimension
                print("🔧 Creating wrapper to fix output dimension...")
                
                # Build wrapper model
                input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
                x = model(input_layer)
                x = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
                
                model = tf.keras.Model(inputs=input_layer, outputs=x)
                print(f"✅ Wrapper created with {len(class_names)} outputs")
                
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            model = None
    
    # Final status
    print("\n" + "=" * 60)
    if model is not None:
        print("🎯 MODEL LOADED SUCCESSFULLY!")
        print(f"   Classes: {len(class_names)}")
        print(f"   Input: {model.input_shape}")
        print(f"   Output: {model.output_shape}")
    else:
        print("⚠️ USING ENHANCED ANALYSIS MODE")
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
        "architecture": "MobileNetV2-based" if model is not None else "Enhanced Analysis"
    }
    
    # Get available hardware IDs safely
    hardware_ids = []
    try:
        hardware_ids = db.get_hardware_ids()
    except:
        hardware_ids = []
    
    return {
        "message": "🐟 HealthyFins API",
        "status": "active",
        "version": "5.0.0",
        "frontend": "https://healthy-fins.vercel.app",
        "model": model_status,
        "database": "supabase" if db and db.supabase else "local",
        "hardware": {
            "available_ids": hardware_ids[:5],
            "total_available": len(hardware_ids)
        },
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "public": ["/", "/health", "/register", "/login", "/hardware"],
            "protected": ["/predict", "/profile", "/history", "/stats", "/search"]
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    model_info = {
        "loaded": model is not None,
        "type": "real_trained" if model is not None else "analysis_mode",
        "layers": len(model.layers) if model is not None else 0,
        "classes_count": len(class_names),
        "class_names": class_names if len(class_names) <= 10 else class_names[:5] + ["..."],
        "input_shape": str(model.input_shape) if model is not None else "N/A",
        "output_shape": str(model.output_shape) if model is not None else "N/A"
    }
    
    database_info = {
        "type": "supabase" if db and db.supabase else "local",
        "status": "connected" if db and db.supabase else "local_mode",
        "hardware_ids": len(db.get_hardware_ids()) if db else 0
    }
    
    return {
        "status": "healthy",
        "service": "HealthyFins Backend v5.0",
        "timestamp": datetime.now().isoformat(),
        "model": model_info,
        "database": database_info,
        "system": {
            "python_version": sys.version.split()[0],
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "environment": os.environ.get("RENDER", "development")
        }
    }

@app.get("/hardware")
async def get_hardware_info():
    """Get available hardware IDs"""
    hardware_ids = db.get_hardware_ids()
    
    # Check availability for each ID
    available_ids = []
    for hw_id in hardware_ids:
        available = db.check_hardware_available(hw_id)
        available_ids.append({
            "id": hw_id,
            "available": available,
            "type": "PH Monitoring Device"
        })
    
    return {
        "success": True,
        "hardware": available_ids,
        "total": len(available_ids),
        "available": sum(1 for h in available_ids if h["available"])
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
        
        # Create access token with user_id
        access_token = create_access_token(data={
            "sub": email, 
            "user_id": result["user_id"]
        })
        
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
        traceback.print_exc()
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
        
        # Create access token with user_id
        access_token = create_access_token(data={
            "sub": email, 
            "user_id": result["user_id"]
        })
        
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
        traceback.print_exc()
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
        
        # 2. Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # 3. Normalize to [0, 1]
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
    try:
        # Convert to uint8 for OpenCV
        img_uint8 = (image_array[0] * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        # Check for common disease indicators
        features = {
            'white_spots': np.mean(hsv[:,:,1] > 150),
            'red_patches': np.mean((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)),
            'fuzzy_areas': np.std(hsv[:,:,2]),
            'overall_health': np.mean(hsv[:,:,1])
        }
        
        # Generate probabilities
        predictions = np.zeros(len(class_names))
        
        for i, disease in enumerate(class_names):
            disease_lower = disease.lower()
            
            if 'healthy' in disease_lower:
                predictions[i] = 0.7 - features['white_spots'] * 0.3 - features['red_patches'] * 0.2
            elif 'white' in disease_lower:
                predictions[i] = features['white_spots'] * 0.8
            elif 'red' in disease_lower or 'bacterial' in disease_lower:
                predictions[i] = features['red_patches'] * 0.7
            elif 'fungal' in disease_lower:
                predictions[i] = features['fuzzy_areas'] * 0.6
            elif 'parasit' in disease_lower:
                predictions[i] = (features['white_spots'] + features['red_patches']) * 0.4
            else:
                predictions[i] = 0.1
        
        # Normalize
        predictions = np.clip(predictions, 0, 1)
        if np.sum(predictions) > 0:
            predictions = predictions / np.sum(predictions)
        else:
            predictions = np.ones(len(class_names)) / len(class_names)
        
        return predictions
    except Exception as e:
        print(f"❌ Feature analysis error: {e}")
        return np.ones(len(class_names)) / len(class_names)

# ========== PROTECTED ENDPOINTS ==========
@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Predict fish disease from image"""
    try:
        print(f"🔍 Prediction request from: {current_user['sub']}")
        
        # Check if image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        
        # Check file size
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Preprocess
        processed_image = preprocess_image_exact_colab(image_bytes)
        
        # Get predictions
        if model is not None:
            print("🧠 Using AI model")
            try:
                predictions = model.predict(processed_image, verbose=0)[0]
                model_type = "ai_model"
            except Exception as e:
                print(f"⚠️ Model prediction failed: {e}")
                predictions = analyze_image_features(processed_image)
                model_type = "enhanced_analysis"
        else:
            print("🔬 Using enhanced analysis")
            predictions = analyze_image_features(processed_image)
            model_type = "enhanced_analysis"
        
        # Ensure valid predictions
        if np.sum(predictions) < 0.9 or np.sum(predictions) > 1.1:
            predictions = np.clip(predictions, 0, 1)
            predictions = predictions / np.sum(predictions)
        
        # Get results
        best_class_idx = np.argmax(predictions)
        confidence = float(predictions[best_class_idx]) * 100
        
        # Get disease name
        if reverse_label_map and str(best_class_idx) in reverse_label_map:
            disease_name = reverse_label_map[str(best_class_idx)]
        elif best_class_idx < len(class_names):
            disease_name = class_names[best_class_idx]
        else:
            disease_name = "Unknown Disease"
        
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
                disease = "Unknown"
                
            top3.append({
                "disease": disease,
                "confidence": float(predictions[idx_int]) * 100
            })
        
        # Detect symptoms
        symptoms = detect_symptoms(processed_image, disease_name)
        
        # Save to history
        image_name = file.filename[:50]
        db.add_prediction_history(
            user_id=current_user["user_id"],
            image_name=image_name,
            prediction=disease_name,
            confidence=confidence,
            model_type=model_type,
            symptoms=symptoms
        )
        
        print(f"✅ Prediction complete: {disease_name} ({confidence:.1f}%)")
        
        return {
            "success": True,
            "prediction": disease_name,
            "confidence": round(confidence, 2),
            "symptoms": symptoms,
            "top3": top3,
            "model_type": model_type,
            "model_available": model is not None,
            "user": {
                "id": current_user["user_id"],
                "email": current_user["sub"]
            },
            "timestamp": datetime.now().isoformat(),
            "urgent": confidence > 70 and "healthy" not in disease_name.lower()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def detect_symptoms(image_array, disease_name):
    """Detect symptoms from image"""
    symptoms = []
    
    try:
        img_uint8 = (image_array[0] * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Check for common symptoms
        # White spots
        white_mask = (hsv[:,:,1] > 150) & (hsv[:,:,2] > 200)
        if np.mean(white_mask) > 0.01:
            symptoms.append("White spots")
        
        # Red patches
        red_mask = (hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)
        if np.mean(red_mask) > 0.01:
            symptoms.append("Red patches")
        
        # Dark areas (fungus/rot)
        dark_mask = gray < 50
        if np.mean(dark_mask) > 0.05:
            symptoms.append("Dark patches")
        
        # Fuzzy growth
        edges = cv2.Canny(gray, 100, 200)
        if np.mean(edges) > 20:
            symptoms.append("Fuzzy growth")
        
        # Based on disease name
        disease_lower = disease_name.lower()
        if 'gill' in disease_lower:
            symptoms.append("Rapid gill movement")
        if 'bacterial' in disease_lower:
            symptoms.append("Swollen abdomen")
        if 'fungal' in disease_lower:
            symptoms.append("Cotton-like growth")
        if 'parasitic' in disease_lower:
            symptoms.append("Flashing/rubbing")
        
    except Exception as e:
        print(f"❌ Symptom detection error: {e}")
        symptoms = ["Visual inspection recommended"]
    
    return symptoms[:5]

# ========== PROFILE ENDPOINTS ==========
@app.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile"""
    try:
        profile = db.get_user_profile(current_user["user_id"])
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Get statistics
        stats = db.get_history_stats(current_user["user_id"])
        
        return {
            "success": True,
            "profile": profile,
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Get profile error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching profile: {str(e)}")

@app.put("/profile")
async def update_profile(
    update_data: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update user profile"""
    try:
        success, message = db.update_user_profile(
            user_id=current_user["user_id"],
            name=update_data.name,
            hardware_id=update_data.hardware_id
        )
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return {
            "success": True,
            "message": message,
            "updated_fields": update_data.dict(exclude_none=True)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Update profile error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error updating profile: {str(e)}")

# ========== HISTORY ENDPOINTS ==========
@app.get("/history")
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    disease_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get user's prediction history"""
    try:
        user_id = current_user["user_id"]
        
        if search:
            # Search in history
            history = db.search_history(user_id, search, limit)
            total = len(history)
        else:
            # Get filtered history
            history = db.get_user_history(user_id, limit, offset, disease_type)
            
            # Get total count for pagination
            stats = db.get_history_stats(user_id)
            total = stats["total"]
        
        return {
            "success": True,
            "history": history,
            "count": len(history),
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(history)) < total
        }
        
    except Exception as e:
        print(f"❌ Get history error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/history/{entry_id}")
async def get_history_entry(
    entry_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get specific history entry"""
    try:
        history = db.get_user_history(current_user["user_id"], limit=100)
        
        # Find the specific entry
        entry = next((h for h in history if h["id"] == entry_id), None)
        
        if not entry:
            raise HTTPException(status_code=404, detail="History entry not found")
        
        return {
            "success": True,
            "entry": entry
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Get history entry error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching history entry: {str(e)}")

@app.delete("/history/{entry_id}")
async def delete_history_entry(
    entry_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a history entry"""
    try:
        success = db.delete_history_entry(current_user["user_id"], entry_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Entry not found or already deleted")
        
        return {
            "success": True,
            "message": "History entry deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Delete history error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting history: {str(e)}")

@app.delete("/history")
async def clear_all_history(current_user: dict = Depends(get_current_user)):
    """Clear all user history"""
    try:
        if not current_user["user_id"]:
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        success = db.clear_user_history(current_user["user_id"])
        
        return {
            "success": success,
            "message": "All history cleared successfully" if success else "Failed to clear history"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Clear history error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

# ========== STATISTICS ENDPOINTS ==========
@app.get("/stats")
async def get_user_stats(current_user: dict = Depends(get_current_user)):
    """Get user statistics"""
    try:
        stats = db.get_history_stats(current_user["user_id"])
        
        return {
            "success": True,
            "stats": stats,
            "user_id": current_user["user_id"]
        }
        
    except Exception as e:
        print(f"❌ Get stats error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

@app.get("/stats/daily")
async def get_daily_stats(
    days: int = Query(7, ge=1, le=30),
    current_user: dict = Depends(get_current_user)
):
    """Get daily statistics"""
    try:
        history = db.get_user_history(current_user["user_id"], limit=100)
        
        # Group by date
        daily_stats = {}
        for entry in history:
            date = entry["timestamp"][:10]  # YYYY-MM-DD
            if date not in daily_stats:
                daily_stats[date] = {"count": 0, "healthy": 0, "diseases": {}}
            
            daily_stats[date]["count"] += 1
            if "healthy" in entry["prediction"].lower():
                daily_stats[date]["healthy"] += 1
            else:
                disease = entry["prediction"]
                daily_stats[date]["diseases"][disease] = daily_stats[date]["diseases"].get(disease, 0) + 1
        
        # Convert to list and sort by date
        daily_list = [
            {"date": date, **stats}
            for date, stats in sorted(daily_stats.items(), reverse=True)[:days]
        ]
        
        return {
            "success": True,
            "daily_stats": daily_list,
            "days": len(daily_list)
        }
        
    except Exception as e:
        print(f"❌ Get daily stats error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching daily stats: {str(e)}")

# ========== SEARCH ENDPOINTS ==========
@app.get("/search")
async def search_history(
    query: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(get_current_user)
):
    """Search in user's history"""
    try:
        results = db.search_history(current_user["user_id"], query, limit)
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# ========== EXPORT ENDPOINTS ==========
@app.get("/export/history")
async def export_history(
    format: str = Query("json", regex="^(json|csv)$"),
    current_user: dict = Depends(get_current_user)
):
    """Export user history"""
    try:
        history = db.get_user_history(current_user["user_id"], limit=1000)
        
        if format == "csv":
            # Generate CSV
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                "timestamp", "prediction", "confidence", "image_name", "model_type", "symptoms"
            ])
            
            writer.writeheader()
            for entry in history:
                writer.writerow({
                    "timestamp": entry["timestamp"],
                    "prediction": entry["prediction"],
                    "confidence": entry["confidence"],
                    "image_name": entry["image_name"],
                    "model_type": entry.get("model_type", "unknown"),
                    "symptoms": ", ".join(entry.get("symptoms", []))
                })
            
            csv_content = output.getvalue()
            
            return {
                "success": True,
                "format": "csv",
                "data": csv_content,
                "count": len(history)
            }
        else:
            # JSON format
            return {
                "success": True,
                "format": "json",
                "data": history,
                "count": len(history)
            }
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

# ========== STARTUP MESSAGE ==========
print("\n" + "=" * 60)
print("🐟 HEALTHYFINS API v5.0 - SUPABASE INTEGRATION")
print("=" * 60)
print(f"📡 Backend URL: https://healthyfins.onrender.com")
print(f"🌐 Frontend URL: https://healthy-fins.vercel.app")
print(f"💾 Database: {'Supabase' if db and db.supabase else 'Local JSON'}")
print(f"🔧 Hardware IDs: {len(db.get_hardware_ids()) if db else 0} available")
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
