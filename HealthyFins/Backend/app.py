# app.py - COMPLETE FIXED VERSION WITH SSL KAFKA FIX & RENDER HEALTH CHECK
import os
import sys
import asyncio
import ssl  # Added SSL import for Kafka

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import json
from datetime import datetime
from typing import Optional, List
import traceback
import uuid
from pydantic import BaseModel
from aiokafka import AIOKafkaConsumer

# Import our modules
try:
    from database import db
    print(f"✅ Successfully imported db: {db}")
except Exception as e:
    print(f"❌ Failed to import db: {e}")
    db = None

try:
    import iot_bridge
    print("✅ Successfully imported iot_bridge")
except ImportError:
    print("⚠️ iot_bridge module not found")
    iot_bridge = None

from auth import create_access_token, get_current_user

# Get base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== CREATE FASTAPI APP ==========
app = FastAPI(
    title="HealthyFins API",
    description="AI Fish Disease Detection System",
    version="5.0.0"
)

# ========== CORS CONFIGURATION ==========
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

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

# ========== PH MONITORING MODELS ==========
class SensorData(BaseModel):
    ph: float
    temperature: Optional[float] = None
    turbidity: Optional[float] = None
    hardware_id: str
    timestamp: Optional[str] = None

# ========== IN-MEMORY STORAGE ==========
latest_sensor_readings = {}

# ========== KAFKA CONSUMER TASK ==========
async def consume_kafka_stream():
    """Background task to consume from Aiven Kafka"""
    print("🎧 Starting Kafka Consumer Task...")
    AIVEN_KAFKA_SERVER = os.getenv("AIVEN_KAFKA_SERVER")
    AIVEN_KAFKA_USER = os.getenv("AIVEN_KAFKA_USER")
    AIVEN_KAFKA_PASS = os.getenv("AIVEN_KAFKA_PASS")
    
    if not all([AIVEN_KAFKA_SERVER, AIVEN_KAFKA_USER, AIVEN_KAFKA_PASS]):
        print("⚠️ Missing Aiven Kafka credentials. Consumer will not start.")
        return

    # Create the SSL Context required by aiokafka
    ssl_context = ssl.create_default_context()

    try:
        consumer = AIOKafkaConsumer(
            'healthyfins-telemetry',
            bootstrap_servers=AIVEN_KAFKA_SERVER,
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_plain_username=AIVEN_KAFKA_USER,
            sasl_plain_password=AIVEN_KAFKA_PASS,
            ssl_context=ssl_context,  # Added SSL Context
            group_id="fastapi-group"
        )
        await consumer.start()
        print("✅ Kafka Consumer connected and listening!")
        
        try:
            async for msg in consumer:
                data = json.loads(msg.value.decode('utf-8'))
                hardware_id = data.get("hardware_id", "unknown")
                
                # Assign server timestamp if missing
                if "timestamp" not in data or not data["timestamp"]:
                    data["timestamp"] = datetime.now().isoformat()
                    
                latest_sensor_readings[hardware_id] = {
                    "ph": data.get("ph", 7.0),
                    "temperature": data.get("temperature", 25.0),
                    "turbidity": data.get("turbidity", 10.0),
                    "timestamp": data["timestamp"],
                    "hardware_id": hardware_id
                }
                print(f"💾 Kafka Consumer updated state for {hardware_id}: pH {data.get('ph')}")
        finally:
            await consumer.stop()
    except Exception as e:
        print(f"❌ Kafka Consumer error: {e}")

# ========== MODEL LOADING - FIXED VERSION ==========
model = None
class_names = []
reverse_label_map = {}

def custom_load_model(model_path):
    """Custom model loader that handles batch_shape error"""
    try:
        print("🔄 Using custom model loader...")
        
        # Register custom InputLayer to handle batch_shape
        class CompatibleInputLayer(tf.keras.layers.InputLayer):
            def __init__(self, **kwargs):
                if 'batch_shape' in kwargs:
                    kwargs['shape'] = kwargs['batch_shape'][1:]
                    del kwargs['batch_shape']
                super().__init__(**kwargs)
        
        # Try loading with custom objects
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'InputLayer': CompatibleInputLayer},
            compile=False
        )
        print("✅ Model loaded with compatibility fix!")
        return model, True
        
    except Exception as e:
        print(f"❌ Custom loader failed: {e}")
        return None, False

def load_model_from_weights(model_path):
    """Rebuild model architecture and load weights"""
    try:
        print("🏗️ Rebuilding model from scratch...")
        
        # Build the same architecture as your trained model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        # Try to load weights
        model.load_weights(model_path)
        print("✅ Weights loaded successfully!")
        return model, True
        
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        return None, False

@app.on_event("startup")
async def startup_event():
    """Load AI model and start Background Streams on startup"""
    global model, class_names, reverse_label_map
    
    print("=" * 60)
    print("🐟 HEALTHYFINS - STARTUP SEQUENCE")
    print("=" * 60)
    
    # 1. START PIPELINES
    if iot_bridge:
        iot_bridge.start_mqtt_bridge()
    asyncio.create_task(consume_kafka_stream())
    
    # 2. LOAD MODEL
    model_path = os.path.join(BASE_DIR, 'models', 'fish_disease_model_final.h5')
    info_path = os.path.join(BASE_DIR, 'models', 'model_info_final.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
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
            class_names = [
                "Bacterial Red disease", "Parasitic diseases", 
                "Viral diseases White tail disease", "Fungal diseases Saprolegniasis",
                "Bacterial diseases - Aeromoniasis", "Bacterial gill disease",
                "Healthy Fish", "EUS_Ulcerative_Syndrome (arg)"
            ]
            reverse_label_map = {str(i): class_names[i] for i in range(len(class_names))}
    else:
        print("⚠️ Info file not found, using default classes")
        class_names = [
            "Bacterial Red disease", "Parasitic diseases", 
            "Viral diseases White tail disease", "Fungal diseases Saprolegniasis",
            "Bacterial diseases - Aeromoniasis", "Bacterial gill disease",
            "Healthy Fish", "EUS_Ulcerative_Syndrome (arg)"
        ]
        reverse_label_map = {str(i): class_names[i] for i in range(len(class_names))}
    
    print(f"📊 Total classes: {len(class_names)}")
    print(f"📊 TensorFlow version: {tf.__version__}")
    
    # Try loading strategies
    if os.path.exists(model_path):
        loading_strategies = [
            ("Custom loader", custom_load_model),
            ("Build from weights", load_model_from_weights),
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
            print(f"✅ Model test passed! Output shape: {predictions.shape}")
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            model = None
    
    # Final status
    print("\n" + "=" * 60)
    if model is not None:
        print("🎯 MODEL LOADED SUCCESSFULLY!")
        print(f"   Classes: {len(class_names)}")
    else:
        print("⚠️ USING ENHANCED ANALYSIS MODE")
        print(f"   Classes: {len(class_names)}")
    print("=" * 60)

# ========== HEALTH CHECK ==========
@app.head("/")
async def head_root():
    """Health check ping for Render monitors"""
    return JSONResponse(content={"status": "ok"})

@app.get("/")
async def root():
    """Root endpoint with API info"""
    try:
        model_status = {
            "loaded": model is not None,
            "type": "real_trained" if model is not None else "analysis_mode",
            "classes": len(class_names)
        }
        
        hardware_ids = []
        try:
            if db:
                hardware_ids = db.get_hardware_ids()
        except:
            hardware_ids = []
        
        return {
            "success": True,
            "message": "🐟 HealthyFins API",
            "status": "active",
            "version": "5.0.0",
            "frontend": "https://healthy-fins.vercel.app",
            "model": model_status,
            "database": "connected" if db else "disconnected",
            "hardware": {
                "available_ids": hardware_ids[:5],
                "total_available": len(hardware_ids)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "message": "Error loading API info",
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        model_info = {
            "loaded": model is not None,
            "classes_count": len(class_names),
        }
        
        database_info = {
            "type": "Supabase REST API",
            "status": "connected" if db else "disconnected",
        }
        
        return {
            "success": True,
            "status": "healthy",
            "service": "HealthyFins Backend v5.0",
            "timestamp": datetime.now().isoformat(),
            "model": model_info,
            "database": database_info
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e)
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
        
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        success, result = db.create_user(email, password, name, hardware_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=result)
        
        access_token = create_access_token(data={
            "sub": email, 
            "user_id": result["user_id"]
        })
        
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
        
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        success, result = db.authenticate_user(email, password)
        
        if not success:
            raise HTTPException(status_code=401, detail=result)
        
        access_token = create_access_token(data={
            "sub": email, 
            "user_id": result["user_id"]
        })
        
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
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def analyze_image_features(image_array):
    """Enhanced image analysis when model isn't available"""
    try:
        img_uint8 = (image_array[0] * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        features = {
            'white_spots': np.mean(hsv[:,:,1] > 150),
            'red_patches': np.mean((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)),
            'fuzzy_areas': np.std(hsv[:,:,2]),
        }
        
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
        
        predictions = np.clip(predictions, 0, 1)
        if np.sum(predictions) > 0:
            predictions = predictions / np.sum(predictions)
        else:
            predictions = np.ones(len(class_names)) / len(class_names)
        
        return predictions
    except Exception as e:
        print(f"❌ Feature analysis error: {e}")
        return np.ones(len(class_names)) / len(class_names)

def detect_symptoms(image_array, disease_name):
    """Detect symptoms from image"""
    symptoms = []
    try:
        img_uint8 = (image_array[0] * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        white_mask = (hsv[:,:,1] > 150) & (hsv[:,:,2] > 200)
        if np.mean(white_mask) > 0.01:
            symptoms.append("White spots")
        
        red_mask = (hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)
        if np.mean(red_mask) > 0.01:
            symptoms.append("Red patches")
        
        dark_mask = gray < 50
        if np.mean(dark_mask) > 0.05:
            symptoms.append("Dark patches")
        
        edges = cv2.Canny(gray, 100, 200)
        if np.mean(edges) > 20:
            symptoms.append("Fuzzy growth")
        
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

# ========== PROTECTED ENDPOINTS ==========
@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Predict fish disease from image"""
    try:
        print(f"🔍 Prediction request from: {current_user['sub']}")
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        processed_image = preprocess_image_exact_colab(image_bytes)
        
        if model is not None:
            try:
                predictions = model.predict(processed_image, verbose=0)[0]
                model_type = "ai_model"
            except Exception as e:
                print(f"⚠️ Model prediction failed: {e}")
                predictions = analyze_image_features(processed_image)
                model_type = "enhanced_analysis"
        else:
            predictions = analyze_image_features(processed_image)
            model_type = "enhanced_analysis"
        
        if np.sum(predictions) < 0.9 or np.sum(predictions) > 1.1:
            predictions = np.clip(predictions, 0, 1)
            predictions = predictions / np.sum(predictions)
        
        best_class_idx = np.argmax(predictions)
        confidence = float(predictions[best_class_idx]) * 100
        
        if reverse_label_map and str(best_class_idx) in reverse_label_map:
            disease_name = reverse_label_map[str(best_class_idx)]
        elif best_class_idx < len(class_names):
            disease_name = class_names[best_class_idx]
        else:
            disease_name = "Unknown Disease"
        
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
        
        symptoms = detect_symptoms(processed_image, disease_name)
        
        if db:
            image_name = file.filename[:50]
            db.add_prediction_history(
                user_id=current_user["user_id"],
                image_name=image_name,
                prediction=disease_name,
                confidence=confidence,
                model_type=model_type,
                symptoms=symptoms
            )
        
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

# ========== PROFILE ENDPOINTS ==========
@app.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile"""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        profile = db.get_user_profile(current_user["user_id"])
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
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
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
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

# ========== PH MONITORING ENDPOINTS ==========
@app.post("/ph-monitoring/data")
async def receive_sensor_data(data: SensorData):
    """Fallback manual HTTP POST (The stream is handled via Kafka)"""
    try:
        if not data.timestamp:
            data.timestamp = datetime.now().isoformat()
        
        latest_sensor_readings[data.hardware_id] = {
            "ph": data.ph,
            "temperature": data.temperature,
            "turbidity": data.turbidity,
            "timestamp": data.timestamp,
            "hardware_id": data.hardware_id
        }
        
        print(f"📊 Manual HTTP POST received from {data.hardware_id}: pH={data.ph}")
        
        return {
            "success": True,
            "message": "Sensor data received via HTTP",
            "timestamp": data.timestamp
        }
        
    except Exception as e:
        print(f"❌ Error receiving sensor data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ph-monitoring/latest")
async def get_latest_sensor_data(
    hardware_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get latest sensor readings (populated by Kafka consumer)"""
    try:
        if not hardware_id:
            if not db:
                return {
                    "success": True,
                    "data": {
                        "ph": 7.2,
                        "temperature": 25.5,
                        "turbidity": 12,
                        "timestamp": datetime.now().isoformat(),
                        "status": "mock",
                        "hardware_id": "unknown"
                    }
                }
            
            user_profile = db.get_user_profile(current_user["user_id"])
            if not user_profile:
                raise HTTPException(status_code=404, detail="User profile not found")
            
            hardware_id = user_profile.get("hardware_id")
        
        if not hardware_id:
            return {
                "success": True,
                "data": {
                    "ph": 7.2,
                    "temperature": 25.5,
                    "turbidity": 12,
                    "timestamp": datetime.now().isoformat(),
                    "status": "mock",
                    "hardware_id": "none",
                    "message": "No hardware configured. Add hardware ID in profile."
                }
            }
        
        reading = latest_sensor_readings.get(hardware_id)
        
        if reading:
            return {
                "success": True,
                "data": {
                    **reading,
                    "status": "real"
                }
            }
        else:
            return {
                "success": True,
                "data": {
                    "ph": 7.0,
                    "temperature": 26.0,
                    "turbidity": 10,
                    "timestamp": None,
                    "status": "waiting",
                    "hardware_id": hardware_id,
                    "message": f"Waiting for data from Kafka stream for device {hardware_id}"
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting sensor data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== HISTORY, STATS, SEARCH & EXPORT (Unchanged) ==========
@app.get("/history")
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    disease_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        if not db: raise HTTPException(status_code=500, detail="Database not connected")
        user_id = current_user["user_id"]
        
        if search:
            history = db.search_history(user_id, search, limit)
            total = len(history)
        else:
            history = db.get_user_history(user_id, limit, offset, disease_type)
            stats = db.get_history_stats(user_id)
            total = stats["total"]
        
        return {
            "success": True, "history": history, "count": len(history),
            "total": total, "limit": limit, "offset": offset,
            "has_more": (offset + len(history)) < total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/history/{entry_id}")
async def get_history_entry(entry_id: str, current_user: dict = Depends(get_current_user)):
    try:
        if not db: raise HTTPException(status_code=500, detail="Database not connected")
        history = db.get_user_history(current_user["user_id"], limit=100)
        entry = next((h for h in history if h["id"] == entry_id), None)
        if not entry: raise HTTPException(status_code=404, detail="Entry not found")
        return {"success": True, "entry": entry}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{entry_id}")
async def delete_history_entry(entry_id: str, current_user: dict = Depends(get_current_user)):
    try:
        if not db: raise HTTPException(status_code=500, detail="Database not connected")
        success = db.delete_history_entry(current_user["user_id"], entry_id)
        if not success: raise HTTPException(status_code=404, detail="Not found")
        return {"success": True, "message": "Deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history")
async def clear_all_history(current_user: dict = Depends(get_current_user)):
    try:
        if not db: raise HTTPException(status_code=500, detail="Database not connected")
        success = db.clear_user_history(current_user["user_id"])
        return {"success": success, "message": "Cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_user_stats(current_user: dict = Depends(get_current_user)):
    try:
        if not db: raise HTTPException(status_code=500, detail="Database not connected")
        stats = db.get_history_stats(current_user["user_id"])
        return {"success": True, "stats": stats, "user_id": current_user["user_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_history(
    query: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(get_current_user)
):
    try:
        if not db: raise HTTPException(status_code=500, detail="Database not connected")
        results = db.search_history(current_user["user_id"], query, limit)
        return {"success": True, "query": query, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/history")
async def export_history(
    format: str = Query("json", regex="^(json|csv)$"),
    current_user: dict = Depends(get_current_user)
):
    try:
        if not db: raise HTTPException(status_code=500, detail="Database not connected")
        history = db.get_user_history(current_user["user_id"], limit=1000)
        
        if format == "csv":
            import csv, io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["timestamp", "prediction", "confidence", "image_name", "model_type", "symptoms"])
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
            return {"success": True, "format": "csv", "data": output.getvalue(), "count": len(history)}
        return {"success": True, "format": "json", "data": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== STARTUP MESSAGE ==========
print("\n" + "=" * 60)
print("🐟 HEALTHYFINS API v5.0 - EVENT-DRIVEN KAFKA PIPELINE (AIVEN)")
print("=" * 60)
print(f"💾 Database: Supabase REST API")
print(f"🔧 Hardware IDs: {len(db.get_hardware_ids()) if db else 0} available")
print(f"📊 PH Monitoring: Real-time Kafka Consumer enabled")
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
