"""
FastAPI MobileNetV2 Image Classification API
Deploy to Render/Vercel with this file
"""

import os
import io
import json
import h5py
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

# Initialize FastAPI app
app = FastAPI(
    title="MobileNetV2 Image Classification API",
    description="Classify images into 8 categories using MobileNetV2",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_names = []
model_metadata = {}

def load_model():
    """Load the model and class names from the H5 file"""
    global model, class_names, model_metadata
    
    try:
        MODEL_PATH = "deployment_ready_model.h5"
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        print(f"🚀 Loading model from: {MODEL_PATH}")
        
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Load metadata from H5 file
        with h5py.File(MODEL_PATH, 'r') as f:
            # Get class names
            if 'class_names' in f.attrs:
                class_names = json.loads(f.attrs['class_names'])
            else:
                class_names = [f"Class_{i}" for i in range(8)]
            
            # Get all metadata
            model_metadata = {}
            for key in f.attrs:
                try:
                    value = f.attrs[key]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    model_metadata[key] = value
                except:
                    model_metadata[key] = "binary_data"
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Number of classes: {len(class_names)}")
        print(f"   Class names: {class_names}")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model when the application starts"""
    load_model()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for MobileNetV2
    The model already has built-in preprocessing, but we ensure proper format
    
    Args:
        image: PIL Image object
        
    Returns:
        numpy array ready for prediction
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array (uint8, 0-255)
    img_array = np.array(image)
    
    # Add batch dimension
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(image_array: np.ndarray) -> dict:
    """
    Make prediction on image array
    
    Args:
        image_array: numpy array of image
        
    Returns:
        dict with prediction results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Make prediction
        predictions = model.predict(image_array, verbose=0)[0]
        
        # Get top predictions
        top_indices = np.argsort(predictions)[-5:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                "class_index": int(idx),
                "class_name": class_names[idx] if idx < len(class_names) else f"Class_{idx}",
                "confidence": float(predictions[idx]),
                "confidence_percentage": f"{predictions[idx] * 100:.2f}%"
            })
        
        return {
            "success": True,
            "predictions": results,
            "top_prediction": results[0],
            "all_probabilities": predictions.tolist()
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MobileNetV2 Image Classification API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload image for classification",
            "GET /health": "Health check",
            "GET /classes": "Get class names",
            "GET /info": "Get model information",
            "GET /docs": "API documentation (Swagger UI)"
        }
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file to classify"),
    return_image_info: bool = False
):
    """
    Predict image class
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        return_image_info: Whether to include image information in response
        
    Returns:
        JSON with prediction results
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (max 10MB)
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get image info
        image_info = {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": file_size
        }
        
        # Preprocess image
        image_array = preprocess_image(image)
        
        # Make prediction
        result = predict_image(image_array)
        
        # Add image info if requested
        if return_image_info:
            result["image_info"] = image_info
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is not None:
        return {
            "status": "healthy",
            "model_loaded": True,
            "model": "MobileNetV2",
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "num_classes": len(class_names)
        }
    else:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": "Model not loaded"
        }

@app.get("/classes")
async def get_classes():
    """Get list of class names"""
    return {
        "success": True,
        "classes": class_names,
        "count": len(class_names)
    }

@app.get("/info")
async def get_model_info():
    """Get model information and metadata"""
    return {
        "model": "MobileNetV2 with custom classifier",
        "preprocessing": "Built-in: resize to 224x224, scale to [-1, 1]",
        "input_format": "RGB image, uint8 (0-255), any size",
        "output_format": f"{len(class_names)}-class probabilities (softmax)",
        "classes": class_names,
        "metadata": model_metadata
    }

@app.get("/test-prediction")
async def test_prediction():
    """Test endpoint with a random image"""
    try:
        # Create a random test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_pil = Image.fromarray(test_image)
        
        # Preprocess and predict
        image_array = preprocess_image(test_pil)
        result = predict_image(image_array)
        
        return {
            "message": "Test prediction successful",
            "test_image": "Random 224x224 image",
            "prediction": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all other exceptions"""
    print(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Render/Vercel provides this)
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting FastAPI server on port {port}...")
    print(f"📚 API Documentation: http://localhost:{port}/docs")
    
    uvicorn.run(
        "app:app",  # Change to "__main__:app" if running directly
        host="0.0.0.0",
        port=port,
        reload=False  # Set to True for development only
    )
