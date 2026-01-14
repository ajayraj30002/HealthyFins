# app.py - COMPLETE Flask app for Render/Vercel deployments
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import h5py
from flask import Flask, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Load model ONCE when app starts
print("🚀 Loading MobileNetV2 model...")
MODEL_PATH = 'deployment_ready_model.h5'

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # Load class names from model metadata
    with h5py.File(MODEL_PATH, 'r') as f:
        if 'class_names' in f.attrs:
            class_names = json.loads(f.attrs['class_names'])
        else:
            class_names = [f'Class_{i}' for i in range(8)]
    
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Classes: {len(class_names)}")
    print(f"   Class names: {class_names}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    class_names = []

def predict_image(image_bytes):
    """
    Predict from image bytes
    
    Args:
        image_bytes: Bytes of image file (JPEG, PNG, etc.)
        
    Returns:
        dict with predictions
    """
    if model is None:
        return {'success': False, 'error': 'Model not loaded'}
    
    try:
        # Convert bytes to image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get original image info
        original_size = img.size
        original_format = img.format
        
        # Convert to numpy array (uint8, 0-255)
        # Model automatically handles:
        # 1. Resizing to 224x224
        # 2. Scaling from [0,255] to [-1,1]
        img_array = np.array(img)
        
        # Debug info
        print(f"📊 Image info: {original_size}, format: {original_format}, dtype: {img_array.dtype}, shape: {img_array.shape}")
        
        # Add batch dimension if needed
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Predict (NO manual preprocessing needed!)
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        # Prepare results
        result = {
            'success': True,
            'predictions': [],
            'top_prediction': None,
            'image_info': {
                'original_size': original_size,
                'format': original_format,
                'processed_to': '224x224'
            }
        }
        
        for idx in top_indices:
            prediction_data = {
                'class_index': int(idx),
                'class_name': class_names[idx] if idx < len(class_names) else f'Class_{idx}',
                'confidence': float(predictions[idx]),
                'confidence_percentage': f"{predictions[idx] * 100:.2f}%"
            }
            result['predictions'].append(prediction_data)
        
        result['top_prediction'] = result['predictions'][0]
        
        return result
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    """Home page with upload form"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MobileNetV2 Image Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            .upload-form { margin: 30px 0; }
            .result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
            .prediction { margin: 10px 0; padding: 10px; background: white; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🚀 MobileNetV2 Image Classifier</h1>
        <p>Upload an image for classification (8 classes)</p>
        
        <div class="upload-form">
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <br><br>
                <input type="submit" value="Classify Image">
            </form>
        </div>
        
        <div>
            <h3>API Usage:</h3>
            <pre>POST /predict with multipart/form-data containing 'image' file</pre>
        </div>
        
        <div>
            <h3>Classes:</h3>
            <ul>
    ''' + ''.join([f'<li>{name}</li>' for name in class_names]) + '''
            </ul>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        # Check if file is empty
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' not in image_file.filename or \
           image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Read image bytes
        image_bytes = image_file.read()
        
        # Check file size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            return jsonify({'success': False, 'error': 'File too large (max 10MB)'}), 400
        
        # Predict
        result = predict_image(image_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_name': 'MobileNetV2',
            'num_classes': len(class_names),
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape)
        })
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of class names"""
    return jsonify({
        'success': True,
        'classes': class_names,
        'count': len(class_names)
    })

@app.route('/info', methods=['GET'])
def get_info():
    """Get model information"""
    return jsonify({
        'model': 'MobileNetV2 with custom classifier',
        'preprocessing': 'Built-in: resize to 224x224, scale to [-1, 1]',
        'input_format': 'RGB image, uint8 (0-255), any size',
        'output_format': '8-class probabilities (softmax)',
        'classes': class_names
    })

# ==================== MAIN ====================

if __name__ == '__main__':
    # Get port from environment variable (Render/Vercel provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    print(f"🌐 Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
