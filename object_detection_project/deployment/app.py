"""
Hard Hat / PPE Detection - Deployment App
==========================================
Simple Flask app for deploying PPE detection model.
"""

from flask import Flask, request, jsonify, render_template_string
from pathlib import Path
import torch
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load model
MODEL_PATH = Path('runs/detect/ppe_detection_phase2/weights/best.pt')
if not MODEL_PATH.exists():
    MODEL_PATH = Path('runs/detect/ppe_detection_phase1/weights/best.pt')

if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))
    print(f"✅ Loaded model: {MODEL_PATH}")
else:
    model = None
    print("⚠️  Model not found. Please train the model first.")

device = 0 if torch.cuda.is_available() else 'cpu'
class_names = ['head', 'helmet', 'person']


@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hard Hat / PPE Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background: #45a049; }
            #result { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Hard Hat / PPE Detection</h1>
        <div class="upload-area">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="imageInput" accept="image/*" required>
                <br><br>
                <button type="submit">Detect PPE</button>
            </form>
        </div>
        <div id="result"></div>
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData();
                formData.append('image', document.getElementById('imageInput').files[0]);
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                document.getElementById('result').innerHTML = 
                    '<h2>Results:</h2><pre>' + JSON.stringify(data, null, 2) + '</pre>';
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image
        image = Image.open(file.stream).convert('RGB')
        
        # Run prediction
        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            device=device,
            verbose=False
        )
        
        # Format results
        detections = []
        if results[0].boxes is not None:
            for i in range(len(results[0].boxes)):
                cls_id = int(results[0].boxes.cls[i])
                conf = float(results[0].boxes.conf[i])
                box = results[0].boxes.xyxy[i].cpu().numpy().tolist()
                
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': box
                })
        
        return jsonify({
            'detections': detections,
            'count': len(detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)




