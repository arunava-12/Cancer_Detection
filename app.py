# app.py (updated)
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = 'your-secret-key-123'  # Change this for production

# Create upload directory immediately
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model = load_model(r"D:\VS-Code\Cancer_Detection\bestmodel.h5")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Generate secure filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = secure_filename(f"{timestamp}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save file
            file.save(filepath)
            
            # Process and predict
            processed_img = preprocess_image(filepath)
            prediction = model.predict(processed_img)
            
            # Format results
            tumor_prob = float(prediction[0][0])
            result = "Tumor Detected" if tumor_prob > 0.5 else "No Tumor"
            confidence = tumor_prob if result == "Tumor Detected" else 1 - tumor_prob
            
            return render_template('result.html', 
                                 result=result,
                                 confidence=round(confidence * 100, 2),
                                 uploaded_image=filename)
        
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
