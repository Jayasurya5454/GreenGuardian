from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import io

app = Flask(__name__)

model_path = 'model.h5'
model = load_model(model_path)
class_labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry__Powdery_mildew',
    6: 'Cherry__healthy',
    7: 'Corn_Cercospora_spot',
    8: 'Corn Common_rust',
    9: 'Corn_Leaf_Blight',
    10: 'Corn__healthy',
    11: 'Grape_Black_rot',
    12: 'Grape_Esca_(Black_Measles)',
    13: 'Grape_Isariopsis_Leaf_Spot',
    14: 'Grape_healthy',
    15: 'Orange__Citrus_greening',
    16: 'Peach__Bacterial_spot',
    17: 'Peach__healthy',
    18: 'Pepper bell_Bacterial_spot',
    19: 'Pepper bell_healthy',
    20: 'Potato_Early_blight',
    21: 'Potato_Late_blight',
    22: 'Potato_healthy',
    23: 'Raspberry_healthy',
    24: 'Soybean_healthy',
    25: 'Squash_Powdery_mildew',
    26: 'Strawberry_Leaf_scorch',
    27: 'Strawberry_healthy',
    28: 'Tomato_Bacterial_spot',
    29: 'Tomato_Early_blight',
    30: 'Tomato_Late_blight',
    31: 'Tomato_Leaf_Mold',
    32: 'Tomato_Septoria_leaf_spot',
    33: 'Tomato_Spider_mites ',
    34: 'Tomato_Target_Spot',
    35: 'Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato_mosaic_virus',
    37: 'Tomato__healthy'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'
        
        file = request.files['image']

        if file.filename == '':
            return 'No selected file'

        try:
            img = image.load_img(io.BytesIO(file.read()), target_size=(100, 100))  
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = model.predict(img_array)

            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            return render_template('result.html', predicted_class=predicted_class_label)
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)
