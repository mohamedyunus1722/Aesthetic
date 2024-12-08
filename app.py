from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os

app = Flask(__name__)

def load_models_():
  

    ava_mobilenet_model = tf.keras.models.load_model('Models/AVA_mobilenet_model.keras')
    ava_xception_model = tf.keras.models.load_model('Models/AVA_Xception_50_model.keras')
    ava_densenet_model = tf.keras.models.load_model('Models/AVA_densenet_model.keras')
    ava_efficient_model = tf.keras.models.load_model('Models/AVA_efficientnet_model.keras')
    ava_resnet_model = tf.keras.models.load_model('Models/AVA_resnet_50_model.keras')
    ava_vgg_model = tf.keras.models.load_model('Models/AVA_Xception_model.keras')
    ava_inceptionresnetV2_model = tf.keras.models.load_model('Models/AVA_inception_resnet_v2_model.keras')

    return ava_mobilenet_model, ava_xception_model, ava_densenet_model, ava_efficient_model, ava_resnet_model, ava_vgg_model, ava_inceptionresnetV2_model

ava_densenet_model, ava_efficientnet_model, ava_vgg_model,ava_resnet_model, ava_xception_model, ava_mobilenet_model,ava_model= load_models_()

models_ava = [ava_densenet_model, ava_efficientnet_model, ava_vgg_model,ava_resnet_model, ava_xception_model, ava_mobilenet_model,ava_model]

def ensemble_predict(models, dataset):
    all_predictions = []


    for model in models:
        predictions = model.predict(dataset)
        all_predictions.append(predictions.flatten())  


    ensemble_predictions = np.mean(np.stack(all_predictions, axis=0), axis=0)
    return ensemble_predictions

def get_prediction(image_path, models_ava):

    image = Image.open(image_path)
    

    if image.mode != 'RGB':
        image = image.convert('RGB')
    

    image = image.resize((224, 224))
    
    
    image = np.array(image) / 255.0 
    

    print(f"Image shape before reshaping: {image.shape}")  
    
    
    image = image.reshape(1, 224, 224, 3)
    
   
    ava_predictions = ensemble_predict(models_ava, image)
    
    return ava_predictions
    

@app.route('/')
def index():
    return render_template('index.html')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if file:
        
        img = Image.open(file.stream)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        img.save(img_path)
   
        ava_preds = get_prediction(img_path, models_ava)
        
        return jsonify({
            'image_url': f'/static/uploads/{file.filename}',
            'predictions': (ava_preds * 2).tolist() 
        })
if __name__ == "__main__":
    app.run(debug=True)