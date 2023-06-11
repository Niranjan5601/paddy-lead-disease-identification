from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
import numpy as np 
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/', methods=['GET'])
def upload_file():
    
   
    return render_template('intro.html')



@app.route('/upload', methods=['POST','GET'])
def upload():
    return render_template('upload.html')


@app.route('/next', methods=['POST','GET'])
def next():
    return render_template('next.html')


@app.route('/predict', methods=['POST'])
def predict():

    for f in os.listdir('static/images'):
        if f.endswith(".jpg"):
             os.remove(os.path.join('static/images', f))
        elif f.endswith(".png"):
            os.remove(os.path.join('static/images', f))
        elif f.endswith(".jpeg"):
            os.remove(os.path.join('static/images', f))
    
    imagefile= request.files['file']
    img = Image.open(imagefile)

   
    img.save(os.path.join('uploads/', imagefile.filename))

   
    img.save(os.path.join('static/images/', imagefile.filename))
    image_path="uploads/"+ imagefile.filename
    
    static_img = "static/images/"+imagefile.filename
    
   
    
   
    value= {
        0: 'Bacterial leaf blight', 1: 'Bacterial leaf streak', 2: 'Bacterial panicle blight', 3: 'Blast', 4: 'Brown spot', 5: 'Dead heart', 6: 'Downy mildew', 7: 'Hispa', 8: 'Normal', 9: 'Tungro'
    }
    
    test_data = ImageDataGenerator(rescale=1.0/255).flow_from_directory(    
    directory="uploads/",
    target_size=(256, 256),
    batch_size=16,
    classes=['.'],
    shuffle=False,)

    load_model = tf.keras.models.load_model('my_model.json')
    prediction = np.argmax(load_model.predict(test_data),axis=1)
    output=  value[prediction[0]]
    os.remove(image_path)
    
    return render_template('next.html',user_image=static_img,prediction = output)

if __name__ == '__main__':
    app.run(debug=True)


