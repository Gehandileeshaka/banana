from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

app = Flask(__name__)
model = tf.keras.models.load_model('PanamaIdentificator.h5')


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = Image.open(image_path)
    image = image.resize((400, 400))  # Resize the image
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image / 255.0  # Normalize the image


    yhat = model.predict(image)
    prediction_confidence = yhat[0][0]

    # Customize based on your classes and threshold
    if prediction_confidence > 0.5:  # Adjust this threshold as needed
        classification = 'panama'
        confidence = f'{prediction_confidence * 100:.2f}%'
    elif prediction_confidence < 0.2:  # Adjust this threshold as needed
        classification = 'NotPanama'
        confidence = f'{(1 - prediction_confidence) * 100:.2f}%'
    else:
        classification = 'No match'
        confidence = 'N/A'

    return render_template('index.html', prediction=classification, confidence=confidence)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
