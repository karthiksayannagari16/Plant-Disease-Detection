from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import json
import os
import gdown

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("model", exist_ok=True)

# Rebuild architecture
model = models.Sequential()

model.add(
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(224, 224, 3)
    )
)

model.add(layers.MaxPooling2D(2, 2))

model.add(
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)

model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())

model.add(
    layers.Dense(
        256,
        activation='relu'
    )
)

model.add(
    layers.Dense(
        38,
        activation='softmax'
    )
)

# Model path
MODEL_PATH = "model/plant_weights.weights.h5"

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):

    print("Model not found. Downloading...")

    file_id = "1800-TPvwtmnk4pY6EcM9-UbHaceausSN"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, MODEL_PATH, quiet=False)

    print("Model downloaded successfully.")

# Load model weights
model.load_weights(MODEL_PATH)

# Load class labels
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)


# Preprocess image
def preprocess_image(path):

    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img)

    img = img.astype('float32') / 255.0

    img = np.expand_dims(img, axis=0)

    return img


@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None
    image_path = None

    if request.method == 'POST':

        file = request.files.get('image')

        if file and file.filename != "":

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            image = preprocess_image(filepath)

            pred = model.predict(image)

            idx = np.argmax(pred)

            prediction = class_indices[str(idx)]

            image_path = filepath

    return render_template(
        'index.html',
        prediction=prediction,
        image_path=image_path
    )


if __name__ == '__main__':
    app.run(debug=True)