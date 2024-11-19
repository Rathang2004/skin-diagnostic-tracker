from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model("skin_diagnostic_model.h5")

app = Flask(__name__)
CORS(app)

classes = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}

@app.route("/diagnose", methods=["POST"])
def diagnose():
    img_file = request.files['image'].read()
    img = Image.open(io.BytesIO(img_file))
    img = img.convert("RGB")
    img = img.resize((256, 256)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    diagnosis = model.predict(img_array)
    diagnosis_class = int(np.argmax(diagnosis, axis=1))
    diagnosis_label = classes.get(diagnosis_class, "Unknown")


    return jsonify({
        "Diagnosis Class Number": int(diagnosis_class),
        "Diagnosis Class": diagnosis_label
         
     }), 201
 

if __name__ == "__main__":
    app.run(debug=True)