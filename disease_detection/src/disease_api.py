import os
import json
import base64
import io

import numpy as np
import tensorflow as tf

from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tomato_disease_model.keras"
)

CLASS_FILE = os.path.join(
    BASE_DIR,
    "models",
    "tomato_class_names.json"
)

IMG_SIZE = (128, 128)


print("=" * 60)
print("       AGRIPREDICT AI")
print("   TOMATO DISEASE DETECTION API")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(CLASS_FILE, "r") as file:
    CLASS_NAMES = json.load(file)

print("Model loaded successfully.")
print("Number of classes:", len(CLASS_NAMES))


def predict_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][predicted_index]
    )

    disease = CLASS_NAMES[predicted_index]

    disease = disease.replace(
        "Tomato___",
        ""
    )

    disease = disease.replace(
        "_",
        " "
    )

    return disease, confidence


class DiseaseHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):

        response = json.dumps(
            data
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, OPTIONS"
        )

        self.end_headers()

        self.wfile.write(
            response
        )


    def do_OPTIONS(self):

        self.send_json({
            "success": True
        })


    def do_POST(self):

        if self.path != "/disease_predict":

            self.send_json({
                "success": False,
                "error": "Invalid endpoint"
            }, 404)

            return

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(
                content_length
            )

            data = json.loads(
                body.decode("utf-8")
            )

            image_data = data.get(
                "image"
            )

            if not image_data:

                self.send_json({
                    "success": False,
                    "error": "No image provided"
                }, 400)

                return

            if "," in image_data:

                image_data = image_data.split(
                    ",",
                    1
                )[1]

            image_bytes = base64.b64decode(
                image_data
            )

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            disease, confidence = predict_image(
                image
            )

            self.send_json({
                "success": True,
                "disease": disease,
                "confidence": round(
                    confidence * 100,
                    2
                )
            })

        except Exception as error:

            print("ERROR:", error)

            self.send_json({
                "success": False,
                "error": str(error)
            }, 500)


HOST = "127.0.0.1"
PORT = 8001

server = HTTPServer(
    (HOST, PORT),
    DiseaseHandler
)

print("\n" + "=" * 60)
print("Disease API is running")
print("=" * 60)

print("\nURL:")
print(
    "http://127.0.0.1:8001/disease_predict"
)

print("\nWaiting for image predictions...")

server.serve_forever()