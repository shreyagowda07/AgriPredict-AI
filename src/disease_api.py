import os
import json
import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer


# Project configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "tomato_disease_model_v2.keras"
)

CLASS_FILE = os.path.join(
    BASE_DIR, "models", "tomato_class_names_v2.json"
)

IMG_SIZE = (160, 160)


# Load the improved model
print("=" * 55)
print("          AGRIPREDICT AI")
print("   TOMATO DISEASE DETECTION API")
print("=" * 55)

print("Loading improved model...")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.isfile(CLASS_FILE):
    raise FileNotFoundError(f"Class file not found: {CLASS_FILE}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r", encoding="utf-8") as file:
    CLASS_NAMES = json.load(file)

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError("Model output and class names do not match.")

print("Model loaded successfully!")
print("Number of classes:", len(CLASS_NAMES))


# Predict a tomato leaf image
def predict_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    index = int(np.argmax(predictions))
    confidence = float(predictions[index])

    disease = CLASS_NAMES[index]
    disease = disease.replace("Tomato___", "").replace("_", " ")

    return disease, confidence


# API request handler
class DiseaseHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(response)

    def do_OPTIONS(self):
        self.send_json({"success": True})

    def do_GET(self):
        if self.path in ("/", "/health"):
            self.send_json({
                "success": True,
                "message": "AgriPredict AI API is running",
                "model": "MobileNetV2",
                "classes": len(CLASS_NAMES)
            })
        else:
            self.send_json({"success": False, "error": "Endpoint not found"}, 404)

    def do_POST(self):
        if self.path != "/disease_predict":
            self.send_json({"success": False, "error": "Invalid endpoint"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))

            if length <= 0 or length > 15 * 1024 * 1024:
                self.send_json({
                    "success": False,
                    "error": "Invalid request size"
                }, 400)
                return

            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            image_data = data.get("image")

            if not isinstance(image_data, str) or not image_data:
                self.send_json({
                    "success": False,
                    "error": "Please provide an image"
                }, 400)
                return

            if "," in image_data:
                image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data, validate=True)

            with Image.open(io.BytesIO(image_bytes)) as uploaded:
                uploaded.load()
                image = uploaded.convert("RGB")

            disease, confidence = predict_image(image)

            self.send_json({
                "success": True,
                "disease": disease,
                "confidence": round(confidence * 100, 2),
                "message": "Prediction completed"
            })

            print(f"Prediction: {disease} ({confidence * 100:.2f}%)")

        except Exception as error:
            print("Prediction error:", error)
            self.send_json({
                "success": False,
                "error": "Could not process this image. Please try another."
            }, 400)


# Start the local server
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8001"))

server = HTTPServer((HOST, PORT), DiseaseHandler)

print()
print("API is running!")
print("Health check: http://127.0.0.1:8001/health")
print("Prediction endpoint: http://127.0.0.1:8001/disease_predict")
print("Press Ctrl+C to stop the server.")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nAPI server stopped.")
finally:
    server.server_close()