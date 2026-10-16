import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
print("        AGRIPREDICT AI")
print("    TOMATO DISEASE PREDICTION")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = json.load(f)

print("Model loaded successfully.")
print("Number of classes:", len(CLASS_NAMES))


def predict_tomato_disease(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))

    confidence = float(predictions[0][predicted_index])

    disease = CLASS_NAMES[predicted_index]

    disease = disease.replace("Tomato___", "")
    disease = disease.replace("_", " ")

    return disease, confidence


if __name__ == "__main__":

    image_path = input("\nEnter image path: ").strip()

    if not os.path.exists(image_path):

        print("\nERROR: Image file not found.")

    else:

        disease, confidence = predict_tomato_disease(image_path)

        print("\n" + "=" * 60)
        print("        PREDICTION RESULT")
        print("=" * 60)

        print("\nDisease:")
        print(disease)

        print("\nConfidence:")
        print(f"{confidence * 100:.2f}%")

        print("\n" + "=" * 60)