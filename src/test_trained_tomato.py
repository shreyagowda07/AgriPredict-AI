
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

base = Path(__file__).resolve().parent.parent
model = tf.keras.models.load_model(
    base / "models" / "tomato_disease_model_trained.keras"
)

with open(base / "models" / "tomato_class_names_trained.json", encoding="utf-8") as f:
    classes = json.load(f)

image_path = (
    base / "data" / "plantvillage" / "raw" / "color"
    / "Tomato___healthy"
    / "fd09836b-96be-498a-a720-e17a54522de7___RS_HL 9687.JPG"
)

image = Image.open(image_path).convert("RGB").resize((128, 128))
arr = np.asarray(image, dtype=np.float32)[None, ...]

predictions = model.predict(arr, verbose=0)[0]
index = int(np.argmax(predictions))

print("\n=== AGRIPREDICT AI TEST ===")
print("Image:", image_path.name)
print("Predicted class:", classes[index])
print(f"Confidence: {predictions[index] * 100:.2f}%")
