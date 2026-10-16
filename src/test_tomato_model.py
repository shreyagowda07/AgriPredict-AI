import json
from pathlib import Path
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "tomato_disease_model_trained.keras"
CLASS_PATH = PROJECT_DIR / "models" / "tomato_class_names_trained.json"

print("Loading trained tomato disease model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("\nMODEL LOADED SUCCESSFULLY!")
print("Number of classes:", len(class_names))
print("Model output shape:", model.output_shape)

print("\nTomato categories:")
for name in class_names:
    print("-", name)

print("\nModel is ready for image prediction testing.")
