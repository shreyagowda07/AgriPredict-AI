import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================
# AGRIPREDICT AI
# Plant Disease Detection
# ==============================

print("=" * 50)
print("       AGRIPREDICT AI")
print("   PLANT DISEASE DETECTION")
print("=" * 50)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create folders if they don't exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("\nProject folders are ready.")

# Disease classes
CLASS_NAMES = [
    "Healthy",
    "Bacterial Disease",
    "Fungal Disease",
    "Viral Disease"
]

print("\nDisease classes:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")

# Save class names
class_file = os.path.join(MODEL_DIR, "class_names.json")

with open(class_file, "w") as f:
    json.dump(CLASS_NAMES, f, indent=4)

print("\nClass names saved.")

# Create a simple CNN model
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(CLASS_NAMES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model created successfully.")

# Show model structure
model.summary()

# Save initial model
model_path = os.path.join(MODEL_DIR, "plant_disease_model.keras")
model.save(model_path)

print("\nInitial model saved at:")
print(model_path)

print("\n==========================================")
print("Disease detection setup completed!")
print("==========================================")