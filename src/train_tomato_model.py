
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "plantvillage" / "raw" / "color"
MODEL_DIR = PROJECT_DIR / "models"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

print("\n====================================")
print(" AGRIPREDICT AI - TOMATO MODEL TRAINING")
print("====================================")

# Load only the 10 tomato categories
tomato_folders = sorted(
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir() and folder.name.startswith("Tomato___")
)

if len(tomato_folders) != 10:
    raise RuntimeError(
        f"Expected 10 tomato folders, found {len(tomato_folders)}"
    )

print(f"\nTomato categories found: {len(tomato_folders)}")

# Create training and validation datasets
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=[folder.name for folder in tomato_folders],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=SEED
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=[folder.name for folder in tomato_folders],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=SEED
)

class_names = train_ds.class_names
print("\nClasses:")
for name in class_names:
    print("-", name)

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Build a CNN classification model
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the model
print("\nTraining started. This may take a while on your computer...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save to new filenames to protect your existing model
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "tomato_disease_model_trained.keras"
classes_path = MODEL_DIR / "tomato_class_names_trained.json"

model.save(model_path)

with open(classes_path, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print("\n====================================")
print(" TRAINING COMPLETED")
print("====================================")
print(f"Model saved: {model_path}")
print(f"Classes saved: {classes_path}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print("\nNote: Validation accuracy is an estimate, not a guarantee.")
