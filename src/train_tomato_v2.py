
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "plantvillage" / "raw" / "color"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42

print("🌱 AGRIPREDICT AI - IMPROVED TOMATO MODEL")
print("Checking tomato dataset...")

# Find tomato disease folders
tomato_folders = sorted(
    folder.name
    for folder in DATA_DIR.iterdir()
    if folder.is_dir() and folder.name.startswith("Tomato___")
)

if len(tomato_folders) != 10:
    raise ValueError(
        f"Expected 10 tomato classes, but found {len(tomato_folders)}. "
        f"Check dataset path: {DATA_DIR}"
    )

print("Tomato classes:", len(tomato_folders))
for name in tomato_folders:
    print("-", name)

# Load training and validation data
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=tomato_folders,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=tomato_folders,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_data.class_names
print("Training and validation datasets loaded.")

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)

# Data augmentation helps the model learn variations
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1)
], name="data_augmentation")

# Load pretrained MobileNetV2
print("Loading MobileNetV2 pretrained model...")
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Build the new classifier
inputs = layers.Input(shape=(160, 160, 3))
x = augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save the best version of this model
model_path = MODEL_DIR / "tomato_disease_model_v2.keras"
checkpoint = ModelCheckpoint(
    str(model_path),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

print("\n🚀 Starting training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# Save class names separately
class_file = MODEL_DIR / "tomato_class_names_v2.json"
with open(class_file, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print("\n✅ TRAINING COMPLETED!")
print("Model saved to:", model_path)
print("Class names saved to:", class_file)
print("Final training accuracy:",
      round(history.history["accuracy"][-1] * 100, 2), "%")
print("Final validation accuracy:",
      round(history.history["val_accuracy"][-1] * 100, 2), "%")
print("\nNext, we will test the improved model.")