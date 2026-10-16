
import json
from pathlib import Path

import tensorflow as tf
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "tomato_disease_model_v2.keras"
CLASS_PATH = PROJECT_DIR / "models" / "tomato_class_names_v2.json"
DATA_DIR = PROJECT_DIR / "data" / "plantvillage" / "raw" / "color"

print("🌱 AGRIPREDICT AI - MODEL TESTING")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

# Find sample images from tomato disease folders
test_images = []

for folder_name in class_names:
    folder = DATA_DIR / folder_name

    if folder.exists():
        images = list(folder.glob("*.JPG"))
        images += list(folder.glob("*.jpg"))
        images += list(folder.glob("*.JPEG"))
        images += list(folder.glob("*.jpeg"))

        if images:
            test_images.append((folder_name, images[0]))

correct = 0
total = 0

for actual_class, image_path in test_images:
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(160, 160)
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    actual_label = actual_class.replace("Tomato___", "")
    predicted_label = predicted_class.replace("Tomato___", "")

    is_correct = actual_class == predicted_class

    if is_correct:
        correct += 1

    total += 1

    result = "CORRECT ✅" if is_correct else "WRONG ❌"

    print("\nImage:", image_path.name)
    print("Actual:", actual_label)
    print("Predicted:", predicted_label)
    print("Confidence:", round(confidence, 2), "%")
    print("Result:", result)

print("\n========== TEST SUMMARY ==========")
print("Images tested:", total)
print("Correct predictions:", correct)

if total:
    print("Sample accuracy:", round(correct / total * 100, 2), "%")

print("\nTesting completed!")
