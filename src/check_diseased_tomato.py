
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

base = Path(__file__).resolve().parent.parent
data = base / "data" / "plantvillage" / "raw" / "color"
model = tf.keras.models.load_model(
    base / "models" / "tomato_disease_model_trained.keras"
)

with open(base / "models" / "tomato_class_names_trained.json", encoding="utf-8") as f:
    classes = json.load(f)

folder = data / "Tomato___Early_blight"
images = sorted(folder.glob("*.JPG"))[:10]
if not images:
    images = sorted(folder.glob("*.jpg"))[:10]

correct = 0
print("\nEXPECTED CLASS: Tomato___Early_blight")

for path in images:
    image = Image.open(path).convert("RGB").resize((128, 128))
    arr = np.asarray(image, dtype=np.float32)[None, ...]
    scores = model.predict(arr, verbose=0)[0]
    index = int(np.argmax(scores))
    predicted = classes[index]

    if predicted == "Tomato___Early_blight":
        correct += 1

    print(f"{path.name}: {predicted} ({scores[index] * 100:.1f}%)")

print(f"\nCorrect Early Blight predictions: {correct}/{len(images)}")
