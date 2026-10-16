
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

base = Path(__file__).resolve().parent.parent
data_dir = base / "data" / "plantvillage" / "raw" / "color"
model = tf.keras.models.load_model(
    base / "models" / "tomato_disease_model_trained.keras"
)

with open(base / "models" / "tomato_class_names_trained.json", encoding="utf-8") as f:
    classes = json.load(f)

folder = data_dir / "Tomato___healthy"
images = list(folder.glob("*.JPG"))[:10]

if not images:
    images = list(folder.glob("*.jpg"))[:10]

print("\nEXPECTED CLASS: Tomato___healthy")
correct = 0

for path in images:
    image = Image.open(path).convert("RGB").resize((128, 128))
    arr = np.asarray(image, dtype=np.float32)[None, ...]
    scores = model.predict(arr, verbose=0)[0]
    index = int(np.argmax(scores))
    prediction = classes[index]

    if prediction == "Tomato___healthy":
        correct += 1

    print(f"{path.name}: {prediction} ({scores[index] * 100:.1f}%)")

print(f"\nCorrect healthy predictions: {correct}/{len(images)}")
