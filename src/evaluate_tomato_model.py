
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

base = Path(__file__).resolve().parent.parent
data_dir = base / "data" / "plantvillage" / "raw" / "color"
model_path = base / "models" / "tomato_disease_model_trained.keras"
class_path = base / "models" / "tomato_class_names_trained.json"

with open(class_path, encoding="utf-8") as f:
    classes = json.load(f)

model = tf.keras.models.load_model(model_path)

# Use a separate, reproducible sample from each class.
rng = np.random.default_rng(42)
total = 0
correct = 0

print("\n=== AGRIPREDICT AI: CLASS EVALUATION ===")

for class_name in classes:
    folder = data_dir / class_name
    files = list(folder.glob("*.JPG")) + list(folder.glob("*.jpg"))
    if not files:
        print(f"\n{class_name}: No images found")
        continue

    chosen = rng.choice(files, size=min(20, len(files)), replace=False)
    class_correct = 0

    for path in chosen:
        image = tf.keras.utils.load_img(
            path, target_size=(128, 128), color_mode="rgb"
        )
        arr = tf.keras.utils.img_to_array(image)[None, ...]
        scores = model.predict(arr, verbose=0)[0]
        predicted = classes[int(np.argmax(scores))]

        total += 1
        if predicted == class_name:
            correct += 1
            class_correct += 1

    print(
        f"{class_name}: {class_correct}/{len(chosen)} "
        f"({class_correct / len(chosen) * 100:.1f}%)"
    )

print("\nOverall sample accuracy:")
print(f"{correct}/{total} ({correct / total * 100:.1f}%)" if total else "No images tested")
print("\nNote: This is a diagnostic sample, not a formal test-set score.")
