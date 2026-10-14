import tensorflow_datasets as tfds

print("Loading PlantVillage dataset...")

dataset, info = tfds.load(
    "plant_village",
    split="train",
    with_info=True
)

print("\nDataset loaded successfully!")
print(info)