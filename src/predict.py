import json


# Load the trained model
with open("models/crop_yield_model.json", "r") as file:
    model = json.load(file)


# Get model parameters
weights = model["weights"]
bias = model["bias"]
means = model["means"]
stds = model["stds"]


# New farm information
rainfall = 1200
temperature = 28
soil_ph = 6.5
fertilizer = 100
area = 2.0


# Store input values
features = [
    rainfall,
    temperature,
    soil_ph,
    fertilizer,
    area
]


# Normalize input
normalized_features = []

for i in range(5):

    normalized_value = (
        features[i] - means[i]
    ) / stds[i]

    normalized_features.append(normalized_value)


# Predict yield
prediction = bias

for i in range(5):
    prediction += (
        weights[i] * normalized_features[i]
    )


# Display result
print()
print("==============================")
print("     CROP YIELD PREDICTION")
print("==============================")

print()
print("Crop: Rice")
print("Rainfall:", rainfall, "mm")
print("Temperature:", temperature, "°C")
print("Soil pH:", soil_ph)
print("Fertilizer:", fertilizer, "kg")
print("Area:", area, "hectares")

print()
print("Predicted Yield:", round(prediction, 2), "tons")