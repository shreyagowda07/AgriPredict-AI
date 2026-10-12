import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# CROP YIELD MODEL TRAINING
# ==========================================

print()
print("========================================")
print("       CROP YIELD ML TRAINING")
print("========================================")
print()


# Load dataset
data_path = "data/raw/crop_yield.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Training samples:", len(df))
print()


# Features
X = df[
    [
        "crop",
        "rainfall_mm",
        "temperature_c",
        "soil_ph",
        "fertilizer_kg",
        "area_hectare"
    ]
]

# Target
y = df["yield_ton"]


# Categorical and numerical features
categorical_features = ["crop"]

numerical_features = [
    "rainfall_mm",
    "temperature_c",
    "soil_ph",
    "fertilizer_kg",
    "area_hectare"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "crop",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numbers",
            StandardScaler(),
            numerical_features
        )
    ]
)


# Linear Regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# Train
print("Training model...")

model.fit(X, y)

print("Training completed successfully!")
print()


# Evaluate
predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)


print("========================================")
print("          MODEL EVALUATION")
print("========================================")
print()

print("Model: Linear Regression")
print("Training samples:", len(df))
print("Mean Absolute Error:", round(mae, 4))
print("R2 Score:", round(r2, 4))

print()


# Save model
model_path = "models/crop_yield_model.joblib"

joblib.dump(model, model_path)

print("Model saved successfully!")
print("Location:", model_path)
print()
