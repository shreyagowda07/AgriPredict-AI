import joblib
import pandas as pd

MODEL_PATH = "models/crop_yield_model.joblib"

model = joblib.load(MODEL_PATH)


def predict_yield(data):
    input_data = pd.DataFrame([{
        "crop": data["crop"],
        "rainfall_mm": float(data["rainfall"]),
        "temperature_c": float(data["temperature"]),
        "soil_ph": float(data["soil_ph"]),
        "fertilizer_kg": float(data["fertilizer"]),
        "area_hectare": float(data["area"])
    }])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": round(float(prediction), 2),
        "unit": "tons"
    }