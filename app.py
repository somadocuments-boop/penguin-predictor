from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Penguin Species Predictor")

# Load model artifacts once, at startup
model = joblib.load('penguin_model.pkl')
scaler = joblib.load('scaler.pkl')
le_species = joblib.load('label_encoder.pkl')

# Serve everything inside the "static" folder (index.html, css, js, images)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define the expected input shape
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


@app.get("/")
def home():
    # Serve the HTML form as the homepage
    return FileResponse("static/index.html")


@app.post("/predict")
def predict(features: PenguinFeatures):
    input_data = np.array([[
        features.bill_length_mm,
        features.bill_depth_mm,
        features.flipper_length_mm,
        features.body_mass_g
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    species = le_species.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    return {
        "predicted_species": species,
        "confidence": round(float(max(probabilities)), 3)
    }