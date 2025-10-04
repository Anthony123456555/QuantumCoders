from fastapi import FastAPI, UploadFile, File
import pandas as pd
from load_data import load_all_data
from preprocess import preprocess_data
from src.models.train_model import train_and_evaluate

app = FastAPI(title="Exoplanet Detector API")

print("🚀 Initialisation API et chargement des données...")
dataframes = load_all_data("data")
merged_df = preprocess_data(dataframes)
model = train_and_evaluate(merged_df)

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API Exoplanet Detector 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if model is None:
        return {"error": "Modèle non entraîné."}

    numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna()
    if numeric_df.empty:
        return {"error": "Pas de colonnes numériques exploitables."}

    predictions = model.predict(numeric_df)
    return {"predictions": predictions.tolist()}
