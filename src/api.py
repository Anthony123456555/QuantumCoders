from fastapi import FastAPI, UploadFile, File
import pandas as pd
from load_data import load_all_data
from preprocess import preprocess_data
from src.models.train_model import train_and_evaluate

app = FastAPI(title="Exoplanet Detector API")

print("🚀 Initializing API and loading data...")
dataframes = load_all_data("data")
merged_df = preprocess_data(dataframes)
model = train_and_evaluate(merged_df)

@app.get("/")
def root():
    return {"message": "Welcome to the Exoplanet Detector API 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return {"error": f"Error reading CSV file: {e}"}

    if model is None:
        return {"error": "Model not trained or loaded."}
    numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna()

    if numeric_df.empty:
        return {"error": "No usable numeric columns found after processing."}

    predictions = model.predict(numeric_df)
    return {"predictions": predictions.tolist()}