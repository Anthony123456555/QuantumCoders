from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
import pandas as pd
import joblib
from io import BytesIO
from src.data.preprocess import preprocess_tabular
from src.models.predict import load as load_model, predict_from_array

app = FastAPI(title="Exoplanet Detector API")

try:
    load_model()
except Exception:
    pass

class PredictResponse(BaseModel):
    label: str
    prob: float

@app.post('/predict', response_model=PredictResponse)
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    if 'label' not in df.columns:
        df['label'] = 0
    from src.data.preprocess import preprocess_tabular
    X_num, _, _ = preprocess_tabular(df, target_col='label')
    from src.models.predict import _model
    if _model is None:
        return {'label':'model_not_loaded','prob':0.0}
    labels, probs = predict_from_array(X_num)
    avg_prob = float(sum(probs)/len(probs))
    avg_label = 'candidate' if avg_prob > 0.5 else 'not_candidate'
    return {'label': avg_label, 'prob': avg_prob}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
