import joblib
import numpy as np

_model = None
_preproc = None

def load(model_path='models/baseline_xgb.pkl', preproc_path='models/preproc.joblib'):
    global _model, _preproc
    _model = joblib.load(model_path)
    _preproc = joblib.load(preproc_path)

def predict_from_array(X_array):
    if _model is None or _preproc is None:
        raise RuntimeError('Model not loaded. Call load() first.')
    probs = _model.predict_proba(X_array)[:,1]
    labels = (_model.predict(X_array)).tolist()
    return labels, probs.tolist()
