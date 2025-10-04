import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

from models.train_model import train_with_xgboost
from data.preprocess import preprocess_tabular

MODEL_PATH = "../models/xgb_model.pkl"

st.set_page_config(page_title="Exoplanet Classifier", page_icon=":satellite:", layout="centered")
st.title("🚀 NASA Exoplanet Classifier")
st.write("Ce projet utilise un modèle IA entraîné sur les données exoplanètes de la NASA (Kepler, K2, TESS, etc).")

@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model(MODEL_PATH)

if model is None:
    st.warning("Aucun modèle entraîné n'a été trouvé. Merci d'entraîner d'abord votre modèle avec les données NASA via l'interface CLI.")
    st.stop()
st.header("1️⃣ Données à prédire")
input_mode = st.radio("Comment voulez-vous fournir les données ?", ("Formulaire manuel", "Téléverser un fichier CSV"))

if input_mode == "Formulaire manuel":
    pl_orbper = st.number_input("Période orbitale (jours)", min_value=0.0, value=365.0)
    pl_rade = st.number_input("Rayon planétaire (Terre = 1)", min_value=0.0, value=1.0)
    stellar_teff = st.number_input("Température effective étoile (K)", min_value=0.0, value=5778.0)
    pl_eqt = st.number_input("Température d'équilibre planète (K)", min_value=0.0, value=300.0)

    input_dict = {
        "pl_orbper": pl_orbper,
        "pl_rade": pl_rade,
        "st_teff": stellar_teff,
        "pl_eqt": pl_eqt,
    }
    input_df = pd.DataFrame([input_dict])

else:
    uploaded_file = st.file_uploader("Choisissez un fichier CSV avec vos données", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.warning("Veuillez uploader un fichier CSV pour continuer.")
        st.stop()

st.write("Aperçu de vos données :")
st.write(input_df)
st.header("2️⃣ Prédiction")
if st.button("Prédire"):
    try:
        X, _, _ = preprocess_tabular(input_df, target_col="label")  # La colonne label n'existe pas ici mais c'est ok
        proba = model.predict_proba(X)
        pred = model.predict(X)
        st.success(f"Classe prédite : **{pred[0]}**")
        st.write(f"Probabilités (par classe) : {proba[0]}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

if hasattr(model, "feature_importances_"):
    st.header("3️⃣ Importance des variables")
    importances = model.feature_importances_
    cols = input_df.columns if input_mode == "Téléverser un fichier CSV" else list(input_dict.keys())
    imp_df = pd.DataFrame({"Variable": cols, "Importance": importances})
    st.bar_chart(imp_df.set_index("Variable"))