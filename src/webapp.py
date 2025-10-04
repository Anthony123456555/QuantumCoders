import streamlit as st
import os
import joblib
import pandas as pd
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(CURRENT_DIR, "models", "xgb_model.pkl")


st.set_page_config(
    page_title="NASA Exoplanet Classifier",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🚀 NASA Exoplanet Classifier")
st.markdown("Ce projet utilise un modèle IA entraîné sur les données exoplanètes de la NASA (Kepler, K2, TESS, etc).")


@st.cache_resource
def load_trained_model(path):
    """Tente de charger le modèle entraîné."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

model = load_trained_model(MODEL_FILE_PATH)

if model is None:
    st.error(
        f"Aucun modèle entraîné n'a été trouvé à l'emplacement : `{MODEL_FILE_PATH}`. "
        "Merci d'entraîner d'abord votre modèle via l'interface CLI (`python src/app.py`)."
    )
    st.warning("Aucun modèle entraîné n'a été trouvé. Merci d'entraîner d'abord votre modèle avec les données NASA via l'interface CLI.")
else:

    st.success("✅ Modèle entraîné chargé avec succès !")

    st.sidebar.header("Analyse d'une exoplanète")

    with st.sidebar.form("prediction_form"):
        st.subheader("Entrez les paramètres clés:")

        koi_score = st.number_input("Score KOI", value=0.5, min_value=0.0, max_value=1.0)
        koi_fpflag_nt = st.selectbox("Non-transit FP Flag", options=[0, 1], index=0)
        koi_fpflag_ss = st.selectbox("Stellar Single FP Flag", options=[0, 1], index=0)

        submitted = st.form_submit_button("Classer l'exoplanète")

    if submitted:
        input_data = pd.DataFrame({
            'koi_score': [koi_score],
            'koi_fpflag_nt': [koi_fpflag_nt],
            'koi_fpflag_ss': [koi_fpflag_ss]
        })

        try:
            prediction = model.predict(input_data)[0]
            st.subheader("Résultat de la Classification")
            if prediction == 0:
                st.metric("Disposition", "CANDIDATE", "Probable exoplanète")
            elif prediction == 1:
                st.metric("Disposition", "FALSE POSITIVE", "- Fausse alerte")
            else:
                st.metric("Disposition", "CONFIRMED", "Planète confirmée")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction. Vérifiez les features d'entrée. Erreur: {e}")

st.markdown("---")