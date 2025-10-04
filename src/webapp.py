import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np

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
        st.error(f"Erreur: Le fichier modèle est introuvable à l'emplacement: `{path}`")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

model = load_trained_model(MODEL_FILE_PATH)
CLASS_MAP = {
    0: "CANDIDATE (Probable exoplanète)",
    1: "FALSE POSITIVE (Fausse alerte)",
    2: "CONFIRMED (Planète confirmée)"
}

if model is None:
    st.error(
        f"Aucun modèle entraîné n'a été trouvé à l'emplacement : `{MODEL_FILE_PATH}`. "
        "Merci d'entraîner d'abord votre modèle."
    )
else:
    try:
        MODEL_EXPECTED_FEATURES_ALL = model.get_booster().feature_names
        st.success("✅ Modèle entraîné et structure des features chargée avec succès !")
    except Exception as e:
        st.error(f"Erreur critique: Impossible d'extraire la liste des features du modèle. Le modèle ne peut pas être utilisé. Erreur: {e}")
        MODEL_EXPECTED_FEATURES_ALL = None
        st.stop()
    tab_single, tab_batch = st.tabs(["Classification Unitaire (Manuelle)", "Classification par Lot (CSV)"])
    with tab_single:
        MODEL_EXPECTED_FEATURES = MODEL_EXPECTED_FEATURES_ALL

        st.sidebar.header("Analyse d'une exoplanète")
        with st.sidebar.form("prediction_form"):
            st.subheader("Entrez les paramètres clés:")
            koi_score = st.number_input("KOI Score (Importance du signal)", value=1.000, min_value=0.0, max_value=1.0)
            koi_fpflag_nt = st.selectbox("Non-transit FP Flag (0: OK, 1: Problème de lumière)", options=[0, 1], index=0)
            koi_fpflag_ss = st.selectbox("Stellar Single FP Flag (0: OK, 1: Étoile Binaire suspecte)", options=[0, 1], index=0)
            koi_period = st.number_input("Période Orbitale (jours)", value=11.23, min_value=0.0)
            koi_depth = st.number_input("Profondeur du Transit (ppm)", value=2700.0, min_value=0.0)
            koi_impact = st.number_input("Paramètre d'Impact", value=0.6, min_value=0.0, max_value=1.0)
            kepoi_name = st.text_input("Nom de l'objet KOI (Ex: K00082.01)", value="K00082.01")


            submitted = st.form_submit_button("Classer l'exoplanète")

        if submitted:
            user_input_data = {
                'koi_score': koi_score,
                'koi_fpflag_nt': koi_fpflag_nt,
                'koi_fpflag_ss': koi_fpflag_ss,
                'koi_period': koi_period,
                'koi_depth': koi_depth,
                'koi_impact': koi_impact,
            }
            df_final = pd.DataFrame(0.0, index=[0], columns=MODEL_EXPECTED_FEATURES)
            for col, value in user_input_data.items():
                if col in df_final.columns:
                    df_final.loc[0, col] = value
            one_hot_col_name = f'kepoi_name_{kepoi_name}'

            if one_hot_col_name in df_final.columns:
                df_final.loc[0, one_hot_col_name] = 1.0
            else:
                st.warning(f"Le nom d'objet '{kepoi_name}' n'a pas été vu à l'entraînement. L'encodage One-Hot sera ignoré pour cette feature.")
            df_final = df_final[MODEL_EXPECTED_FEATURES]

            st.caption(f"DataFrame prêt pour la prédiction. Shape: {df_final.shape}")
            try:
                prediction_proba = model.predict_proba(df_final)[0]
                prediction = np.argmax(prediction_proba)

                st.subheader("Résultat de la Classification")

                st.metric(
                    "Disposition Prédite",
                    CLASS_MAP.get(prediction, "Inconnu"),
                    f"Confiance: {prediction_proba[prediction]:.2%}"
                )

                st.write("---")
                st.markdown("**Probabilités pour chaque classe :**")
                for i, prob in enumerate(prediction_proba):
                    st.write(f"- {CLASS_MAP.get(i, f'Classe {i}')}: **{prob:.2%}**")


            except Exception as e:
                st.error(f"Erreur lors de la prédiction unitaire. Erreur: {e}")
    with tab_batch:

        # --- Définition des features pour le mode batch (Exclut l'OHE du nom de l'objet) ---
        MODEL_EXPECTED_FEATURES_BATCH = [
            f for f in MODEL_EXPECTED_FEATURES_ALL if not f.startswith('kepoi_name_')
        ]

        st.header("Classification par Lot (Fichier CSV)")
        st.markdown(
            "⚠️ **Alerte :** L'encodage du nom d'objet (`kepoi_name`) est ignoré dans ce mode pour des raisons de compatibilité. "
            "La classification est basée uniquement sur les features numériques (score, période, etc.)."
        )

        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

        if uploaded_file is not None:

            df_batch = None

            # 1. Tente de lire le fichier avec le délimiteur virgule (par défaut)
            try:
                df_batch = pd.read_csv(uploaded_file, on_bad_lines='skip')
                st.success(f"Fichier chargé (délimiteur : virgule). {len(df_batch)} lignes à classer.")
            except Exception as e:
                try:
                    uploaded_file.seek(0) # Réinitialise le pointeur du fichier
                    df_batch = pd.read_csv(uploaded_file, sep=';', on_bad_lines='skip')
                    st.success(f"Fichier chargé (délimiteur : point-virgule). {len(df_batch)} lignes à classer.")
                except Exception as e_semi:
                    st.error(f"Erreur lors du traitement du fichier. Impossible de lire le CSV : {e_semi}")
                    st.warning("Veuillez vérifier le format de votre CSV. Il pourrait ne pas être formaté correctement (mauvais délimiteur ou problème d'encodage).")
                    # CORRECTION: Remplacement de 'return' par 'st.stop()' pour éviter la 'SyntaxError'.
                    st.stop()

            if df_batch is None or df_batch.empty:
                st.warning("Le fichier CSV est vide ou les lignes étaient illisibles. Veuillez vérifier son contenu.")
                # CORRECTION: Remplacement de 'return' par 'st.stop()'
                st.stop()


            # Conserver une colonne d'index temporaire si le nom de KOI est présent
            if 'kepoi_name' in df_batch.columns:
                df_batch_display = df_batch[['kepoi_name']].copy()
                # On retire 'kepoi_name' car elle est catégorielle et doit être traitée
                df_batch = df_batch.drop(columns=['kepoi_name'], errors='ignore')
            else:
                df_batch_display = pd.DataFrame(index=df_batch.index)
                df_batch_display['Index'] = df_batch.index


            # 1. Créer le DataFrame final en garantissant toutes les features attendues pour le BATCH
            # Remplir par défaut avec des zéros
            df_final_batch = pd.DataFrame(0.0, index=df_batch.index, columns=MODEL_EXPECTED_FEATURES_BATCH)

            # 2. Remplir le DataFrame final avec les colonnes disponibles dans le fichier chargé
            for col in df_batch.columns:
                if col in df_final_batch.columns:
                    # Assurez-vous que les types correspondent à ceux de l'entraînement (float/int)
                    try:
                        df_final_batch[col] = df_batch[col].astype(df_final_batch[col].dtype)
                    except Exception:
                        df_final_batch[col] = df_batch[col] # Utiliser le type par défaut si la conversion échoue

            # 3. Assurer l'ordre exact des colonnes
            df_final_batch = df_final_batch[MODEL_EXPECTED_FEATURES_BATCH]

            # ----------------------------------------------------------------------
            # PRÉDICTION PAR LOT
            # ----------------------------------------------------------------------
            st.subheader("Démarrage de la Classification...")
            with st.spinner('Classification en cours, veuillez patienter...'):
                # Effectuer la prédiction
                prediction_proba_batch = model.predict_proba(df_final_batch)
                prediction_batch = np.argmax(prediction_proba_batch, axis=1)

                # Ajouter les résultats au DataFrame d'affichage
                df_batch_display['Prediction Class'] = prediction_batch
                df_batch_display['Disposition'] = df_batch_display['Prediction Class'].map(CLASS_MAP)

                # Ajouter la probabilité de la classe prédite
                df_batch_display['Confiance (%)'] = [
                    f"{prediction_proba_batch[i, pred]:.2%}"
                    for i, pred in enumerate(prediction_batch)
                ]

            st.success("Classification terminée !")
            st.markdown("### Aperçu des résultats")
            st.dataframe(df_batch_display)

            csv_output = df_batch_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les résultats complets (CSV)",
                data=csv_output,
                file_name='exoplanet_classification_results.csv',
                mime='text/csv',
            )


st.markdown("---")
