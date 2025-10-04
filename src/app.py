import sys
import os

# Ajoute src/ au PYTHONPATH pour les imports relatifs
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.loaders import load_all_data
from data.preprocess import preprocess_data
from models.visualize import visualize_data
from models.train_model import train_with_xgboost

def main():
    print("🚀 Chargement des données...")
    # Utilise le chemin absolu vers le dossier data, adapté à partir de src/
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    dataframes = load_all_data(data_dir)

    if not dataframes or len(dataframes) == 0:
        print("❌ Aucun fichier de données n'a été chargé.")
        return

    print("🧹 Prétraitement...")
    merged_df = preprocess_data(dataframes)

    if merged_df is None or merged_df.empty:
        print("❌ Le prétraitement a échoué ou aucun DataFrame fusionné n'est disponible.")
        return

    print("📊 Visualisation...")
    visualize_data(merged_df)

    # --- Détection automatique de la colonne cible ---
    # Adapter ici selon tes données : pour Kepler, c'est souvent "koi_disposition"
    dispo_col = None
    for col in merged_df.columns:
        if col.lower() in ["koi_disposition", "disposition", "pl_disposition"]:
            dispo_col = col
            break

    if dispo_col:
        print(f"✅ Colonne cible détectée : {dispo_col}")
        label_map = {"CONFIRMED": 2, "CANDIDATE": 1, "FALSE POSITIVE": 0}
        merged_df = merged_df[merged_df[dispo_col].isin(label_map.keys())].copy()
        merged_df[dispo_col] = merged_df[dispo_col].map(label_map)
        target_col = dispo_col
    elif "label" in merged_df.columns:
        print("✅ Colonne cible détectée : label")
        target_col = "label"
    else:
        print("❌ Impossible de trouver une colonne cible (ex: koi_disposition, disposition, label).")
        print("Colonnes disponibles :", merged_df.columns.tolist())
        return

    print("🤖 Entraînement du modèle...")
    # Chemin de sauvegarde : src/models/xgb_model.pkl
    model_path = os.path.join(os.path.dirname(__file__), "models", "xgb_model.pkl")
    train_with_xgboost(merged_df, target_col=target_col, model_path=model_path)

if __name__ == "__main__":
    main()