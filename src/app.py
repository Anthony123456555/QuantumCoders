from load_data import load_all_data
from preprocess import preprocess_data
from train_model import train_and_evaluate
from visualize import visualize_data

def main():
    print("🚀 Chargement des données...")
    dataframes = load_all_data("data")

    print("🧹 Prétraitement...")
    merged_df = preprocess_data(dataframes)

    print("📊 Visualisation...")
    visualize_data(merged_df)

    print("🤖 Entraînement du modèle...")
    train_and_evaluate(merged_df)

if __name__ == "__main__":
    main()
