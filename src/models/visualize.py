import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data.loaders import load_all_data
from data.preprocess import preprocess_data
from models.train_model import train_with_xgboost

def visualize_data(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    few_numeric_cols = numeric_cols[:4]
    print(f"Variables displayed in the Pairplot: {few_numeric_cols.tolist()}")
    sns.pairplot(df[few_numeric_cols].sample(min(100, len(df))))

    plt.tight_layout()
    plt.savefig("pairplot.png")
    plt.close()
    print("✅ Pairplot saved to pairplot.png")


def main():
    print("🚀 Loading data...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    dataframes = load_all_data(data_dir)

    if not dataframes or len(dataframes) == 0:
        print("❌ No data files were loaded.")
        return

    print("🧹 Preprocessing...")
    merged_df = preprocess_data(dataframes)

    if merged_df is None or merged_df.empty:
        print("❌ Preprocessing failed or no merged DataFrame is available.")
        return

    print("📊 Visualization...")
    visualize_data(merged_df)

    dispo_col = None
    for col in merged_df.columns:
        if col.lower() in ["koi_disposition", "disposition", "pl_disposition"]:
            dispo_col = col
            break

    if dispo_col:
        print(f"✅ Target column detected: {dispo_col}")
        label_map = {"CONFIRMED": 2, "CANDIDATE": 1, "FALSE POSITIVE": 0}
        merged_df = merged_df[merged_df[dispo_col].isin(label_map.keys())].copy()
        merged_df[dispo_col] = merged_df[dispo_col].map(label_map)
        target_col = dispo_col

    elif "label" in merged_df.columns:
        print("✅ Target column detected: label")
        target_col = "label"
    else:
        print("❌ Could not find a target column (e.g., koi_disposition, disposition, label).")
        print("Available columns:", merged_df.columns.tolist())
        return

    print("🤖 Training model...")
    model_path = os.path.join(os.path.dirname(__file__), "models", "xgb_model.pkl")
    train_with_xgboost(merged_df, target_col=target_col, model_path=model_path)

if __name__ == "__main__":
    main()
