import sys
import os
# Add the current directory to the system path to allow local imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Assuming these modules exist in your project structure (e.g., in a 'data' and 'models' folder)
from data.loaders import load_all_data
from data.preprocess import preprocess_data
from models.visualize import visualize_data
from models.train_model import train_with_xgboost

def main():
    print("🚀 Loading data...")
    # Navigate up one directory (to the project root, typically) and into the 'data' folder
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

    # --- Target Column Detection ---
    dispo_col = None
    # Search for the target column name (case-insensitive)
    for col in merged_df.columns:
        if col.lower() in ["koi_disposition", "disposition", "pl_disposition"]:
            dispo_col = col
            break

    if dispo_col:
        print(f"✅ Target column detected: {dispo_col}")

        # Mapping the disposition strings to numerical labels for the XGBoost model
        label_map = {"CONFIRMED": 2, "CANDIDATE": 1, "FALSE POSITIVE": 0}

        # Filter out rows with unmapped dispositions and create a copy to avoid warnings
        merged_df = merged_df[merged_df[dispo_col].isin(label_map.keys())].copy()

        # Apply the mapping
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
    # Define the path where the trained model will be saved
    model_path = os.path.join(os.path.dirname(__file__), "models", "xgb_model.pkl")
    train_with_xgboost(merged_df, target_col=target_col, model_path=model_path)

if __name__ == "__main__":
    main()
