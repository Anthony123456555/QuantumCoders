import pandas as pd
import os

def load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"unavailable file : {file_path}")
    return pd.read_csv(file_path)

def load_all_data(data_dir: str = "data"):
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    dataframes = {f: load_csv(os.path.join(data_dir, f)) for f in files}
    return dataframes
