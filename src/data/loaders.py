import pandas as pd
from pathlib import Path

def load_catalog(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(p)
    return df
