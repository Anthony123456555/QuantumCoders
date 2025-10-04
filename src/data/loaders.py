import os
import pandas as pd

def load_all_data(data_dir):
    dataframes = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath, comment='#')
                if df.empty or df.shape[1] < 2:
                    with open(filepath, encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if not line.startswith('#') and line.strip():
                                print(f"Première ligne non-commentaire {i+1}: {line.strip()}")
                                break
                    df = pd.read_csv(filepath, skiprows=55)
                dataframes.append(df)
                print(f"✅ Chargé: {filename}, {df.shape}")
            except Exception as e:
                print(f"Erreur lors du chargement de {filename}: {e}")
    return dataframes