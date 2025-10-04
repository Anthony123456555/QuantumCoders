import pandas as pd

def preprocess_data(dataframes: dict) -> pd.DataFrame:
    df_list = []
    for name, df in dataframes.items():
        df["source"] = name
        df_list.append(df)

    merged = pd.concat(df_list, ignore_index=True)

    if "period" in merged.columns:
        merged = merged.dropna(subset=["period"])

    return merged
