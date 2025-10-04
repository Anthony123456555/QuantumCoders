import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_tabular(df, target_col='label'):
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in dataframe columns")
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(X) == 0 or len(num_cols) == 0:
        return np.array([]), y, {'imputer': None, 'scaler': None, 'num_cols': []}

    imputer = SimpleImputer(strategy='median')
    X_num = imputer.fit_transform(X[num_cols])
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    return X_num, y, {'imputer': imputer, 'scaler': scaler, 'num_cols': list(num_cols)}

def preprocess_data(df_list, target_col="koi_disposition"):
    if not df_list:
        raise ValueError("Aucun DataFrame à concaténer. Vérifie ta liste de fichiers ou ton chargement de données.")
    merged_df = pd.concat(df_list, ignore_index=True)

    print(f"DEBUG: Taille du DataFrame fusionné AVANT nettoyage: {len(merged_df)}")
    rows_before_target_dropna = len(merged_df)
    merged_df.dropna(subset=[target_col], inplace=True)
    rows_after_target_dropna = len(merged_df)

    print(f"DEBUG: Lignes supprimées (cible manquante): {rows_before_target_dropna - rows_after_target_dropna}")

    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()

    cols_to_impute = [col for col in numeric_cols if col != target_col]

    for col in cols_to_impute:
        median_value = merged_df[col].median()
        if pd.isna(median_value):
            merged_df[col] = merged_df[col].fillna(0)
        else:
            merged_df[col] = merged_df[col].fillna(median_value)

    if len(merged_df) == 0:
        raise ValueError("Le DataFrame est vide après la suppression des NaN sur la colonne cible. Vérifiez vos données.")

    print(f"DEBUG: Taille du DataFrame fusionné APRÈS nettoyage: {len(merged_df)}")

    return merged_df