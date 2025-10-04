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
    imputer = SimpleImputer(strategy='median')
    X_num = imputer.fit_transform(X[num_cols])
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    return X_num, y, {'imputer': imputer, 'scaler': scaler, 'num_cols': list(num_cols)}

def preprocess_data(df_list, target_col="label"):
    if not df_list:
        raise ValueError("Aucun DataFrame à concaténer. Vérifie ta liste de fichiers ou ton chargement de données.")
    merged = pd.concat(df_list, ignore_index=True)
    return merged