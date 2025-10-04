import numpy as np
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
