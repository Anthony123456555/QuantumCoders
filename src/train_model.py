from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate(df):
    numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna()

    if "label" not in numeric_df.columns:
        print("⚠️ Pas de colonne 'label' trouvée, impossible d'entraîner un modèle.")
        return None

    X = numeric_df.drop(columns=["label"])
    y = numeric_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy du modèle : {acc:.2f}")

    return model
