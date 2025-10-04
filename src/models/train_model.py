import os
import pandas as pd
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

def train_with_xgboost(df, target_col="label", model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")
    else:
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)

    print(f"📂 Chemin de sauvegarde du modèle : {os.path.abspath(model_path)}")

    numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna()
    if target_col not in numeric_df.columns:
        print(f"⚠️ Pas de colonne '{target_col}' trouvée dans les colonnes {numeric_df.columns.tolist()}.")
        return None

    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    print("✅ Best hyperparameters:", study.best_params)

    best_model = XGBClassifier(**study.best_params,
                               use_label_encoder=False,
                               eval_metric="logloss",
                               random_state=42)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"🎯 Accuracy: {acc:.3f}, ROC AUC: {auc:.3f}")

    # Création du dossier si besoin
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"💾 Modèle sauvegardé dans {os.path.abspath(model_path)}")

    return best_model