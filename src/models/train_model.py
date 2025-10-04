import os
import pandas as pd
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def train_with_xgboost(df, target_col="koi_disposition", model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")
    else:
        model_path = os.path.abspath(model_path)

    print(f"📂 Chemin de sauvegarde du modèle : {os.path.abspath(model_path)}")
    if target_col not in df.columns:
        print(f"⚠️ La colonne cible '{target_col}' n'a pas été trouvée.")
        return None

    target_col_final = target_col

    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        df['target_encoded'] = le.fit_transform(df[target_col])
        print(f"✅ Colonne cible '{target_col}' encodée numériquement.")

        target_col_final = 'target_encoded'

        if len(le.classes_) < 2:
            print("ERREUR: Moins de 2 classes uniques dans la colonne cible. L'entraînement est impossible.")
            return None

        df = df.drop(columns=[target_col])

    y = df[target_col_final]
    X = df.drop(columns=[target_col_final])
    X = pd.get_dummies(X, drop_first=True)
    if X.isnull().sum().sum() > 0:
        print("AVERTISSEMENT: Imputation finale des NaN restants dans les features (méthode médiane).")
        X.fillna(X.median(), inplace=True)

    if len(y) == 0:
        print("ERREUR FATALE: Zéro échantillon après la préparation finale. L'entraînement est impossible.")
        return None

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
            # 'use_label_encoder' est retiré pour éviter l'avertissement XGBoost
            "eval_metric": "logloss",
            "random_state": 42
        }
        model = XGBClassifier(**params)
        # Utiliser le score AUC si la tâche est binaire, sinon accuracy
        scoring_metric = "roc_auc" if len(y.unique()) == 2 else "accuracy"
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring_metric)
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    print("✅ Best hyperparameters:", study.best_params)

    best_model = XGBClassifier(**study.best_params,
                               eval_metric="logloss",
                               random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if len(y.unique()) == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"🎯 Accuracy: {acc:.3f}, ROC AUC: {auc:.3f}")
    else:
        print(f"🎯 Accuracy: {acc:.3f}")
        print("NOTE: ROC AUC non calculé (tâche multi-classe).")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"💾 Modèle sauvegardé dans {os.path.abspath(model_path)}")

    return best_model