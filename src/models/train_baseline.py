import argparse
import joblib
from src.data.loaders import load_catalog
from src.data.preprocess import preprocess_tabular
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

def main(input_csv):
    df = load_catalog(input_csv)
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'false_positive':0,'candidate':1,'confirmed':1}).fillna(0).astype(int)
    X, y, preproc = preprocess_tabular(df, target_col='label')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, preds))
    try:
        auc = roc_auc_score(y_test, probs)
        print('ROC AUC:', auc)
    except Exception:
        pass
    joblib.dump(model, 'models/baseline_xgb.pkl')
    joblib.dump(preproc, 'models/preproc.joblib')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input_csv', required=True)
    args = parser.parse_args()
    main(args.input_csv)
