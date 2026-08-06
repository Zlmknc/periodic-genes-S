import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(PROCESSED / "split_train.csv")
feature_cols = [c for c in train.columns if c not in ["ORF", "label"]]
X_train, y_train = train[feature_cols], train["label"].values

pos_ratio = y_train.mean()
scale_pos_weight = (1 - pos_ratio) / pos_ratio

N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def make_pipeline(model, needs_scaling):
    if needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("clf", model)])
    return Pipeline([("clf", model)])

models = {
    "Random Forest": (RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=2,
                                              class_weight="balanced_subsample", random_state=42, n_jobs=-1), False),
    "Extra Trees": (ExtraTreesClassifier(n_estimators=500, max_depth=12,
                                          class_weight="balanced_subsample", random_state=42, n_jobs=-1), False),
    "Gradient Boosting": (GradientBoostingClassifier(n_estimators=300, max_depth=4, random_state=42), False),
    "LightGBM": (lgb.LGBMClassifier(n_estimators=300, scale_pos_weight=scale_pos_weight,
                                     random_state=42, verbose=-1), False),
    "XGBoost": (xgb.XGBClassifier(n_estimators=300, scale_pos_weight=scale_pos_weight,
                                   random_state=42, eval_metric="logloss"), False),
    "CatBoost": (CatBoostClassifier(iterations=300, class_weights=[1, scale_pos_weight],
                                 random_state=42, verbose=0, allow_writing_files=False), False),
    "Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42), True),
    "SVM (RBF)": (SVC(class_weight="balanced", probability=True, random_state=42), True),
    "KNN": (KNeighborsClassifier(n_neighbors=15), True),
    "Naive Bayes": (GaussianNB(), True),
    "MLP": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42), True),
}

results = []
oof_predictions = {}   # sonraki adımda eşik optimizasyonu için saklıyoruz

for name, (model, needs_scaling) in models.items():
    
    if name == "CatBoost":
        # CatBoost, sklearn clone() ile uyumsuz (class_weights parametresi
        # nedeniyle) -> cross_val_predict yerine manuel fold döngüsü.
        oof_proba = np.zeros(len(y_train))
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr = y_train[tr_idx]

            fold_pos_ratio = y_tr.mean()
            fold_spw = (1 - fold_pos_ratio) / fold_pos_ratio

            fold_model = CatBoostClassifier(
                iterations=300, class_weights=[1, fold_spw],
                random_state=42, verbose=0, allow_writing_files=False
            )
            fold_model.fit(X_tr, y_tr)
            oof_proba[va_idx] = fold_model.predict_proba(X_va)[:, 1]
            print(f"  CatBoost fold {fold_idx}/{N_FOLDS} tamamlandı")
    else:
        pipe = make_pipeline(model, needs_scaling)
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv,
                                       method="predict_proba", n_jobs=-1)[:, 1]

    oof_pred = (oof_proba >= 0.5).astype(int)

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_train, oof_pred),
        "Precision_M": precision_score(y_train, oof_pred, average="macro"),
        "Recall_M": recall_score(y_train, oof_pred, average="macro"),
        "F1_M": f1_score(y_train, oof_pred, average="macro"),
        "Kappa": cohen_kappa_score(y_train, oof_pred),
        "MCC": matthews_corrcoef(y_train, oof_pred),
        "AUC": roc_auc_score(y_train, oof_proba),
    }
    results.append(row)
    oof_predictions[name] = oof_proba
    print(f"{name:20s} F1_M={row['F1_M']:.4f}  AUC={row['AUC']:.4f}  MCC={row['MCC']:.4f}")

results_df = pd.DataFrame(results).sort_values("F1_M", ascending=False)
results_df.to_csv(RESULTS / "stage1_model_comparison_CV.csv", index=False)

print(f"\n=== {N_FOLDS}-Fold Stratified CV Sonuçları (out-of-fold tahminler, train seti üzerinde) ===")
print(results_df.to_string(index=False))

# Out-of-fold olasılıkları sakla (eşik optimizasyonu adımında kullanılacak)
oof_df = pd.DataFrame(oof_predictions)
oof_df["label"] = y_train
oof_df["ORF"] = train["ORF"].values
oof_df.to_csv(PROCESSED / "oof_predictions.csv", index=False)

# --- Final modelleri TÜM train seti üzerinde yeniden eğitip kaydet ---
for name, (model, needs_scaling) in models.items():
    pipe = make_pipeline(model, needs_scaling)
    pipe.fit(X_train, y_train)
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(pipe, MODELS_DIR / f"stage1_{safe_name}.pkl")

print(f"\nModeller (tüm train seti üzerinde yeniden eğitilmiş) kaydedildi: {MODELS_DIR}")