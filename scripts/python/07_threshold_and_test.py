import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score)
import joblib

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

oof = pd.read_csv(PROCESSED / "oof_predictions.csv")
y_train = oof["label"].values
model_names = [c for c in oof.columns if c not in ["label", "ORF"]]

# =========================================================
# 1) EŞİK OPTİMİZASYONU — SADECE out-of-fold (train) tahminleri üzerinde.
# Test setine burada HİÇ dokunmuyoruz.
# =========================================================

BEST_MODEL = "LightGBM"   # CV sonucuna göre en iyi model; değiştirmek isterseniz bu satırın güncellenmesi gerekir

thresholds = np.arange(0.20, 0.55, 0.01)
threshold_results = []

for t in thresholds:
    pred = (oof[BEST_MODEL].values >= t).astype(int)
    threshold_results.append({
        "threshold": round(t, 2),
        "F1_M": f1_score(y_train, pred, average="macro"),
        "Recall_pos": recall_score(y_train, pred, pos_label=1),
        "Precision_pos": precision_score(y_train, pred, pos_label=1),
        "MCC": matthews_corrcoef(y_train, pred),
    })

thr_df = pd.DataFrame(threshold_results)
thr_df.to_csv(RESULTS / "threshold_search_train_oof.csv", index=False)

best_row = thr_df.loc[thr_df["F1_M"].idxmax()]
best_threshold = best_row["threshold"]
print("=== Eşik Tarama Sonuçları (Train OOF üzerinde) ===")
print(thr_df.to_string(index=False))
print(f"\nEn iyi eşik (F1_M'e göre): {best_threshold}  (F1_M={best_row['F1_M']:.4f})")

# =========================================================
# 2) NİHAİ TEST DEĞERLENDİRMESİ — test setine BURADA, TEK SEFER dokunuyoruz.
# =========================================================

test = pd.read_csv(PROCESSED / "split_test.csv")
feature_cols = [c for c in test.columns if c not in ["ORF", "label"]]
X_test, y_test = test[feature_cols], test["label"].values

safe_name = BEST_MODEL.replace(" ", "_").replace("(", "").replace(")", "")
model = joblib.load(MODELS_DIR / f"stage1_{safe_name}.pkl")

proba_test = model.predict_proba(X_test)[:, 1]

final_results = []
for label_t, t in [("default_0.50", 0.50), ("optimized", best_threshold)]:
    pred = (proba_test >= t).astype(int)
    final_results.append({
        "threshold_type": label_t,
        "threshold": t,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision_M": precision_score(y_test, pred, average="macro"),
        "Recall_M": recall_score(y_test, pred, average="macro"),
        "F1_M": f1_score(y_test, pred, average="macro"),
        "Kappa": cohen_kappa_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
        "AUC": roc_auc_score(y_test, proba_test),   # eşikten bağımsız, sabit
    })

final_df = pd.DataFrame(final_results)
final_df.to_csv(RESULTS / "final_test_evaluation.csv", index=False)

print(f"\n=== NİHAİ TEST SONUCU ({BEST_MODEL}) — test seti sadece burada kullanıldı ===")
print(final_df.to_string(index=False))