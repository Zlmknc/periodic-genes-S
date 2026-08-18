import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score)
import lightgbm as lgb
import joblib

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

train = pd.read_csv(PROCESSED / "split_train.csv")
test = pd.read_csv(PROCESSED / "split_test.csv")

# =========================================================
# SADECE elu_* sütunlarını al. Bunlar 800-genlik etiketin
# türetilmesine hiç karışmadı (etiket alpha+cdc15+cdc28'den
# geldi) -> bu, modelin "etiketleme algoritmasının imzasını
# ezberlemesi" ihtimaline karşı BAĞIMSIZ bir test.
# =========================================================

elu_cols = [c for c in train.columns if c.startswith("elu_")]
print(f"Kullanılan elu özellikleri ({len(elu_cols)} adet):", elu_cols)

X_train_elu, y_train = train[elu_cols], train["label"].values
X_test_elu, y_test = test[elu_cols], test["label"].values

pos_ratio = y_train.mean()
scale_pos_weight = (1 - pos_ratio) / pos_ratio

model = lgb.LGBMClassifier(n_estimators=300, scale_pos_weight=scale_pos_weight,
                            random_state=42, verbose=-1)

# --- 5-fold CV (train üzerinde, out-of-fold) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(model, X_train_elu, y_train, cv=cv,
                               method="predict_proba", n_jobs=-1)[:, 1]
oof_pred = (oof_proba >= 0.5).astype(int)

print("\n=== Sadece-ELU Modeli — 5-Fold CV (Train seti) ===")
print(f"F1_M={f1_score(y_train, oof_pred, average='macro'):.4f}  "
      f"AUC={roc_auc_score(y_train, oof_proba):.4f}  "
      f"MCC={matthews_corrcoef(y_train, oof_pred):.4f}")

# --- Test setinde nihai değerlendirme (tek sefer) ---
model.fit(X_train_elu, y_train)
proba_test = model.predict_proba(X_test_elu)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)

elu_test_result = {
    "Model": "LightGBM (yalnizca ELU ozellikleri)",
    "Accuracy": accuracy_score(y_test, pred_test),
    "Precision_M": precision_score(y_test, pred_test, average="macro"),
    "Recall_M": recall_score(y_test, pred_test, average="macro"),
    "F1_M": f1_score(y_test, pred_test, average="macro"),
    "Kappa": cohen_kappa_score(y_test, pred_test),
    "MCC": matthews_corrcoef(y_test, pred_test),
    "AUC": roc_auc_score(y_test, proba_test),
}

print("\n=== Sadece-ELU Modeli — NİHAİ TEST SONUCU ===")
for k, v in elu_test_result.items():
    print(f"{k}: {v}")

# --- Karşılaştırma tablosu: Fisher G-test vs Sadece-ELU vs Tam model ---
fisher = pd.read_csv(RESULTS / "baseline_fisher_g_test.csv").set_index("metric")["value"]
full_model = pd.read_csv(RESULTS / "final_test_evaluation.csv")
full_model_default = full_model[full_model["threshold_type"] == "default_0.50"].iloc[0]

comparison = pd.DataFrame([
    {"Yaklaşım": "Fisher G-test (klasik, baseline)",
     "AUC": fisher["AUC"], "F1_M": fisher["Macro_F1"], "MCC": fisher["MCC"]},
    {"Yaklaşım": "Sadece-ELU (bağımsız doğrulama)",
     "AUC": elu_test_result["AUC"], "F1_M": elu_test_result["F1_M"], "MCC": elu_test_result["MCC"]},
    {"Yaklaşım": "Tam model (alpha+cdc15+cdc28+elu, tüm özellikler)",
     "AUC": full_model_default["AUC"], "F1_M": full_model_default["F1_M"], "MCC": full_model_default["MCC"]},
])

comparison.to_csv(RESULTS / "elu_independent_validation_comparison.csv", index=False)
print("\n=== KARŞILAŞTIRMA TABLOSU (Test seti) ===")
print(comparison.to_string(index=False))

joblib.dump(model, MODELS_DIR / "elu_only_lightgbm.pkl")