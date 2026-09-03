import os
import sys

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import lightgbm as lgb
import joblib

# ---------------------------------------------------------
# Dizin Yapılandırması
# ---------------------------------------------------------
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
base_elu_cols = [c for c in train.columns if c.startswith("elu_")]
print(f"Mevcut ELU temel özellikleri ({len(base_elu_cols)} adet):", base_elu_cols)


def enrich_elu_features(df, cols):
    """
    Mevcut ELU zaman serisi özetlerinden (Lomb-Scargle, genlik, faz, otokorelasyon)
    fiziksel dalga formu tutarlılığı ve spektral kaliteyi ölçen ek özellikler türetir.
    """
    df_feat = df[cols].copy()

    # 1. Sinyal / Gürültü Oranı Yaklaşımı (SNR Proxy)
    if "elu_sin_amplitude" in df_feat.columns and "elu_std" in df_feat.columns:
        df_feat["elu_snr_proxy"] = df_feat["elu_sin_amplitude"] / (df_feat["elu_std"] + 1e-6)

    # 2. Spektral Konsantrasyon (Bant Gücünün Maksimum Güce Oranı)
    if "elu_ls_band_power" in df_feat.columns and "elu_ls_max_power" in df_feat.columns:
        df_feat["elu_ls_power_ratio"] = df_feat["elu_ls_max_power"] / (df_feat["elu_ls_band_power"] + 1e-6)

    # 3. Faz Uyum Skoru (Hücre döngüsü periyodunda normalize edilmiş faz)
    if "elu_sin_phase" in df_feat.columns and "elu_ls_dominant_period" in df_feat.columns:
        df_feat["elu_phase_period_mod"] = np.sin(
            df_feat["elu_sin_phase"] * (2 * np.pi / (df_feat["elu_ls_dominant_period"] + 1e-6))
        )

    # 4. Gecikmeli Otokorelasyon Gradyanı (Dalga sönümlenme hızı)
    if "elu_autocorr_lag1" in df_feat.columns and "elu_autocorr_lag2" in df_feat.columns:
        df_feat["elu_autocorr_decay"] = df_feat["elu_autocorr_lag1"] - df_feat["elu_autocorr_lag2"]

    # 5. Dalga Formu Dinamik Aralığı
    if "elu_range" in df_feat.columns and "elu_std" in df_feat.columns:
        df_feat["elu_crest_factor"] = df_feat["elu_range"] / (df_feat["elu_std"] + 1e-6)

    return df_feat


X_train_elu = enrich_elu_features(train, base_elu_cols)
X_test_elu = enrich_elu_features(test, base_elu_cols)

y_train = train["label"].values
y_test = test["label"].values

print(f"Zenginleştirilmiş toplam ELU özellik sayısı: {X_train_elu.shape[1]}")

pos_ratio = y_train.mean()
scale_pos_weight = (1 - pos_ratio) / pos_ratio

# Küçük özellik uzayında genellemeyi artıran, aşırı öğrenmeyi önleyen hiperparametreler
model = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_child_samples=25,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=4,
    verbose=-1,
)

# --- 5-fold CV (train üzerinde, out-of-fold) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(
    model, X_train_elu, y_train, cv=cv, method="predict_proba", n_jobs=1
)[:, 1]
oof_pred = (oof_proba >= 0.5).astype(int)

print("\n=== Sadece-ELU Modeli — 5-Fold CV (Train seti) ===")
print(
    f"F1_M={f1_score(y_train, oof_pred, average='macro'):.4f}  "
    f"AUC={roc_auc_score(y_train, oof_proba):.4f}  "
    f"MCC={matthews_corrcoef(y_train, oof_pred):.4f}"
)

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
fisher_table = pd.read_csv(RESULTS / "baseline_fisher_g_test.csv")
fisher_combined = fisher_table[fisher_table["Yaklaşım"].str.contains("Kombine")].iloc[0]
fisher_cdc15 = fisher_table[fisher_table["Yaklaşım"].str.contains(r"\(cdc15\)")].iloc[0]

full_model = pd.read_csv(RESULTS / "final_test_evaluation.csv")
full_model_default = full_model[full_model["threshold_type"] == "default_0.50"].iloc[0]

comparison = pd.DataFrame([
    {
        "Yaklaşım": "Fisher G-test (yalnızca cdc15)",
        "AUC": fisher_cdc15["AUC"],
        "F1_M": fisher_cdc15["Macro_F1"],
        "MCC": fisher_cdc15["MCC"],
    },
    {
        "Yaklaşım": "Fisher Kombine Testi (4 deney, meta-analiz)",
        "AUC": fisher_combined["AUC"],
        "F1_M": fisher_combined["Macro_F1"],
        "MCC": fisher_combined["MCC"],
    },
    {
        "Yaklaşım": "Sadece-ELU (bağımsız ML doğrulama)",
        "AUC": elu_test_result["AUC"],
        "F1_M": elu_test_result["F1_M"],
        "MCC": elu_test_result["MCC"],
    },
    {
        "Yaklaşım": "Tam model (alpha+cdc15+cdc28+elu, tüm özellikler)",
        "AUC": full_model_default["AUC"],
        "F1_M": full_model_default["F1_M"],
        "MCC": full_model_default["MCC"],
    },
])

comparison.to_csv(RESULTS / "elu_independent_validation_comparison.csv", index=False)
print("\n=== KARŞILAŞTIRMA TABLOSU (Test seti) ===")
print(comparison.to_string(index=False))

joblib.dump(model, MODELS_DIR / "elu_only_lightgbm.pkl")