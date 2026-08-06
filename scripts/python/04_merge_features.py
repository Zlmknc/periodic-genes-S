import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

LABEL_SOURCE_EXPERIMENTS = ["alpha", "cdc15", "cdc28"]   # 800-gen listesi bunlardan türetildi
INDEPENDENT_EXPERIMENT = "elu"                             # etikette KULLANILMADI -> döngüsellikten bağımsız

labels = pd.read_csv(PROCESSED / "labels.csv")

# --- Deney bazlı özellik dosyalarını yükle ---
feature_dfs = {}
for name in LABEL_SOURCE_EXPERIMENTS + [INDEPENDENT_EXPERIMENT]:
    feature_dfs[name] = pd.read_csv(PROCESSED / f"{name}_features.csv")

# --- Hepsini ORF üzerinden birleştir ---
merged = labels.copy()
for name, df in feature_dfs.items():
    merged = merged.merge(df, on="ORF", how="left")

print("Birleşim sonrası boyut:", merged.shape)
print("Eksik satır sayısı (herhangi bir deneyde gen kayıpsa):", merged.isnull().any(axis=1).sum())

# =========================================================
# ÇAPRAZ-DENEY TUTARLILIK ÖZELLİKLERİ
# Sadece etiketin türetildiği 3 deneyi (alpha, cdc15, cdc28) kullanıyoruz.
# Mantık: bir gen GERÇEKTEN periyodikse, farklı senkronizasyon yöntemleriyle
# elde edilen bağımsız deneylerin HEPSİNDE tutarlı bir periyodiklik sinyali
# göstermesi beklenir. Tek deneyde yüksek, diğerlerinde düşük çıkan bir
# istatistik muhtemelen gürültüdür (deneye özgü artefakt).
# =========================================================

def cross_experiment_stats(row, feature_name, experiments):
    vals = np.array([row[f"{e}_{feature_name}"] for e in experiments], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return pd.Series({f"cross_{feature_name}_mean": 0.0,
                           f"cross_{feature_name}_std": 0.0,
                           f"cross_{feature_name}_min": 0.0})
    return pd.Series({
        f"cross_{feature_name}_mean": vals.mean(),
        f"cross_{feature_name}_std": vals.std(),
        f"cross_{feature_name}_min": vals.min(),   # "en zayıf halka" - hepsi tutarlı mı?
    })

cross_features_to_build = ["autocorr_lag1", "autocorr_lag2", "ls_band_power", "ls_max_power", "std"]

for feat in cross_features_to_build:
    stats = merged.apply(lambda r: cross_experiment_stats(r, feat, LABEL_SOURCE_EXPERIMENTS), axis=1)
    merged = pd.concat([merged, stats], axis=1)

# "Kaç deneyde otokorelasyon belirgin şekilde pozitif (>0.3)" - basit oy sayma özelliği
def consistency_vote(row, threshold=0.3):
    vals = [row[f"{e}_autocorr_lag1"] for e in LABEL_SOURCE_EXPERIMENTS]
    return sum(1 for v in vals if pd.notna(v) and v > threshold)

merged["n_experiments_high_autocorr"] = merged.apply(consistency_vote, axis=1)

print("\ncross_autocorr_lag1_mean istatistikleri:")
print(merged["cross_autocorr_lag1_mean"].describe())

print("\nn_experiments_high_autocorr dağılımı (etikete göre):")
print(merged.groupby("label")["n_experiments_high_autocorr"].value_counts(normalize=True).unstack().round(3))

# --- Kalan NaN'ları 0 ile doldur (özellik hesaplanamadıysa "sinyal yok" kabul ediyoruz) ---
feature_cols = [c for c in merged.columns if c not in ["ORF", "label"]]
merged[feature_cols] = merged[feature_cols].fillna(0.0)

merged.to_csv(PROCESSED / "final_dataset.csv", index=False)
print(f"\nNihai veri seti kaydedildi: {merged.shape[0]} gen x {len(feature_cols)} özellik")
print("Sütunlar:", feature_cols)