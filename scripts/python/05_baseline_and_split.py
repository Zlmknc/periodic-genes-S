import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROCESSED / "final_dataset.csv")
X = df.drop(columns=["ORF", "label"])
y = df["label"].values

# =========================================================
# FISHER G-TEST — artık 4 deneyin HER BİRİNDE ayrı hesaplanıyor,
# sonra Fisher'ın kombine testi (meta-analiz) ile birleştiriliyor.
# Bu, "ML dört deneyi kullanıyor, klasik test tek deneyi kullanıyor,
# karşılaştırma adil değil" eleştirisini kapatan, tamamen meşru
# (hiçbir ML/label bilgisi kullanmayan) bir güçlendirmedir.
# =========================================================

def fisher_g_test(series):
    v = series[~np.isnan(series)]
    n = len(v)
    if n < 4:
        return 0.0, 1.0
    v = v - v.mean()
    power = np.abs(np.fft.rfft(v))[1:] ** 2
    if power.sum() == 0:
        return 0.0, 1.0
    g_stat = power.max() / power.sum()
    m = len(power)
    p_val = m * (1 - g_stat) ** (m - 1)
    p_val = min(max(p_val, 1e-12), 1.0)   # log(0) hatasını önlemek için alt sınır
    return g_stat, p_val

EXPERIMENTS = ["alpha", "cdc15", "cdc28", "elu"]
p_values_per_experiment = {}

print("=== Fisher G-test — deney bazlı sonuçlar ===")
for exp in EXPERIMENTS:
    raw = pd.read_csv(PROCESSED / f"{exp}_aligned.csv").set_index("ORF")
    raw = raw.loc[df["ORF"]]   # sıralamayı final_dataset ile eşitle

    p_vals = []
    for _, row in raw.iterrows():
        _, p = fisher_g_test(row.values.astype(float))
        p_vals.append(p)
    p_vals = np.array(p_vals)
    p_values_per_experiment[exp] = p_vals

    score = 1 - p_vals
    auc = roc_auc_score(y, score)
    pred = (p_vals < 0.05).astype(int)
    f1 = f1_score(y, pred, average="macro")
    mcc = matthews_corrcoef(y, pred)
    print(f"{exp:8s} AUC={auc:.4f}  Macro F1={f1:.4f}  MCC={mcc:.4f}")

# --- Fisher'ın kombine testi: X^2 = -2 * sum(ln(p_i)), df = 2k ---
p_matrix = np.column_stack([p_values_per_experiment[e] for e in EXPERIMENTS])
combined_stat = -2 * np.sum(np.log(p_matrix), axis=1)
combined_p = chi2.sf(combined_stat, df=2 * len(EXPERIMENTS))

combined_score = 1 - combined_p
combined_auc = roc_auc_score(y, combined_score)
combined_pred = (combined_p < 0.05).astype(int)
combined_f1 = f1_score(y, combined_pred, average="macro")
combined_mcc = matthews_corrcoef(y, combined_pred)

print(f"\n=== Fisher Kombine Testi (4 deney birleşik, X^2 meta-analiz) ===")
print(f"AUC={combined_auc:.4f}  Macro F1={combined_f1:.4f}  MCC={combined_mcc:.4f}")

baseline_summary = pd.DataFrame([
    *[{"Yaklaşım": f"Fisher G-test ({e})",
       "AUC": roc_auc_score(y, 1 - p_values_per_experiment[e]),
       "Macro_F1": f1_score(y, (p_values_per_experiment[e] < 0.05).astype(int), average="macro"),
       "MCC": matthews_corrcoef(y, (p_values_per_experiment[e] < 0.05).astype(int))}
      for e in EXPERIMENTS],
    {"Yaklaşım": "Fisher Kombine Testi (4 deney, meta-analiz)",
     "AUC": combined_auc, "Macro_F1": combined_f1, "MCC": combined_mcc},
])
baseline_summary.to_csv(RESULTS / "baseline_fisher_g_test.csv", index=False)
print("\n", baseline_summary.to_string(index=False))

# =========================================================
# 2) TRAIN / TEST AYRIMI (%80 / %20)
# Train kısmı, model karşılaştırması ve eşik optimizasyonu için
# 5-fold Stratified CV ile kullanılacak (bkz. 06_train_models.py).
# Test seti yalnızca en sonda, bir kez kullanılacak.
# =========================================================

X_train, X_test, y_train, y_test, orf_train, orf_test = train_test_split(
    X, y, df["ORF"], test_size=0.20, stratify=y, random_state=42
)

print(f"\nTrain (CV için): {X_train.shape[0]} ({y_train.mean():.2%} pozitif)")
print(f"Test:            {X_test.shape[0]} ({y_test.mean():.2%} pozitif)")

for name, Xs, ys, orfs in [("train", X_train, y_train, orf_train),
                            ("test", X_test, y_test, orf_test)]:
    out = Xs.copy()
    out["label"] = ys
    out["ORF"] = orfs.values
    out.to_csv(PROCESSED / f"split_{name}.csv", index=False)

print("\nSplit dosyaları kaydedildi: split_train.csv, split_test.csv")