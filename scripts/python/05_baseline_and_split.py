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
# 1) FISHER'S G-TEST BASELINE (klasik istatistiksel yöntem)
# ML olmadan, sadece cdc15 serisindeki en güçlü periyodogram
# bileşeninin, beklenen (gürültü altında) dağılımına göre ne
# kadar aşırı olduğunu test eder. Modelin üstüne inşa ettiği
# "sıfır noktası" budur.
# =========================================================

def fisher_g_test(cdc15_series):
    """Basitleştirilmiş Fisher G-istatistiği: FFT güç spektrumundaki
    en büyük bileşenin toplam güce oranı. NaN'lar çıkarılır."""
    v = cdc15_series[~np.isnan(cdc15_series)]
    n = len(v)
    if n < 4:
        return 0.0, 1.0
    v = v - v.mean()
    power = np.abs(np.fft.rfft(v))[1:] ** 2   # DC bileşeni haric
    if power.sum() == 0:
        return 0.0, 1.0
    g_stat = power.max() / power.sum()
    m = len(power)
    # Fisher'in asimptotik p-deger yaklaşımı
    p_val = m * (1 - g_stat) ** (m - 1)
    p_val = min(max(p_val, 0.0), 1.0)
    return g_stat, p_val

cdc15_raw = pd.read_csv(PROCESSED / "cdc15_aligned.csv").set_index("ORF")
cdc15_raw = cdc15_raw.loc[df["ORF"]]   # sıralamayı final_dataset ile eşitle

g_stats, p_vals = [], []
for _, row in cdc15_raw.iterrows():
    g, p = fisher_g_test(row.values.astype(float))
    g_stats.append(g)
    p_vals.append(p)

baseline_score = 1 - np.array(p_vals)   # düşük p-değeri = yüksek periyodiklik skoru
baseline_auc = roc_auc_score(y, baseline_score)
print(f"Fisher G-test baseline AUC: {baseline_auc:.4f}")

baseline_pred = (np.array(p_vals) < 0.05).astype(int)
baseline_f1 = f1_score(y, baseline_pred, average="macro")
baseline_mcc = matthews_corrcoef(y, baseline_pred)
print(f"Fisher G-test baseline (p<0.05) -> Macro F1: {baseline_f1:.4f}, MCC: {baseline_mcc:.4f}")

pd.DataFrame({
    "metric": ["AUC", "Macro_F1", "MCC"],
    "value": [baseline_auc, baseline_f1, baseline_mcc]
}).to_csv(RESULTS / "baseline_fisher_g_test.csv", index=False)

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