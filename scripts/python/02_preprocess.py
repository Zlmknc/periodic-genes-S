import pandas as pd
import numpy as np
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = ["alpha", "cdc15", "cdc28", "elu"]

def load_experiment(name):
    df = pd.read_csv(RAW / f"spellman_{name}.csv")
    df = df.set_index("ORF")
    times = []
    for col in df.columns:
        m = re.match(rf"{name}_(\d+)", col)
        if not m:
            raise ValueError(f"Beklenmeyen sütun ismi: {col}")
        times.append(int(m.group(1)))
    times = np.array(times)
    order = np.argsort(times)          # zaman sırasına göre sütunları diz
    df = df.iloc[:, order]
    times = times[order]
    return df, times

data = {}
for name in EXPERIMENTS:
    df, times = load_experiment(name)
    data[name] = {"matrix": df, "times": times}
    print(f"{name}: {df.shape[0]} gen x {df.shape[1]} zaman noktası, zamanlar(dk)={times.tolist()}")

# --- Etiket, sadece alpha+cdc15+cdc28 ortak genleri üzerinden (orijinal 800-gen listesi bu 3 deneyden türetildi) ---
periodic = set(x.strip() for x in open(RAW / "spellman_periodic_genes.txt").read().split())
label_genes = set.intersection(*[set(data[n]["matrix"].index) for n in ["alpha", "cdc15", "cdc28"]])
print(f"\nalpha+cdc15+cdc28 ortak gen sayısı: {len(label_genes)}")

label_df = pd.DataFrame({"ORF": sorted(label_genes)})
label_df["label"] = label_df["ORF"].isin(periodic).astype(int)
print("Etiket dağılımı:\n", label_df["label"].value_counts())
print("Pozitif oran: {:.2%}".format(label_df["label"].mean()))

label_df.to_csv(PROCESSED / "labels.csv", index=False)

# --- Her deneyin hizalanmış (zamana göre sıralı) matrisini ve zaman dizisini kaydet ---
for name in EXPERIMENTS:
    out = data[name]["matrix"].reset_index()
    out.to_csv(PROCESSED / f"{name}_aligned.csv", index=False)
    np.save(PROCESSED / f"{name}_times.npy", data[name]["times"])

print("\nİşlenmiş dosyalar kaydedildi:", PROCESSED)