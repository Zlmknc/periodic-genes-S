import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

files = {
    "combined": "spellman_all_combined.csv",
    "alpha": "spellman_alpha.csv",
    "cdc15": "spellman_cdc15.csv",
    "cdc28": "spellman_cdc28.csv",
    "elu": "spellman_elu.csv",
    "metadata": "spellman_metadata.csv",
}

dfs = {}
for name, fname in files.items():
    path = RAW / fname
    df = pd.read_csv(path)
    dfs[name] = df
    print(f"\n=== {name} ({fname}) ===")
    print("Boyut:", df.shape)
    print("İlk 3 sütun:", df.columns[:3].tolist())
    print("Son 3 sütun:", df.columns[-3:].tolist())

# Beklenen zaman noktası sayıları (literatürle karşılaştırma için)
expected = {"alpha": 18, "cdc15": 24, "cdc28": 17, "elu": 14}
print("\n--- Beklenen vs Gerçek zaman noktası sayısı (ORF sütunu hariç) ---")
for name, exp_n in expected.items():
    actual_n = dfs[name].shape[1] - 1  # ORF sütununu çıkar
    status = "OK" if abs(actual_n - exp_n) <= 2 else "KONTROL ET"
    print(f"{name}: beklenen~{exp_n}, gerçek={actual_n}  -> {status}")

# Eksik değer kontrolü
print("\n--- Eksik değer (NaN) oranları ---")
for name in ["alpha", "cdc15", "cdc28", "elu"]:
    df = dfs[name].drop(columns=["ORF"])
    nan_ratio = df.isnull().mean().mean()
    print(f"{name}: ortalama NaN oranı = {nan_ratio:.2%}")

# 800 genlik periyodik liste ile örtüşme (bu dosyayı da data/raw içine koyun)
periodic_path = RAW / "spellman_periodic_genes.txt"
if periodic_path.exists():
    periodic = set(x.strip() for x in open(periodic_path).read().split())
    print(f"\n--- Periyodik gen listesi: {len(periodic)} gen ---")
    for name in ["alpha", "cdc15", "cdc28", "elu"]:
        genes_in_exp = set(dfs[name]["ORF"].astype(str))
        overlap = periodic & genes_in_exp
        print(f"{name}: {len(overlap)}/{len(periodic)} periyodik gen bulundu "
              f"({len(overlap)/len(periodic):.1%})")
else:
    print("\nUYARI: spellman_periodic_genes.txt data/raw içinde bulunamadı, oraya kopyalayın.")