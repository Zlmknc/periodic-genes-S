import pandas as pd
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"

# Overlap ettikleri gerçek genler
overlap_map = {
    "YAL004W": "YAL005C",   # SSA1
    "YOR331C": "YOR332W",   # VMA4
}

labels = pd.read_csv(PROCESSED / "labels.csv")

for dubious, real_gene in overlap_map.items():
    real_label = labels[labels["ORF"] == real_gene]
    if not real_label.empty:
        is_periodic = real_label.iloc[0]["label"]
        print(f"{dubious} (dubious) <-> {real_gene} (gerçek gen): "
              f"{real_gene} 800-gen listesinde periyodik mi? {'EVET' if is_periodic else 'HAYIR'}")
    else:
        print(f"{real_gene} veri setinde bulunamadı.")