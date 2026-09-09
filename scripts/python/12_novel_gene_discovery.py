import pandas as pd
import numpy as np
from pathlib import Path
import joblib

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Tüm veri seti (train+test birleşik) - artık final model seçimi bitti,
# keşif için TÜM genomu kullanmak istiyoruz.
full = pd.read_csv(PROCESSED / "final_dataset.csv")
feature_cols = [c for c in full.columns if c not in ["ORF", "label"]]

loaded = joblib.load(MODELS_DIR / "stage1_LightGBM.pkl")
model = loaded.named_steps["clf"] if hasattr(loaded, "named_steps") else loaded

proba_all = model.predict_proba(full[feature_cols])[:, 1]
full["periodic_probability"] = proba_all

# --- Bilinen 800 genin dışındaki, ama modelin YÜKSEK GÜVENLE periyodik dediği genler ---
known_positive = full[full["label"] == 1]
candidates = full[(full["label"] == 0) & (full["periodic_probability"] >= 0.60)].copy()
candidates = candidates.sort_values("periodic_probability", ascending=False)

print(f"Bilinen periyodik gen sayısı (800-gen listesi): {len(known_positive)}")
print(f"Yeni aday sayısı (label=0 ama olasılık>=0.80): {len(candidates)}")
print("\nEn yüksek olasılıklı ilk 20 aday:")
print(candidates[["ORF", "periodic_probability"]].head(20).to_string(index=False))

# =========================================================
# Dubious ORF filtresi — SGD'de "unlikely to encode a functional
# protein" olarak işaretlenmiş, doğrulanmış bir genle tamamen
# örtüşen ORF'ler, muhtemelen o genin sinyalini yansıtıyor.
# Bkz. 15_check_overlap_artifact.py doğrulaması: YOR331C'nin
# örtüştüğü VMA4/YOR332W zaten 800-gen listesinde periyodik.
# =========================================================
DUBIOUS_ORFS = {
    "YAL004W": "SSA1/YAL005C ile tamamen örtüşüyor (dubious ORF)",
    "YOR331C": "VMA4/YOR332W ile tamamen örtüşüyor (dubious ORF, VMA4 zaten periyodik)",
}

removed = candidates[candidates["ORF"].isin(DUBIOUS_ORFS.keys())]
if not removed.empty:
    print(f"\nDubious ORF filtresiyle çıkarılan {len(removed)} aday:")
    for _, row in removed.iterrows():
        print(f"  {row['ORF']} (p={row['periodic_probability']:.4f}) -> {DUBIOUS_ORFS[row['ORF']]}")

candidates_clean = candidates[~candidates["ORF"].isin(DUBIOUS_ORFS.keys())].copy()
print(f"\nTemiz aday sayısı (dubious ORF'ler çıkarıldıktan sonra): {len(candidates_clean)}")

# Hem tam (ham) hem temiz listeyi ayrı ayrı kaydedelim -- şeffaflık için
candidates.to_csv(RESULTS / "novel_periodic_gene_candidates_raw.csv", index=False)
candidates_clean.to_csv(RESULTS / "novel_periodic_gene_candidates.csv", index=False)

with open(RESULTS.parent / "novel_candidates_orf_list.txt", "w") as f:
    f.write("\n".join(candidates_clean["ORF"].tolist()))

print(f"\nTemiz aday listesi kaydedildi: {RESULTS / 'novel_periodic_gene_candidates.csv'}")
print(f"Ham (filtresiz) liste ayrıca saklandı: {RESULTS / 'novel_periodic_gene_candidates_raw.csv'}")