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

candidates[["ORF", "periodic_probability"]].to_csv(
    RESULTS / "novel_periodic_gene_candidates.csv", index=False)

# GO zenginleştirme analizi için sade bir ORF listesi de kaydedelim
with open(RESULTS.parent / "novel_candidates_orf_list.txt", "w") as f:
    f.write("\n".join(candidates["ORF"].tolist()))

print(f"\nAday listesi kaydedildi: {RESULTS / 'novel_periodic_gene_candidates.csv'}")
print(f"Sade ORF listesi (GO analizi için): {RESULTS.parent / 'novel_candidates_orf_list.txt'}")