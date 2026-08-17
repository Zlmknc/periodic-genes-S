import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
FIGURES = Path(__file__).resolve().parent.parent.parent / "results" / "figures"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
FIGURES.mkdir(parents=True, exist_ok=True)

BEST_MODEL = "LightGBM"
safe_name = BEST_MODEL.replace(" ", "_")
loaded = joblib.load(MODELS_DIR / f"stage1_{safe_name}.pkl")

# 06_train_models.py'de model bir sklearn Pipeline içinde kaydedilmişti
# (needs_scaling=False olsa bile Pipeline([("clf", model)]) kullanılmıştı).
# SHAP TreeExplainer ham ağaç modelini bekliyor, Pipeline'ı değil.
if hasattr(loaded, "named_steps"):
    model = loaded.named_steps["clf"]
else:
    model = loaded

train = pd.read_csv(PROCESSED / "split_train.csv")
feature_cols = [c for c in train.columns if c not in ["ORF", "label"]]
X_train = train[feature_cols]

# Hesaplama süresini kısaltmak için örneklem alalım (SHAP tüm veri için yavaş olabilir)
sample = X_train.sample(n=min(1000, len(X_train)), random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# LightGBM binary classification için shap_values bazen liste, bazen tek array döner
if isinstance(shap_values, list):
    shap_values = shap_values[1]   # pozitif sınıf

# --- Özet grafik (en önemli 20 özellik) ---
plt.figure()
shap.summary_plot(shap_values, sample, max_display=20, show=False)
plt.tight_layout()
plt.savefig(FIGURES / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Ortalama mutlak SHAP değeri tablosu ---
mean_abs_shap = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

mean_abs_shap.to_csv(Path(__file__).resolve().parent.parent.parent / "results" / "tables" / "shap_importance.csv", index=False)
print(mean_abs_shap.head(20).to_string(index=False))
print(f"\nGrafik kaydedildi: {FIGURES / 'shap_summary.png'}")