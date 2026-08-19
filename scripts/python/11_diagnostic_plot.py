import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"
FIGURES = Path(__file__).resolve().parent.parent.parent / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

diag = pd.read_csv(RESULTS / "single_experiment_diagnostic.csv")

n_timepoints = {"alpha": 18, "cdc15": 24, "cdc28": 17, "elu": 14}
diag["n_timepoints"] = diag["Deney"].map(n_timepoints)

plt.figure(figsize=(6, 4))
colors = ["#1f77b4" if used else "#d62728" for used in diag["Etikette_kullanildi_mi"]]
plt.scatter(diag["n_timepoints"], diag["AUC"], s=100, c=colors, zorder=3)

for _, row in diag.iterrows():
    plt.annotate(row["Deney"], (row["n_timepoints"], row["AUC"]),
                 textcoords="offset points", xytext=(8, 5), fontsize=10)

plt.xlabel("Zaman noktası sayısı")
plt.ylabel("Tekil-deney AUC (Train, 5-fold CV)")
plt.title("Zaman noktası sayısı ile tekil-deney başarımı ilişkisi")
plt.grid(alpha=0.3, zorder=0)

# lejant: mavi = etikete karışan deney, kırmızı = bağımsız (elu)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10,
           label='Etikete karışan deney (alpha/cdc15/cdc28)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10,
           label='Bağımsız deney (elu)'),
]
plt.legend(handles=legend_elements, loc="lower right", fontsize=8)

plt.tight_layout()
out_path = FIGURES / "n_timepoints_vs_auc.png"
plt.savefig(out_path, dpi=150)
print(f"Grafik kaydedildi: {out_path}")