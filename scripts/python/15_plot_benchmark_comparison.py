from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. Dizin Yapılandırması ve Veri Yükleme
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TABLE_PATH = BASE_DIR / "results" / "tables" / "elu_independent_validation_comparison.csv"
FIG_DIR = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TABLE_PATH)

label_map = {
    "Fisher G-test (yalnızca cdc15)": "Fisher G-test\n(cdc15)",
    "Fisher Kombine Testi (4 deney, meta-analiz)": "Fisher Combined\n(4 Datasets)",
    "Sadece-ELU (bağımsız ML doğrulama)": "ELU-Only LightGBM\n(Independent)",
    "Tam model (alpha+cdc15+cdc28+elu, tüm özellikler)": "Full LightGBM\n(Integrated 4 Datasets)",
}

df["Label_Clean"] = df["Yaklaşım"].map(lambda x: label_map.get(x, x))

# ----------------------------------------------------------------------
# 2. Tipografi ve Stil Ayarları ('grid.axis' kaldırıldı)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

colors = {
    "AUC": "#1f77b4",     # Çelik mavisi
    "F1_M": "#ff7f0e",    # Amber
    "MCC": "#2ca02c",     # Yeşil
}

metrics = ["AUC", "F1_M", "MCC"]
metric_names = ["ROC-AUC", "Macro F1 (F1_M)", "Matthews Corr. (MCC)"]

n_groups = len(df)
n_metrics = len(metrics)

bar_width = 0.22
indices = np.arange(n_groups)

# ----------------------------------------------------------------------
# 3. Çizim
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)

# Kılavuz çizgisi doğrudan eksen üzerinden yalnızca Y eksenine uygulanır:
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

for i, (metric, display_name) in enumerate(zip(metrics, metric_names)):
    offsets = indices + (i - (n_metrics - 1) / 2) * bar_width
    values = df[metric].values

    bars = ax.bar(
        offsets,
        values,
        bar_width,
        label=display_name,
        color=colors[metric],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

ax.set_ylabel("Metric Score", fontweight="semibold")
ax.set_ylim(0, 1.12)
ax.set_xticks(indices)
ax.set_xticklabels(df["Label_Clean"], fontweight="medium")

# Gruplar arası ayrım çizgisi
ax.axvline(x=1.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(0.75, 1.06, "Single Experiment / Classical Baseline", ha="center", fontsize=9.5, style="italic", color="#444")
ax.text(2.5, 1.06, "Multi-Condition & ML Models", ha="center", fontsize=9.5, style="italic", color="#444")

ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")
plt.title("Performance Benchmark Across Single/Multi-Omic Datasets and Methodologies", pad=24, fontweight="bold")

plt.tight_layout()

# ----------------------------------------------------------------------
# 4. Kaydetme
# ----------------------------------------------------------------------
png_path = FIG_DIR / "model_comparison_benchmark.png"
pdf_path = FIG_DIR / "model_comparison_benchmark.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.close()

print(f"[+] Grafikler başarıyla kaydedildi:\n - {png_path}\n - {pdf_path}")