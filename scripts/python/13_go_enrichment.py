import os
import warnings
from pathlib import Path
from gprofiler import GProfiler
import pandas as pd

# Loky uyarısını engelle
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"
csv_path = RESULTS / "novel_periodic_gene_candidates.csv"

# 1. CSV'yi virgül ayracıyla garantiye alarak oku
candidates = pd.read_csv(csv_path, sep=",")

# Eğer tek sütun algılandıysa düzelt
if "ORF" not in candidates.columns:
    candidates = pd.read_csv(csv_path, sep=None, engine="python")
    if "ORF" not in candidates.columns:
        # İlk sütunu bölerek al
        first_col = candidates.columns[0]
        candidates = candidates[first_col].str.split(",", expand=True)
        candidates.columns = ["ORF", "periodic_probability"]

# 2. Sadece temiz ORF adlarını al (virgül veya sayı artığı kalmasın)
gene_list = [
    str(x).split(",")[0].strip() 
    for x in candidates["ORF"].dropna().tolist() 
    if str(x).startswith("Y")
]

print(f"GO zenginleştirme analizi için {len(gene_list)} geçerli maya geni kullanılıyor:")
print(gene_list)

# 3. g:Profiler Analizi
gp = GProfiler(return_dataframe=True)

# Mayada yanıltıcı insan yolakları getiren KEGG yerine GO terimleri kullanılır
results = gp.profile(
    organism="scerevisiae",
    query=gene_list,
    sources=["GO:BP", "GO:CC", "GO:MF"],  # Mayaya özgü ve güvenilir kaynaklar
    user_threshold=0.05,
    significance_threshold_method="g_SCS",  # g:Profiler'ın önerdiği çoklu test düzeltmesi
    no_evidences=False,  # Eşleşen genleri görmek için
)

if results.empty:
    print("\n[!] Eşik altı (g_SCS < 0.05) anlamlı terim bulunamadı.")
    print("Aday gen sayısı (~20) az olduğu için çoklu test düzeltmesi katı kalmış olabilir.")
    print("Ham p-değerlerini görmek için user_threshold=1.0 ile en üstteki terimlere bakılabilir.")
else:
    results_sorted = results.sort_values("p_value")
    cols_to_show = ["source", "native", "name", "p_value", "intersection_size", "intersections"]
    print("\n=== Anlamlı GO Terimleri ===")
    print(results_sorted[cols_to_show].head(25).to_string(index=False))

results.to_csv(RESULTS / "go_enrichment_results.csv", index=False)
print(f"\nTüm sonuçlar kaydedildi: {RESULTS / 'go_enrichment_results.csv'}")