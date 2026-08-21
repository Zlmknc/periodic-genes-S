import pandas as pd
from pathlib import Path
from gprofiler import GProfiler

RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"

candidates = pd.read_csv(RESULTS / "novel_periodic_gene_candidates.csv")
gene_list = candidates["ORF"].tolist()
print(f"GO zenginleştirme analizi için {len(gene_list)} gen kullanılıyor.")

gp = GProfiler(return_dataframe=True)

results = gp.profile(
    organism="scerevisiae",
    query=gene_list,
    sources=["GO:BP", "GO:CC", "GO:MF", "KEGG"],
    user_threshold=0.05,
    significance_threshold_method="fdr",
)

if results.empty:
    print("\nHiçbir terimde istatistiksel olarak anlamlı zenginleşme bulunamadı (FDR<0.05).")
    print("Bu, 15 genlik küçük aday listesiyle beklenebilir bir durum; yine de raporda")
    print("dürüstçe belirtilmeli ve ham p-değerleriyle (eşiksiz) tekrar bakılabilir.")
else:
    results_sorted = results.sort_values("p_value")
    cols_to_show = ["source", "native", "name", "p_value", "intersection_size", "term_size"]
    print("\n=== Anlamlı GO/KEGG Terimleri (FDR < 0.05) ===")
    print(results_sorted[cols_to_show].head(20).to_string(index=False))

results.to_csv(RESULTS / "go_enrichment_results.csv", index=False)
print(f"\nTüm sonuçlar kaydedildi: {RESULTS / 'go_enrichment_results.csv'}")

# --- Karşılaştırma için: bilinen 800 genin GO profili de bakılabilir (opsiyonel referans) ---