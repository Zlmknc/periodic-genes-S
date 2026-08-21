import pandas as pd
from pathlib import Path
from gprofiler import GProfiler

RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"

candidates = pd.read_csv(RESULTS / "novel_periodic_gene_candidates.csv")

gp = GProfiler(return_dataframe=True)
conv = gp.convert(organism="scerevisiae", query=candidates["ORF"].tolist(),
                   target_namespace="ENSG")  # standart gen ismi icin

# g:Convert bazi alan isimlerini farkli dondurebilir; kontrol edip birlestirelim
print(conv.columns.tolist())
print(conv.head(20))

merged = candidates.merge(conv[["incoming", "name", "description"]],
                           left_on="ORF", right_on="incoming", how="left")
merged.to_csv(RESULTS / "novel_candidates_annotated.csv", index=False)
print(merged[["ORF", "name", "description", "periodic_probability"]].to_string(index=False))