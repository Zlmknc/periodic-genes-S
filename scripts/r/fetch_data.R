# =========================================================
# fetch_data.R
# Spellman et al. (1998) maya hücre döngüsü veri setini
# Bioconductor 'yeastCC' paketinden çeker ve deneylere
# (alpha, cdc15, cdc28, elu) göre ayırarak CSV olarak kaydeder.
# =========================================================

# 1) Paket kurulumu (yalnızca ilk seferde gerekir)
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!require("yeastCC", quietly = TRUE)) BiocManager::install("yeastCC", update = FALSE, ask = FALSE)
if (!require("Biobase", quietly = TRUE)) BiocManager::install("Biobase", update = FALSE, ask = FALSE)

library(yeastCC)
library(Biobase)

# 2) Veriyi yükle
data(spYCCES)

# 3) Yapıyı keşfet -- konsol çıktısını mutlaka inceleyin
cat("Boyut (gen x örnek):\n"); print(dim(spYCCES))
cat("\nİlk 10 sütun ismi:\n"); print(head(colnames(exprs(spYCCES)), 10))
cat("\npData sütunları:\n"); print(colnames(pData(spYCCES)))
cat("\npData ilk satırlar:\n"); print(head(pData(spYCCES)))

# 4) Ham ifade matrisini ve metadata'yı al
expr  <- exprs(spYCCES)          # genler x örnekler
meta  <- pData(spYCCES)          # her örneğin (sütunun) deney/zaman bilgisi
genes <- rownames(expr)          # ORF isimleri (YAL001C gibi)

# 5) Tüm birleşik veriyi kaydet (satır = gen, sütun = örnek)
out_dir <- "../../data/raw"      # scripts/r/ içinden çalıştırıldığı için proje köküne göre yol
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(data.frame(ORF = genes, expr, check.names = FALSE),
          file.path(out_dir, "spellman_all_combined.csv"), row.names = FALSE)
write.csv(meta, file.path(out_dir, "spellman_metadata.csv"), row.names = TRUE)

# 6) Deneye göre otomatik ayırma (sütun ismine göre eşleştirme)
#    NOT: Adım 3'teki "pData sütunları" çıktısına bakıp eşleşmenin
#    doğru olduğunu teyit edin. Sütun isimleri beklenenden farklıysa
#    aşağıdaki pattern'leri güncelleyin.

split_by_pattern <- function(colnames_vec, pattern) {
  grepl(pattern, colnames_vec, ignore.case = TRUE)
}

for (exp_name in c("alpha", "cdc15", "cdc28", "elu")) {
  idx <- split_by_pattern(colnames(expr), exp_name)
  if (sum(idx) == 0) {
    cat(sprintf("UYARI: '%s' için sütun ismiyle eşleşme bulunamadı, pData'ya bakılmalı.\n", exp_name))
    next
  }
  sub_expr <- expr[, idx, drop = FALSE]
  df_out <- data.frame(ORF = genes, sub_expr, check.names = FALSE)
  write.csv(df_out, file.path(out_dir, paste0("spellman_", exp_name, ".csv")), row.names = FALSE)
  cat(sprintf("%s kaydedildi: %d gen x %d zaman noktasi\n", exp_name, nrow(df_out), sum(idx)))
}

cat("\nTüm dosyalar '", out_dir, "' klasörüne kaydedildi.\n", sep = "")