# Maya Hücre Döngüsünde Periyodik Genlerin Çoklu Deney Entegrasyonu ile Makine Öğrenmesi Temelli Sınıflandırılması

Spellman ve arkadaşlarının (1998) dört senkronizasyon deneyinin (alfa faktörü, cdc15, cdc28, elutriation) tamamı kullanılarak, maya (*Saccharomyces cerevisiae*) hücre döngüsünde periyodik olarak ifade edilen genlerin sızıntısız, yorumlanabilir ve biyolojik olarak doğrulanmış bir makine öğrenmesi hattıyla sınıflandırılması.

Bu proje, tek bir deneye (CDC15) dayanan ve test setinde eşik optimizasyonu yapan önceki bir çalışmanın metodolojik zafiyetlerini gidermek amacıyla sıfırdan tasarlanmıştır.

## İçindekiler

- [Öne Çıkan Sonuçlar](#öne-çıkan-sonuçlar)
- [Proje Neden Var](#proje-neden-var)
- [Metodoloji Özeti](#metodoloji-özeti)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma Sırası](#çalıştırma-sırası)
- [Sonuçlar](#sonuçlar)
- [Sınırlılıklar](#sınırlılıklar)
- [Kaynakça](#kaynakça)
- [Lisans](#lisans)

## Öne Çıkan Sonuçlar

| Yaklaşım | AUC | Makro F1 | MCC |
|---|---|---|---|
| Fisher G-test (klasik taban çizgisi) | 0.6622 | 0.5649 | 0.1675 |
| Sadece-ELU modeli (bağımsız doğrulama) | 0.6455 | 0.6073 | 0.2355 |
| **Nihai model (LightGBM, 4 deney + çapraz özellikler)** | **0.9707** | **0.8871** | **0.7774** |

- 4 deneyin tamamı (alfa, cdc15, cdc28, elutriation), filtrelenmemiş 6178 ORF üzerinde entegre edildi.
- Eşik optimizasyonu **yalnızca eğitim setinin kutu-dışı (out-of-fold) tahminleri üzerinde** yapıldı; test seti tek bir kez, en sonda kullanıldı.
- Döngüsellik riski, etiketin türetilmesine hiç katılmayan **elutriation** deneyiyle bağımsız olarak test edildi.
- Model, altın standart dışındaki genlere uygulandı; adaylar arasında hücre duvarı biyogenezi ile ilişkili genlerin istatistiksel olarak anlamlı biçimde zenginleştiği bulundu (GO:0009272, p_adj=0.019).

## Proje Neden Var

İncelenen bir önceki çalışmada üç kritik sorun tespit edildi:

1. **Veri sızıntısı** — sınıflandırma eşiği doğrudan test seti üzerinde aranmış, aynı test seti üç model aşamasında tekrar tekrar kullanılmış.
2. **Döngüsellik riski** — 800 genlik altın standart etiket, alfa+cdc15+cdc28'in Fourier/korelasyon analiziyle türetilmişken, model özellikleri de benzer spektral yöntemlerle (FFT, otokorelasyon) üretilmiş.
3. **Tek deneye bağımlılık** — yalnızca CDC15 kullanılmış, Spellman'ın yayımladığı diğer üç deney (alfa, cdc28, elutriation) hiç kullanılmamış.

Bu proje, veri setinden başlayarak bu üç sorunu doğrudan hedef alan bir yeniden tasarımdır.

## Metodoloji Özeti

```
Bioconductor yeastCC (R)
        │
        ▼
4 deney: alpha / cdc15 / cdc28 / elu  (ham, filtrelenmemiş, 6178 ORF)
        │
        ▼
Deney-içi özellikler: istatistikler, Lomb–Scargle periodogramı,
sinüzoidal eğri uydurma, otokorelasyon
        │
        ▼
Çapraz-deney tutarlılık özellikleri (alpha+cdc15+cdc28 ortak sinyali)
        │
        ▼
Fisher G-test taban çizgisi  │  Train/Test ayrımı (%80/%20, stratified)
        │
        ▼
5-fold Stratified CV → 11 model karşılaştırması (kutu-dışı tahminler)
        │
        ▼
Eşik optimizasyonu (yalnızca train OOF üzerinde)
        │
        ▼
Nihai test değerlendirmesi (test seti burada TEK SEFER kullanılır)
        │
        ▼
SHAP yorumlanabilirlik  │  ELU ile bağımsız doğrulama
        │
        ▼
Yeni gen keşfi → GO zenginleştirme (g:Profiler)
```

Lomb–Scargle periodogramı, klasik FFT'nin aksine eşit olmayan zaman noktalarını (Spellman verisindeki eksik ölçümler nedeniyle) doğru biçimde ele alır.

## Proje Yapısı

```
periodic-genes-S/
├── data/
│   ├── raw/              # yeastCC'den çekilen 6 ham CSV + periyodik gen listesi
│   └── processed/        # işlenmiş özellik matrisleri, split'ler, OOF tahminler
├── scripts/
│   ├── r/
│   │   └── fetch_data.R              # Bioconductor yeastCC'den veri çekme
│   └── python/
│       ├── 01_validate_data.py       # veri doğrulama
│       ├── 02_preprocess.py          # deney hizalama + etiket oluşturma
│       ├── 03_feature_engineering.py # istatistik + Lomb-Scargle + sinüs fit
│       ├── 04_merge_features.py      # çapraz-deney özellikleri
│       ├── 05_baseline_and_split.py  # Fisher G-test + train/test ayrımı
│       ├── 06_train_models.py        # 11 model, 5-fold CV
│       ├── 07_threshold_and_test.py  # eşik optimizasyonu + nihai test
│       ├── 08_shap_analysis.py       # SHAP yorumlanabilirlik
│       ├── 09_elu_only_validation.py # bağımsız (ELU) doğrulama
│       ├── 10_single_experiment_diagnostic.py
│       ├── 11_diagnostic_plot.py
│       ├── 12_novel_gene_discovery.py
│       ├── 13_go_enrichment.py       # GO zenginleştirme (g:Profiler)
│       └── 14_annotate_candidates.py # aday genlerin isim/fonksiyon eşleştirmesi
├── results/
│   ├── tables/            # tüm sonuç tabloları (CSV)
│   └── figures/           # SHAP grafiği, tanı grafiği
├── models/                 # eğitilmiş model dosyaları (.pkl)
├── requirements.txt
└── README.md
```

## Kurulum

### Python ortamı

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### R ortamı (veri temini için)

```r
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("yeastCC")
BiocManager::install("Biobase")
```

## Çalıştırma Sırası

```bash
# 1) Veriyi çek (R)
Rscript scripts/r/fetch_data.R

# 2) Python pipeline'ı sırasıyla çalıştır
cd scripts/python
python 01_validate_data.py
python 02_preprocess.py
python 03_feature_engineering.py
python 04_merge_features.py
python 05_baseline_and_split.py
python 06_train_models.py
python 07_threshold_and_test.py
python 08_shap_analysis.py
python 09_elu_only_validation.py
python 10_single_experiment_diagnostic.py
python 11_diagnostic_plot.py
python 12_novel_gene_discovery.py
python 13_go_enrichment.py       # internet bağlantısı gerektirir (g:Profiler API)
python 14_annotate_candidates.py
```

Her script, `data/processed/`, `results/tables/`, `results/figures/` ve `models/` altına çıktılarını yazar; script'ler birbirinin çıktısına bağımlı olduğundan sıra önemlidir.

## Sonuçlar

Ayrıntılı sonuç tabloları `results/tables/` altında, grafikler `results/figures/` altında bulunur. Tam metodoloji, bulgular ve tartışma için proje kapsamında hazırlanan tez dokümanına bakınız.

Öne çıkan bulgular:

- **Model karşılaştırması**: LightGBM, 11 model arasında en iyi performansı gösterdi (5-fold CV: F1=0.9024, AUC=0.9703).
- **SHAP analizi**: En önemli özelliklerden ikisi, bu projede geliştirilen çapraz-deney tutarlılık özellikleridir (`cross_ls_band_power_mean`, `cross_std_mean`).
- **Çapraz-deney tutarlılığı**: Periyodik genlerin %73.4'ü 2-3 deneyde birden yüksek otokorelasyon gösterirken, periyodik olmayanlarda bu oran yalnızca %19.
- **Yeni gen keşfi**: 16 yüksek-güvenli aday gen arasında SPS100, SPS1, DPM1, RHO1 gibi hücre duvarı/spor duvarı biyogenezi ile ilişkili genler bulundu (GO:0009272, p_adj=0.019).

## Sınırlılıklar

- Bağımsız gold-standard listeleriyle (Pramila 2006, Rowicka 2007) dış doğrulama yapılmamıştır.
- Modeller arası istatistiksel anlamlılık testi (5×2cv vb.) uygulanmamıştır.
- Tek `random_state` (42) kullanılmıştır; çoklu tohum tekrarı yoktur.
- Hiperparametreler manuel seçilmiştir, sistematik arama (grid/random search) yapılmamıştır.
- GO zenginleştirmeye katkı sağlayan genlerin kimliği, g:Profiler arayüzünden yarı-elle çıkarılmıştır; API üzerinden programatik doğrulama önerilir.

Ayrıntılı sınırlılık tartışması için tez dokümanının 7. bölümüne bakınız.

## Kaynakça

- Spellman, P. T., et al. (1998). Comprehensive identification of cell cycle–regulated genes of the yeast *Saccharomyces cerevisiae* by microarray hybridization. *Molecular Biology of the Cell*, 9(12), 3273–3297.
- Liew, A. W. C., et al. (2007). Spectral estimation in unevenly sampled space of periodically expressed microarray time series data. *BMC Bioinformatics*, 8(1), 137.
- Wu, W. S., & Li, W. H. (2008). Systematic identification of yeast cell cycle transcription factors using multiple data sources. *BMC Bioinformatics*, 9(1), 522.
- Cheng, C., et al. (2013). Identification of yeast cell cycle regulated genes based on genomic features. *BMC Systems Biology*, 7(1), 70.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Reimand, J., et al. (2007). g:Profiler — a web-based toolset for functional profiling of gene lists. *Nucleic Acids Research*, 35(suppl_2), W193–W200.

## Lisans

Bu proje akademik/eğitim amaçlıdır. Kullanılan veri seti (Spellman et al. 1998) Bioconductor `yeastCC` paketi aracılığıyla temin edilmiştir; orijinal veri lisansı ve atıf koşulları için ilgili yayına bakınız.
