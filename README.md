# Maya Hücre Döngüsünde Periyodik Genlerin Çoklu Deney Entegrasyonu ile Makine Öğrenmesi Temelli Sınıflandırılması

Spellman ve arkadaşlarının (1998) dört senkronizasyon deneyinin (alfa faktörü, cdc15, cdc28, elutriation) tamamı kullanılarak, maya (*Saccharomyces cerevisiae*) hücre döngüsünde periyodik olarak ifade edilen genlerin sızıntısız, yorumlanabilir ve biyolojik olarak doğrulanmış bir makine öğrenmesi hattıyla sınıflandırılması; ardından modelin altın standart dışındaki genlere uygulanarak yeni aday genlerin keşfi ve fonksiyonel doğrulama işlemleri yapılmıştır.

## Projenin Temelleri

Bu çalışma eğitim dönemi süresince yapılan bir ders projesi olan, *"Classification of Periodic Genes in Yeast Gene Expression Data Using Machine Learning"* başlıklı proje kapsamında ilk versiyonu üzerinden geliştirilmiştir: yalnızca CDC15 senkronizasyon deneyinin "temizlenmiş" 4381 genlik alt kümesi üzerinde, 13 temel istatistiksel/FFT/otokorelasyon özniteliğiyle başlayan, sırasıyla sinüzoidal eğri uydurma özellikleri eklenen ve SMOTE/SMOTETomek ile dengelenen bir VotingClassifier ensemble modeline uzanan üç aşamalı bir çalışmaydı. Bu ilk versiyon, LightGBM ile Aşama 1'de makro F1=0.8413'e, nihai ensemble modelle Aşama 3'te makro F1=0.8168 ve AUC=0.9056'ya ulaşmıştır.

Bu proje, ilk çalışmanın ayrıntılı bir öz-değerlendirmesiyle başlatılmıştır. Değerlendirme sürecinde üç kritik metodolojik zafiyet tespit edildi:

1. **Eşik optimizasyonunun test seti üzerinde yapılması** ve aynı test setinin üç aşama boyunca tekrar tekrar kullanılması — klasik bir veri sızıntısı deseni.
2. **Etiket–özellik döngüselliği riski**: 800 genlik altın standart liste zaten alfa+cdc15+cdc28'in Fourier/korelasyon analiziyle türetilmişken, model özniteliklerinin de benzer spektral yöntemlerle (FFT, otokorelasyon) üretilmesi.
3. **Tek deneye (CDC15) bağımlılık** — Spellman'ın yayımladığı diğer üç deneyin (alfa, cdc28, elutriation) hiç kullanılmaması.

Bu proje, ilk çalışmanın genel araştırma sorusunu (maya genlerinde periyodikliğin ML ile sınıflandırılması) koruyarak, bu üç sorunu doğrudan hedef alan, veri setinden başlayan bir yeniden inşa sürecidir. Bu depo, ilk projenin, **eleştirisi üzerine inşa edilmiş, metodolojik olarak yeniden tasarlanmış devamıdır.**

## İlk versiyon ile farklar 

| | İlk versiyon | Bu proje (son hâli) |
|---|---|---|
| Veri kapsamı | Yalnızca CDC15, "temizlenmiş" 4381 gen (800/800 periyodik, gen kaybı: %20.9) | 4 deneyin tamamı (alfa, cdc15, cdc28, elu), filtrelenme uygulanmadan kullanılan 6178 gen |
| Periyodiklik tespiti | Klasik FFT (eşit örnekleme varsayar) | Lomb–Scargle periodogramı (eşit olmayan zaman noktalarını ele alır) + sınırlandırılmış (bounds'lu) sinüzoidal eğri uydurma |
| Özellik mühendisliği | Tek deneyden 13→17 öznitelik | 4 deneyden 68 öznitelik, deneyler arası **çapraz-tutarlılık öznitelikleri** dahil |
| Sınıf dengesizliği | SMOTE / SMOTETomek | Algoritma düzeyinde sınıf ağırlıklandırma (sentetik örnekleme bulunmamaktadır) |
| Taban çizgisi (baseline) karşılaştırması | Baseline Karşılaştırması Bulunmamaktadır | Fisher G-testi (tekil deney + 4 deneyli meta-analiz kombine test) |
| Değerlendirme protokolü | Eşik optimizasyonu test setinde, 3 kez kullanılmış | Eşik optimizasyonu yalnızca eğitim setinin 5-fold kutu-dışı (out-of-fold) tahminlerinde; test seti tek sefer |
| Model yorumlanabilirliği | Bulunmamaktadır | SHAP analizi |
| Döngüsellik kontrolü | Tartışılmamıştır | Etiketin türetilmesine hiç katılmayan elutriation deneyiyle bağımsız doğrulama |
| Yeni gen keşfi | Literatürde bahsedilmiş | Güven eşiğiyle (≥0.80) aday tarama, örtüşen (dubious) ORF artefaktlarının tespiti/filtrelenmesi, GO zenginleştirme ile biyolojik doğrulama |

## Projenin Öne Çıkan Sonuçları 

| Yaklaşım | AUC | Makro F1 | MCC |
|---|---|---|---|
| Fisher G-test (yalnızca cdc15, klasik taban çizgisi) | 0.6622 | 0.5649 | 0.1675 |
| Fisher Kombine Testi (4 deney, meta-analiz) | 0.7871 | 0.5731 | 0.2848 |
| Sadece-ELU modeli (bağımsız doğrulama, etikete karışmayan tek deney) | 0.6938 | 0.5954 | 0.2324 |
| **Nihai model (LightGBM, 4 deney + çapraz özellikler)** | **0.9625** | **0.8828** | **0.7666** |

- İlk versiyondaki en iyi sonuç (Aşama 1 LightGBM, F1=0.8413) bu projede belirgin biçimde aşılmıştır.
- Eşik optimizasyonunun test setinde değil, yalnızca eğitim setinde yapılması sayesinde, "en iyi" eşik ile varsayılan eşik arasındaki fark test setinde ihmal edilebilir düzeyde çıktı — ilk versiyondaki sızıntı riskinin bu tasarımda pratikte önemli bir fark yaratmadığı, metodolojinin doğru kurulduğu gösterilmiştir.
- SHAP analizinde en belirleyici özniteliklerden ikisi, bu projede geliştirilen çapraz-deney tutarlılık öznitelikleridir.
- Yalnızca elutriation özellikleriyle eğitilen bağımsız model, klasik tekil taban çizgisini geçti ama 4-deneyli kombine testin ve tam modelin belirgin altında kalmıştır — bu, nihai modelin başarısının gerçek çoklu-deney entegrasyonundan geldiğine işaret ediyor.
- Nihai model, 800 genlik altın standart dışındaki genlere uygulandı; güven eşiği ile (≥0.80) periyodik tahmin edilen adaylar arasında iki tanesi ("dubious ORF", komşu doğrulanmış genlerle tamamen örtüşen açık okuma çerçeveleri) tespit edilip filtrelenmiştir. Kalan 19 temiz adayla yapılan GO zenginleştirme analizi, hücre duvarı/spor duvarı biyogenezi ile ilgili genlerde istatistiksel olarak zenginleşme GO:0009272, p=0.0391 seviyesinde bir anlamlılık bulunmuştur  — katkı sağlayan genler: FLC3, SPS1, SPS100, RHO1.

## Metodoloji Özeti

```
Bioconductor yeastCC (R)
        │
        ▼
4 deney: alpha / cdc15 / cdc28 / elu  (ham, filtrelenmemiş, 6178 ORF)
        │
        ▼
Deney-içi özellikler: istatistikler, Lomb–Scargle periodogramı,
sınırlandırılmış sinüzoidal eğri uydurma, otokorelasyon
        │
        ▼
Çapraz-deney tutarlılık özellikleri (alpha+cdc15+cdc28 ortak sinyali)
        │
        ▼
Fisher G-test taban çizgisi (tekil + 4 deneyli kombine)  │  Train/Test ayrımı (%80/%20)
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
SHAP yorumlanabilirlik  │  ELU ile bağımsız doğrulama  │  Tekil-deney tanı analizi
        │
        ▼
Yeni gen keşfi (eşik=0.80) → Dubious ORF filtresi → GO zenginleştirme (g:Profiler)
```

## Proje Yapısı

```
periodic-genes-S/
├── data/
│   ├── raw/              # yeastCC'den çekilen 6 ham CSV + periyodik gen listesi
│   └── processed/        # işlenmiş özellik matrisleri, split'ler, OOF tahminler
├── scripts/
│   ├── r/
│   │   └── fetch_data.R                       # Bioconductor yeastCC'den veri çekme
│   └── python/
│       ├── 01_validate_data.py                # veri doğrulama
│       ├── 02_preprocess.py                   # deney hizalama + etiket oluşturma
│       ├── 03_feature_engineering.py          # istatistik + Lomb-Scargle + sinüs fit
│       ├── 04_merge_features.py               # çapraz-deney özellikleri
│       ├── 05_baseline_and_split.py           # Fisher G-test (tekil+kombine) + train/test ayrımı
│       ├── 06_train_models.py                 # 11 model, 5-fold CV
│       ├── 07_threshold_and_test.py           # eşik optimizasyonu + nihai test
│       ├── 08_shap_analysis.py                # SHAP yorumlanabilirlik
│       ├── 09_elu_only_validation.py          # bağımsız (ELU) doğrulama
│       ├── 10_single_experiment_diagnostic.py # tekil deney tanı analizi
│       ├── 11_diagnostic_plot.py              # zaman noktası sayısı vs AUC grafiği
│       ├── 12_novel_gene_discovery.py         # yeni aday tarama + dubious ORF filtresi
│       ├── 13_go_enrichment.py                # GO zenginleştirme (g:Profiler)
│       ├── 14_annotate_candidates.py          # aday genlerin isim/fonksiyon eşleştirmesi
│       └── 15_check_overlap_artifact.py       # örtüşen ORF artefaktı doğrulaması
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
python 15_check_overlap_artifact.py
```

Her script çıktısını `data/processed/`, `results/tables/`, `results/figures/` ve `models/` altına yazar; script'ler birbirinin çıktısına bağımlı olduğundan sıra önemlidir.

## Sınırlılıklar

- Bağımsız gold-standard listeleriyle (Pramila 2006, Rowicka 2007) dış doğrulama yapılmamıştır; elutriation-only doğrulama bunun kısmi bir ikamesidir.
- Modeller arası istatistiksel anlamlılık testi (5×2cv vb.) uygulanmamıştır.
- Tek `random_state` (42) kullanılmıştır; çoklu tohum tekrarı yoktur.
- Hiperparametreler manuel seçilmiştir, sistematik arama (grid/random search) yapılmamıştır.
- Yeni gen keşfi analizinde aday sayısı (n=19) istatistiksel güç açısından sınırlıdır; GO zenginleştirme sonucu hipotez üretici olarak değerlendirilmeli, deneysel doğrulama gerektirir.
- Yeni-aday eşiği (0.80) ile sınıflandırma performans eşiği (0.50) kasıtlı olarak farklıdır; bu ayrımın gerekçesi Yöntem bölümünde açıklanmıştır.

## Kaynakça

- Akıncı, Ö. *Classification of Periodic Genes in Yeast Gene Expression Data Using Machine Learning*. Yalova Üniversitesi, Bilgisayar Mühendisliği Yüksek Lisans Programı — bu projenin çıkış noktası olan ilk ders çalışması.
- Spellman, P. T., et al. (1998). Comprehensive identification of cell cycle–regulated genes of the yeast *Saccharomyces cerevisiae* by microarray hybridization. *Molecular Biology of the Cell*, 9(12), 3273–3297.
- Liew, A. W. C., et al. (2007). Spectral estimation in unevenly sampled space of periodically expressed microarray time series data. *BMC Bioinformatics*, 8(1), 137.
- Wu, W. S., & Li, W. H. (2008). Systematic identification of yeast cell cycle transcription factors using multiple data sources. *BMC Bioinformatics*, 9(1), 522.
- Cheng, C., et al. (2013). Identification of yeast cell cycle regulated genes based on genomic features. *BMC Systems Biology*, 7(1), 70.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Reimand, J., et al. (2007). g:Profiler — a web-based toolset for functional profiling of gene lists. *Nucleic Acids Research*, 35(suppl_2), W193–W200.

## Lisans

Bu proje akademik/eğitim amaçlıdır. Kullanılan veri seti (Spellman et al. 1998) Bioconductor `yeastCC` paketi aracılığıyla temin edilmiştir; orijinal veri lisansı ve atıf koşulları için ilgili yayına bakınız.
