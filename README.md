## Proje Adı: Solar-XAI-Forecasting

### Özet: K-Means tabanlı senaryo ayrıştırma ve Autoformer mimarisi ile güneş enerjisi tahmini.

### Yenilik: Geçiş dönemlerine özel adaptif sinyal filtreleme entegrasyonu.

`Kurulum: pip install -r requirements.txt`

## Proje Klasör Yapısı:
`
solar-forecasting-xai/
├── data/                   # Veri yönetimi
│   ├── raw/                # NREL'den gelen ham CSV/Parquet dosyaları (Git'e eklenmez)
│   └── processed/          # Temizlenmiş, filtrelenmiş ve kümelenmiş veriler
├── notebooks/              # Deney alanı (Jupyter Notebooks)
│   ├── 01_data_acquisition.ipynb   # API ile veri çekme
│   ├── 02_eda_analysis.ipynb       # Keşifsel veri analizi
│   ├── 03_clustering_adaptive.ipynb # K-Means ve Adaptif Filtreleme testleri
│   ├── 04_model_training.ipynb     # Autoformer/PatchTST eğitimleri
│   └── 05_xai_evaluation.ipynb      # SHAP analizleri ve raporlama
├── src/                    # Projenin motoru (Python Modülleri)
│   ├── __init__.py
│   ├── data_loader.py      # Veri okuma ve API işlemleri
│   ├── filters.py          # 12 farklı filtre (Wavelet, Hampel, vb.) fonksiyonları
│   ├── clustering.py       # K-Means senaryo ayırma kodları
│   ├── models/             # Derin öğrenme mimarileri
│   │   ├── __init__.py
│   │   ├── autoformer.py
│   │   └── patchtst.py
│   └── utils.py            # Yardımcı fonksiyonlar (Plotting, metrics)
├── outputs/                # Çıktılar
│   ├── models/             # Eğitilmiş .pth veya .h5 dosyaları
│   ├── plots/              # Makale için kaydedilen grafikler
│   └── reports/            # Tahmin sonuçları (CSV/Excel)
├── config/                 # Parametre yönetimi
│   └── config.yaml         # API anahtarları, model hiperparametreleri
├── .gitignore              # GitHub'a gönderilmeyecek dosyalar
├── requirements.txt        # Kütüphane listesi
└── README.md               # Proje tanıtımı ve kurulum rehberi
`