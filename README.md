## Proje Adı: Solar-XAI-Forecasting

### Özet: K-Means tabanlı senaryo ayrıştırma ve Autoformer mimarisi ile güneş enerjisi tahmini.

### Yenilik: Geçiş dönemlerine özel adaptif sinyal filtreleme entegrasyonu.

`Kurulum: pip install -r requirements.txt`


### 📂 Proje Klasör Yapısı
```
solar-forecasting-xai/
├── data/                         # Veri yönetimi
│   ├── raw/                      # NREL'den gelen ham veriler (Git-ignored)
│   └── processed/                # Temizlenmiş ve kümelenmiş veriler
├── notebooks/                    # Deney alanı (Jupyter Notebooks)
│   ├── 01_data_acquisition.ipynb # API ile veri çekme
│   ├── 02_03_eda_analysis.ipynb  # Keşifsel veri analizi ve Clustering
│   ├── 04_model_training.ipynb   # Autoformer/PatchTST eğitimleri
│   └── 05_xai_evaluation.ipynb   # SHAP analizleri ve raporlama
├── src/                          # Projenin motoru (Python Modülleri)
│   ├── __init__.py
│   ├── data_loader.py            # Veri okuma ve API işlemleri
│   ├── filters.py                # Farklı filtre fonksiyonu
│   ├── models/                   # Derin öğrenme mimarileri
│   │   ├── __init__.py
│   │   ├── autoformer.py
│   │   └── patchtst.py
│   └── utils.py                  # Yardımcı fonksiyonlar (Plot, metrics)
├── outputs/                      # Çıktılar
│   ├── models/                   # Eğitilmiş .pth / .h5 dosyaları
│   ├── plots/                    # Grafik çıktıları
│   └── reports/                  # CSV/Excel raporları
├── .gitignore                    # Git dışı bırakılacaklar
├── requirements.txt              # Kütüphane listesi
└── README.md                     # Proje dokümantasyonu
```