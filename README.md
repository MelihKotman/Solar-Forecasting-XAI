# Regime-Aware Adaptive Signal Filtering for Transformer-Based Solar Irradiance Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![PyWavelets](https://img.shields.io/badge/PyWavelets-00599C?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-blueviolet?style=for-the-badge)
![pvlib](https://img.shields.io/badge/pvlib-Solar%20Modeling-yellow?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

##  Overview

This project proposes an end-to-end pipeline for **Global Horizontal Irradiance (GHI) forecasting** by combining

- 🌤️ Regime-aware adaptive signal filtering
- 🤖 Transformer-based deep learning
- ☀️ Astronomical feature engineering
- 📈 Explainable AI (SHAP)
- 🌍 Out-of-Distribution (OOD) generalization analysis

The pipeline automatically identifies daily weather regimes and applies the most appropriate signal processing technique before training Transformer models.

---

#  Project Structure

```
.
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_03_eda_analysis.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_results.ipynb
│   ├── 06_xai_evaluation.ipynb
│   └── 07_ood_generalization.ipynb
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── filters.py
│   └── utils.py
│
├── models/
│
├── requirements.txt
└── README.md
```

---

#  Pipeline

![Pipeline](outputs/plots/pipeline.drawio.png)

---

# 🔬 Signal Processing

The preprocessing stage applies different filters depending on the detected weather regime.

| Regime | Filter |
|---------|--------|
| ☀️ Sunny | Savitzky-Golay |
| ☁️ Cloudy | Moving Average |
| ⛈️ Chaotic | Wavelet Transform |

---

#  Models

Current deep learning models include

- Custom PatchTST
- Custom Autoformer

Future models may include

- Informer
- iTransformer
- DLinear
- TimesNet

---

#  Explainability

The project includes

- SHAP feature importance
- Temporal explanations
- Model interpretation
- Feature contribution analysis

---

#  OOD Evaluation

Models are evaluated on multiple unseen geographical locations to measure

- Generalization
- Robustness
- Climate transferability

---

#  Installation

```bash
git clone <repository_url>

cd repository

pip install -r requirements.txt
```

---

# ▶ Workflow

Run notebooks sequentially.

```
01_data_acquisition.ipynb
        ↓
02_03_eda_analysis.ipynb
        ↓
04_model_training.ipynb
        ↓
05_model_results.ipynb
        ↓
06_xai_evaluation.ipynb
        ↓
07_ood_generalization.ipynb
```

---

#  Main Libraries

- PyTorch
- NumPy
- Pandas
- SciPy
- PyWavelets
- Scikit-Learn
- pvlib
- SHAP
- Matplotlib
- Jupyter Notebook

---

#  Citation

If you use this project in your research, please cite the corresponding publication.

---

#  Author

**Melih Yiğit Kotman**

Department of Computer Engineering

Bolu Abant İzzet Baysal University