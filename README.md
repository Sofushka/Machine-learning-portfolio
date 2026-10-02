# Data Science & Machine Learning Portfolio

Welcome! This repository serves as a showcase of my end-to-end data science projects, focusing on advanced tabular data pipelines, machine learning ensembles, and deep learning architectures across NLP and Computer Vision. 

Each project contains a complete production-grade pipeline: from exploratory data analysis (EDA) and robust handling of real-world data anomalies to explainable AI (XAI), transformer fine-tuning, and interactive deployment.

---

## 🛠️ Core Technical Stack

* **Languages:** Python, SQL
* **Deep Learning & NLP:** PyTorch, TensorFlow, Keras, Hugging Face Transformers, KerasTuner, Weights & Biases (WandB), spaCy
* **Computer Vision & Spatial AI:** Kornia, OpenCV, LoFTR, Geometric Verification (USAC_MAGSAC/RANSAC)
* **Tabular ML Frameworks:** Scikit-Learn, XGBoost, LightGBM, CatBoost, Optuna
* **Explainable AI & Analysis:** SHAP, Pandas, NumPy, VIF Analysis
* **Visualization & Dashboards:** Plotly (2D/3D), Streamlit, Matplotlib, Seaborn

---

## 📊 Projects Overview

| Project | Domain / Category | Key Tech Stack | Primary Performance / Metric |
| :--- | :--- | :--- | :--- |
| **[E-Commerce Review Analysis](#1-e-commerce-review-analysis--multi-task-rating-prediction)** | Deep Learning / NLP / Multi-Task | DeBERTa-v3, BiLSTM, KerasTuner, Focal Loss | **0.78 Leaderboard Score** |
| **[Mountain Named Entity Recognition](#2-mountain-named-entity-recognition-ner)** | NLP / Token Classification | BERT, Hugging Face Hub, PyTorch | **0.8997 Macro F1-Score** |
| **[Satellite Image Keypoint Matching](#3-satellite-image-keypoint-matching)** | Computer Vision / Remote Sensing | LoFTR, Kornia, USAC_MAGSAC | Robust seasonal alignment |
| **[Credit Risk & Default Prediction](#4-credit-risk--default-prediction-pipeline)** | Financial ML / Imbalanced Tabular | LightGBM, CatBoost, XGBoost, Optuna, SHAP | **0.847 ROC-AUC** (PR-Optimized) |
| **[USA Crime Rate Regression](#5-usa-crime-rate-regression-pipeline)** | Tabular Regression / Analytics | XGBoost, Scikit-Learn, VIF Analysis | **$R^2 = 0.64$, $\text{MAE} = 213$** |

---

## 📁 Featured Projects

### 1. E-Commerce Review Analysis & Multi-Task Rating Prediction
An end-to-end GPU-accelerated Deep Learning pipeline built to concurrently predict customer product ratings (1–5 Stars) and recommendation decisions (Binary: Yes/No) from unstructured review text and customer metadata.

* **Multi-Task Architecture:** Simultaneously optimizes multi-class star rating prediction using Categorical Focal Loss ($\gamma = 2.0$) and binary recommendation probability using Binary Cross-Entropy.
* **Class Imbalance Strategy:** Solved heavy 5-star rating skew via Focal Loss paired with class-weighted loss functions and hybrid over/under-sampling (`imblearn`).
* **Model Benchmarking:** Evaluated 3 distinct architectures on public benchmarks:

| Feature / Metric | Model 1: BoW + Dense ANN | Model 2: GloVe + CNN-BiLSTM | Model 3: DeBERTa-v3 Transformer |
| :--- | :--- | :--- | :--- |
| **Public Leaderboard Score** | 0.61 | 0.68 | **0.78 (Winner)** |
| **Text Feature Extraction** | TF-IDF (4,000 N-grams) | Pretrained GloVe (100d) | Subword Attention (`deberta-v3-small`) |
| **Text Preprocessing** | spaCy Lemmatization | Custom Regex Standardization | Metadata Text Linearization |
| **Optimization** | KerasTuner (Hyperband) | KerasTuner (Hyperband) | Two-Stage Fine-Tuning |

* **Tech Stack:** PyTorch, TensorFlow/Keras, Hugging Face Transformers, KerasTuner, Weights & Biases (WandB), spaCy, Pandas, NumPy.

---

### 2. Mountain Named Entity Recognition (NER)
🔗 **Hugging Face Model Hub:** [`TerentievaSof/bert-base-mountain-ner`](https://huggingface.co/TerentievaSof/bert-base-mountain-ner)

An automated token classification pipeline fine-tuned to extract mountain geographical entities (`B-MOUNTAIN`, `I-MOUNTAIN`) from unstructured English text.

* **Architecture:** Fine-tuned `dslim/bert-base-NER` using BIO (Beginning, Inside, Outside) tagging scheme.
* **Optimization & Training:** Trained with AdamW ($\text{learning\_rate} = 2\text{e-}5$, linear warmup ratio $0.1$) and evaluated using per-epoch macro F1-score with Early Stopping. Best weights automatically published to Hugging Face Hub.
* **Test Evaluation Results:**

| Metric | Score |
| :--- | :--- |
| **Precision** | 0.8930 |
| **Recall** | 0.9065 |
| **Macro F1-Score** | **0.8997** |

* **Tech Stack:** PyTorch, Hugging Face Transformers/Datasets, Scikit-Learn, Python.

---

### 3. Satellite Image Keypoint Matching
An end-to-end Deep Learning and Spatial AI pipeline for detecting keypoint correspondences between multi-temporal Sentinel-2 satellite imagery affected by heavy seasonal changes, illumination shifts, and vegetation growth.

* **Detector-Free Transformer Matching:** Utilizes **LoFTR** (Local Feature Transformer) via `kornia.feature` to leverage self and cross-attention mechanisms for dense keypoint extraction in low-texture and seasonal terrain.
* **Geometric Verification:** Applied **USAC_MAGSAC (RANSAC)** homography filtering to separate true correspondences (inliers) from spurious matches (outliers).
* **Data Pipeline:** Custom tiling pipeline processing 512x512 multi-temporal satellite image pairs.
* **Tech Stack:** PyTorch, Kornia, OpenCV, NumPy, Matplotlib.

---

### 4. Credit Risk & Default Prediction Pipeline
🔗 [View Live Interactive Report](#)

An end-to-end financial machine learning pipeline designed to forecast borrower default risk while resolving severe class imbalance and maintaining full model interpretability for credit underwriting.

* **Optimized Ensemble:** Built a high-performance ensemble blending LightGBM, CatBoost, and XGBoost, tuned over 100 trials via Optuna targeting PR-AUC.
* **Architecture-Aware Imputation:** Routed missing financial disclosures directly through tree-split algorithms in LightGBM/CatBoost to preserve non-disclosure risk signals rather than applying naive mean/median fills.
* **Leakage-Free Validation:** Strict 5-Fold Stratified Cross-Validation with out-of-fold target encoding and feature engineering.
* **Explainable AI (XAI):** Integrated **SHAP** workflows to break down individual borrower risk scores into transparent, quantifiable feature contributions.

| Model Architecture | ROC-AUC | Primary Optimization Metric |
| :--- | :--- | :--- |
| **Optimized Ensemble (LightGBM + CatBoost + XGBoost)** | **0.847** | Average Precision (PR-AUC) |

* **Tech Stack:** LightGBM, CatBoost, XGBoost, Scikit-Learn, Optuna, SHAP, Pandas, NumPy, Plotly.

---

### 5. USA Crime Rate Regression Pipeline
A socio-economic regression pipeline analyzing 2,000 regional records across the United States to model and predict local crime rates.

* **Multicollinearity Mitigation:** Utilized Variance Inflation Factor (VIF) and correlation matrices across demographic and law enforcement features to eliminate redundant signals and stabilize feature importance calculations.
* **Model Comparison:**

| Model Architecture | $R^2$ Score | Mean Absolute Error (MAE) |
| :--- | :--- | :--- |
| **XGBRegressor (Tuned)** | **0.64** | **213** |
| **RandomForestRegressor** | 0.62 | 235 |

* **Business Impact:** Output pipeline maps residual errors to highlight anomalous regions where crime rates deviate significantly from baseline socio-economic indicators.
* **Tech Stack:** XGBoost, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.

