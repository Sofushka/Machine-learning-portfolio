# E-Commerce Review Analysis & Multi-Task Rating Prediction

An end-to-end GPU-accelerated Deep Learning pipeline designed to predict customer product ratings (**1–5 Stars**) and recommendation decisions (**Binary: Yes/No**) from e-commerce text reviews and customer metadata.

This repository implements and compares three distinct deep learning architectures, ranging from feature-engineered baseline neural networks to state-of-the-art transformer fine-tuning.

---

## Project Overview

E-commerce review datasets present unique challenges: extreme class imbalance (heavily skewed toward 5-star ratings), noisy unstructured text, and high-cardinality metadata. This project addresses these challenges using a **Multi-Task Learning (MTL)** framework that simultaneously optimizes two complementary targets:

1. **`Rating`**: Multi-class classification (5 classes: 1 to 5 stars) evaluated using **Categorical Focal Loss** ($\gamma = 2.0$) and class-weighted cross-entropy to handle imbalanced distributions.
2. **`Recommended`**: Binary classification (0 or 1) evaluated using **Binary Cross-Entropy**.

---

## Architecture & Leaderboard Benchmark

| Feature / Metric | Model 1: BoW + Dense ANN | Model 2: GloVe + CNN-BiLSTM | Model 3: DeBERTa-v3 Transformer |
| :--- | :--- | :--- | :--- |
| **Public Leaderboard Score** | **0.61** | **0.68** | **0.78 (Winner)** |
| **Text Feature Extraction** | TF-IDF (4,000 N-grams) | Pretrained GloVe (100d) | Subword Attention (`deberta-v3-small`) |
| **Text Preprocessing** | spaCy Lemmatization (Negation-preserved) | Custom Regex Standardization | Metadata Text Linearization |
| **Metadata Handling** | One-Hot Encoding + Dense Concatenation | Entity Embedding Layers | Textual Prompt Linearization |
| **Resampling Strategy** | Class Weights in Loss Function | Hybrid Over/Under-Sampling (`imblearn`) | Class Weights + Focal Loss |
| **Optimization** | KerasTuner (Hyperband Search) | KerasTuner (Hyperband Search) | Two-Stage Fine-Tuning (Warmup $\rightarrow$ Unfreeze) |
| **Pooling Mechanism** | N/A (Dense Multi-Layer) | Global MaxPooling 1D | Masked Token Mean Pooling |

---

## Key Results & Project Conclusions

### Key Achievements
* **Transformer Dominance (DeBERTa-v3):** Achieved top performance on the competition benchmark with a **Public Score of 0.78** (Weighted Accuracy ~70%, F1 = 0.70).
* **Unified Multi-Task Learning:** Successfully built a single neural network architecture capable of concurrently predicting both granular star ratings (1–5) and binary purchase recommendations (0/1).
* **Handling Class Imbalance:** Combining **Categorical Focal Loss** with balanced class weights significantly improved classification performance on minority classes (1–3 star ratings).

### Engineering & Architecture Lessons
* **Accuracy vs. Inference Speed:** BiLSTM trained nearly 10x faster than DeBERTa while maintaining competitive performance (0.68 vs 0.78 public score). This makes BiLSTM an optimal candidate for production deployment in resource-constrained environments.
* **Data Quality & Preprocessing:** Domain-specific noise reduction, selecting sequence lengths based on empirical token distributions (P95 = 110/153), and log-transforming right-skewed numerical features (like feedback counts) proved critical for model convergence.
* **MLOps Best Practices:** Leveraging **KerasTuner** for automated hyperparameter tuning and **Weights & Biases (WandB)** for metric logging ensured full reproducibility and accelerated experimentation.

### Core Skills & Techniques Demonstrated
* **ETL & Data Engineering:** Imputation, categorical text linearization, sampling strategies via `imblearn`, and input data pipeline construction.
* **Deep Learning & Applied NLP:** Hands-on benchmarking of Bag-of-Words, static GloVe embeddings, BiLSTM/Conv1D hybrid layers, and Hugging Face Transformers.
* **Model Evaluation:** Diagnostic analysis via normalized confusion matrices, PR-AUC, F1-scores, and error profiling.

### Future Roadmap & Enhancements
* **Explainable AI (SHAP/LIME):** Integrate SHAP value visualizations to highlight exact keyword markers driving positive and negative predictions.
* **Production API:** Containerize the DeBERTa model using **Docker** and serve predictions via a **FastAPI** REST endpoint.
* **Real-time Monitoring:** Implement tracking pipelines to detect **Data Drift** and **Model Drift** in live e-commerce streams.
