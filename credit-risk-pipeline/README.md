# Credit Risk & Default Prediction Pipeline

🔗 [Click Here to View the Live Interactive Report](https://sofushka.github.io/Machine-learning-portfolio/credit-risk-pipeline/Credit_Risk_And_Default_Prediction_Project.html)

An end-to-end, production-ready machine learning pipeline built to predict borrower default risk. This project tackles the core challenges inherent in credit scoring datasets: extreme class imbalance, missing financial disclosures, and the critical requirement for model transparency (Explainable AI) in financial decision-making.
 
---

## 🛠️ Tech Stack & Advanced Concepts
*   **Modeling Frameworks:** LightGBM, CatBoost, XGBoost, Scikit-Learn
*   **Hyperparameter Tuning:** Optuna
*   **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
*   **Data Stack:** Pandas, NumPy, Plotly, Matplotlib
*   **Core Methodologies:** Stratified K-Fold Cross-Validation, Feature Engineering, Precision-Recall Optimization, Advanced Missing Value (`NaN`) Strategy

---

## 🚀 Key Engineering Challenges & Solutions

### 1. Robust Imbalanced Data Strategy
*   **The Problem:** Credit default events represent a tiny fraction of total loans (highly skewed target variable). Standard metrics like accuracy or ROC-AUC give a false sense of security while missing actual high-risk defaults.
*   **The Solution:** Structured the training objective to directly optimize for **Average Precision (PR-AUC)**. Adjusted internal class-weight matrices across ensemble models to aggressively penalize false negatives (missed defaults) while maintaining an optimal precision-recall trade-off for business utility.

### 2. Missing Value (`NaN`) Strategy for Tree-Ensembles
*   **The Problem:** Financial ratios (e.g., debt-to-income) often contain missing values that carry distinct behavioral signals rather than just random omissions.
*   **The Solution:** Designed an architecture-aware imputation strategy. For `LightGBM` and `CatBoost`, missing entries were routed dynamically during split-finding to capture the inherent risk signaling of non-disclosures, avoiding naive mean/median fills that dilute predictive signals.

### 3. Leakage-Free Validation & Feature Engineering
*   **Validation:** Implemented a rigorous **5-Fold Stratified Cross-Validation** scheme. 
*   **Feature Engineering:** Developed robust target encoding for high-cardinality categorical features and computed interactive risk ratios (e.g., payment-to-income volatility). All data transformations and encodings were fit strictly within each cross-validation fold to prevent data leakage and guarantee stable out-of-sample generalization.

### 4. Explainable AI (XAI) & Model Interpretability
*   **The Problem:** Modern gradient boosting ensembles are "black boxes," making them legally and operationally difficult to deploy in credit underwriting.
*   **The Solution:** Integrated **SHAP** workflows into the evaluation pipeline. This provides absolute mathematical clarity regarding feature importance, breaking down individual borrower risk scores into transparent, human-readable contributions (e.g., quantifying exactly how an increased debt-to-income ratio shifts the prediction toward a default state).

---

## 📊 Model Performance Evaluation

Models were tuned via automated Bayesian optimization using **Optuna** over 100 trials, prioritizing Average Precision:

| Model Architecture | ROC-AUC | 
| :--- | :---: |
| **Ensemble(LightGBM, CatBoost, XGBoost)** (Optimized) | **0.847** | 
