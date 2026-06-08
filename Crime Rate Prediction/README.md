Crime Rate Prediction | Python (Scikit-learn, XGBoost) 

• Analyzed 2000 socio-economic and law enforcement records in the USA (1990). 

• Performed data cleaning and feature engineering, identified key factors of crime rates (% of children 
born out of wedlock; % of two-parent families; % of White population). 

• Trained models: RandomForest (R² = 0.62, MAE = 235) and XGBRegressor (R² = 0.64, MAE = 213). 

• Results can help identify high-risk areas for preventive measures. 


# USA Crime Rate Regression Pipeline

An end-to-end predictive modeling pipeline built to analyze and forecast regional crime rates across the United States using socio-economic and law enforcement metrics. This project focuses on handling highly correlated features, robust data preprocessing, and optimizing ensemble models for precise regression performance.

---

## 🛠️ Tech Stack & Key Concepts
*   **Modeling Frameworks:** XGBoost (`XGBRegressor`), Scikit-Learn (`RandomForestRegressor`)
*   **Data Stack:** Pandas, NumPy, Matplotlib, Seaborn
*   **Core Concepts:** Tabular Regression, Multicollinearity Mitigation, Advanced Feature Engineering, Residual Analysis

---

## 🚀 Key Engineering & Pipeline Steps

### 1. Data Preprocessing & Feature Engineering
*   **Data Integrity:** Cleaned and imputed missing values across 2,000 socio-economic and law enforcement records. 
*   **Multicollinearity Handling:** Analyzed highly correlated variables (such as demographic indicators and family structures) using Variance Inflation Factor (VIF) and correlation matrices to prevent model overfitting and ensure reliable feature importances.
*   **Feature Extraction:** Engineered interaction features combining community infrastructure metrics and law enforcement coverage to maximize predictive signal.

### 2. Model Training & Optimization
Evaluated multiple tree-based ensembles to handle non-linear relationships in the data. Models were tuned via cross-validation and evaluated using R-squared ($R^2$) and Mean Absolute Error (MAE):

| Model Architecture | $R^2$ Score | Mean Absolute Error (MAE) |
| :--- | :---: | :---: |
| **XGBRegressor** (Tuned) | **0.64** | **213** |
| **RandomForestRegressor** | 0.62 | 235 |

### 3. Model Interpretability & Business Impact
*   **Key Predictors:** Isolated the highest-impact socio-economic features contributing to crime variances (including regional family-structure distributions and community demographic metrics).
*   **Actionable Insights:** Designed the output pipeline to map out residual errors, allowing stakeholders to easily identify high-risk regions where unexpected spikes in crime rate defy average socio-economic baselines.

---

## 📂 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the complete pipeline notebook or main script: `python src/train.py`
