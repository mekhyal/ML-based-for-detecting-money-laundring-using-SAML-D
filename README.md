# ML-Based Detection of Money Laundering Using SAML-D

A data mining project applying supervised classification, unsupervised clustering, and regression techniques to detect Anti-Money Laundering (AML) patterns in large-scale synthetic financial transaction data.

---

## Dataset

**Synthetic Anti-Money Laundering Dataset (SAML-D)**  
Source: [Kaggle – berkanoztas/synthetic-transaction-monitoring-dataset-aml](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)  
License: CC-BY-NC-SA-4.0

| Property | Value |
|---|---|
| Total records | ~9.5 million |
| Features | 12 raw attributes |
| Laundering cases | 9,873 (~0.1%) |
| Normal cases | 9,494,979 (~99.9%) |

Raw features include: `Time`, `Date`, `Sender_account`, `Receiver_account`, `Amount`, `Payment_currency`, `Received_currency`, `Sender_bank_location`, `Receiver_bank_location`, `Payment_type`, `Is_laundering`, `Laundering_type`.

---

## Project Structure

```
├── Preprocessing_SAML_D.ipynb      # P2 – Data cleaning & feature engineering
├── MoneyLaundringDetection.ipynb   # P3 – Classification, Clustering & Regression
└── README.md
```

---

## Notebooks

### 1. `Preprocessing_SAML_D.ipynb` — P2: Corrected Preprocessing

This notebook prepares the raw SAML-D dataset for modeling with a strict **no-leakage** pipeline:

**Steps performed:**
- Load dataset via `kagglehub`
- Normalize and clean categorical columns (consistent casing, special character removal)
- Engineer grouped features to reduce cardinality:
  - `Sender/Receiver_bank_location` → `uk` / `international`
  - `Payment_currency` / `Received_currency` → regional groups (uk_pounds, europe, middle_east, asia, africa, americas)
- Log-transform transaction `Amount` (`Amount_log`) to reduce skewness
- Bin `Amount` into quartiles (`Amount_Q4`)
- Drop leakage-prone columns: raw identifiers, original ungrouped columns, and `Laundering_type`
- **Split before fitting** (70% train / 15% validation / 15% test, stratified)
- Fit imputation, scaling (`RobustScaler`), and one-hot encoding **on training data only**
- Transform validation and test sets with the fitted preprocessor
- Save outputs to `p2_outputs/` for direct use in P3

**Output files saved:**
```
p2_outputs/
├── X_train_processed.npz
├── X_val_processed.npz
├── X_test_processed.npz
├── y_train.csv
├── y_val.csv
├── y_test.csv
├── preprocessor.pkl
├── feature_names.json
└── preprocessing_metadata.json
```

**Final feature count after encoding:** 30 features  
**Split sizes:** Train: 6,653,396 | Val: 1,425,728 | Test: 1,425,728

---

### 2. `MoneyLaundringDetection.ipynb` — P3: Modeling

Loads the preprocessed outputs from P2 and applies three data mining tasks.

#### 2.1 Classification

Seven models are trained and evaluated on precision, recall, F1-score, and accuracy. Class imbalance (~0.1% positives) is addressed using **SMOTE** and a custom **probability threshold (0.3)** for applicable models.

| Model | Notes |
|---|---|
| Naive Bayes | Dense input; baseline probabilistic model |
| Decision Tree | Captures deterministic rule-like patterns |
| Logistic Regression | Stable linear baseline with `class_weight='balanced'` |
| Random Forest | Ensemble; 100 estimators |
| XGBoost + SMOTE | GPU-accelerated (`tree_method='hist'`), threshold = 0.3 |
| Gradient Boosting + SMOTE | Threshold = 0.3 |

Results are compiled into a summary DataFrame ranked by F1-score.

#### 2.2 Clustering (Unsupervised)

K-Means clustering is applied to group transaction behavior without using labels.

- **Elbow Method** and **Silhouette Score** used to select optimal `k = 6`
- Two clustering configurations compared:
  - **Full feature set** (includes rule-based indicators)
  - **Clean feature set** (excludes rule-based suspicious indicators for unbiased discovery)
- PCA (2 components) used for visualization
- Cluster summaries include: laundering rate, average log-amount, cash withdrawal rate, cross-border rate

#### 2.3 Regression

Multiple Linear Regression is applied to predict `Amount_log_scaled` as a continuous target.

- Split: 80% train / 10% validation / 10% test
- Metrics: SSE, RMSE, MAE, R²
- Diagnostics: Residuals vs. Predicted, Residual Distribution, Actual vs. Predicted plots

---

## Key Findings

**Classification:** Models like Decision Trees and Random Forests achieved near-perfect performance due to the dataset's rule-based label structure. This confirms correct pipeline implementation, though it limits insight into real-world predictive uncertainty.

**Clustering:** After removing rule-based features, laundering cases were distributed across clusters rather than concentrated in one, indicating that money laundering does not map to a single behavioral pattern. Clustering serves as an exploratory tool rather than a direct detector.

**Regression:** Strong fit with low RMSE and satisfactory R², showing that transaction amount is well-explained by payment type, currency, and location features. Regression complements the other methods by modeling behavior rather than detecting laundering directly.

---

## How to Run

### Prerequisites

```bash
pip install kagglehub pandas numpy scikit-learn scipy imbalanced-learn xgboost matplotlib seaborn joblib
```

### Steps

1. Open `Preprocessing_SAML_D.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells. This downloads the dataset from Kaggle and saves the processed files to `p2_outputs/`.

2. Open `MoneyLaundringDetection.ipynb` in Google Colab. Update `OUTPUT_PATH` to point to the `p2_outputs/` directory and run all cells.

> **Note:** XGBoost with `device='cuda'` requires a GPU runtime. In Colab, go to **Runtime → Change runtime type → GPU**.

---

## Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab
- **Libraries:** `scikit-learn`, `xgboost`, `imbalanced-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`, `kagglehub`
