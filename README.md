# 🛡️ Fraud Detection System
### End-to-end ML pipeline for detecting fraudulent transactions using LightGBM + SMOTE

---

## 📌 Key Results

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.9730** |
| Accuracy | **97%** |
| Precision (Fraud) | **0.95** |
| Recall (Fraud) | **1.00** |
| F1-Score (Fraud) | **0.97** |

---

## 📋 Project Overview

Fraud detection is a classic **imbalanced binary classification** problem —
fraudulent transactions represent only 8.74% of all records. A naive model
that predicts "legitimate" every time achieves 91% accuracy but catches zero
fraud cases. This project builds a robust pipeline that correctly identifies
fraudulent transactions with a **Recall of 1.00** — meaning no fraud case
is missed.

**Core challenges addressed:**
- Class imbalance (91.26% legitimate vs 8.74% fraudulent)
- Large-scale data (1,000,000 transaction records)
- Model deployment via interactive Streamlit web application

---

## 🗂️ Project Structure

```
Fraud_Detection_System/
├── fraud_detection.ipynb       ← Full analysis notebook
├── fraud_detection_app.py      ← Streamlit web application
├── fraud_detection_model.jb    ← Saved trained model (joblib)
├── sample_transactions.csv     ← Sample input data for app testing
├── requirements.txt            ← Python dependencies
└── README.md                   ← Project documentation
```

---

## 🔄 Methodology

### 1. Exploratory Data Analysis
- Class distribution analysis confirming 8.74% fraud rate
- Correlation heatmap identifying strongest fraud signals
- Statistical summary of all 7 transaction features

### 2. Data Preprocessing
- Feature/target separation (X = 7 predictors, y = fraud label)
- No feature scaling required — LightGBM is tree-based and scale-invariant
- Stratified train/test split preserving class ratios

### 3. Handling Class Imbalance — SMOTE
- **SMOTE (Synthetic Minority Oversampling Technique)** applied to
  training set only
- Generates synthetic fraud samples by interpolating between existing
  minority observations
- Applied **after** train/test split to prevent data leakage

### 4. Model — LightGBM Classifier
**LightGBM** (Light Gradient Boosting Machine) by Microsoft was selected for:
- Leaf-wise tree growth → faster convergence than XGBoost
- Histogram-based algorithm → lower memory on 1M+ rows
- Native `is_unbalance` parameter for imbalanced data handling

**Key hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `boosting_type` | gbdt | Gradient Boosted Decision Trees |
| `num_leaves` | 31 | Controls model complexity |
| `learning_rate` | 0.05 | Conservative — reduces overfitting |
| `n_estimators` | 200 | Number of boosting rounds |
| `is_unbalance` | True | Upweights minority class |

### 5. Evaluation
- Confusion matrix with visual heatmap
- ROC curve with AUC score
- Full classification report (precision, recall, F1 per class)
- Feature importance chart (split-based)

---

## 📊 Key Visualizations

| Visual | Insight |
|--------|---------|
| Class Distribution (pre/post SMOTE) | Confirms balancing effect |
| Correlation Heatmap | `ratio_to_median_purchase_price` most correlated with fraud |
| Confusion Matrix | Near-perfect separation of fraud vs legitimate |
| ROC Curve (AUC = 0.9730) | Strong discriminative ability |
| Feature Importance | Top fraud signals identified |

---

## 🌐 Streamlit Web Application

The trained model is deployed as an interactive web application built
with Streamlit, featuring:

- **Single transaction prediction** — input feature values manually
- **Batch processing** — upload a CSV for bulk fraud scoring
- **Risk scoring** — probability output per transaction
- **Model information** — performance metrics and feature descriptions

**To run the app locally:**
```bash
streamlit run fraud_detection_app.py
```

---

## 🗃️ Dataset

- **Source:** Synthetic credit card transaction dataset
- **Size:** 1,000,000 transactions
- **Fraud Rate:** 8.74% (87,403 fraudulent cases)
- **Features:** 7 transaction attributes

| Feature | Type | Description |
|---------|------|-------------|
| `distance_from_home` | Continuous | Distance from cardholder's home |
| `distance_from_last_transaction` | Continuous | Distance from previous transaction |
| `ratio_to_median_purchase_price` | Continuous | Ratio to typical spend amount |
| `repeat_retailer` | Binary | Transaction with known merchant |
| `used_chip` | Binary | EMV chip technology used |
| `used_pin_number` | Binary | PIN verification used |
| `online_order` | Binary | Remote/online transaction |

---

## 🛠️ Tools & Libraries

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-FF6600?style=flat&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logoColor=white)

| Library | Purpose |
|---------|---------|
| `lightgbm` | Gradient boosting classifier |
| `scikit-learn` | Preprocessing, metrics, evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `pandas / numpy` | Data manipulation |
| `matplotlib / seaborn` | Visualizations |
| `streamlit` | Web application deployment |
| `joblib` | Model serialization |

---

## ▶️ How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/otene-daniel/Fraud_Detection_System.git
cd Fraud_Detection_System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis notebook
jupyter notebook fraud_detection.ipynb

# 4. Launch the web application
streamlit run fraud_detection_app.py
```

---

**Author:** Daniel Otene
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/otene-daniel)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](www.linkedin.com/in/otene-daniel-441b45209)
