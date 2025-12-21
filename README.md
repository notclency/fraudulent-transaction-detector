# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

A high-precision machine learning classifier designed to identify fraudulent transactions in highly imbalanced datasets. This project explores the trade-offs between **Random Undersampling** vs. **SMOTE Oversampling** and determines the optimal strategy for minimizing financial loss and customer friction.

**[🌐 View Live Analysis Dashboard](https://clencytabe.vercel.app/projects/fraud-detection)**

---

## 🚀 Key Features

### 1. Handling Extreme Imbalance (0.17% Fraud)
The dataset contains **284,807 transactions** but only **492 frauds**. A standard model would achieve 99.8% accuracy by simply guessing "Legit" every time. This project solves that by:
- **Resampling:** Comparing Random Undersampling (removing legit cases) vs. SMOTE (generating synthetic fraud cases).
- **Metric Focus:** Optimizing for **Precision-Recall (AUPRC)** rather than raw Accuracy.

### 2. Advanced Visualizations
- **Dynamic Confusion Matrices:** Visualizing the cost of False Positives (Customer Friction) vs. False Negatives (Financial Loss).
- **Correlation Heatmaps:** Showing how balancing the dataset reveals hidden feature correlations (specifically `V14`, `V12`, and `V10`) that are invisible in the imbalanced data.

### 3. Model Benchmarking
Compared performance across multiple classifiers:
- **Logistic Regression** (The Winner)
- **K-Nearest Neighbors (KNN)**
- **Support Vector Classifier (SVC)**
- **Decision Tree**

---

## 📊 The Winning Strategy

**Winner:** `Logistic Regression` + `SMOTE`

Implementing **SMOTE** on our imbalanced dataset helped us fix the label imbalance. However, the Neural Network on the oversampled data sometimes predicted fewer correct fraud transactions than the undersampled model.

Crucially, the **undersampled model misclassified a large number of non-fraud transactions as fraud**. In a real-world scenario, this would mean blocking legitimate cards, leading to customer complaints and financial loss.

Therefore, **Logistic Regression with SMOTE** remains the superior choice. It significantly reduced False Positives (from **1,715** down to **12**), balancing risk mitigation with a superior customer experience.

---

## 📂 Dataset Access

Due to the file size (>150MB) and licensing, the raw data is not hosted in this repository.

**You can download it directly from Kaggle:**
👉 **[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)**

### Setup Instructions
1. Download `creditcard.csv` from the link above.
2. Place it in the `data/` folder of this project.
3. Run the notebook.

---

## 🛠️ Tech Stack

- **Data Processing:** Pandas, NumPy
- **Scaling:** RobustScaler (to handle outliers in transaction amounts)
- **Sampling:** Imbalanced-Learn (SMOTE, NearMiss)
- **Modeling:** Scikit-Learn (LogisticRegression, SVC, KNeighbors, DecisionTree)
- **Visualization:** Seaborn, Matplotlib

---

## 📸 Results Snapshot

| Metric | Undersampling | SMOTE (Winner) |
| :--- | :---: | :---: |
| **Accuracy** | 94.2% | **99.9%** |
| **False Positives** | 1,715 (High Friction) | **12 (Low Friction)** |
| **Recall (Fraud)** | 91.0% | **85.0%** |
| **Verdict** | Too risky for CX | **Production Ready** |

---

## 📬 Contact
**Clency Tabe**  
Data Science & Computer Science Student  
[LinkedIn](https://linkedin.com/in/clency-tabe) | [Portfolio](https://clencytabe.com)
