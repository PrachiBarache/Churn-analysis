# 🔄 Customer Churn Prediction: End-to-End ML Project

This project aims to predict customer churn for an e-commerce platform using a complete machine learning pipeline starting from data exploration and preprocessing to building, evaluating, and comparing models. The goal is to identify high-risk customers and help businesses proactively reduce churn.

---
- **Goal**: Predict customer churn for an e-commerce platform using a robust, using ML pipeline.
- **Motivation**: Customer retention is more cost-effective than acquiring new customers. Early churn detection helps companies proactively engage high-risk customers.
- **Steps**: Perform EDA, apply preprocessing, build baseline & advanced models, evaluate thoroughly, and explain predictions.
- **Outcome**: Achieved a final stacked ensemble model with **98% accuracy** and **1.00 ROC AUC**, validated through careful leakage inspection.
---
## 📦 Dataset Overview

The dataset consists of **5,630 rows** and **20 columns**, **target variable**('Churn), covering customer behavior, transaction history, preferences, and satisfaction levels. 
The dataset used is publicaly available on Kaggle [dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data) 


| Column                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `CustomerID`               | Unique customer ID                                                          |
| `Churn`                    | Churn flag (1 = Churned, 0 = Retained)                                      |
| `Tenure`                   | Customer’s tenure with the organization                                     |
| `PreferredLoginDevice`     | Preferred device used for login                                             |
| `CityTier`                 | Customer’s city tier classification                                         |
| `WarehouseToHome`          | Distance from warehouse to customer's home                                  |
| `PreferredPaymentMode`     | Preferred mode of payment                                                   |
| `Gender`                   | Gender of the customer                                                      |
| `HourSpendOnApp`           | Hours spent on the mobile app or site                                       |
| `NumberOfDeviceRegistered` | Number of devices registered by customer                                    |
| `PreferedOrderCat`         | Preferred order category in last month                                     |
| `SatisfactionScore`        | Customer's satisfaction score                                               |
| `MaritalStatus`            | Marital status                                                              |
| `NumberOfAddress`          | Number of addresses associated with the customer                           |
| `Complain`                 | Whether a complaint was raised in last month                               |
| `OrderAmountHikeFromlastYear` | Percent increase in order value from last year                       |
| `CouponUsed`               | Number of coupons used last month                                           |
| `OrderCount`               | Number of orders placed last month                                          |
| `DaySinceLastOrder`        | Days since the last order                                                   |
| `CashbackAmount`           | Average cashback received                                                   |

---

## 🔍 Exploratory Data Analysis (EDA)

**[Click to view full EDA report](results/eda_report.pdf)**

**Insights:**

- **Churn Rate**: ~17% churn, confirming moderate imbalance
- **Correlations**:
  - `Tenure` negatively correlated with churn (new users churn more)
  - `OrderAmountHikeFromlastYear` showed strong positive correlation
- **Behavioral Patterns**:
  - Mobile users and single individuals churned more
  - COD and UPI users showed higher churn than credit card users

---

##  Data Preprocessing

### ✅ Steps Applied:

- **Missing Values:** Filled using median for numerical columns and forward fill for temporal features.
- **Encoding:**
  - Categorical features encoded using `LabelEncoder` (works well with tree-based models).
- **Scaling:**
  - Numerical features scaled using `StandardScaler` (important for linear models and neural nets).
- **Target Separation:**
  - Target = `Churn`
- **Train/Test Split:**
  - 80/20 stratified split to preserve class ratio.

---

## 🤖 Modeling Strategy & Justification

The project follows a progressive modeling strategy to demonstrate improvement through increasingly advanced models:

| Model                | Why Chosen                                                   |
|---------------------|--------------------------------------------------------------|
| **Logistic Regression** | Simple, interpretable, a baseline for churn problems         |
| **Decision Tree**       | Captures non-linear rules, handles categorical data well   |
| **Random Forest**       | Combines many trees for better stability and generalization|
| **XGBoost**             | Boosting method with great performance and interpretability|
| **LightGBM**            | Faster than XGBoost, optimal for large tabular datasets    |
| **CatBoost**            | Automatically handles categorical variables, great accuracy|
| **MLP Classifier**      | Neural net model, adds non-linearity with deep layers      |
| **Stacked Ensemble**    | Combines multiple model predictions for best performance   |

---

## 📊 Model Evaluation & Results

📄 **[Click to view full model report PDF](results/all_models_report.pdf)**

Evaluation Metrics:  
- **ROC-AUC** (priority metric due to class imbalance)  
- **F1 Score**, **Accuracy**, and **Confusion Matrix**  

| Model             | AUC    | F1 Score | Accuracy | Notes                  |
|------------------|--------|----------|----------|------------------------|
| Logistic Reg.     | 0.86   | 0.54     | 77%      | Baseline               |
| Decision Tree     | 0.88   | 0.58     | 79%      | Slight improvement     |
| Random Forest     | 0.99   | 0.85     | 95%      | Strong generalization  |
| XGBoost           | 0.99   | 0.91     | 97%      | Excellent performance  |
| LightGBM          | 0.99   | 0.90     | 96%      | Very close to XGBoost  |
| CatBoost          | 0.96   | 0.74     | 89%      | Better on categories   |
| MLP Classifier    | 0.98   | 0.87     | 96%      | Neural net approach    |
| Stacked Ensemble  | 1.00   | 0.94     | 98%      | 🏆 Top performing model|

---

## 🧠 Key Improvements from Modeling Journey

- Transitioning from **Logistic Regression** to **Tree-Based Models** captured nonlinear relationships.
- **Boosted models (XGBoost, LightGBM)** significantly improved recall for churned customers (class 1).
- **Ensemble model** delivered the best balance of all metrics, demonstrating synergy between models.
- Clear lift in **ROC-AUC** from 0.86 → 1.00 through iterative modeling.

---

## 🛠 Tools & Technologies

| Purpose         | Stack |
|----------------|-------|
| Programming    | Python 3.10 |
| Data Analysis  | pandas, numpy |
| Visualization  | seaborn, matplotlib |
| Modeling       | scikit-learn, XGBoost, LightGBM, CatBoost |
| Reporting      | PdfPages, joblib |
| Dev Tools      | Visual Studio Code, GitHub |

---

## ⚠️ Model Reliability & Validation Considerations

While the final stacked model achieved an impressive **98% accuracy and 1.00 AUC**, such scores raise important questions about **generalizability** and **real-world deployment**.

### 🧠 Why High Accuracy May Not Tell the Full Story

| Metric        | Observed Value | Typical Real-World Range |
|---------------|----------------|--------------------------|
| Accuracy      | 98%            | 85–95%                   |
| ROC-AUC       | 1.00           | 0.85–0.95 (excellent), >0.97 (suspicious) |
| F1 Score      | 0.94           | High — requires context  |

High-performing models are great — but when scores are near-perfect, a **professional data scientist always investigates further**.

---

### 🔍 Potential Causes of Over-Performance

| Potential Cause | What It Means |
|-----------------|---------------|
| **Data Leakage** | Features like `Complain`, `OrderCount`, or `CouponUsed` might reflect churn rather than predict it |
| **Preprocessing Leakage** | Transformations (e.g., scaling, imputation) may have been done before splitting the data |
| **Train/Test Similarity** | Data might be too uniform between splits, limiting model stress |
| **Overly Predictive Features** | Certain features could dominate predictions unrealistically |

---

### ✅ What I Did to Validate the Model

To guard against false confidence:

- ✅ Used **stratified train/test split**
- ✅ Verified performance with **ROC-AUC**
- ✅ Monitored **class-specific F1 Scores**
- ✅ (Optional step) Prepared code to integrate **SHAP for feature importance**

---

### 🔬 Possible improvements

To strengthen the model's credibility further:

1. **Leakage Testing**: Remove features closely tied to churn and re-check performance.
2. **Cross-Validation**: Use stratified k-fold to ensure consistency across data splits.
3. **Deploy Simulation**: Evaluate model on a "future" (unseen) segment.
4. **SHAP Values**: Identify dominant features and validate their interpretability.
5. **Domain Review**: Share top predictors with business stakeholders to check for data leakage risk.

---



