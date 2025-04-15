# 🔄 Customer Churn Prediction: End-to-End ML Project

This project aims to predict customer churn for an e-commerce platform using a complete machine learning pipeline starting from data exploration and preprocessing to building, evaluating, and comparing models. The goal is to identify high-risk customers and help businesses proactively reduce churn.

---
## 📦 Dataset Overview

The dataset consists of **5,630 rows** and **20 columns**, covering customer behavior, transaction history, preferences, and satisfaction levels.

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

📄 **[Click to view full EDA report](results/eda_report.pdf)**

**Insights:**

- **Churn Distribution:** Only ~17% of users churned, indicating moderate class imbalance.
- **Tenure:** New customers (<6 months) are more likely to churn.
- **Complain & SatisfactionScore:** Low satisfaction and recent complaints correlate with churn.
- **Preferred Categories:** Customers who prefer 'Mobile' or 'Fashion' are more likely to churn.
- **Correlations:**
  - `OrderAmountHikeFromlastYear` has moderate positive correlation with churn.
  - `Tenure` has a strong negative correlation with churn.

---

## 🧹 Data Preprocessing

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
