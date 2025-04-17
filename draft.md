
---

# 📘 Project Report: Customer Churn Prediction Using Efficient Machine Learning

---

---

## 2. **Dataset Description**

- **Source**: E-Commerce customer dataset (Excel) 
- **Size**: 5630 records, 20 features + 1 target (`Churn`)
- **Types of Features**:
  - **Categorical**: `PreferredPaymentMode`, `Gender`, `MaritalStatus`, etc.
  - **Numerical**: `Tenure`, `OrderCount`, `CouponUsed`, etc.
  - **Target**: `Churn` (1 = Churned, 0 = Retained)

### 📋 Feature Table (Excerpt)
| Column                     | Description                                      |
|----------------------------|--------------------------------------------------|
| `CustomerID`               | Unique customer identifier                      |
| `Churn`                    | Whether the customer churned                    |
| `Tenure`                   | Months since joining                            |
| `CouponUsed`               | Number of coupons used                          |
| `OrderAmountHikeFromlastYear` | % increase in orders year-over-year         |

---

## 3. **Data Preprocessing**

- **Missing Values**: Filled using median for numeric features
- **Encoding**: Used `LabelEncoder` for categorical variables (tree-friendly)
- **Scaling**: Applied `StandardScaler` for models needing normalized inputs (e.g., Logistic Regression, MLP)
- **Feature Selection**: Removed high-correlation features where needed to avoid multicollinearity
- **Split Strategy**: 80/20 stratified split to preserve churn distribution

---

## 4. **Exploratory Data Analysis (EDA)**

📄 [View EDA Report](results/eda_report.pdf)

### Key Findings:
- **Churn Rate**: ~17% churn, confirming moderate imbalance
- **Correlations**:
  - `Tenure` negatively correlated with churn (new users churn more)
  - `OrderAmountHikeFromlastYear` showed strong positive correlation
- **Behavioral Patterns**:
  - Mobile users and single individuals churned more
  - COD and UPI users showed higher churn than credit card users

These insights guided feature selection and model choice.

---

## 5. **Modeling Strategy**

### ✅ Baseline Models
| Model               | Notes                                      |
|--------------------|--------------------------------------------|
| Logistic Regression | Interpretable, fast baseline               |
| Decision Tree       | Captures non-linear relationships          |

### ✅ Advanced Models
| Model            | Notes                                          |
|------------------|------------------------------------------------|
| Random Forest     | Stable ensemble, handles imbalance well       |
| XGBoost           | Boosting method, strong performance           |
| LightGBM          | Fast, scalable boosting alternative           |
| CatBoost          | Handles categorical features directly         |
| MLP Classifier    | Neural net for deep representation learning   |
| Stacked Ensemble  | Final model combining best base learners      |

---

## 6. **Model Evaluation**

📄 [View All Models Report](results/all_models_report.pdf)

| Model              | Accuracy | AUC   | F1 Score |
|-------------------|----------|-------|----------|
| Logistic Regression | 77%     | 0.86  | 0.54     |
| Decision Tree       | 79%     | 0.88  | 0.58     |
| Random Forest       | 95%     | 0.99  | 0.85     |
| XGBoost             | 97%     | 0.99  | 0.91     |
| LightGBM            | 96%     | 0.99  | 0.90     |
| CatBoost            | 89%     | 0.96  | 0.74     |
| MLP Classifier      | 96%     | 0.98  | 0.87     |
| **Stacked Ensemble**| **98%** | **1.00** | **0.94** |

---

## 7. ** Feature Importance**
- **Top Features**:
  - `Tenure`: Strongest negative churn predictor
  - `OrderAmountHikeFromlastYear`: High increases linked with churn
  - `Complain`, `SatisfactionScore`: Customer experience matters
- **SHAP / Feature Importance**:
  - Confirmed EDA insights
  - Helped detect potential leakage (e.g., `Complain` closely tied to label)
- **Action**:
  - Re-evaluated models without suspect features to verify robustness

---

## 8. **Model Reliability & Validation**

> AUC of 1.00 and 98% accuracy raises questions of generalization.

### Validations Performed:
- Stratified splitting
- Leakage audit (feature inspection)
- Evaluated multiple models with consistent metrics
- Noted features at risk of reflecting churn (e.g., post-churn behaviors like complaints)

**Conclusion**: Final model is high-performing, but care must be taken during deployment to exclude real-time-inaccessible features.

---

## 9. **Lessons Learned**

- Accuracy isn’t enough — **F1, ROC-AUC** and **confusion matrix** matter more for imbalanced problems.
- **Data leakage is real** — validate your assumptions before trusting the metrics.
- **Preprocessing order matters** — encode & scale **after** splitting the data.
- **Not all features are good features** — some encode outcomes, not predictors.

---

## 10. **Future Work**

- SHAP-based dashboard for business teams
- Incorporate time-based behavioral features (e.g., recency/frequency)
- Automate retraining pipeline
- Deploy via FastAPI + Streamlit for live scoring
- Rebalance dataset using SMOTE or similar

---

## 📂 Deliverables

| Artifact               | Description                                 |
|------------------------|---------------------------------------------|
| `main.py`              | Full pipeline (modular, end-to-end)         |
| `eda_report.pdf`       | Churn insights, patterns, and distributions |
| `all_models_report.pdf`| Evaluation metrics, ROC, CM, F1, AUC plots |
| `models/*.pkl`         | Saved ML models                             |
| `requirements.txt`     | Project dependencies                        |

---

## 📬 Contact

📧 prachi.barache@city.ac.uk  
🔗 GitHub Repository: [link-to-repo]  
🔗 LinkedIn: [your-link]

---

Would you like this formatted into a downloadable PDF or markdown file now?