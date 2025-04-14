"""
Exploratory Data Analysis for Customer Churn Prediction
Author: [Your Name]
Purpose: Understand structure, trends, and key patterns in the dataset
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Excel file from data folder
df = pd.read_excel("../data/E Commerce Dataset.xlsx", sheet_name="E Comm")

# Set plot styling
sns.set(style="whitegrid")

# 1. Basic Dataset Info
print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print("\n--- Column Data Types ---")
print(df.dtypes)
print("\n--- Missing Values ---")
print(df.isnull().sum().sort_values(ascending=False))

# 2. Target Variable - Churn Distribution
print("\n--- Churn Class Distribution ---")
print(df['Churn'].value_counts(normalize=True))

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title("Churn Distribution")
plt.xticks([0, 1], ['Retained', 'Churned'])
plt.ylabel("Count")
plt.show()

# 3. Distributions of Numeric Features
numeric_cols = df.select_dtypes(include='number').columns.drop(['CustomerID', 'Churn'])

for col in numeric_cols:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[col], kde=True, ax=ax[0], color="steelblue")
    sns.boxplot(x='Churn', y=col, data=df, ax=ax[1], palette="Set2")
    ax[0].set_title(f"Distribution of {col}")
    ax[1].set_title(f"{col} vs Churn")
    plt.tight_layout()
    plt.show()

# 4. Categorical Features vs. Churn
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='Churn', palette='pastel')
    plt.title(f"Churn by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 5. Correlation Matrix
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include='number')  # Only numeric columns
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 6. Segment-Based Churn Rates (Business Insights)
segment_cols = ['CityTier', 'Gender', 'PreferredPaymentMode']

for col in segment_cols:
    plt.figure(figsize=(6, 4))
    churn_rate = df.groupby(col)['Churn'].mean().sort_values()
    churn_rate.plot(kind='barh', color='tomato')
    plt.title(f"Churn Rate by {col}")
    plt.xlabel("Churn Rate")
    plt.tight_layout()
    plt.show()

print("\n✅ EDA Complete. You can now use these insights for feature engineering.")
