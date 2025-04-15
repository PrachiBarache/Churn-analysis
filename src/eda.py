import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

def generate_eda_report(df, save_path="results/eda_report.pdf"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set(style="whitegrid")

    with PdfPages(save_path) as pdf:

        # 1. Target Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Churn', data=df, palette='pastel')
        plt.title("Churn Distribution")
        pdf.savefig()
        plt.close()

        # 2. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        numeric_df = df.select_dtypes(include='number')
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        pdf.savefig()
        plt.close()

        # 3. Numeric Feature Distributions
        for col in numeric_df.columns.drop('Churn', errors='ignore'):
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"{col} Distribution")
            pdf.savefig()
            plt.close()

        # 4. Categorical Features vs. Churn
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x=col, hue='Churn', palette='Set2')
            plt.title(f"Churn by {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"📊 EDA PDF report saved to {save_path}")
