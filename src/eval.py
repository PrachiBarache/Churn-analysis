import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from matplotlib.backends.backend_pdf import PdfPages

def evaluate_model_pdf(model, X_test, y_test, model_name="Model", pdf_path="results/model_report.pdf"):
    """
    Evaluate model and save confusion matrix, ROC curve, and report to a PDF page.
    """
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=False)

    # Create PDF directory
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    # Open PDF in append mode
    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        pdf.savefig()
        plt.close()

        # --- Page 2: ROC Curve ---
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve")
        plt.legend()
        pdf.savefig()
        plt.close()

        # --- Page 3: Classification Report (as image) ---
        plt.figure(figsize=(6, 4))
        plt.axis('off')
        plt.text(0, 1, f"{model_name} - Classification Report\n\n{report}", fontsize=10, family='monospace')
        pdf.savefig()
        plt.close()

    print(f"✅ {model_name} evaluation pages added to PDF → {pdf_path}")
