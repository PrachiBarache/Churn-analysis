import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

def evaluate_model(model, X_test, y_test, model_name, pdf):
     
    #Evaluate a trained model and save results to a shared multi-page PDF.
     
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    # Page 1: Confusion Matrix
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

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", linewidth=2, color='navy')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
    plt.title(f"{model_name} - ROC Curve", fontsize=14)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='lower right', fontsize=10)

    #  Ensure clean save layout
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')  # 🔧 fixes overlap/overflow
    plt.close()

    # --- Page 3: Classification Report ---
    plt.figure(figsize=(6, 4))
    plt.axis('off')
    plt.text(0, 1, f"{model_name} - Classification Report\n\n{report}",
            fontsize=10, family='monospace', wrap=True)
    pdf.savefig(bbox_inches='tight')
    plt.close()

