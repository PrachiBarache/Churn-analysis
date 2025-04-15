from src.data_load import *
from src.preprocessing import *
from src.baseline_train import *
from src.eval import *
from src.adv_train import *
from src.eda import generate_eda_report
from matplotlib.backends.backend_pdf import PdfPages

# Load and preprocess data
df = data_loader("data/E Commerce Dataset.xlsx", sheet_name="E Comm")
generate_eda_report(df, save_path="results/eda_report.pdf")

X_train, X_test, y_train, y_test = preprocess_data(df, target_col="Churn")

# Create shared PDF for all models
with PdfPages("results/all_models_report.pdf") as pdf:
    run_baseline_models(X_train, X_test, y_train, y_test, pdf)
    run_advanced_models(X_train, X_test, y_train, y_test, pdf)

print("\n All models trained and evaluated.")
