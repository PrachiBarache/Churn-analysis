from src.data_load import *
from src.preprocessing import *
from src.baseline_train import *
from src.eval import *
from src.adv_train import*
# 1. Load the data
file_path = "data/E Commerce Dataset.xlsx"
sheet_name = "E Comm"
df = data_loader(file_path, sheet_name=sheet_name)
# 2. Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(df, target_col="Churn")

# 3. Train baseline Logistic Regression model
log_model = train_logistic_model(X_train, y_train, save_path="models/logistic_model.pkl")


# 4. Evaluate the model
evaluate_model(log_model, X_test, y_test, model_name="Logistic Regression",pdf_path=)

print("\n✅ Churn Prediction Pipeline Complete.")

run_baseline_models(X_train, X_test, y_train, y_test)
run_advanced_models(X_train, X_test, y_train, y_test)

print("\n✅ All models trained and evaluated.")
