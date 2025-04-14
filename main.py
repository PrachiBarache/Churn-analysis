from src.data_load import data_loader
from src.preprocessing import preprocess_data
from src.baseline_train import train_logistic_model
from src.eval import evaluate_model

# 1. Load the data
file_path = "data/E Commerce Dataset.xlsx"
sheet_name = "E Comm"
df = data_loader(file_path, sheet_name=sheet_name)

# 2. Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(df, target_col="Churn")

# 3. Train baseline Logistic Regression model
log_model = train_logistic_model(X_train, y_train, save_path="src/__pycache__/logistic_model.pkl")

# 4. Evaluate the model
evaluate_model(log_model, X_test, y_test, model_name="Logistic Regression")

print("\n✅ Churn Prediction Pipeline Complete.")

from src.train_lightgbm import train_lightgbm
from src.evaluate import evaluate_model

lgb_model = train_lightgbm(X_train, y_train)
evaluate_model(lgb_model, X_test, y_test, model_name="LightGBM")
