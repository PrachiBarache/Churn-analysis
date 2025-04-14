import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_logistic_model(X_train, y_train, save_path="models/logistic_model.pkl"):
    """
    Trains a baseline logistic regression model with class balancing.

    Parameters:
        X_train (array): Training features
        y_train (array): Training labels
        save_path (str): Path to save the trained model

    Returns:
        model: Trained LogisticRegression model
    """

    # Logistic Regression with balanced class weights to handle class imbalance
    model = LogisticRegression(class_weight='balanced', max_iter=1000)

    # Fit model to training data
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, save_path)
    print(f"✅ Logistic Regression model saved to {save_path}")

    return model

def train_decision_tree(X_train, y_train, save_path="models/decision_tree.pkl"):
    """
    Trains a basic Decision Tree classifier.
    """
    model = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"✅ Decision Tree saved at {save_path}")
    return model

def train_random_forest(X_train, y_train, save_path="models/random_forest.pkl"):
    """
    Trains a Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"✅ Random Forest saved at {save_path}")
    return model


def train_xgboost(X_train, y_train, save_path="models/xgboost_model.pkl"):
    """
    Trains an XGBoost classifier.
    """
    model = xgb.XGBClassifier(
        scale_pos_weight=5,  # Ratio of negative to positive class
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"✅ XGBoost model saved at {save_path}")
    return model