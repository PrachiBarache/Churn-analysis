import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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
