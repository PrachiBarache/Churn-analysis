import joblib
import lightgbm as lgb
import os
from src.eval import *
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


def train_lightgbm(X_train, y_train, save_path="models/lightgbm_model.pkl",pdf_path="results/all_models_report.pdf"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    """
    Trains a LightGBM classifier.
    """
    model = lgb.LGBMClassifier(
        class_weight='balanced',  # Handle class imbalance
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"✅ LightGBM model saved at {save_path}")
    return model



def train_catboost(X_train, y_train, save_path="models/catboost_model.pkl"):
    """
    Trains a CatBoost classifier.
    """
    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        auto_class_weights='Balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"✅ CatBoost model saved at {save_path}")
    return model

def train_stacked_model(X_train, y_train, save_path="models/stacked_model.pkl"):
    """
    Trains a stacked ensemble model.
    """
    base_learners = [
        ('dt', DecisionTreeClassifier(max_depth=5, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced'))
    ]
    
    stack_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        passthrough=True
    )

    stack_model.fit(X_train, y_train)
    joblib.dump(stack_model, save_path)
    print(f"✅ Stacked model saved at {save_path}")
    return stack_model

import joblib
from sklearn.neural_network import MLPClassifier

def train_mlp_model(X_train, y_train, save_path="models/mlp_model.pkl"):
    """
    Trains a multi-layer perceptron (MLP) neural network.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"✅ MLP model saved at {save_path}")
    return model


def run_advanced_models(X_train, X_test, y_train, y_test,pdf):
    """
    Train and evaluate advanced models.
    """
    print("\n🔸 Training LightGBM")
    lgb_model = train_lightgbm(X_train, y_train)
    evaluate_model(lgb_model, X_test, y_test, "LightGBM", pdf)

    print("\n🔸 Training CatBoost")
    cat_model = train_catboost(X_train, y_train)
    evaluate_model(cat_model, X_test, y_test, "CatBoost", pdf)

    print("\n🔸 Training MLP (Neural Net)")
    mlp_model = train_mlp_model(X_train, y_train)
    evaluate_model(mlp_model, X_test, y_test, "MLP Classifier",pdf)

    print("\n🔸 Training Stacked Ensemble")
    stack_model = train_stacked_model(X_train, y_train)
    evaluate_model(stack_model, X_test, y_test, "Stacked Ensemble",pdf)