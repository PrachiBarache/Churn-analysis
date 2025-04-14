import joblib
import lightgbm as lgb

def train_lightgbm(X_train, y_train, save_path="models/lightgbm_model.pkl"):
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


import joblib
from catboost import CatBoostClassifier

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

import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
