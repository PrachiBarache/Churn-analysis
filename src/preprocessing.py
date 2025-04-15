import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

   # Clean and prepare features for modeling.

def preprocess_data(df: pd.DataFrame, target_col: str):     
    df = df.copy()

    # Drop irrelevant or ID columns
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categorical columns using LabelEncoder
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test
