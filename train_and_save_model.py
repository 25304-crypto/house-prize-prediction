# -*- coding: utf-8 -*-
"""Train and save the house price prediction model"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 42
CSV_PATH = r"C:\Users\VICTUS\Desktop\house_pre\housing.csv"
TARGET_COL = "median_house_value"

# Load data
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Define preprocessing pipeline
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create and train the model with best parameters
model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", HistGradientBoostingRegressor(
            l2_regularization=0.1,
            learning_rate=0.1,
            max_depth=None,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            random_state=RANDOM_STATE
        ))
    ]
)

print("Training model...")
model.fit(X_train, y_train)

# Save the model
model_path = r"C:\Users\VICTUS\Desktop\house_pre\house_price_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")

# Display performance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("\n=== MODEL PERFORMANCE ===")
print(f"Train RMSE: {root_mean_squared_error(y_train, train_pred):.3f}")
print(f"Test RMSE:  {root_mean_squared_error(y_test, test_pred):.3f}")
print(f"Train MAE:  {mean_absolute_error(y_train, train_pred):.3f}")
print(f"Test MAE:   {mean_absolute_error(y_test, test_pred):.3f}")
print(f"Train R2:   {r2_score(y_train, train_pred):.3f}")
print(f"Test R2:    {r2_score(y_test, test_pred):.3f}")
