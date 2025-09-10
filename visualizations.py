import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Models
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- Load Dataset ---
data_path = "data/raw/upi_transactions_2024.csv"
df = pd.read_csv(data_path)

# Assuming the last column is the target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# --- Load Models ---
model_files = {
    "RandomForest": "models/RandomForest.pkl",
    "LogisticRegression": "models/LogisticRegression.pkl",
    "XGBoost": "models/XGBoost.json",
    "LightGBM": "models/LightGBM.txt",
    "CatBoost": "models/CatBoost.cbm"
}

models = {}
for name, path in model_files.items():
    if name == "CatBoost":
        model = CatBoostClassifier()
        model.load_model(path)
        models[name] = model
    elif name == "XGBoost":
        model = XGBClassifier()
        model.load_model(path)
        models[name] = model
    elif name == "LightGBM":
        model = lgb.Booster(model_file=path)
        models[name] = model
    else:  # scikit-learn models
        with open(path, "rb") as f:
            models[name] = pickle.load(f)

print("All models loaded successfully!")

# --- Visualizations ---
for name, model in models.items():
    print(f"\n--- {name} ---")

    if name == "LightGBM":
        y_pred = np.round(model.predict(X))
    else:
        y_pred = model.predict(X)

    # Classification report
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Feature importance (if supported)
    try:
        if name == "LightGBM":
            importance = model.feature_importance()
        else:
            importance = model.feature_importances_

        plt.figure(figsize=(8, 5))
        sns.barplot(x=X.columns, y=importance)
        plt.xticks(rotation=90)
        plt.title(f"{name} Feature Importance")
        plt.show()
    except AttributeError:
        print(f"Feature importance not available for {name}")

