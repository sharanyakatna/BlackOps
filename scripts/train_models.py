import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import os
import joblib

# ==============================
# 1️⃣ Paths and Dataset
# ==============================
DATA_PATH = "data/raw/upi_transactions_2024.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ==============================
# 2️⃣ Load Data
# ==============================
df = pd.read_csv(DATA_PATH)
print(f"📂 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ==============================
# 3️⃣ Set Target Column
# ==============================
target_col = "transaction_status"
# Map target to numeric (0: FAILED, 1: SUCCESS)
df[target_col] = df[target_col].map({'FAILED': 0, 'SUCCESS': 1})
print(f"Target classes: {df[target_col].unique()}")

# ==============================
# 4️⃣ Drop ID Columns
# ==============================
id_cols = ['transaction id']  # adjust if more
df.drop(columns=id_cols, inplace=True, errors='ignore')

# ==============================
# 5️⃣ Encode Categorical Columns
# ==============================
categorical_cols = [
    'timestamp', 'transaction type', 'merchant_category', 
    'sender_age_group', 'receiver_age_group', 'sender_state', 
    'sender_bank', 'receiver_bank', 'device_type', 
    'network_type', 'day_of_week'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ==============================
# 6️⃣ Separate Features & Target
# ==============================
X = df.drop(columns=[target_col])
y = df[target_col]

# ==============================
# 7️⃣ Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
print(f"Training on {X_train.shape[0]} rows, testing on {X_test.shape[0]} rows")

# ==============================
# 8️⃣ Handle Imbalanced Classes
# ==============================
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print(f"After SMOTE, training data shape: {X_train.shape}")

# ==============================
# 9️⃣ Scale Numeric Features
# ==============================
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ==============================
# 10️⃣ Define Models
# ==============================
models = {
    "RandomForest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=sum(y_train==0)/sum(y_train==1)),
    "LightGBM": LGBMClassifier(class_weight='balanced'),
    "CatBoost": CatBoostClassifier(verbose=0, class_weights=[sum(y_train==0), sum(y_train==1)])
}

# ==============================
# 11️⃣ Train, Evaluate & Save
# ==============================
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✅ Accuracy: {acc:.4f}")
    print(f"   📊 Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    # Save model
    save_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, save_path)
    print(f"   💾 Saved model to {save_path}")

print("\n✅ Training completed! All models and metrics saved.")

