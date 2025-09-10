import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # for handling class imbalance
import os

# -------------------------------
# Step 1: Load the dataset
# -------------------------------
df = pd.read_csv('data/raw/upi_transactions_2024.csv')

# -------------------------------
# Step 2: Basic checks
# -------------------------------
print("Initial Data Info:")
print(df.info())
print(df.describe())
print("Missing values per column:\n", df.isna().sum())

# -------------------------------
# Step 3: Drop irrelevant columns
# -------------------------------
if 'transaction id' in df.columns:
    df = df.drop(['transaction id'], axis=1)

# -------------------------------
# Step 4: Handle timestamp
# -------------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df = df.drop('timestamp', axis=1)

# -------------------------------
# Step 5: Identify categorical and numeric features
# -------------------------------
categorical_cols = [
    'transaction type', 'merchant_category', 'transaction_status',
    'sender_age_group', 'receiver_age_group', 'sender_state',
    'sender_bank', 'receiver_bank', 'device_type', 'network_type', 'day_of_week'
]

numeric_cols = ['amount (INR)', 'hour_of_day', 'month', 'day', 'hour', 'is_weekend']

# -------------------------------
# Step 6: Encode categorical variables
# -------------------------------
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -------------------------------
# Step 7: Scale numeric features
# -------------------------------
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------------------------------
# Step 8: Handle imbalance (optional)
# -------------------------------
X = df.drop('fraud_flag', axis=1)
y = df['fraud_flag']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Save preprocessed data
cleaned_data_path = 'data/upi_transactions_2024_cleaned.csv'
preprocessed_df = pd.concat([X_res, y_res], axis=1)
os.makedirs('data', exist_ok=True)
preprocessed_df.to_csv(cleaned_data_path, index=False)

print(f"Preprocessing complete. Cleaned data saved to '{cleaned_data_path}'")

