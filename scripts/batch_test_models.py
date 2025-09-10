import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Paths
DATA_PATH = "data/upi_transactions_2024.csv"
MODEL_DIR = "models/"
RESULTS_DIR = "results/"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Detect target column (binary classification only)
binary_cols = [col for col in df.columns if df[col].nunique() == 2]
target_col = binary_cols[0] if binary_cols else None

if target_col:
    print(f"Detected target column: {target_col}")
    X = df.drop(columns=[target_col])
    y_true = df[target_col]
else:
    print("⚠️ No target column found. Running in prediction-only mode.")
    X = df
    y_true = None

# Loop over all saved models
for model_file in os.listdir(MODEL_DIR):
    if not model_file.endswith(".pkl"):
        continue

    model_path = os.path.join(MODEL_DIR, model_file)
    model_name = model_file.replace(".pkl", "")

    try:
        model = joblib.load(model_path)
        print(f"Loaded model: {model_name}")

        # Run predictions
        y_pred = model.predict(X)

        # Save predictions
        result_path = os.path.join(RESULTS_DIR, f"{model_name}_predictions.csv")
        pd.DataFrame({"prediction": y_pred}).to_csv(result_path, index=False)
        print(f"✅ Predictions saved to {result_path}")

        # If ground truth exists, evaluate
        if y_true is not None:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            print(f"📊 {model_name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
            print(f"Confusion Matrix:\n{cm}\n")

            # Save evaluation results
            with open(os.path.join(RESULTS_DIR, f"{model_name}_metrics.txt"), "w") as f:
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Confusion Matrix:\n{cm}\n")

    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")

