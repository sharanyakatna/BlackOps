from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load your model(s) once at startup
MODEL_PATH = "models/catboost_model.pkl"  # adjust path/name if needed
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    print("⚠️ Model file not found, please check path.")

@app.route("/")
def home():
    return {"message": "Blackops API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        features = data.get("features")  # expecting list of features
        
        if not features:
            return jsonify({"error": "No features provided"}), 400

        prediction = model.predict([features])[0]
        return jsonify({"prediction": str(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

