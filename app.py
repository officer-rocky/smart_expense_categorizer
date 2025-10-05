# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("models/model.joblib")

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    texts = data.get("texts") or ([data.get("text")] if data.get("text") else [])
    if not texts:
        return jsonify({"error":"No text provided"}), 400

    texts_clean = [clean_text(t) for t in texts]
    preds = model.predict(texts_clean).tolist()
    probs = model.predict_proba(texts_clean).tolist()
    return jsonify({"preds": preds, "probs": probs, "classes": model.classes_.tolist()})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
