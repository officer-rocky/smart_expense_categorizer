# infer.py
import joblib
import sys
import re

# Load model
model = joblib.load("models/model.joblib")

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def predict(texts):
    texts = [clean_text(t) for t in texts]
    preds = model.predict(texts)
    probs = model.predict_proba(texts)
    return preds, probs

if __name__ == "__main__":
    samples = sys.argv[1:] or ["Swiggy order #123 250", "Salary Oct 50000", "Uber ride 220"]
    preds, probs = predict(samples)
    for t, p, pr in zip(samples, preds, probs):
        print(f"Text: {t}\nPred: {p}\nProbabilities:")
        for cls, prob in zip(model.classes_, pr):
            print(f"{cls}: {prob:.2%}")
        print("-"*30)
