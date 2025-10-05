# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re

# Create models directory
os.makedirs("models", exist_ok=True)

# Preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# 1) Load data
df = pd.read_csv("data/transactions.csv")
df = df.dropna(subset=["text","label"])
df["text"] = df["text"].apply(clean_text)
X = df["text"]
y = df["label"].astype(str)

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
])

# 4) Grid search
params = {
    "tfidf__min_df": [1, 2],
    "tfidf__ngram_range": [(1,1),(1,2)],
    "clf__C": [0.5, 1.0, 2.0]
}

try:
    grid = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print("Grid search done. Best params:", grid.best_params_)
except Exception as e:
    print("Grid search failed. Using default pipeline. Error:", e)
    pipeline.fit(X_train, y_train)
    best_model = pipeline

# 5) Evaluate
y_pred = best_model.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 6) Save
joblib.dump(best_model, "models/model.joblib")
print("Saved model to models/model.joblib")
