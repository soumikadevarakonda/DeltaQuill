import os
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------
# Paths
# ---------------------------
DATA_PATH = "datasets/final/stylometry_dataset.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------
# Book-Level Split
# ---------------------------
train_books = [
    "crime_and_punishment",
    "1984",
    "the_5am_club"
]

test_books = [
    "white_nights",
    "animal_farm",
    "wealth_money_cant_buy"
]

train_df = df[df["book"].isin(train_books)]
test_df = df[df["book"].isin(test_books)]

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

# ---------------------------
# TF-IDF Vectorizer
# ---------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    lowercase=True
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["author"]
y_test = test_df["author"]

print("Feature dimension:", X_train.shape[1])

# ---------------------------
# Model
# ---------------------------
model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs"
)

model.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = model.predict(X_test)

print("\nBook-level Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------
# Save Model + Vectorizer
# ---------------------------
with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")