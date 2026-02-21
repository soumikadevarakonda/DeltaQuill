import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from stylometric_features import extract_features

DATA_PATH = "datasets/final/stylometry_dataset.csv"

df = pd.read_csv(DATA_PATH)

X_text = df["text"]
y = df["author"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Stylometric features
X_train_style = np.array([extract_features(t) for t in X_train_text])
X_test_style = np.array([extract_features(t) for t in X_test_text])

# Combine
X_train_combined = hstack([X_train_tfidf, X_train_style])
X_test_combined = hstack([X_test_tfidf, X_test_style])

model = LogisticRegression(max_iter=2000)
model.fit(X_train_combined, y_train)

y_pred = model.predict(X_test_combined)

print("Combined Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))