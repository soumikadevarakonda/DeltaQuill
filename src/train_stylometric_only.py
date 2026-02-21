import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from stylometric_features import extract_features

DATA_PATH = "datasets/final/stylometry_dataset.csv"

df = pd.read_csv(DATA_PATH)

X_features = np.array([extract_features(text) for text in df["text"]])
y = df["author"]

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Stylometry-only Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))