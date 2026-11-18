"""Inference helpers for the DeltaQuill Streamlit app.

Keeps model loading, prediction, explanation and author-profile logic
separate from the UI code in app/app.py.
"""
import os
import pickle

import numpy as np
import pandas as pd

from src.stylometric_features import extract_features

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "final", "stylometry_dataset.csv")

AUTHORS = {
    "dostoevsky": {
        "name": "Fyodor Dostoevsky",
        "train_book": "Crime and Punishment",
        "test_book": "White Nights",
        "style": "Long, introspective sentences with psychological depth.",
    },
    "orwell": {
        "name": "George Orwell",
        "train_book": "1984",
        "test_book": "Animal Farm",
        "style": "Plain, declarative prose with a political edge.",
    },
    "sharma": {
        "name": "Robin Sharma",
        "train_book": "The 5 AM Club",
        "test_book": "The Wealth Money Can't Buy",
        "style": "Motivational, structured, second-person guidance.",
    },
}

TRAIN_BOOKS = ["crime_and_punishment", "1984", "the_5am_club"]
TEST_BOOKS = ["white_nights", "animal_farm", "wealth_money_cant_buy"]

# Display names for the stylometric feature vector (first 8 entries).
METRIC_NAMES = [
    "Avg word length",
    "Avg sentence length",
    "Sentence length std dev",
    "Commas per 100 words",
    "Semicolons per 100 words",
    "Exclamations per 100 words",
    "Questions per 100 words",
    "Type-token ratio",
]


def load_artifacts():
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict(model, vectorizer, text):
    """Return (predicted_label, probabilities DataFrame, tfidf vector)."""
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    df = pd.DataFrame({"author": model.classes_, "probability": probs})
    df["name"] = df["author"].map(lambda a: AUTHORS.get(a, {}).get("name", a.title()))
    df = df.sort_values("probability", ascending=False).reset_index(drop=True)
    return df.loc[0, "author"], df, vec


def top_terms(model, vectorizer, vec, label, n=10):
    """Terms in the input that pushed the score most toward `label`.

    Contribution of term j = tfidf_j * coef[label, j] (linear model).
    """
    class_idx = list(model.classes_).index(label)
    row = vec.tocsr()
    idx = row.indices
    if len(idx) == 0:
        return pd.DataFrame(columns=["term", "contribution"])
    contrib = row.data * model.coef_[class_idx, idx]
    terms = vectorizer.get_feature_names_out()[idx]
    df = pd.DataFrame({"term": terms, "contribution": contrib})
    df = df[df["contribution"] > 0]
    return df.sort_values("contribution", ascending=False).head(n).reset_index(drop=True)


def style_metrics(text):
    """Comparable stylometric metrics (counts normalised per 100 words).

    Text is lower-cased to match the preprocessing applied to the dataset.
    """
    text = text.lower()
    feats = extract_features(text)
    n_words = max(len(text.split()), 1)
    per100 = lambda c: c * 100.0 / n_words
    return [
        feats[0],
        feats[1],
        feats[2],
        per100(feats[3]),
        per100(feats[4]),
        per100(feats[5]),
        per100(feats[6]),
        feats[7],
    ]


def load_dataset():
    return pd.read_csv(DATA_PATH)


def author_profiles(df):
    """Average stylometric metrics per author across the whole dataset."""
    rows = {}
    for author, group in df.groupby("author"):
        rows[author] = np.mean([style_metrics(t) for t in group["text"]], axis=0)
    return pd.DataFrame(rows, index=METRIC_NAMES)


def evaluate(model, vectorizer, df):
    """Book-level held-out accuracy and dataset sizes."""
    test = df[df["book"].isin(TEST_BOOKS)]
    acc = float((model.predict(vectorizer.transform(test["text"])) == test["author"]).mean())
    return {
        "accuracy": acc,
        "train_chunks": int(df["book"].isin(TRAIN_BOOKS).sum()),
        "test_chunks": int(len(test)),
        "features": int(len(vectorizer.vocabulary_)),
    }
