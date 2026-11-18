import sys
import os

# Allow imports from project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import pickle
import pandas as pd
from src.stylometric_features import extract_features

MODEL_DIR = os.path.join(ROOT_DIR, "models")

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Stylometric Fingerprinting", layout="centered")

st.title("Stylometric Fingerprinting Engine")
st.write("Paste text below to detect its authorial signature.")

# ---------------------------
# Load Model + Vectorizer
# ---------------------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------------------------
# Input Area
# ---------------------------
user_input = st.text_area("Input Text", height=250)

if st.button("Analyze"):

    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # TF-IDF features (ONLY these go into the model)
        tfidf_vec = vectorizer.transform([user_input])

        if tfidf_vec.nnz == 0:
            st.warning(
                "None of the words in this text appear in the model's vocabulary, "
                "so the prediction below is not meaningful. Try a longer passage."
            )
        elif len(user_input.split()) < 50:
            st.info("Short inputs give less reliable predictions. 50+ words recommended.")

        # Predict
        prediction = model.predict(tfidf_vec)[0]
        probabilities = model.predict_proba(tfidf_vec)[0]

        # ---------------------------
        # Display Results
        # ---------------------------
        st.subheader("Prediction")
        st.success(f"Predicted Author: {prediction.capitalize()}")

        st.subheader("Confidence Scores")

        df = pd.DataFrame({
            "Author": [c.capitalize() for c in model.classes_],
            "Probability": probabilities
        })

        st.bar_chart(df.set_index("Author"))

        # ---------------------------
        # Stylometric Metrics (Display Only)
        # ---------------------------
        style_features = extract_features(user_input)

        st.subheader("Stylometric Metrics")

        metrics = pd.DataFrame({
            "Metric": [
                "Average word length",
                "Average sentence length (words)",
                "Sentence length std dev",
                "Commas",
                "Semicolons",
                "Exclamation marks",
                "Question marks",
                "Type-token ratio",
            ],
            "Value": [
                f"{style_features[0]:.2f}",
                f"{style_features[1]:.2f}",
                f"{style_features[2]:.2f}",
                str(style_features[3]),
                str(style_features[4]),
                str(style_features[5]),
                str(style_features[6]),
                f"{style_features[7]:.3f}",
            ],
        })
        st.table(metrics.set_index("Metric"))
