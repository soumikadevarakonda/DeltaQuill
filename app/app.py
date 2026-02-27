import sys
import os

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from src.stylometric_features import extract_features

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
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
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

        # Debug line (remove later if you want)
        st.write("Vector shape:", tfidf_vec.shape)

        # Predict
        prediction = model.predict(tfidf_vec)[0]
        probabilities = model.predict_proba(tfidf_vec)[0]

        # ---------------------------
        # Display Results
        # ---------------------------
        st.subheader("Prediction")
        st.success(f"Predicted Author: {prediction}")

        st.subheader("Confidence Scores")

        df = pd.DataFrame({
            "Author": model.classes_,
            "Probability": probabilities
        })

        st.bar_chart(df.set_index("Author"))

        # ---------------------------
        # Stylometric Metrics (Display Only)
        # ---------------------------
        style_features = extract_features(user_input)

        st.subheader("Stylometric Metrics")

        st.write(f"Average word length: {style_features[0]:.2f}")
        st.write(f"Average sentence length: {style_features[1]:.2f}")
        st.write(f"Sentence length std dev: {style_features[2]:.2f}")
        st.write(f"Comma count: {style_features[3]}")
        st.write(f"Type-token ratio: {style_features[7]:.3f}")