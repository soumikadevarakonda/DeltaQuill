import pandas as pd
import numpy as np
from stylometric_features import extract_features

df = pd.read_csv("datasets/final/stylometry_dataset.csv")

profiles = {}

for author in df["author"].unique():
    texts = df[df["author"] == author]["text"]
    features = np.array([extract_features(t) for t in texts])
    profiles[author] = features.mean(axis=0)

for author in profiles:
    print("\nAuthor:", author)
    print("Avg word length:", profiles[author][0])
    print("Avg sentence length:", profiles[author][1])
    print("Type-token ratio:", profiles[author][7])