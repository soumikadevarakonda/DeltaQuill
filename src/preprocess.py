import re
import os

PROCESSED_PATH = "datasets/processed"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)  # remove pure numbers
    return text.strip()

for file in os.listdir(PROCESSED_PATH):
    path = os.path.join(PROCESSED_PATH, file)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_text(text)

    with open(path, "w", encoding="utf-8") as f:
        f.write(cleaned)

print("Cleaning complete.")