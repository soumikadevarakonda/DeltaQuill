import os
import csv
from collections import Counter

PROCESSED_PATH = "datasets/processed"
OUTPUT_PATH = "datasets/final"
CHUNK_SIZE = 1000

os.makedirs(OUTPUT_PATH, exist_ok=True)

dataset = []

for file in os.listdir(PROCESSED_PATH):
    author = file.split("_")[0]
    path = os.path.join(PROCESSED_PATH, file)

    with open(path, "r", encoding="utf-8") as f:
        words = f.read().split()

    for i in range(0, len(words), CHUNK_SIZE):
        chunk = words[i:i+CHUNK_SIZE]
        if len(chunk) == CHUNK_SIZE:
            dataset.append([author, " ".join(chunk)])

# Print distribution
counts = Counter([row[0] for row in dataset])
print("Chunks per author:", counts)

# Balance dataset
min_count = min(counts.values())
balanced = []

author_seen = {author: 0 for author in counts}

for row in dataset:
    author = row[0]
    if author_seen[author] < min_count:
        balanced.append(row)
        author_seen[author] += 1

print("Balanced size:", len(balanced))

# Save to CSV
with open(os.path.join(OUTPUT_PATH, "stylometry_dataset.csv"), 
          "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["author", "text"])
    writer.writerows(balanced)

print("Dataset saved.")