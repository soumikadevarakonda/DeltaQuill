import os
import csv
from collections import defaultdict

PROCESSED_PATH = "datasets/processed"
OUTPUT_PATH = "datasets/final"
CHUNK_SIZE = 1000

os.makedirs(OUTPUT_PATH, exist_ok=True)

dataset = []

# -------- STEP 1: COLLECT ALL CHUNKS --------
for file in os.listdir(PROCESSED_PATH):
    if file.endswith(".txt"):
        filename = file.replace(".txt", "")
        parts = filename.split("_", 1)

        author = parts[0]
        book = parts[1] if len(parts) > 1 else "unknown"

        path = os.path.join(PROCESSED_PATH, file)

        with open(path, "r", encoding="utf-8") as f:
            words = f.read().split()

        for i in range(0, len(words), CHUNK_SIZE):
            chunk = words[i:i+CHUNK_SIZE]
            if len(chunk) == CHUNK_SIZE:
                dataset.append([author, book, " ".join(chunk)])

print("\nTotal raw chunks collected:", len(dataset))

# -------- STEP 2: GROUP BY AUTHOR AND BOOK --------
by_author_book = defaultdict(lambda: defaultdict(list))

for row in dataset:
    author, book, text = row
    by_author_book[author][book].append(row)

# Show raw distribution
print("\nRaw distribution per author per book:")
for author in by_author_book:
    for book in by_author_book[author]:
        print(f"{author} - {book}: {len(by_author_book[author][book])}")

# -------- STEP 3: BALANCE PER AUTHOR (EQUAL BOOK CONTRIBUTION) --------

# Total chunks per author
author_totals = {
    author: sum(len(by_author_book[author][book])
                for book in by_author_book[author])
    for author in by_author_book
}

min_total = min(author_totals.values())
print("\nMinimum total chunks among authors:", min_total)

balanced = []

for author in by_author_book:
    books = list(by_author_book[author].keys())
    per_book_limit = min_total // len(books)

    for book in books:
        balanced.extend(by_author_book[author][book][:per_book_limit])

print("Final balanced dataset size:", len(balanced))

# -------- STEP 4: SAVE --------
with open(os.path.join(OUTPUT_PATH, "stylometry_dataset.csv"),
          "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["author", "book", "text"])
    writer.writerows(balanced)

print("\nDataset saved with book column and balanced representation.")