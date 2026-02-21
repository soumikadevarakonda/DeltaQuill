from ebooklib import epub
from bs4 import BeautifulSoup
import os

RAW_PATH = "datasets/raw"
PROCESSED_PATH = "datasets/processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

def extract_text_from_epub(file_path):
    book = epub.read_epub(file_path)
    text = ""

    for item in book.get_items():
        if item.get_type() == 9:  # document
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text += soup.get_text(separator=" ") + " "

    return text

for author in os.listdir(RAW_PATH):
    author_path = os.path.join(RAW_PATH, author)
    if os.path.isdir(author_path):
        for file in os.listdir(author_path):
            if file.endswith(".epub"):
                full_path = os.path.join(author_path, file)
                text = extract_text_from_epub(full_path)

                output_file = os.path.join(
                    PROCESSED_PATH,
                    f"{author}_{file.replace('.epub', '.txt')}"
                )

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text)

print("Extraction complete.")