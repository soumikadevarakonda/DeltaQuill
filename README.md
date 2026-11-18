# DeltaQuill

### Stylometric Fingerprinting: Writing Analysis and Authorship Detection Engine

**DeltaQuill** is a stylometry framework designed to identify authorship by analyzing lexical, syntactic, and statistical patterns in writing. By applying classical machine learning to distinctive "writing signatures," it achieves high-accuracy attribution across different literary styles.

---

## 🚀 Project Overview

DeltaQuill performs **supervised multiclass classification** to predict the most likely author of an input text. The system is built to ensure **book-level generalization**, meaning the model learns an author's fundamental style rather than just memorizing specific plot points or vocabulary from a single book.

### Authors Included

| Author | Training Book | Testing Book |
| --- | --- | --- |
| **Fyodor Dostoevsky** | *Crime and Punishment* | *White Nights* |
| **George Orwell** | *1984* | *Animal Farm* |
| **Robin Sharma** | *The 5 AM Club* | *The Wealth Money Can't Buy* |

---

## 🛠 Methodology

### 1. Data Pipeline

* **Extraction:** Text is pulled from EPUB files.
* **Processing:** Cleaning, normalization, and segmenting into chunks.
* **Storage:** Data is stored in a structured CSV format containing the text chunk, author label, and book identifier.

### 2. Feature Engineering

The engine utilizes two primary layers of analysis:

**A. TF-IDF Vectorization**

* **N-gram Range:** (1, 2) — captures both individual words and two-word phrases.
* **Max Features:** 5,000 dimensions.
* **Min DF:** 2 (filters out ultra-rare noise).

**B. Stylometric Metrics (UI-Integrated)**
The following metadata is extracted for interpretability:

* Average word and sentence length.
* Sentence length standard deviation (rhythm analysis).
* Punctuation frequencies and Type-Token Ratio (vocabulary richness).

### 3. Machine Learning Model

The system employs **Logistic Regression**.

* **Why?** It handles high-dimensional sparse data efficiently and provides clear class probabilities.
* **Evaluation:** By splitting data by **book** rather than by random chunks, we prevent "data leakage," ensuring the model recognizes the author even in a completely new narrative context.
* **Accuracy:** ~98% on book-level evaluation.

---

## 📂 Project Structure

```text
DeltaQuill/
│
├── .streamlit/
│   └── config.toml                # Streamlit theme
├── app/
│   └── app.py                     # Streamlit interface
├── src/
│   ├── extract_text.py            # EPUB -> raw text
│   ├── preprocess.py              # Cleaning and normalization
│   ├── chunking.py                # 1000-word chunks -> CSV dataset
│   ├── stylometric_features.py    # Stylometric feature extraction
│   ├── predictor.py               # Inference, explanations, author profiles (used by the app)
│   ├── train_book_split.py        # Primary training script (book-level split)
│   ├── train_model.py             # Alternative: random chunk split
│   ├── train_combined.py          # Experiment: TF-IDF + stylometric features
│   └── train_stylometric_only.py  # Experiment: stylometric features only
├── datasets/
│   ├── raw/                       # Source EPUBs
│   ├── processed/                 # Extracted text
│   └── final/
│       └── stylometry_dataset.csv # Processed data
├── models/
│   ├── model.pkl                  # Trained Logistic Regression model
│   └── vectorizer.pkl             # Fitted TF-IDF Vectorizer
├── requirements.txt
└── README.md

```

---

## 💻 Running the Project

Install dependencies (run all commands from the project root):

```bash
pip install -r requirements.txt
```

### 1. Train the Model

Run the split-validation script to generate the model artifacts:

```bash
python src/train_book_split.py

```

### 2. Launch the Interface

Start the Streamlit web application:

```bash
streamlit run app/app.py

```

> **Deployment note:** The app is deployed on Streamlit Community Cloud with `app/app.py` as the entry point. `scikit-learn` is pinned in `requirements.txt` to the version used to pickle the model; if you retrain with a different version, update the pin to match.

---

## 📝 Sample Predictions

> **Dostoevsky-Style:** *"I analyze myself relentlessly, as though I were both the criminal and the judge..."*
> **Orwell-Style:** *"The government reduced the ration again. The announcement was brief. No explanation was offered."*

---

## ⚠️ Limitations & Future Roadmap

* **Current Limits:** Forced-choice classification (cannot detect "Unknown"), limited author pool, and stylometric stats are currently for display rather than training.
* **Future Enhancements:**
* Integrate a **Hybrid Model** (TF-IDF + Stylometric features).
* Add **"Out-of-Distribution" detection** for unknown authors.
* Support for PDF/TXT file uploads.
* Expanded dataset with more diverse literary figures.
