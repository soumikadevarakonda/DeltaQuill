
DeltaQuill
Stylometric Fingerprinting: Writing Analysis and Authorship Detection Engine

DeltaQuill is a stylometry framework that identifies authorship by analyzing lexical, syntactic, and statistical patterns in writing.

It applies classical machine learning techniques to detect distinctive writing signatures across texts. The system leverages:

TF-IDF vectorization (unigrams and bigrams)

Stylometric feature extraction

Logistic Regression for multiclass classification

Book-level evaluation to prevent data leakage

The result is a high-accuracy authorship attribution engine with an interactive Streamlit interface.

Project Overview

DeltaQuill performs supervised multiclass classification to predict the most likely author of an input text.

Authors Included

Fyodor Dostoevsky

George Orwell

Robin Sharma

Each author contributes:

One book used for training

One different book used for testing

This ensures book-level generalization, not chunk-level memorization.

Methodology
1. Data Pipeline

Text extracted from EPUB files

Preprocessing and cleaning

Chunking into smaller segments

Stored as structured CSV dataset

Each dataset entry contains:

Text chunk

Author label

Book identifier

2. Feature Engineering
TF-IDF Vectorization

Configuration:

ngram_range=(1,2)

max_features=5000

min_df=2

Lowercasing enabled

Each text chunk becomes a 5000-dimensional vector representing lexical patterns.

Stylometric Metrics (Displayed in UI)

The system also extracts:

Average word length

Average sentence length

Sentence length standard deviation

Punctuation frequencies

Type-token ratio

Function word frequencies

Currently, these are displayed for interpretability and analysis.

3. Machine Learning Model

Algorithm Used:

Logistic Regression (Multiclass Classification)

Why Logistic Regression?

Effective for high-dimensional sparse data

Produces interpretable class probabilities

Strong performance in text classification tasks

The model learns linear decision boundaries in TF-IDF feature space.

Evaluation Strategy

Book-Level Split:

Training Books:

Crime and Punishment

1984

The 5 AM Club

Testing Books:

White Nights

Animal Farm

The Wealth Money Can't Buy

This prevents stylistic leakage across chunks of the same book.

Accuracy achieved: ~98% (book-level evaluation)

Project Structure
DeltaQuill/
│
├── app/
│   └── app.py
│
├── src/
│   └── stylometric_features.py
│
├── datasets/
│   └── final/
│       └── stylometry_dataset.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── train_book_split.py
├── train_model.py
├── train_combined.py
├── train_stylometric_only.py
└── README.md
Running the Project
1. Train the Model
python train_book_split.py

This generates:

models/model.pkl

models/vectorizer.pkl

2. Launch Streamlit App
streamlit run app/app.py

Open the local URL provided in the terminal.

Sample Texts for Prediction
Dostoevsky-Style Sample
I have often wondered whether my thoughts are truly my own, or merely echoes of conversations I have rehearsed in solitude. There is something humiliating in recognizing one’s weakness so clearly and yet being unable to resist it. I analyze myself relentlessly, as though I were both the criminal and the judge, and in doing so I discover not innocence, but an even subtler guilt.
Orwell-Style Sample
The government reduced the ration again. The announcement was brief. No explanation was offered. Citizens were expected to accept it.
Key Concepts Demonstrated

Supervised Machine Learning

Text Vectorization using TF-IDF

N-gram Feature Engineering

Multiclass Logistic Regression

Book-Level Validation Strategy

Streamlit Deployment

Limitations

Forced-choice classification among three authors

No unknown-author detection

Limited number of books per author

Stylometric features not yet integrated into training pipeline

Future Enhancements

Integrate stylometric + TF-IDF hybrid model

Add unknown-author detection

Expand dataset with additional authors

Add file upload support (PDF/TXT)

Improve interpretability visualizations