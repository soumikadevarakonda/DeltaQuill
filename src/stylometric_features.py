import re
import numpy as np

FUNCTION_WORDS = [
    "and", "but", "however", "therefore", "because",
    "although", "though", "while", "if", "then"
]

def extract_features(text):
    words = text.split()
    sentences = re.split(r'[.!?]+', text)

    word_count = len(words)
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len = np.mean(sentence_lengths) if sentence_lengths else 0
    std_sent_len = np.std(sentence_lengths) if sentence_lengths else 0

    comma_freq = text.count(',')
    semicolon_freq = text.count(';')
    exclaim_freq = text.count('!')
    question_freq = text.count('?')

    type_token_ratio = len(set(words)) / word_count if word_count else 0

    function_word_freq = [
        words.count(fw) / word_count if word_count else 0
        for fw in FUNCTION_WORDS
    ]

    return [
        avg_word_len,
        avg_sent_len,
        std_sent_len,
        comma_freq,
        semicolon_freq,
        exclaim_freq,
        question_freq,
        type_token_ratio,
        *function_word_freq
    ]