"""This module contains the main script for the project.
It reads individual a data file containing German sentences and text, feeds the text to the lemmatizer and generates
the word-frequency dict.
"""

from collections import Counter
import csv
from typing import Tuple
import spacy
from tqdm import tqdm

BATCH_SIZE = 1000
SPACY_MODEL = "de_core_news_lg"
INPUT_FILES = [
    "./data/raw/deu_mixed-typical_2011_300K-sentences.txt",
    "./data/raw/deu_wikipedia_2021_300K-sentences.txt",
    "./data/raw/deu-de_web-public_2019_300K-sentences.txt",
]
OUTPUT_FILE = "./data/processed/freqs_dict.csv"
LEMMA = "lemma"
POS = "pos"
FREQUENCY = "frequency"

print("Loading spaCy model")
nlp = spacy.load(SPACY_MODEL)


def lemmatize_text(filename: str) -> Counter:
    """Processes an input file and generates word-frequency dict for it."""
    print("Calculating the workload")
    with open(filename, "r", encoding="utf8") as f:
        total_lines = sum(1 for _ in f)

    curr_freqs: Counter[Tuple[str, str]] = Counter()

    print("Starting lemmatization process...")
    with open(filename, "r", encoding="utf8") as f:
        for doc in tqdm(nlp.pipe(f, batch_size=BATCH_SIZE), total=total_lines):
            for token in doc:
                if not token.is_punct and not token.is_space:
                    curr_freqs[(token.lemma_, token.pos_)] += 1
    print("Finished lemmatization")

    return curr_freqs


freqs_dict: Counter[Tuple[str, str]] = Counter()

for input_file in INPUT_FILES:
    print(f"Lemmatizing {input_file}")
    freqs_dict = freqs_dict + lemmatize_text(input_file)

print("Exporting to file...")
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([LEMMA, POS, FREQUENCY])
    for (lemma, pos), count in freqs_dict.most_common():
        writer.writerow([lemma, pos, count])
print(f"Counter saved to {OUTPUT_FILE}")
