"""This module contains the main script for the project.
It reads individual a data file containing German sentences and text, feeds the text to the lemmatizer and generates
the word-frequency dict.
"""

from collections import Counter
import csv
import spacy
from tqdm import tqdm

BATCH_SIZE = 1000
SPACY_MODEL = "de_core_news_lg"
INPUT_FILE = "./data/raw/deu_news_2024_1M-sentences.txt"
OUTPUT_FILE = "./data/reports/word_freq_dict.csv"
LEMMA = "lemma"
POS = "pos"
FREQUENCY = "frequency"

print("Loading spaCy model")
nlp = spacy.load(SPACY_MODEL)

print("Calculating the workload")
with open(INPUT_FILE, "r", encoding="utf8") as f:
    total_lines = sum(1 for _ in f)

print("Starting lemmatization process...")
wore_freq = Counter()
with open(INPUT_FILE, "r", encoding="utf8") as f:
    for doc in tqdm(nlp.pipe(f, batch_size=BATCH_SIZE), total=total_lines):
        for token in doc:
            if not token.is_punct and not token.is_space:
                wore_freq[(token.lemma_, token.pos_)] += 1
print("Finished lemmatization")

print("Exporting to file...")
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([LEMMA, POS, FREQUENCY])
    for (lemma, pos), count in wore_freq.most_common():
        writer.writerow([lemma, pos, count])
print(f"Counter saved to {OUTPUT_FILE}")
