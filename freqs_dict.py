"""This module contains the main script for the project.
It reads individual a data file containing German sentences and text, feeds the text to the lemmatizer and generates
the word-frequency dict.
"""

import tempfile
import os
from collections import Counter
from typing import Tuple
import csv
import spacy
from tqdm import tqdm
import PyPDF2

BATCH_SIZE = 1000
SPACY_MODEL = "de_core_news_lg"
INPUT_FILES = [
    "./data/raw/Norwegische Elefanten B1.pdf",
    "./data/raw/Der Tote im See.pdf",
    "./data/raw/Jungs sind keine Regenschirme.pdf"
    ]
OUTPUT_FILE = "./data/processed/story_book.csv"
LEMMA = "lemma"
POS = "pos"
FREQUENCY = "frequency"

print("Loading spaCy model")
nlp = spacy.load(SPACY_MODEL)


def open_PDF(filename: str) -> str:
    """Opens a PDF file and extracts the text from it, saving it to a temporary text file."""
    # Create a temporary text file with the same name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"{base_name}_temp.txt")

    # Convert PDF to text
    extracted_text = ""
    with open(filename, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() + "\n"

    # Write the extracted text to the temporary file
    with open(temp_file_path, "w", encoding="utf-8") as temp_file:
        temp_file.write(extracted_text)

    return temp_file_path


def lemmatize_text(filename: str) -> Counter:
    """Processes an input file and generates word-frequency dict for it."""
    # Calculating the workload
    with open(filename, "r", encoding="utf8") as f:
        total_lines = sum(1 for _ in f)

    # Processing the file
    curr_freqs: Counter[Tuple[str, str]] = Counter()
    with open(filename, "r", encoding="utf8") as f:
        for doc in tqdm(nlp.pipe(f, batch_size=BATCH_SIZE), total=total_lines):
            for token in doc:
                if not token.is_punct and not token.is_space:
                    curr_freqs[(token.lemma_, token.pos_)] += 1

    return curr_freqs


freqs_dict: Counter[Tuple[str, str]] = Counter()

for input_file in INPUT_FILES:
    print(f"Lemmatizing {input_file}...")

    # Check if the input file is a PDF and convert it to text if needed
    if input_file.endswith(".pdf"):
        print("PDF file detected. Converting to text...")
        input_file = open_PDF(input_file)
        print(f"PDF was extracted to the temporary file: {input_file}")

    freqs_dict = freqs_dict + lemmatize_text(input_file)

print("Exporting to file...")
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([LEMMA, POS, FREQUENCY])
    for (lemma, pos), count in freqs_dict.most_common():
        writer.writerow([lemma, pos, count])
print(f"Counter saved to {OUTPUT_FILE}")
