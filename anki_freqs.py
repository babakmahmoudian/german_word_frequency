"""Compares the given Anki collection with the processed word-frequency dictionary and
identifies missing/extra words and the words' frequencies."""

import logging
import re
import requests
import pandas as pd
import spacy
from configs import (
    ANKI_API_URL,
    ANKI_QUERY,
    ANKI_REQUEST_TIMEOUT,
    ANKI_WORD_FIELD,
    ANKIFREQ_FILE,
    SPACY_MODEL,
    LEMMA_BATCH_SIZE,
    FREQDICT_FILE,
    LEMMA_COL,
    FREQUENCY_COL,
    WORD_COL,
    RANK_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting Anki frequency generation process.")

logging.info("Reading frequency data from: %s", FREQDICT_FILE)
frequency_df = pd.read_csv(
    FREQDICT_FILE,
    index_col=None,
    header=0,
    usecols=[LEMMA_COL, FREQUENCY_COL, RANK_COL],
)

logging.info("Fetching note IDs from Anki collection using AnkiConnect API")
note_ids = requests.post(
    timeout=ANKI_REQUEST_TIMEOUT,
    url=ANKI_API_URL,
    json={"action": "findNotes", "version": 6, "params": {"query": ANKI_QUERY}},
).json()["result"]
logging.info("Total of %d notes found in Anki collection.", len(note_ids))

logging.info("Fetching note information for the retrieved note IDs")
note_info = requests.post(
    timeout=ANKI_REQUEST_TIMEOUT,
    url=ANKI_API_URL,
    json={"action": "notesInfo", "version": 6, "params": {"notes": note_ids}},
).json()["result"]

logging.info("Extracting words from the note information")
anki_words = []
for note in note_info:
    word_field_value = note["fields"][ANKI_WORD_FIELD]["value"]
    word_field_value = re.sub(r'[^\w\s]', '', word_field_value)
    anki_words.extend(word_field_value.split())

anki_df = pd.DataFrame(anki_words, columns=[WORD_COL])

logging.info("Loading spaCy model: %s", SPACY_MODEL)
nlp = spacy.load(SPACY_MODEL)

logging.info("Stripping leading and trailing whitespace from words")
anki_df[WORD_COL] = anki_df[WORD_COL].str.strip()

logging.info("Removing duplicate instances")
anki_df = anki_df.drop_duplicates(subset=[WORD_COL], keep="first")

logging.info("Lemmatizing Anki words")
lemmas = []
for batch_start in range(0, len(anki_df), LEMMA_BATCH_SIZE):
    batch = anki_df[WORD_COL][batch_start: batch_start + LEMMA_BATCH_SIZE]
    for doc in nlp.pipe(batch, batch_size=LEMMA_BATCH_SIZE):
        lemmas.append(doc[0].lemma_.lower())
anki_df[LEMMA_COL] = lemmas

logging.info("Converting lemmas to lowercase")
anki_df[LEMMA_COL] = anki_df[LEMMA_COL].str.lower()

logging.info("Merging Anki words with frequency data")
anki_df = anki_df.merge(frequency_df, left_on=LEMMA_COL, right_on=LEMMA_COL, how="left")

logging.info("Filling missing frequency and rank values with 0")
anki_df[FREQUENCY_COL] = anki_df[FREQUENCY_COL].fillna(0).astype(int)
anki_df[RANK_COL] = anki_df[RANK_COL].fillna(0).astype(int)

logging.info("Removing duplicate instances based on the %s column.", WORD_COL)
anki_df = anki_df.drop_duplicates(subset=[WORD_COL], keep="first")

logging.info("Sorting the Anki words")
anki_df = anki_df.sort_values(by=WORD_COL, ascending=True)

logging.info("Saving Anki frequencies to: %s", ANKIFREQ_FILE)
anki_df[[WORD_COL, LEMMA_COL, FREQUENCY_COL, RANK_COL]].to_csv(ANKIFREQ_FILE, index=False)

logging.info("Anki frequency generation process completed.")
