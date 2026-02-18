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
    ANKI_NOTETYPE_COL,
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

# NOTE: For this step, Anki should be open with the addon "AnkiConnect" enabled
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

logging.info("Extracting words and notetypes from the note information")
anki_words = []
anki_notetypes = []
for note in note_info:
    word = note["fields"][ANKI_WORD_FIELD]["value"]
    notetype = note["modelName"]
    anki_words.append(word)
    anki_notetypes.append(notetype)

logging.info("Setting up the output dataframe")
extracted_words = []
extracted_orig_words = []
extracted_notetypes = []
for _, (notetype, whole_word) in enumerate(zip(anki_notetypes, anki_words)):
    combined_words = re.sub(r"[^\w\s]", "", whole_word)
    single_words = combined_words.split()
    for single_word in single_words:
        if (single_word in anki_words) and (len(single_words) > 1):
            continue
        extracted_words.append(single_word)
        extracted_orig_words.append(whole_word)
        extracted_notetypes.append(notetype)
anki_df = pd.DataFrame(
    {ANKI_NOTETYPE_COL: extracted_notetypes, ANKI_WORD_FIELD: extracted_orig_words, WORD_COL: extracted_words}
)

logging.info("Converting word to lowercase")
anki_df[WORD_COL] = anki_df[WORD_COL].str.lower()

logging.info("Stripping leading and trailing whitespace from words")
anki_df[WORD_COL] = anki_df[WORD_COL].str.strip()

logging.info("Removing duplicate instances")
anki_df = anki_df.drop_duplicates(subset=[WORD_COL], keep="first")

logging.info("Loading spaCy model: %s", SPACY_MODEL)
nlp = spacy.load(SPACY_MODEL)

logging.info("Lemmatizing Anki words...")
lemmas = []
for batch_start in range(0, len(anki_df), LEMMA_BATCH_SIZE):
    batch = anki_df[WORD_COL][batch_start : batch_start + LEMMA_BATCH_SIZE]
    for doc in nlp.pipe(batch, batch_size=LEMMA_BATCH_SIZE):
        lemmas.append(doc[0].lemma_.lower())
anki_df[LEMMA_COL] = lemmas
logging.info("Finished Lemmatization")

logging.info("Converting lemmas to lowercase")
anki_df[LEMMA_COL] = anki_df[LEMMA_COL].str.lower()

logging.info("Merging Anki words with frequency data")
anki_df = anki_df.merge(frequency_df, left_on=LEMMA_COL, right_on=LEMMA_COL, how="left")

logging.info("Filling missing frequency and rank values with 0")
anki_df[FREQUENCY_COL] = anki_df[FREQUENCY_COL].fillna(0).astype(int)
anki_df[RANK_COL] = anki_df[RANK_COL].fillna(0).astype(int)

logging.info("Sorting the Anki words")
anki_df = anki_df.sort_values(by=FREQUENCY_COL, ascending=False)

logging.info("Saving Anki frequencies to: %s", ANKIFREQ_FILE)
anki_df[[ANKI_NOTETYPE_COL, ANKI_WORD_FIELD, WORD_COL, LEMMA_COL, FREQUENCY_COL, RANK_COL]].to_csv(
    ANKIFREQ_FILE, index=False
)

logging.info("Anki frequency generation process completed.")
