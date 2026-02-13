"""Compares the given Anki collection with the processed word-frequency dictionary and
identifies missing/extra words and the words' frequencies."""

import logging
import pandas as pd
import spacy
from configs import (
    INPUT_ANKI,
    ANKI_WORD_COL_INDEX,
    ANKI_SKIP_ROWS,
    ANKI_SEPERATOR,
    OUTPUT_ANKI,
    SPACY_MODEL,
    LEMMA_BATCH_SIZE,
    OUTPUT_FILE,
    LEMMA_COL_TITLE,
    FREQUENCY_COL_TITLE,
    WORD_COL_TITLE,
    RANK_COL_TITLE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting Anki frequency generation process.")

logging.info("Reading Anki collection from: %s", INPUT_ANKI)
anki_df = pd.read_csv(
    INPUT_ANKI,
    usecols=[ANKI_WORD_COL_INDEX],
    sep=ANKI_SEPERATOR,
    skiprows=ANKI_SKIP_ROWS,
    header=None,
    names=[WORD_COL_TITLE],
    index_col=None,
)

logging.info("Reading frequency data from: %s", OUTPUT_FILE)
frequency_df = pd.read_csv(
    OUTPUT_FILE,
    index_col=None,
    header=0,
    usecols=[LEMMA_COL_TITLE, FREQUENCY_COL_TITLE, RANK_COL_TITLE],
)

logging.info("Loading spaCy model: %s", SPACY_MODEL)
nlp = spacy.load(SPACY_MODEL)

logging.info("Lemmatizing Anki words")
anki_df[LEMMA_COL_TITLE] = [doc[-1].lemma_ for doc in nlp.pipe(anki_df[WORD_COL_TITLE], batch_size=LEMMA_BATCH_SIZE)]

logging.info("Merging Anki words with frequency data")
anki_df = anki_df.merge(frequency_df, left_on=LEMMA_COL_TITLE, right_on=LEMMA_COL_TITLE, how="left")

anki_df[FREQUENCY_COL_TITLE] = anki_df[FREQUENCY_COL_TITLE].fillna(0).astype(int)
anki_df[RANK_COL_TITLE] = anki_df[RANK_COL_TITLE].fillna(0).astype(int)

logging.info("Sorting the Anki words by frequency")
anki_df = anki_df.sort_values(by=FREQUENCY_COL_TITLE, ascending=False, na_position="last")

logging.info("Saving Anki frequencies to: %s", OUTPUT_ANKI)
anki_df[[WORD_COL_TITLE, LEMMA_COL_TITLE, FREQUENCY_COL_TITLE, RANK_COL_TITLE]].to_csv(OUTPUT_ANKI, index=False)

logging.info("Anki frequency generation process completed.")
