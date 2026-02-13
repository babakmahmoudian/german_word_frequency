"""This module contains the main script for the project.
It reads individual data files, creates a the consolidated raw dataset, and performs the cleaning and preprocessing
steps. Afterwards, generates the word-frequency dictionary and saves it to a file.
"""

import os
import logging
import spacy
import pandas as pd
from configs import (
    LEMMA_BATCH_SIZE,
    SPACY_MODEL,
    INPUT_PATH,
    WORD_COL_INDEX,
    FREQUENCY_COL_INDEX,
    SEPARATOR,
    INPUT_PREFIX,
    WORD_COL_TITLE,
    FREQUENCY_COL_TITLE,
    LEMMA_COL_TITLE,
    RANK_COL_TITLE,
    OUTPUT_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_data() -> pd.DataFrame:
    """Reads the input files and returns a consolidated DataFrame."""

    df = pd.DataFrame()

    input_files = [
        f"{INPUT_PATH}/{filename}" for filename in os.listdir(INPUT_PATH) if filename.startswith(INPUT_PREFIX)
    ]
    for filename in input_files:
        logging.info("Reading file: %s", filename)
        next_df = pd.read_csv(
            filename,
            sep=SEPARATOR,
            usecols=[WORD_COL_INDEX, FREQUENCY_COL_INDEX],
            header=None,
        )
        df = pd.concat([df, next_df])

    df.columns = [WORD_COL_TITLE, FREQUENCY_COL_TITLE]

    logging.info("Consolidating data by summing frequencies of duplicate words...")
    df = df.groupby(WORD_COL_TITLE, as_index=False).agg({FREQUENCY_COL_TITLE: "sum"})

    return df


def process_words(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the given dataframe and returns the processed dataframe."""

    logging.info("Removing words with non-alphabetic characters...")
    # FIXME: Fails to filter out words with invisible unicode characters.
    df = df[df[WORD_COL_TITLE].str.isalpha()]

    logging.info("Converting words to lowercase...")
    df[WORD_COL_TITLE] = df[WORD_COL_TITLE].str.lower()

    total_batches = (len(df) + LEMMA_BATCH_SIZE - 1) // LEMMA_BATCH_SIZE
    next_log_pct = 1
    logging.info("Lemmatizing words using spaCy; total words: %d; total batched: %d", len(df), total_batches)
    lemmas = []
    for batch_start in range(0, len(df), LEMMA_BATCH_SIZE):
        batch = df[WORD_COL_TITLE][batch_start : batch_start + LEMMA_BATCH_SIZE]
        for doc in nlp.pipe(batch, batch_size=LEMMA_BATCH_SIZE):
            # TODO: Filter out proper nouns (people, places, etc.)
            lemmas.append(doc[0].lemma_)

        batch_num = batch_start // LEMMA_BATCH_SIZE + 1
        pct_complete = batch_num / total_batches * 100
        if pct_complete >= next_log_pct:
            logging.info("Processed batch %d/%d (%.2f%% complete)", batch_num, total_batches, pct_complete)
            next_log_pct += 1
    df[LEMMA_COL_TITLE] = lemmas
    logging.info("Lemmatization completed.")

    return df


logging.info("Starting the dictionary generation process...")

logging.info("Loading spaCy model: %s", SPACY_MODEL)
nlp = spacy.load(SPACY_MODEL)

logging.info("Reading and consolidating input data files...")
freq_df = get_data()

# NOTE: Use this for testing purposes:
# # Shrink the dataset for testing
# MAX_ROWS = 100000
# logging.info("Shrinking the dataset to the top %d rows for testing purposes...", MAX_ROWS)
# freq_df = freq_df.head(MAX_ROWS)

logging.info("Processing words (lowercasing and lemmatization)...")
freq_df = process_words(freq_df)

logging.info("Aggregating frequencies by lemma...")
freq_df = freq_df.groupby(LEMMA_COL_TITLE, as_index=False).agg({FREQUENCY_COL_TITLE: "sum", WORD_COL_TITLE: "first"})

logging.info("Calculating ranks based on frequency...")
freq_df[RANK_COL_TITLE] = freq_df[FREQUENCY_COL_TITLE].rank(method="min", ascending=False).astype(int)

logging.info("Sorting the DataFrame by frequency in descending order...")
freq_df = freq_df.sort_values(by=FREQUENCY_COL_TITLE, ascending=False)

logging.info("Saving the dictionary to file: %s", OUTPUT_FILE)
freq_df[[WORD_COL_TITLE, FREQUENCY_COL_TITLE, LEMMA_COL_TITLE, RANK_COL_TITLE]].to_csv(
    OUTPUT_FILE,
    index=False,
)

logging.info("Dictionary generation process completed successfully.")
