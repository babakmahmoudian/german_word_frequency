"""This module contains the main script for the project.
   It reads individual data files, creates a the consolidated raw dataset,
   and performs the cleaning and preprocessing steps. Afterwards, generates
   the word-frequency dictionary and saves it to a file.
"""

import os
import spacy
import pandas as pd
from tqdm import tqdm
from configs import (INPUT_PATH,
                     SEPARATOR,
                     WORD_COL_INDEX,
                     FREQUENCY_COL_INDEX,
                     OUTPUT_FILE,
                     LEMMA_BATCH_SIZE,
                     SPACY_MODEL)


def get_data() -> pd.DataFrame:
    """Reads the input files and returns a consolidated DataFrame."""

    df = pd.DataFrame()

    input_files = [f"{INPUT_PATH}/{filename}"
                   for filename in os.listdir(INPUT_PATH)]
    for filename in input_files:
        next_df = pd.read_csv(filename,
                              sep=SEPARATOR,
                              usecols=[WORD_COL_INDEX, FREQUENCY_COL_INDEX],
                              index_col=0,
                              header=None)
        df = pd.concat([df, next_df])

    # Aggregating the frequencies by word
    df = df.groupby(df.index).sum()

    return df


def process_words(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the given dataframe and returns the processed dataframe."""

    # Initially remove words with non-alphabetic characters
    # This might leave out some words containing invisible character.
    df = df[df.index.str.isalpha()]

    # Lowercase the words
    df.index = df.index.str.lower()

    # Lemmatization
    # TODO: Filter out proper nouns (people, places, etc.)
    lemmas = []
    for doc in tqdm(nlp.pipe(df.index, batch_size=LEMMA_BATCH_SIZE),
                    total=len(df),
                    desc="Lemmatizing words"):
        lemmas.append(doc[0].lemma_)

    # Create the output DataFrame
    df = pd.DataFrame(df[FREQUENCY_COL_INDEX].values, index=lemmas)

    # Aggregate the frequencies by lemma and sort
    df = df.groupby(df.index).sum().sort_values(
        by=df.columns[0], ascending=False)

    # Once again remove words with non-alphabetic characters
    # By now, lemmatization will have already transformed the non-alphabetic
    # characters into '-' or removed them.
    df = df[df.index.str.isalpha()]

    return df


print("Loading the spaCy model for lemmatization...")
nlp = spacy.load(SPACY_MODEL)

print("Reading the raw data files...")
freq_df = get_data()

print("Processing the words in the DataFrame...")
freq_df = process_words(freq_df)

print("Saving the processed DataFrame to a file...")
freq_df.to_csv(OUTPUT_FILE, header=False)
print("Done!")
