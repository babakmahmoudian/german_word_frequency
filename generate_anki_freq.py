"""Compares the given Anki collection with the processed word-frequency
dictionary and identifies missing/extra words and the words' frequencies."""


import pandas as pd
import spacy
from tqdm import tqdm
from configs import (
    INPUT_ANKI, ANKI_WORD_COL_INDEX, ANKI_SKIP_ROWS, ANKI_SEPERATOR,
    OUTPUT_ANKI,
    SPACY_MODEL, LEMMA_BATCH_SIZE,
    OUTPUT_FILE,
    LEMMA_COL_TITLE, FREQUENCY_COL_TITLE,
)

print("Reading the input data files...")
anki_df = pd.read_csv(INPUT_ANKI,
                      usecols=[ANKI_WORD_COL_INDEX],
                      sep=ANKI_SEPERATOR,
                      skiprows=ANKI_SKIP_ROWS,
                      header=None,
                      index_col=0)
frequency_df = pd.read_csv(OUTPUT_FILE, index_col=0)

print("Loading the spaCy model for lemmatization...")
nlp = spacy.load(SPACY_MODEL)

print("Lemmatizing Anki words...")
anki_lemmas = []
for doc in tqdm(nlp.pipe(anki_df.index, batch_size=LEMMA_BATCH_SIZE),
                total=len(anki_df),
                desc="Lemmatizing words"):
    anki_lemmas.append(doc[-1].lemma_)
anki_df[LEMMA_COL_TITLE] = anki_lemmas

print("Merging the Anki words with their frequencies...")
anki_df = anki_df.merge(frequency_df, left_on=LEMMA_COL_TITLE,
                        right_index=True, how='left')

anki_df[FREQUENCY_COL_TITLE] = anki_df[FREQUENCY_COL_TITLE].fillna(0).astype(int)

print("Sorting the Anki words by frequency...")
anki_df = anki_df.sort_values(by=FREQUENCY_COL_TITLE, ascending=False, na_position='last')

print("Saving the Anki words with their frequencies to a file...")
anki_df.to_csv(OUTPUT_ANKI, header=[LEMMA_COL_TITLE, FREQUENCY_COL_TITLE])
print("Anki frequencies saved to:", OUTPUT_ANKI)
