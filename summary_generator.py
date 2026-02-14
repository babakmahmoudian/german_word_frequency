"""Module uses the generated anki frequencies to summarize the anki collection
and generate a report.
It compares the anki words with the top 5000 most frequents in the frequency
dictionary; pinpointing which anki words are in the top 5000 and which are not.
It also suggests which of the top 5000 words are missing from the anki collection.
"""

import logging
import pandas as pd
from configs import (
    OUTPUT_ANKI,
    OUTPUT_ANKI_SUMMARY,
    LEMMA_COL_TITLE,
    FREQUENCY_COL_TITLE,
    RANK_COL_TITLE,
    WORD_COL_TITLE,
    OUTPUT_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting the generation of the Anki collection summary report...")

logging.info("Reading Anki collection data from: %s", OUTPUT_ANKI)
df_anki = pd.read_csv(
    OUTPUT_ANKI, index_col=None, usecols=[WORD_COL_TITLE, LEMMA_COL_TITLE, FREQUENCY_COL_TITLE, RANK_COL_TITLE]
)

logging.info("Reading frequency data from: %s", OUTPUT_FILE)
df_freq = pd.read_csv(OUTPUT_FILE, index_col=None, usecols=[WORD_COL_TITLE, LEMMA_COL_TITLE, RANK_COL_TITLE])

logging.info("Extracting Anki words in the top 5000 most frequent words")
anki_in_top_5000 = (
    df_anki[(df_anki[RANK_COL_TITLE] > 0) & (df_anki[RANK_COL_TITLE] <= 5000)].copy().sort_values(by=WORD_COL_TITLE)
)

logging.info("Extracting Anki words NOT in the top 5000 most frequent words")
anki_not_in_top_5000 = (
    df_anki[(df_anki[RANK_COL_TITLE] > 5000) | (df_anki[RANK_COL_TITLE] == 0)].copy().sort_values(by=RANK_COL_TITLE)
)

logging.info("Identifying top 5000 most frequent words missing from Anki collection")
df_freq_lower = df_freq.copy()
df_freq_lower[LEMMA_COL_TITLE] = df_freq_lower[LEMMA_COL_TITLE].str.lower()
df_anki_lower = df_anki.copy()
df_anki_lower[LEMMA_COL_TITLE] = df_anki_lower[LEMMA_COL_TITLE].str.lower()
top_5000_missing_from_anki = (
    df_freq_lower[df_freq_lower[RANK_COL_TITLE] <= 5000][
        ~df_freq_lower[LEMMA_COL_TITLE].isin(df_anki_lower[LEMMA_COL_TITLE])
    ]
    .copy()
    .sort_values(by=RANK_COL_TITLE)
)

logging.info("Saving summary report...")
with open(OUTPUT_ANKI_SUMMARY, "w", encoding="utf-8") as f:
    f.write("Anki Words in the Top 5000 Most Frequent Words:\n")
    f.write(anki_in_top_5000.to_string(index=False))
    f.write("\n\nAnki Words NOT in the Top 5000 Most Frequent Words:\n")
    f.write(anki_not_in_top_5000.to_string(index=False))
    f.write("\n\nTop 5000 Most Frequent Words Missing from Anki Collection:\n")
    f.write(top_5000_missing_from_anki.to_string(index=False))
logging.info("Summary report saved to: %s", OUTPUT_ANKI_SUMMARY)
