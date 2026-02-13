"""Module uses the generated anki frequencies to summarize the anki collection
and generate a report.
It compares the anki words with the top 5000 most frequents in the frequency
dictionary; pinpointing which anki words are in the top 5000 and which are not.
It also suggests which of the top 5000 words are missing from the anki collection.
"""

import pandas as pd
from configs import (
    OUTPUT_ANKI, OUTPUT_ANKI_SUMMARY,
    LEMMA_COL_TITLE, FREQUENCY_COL_TITLE, RANK_COL_TITLE,
    OUTPUT_FILE
)

print("Loading data...")
df_anki = pd.read_csv(OUTPUT_ANKI, index_col=0)
df_freq = pd.read_csv(OUTPUT_FILE, index_col=None)

# FIXME: Remove this and read the rank column directly from the frequency file
print("Calculating ranks...")
df_freq[RANK_COL_TITLE] = df_freq[FREQUENCY_COL_TITLE].rank(
    method='min', ascending=False).astype(int)


print("Annotating Anki data with frequency ranks...")
df_anki = df_anki.merge(
    df_freq[[LEMMA_COL_TITLE, RANK_COL_TITLE]],
    left_on=LEMMA_COL_TITLE, right_on=LEMMA_COL_TITLE,
    how='left')

print("Extracting Anki words in the top 5000 most frequent words...")
anki_in_top_5000 = df_anki[df_anki[RANK_COL_TITLE] <= 5000].copy()

print("Extracting Anki words NOT in the top 5000 most frequent words...")
anki_not_in_top_5000 = df_anki[df_anki[RANK_COL_TITLE] > 5000].copy()

print("Identifying top 5000 most frequent words missing from Anki...")
top_5000_missing_from_anki = df_freq[df_freq[RANK_COL_TITLE] <= 5000][
    ~df_freq[LEMMA_COL_TITLE].isin(df_anki[LEMMA_COL_TITLE])
].copy()

print("Saving summary report...")
with open(OUTPUT_ANKI_SUMMARY, 'w', encoding='utf-8') as f:
    f.write("Anki Words in the Top 5000 Most Frequent Words:\n")
    f.write(anki_in_top_5000.to_string(index=False))
    f.write("\n\nAnki Words NOT in the Top 5000 Most Frequent Words:\n")
    f.write(anki_not_in_top_5000.to_string(index=False))
    f.write("\n\nTop 5000 Most Frequent Words Missing from Anki Collection:\n")
    f.write(top_5000_missing_from_anki.to_string(index=False))
print("Summary report saved to:", OUTPUT_ANKI_SUMMARY)
