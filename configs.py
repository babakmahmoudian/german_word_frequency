"""Configuration settings for the word frequency processing script."""

LEMMA_BATCH_SIZE = 1000
SPACY_MODEL = "de_core_news_lg"

LEMMA_COL_TITLE = 'lemma'
FREQUENCY_COL_TITLE = 'frequency'
RANK_COL_TITLE = 'rank'
WORD_COL_TITLE = 'word'

# List the input files and their full path to be processed
# NOTE: All data files are assumed to follow the same format.
INPUT_PATH = './input_data'
INPUT_PREFIX = 'row_words_'
WORD_COL_INDEX = 1
FREQUENCY_COL_INDEX = 2
SEPARATOR = '\t'

# Specify the output file path for the processed word-frequency dictionary
# NOTE: The output file will be overwritten if it already exists.
OUTPUT_FILE = './output_data/all_frequencies.csv'

# Configuration for processing the Anki collection
INPUT_ANKI = './input_data/anki_words.tsv'
ANKI_SEPERATOR = '\t'
ANKI_SKIP_ROWS = 2
ANKI_WORD_COL_INDEX = 0
OUTPUT_ANKI = './output_data/anki_frequencies.csv'
OUTPUT_ANKI_SUMMARY = './output_data/anki_summary_report.txt'