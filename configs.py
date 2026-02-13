"""Configuration settings for the word frequency processing script."""

# List the input files and their full path to be processed
# NOTE: All files in the input directory are considered relevant.
# NOTE: All data files are assumed to follow the same format.
INPUT_PATH = './data'
WORD_COL_INDEX = 1
FREQUENCY_COL_INDEX = 2
SEPARATOR = '\t'

# Specify the output file path for the processed word-frequency dictionary
# NOTE: The output file will be overwritten if it already exists.
OUTPUT_FILE = './output/processed_words.csv'

LEMMA_BATCH_SIZE = 1000
SPACY_MODEL = "de_core_news_lg"
