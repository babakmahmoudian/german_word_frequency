"""Configuration settings for the word frequency processing script."""

# Internal settings for the processing script
LEMMA_BATCH_SIZE = 1000
SPACY_MODEL = "de_core_news_lg"

# Column names used throughout the code and in output files
LEMMA_COL = "lemma"
MORPH_COL = "morph"
POS_COL = "pos"
FREQUENCY_COL = "frequency"
RANK_COL = "rank"
WORD_COL = "word"
ANKI_NOTETYPE_COL = "notetype"

# List the input files and their full path to be processed
# NOTE: All raw data files are assumed to follow the same format.
INPUT_PATH = "./input_data"
INPUT_PREFIX = "row_words_"
WORD_COL_INDEX = 1
FREQUENCY_COL_INDEX = 2
SEPARATOR = "\t"

# Configuration for processing the Anki collection
ANKI_REQUEST_TIMEOUT = 10
ANKI_API_URL = "http://127.0.0.1:8765"
ANKI_QUERY = "deck:My-German -note:MY-German-Paradigm"
ANKI_WORD_FIELD = "Deutsch"

# Specify the output file path for the processed word-frequency dictionary
# NOTE: The output files will be overwritten if it already exists.
FREQDICT_FILE = "./output_data/all_frequencies.csv"
ANKIFREQ_FILE = "./output_data/anki_frequencies.csv"

# Configuration for the final Anki deck output
MAX_RANK_TO_ADD = 5000
MIN_RANK_TO_REMOVE = 10000
