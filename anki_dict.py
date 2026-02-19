"""This script retrieves notes from Anki using AnkiConnect, processes them, and extracts the lemmas"""

import requests
import spacy
import pandas as pd


BATCH_SIZE = 1000
SPACY_MODEL = "de_core_news_lg"

POSES = {
    "My-German-Noun": "NOUN",
    "My-German-Verb": "VERB",
    "My-German-Modifier": "MODIFIER",
}

URL = "http://127.0.0.1:8765"
QUERY = " OR ".join([f"note:{notetype}" for notetype in POSES.keys()])

WORD = "word"
MAIN_WORD = "main_word"
LEMMA = "lemma"
POS = "pos"

OUTPUT_FILE = "./data/processed/anki_dict.csv"


def extract_main_word(word: str) -> str:
    word_parts = word.split()
    return word_parts[len(word_parts) // 2]


def lemmatize_text(words: pd.Series) -> dict:
    """Processes an input list of words and generates word-frequency dict for it."""
    result = dict()
    for doc in nlp.pipe(words, batch_size=BATCH_SIZE):
        token = doc[0]
        result[token.text] = token.lemma_

    return result


print("Retreieving note IDs from Anki")
note_ids = requests.post(
    URL, timeout=(0.5, 3), json={"action": "findNotes", "version": 6, "params": {"query": QUERY}}
).json()["result"]

print("Retreieving note info from Anki")
note_info = requests.post(
    URL, timeout=(0.5, 3), json={"action": "notesInfo", "version": 6, "params": {"notes": note_ids}}
).json()["result"]

print(f"Total of {len(note_info)} notes were fetched.")

anki_words = []
anki_notetypes = []
for note in note_info:
    anki_notetypes.append(POSES[note["modelName"]])
    anki_words.append(note["fields"]["Deutsch"]["value"])

print("Setting up the Anki dataset for lemmatization")
df = pd.DataFrame(
    {
        POS: anki_notetypes,
        WORD: anki_words,
        MAIN_WORD: [extract_main_word(word) for word in anki_words],
    }
)

print("Loading spaCy model")
nlp = spacy.load(SPACY_MODEL)

print("Extracting lemmas...")
word_lemmas = lemmatize_text(df[MAIN_WORD])
print("Finished lemmatization")

print("Updating the Anki dataset")
df[LEMMA] = df[MAIN_WORD].apply(lambda w: word_lemmas[w])
df = df.drop([MAIN_WORD], axis="columns")

print("Exporting the results")
df.to_csv(OUTPUT_FILE, index=False)
