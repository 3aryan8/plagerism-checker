# ============================================================
# preprocessing.py — Text cleaning + sentence tokenization
# ============================================================

import re
import nltk
from config import MIN_SENTENCE_TOKENS

# Download the Punkt tokenizer data on first run
nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)


def _normalize_whitespace(text: str) -> str:
    """
    Normalise raw text without lowercasing:
      • collapse multiple spaces / tabs
      • collapse multiple blank lines
      • strip leading/trailing whitespace
    This version preserves original casing for display and embedding.
    """
    text = re.sub(r"[ \t]+", " ", text)      # collapse horizontal whitespace
    text = re.sub(r"\n{2,}", "\n", text)      # collapse blank lines
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Tokenise text into sentences using NLTK Punkt.
    • Original casing is preserved (important for SBERT accuracy).
    • Very short / empty sentences are filtered using a lowercase word count.

    Returns:
        list of sentence strings in original case.
    """
    normalized = _normalize_whitespace(text)
    raw_sents   = nltk.sent_tokenize(normalized)

    sentences = []
    for sent in raw_sents:
        sent = sent.strip()
        # filter on lowercased word count — keeps original casing in output
        if sent and len(sent.lower().split()) >= MIN_SENTENCE_TOKENS:
            sentences.append(sent)

    return sentences


def preprocess_documents(docs: dict[str, str]) -> dict[str, list[str]]:
    """
    Accepts  {doc_name: raw_text, ...}
    Returns  {doc_name: [sentence, sentence, ...], ...}
    """
    return {name: split_sentences(text) for name, text in docs.items()}
