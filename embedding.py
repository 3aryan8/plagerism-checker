# ============================================================
# embedding.py — SBERT model loading + sentence encoding
# ============================================================

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from config import SBERT_MODEL


@st.cache_resource
def get_model() -> SentenceTransformer:
    """
    Lazy-load the SentenceTransformer model.
    Caches it directly using Streamlit cache to avoid repeated I/O.
    """
    return SentenceTransformer(SBERT_MODEL)


def encode_sentences(sentences: list[str]) -> np.ndarray:
    """
    Convert a list of sentences into an (N x D) embedding matrix.

    Args:
        sentences: list of sentence strings

    Returns:
        numpy array of shape (len(sentences), embedding_dim)
    """
    if not sentences:
        return np.empty((0, 384))           # 384-dim for MiniLM-L6-v2

    model = get_model()
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return embeddings


def encode_documents(
    processed_docs: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """
    Encode every document's sentences.

    Args:
        processed_docs: {doc_name: [sentence, ...]}

    Returns:
        {doc_name: embedding_matrix}
    """
    return {name: encode_sentences(sents) for name, sents in processed_docs.items()}
