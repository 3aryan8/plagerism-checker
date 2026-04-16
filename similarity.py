# ============================================================
# similarity.py — Pairwise cosine-similarity matrix logic
# ============================================================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


def compute_similarity_matrix(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> np.ndarray:
    """
    Compute an (M x N) cosine-similarity matrix between two embedding sets.

    Args:
        embeddings_a: shape (M, D)
        embeddings_b: shape (N, D)

    Returns:
        similarity matrix of shape (M, N)
        matrix[i][j] = similarity between sentence i of doc A
                        and sentence j of doc B
    """
    if embeddings_a.shape[0] == 0 or embeddings_b.shape[0] == 0:
        return np.zeros((embeddings_a.shape[0], embeddings_b.shape[0]))

    return cosine_similarity(embeddings_a, embeddings_b)


def compute_all_pairwise_similarities(
    embeddings: dict[str, np.ndarray]
) -> dict[tuple[str, str], np.ndarray]:
    """
    For every unique pair of documents, compute a similarity matrix.

    Args:
        embeddings: {doc_name: embedding_matrix}

    Returns:
        {(doc_a, doc_b): similarity_matrix, ...}
        (only upper-triangle pairs, i.e. a < b alphabetically)
    """
    doc_names = list(embeddings.keys())
    pair_matrices: dict[tuple[str, str], np.ndarray] = {}

    for name_a, name_b in combinations(doc_names, 2):
        matrix = compute_similarity_matrix(
            embeddings[name_a],
            embeddings[name_b],
        )
        pair_matrices[(name_a, name_b)] = matrix

    return pair_matrices
