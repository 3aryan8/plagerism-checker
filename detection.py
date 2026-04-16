# ============================================================
# detection.py — Core plagiarism detection logic
# ============================================================

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MatchedPair:
    """Represents a single plagiarised sentence pair."""
    doc_a:       str
    doc_b:       str
    sentence_a:  str
    sentence_b:  str
    similarity:  float
    index_a:     int
    index_b:     int


@dataclass
class PairResult:
    """Aggregated result for one document pair."""
    doc_a:              str
    doc_b:              str
    plagiarism_pct_a:   float          # % of doc-A sentences flagged
    plagiarism_pct_b:   float          # % of doc-B sentences flagged
    overall_pct:        float          # combined %
    matched_pairs:      list[MatchedPair] = field(default_factory=list)
    flagged_indices_a:  set[int]       = field(default_factory=set)
    flagged_indices_b:  set[int]       = field(default_factory=set)


def detect_plagiarism(
    doc_a_name:   str,
    doc_b_name:   str,
    sentences_a:  list[str],
    sentences_b:  list[str],
    sim_matrix:   np.ndarray,
    threshold:    float,
) -> PairResult:
    """
    Apply threshold to the similarity matrix to flag plagiarised sentences.

    Strategy:
      • For every sentence in doc-A, find its best match in doc-B.
      • If that best match score ≥ threshold → mark both sentences as plagiarised.

    Args:
        doc_a_name / doc_b_name : display names of the two documents
        sentences_a / sentences_b : tokenised sentence lists
        sim_matrix : (M x N) cosine similarity matrix
        threshold  : user-defined cut-off (0..1)

    Returns:
        PairResult with match details and plagiarism percentages
    """
    matched_pairs:     list[MatchedPair] = []
    flagged_indices_a: set[int] = set()
    flagged_indices_b: set[int] = set()

    m = len(sentences_a)
    n = len(sentences_b)

    if m == 0 or n == 0:
        return PairResult(
            doc_a=doc_a_name,
            doc_b=doc_b_name,
            plagiarism_pct_a=0.0,
            plagiarism_pct_b=0.0,
            overall_pct=0.0,
        )

    seen_pairs = set()

    # --- Pass 1: scan from doc-A → doc-B --------------------------------
    for i, sent_a in enumerate(sentences_a):
        best_j   = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])

        if best_sim >= threshold:
            flagged_indices_a.add(i)
            flagged_indices_b.add(best_j)
            seen_pairs.add((i, best_j))
            matched_pairs.append(
                MatchedPair(
                    doc_a=doc_a_name,
                    doc_b=doc_b_name,
                    sentence_a=sent_a,
                    sentence_b=sentences_b[best_j],
                    similarity=best_sim,
                    index_a=i,
                    index_b=best_j,
                )
            )

    # --- Pass 2: also scan doc-B → doc-A (catches plagiarism from B's POV)
    for j, sent_b in enumerate(sentences_b):
        best_i   = int(np.argmax(sim_matrix[:, j]))
        best_sim = float(sim_matrix[best_i, j])

        if best_sim >= threshold:
            flagged_indices_a.add(best_i)
            flagged_indices_b.add(j)
            # avoid duplicate pairs already captured in pass 1
            if (best_i, j) not in seen_pairs:
                seen_pairs.add((best_i, j))
                matched_pairs.append(
                    MatchedPair(
                        doc_a=doc_a_name,
                        doc_b=doc_b_name,
                        sentence_a=sentences_a[best_i],
                        sentence_b=sent_b,
                        similarity=best_sim,
                        index_a=best_i,
                        index_b=j,
                    )
                )

    # Sort by similarity descending so the most suspicious pairs appear first
    matched_pairs.sort(key=lambda p: p.similarity, reverse=True)

    pct_a   = (len(flagged_indices_a) / m) * 100 if m > 0 else 0.0
    pct_b   = (len(flagged_indices_b) / n) * 100 if n > 0 else 0.0
    overall_flagged = len(flagged_indices_a) + len(flagged_indices_b)
    overall = (overall_flagged / (m + n)) * 100 if (m + n) > 0 else 0.0

    return PairResult(
        doc_a=doc_a_name,
        doc_b=doc_b_name,
        plagiarism_pct_a=round(pct_a,   2),
        plagiarism_pct_b=round(pct_b,   2),
        overall_pct=round(overall, 2),
        matched_pairs=matched_pairs,
        flagged_indices_a=flagged_indices_a,
        flagged_indices_b=flagged_indices_b,
    )


def run_all_detections(
    processed_docs:   dict[str, list[str]],
    pair_matrices:    dict[tuple[str, str], np.ndarray],
    threshold:        float,
) -> list[PairResult]:
    """
    Run detection for every document pair.

    Returns:
        list of PairResult objects, one per unique pair
    """
    results = []
    for (name_a, name_b), matrix in pair_matrices.items():
        result = detect_plagiarism(
            doc_a_name=name_a,
            doc_b_name=name_b,
            sentences_a=processed_docs[name_a],
            sentences_b=processed_docs[name_b],
            sim_matrix=matrix,
            threshold=threshold,
        )
        results.append(result)
    return results
