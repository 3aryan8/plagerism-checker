# ============================================================
# utils.py — General-purpose helper functions
# ============================================================

from detection import PairResult


def read_uploaded_file(uploaded_file) -> str:
    """
    Read a Streamlit UploadedFile object and return its content as a string.
    Handles both UTF-8 and latin-1 encodings.
    """
    raw = uploaded_file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def score_to_label(pct: float) -> tuple[str, str]:
    """
    Map a plagiarism percentage to a human-readable label and a hex colour.

    Returns:
        (label, hex_colour)
    """
    if pct >= 70:
        return "🔴 High Plagiarism",       "#ff4e4e"
    elif pct >= 40:
        return "🟠 Moderate Plagiarism",   "#ff9a3c"
    elif pct >= 15:
        return "🟡 Low Plagiarism",        "#facc15"
    else:
        return "🟢 Minimal / No Plagiarism", "#4ade80"


def format_summary_stats(result: PairResult) -> dict[str, str]:
    """
    Extract key stats from a PairResult as a display-ready dict.
    """
    return {
        "Document A":            result.doc_a,
        "Document B":            result.doc_b,
        f"{result.doc_a} flagged": f"{result.plagiarism_pct_a:.1f}%",
        f"{result.doc_b} flagged": f"{result.plagiarism_pct_b:.1f}%",
        "Overall score":         f"{result.overall_pct:.1f}%",
        "Matched pairs":         str(len(result.matched_pairs)),
    }


def truncate(text: str, max_len: int = 120) -> str:
    """Truncate text for display purposes."""
    return (text[:max_len] + "…") if len(text) > max_len else text


def results_to_csv(results) -> str:
    """
    Convert a list of PairResult objects to a CSV string for download.
    Columns: doc_a, doc_b, sentence_a, sentence_b, similarity_pct
    """
    lines = ["doc_a,doc_b,sentence_a,sentence_b,similarity_pct"]
    for result in results:
        for pair in result.matched_pairs:
            def _esc(s: str) -> str:
                return '"' + s.replace('"', '""') + '"'
            lines.append(
                f"{_esc(pair.doc_a)},{_esc(pair.doc_b)},"
                f"{_esc(pair.sentence_a)},{_esc(pair.sentence_b)},"
                f"{pair.similarity * 100:.2f}"
            )
    return "\n".join(lines)
