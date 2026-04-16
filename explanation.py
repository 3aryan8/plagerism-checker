# ============================================================
# explanation.py — OpenRouter API integration for AI explanations
# ============================================================

import requests
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
from detection import PairResult


# ── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(result: PairResult) -> str:
    """
    Construct a structured prompt from the detection result so the LLM
    can produce a meaningful plagiarism report.
    """
    top_pairs = result.matched_pairs[:5]   # limit to top 5 for prompt brevity

    pairs_text = ""
    for i, pair in enumerate(top_pairs, 1):
        pairs_text += (
            f"\n  Pair {i} (similarity: {pair.similarity:.2%}):\n"
            f"    Source sentence : \"{pair.sentence_a}\"\n"
            f"    Target sentence : \"{pair.sentence_b}\"\n"
        )

    prompt = f"""You are an academic integrity expert. Analyse the following plagiarism detection report and write a clear, professional explanation.

REPORT SUMMARY
--------------
Document A : {result.doc_a}
Document B : {result.doc_b}
Plagiarism in Document A : {result.plagiarism_pct_a:.1f}%
Plagiarism in Document B : {result.plagiarism_pct_b:.1f}%
Overall plagiarism score : {result.overall_pct:.1f}%
Number of flagged sentence pairs : {len(result.matched_pairs)}

TOP MATCHING SENTENCE PAIRS
----------------------------{pairs_text}

YOUR TASK
---------
1. Summarise the overall plagiarism level in 2-3 sentences.
2. Explain what the matched sentences reveal about how the content was reused (paraphrase, near-copy, idea borrowing, etc.).
3. Give a brief recommendation (cite sources, rewrite, etc.).
4. Keep the tone professional and concise (max 200 words).
"""
    return prompt


# ── API call ─────────────────────────────────────────────────────────────────

def generate_explanation(result: PairResult) -> str:
    """
    Call the OpenRouter API and return an AI-generated explanation string.
    Returns an error message string if the API call fails.
    """
    if not OPENROUTER_API_KEY:
        return (
            "⚠️  OpenRouter API key not configured.  "
            "Set the `OPENROUTER_API_KEY` environment variable or add it to the `.env` file "
            "to enable AI explanations."
        )

    headers = {
        "Authorization":  f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   "https://plagiarism-detector.app",   # required by OpenRouter
        "X-Title":        "Semantic Plagiarism Detector",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role":    "user",
                "content": _build_prompt(result),
            }
        ],
        "max_tokens": 500,
        "temperature": 0.4,
    }

    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        return "⚠️  The OpenRouter API request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️  HTTP error from OpenRouter API: {e.response.status_code} — {e.response.text}"
    except requests.exceptions.ConnectionError:
        return "⚠️  Could not reach the OpenRouter API. Check your internet connection."
    except (KeyError, IndexError):
        return "⚠️  Unexpected response format from OpenRouter API."
    except Exception as e:
        return f"⚠️  Unexpected error: {e}"
