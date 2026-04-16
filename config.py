# ============================================================
# config.py — Central configuration for the plagiarism system
# ============================================================

import os

# ── OpenRouter API ──────────────────────────────────────────
# Set the key in a .env file or as an environment variable:
#   export OPENROUTER_API_KEY="sk-or-v1-..."
# Never hard-code your key directly in this file.
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL    = "mistralai/mistral-7b-instruct:free"  # :free suffix required for free-tier accounts

# ── Sentence-Transformer model ──────────────────────────────
SBERT_MODEL = "all-MiniLM-L6-v2"

# ── Detection defaults ──────────────────────────────────────
DEFAULT_THRESHOLD   = 0.75   # cosine-similarity cut-off
MIN_SENTENCE_TOKENS = 5      # ignore very short sentences (< N words)

# ── Heatmap palette ─────────────────────────────────────────
HEATMAP_COLORMAP = "YlOrRd"  # yellow → orange → red

# ── UI copy ─────────────────────────────────────────────────
APP_TITLE    = "🔍 Semantic Plagiarism Detection System"
APP_SUBTITLE = "Detect meaning-level similarity between documents using AI embeddings"
