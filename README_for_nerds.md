# Semantic Plagiarism Detection System — Technical Reference

> **Audience:** Developers who need to understand, debug, or extend this codebase.  
> All explanations are traceable to specific functions, line numbers, and data types.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Map](#2-repository-map)
3. [Execution Flow (End-to-End)](#3-execution-flow-end-to-end)
4. [Module-by-Module Reference](#4-module-by-module-reference)
   - [config.py](#41-configpy)
   - [preprocessing.py](#42-preprocessingpy)
   - [embedding.py](#43-embeddingpy)
   - [similarity.py](#44-similaritypy)
   - [detection.py](#45-detectionpy)
   - [explanation.py](#46-explanationpy)
   - [visualization.py](#47-visualizationpy)
   - [utils.py](#48-utilspy)
   - [app.py](#49-apppy)
5. [Model: all-MiniLM-L6-v2](#5-model-all-minilm-l6-v2)
6. [Data Structures](#6-data-structures)
7. [Detection Algorithm](#7-detection-algorithm)
8. [OpenRouter Integration](#8-openrouter-integration)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Environment & Dependency Assumptions](#10-environment--dependency-assumptions)
11. [Known Limitations](#11-known-limitations)

---

## 1. System Overview

This is a **pure inference system** — no model training occurs at runtime. The pipeline:

1. Tokenises raw text into sentences (NLTK Punkt)
2. Embeds each sentence into a 384-dimensional dense vector (SBERT `all-MiniLM-L6-v2`)
3. Computes pairwise cosine similarity matrices between all document pairs
4. Applies a bidirectional greedy best-match scan against a user-defined threshold
5. Renders highlighted HTML, a seaborn heatmap, and an AI explanation via the OpenRouter API (Mistral 7B)

**Plagiarism detection is entirely threshold-based.** No classifier is trained. The threshold is a hard cosine-similarity cutoff configurable at runtime.

---

## 2. Repository Map

```
NLP/
├── app.py            Streamlit entry point; orchestrates all stages
├── config.py         All constants: model names, thresholds, API keys
├── preprocessing.py  Text → sentence list (NLTK Punkt + regex normalisation)
├── embedding.py      Sentence list → (N, 384) numpy array via SBERT
├── similarity.py     (N, 384) × (M, 384) → (N, M) cosine similarity matrix
├── detection.py      (N, M) matrix + threshold → flagged pairs + %
├── explanation.py    PairResult → prompt → OpenRouter API → string
├── visualization.py  PairResult → HTML table; sentences → HTML highlights;
│                     similarity matrix → seaborn PNG heatmap
├── utils.py          File I/O helpers, CSV serialiser, score → label mapping
├── requirements.txt  Python package pins
└── .env              Not committed; stores OPENROUTER_API_KEY
```

**Dependency graph (import direction):**

```
app.py
  ├── config.py              (no upstream deps)
  ├── preprocessing.py  ←── config.MIN_SENTENCE_TOKENS
  ├── embedding.py      ←── config.SBERT_MODEL
  ├── similarity.py          (only numpy + sklearn)
  ├── detection.py           (only numpy; defines PairResult, MatchedPair)
  ├── explanation.py    ←── config.OPENROUTER_*, detection.PairResult
  ├── visualization.py  ←── config.HEATMAP_COLORMAP, detection.PairResult
  └── utils.py          ←── detection.PairResult
```

Nothing imports `app.py`. All application state lives in `st.session_state` (implicitly, via Streamlit re-runs).

---

## 3. Execution Flow (End-to-End)

Every user click of **"🔍 Check Plagiarism"** triggers a top-to-bottom re-execution of `app.py`. Streamlit's execution model is synchronous within a session.

```
User submits documents
        │
        ▼
[app.py] reads raw text → docs_raw: dict[str, str]
        │
        ▼ st.spinner
[preprocessing.preprocess_documents(docs_raw)]
  For each document:
    _normalize_whitespace(text)          # regex: collapse whitespace + blank lines
    nltk.sent_tokenize(normalized)       # Punkt tokeniser (punkt_tab model)
    filter: len(sent.lower().split()) >= MIN_SENTENCE_TOKENS (5)
  Returns: processed: dict[str, list[str]]
        │
        ▼ st.spinner
[embedding.encode_documents(processed)]
  For each document:
    model.encode(sentences, batch_size=64) → np.ndarray shape (N, 384)
  Returns: embeddings: dict[str, np.ndarray]
        │
        ▼ st.spinner
[similarity.compute_all_pairwise_similarities(embeddings)]
  For each (A, B) pair from itertools.combinations(doc_names, 2):
    sklearn.cosine_similarity(emb_a, emb_b) → np.ndarray shape (M, N)
  Returns: pair_matrices: dict[tuple[str,str], np.ndarray]
        │
        ▼ st.spinner
[detection.run_all_detections(processed, pair_matrices, threshold)]
  For each pair:
    detect_plagiarism(...) → PairResult
  Returns: results: list[PairResult]
        │
        ▼
[app.py] renders results per pair:
  ├── visualization.build_match_table_html(result)  → HTML string
  ├── visualization.highlight_document(...)          → HTML string
  ├── visualization.render_heatmap(...)              → io.BytesIO (PNG)
  └── explanation.generate_explanation(result)       → str (OpenRouter API call)
```

**No caching occurs on computed results.** Only the SBERT model itself is cached via `@st.cache_resource` (persists across re-runs within the same Streamlit server process).

---

## 4. Module-by-Module Reference

### 4.1 `config.py`

Central constants. No logic. All other modules import from here — edit this file to change global behaviour.

| Name | Type | Default | Effect |
|---|---|---|---|
| `OPENROUTER_API_KEY` | `str` | `os.environ.get(...)` | If empty string, `explanation.py` skips the API call entirely and returns a warning |
| `OPENROUTER_BASE_URL` | `str` | `"https://openrouter.ai/api/v1/chat/completions"` | POST target for the LLM API |
| `OPENROUTER_MODEL` | `str` | `"mistralai/mistral-7b-instruct:free"` | Model ID sent in the JSON payload; the `:free` suffix is required for free-tier accounts |
| `SBERT_MODEL` | `str` | `"all-MiniLM-L6-v2"` | Passed verbatim to `SentenceTransformer(...)` |
| `DEFAULT_THRESHOLD` | `float` | `0.75` | Sidebar slider default; range is `[0.50, 0.99]` |
| `MIN_SENTENCE_TOKENS` | `int` | `5` | Minimum word count per sentence after whitespace split; sentences below this are discarded |
| `HEATMAP_COLORMAP` | `str` | `"YlOrRd"` | Matplotlib colormap name passed to `sns.heatmap(cmap=...)` |
| `APP_TITLE` / `APP_SUBTITLE` | `str` | — | UI display strings only |

API key resolution order: environment variable `OPENROUTER_API_KEY` → empty string fallback. There is **no `.env` auto-loading** in the code (no `python-dotenv`); the `.env` file must be sourced by the shell before launching Streamlit, or the variable set via `export`.

---

### 4.2 `preprocessing.py`

**Imports:** `re`, `nltk`, `config.MIN_SENTENCE_TOKENS`

**NLTK data downloaded at module import time** (not lazily):
```python
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
```
`punkt_tab` is the new binary format required by NLTK ≥ 3.9. Both are downloaded to `~/nltk_data/` by default. If this directory is not writable (e.g., in a container), download will silently fail and `sent_tokenize` will raise a `LookupError` at runtime.

**`_normalize_whitespace(text: str) → str`**
- `re.sub(r"[ \t]+", " ", text)` — collapses all horizontal whitespace (spaces and tabs) to a single space
- `re.sub(r"\n{2,}", "\n", text)` — collapses multiple blank lines to one newline
- Does **not** lowercase; casing is preserved intentionally because SBERT is case-sensitive and benefits from proper capitalisation

**`split_sentences(text: str) → list[str]`**
1. Calls `_normalize_whitespace`
2. Calls `nltk.sent_tokenize(normalized)` — this uses the Punkt unsupervised sentence boundary detection algorithm
3. Filters: `len(sent.lower().split()) >= MIN_SENTENCE_TOKENS` — word count on the lowercased form, but the original-case string is kept in the output list

**`preprocess_documents(docs: dict[str, str]) → dict[str, list[str]]`**
- Simple dict comprehension calling `split_sentences` per document
- Empty documents or documents that produce zero sentences after filtering will be caught in `app.py` and skipped with a warning

---

### 4.3 `embedding.py`

**Imports:** `numpy`, `streamlit`, `sentence_transformers.SentenceTransformer`, `config.SBERT_MODEL`

**`get_model() → SentenceTransformer`** — decorated with `@st.cache_resource`

`@st.cache_resource` stores the return value **globally across all Streamlit sessions** in the same server process — i.e., the model is loaded once per Python interpreter lifetime, not per user. The model is not serialised to disk by this decorator; it lives in memory.

**`encode_sentences(sentences: list[str]) → np.ndarray`**

Calls:
```python
model.encode(
    sentences,
    convert_to_numpy=True,    # returns np.ndarray, not torch.Tensor
    show_progress_bar=False,
    batch_size=64,
)
```

Returns shape `(len(sentences), 384)`. If `sentences` is empty, returns `np.empty((0, 384))` immediately without touching the model.

**`encode_documents(processed_docs: dict[str, list[str]]) → dict[str, np.ndarray]`**
- Dict comprehension over `encode_sentences` per document
- All documents are encoded **sequentially**, not in parallel

---

### 4.4 `similarity.py`

**Imports:** `numpy`, `sklearn.metrics.pairwise.cosine_similarity`, `itertools.combinations`

**`compute_similarity_matrix(embeddings_a, embeddings_b) → np.ndarray`**

Wraps `sklearn.metrics.pairwise.cosine_similarity(embeddings_a, embeddings_b)`.

- Input shapes: `(M, D)` and `(N, D)` where `D = 384`
- Output shape: `(M, N)`
- Cell `[i, j]` = cosine similarity between sentence `i` of doc A and sentence `j` of doc B
- Internally sklearn normalises each row vector and computes dot products: `sim(a, b) = (a · b) / (‖a‖ · ‖b‖)`
- Values are in `[−1, 1]` theoretically, but SBERT embeddings are non-negative in practice after mean pooling + normalisation, so practical range is `[0, 1]`

**`compute_all_pairwise_similarities(embeddings: dict) → dict`**

Uses `itertools.combinations(doc_names, 2)` — generates only upper-triangle pairs (no self-comparisons, no symmetric duplicates). With N documents, produces `N*(N-1)/2` matrices.

---

### 4.5 `detection.py`

**Imports:** `numpy`, `dataclasses`

#### Data Classes

**`MatchedPair`** — a single flagged sentence pair:
| Field | Type | Description |
|---|---|---|
| `doc_a`, `doc_b` | `str` | Document display names |
| `sentence_a`, `sentence_b` | `str` | The actual sentence strings |
| `similarity` | `float` | Cosine similarity score in `[0, 1]` |
| `index_a`, `index_b` | `int` | Original sentence indices in their respective lists |

**`PairResult`** — aggregated result for one document pair:
| Field | Type | Description |
|---|---|---|
| `doc_a`, `doc_b` | `str` | Document names |
| `plagiarism_pct_a` | `float` | `(len(flagged_indices_a) / M) * 100` |
| `plagiarism_pct_b` | `float` | `(len(flagged_indices_b) / N) * 100` |
| `overall_pct` | `float` | `((|flagged_a| + |flagged_b|) / (M + N)) * 100` |
| `matched_pairs` | `list[MatchedPair]` | Sorted descending by similarity |
| `flagged_indices_a` | `set[int]` | Sentence indices flagged in doc A |
| `flagged_indices_b` | `set[int]` | Sentence indices flagged in doc B |

Note: `overall_pct` double-counts sentences that appear in both `flagged_indices_a` and `flagged_indices_b`. This is intentional — it measures the combined exposure of both documents, not unique sentences.

---

### 4.6 `explanation.py`

**Imports:** `requests`, `config.*`, `detection.PairResult`

**`_build_prompt(result: PairResult) → str`**
- Takes the top 5 matched pairs (`result.matched_pairs[:5]`)
- Constructs a structured plain-text prompt with four labelled sections: REPORT SUMMARY, TOP MATCHING SENTENCE PAIRS, YOUR TASK
- Constrains the LLM response to max 200 words in the prompt text (not enforced by `max_tokens`)

**`generate_explanation(result: PairResult) → str`**

Short-circuits immediately if `OPENROUTER_API_KEY` is falsy (empty string or None).

HTTP POST to `OPENROUTER_BASE_URL` with:
```json
{
  "model": "mistralai/mistral-7b-instruct:free",
  "messages": [{"role": "user", "content": "<prompt>"}],
  "max_tokens": 500,
  "temperature": 0.4
}
```

Required headers:
- `Authorization: Bearer <key>` — standard OpenAI-compatible auth
- `HTTP-Referer` — OpenRouter requires this to identify the calling app; currently hardcoded to `"https://plagiarism-detector.app"` (not validated by OpenRouter)
- `X-Title` — display name shown in OpenRouter dashboard

Response extraction: `data["choices"][0]["message"]["content"].strip()`

Timeout: 30 seconds. Error handling catches: `Timeout`, `HTTPError`, `ConnectionError`, `KeyError`/`IndexError` (unexpected response structure), and a generic `Exception` fallback. All errors return a human-readable warning string rather than raising.

---

### 4.7 `visualization.py`

**Imports:** `io`, `html` (stdlib, aliased as `html_lib`), `numpy`, `matplotlib` (Agg backend), `seaborn`, `config.HEATMAP_COLORMAP`, `detection.PairResult`

Matplotlib is forced to the `Agg` (non-interactive raster) backend at import time via `matplotlib.use("Agg")` before any `pyplot` import. This is required because Streamlit runs in a server context with no display.

**`render_heatmap(...) → io.BytesIO`**

1. Truncates sentence labels to 30 characters for axis ticks
2. Computes dynamic figure size: `fig_w = min(2 + N_B * 0.55, 22)`, `fig_h = min(2 + N_A * 0.45, 18)`
3. Annotations (`annot=True`, `fmt=".2f"`) are only added when both dimensions ≤ 12 to avoid label collision
4. Adds a dashed horizontal cyan line on the colorbar at `y=threshold` using `cbar.ax.axhline(...)`
5. Saves to `io.BytesIO` at 130 DPI as PNG, then returns the buffer (seeked to 0)
6. Calls `plt.close(fig)` to release memory — critical in Streamlit where figures accumulate across re-runs

`st.image(buf, use_container_width=True)` in `app.py` decodes this buffer directly.

**`highlight_document(sentences, flagged_indices, doc_name) → str`**

Produces an HTML `<div>` with one `<span>` per sentence. All user-supplied text is passed through `html_lib.escape()` to prevent XSS. Flagged sentences get an inline gradient background style; unflagged sentences get a neutral style.

**`build_match_table_html(result: PairResult) → str`**

Builds a `<table>` string with one `<tr>` per `MatchedPair`. Score colour thresholds:
- ≥ 90% → `#ff4e4e` (red)
- ≥ 80% → `#ff9a3c` (orange)
- < 80% → `#facc15` (yellow)

**Critical implementation note:** All HTML strings must begin at column 0 inside the f-string. Leading whitespace ≥ 4 spaces causes Streamlit's Markdown renderer to treat the content as a fenced code block rather than raw HTML, displaying the raw tag text instead of rendered HTML.

---

### 4.8 `utils.py`

**`read_uploaded_file(uploaded_file) → str`**
- Reads the Streamlit `UploadedFile` object (which exposes `.read()` → `bytes`)
- Attempts UTF-8 decode; falls back to latin-1 on `UnicodeDecodeError`
- No BOM stripping; could silently corrupt files with UTF-8 BOM

**`score_to_label(pct: float) → tuple[str, str]`**

Label thresholds (applied to `overall_pct`):
| `overall_pct` range | Label | Colour |
|---|---|---|
| ≥ 70 | 🔴 High Plagiarism | `#ff4e4e` |
| ≥ 40 | 🟠 Moderate Plagiarism | `#ff9a3c` |
| ≥ 15 | 🟡 Low Plagiarism | `#facc15` |
| < 15 | 🟢 Minimal / No Plagiarism | `#4ade80` |

**`results_to_csv(results: list[PairResult]) → str`**

Produces RFC 4180-compliant CSV. All four text fields (`doc_a`, `doc_b`, `sentence_a`, `sentence_b`) are double-quoted with internal double-quotes escaped as `""`. `similarity_pct` is the cosine score multiplied by 100, formatted to 2 decimal places.

---

### 4.9 `app.py`

**Entry point.** Run with `streamlit run app.py`.

**Page config:** `layout="wide"`, `initial_sidebar_state="expanded"`. Custom CSS is injected via `st.markdown(..., unsafe_allow_html=True)` — imports Google Fonts (`Syne`, `Space Mono`), sets dark theme colours, overrides Streamlit's default metric/button/tab styles.

**Input modes:**
- **Upload mode:** `st.file_uploader(accept_multiple_files=True, type=["txt"])` → each file passed to `utils.read_uploaded_file`
- **Paste mode:** `st.number_input` (2–6) + dynamic `st.text_area` per document; empty text areas are skipped

**Pipeline execution** is guarded behind `if run:` (button press). Each stage is wrapped in `st.spinner(...)`. All five stages run sequentially in `app.py`'s main thread — there is no async execution.

**Per-pair results loop:**

For each `PairResult` in `results`:
1. `st.expander` with overall percentage and label in the title
2. 4 metric columns: overall score, per-doc flagged %, matched pair count
3. 4-tab layout: Matched Pairs → Highlighted Text → Heatmap → AI Explanation

The AI explanation (`generate_explanation`) is called **inside the tab render block**, meaning it only executes when the tab is rendered (because Streamlit renders all tabs on load, this fires immediately for all pairs on page load after analysis).

**CSV download:** `st.download_button` with `mime="text/csv"` and `results_to_csv(results)` as data payload.

---

## 5. Model: `all-MiniLM-L6-v2`

### Architecture

A distilled 6-layer BERT-like transformer encoder (MiniLM variant), fine-tuned by the sentence-transformers library on sentence-pair datasets for semantic similarity.

| Property | Value |
|---|---|
| Architecture | Transformer encoder (BERT-style) |
| Layers | 6 |
| Attention heads | 12 |
| Hidden size | 384 |
| Max input tokens | 256 (longer sequences are truncated) |
| Output | Single 384-dim vector per sentence |
| Pooling | Mean pooling over token embeddings of the last hidden layer |
| Normalisation | L2-normalised output (unit vectors) |
| Parameters | ~22.7 million |
| Training data | 1 billion sentence pairs (NLI, STS, MS-MARCO, etc.) |
| Licence | Apache 2.0 |

### Weights & Storage

On first run, `SentenceTransformer("all-MiniLM-L6-v2")` downloads the model from HuggingFace Hub to:
```
~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/
```
This directory contains `pytorch_model.bin` (or `model.safetensors`), `tokenizer.json`, `config.json`, and sentence-transformers `modules.json`.

**Weights are never updated at runtime.** This system performs inference only. `model.encode()` runs in `torch.no_grad()` context internally (managed by the sentence-transformers library).

### Inputs & Outputs

| | Detail |
|---|---|
| Input | Raw string; tokenised internally by the model's `AutoTokenizer` (WordPiece) |
| Tokenisation | WordPiece, max 256 tokens; `[CLS]` + tokens + `[SEP]`; padded/truncated per batch |
| Output | `np.ndarray` of shape `(N, 384)`, dtype `float32` |
| Batch size | 64 (set in `embedding.py:encode_sentences`) |

Sentences longer than 256 WordPiece tokens are **silently truncated**. No warning is emitted. For academic text, typical sentences are well under this limit, but very long compound sentences may be affected.

### Device

The model runs on whichever device PyTorch defaults to. The code does not explicitly set `device=` in `SentenceTransformer(...)` or `model.encode(...)`. If CUDA is available, `torch` will use it automatically; otherwise CPU inference is used.

---

## 6. Data Structures

All data flows through standard Python types. The only custom types are two `dataclass` objects, both defined in `detection.py`.

```
Raw Input:       dict[str, str]
                 {doc_name: raw_text}

After Preprocess: dict[str, list[str]]
                  {doc_name: [sentence_0, sentence_1, ...]}

After Embedding: dict[str, np.ndarray]
                 {doc_name: array(shape=(N_sentences, 384), dtype=float32)}

After Similarity: dict[tuple[str,str], np.ndarray]
                  {(doc_a, doc_b): array(shape=(M, N), dtype=float64)}
                  ^ float64 from sklearn cosine_similarity

After Detection: list[PairResult]
                 One PairResult per document pair

PairResult contains:
  matched_pairs:     list[MatchedPair]   (sorted by similarity descending)
  flagged_indices_a: set[int]
  flagged_indices_b: set[int]
  plagiarism_pct_a:  float  (rounded to 2 dp)
  plagiarism_pct_b:  float
  overall_pct:       float
```

---

## 7. Detection Algorithm

Defined in `detection.detect_plagiarism()`. The algorithm is a **bidirectional greedy best-match scan** — not maximum bipartite matching, not all-pairs thresholding.

### Algorithm (pseudocode)

```
seen_pairs = set()
flagged_a  = set()
flagged_b  = set()
matches    = []

# Pass 1: For each sentence in A, find its best match in B
for i in range(M):
    j_best = argmax(sim_matrix[i, :])           # index of best B sentence
    score  = sim_matrix[i, j_best]
    if score >= threshold:
        flagged_a.add(i)
        flagged_b.add(j_best)
        seen_pairs.add((i, j_best))
        matches.append(MatchedPair(i, j_best, score))

# Pass 2: For each sentence in B, find its best match in A
for j in range(N):
    i_best = argmax(sim_matrix[:, j])           # index of best A sentence
    score  = sim_matrix[i_best, j]
    if score >= threshold:
        flagged_a.add(i_best)
        flagged_b.add(j)
        if (i_best, j) not in seen_pairs:        # dedup
            seen_pairs.add((i_best, j))
            matches.append(MatchedPair(i_best, j, score))

matches.sort(key=lambda p: p.similarity, reverse=True)
```

### Properties & Edge Cases

- **Not symmetric by default (without Pass 2):** If sentence A_i's best match is B_j but B_j's best match is NOT A_i, Pass 1 alone would miss flagging B_j from B's perspective. Pass 2 corrects this.
- **One-to-many:** A single sentence in A can be matched to multiple sentences in B if multiple B sentences each chose A_i as their best match in Pass 2.
- **Greedy, not optimal:** This is NOT the optimal assignment (which would require the Hungarian algorithm). The algorithm can double-flag sentence pairs.
- **`seen_pairs` deduplication:** Only prevents duplicate `MatchedPair` objects. `flagged_indices_a/b` are sets so they remain deduplicated regardless.
- **Percentage computation:**
  - `pct_a = |flagged_a| / M * 100`
  - `pct_b = |flagged_b| / N * 100`
  - `overall = (|flagged_a| + |flagged_b|) / (M + N) * 100`
  - This means a sentence that's flagged in both `flagged_a` and `flagged_b` contributes to both numerator terms.

---

## 8. OpenRouter Integration

OpenRouter acts as a proxy over multiple LLM providers with a unified OpenAI-compatible API.

**Endpoint:** `POST https://openrouter.ai/api/v1/chat/completions`

**Model selection:** `mistralai/mistral-7b-instruct:free` — the `:free` suffix routes to the free-tier inference backend. Removing it may select paid inference.

**Prompt structure:** The prompt is structured text (not a system/user role split — only a user message). It contains:
- Aggregate statistics from `PairResult` (percentages, pair count)
- Top 5 matched pairs with similarity scores and sentence text
- Explicit task instructions and word limit

**Inference parameters:**
- `temperature: 0.4` — relatively deterministic; low variance in outputs
- `max_tokens: 500` — hard cap on response length

**The `HTTP-Referer` header** is required by OpenRouter's API but is not validated. The hardcoded value `"https://plagiarism-detector.app"` is a placeholder; any non-empty URL is accepted.

**No retry logic.** A single POST is made. On `requests.exceptions.Timeout` (30s), the error string is returned to the UI.

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| No model training | All plagiarism detection is geometric (cosine similarity). Training would require labelled sentence pairs and is outside scope. |
| Casing preserved throughout | SBERT's tokeniser is case-aware. Lowercasing before embedding would degrade similarity quality on proper nouns and title-case sentences. |
| `@st.cache_resource` for SBERT | Prevents reloading ~85MB weights on every Streamlit re-run. Lives in the Python process; not persisted to disk. |
| `itertools.combinations` for pairs | Ensures exactly `N*(N-1)/2` unique pairs. Prevents symmetric duplicates `(A,B)` and `(B,A)`. |
| Bidirectional scan | Catches plagiarism from both directions. A sentence in B can be flagged even if no sentence in A chose it as its best match. |
| `html.escape()` on all user text | Prevents XSS through crafted document content rendered via `unsafe_allow_html=True`. |
| Matplotlib `Agg` backend | Streamlit runs server-side without a display. `Agg` produces raster images without any GUI toolkit dependency. |
| `BytesIO` for heatmap | Avoids writing to disk. The PNG bytes are passed directly to `st.image()`. |
| `plt.close(fig)` after each heatmap | Prevents matplotlib figure accumulation in memory across Streamlit re-runs. |

---

## 10. Environment & Dependency Assumptions

### Python
- Python **3.11** (the `nlpenv` conda environment)
- Type annotations use `list[str]` / `dict[str, ...]` syntax (PEP 585, Python 3.9+)

### NLTK Data
Downloaded to `~/nltk_data/` at first import of `preprocessing.py`. Requires internet and write access. The `punkt_tab` resource is mandatory for NLTK ≥ 3.9; `punkt` is kept for backward compatibility.

### HuggingFace Cache
Model weights downloaded to `~/.cache/huggingface/hub/` on first call to `SentenceTransformer("all-MiniLM-L6-v2")`. Approximately 85MB. Subsequent imports use the local cache.

### GPU
Optional. `sentence-transformers` will use CUDA if `torch.cuda.is_available()`. No explicit device management in this codebase. CPU inference is fully supported and typical for this model size.

### `torchvision`
Required by `transformers` (a transitive dependency of `sentence-transformers`) for several computer-vision model image processors. Not used directly by this project. Without it, Streamlit's file watcher raises `ModuleNotFoundError: No module named 'torchvision'` on startup (harmless to functionality, but noisy).

### Environment Variable
`OPENROUTER_API_KEY` must be present in the **shell environment** before launching Streamlit. `python-dotenv` is **not installed**; `.env` files are not auto-loaded. The variable must be exported manually or via a shell init file.

### Port
Streamlit defaults to `localhost:8501`. No explicit port configuration in this codebase.

---

## 11. Known Limitations

| Limitation | Detail |
|---|---|
| **Max input tokens: 256** | Sentences exceeding 256 WordPiece tokens are truncated before embedding. No warning is emitted. Typically affects sentences > ~200 words. |
| **No `.env` auto-loading** | `OPENROUTER_API_KEY` must be exported to the shell environment manually. The `.env` file is advisory only. |
| **Sequential encoding** | All documents are encoded one after another. No parallelism. For large documents (>500 sentences), this creates visible latency. |
| **All-pairs computation** | With N documents, `N*(N-1)/2` full similarity matrices are computed. Memory scales as `O(N² * M * M * 8 bytes)` in the worst case. Not suitable for N > 10 or very long documents. |
| **No deduplication of identical sentences** | If the same sentence appears in both documents, it is embedded and compared independently, potentially producing multiple equivalent `MatchedPair` entries. |
| **Greedy matching** | The bidirectional best-match algorithm is not globally optimal. It can produce lower overall scores than Hungarian matching when documents share many similar sentences. |
| **`overall_pct` inflation** | A sentence flagged in Pass 1 (from A's POV) and again in Pass 2 (from B's POV) contributes to `pct_a` and `pct_b` separately. The overall score double-counts these sentence positions. |
| **LLM explanation quality** | The explanation is generated from top-5 pairs only. For documents with many matched pairs, the AI report may miss important patterns. No streaming — the full response is awaited before display. |
| **Streamlit re-run semantics** | Every time the user changes the threshold slider, the full pipeline from preprocessing onwards re-executes. Embeddings and similarity matrices are recomputed unnecessarily (only detection needs to rerun on threshold change). |
