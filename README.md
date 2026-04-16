# 🔍 Semantic Plagiarism Detection System

Detect **meaning-level** (not keyword) plagiarism between multiple documents using sentence embeddings, cosine similarity, and AI-generated explanations.

---

## 📁 Project Structure

```
plagiarism_detector/
│
├── app.py              ← Streamlit UI (entry point)
├── config.py           ← API keys & constants
├── preprocessing.py    ← Text cleaning + sentence tokenisation
├── embedding.py        ← SBERT model & sentence encoding
├── similarity.py       ← Cosine-similarity matrix computation
├── detection.py        ← Plagiarism detection logic
├── explanation.py      ← OpenRouter API integration
├── visualization.py    ← Heatmap + HTML sentence highlighting
├── utils.py            ← Helper functions
├── requirements.txt    ← Python dependencies
├── sample_doc_a.txt    ← Sample document for testing
└── sample_doc_b.txt    ← Sample document for testing
```

---

## ⚡ Quick Setup

### 1. Create & activate a virtual environment (recommended)

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~80 MB) from HuggingFace.

### 3. Configure your OpenRouter API key

Open `config.py` and replace the placeholder:

```python
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
```

Get a free key at → https://openrouter.ai

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🎮 How to Use

1. **Choose input mode** — upload `.txt` files or paste text directly.
2. **Add at least 2 documents.**
3. **Adjust the similarity threshold** using the sidebar slider (default: 0.75).
4. Click **"🔍 Check Plagiarism"**.
5. View results in four tabs:
   - **Matched Pairs** — table of flagged sentence pairs with scores
   - **Highlighted Text** — original documents with plagiarised sentences highlighted
   - **Heatmap** — visual similarity matrix between all sentence pairs
   - **AI Explanation** — OpenRouter-generated plagiarism report

---

## 🏗️ Architecture

```
Input Text
    │
    ▼
preprocessing.py  ── clean + sentence-tokenise
    │
    ▼
embedding.py  ── SBERT all-MiniLM-L6-v2 → vectors
    │
    ▼
similarity.py  ── cosine similarity matrix (sklearn)
    │
    ▼
detection.py  ── threshold comparison → flagged pairs + %
    │
    ├──▶ visualization.py  ── heatmap (seaborn) + HTML highlights
    │
    └──▶ explanation.py  ── OpenRouter API → AI summary
```

---

## 🔧 Configuration Options (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | `"YOUR_KEY"` | Your OpenRouter API key |
| `OPENROUTER_MODEL` | `mistralai/mistral-7b-instruct` | LLM model for explanations |
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `DEFAULT_THRESHOLD` | `0.75` | Default similarity cut-off |
| `MIN_SENTENCE_TOKENS` | `5` | Min words per sentence |
| `HEATMAP_COLORMAP` | `YlOrRd` | Matplotlib colormap for heatmap |

---

## 📦 Key Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Web UI |
| `sentence-transformers` | SBERT sentence embeddings |
| `scikit-learn` | Cosine similarity computation |
| `nltk` | Sentence tokenisation |
| `matplotlib` / `seaborn` | Heatmap visualisation |
| `requests` | OpenRouter API calls |

---

## 🧪 Testing with Sample Files

Two sample documents are included (`sample_doc_a.txt`, `sample_doc_b.txt`). They cover the same climate-change topic with semantically similar (but paraphrased) content — perfect for testing at a threshold of ~0.70–0.80.

---

## ⚠️ Troubleshooting

| Issue | Fix |
|---|---|
| Model download fails | Ensure internet access; HuggingFace CDN required on first run |
| `nltk` punkt error | Run `python -c "import nltk; nltk.download('punkt_tab')"` manually |
| OpenRouter returns 401 | Check your API key in `config.py` |
| App is slow first run | SBERT model is loading; subsequent runs use the cache |
