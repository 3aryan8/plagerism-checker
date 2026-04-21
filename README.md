<div align="center">

# 🔍 Semantic Plagiarism Detection System

**Detect meaning-level plagiarism between documents — not just copied words, but copied ideas.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![SBERT](https://img.shields.io/badge/SBERT-MiniLM--L6--v2-orange)](https://www.sbert.net)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Mistral%207B-8B5CF6)](https://openrouter.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 🧠 What Is This?

Traditional plagiarism detectors catch **copy-pasted text**. This system goes further — it detects **semantic plagiarism**: text that has been paraphrased, reworded, or restructured but conveys the same meaning.

It works by converting every sentence in your documents into high-dimensional **vector embeddings** using a state-of-the-art transformer model (SBERT), then measuring the **cosine similarity** between each sentence pair. Flagged pairs are surfaced in an interactive UI with heatmaps, highlighted text, and an AI-generated explanation report.

> **Example:** `"The climate is rapidly warming"` and `"Global temperatures are rising fast"` are **not** textually identical, but this system will flag them — because they mean the same thing.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Semantic Matching** | Uses `all-MiniLM-L6-v2` SBERT embeddings, not naive string comparison |
| 🌡️ **Similarity Heatmap** | Full cosine-similarity matrix visualised as an annotated heatmap |
| 🎨 **Highlighted Text** | Documents shown side-by-side with plagiarised sentences colour-coded |
| 🔗 **Matched Pairs Table** | Ranked list of flagged sentence pairs with similarity scores |
| 🤖 **AI Explanation** | OpenRouter (Mistral 7B) generates a natural-language plagiarism report |
| 📥 **CSV Export** | Download all matched sentence pairs as a CSV file |
| ⚙️ **Adjustable Threshold** | Sidebar slider to tune the similarity cutoff (0.50 – 0.99) |
| 📄 **Multi-Document** | Compare up to 6 documents simultaneously (all unique pairs checked) |
| 🔢 **Dual Input Modes** | Upload `.txt` files **or** paste text directly in the browser |

---

## 🏗️ Architecture

The pipeline is fully modular. Each stage is isolated in its own Python module:

```
 ┌──────────────────────┐
 │   Input Documents    │  Upload .txt files or paste text (up to 6 docs)
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │   preprocessing.py   │  Clean text + NLTK sentence tokenisation
 │                      │  → Removes noise; ignores sentences < 5 words
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │    embedding.py      │  SBERT all-MiniLM-L6-v2 (~80 MB)
 │                      │  → Each sentence → 384-dim dense vector
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │    similarity.py     │  Scikit-learn cosine similarity
 │                      │  → N×M similarity matrix per document pair
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │    detection.py      │  Bidirectional best-match scan
 │                      │  → Flags pairs ≥ threshold; computes % flagged
 └──────┬───────┬───────┘
        │       │
        ▼       ▼
 ┌──────────┐  ┌──────────────────────┐
 │ visuali- │  │    explanation.py    │
 │ zation   │  │  OpenRouter API      │
 │ .py      │  │  → Mistral 7B report │
 │ Heatmap  │  └──────────────────────┘
 │ + HTML   │
 └──────────┘
        │
        ▼
 ┌──────────────────────┐
 │       app.py         │  Streamlit UI — ties everything together
 └──────────────────────┘
```

### Detection Strategy (Bidirectional Scan)

The detector runs two passes to avoid direction-dependent misses:

1. **Pass 1 (A → B):** For every sentence in Document A, find its best-matching sentence in Document B. Flag pairs that exceed the threshold.
2. **Pass 2 (B → A):** Repeat from Document B's perspective, adding any new pairs not already found in Pass 1.

Duplicate pairs are deduplicated before reporting. Results are sorted by descending similarity score.

---

## 📁 Project Structure

```
plagiarism_detector/
│
├── app.py              ← Streamlit UI (entry point)
├── config.py           ← API keys, model names & constants
├── preprocessing.py    ← Text cleaning + NLTK sentence tokenisation
├── embedding.py        ← SBERT model loading & sentence encoding
├── similarity.py       ← Pairwise cosine-similarity matrix computation
├── detection.py        ← Bidirectional plagiarism detection logic
├── explanation.py      ← OpenRouter API integration (Mistral 7B)
├── visualization.py    ← Seaborn heatmap + HTML sentence highlighting
├── utils.py            ← File I/O, CSV export, score labelling helpers
│
├── requirements.txt    ← Python dependencies
├── .env                ← (not committed) store OPENROUTER_API_KEY here
│
├── sample_doc_a.txt    ← Sample climate-change document for testing
└── sample_doc_b.txt    ← Paraphrased matching document for testing
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/plagiarism-detector.git
cd plagiarism-detector
```

### 2. Create & activate a Conda environment

```bash
conda create -n nlpenv python=3.11
conda activate nlpenv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, the SBERT model `all-MiniLM-L6-v2` (~80 MB) will be automatically downloaded from HuggingFace. Subsequent runs use the local cache.

Also download the required NLTK tokenizer data:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### 4. Configure your OpenRouter API key

Create a `.env` file in the project root (it is already git-ignored):

```bash
echo 'OPENROUTER_API_KEY="sk-or-v1-..."' > .env
```

Or set it as an environment variable before launching:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Get a free key at → [openrouter.ai](https://openrouter.ai)

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at **`http://localhost:8501`**.

---

## 🎮 How to Use

1. **Choose an input mode** in the main panel:
   - **📁 Upload text files** — drag & drop one or more `.txt` files
   - **✏️ Paste text** — type or paste directly into text areas (2–6 documents)

2. **Adjust the similarity threshold** in the sidebar slider (default: `0.75`).
   - Lower → catches more potential matches (higher false-positive rate)
   - Higher → only flags very close matches (more conservative)

3. Click **🔍 Check Plagiarism**.

4. For each document pair, results appear in four tabs:

   | Tab | What you see |
   |---|---|
   | 🔗 **Matched Pairs** | Ranked table of flagged sentence pairs with similarity scores |
   | 🎨 **Highlighted Text** | Both documents side-by-side; plagiarised sentences highlighted in red |
   | 🌡️ **Heatmap** | Full sentence × sentence cosine similarity matrix, colour-coded |
   | 🤖 **AI Explanation** | Mistral 7B natural-language report describing the plagiarism found |

5. Use **📥 Download Matched Pairs (CSV)** to export results for further analysis.

---

## 🔧 Configuration Reference (`config.py`)

All tuneable parameters live in one place:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | `""` (reads from env) | Your OpenRouter API key |
| `OPENROUTER_BASE_URL` | OpenRouter endpoint | API base URL |
| `OPENROUTER_MODEL` | `mistralai/mistral-7b-instruct:free` | LLM used for AI explanations |
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformer model name |
| `DEFAULT_THRESHOLD` | `0.75` | Initial cosine similarity cutoff |
| `MIN_SENTENCE_TOKENS` | `5` | Minimum word count per sentence (shorter ones are skipped) |
| `HEATMAP_COLORMAP` | `YlOrRd` | Matplotlib colormap (yellow → orange → red) |
| `APP_TITLE` | `"🔍 Semantic Plagiarism Detection System"` | UI title text |
| `APP_SUBTITLE` | `"Detect meaning-level similarity…"` | UI subtitle text |

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32.0 | Interactive web UI |
| `sentence-transformers` | ≥ 2.7.0 | SBERT sentence embeddings |
| `scikit-learn` | ≥ 1.4.0 | Cosine similarity computation |
| `nltk` | ≥ 3.8.1 | Sentence tokenisation (`punkt_tab`) |
| `matplotlib` | ≥ 3.8.0 | Heatmap rendering |
| `seaborn` | ≥ 0.13.2 | Heatmap styling |
| `numpy` | ≥ 1.26.0 | Matrix operations |
| `torch` | ≥ 2.2.0 | SBERT backend (PyTorch) |
| `requests` | ≥ 2.31.0 | OpenRouter API HTTP calls |

---

## 🧪 Testing with Sample Documents

Two sample documents are included for immediate testing:

- **`sample_doc_a.txt`** — An essay on climate change
- **`sample_doc_b.txt`** — A paraphrased / semantically equivalent version

Upload both files (or select them in the UI) and set the threshold to `0.70`–`0.80` to see the system in action. You should observe:
- Several sentence pairs flagged with high similarity scores
- Both documents highlighted with the relevant sentences
- A clear heatmap showing hotspots of overlap
- An AI explanation summarising the extent and nature of the plagiarism

---

## ⚠️ Troubleshooting

| Issue | Fix |
|---|---|
| **Model download fails** | Ensure you have internet access on first run; HuggingFace CDN is required |
| **`nltk punkt` / `punkt_tab` error** | Run `python -c "import nltk; nltk.download('punkt_tab')"` |
| **OpenRouter 401 Unauthorized** | Verify `OPENROUTER_API_KEY` is set correctly in `.env` or your environment |
| **OpenRouter 429 / rate limit** | Free-tier rate limits apply; wait a moment and retry |
| **App is slow on first load** | SBERT model is loading into memory; all subsequent runs use the in-memory cache |
| **Empty results / no flags** | Try lowering the threshold slider; documents may be less similar than expected |
| **`torch` or CUDA errors** | SBERT runs fine on CPU; ensure `torch` is installed (GPU is optional) |

---

## 🗺️ Roadmap

- [ ] Support for PDF and DOCX input formats
- [ ] Configurable LLM model selection from the sidebar
- [ ] Per-sentence confidence score colour gradient in highlights
- [ ] Batch folder comparison mode
- [ ] Persistent result history / session storage

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss significant changes.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built with ❤️ using <strong>Streamlit</strong> · <strong>SBERT</strong> · <strong>OpenRouter</strong>
</div>
