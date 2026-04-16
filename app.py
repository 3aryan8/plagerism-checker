# ============================================================
# app.py — Streamlit UI for Semantic Plagiarism Detection
# ============================================================

import streamlit as st
from config import APP_TITLE, APP_SUBTITLE, DEFAULT_THRESHOLD

from preprocessing  import preprocess_documents
from embedding      import encode_documents
from similarity     import compute_all_pairwise_similarities
from detection      import run_all_detections
from explanation    import generate_explanation
from visualization  import render_heatmap, highlight_document, build_match_table_html
from utils          import read_uploaded_file, score_to_label, results_to_csv


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"]  { font-family: 'Syne', sans-serif; }
  code, pre                   { font-family: 'Space Mono', monospace !important; }

  /* App background */
  .stApp { background: #0b0b18; color: #d0d0e8; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #10101f !important; }

  /* Headers */
  h1 { font-family: 'Syne', sans-serif; font-weight: 800;
       background: linear-gradient(135deg, #a78bfa, #38bdf8);
       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
       font-size: 2.2rem !important; margin-bottom: 0 !important; }
  h2 { color: #a78bfa !important; font-weight: 700; }
  h3 { color: #7dd3fc !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #13132a; border: 1px solid #2a2a4a;
    border-radius: 10px; padding: 12px;
  }
  [data-testid="stMetricValue"]  { color: #a78bfa !important; font-size: 2rem !important; }
  [data-testid="stMetricLabel"]  { color: #888 !important; }
  [data-testid="stMetricDelta"]  { color: #4ade80 !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white; border: none; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 700;
    padding: 0.6rem 1.8rem; font-size: 1rem;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity: .85; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: #888 !important; font-weight: 600; }
  .stTabs [aria-selected="true"] { color: #a78bfa !important; border-bottom-color: #a78bfa !important; }

  /* Divider */
  hr { border-color: #2a2a4a; }

  /* Expander */
  .streamlit-expanderHeader { color: #a78bfa !important; font-weight: 700; }

  /* Sliders */
  [data-baseweb="slider"] [role="slider"] { background: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    threshold = st.slider(
        "🎯 Similarity Threshold",
        min_value=0.50,
        max_value=0.99,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Sentences with cosine similarity ≥ this value are flagged as plagiarised.",
    )

    st.markdown(f"""
    <div style='background:#1a1a2e;padding:10px;border-radius:8px;font-size:12px;color:#888'>
      <b style='color:#a78bfa'>How it works:</b><br>
      1. Sentences are embedded using <b>SBERT (MiniLM-L6-v2)</b><br>
      2. Cosine similarity is computed between every sentence pair<br>
      3. Pairs above the threshold are flagged<br>
      4. An AI summary is generated via OpenRouter
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#555'>Built with Streamlit · SBERT · OpenRouter</div>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#666;font-size:15px;margin-top:4px'>{APP_SUBTITLE}</p>", unsafe_allow_html=True)
st.markdown("---")


# ── Input Section ─────────────────────────────────────────────────────────────

st.markdown("## 📄 Input Documents")

input_mode = st.radio(
    "Input mode",
    ["📁 Upload text files", "✏️ Paste text"],
    horizontal=True,
    label_visibility="collapsed",
)

docs_raw: dict[str, str] = {}

if input_mode == "📁 Upload text files":
    uploaded = st.file_uploader(
        "Upload two or more `.txt` files",
        type=["txt"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            docs_raw[f.name] = read_uploaded_file(f)
        st.success(f"✅ {len(uploaded)} file(s) loaded.")
else:
    # Paste mode — dynamic text areas
    n_docs = st.number_input("How many documents?", min_value=2, max_value=6, value=2)
    cols = st.columns(2)
    for i in range(int(n_docs)):
        col = cols[i % 2]
        with col:
            name = st.text_input(f"Document {i+1} name", value=f"Document {i+1}", key=f"name_{i}")
            text = st.text_area(f"Paste text for {name}", height=200, key=f"text_{i}")
            if text.strip():
                docs_raw[name] = text

st.markdown("---")

# ── Run Button ────────────────────────────────────────────────────────────────

run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("🔍 Check Plagiarism", use_container_width=True)

# ── Main pipeline ─────────────────────────────────────────────────────────────

if run:
    if len(docs_raw) < 2:
        st.warning("⚠️  Please provide at least **2** documents to compare.")
        st.stop()

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    with st.spinner("📝 Preprocessing documents…"):
        processed = preprocess_documents(docs_raw)

    # Warn about any empty documents
    for name, sents in processed.items():
        if not sents:
            st.warning(f"⚠️  **{name}** produced no usable sentences after preprocessing. It will be skipped.")
    processed = {k: v for k, v in processed.items() if v}

    if len(processed) < 2:
        st.error("Not enough non-empty documents to compare.")
        st.stop()

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    with st.spinner("🧠 Generating sentence embeddings (SBERT)…"):
        embeddings = encode_documents(processed)

    # ── Step 3: Similarity ────────────────────────────────────────────────────
    with st.spinner("📐 Computing similarity matrices…"):
        pair_matrices = compute_all_pairwise_similarities(embeddings)

    # ── Step 4: Detect ────────────────────────────────────────────────────────
    with st.spinner("🔎 Running plagiarism detection…"):
        results = run_all_detections(processed, pair_matrices, threshold)

    # ── Step 5: Display results per pair ─────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Results")

    for result in results:
        pair_title = f"{result.doc_a}  ↔  {result.doc_b}"
        label, color = score_to_label(result.overall_pct)

        with st.expander(f"**{pair_title}** — {label}  ({result.overall_pct:.1f}%)", expanded=True):

            # ── Score metrics ─────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Score",        f"{result.overall_pct:.1f}%")
            m2.metric(f"{result.doc_a} flagged", f"{result.plagiarism_pct_a:.1f}%")
            m3.metric(f"{result.doc_b} flagged", f"{result.plagiarism_pct_b:.1f}%")
            m4.metric("Matched pairs",        str(len(result.matched_pairs)))

            st.markdown(
                f"<div style='background:#1a1a2e;padding:10px 16px;border-radius:8px;"
                f"border-left:4px solid {color};margin:10px 0'>"
                f"<b style='color:{color}'>{label}</b></div>",
                unsafe_allow_html=True,
            )

            tabs = st.tabs(["🔗 Matched Pairs", "🎨 Highlighted Text", "🌡️ Heatmap", "🤖 AI Explanation"])

            # ── Tab 1: Matched pairs table ────────────────────────────────────
            with tabs[0]:
                st.markdown("### Top Matched Sentence Pairs")
                st.markdown(build_match_table_html(result), unsafe_allow_html=True)

            # ── Tab 2: Highlighted documents ─────────────────────────────────
            with tabs[1]:
                st.markdown("### Highlighted Documents")
                st.markdown(
                    "<p style='color:#666;font-size:12px'>"
                    "🔴 Highlighted sentences exceed the similarity threshold.</p>",
                    unsafe_allow_html=True,
                )
                hcol_a, hcol_b = st.columns(2)
                with hcol_a:
                    st.markdown(
                        highlight_document(
                            processed[result.doc_a],
                            result.flagged_indices_a,
                            result.doc_a,
                        ),
                        unsafe_allow_html=True,
                    )
                with hcol_b:
                    st.markdown(
                        highlight_document(
                            processed[result.doc_b],
                            result.flagged_indices_b,
                            result.doc_b,
                        ),
                        unsafe_allow_html=True,
                    )

            # ── Tab 3: Heatmap ────────────────────────────────────────────────
            with tabs[2]:
                st.markdown("### Similarity Heatmap")
                st.markdown(
                    "<p style='color:#666;font-size:12px'>"
                    "Each cell shows cosine similarity between two sentences. "
                    "Warmer colours = higher similarity.</p>",
                    unsafe_allow_html=True,
                )
                sim_matrix = pair_matrices[(result.doc_a, result.doc_b)]
                heatmap_buf = render_heatmap(
                    sim_matrix,
                    result.doc_a,
                    result.doc_b,
                    processed[result.doc_a],
                    processed[result.doc_b],
                    threshold,
                )
                st.image(heatmap_buf, use_container_width=True)

            # ── Tab 4: AI explanation ─────────────────────────────────────────
            with tabs[3]:
                st.markdown("### 🤖 AI-Generated Explanation")
                with st.spinner("Calling OpenRouter API…"):
                    explanation = generate_explanation(result)
                st.markdown(
                    f"<div style='background:#12122a;padding:16px;border-radius:10px;"
                    f"border:1px solid #2a2a4a;color:#d0d0f0;line-height:1.7;font-size:14px'>"
                    f"{explanation}</div>",
                    unsafe_allow_html=True,
                )

    # ── Download results as CSV ──────────────────────────────────────────────
    st.markdown("---")
    csv_data = results_to_csv(results)
    st.download_button(
        label="📥 Download Matched Pairs (CSV)",
        data=csv_data,
        file_name="plagiarism_results.csv",
        mime="text/csv",
    )
    st.markdown(
        "<div style='text-align:center;color:#444;font-size:12px;margin-top:12px'>"
        "Semantic Plagiarism Detection · Powered by SBERT + OpenRouter</div>",
        unsafe_allow_html=True,
    )
