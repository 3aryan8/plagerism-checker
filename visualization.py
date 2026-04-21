# ============================================================
# visualization.py — Heatmap + HTML sentence highlighting
# ============================================================

import io
import html as html_lib          # alias avoids name-shadowing by local variables
import numpy as np
import matplotlib
matplotlib.use("Agg")             # headless backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
from config import HEATMAP_COLORMAP
from detection import PairResult


# ── Heatmap ──────────────────────────────────────────────────────────────────

def render_heatmap(
    sim_matrix: np.ndarray,
    doc_a_name: str,
    doc_b_name: str,
    sentences_a: list[str],
    sentences_b: list[str],
    threshold: float,
) -> io.BytesIO:
    """
    Render a seaborn heatmap of the similarity matrix and return it
    as a PNG image stored in a BytesIO buffer.

    Axis labels are truncated sentence snippets for readability.
    A dashed line marks the threshold value in the colorbar.
    """
    MAX_LABEL_LEN = 30   # characters per axis tick label

    def _label(sent: str) -> str:
        return (sent[:MAX_LABEL_LEN] + "…") if len(sent) > MAX_LABEL_LEN else sent

    x_labels = [_label(s) for s in sentences_b]
    y_labels = [_label(s) for s in sentences_a]

    # Dynamic figure size — scales with document length, capped for readability
    fig_w = min(2 + len(sentences_b) * 0.55, 22)
    fig_h = min(2 + len(sentences_a) * 0.45, 18)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Evaluate once — reused for both annot= and fmt= to stay consistent
    do_annot = sim_matrix.shape[0] <= 12 and sim_matrix.shape[1] <= 12

    sns.heatmap(
        sim_matrix,
        ax=ax,
        cmap=HEATMAP_COLORMAP,
        vmin=0.0,
        vmax=1.0,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.3,
        linecolor="#1e1e2e",
        annot=do_annot,
        fmt=".2f" if do_annot else "",
        annot_kws={"size": 7, "color": "white"},
        cbar_kws={"shrink": 0.8, "label": "Cosine Similarity"},
    )

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    # Threshold line on colorbar
    cbar.ax.axhline(y=threshold, color="cyan", linewidth=1.5, linestyle="--")
    cbar.ax.text(
        1.6, threshold, f" ← threshold ({threshold:.2f})",
        va="center", ha="left", color="cyan", fontsize=7,
        transform=cbar.ax.get_yaxis_transform(),
    )

    ax.set_title(
        f"Sentence-Level Similarity  |  {doc_a_name}  vs  {doc_b_name}",
        color="white", fontsize=11, pad=12,
    )
    ax.set_xlabel(doc_b_name, color="#aaaacc", fontsize=9)
    ax.set_ylabel(doc_a_name, color="#aaaacc", fontsize=9)
    ax.tick_params(colors="#aaaacc", labelsize=7)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Highlighted HTML ─────────────────────────────────────────────────────────

_HIGHLIGHT_STYLE = (
    "background: linear-gradient(120deg, #ff4e4e55, #ff9a3c55);"
    "border-left: 3px solid #ff4e4e;"
    "padding: 3px 8px;"
    "border-radius: 4px;"
    "color: #ffe8e8;"
)

_NORMAL_STYLE = (
    "padding: 3px 8px;"
    "color: #d0d0e8;"
)


def highlight_document(
    sentences: list[str],
    flagged_indices: set[int],
    doc_name: str,
) -> str:
    """
    Return an HTML string of the document's sentences where flagged
    sentences are highlighted in red-orange.
    Uses html_lib.escape() to prevent XSS from user-supplied text.
    """
    lines = [
        f"<h4 style='color:#a78bfa;margin-bottom:6px'>{html_lib.escape(doc_name)}</h4>",
        "<div style='font-size:13px;line-height:1.8'>",
    ]
    for i, sent in enumerate(sentences):
        style = _HIGHLIGHT_STYLE if i in flagged_indices else _NORMAL_STYLE
        lines.append(f"<span style='{style}'>{html_lib.escape(sent)}</span> ")
    lines.append("</div>")
    return "\n".join(lines)


def build_match_table_html(result: PairResult) -> str:
    """
    Build an HTML table of the top matched sentence pairs.
    Variable renamed from `html` to `table_html` to avoid shadowing
    the stdlib `html` module imported at the top of this file.
    """
    if not result.matched_pairs:
        return "<p style='color:#888'>No plagiarised sentence pairs found above the threshold.</p>"

    rows = []
    for i, pair in enumerate(result.matched_pairs, 1):
        pct = pair.similarity * 100
        # colour the score cell: yellow → orange → red
        if pct >= 90:
            score_color = "#ff4e4e"
        elif pct >= 80:
            score_color = "#ff9a3c"
        else:
            score_color = "#facc15"

        rows.append(f"""<tr style='border-bottom:1px solid #2a2a3e'>
  <td style='padding:8px;color:#a78bfa;text-align:center'>{i}</td>
  <td style='padding:8px;color:#e0e0f0'>{html_lib.escape(pair.sentence_a)}</td>
  <td style='padding:8px;color:#e0e0f0'>{html_lib.escape(pair.sentence_b)}</td>
  <td style='padding:8px;text-align:center;font-weight:700;color:{score_color}'>{pct:.1f}%</td>
</tr>""")

    # ⚠️  FIX: renamed `html` → `table_html` to avoid shadowing the stdlib module
    table_html = f"""<table style='width:100%;border-collapse:collapse;background:#12121f;border-radius:8px;overflow:hidden'>
  <thead>
    <tr style='background:#1e1e3f'>
      <th style='padding:10px;color:#a78bfa'>#</th>
      <th style='padding:10px;color:#a78bfa;text-align:left'>{html_lib.escape(result.doc_a)} — sentence</th>
      <th style='padding:10px;color:#a78bfa;text-align:left'>{html_lib.escape(result.doc_b)} — sentence</th>
      <th style='padding:10px;color:#a78bfa'>Score</th>
    </tr>
  </thead>
  <tbody>
    {"".join(rows)}
  </tbody>
</table>"""
    return table_html
