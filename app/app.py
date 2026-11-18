import sys
import os

# Allow imports from project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import altair as alt
import pandas as pd
import streamlit as st

from src import predictor
from src.predictor import AUTHORS, METRIC_NAMES

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="DeltaQuill · Stylometric Fingerprinting",
    page_icon="🪶",
    layout="wide",
)

# Fixed colour per author (validated categorical slots 1-3; colour follows the
# author, never the rank).
AUTHOR_COLORS = {
    "dostoevsky": "#2a78d6",
    "orwell": "#eb6834",
    "sharma": "#1baf7a",
}

SAMPLES = {
    "Dostoevsky-style": (
        "I analyze myself relentlessly, as though I were both the criminal and the judge, "
        "and yet I cannot say, gentlemen, whether I despise myself more for the deed or for "
        "the trembling that came after it. Do you understand me? It was not poverty, no, it "
        "was something else entirely, some feverish idea that had taken root in me, that "
        "whispered to me at night in that cramped little room, until I could no longer tell "
        "whether I was ill or merely wretched, and whether there was any difference at all."
    ),
    "Orwell-style": (
        "The government reduced the ration again. The announcement was brief. No explanation "
        "was offered. In the canteen the workers ate in silence, and the telescreen went on "
        "reporting that production had risen by twelve percent. Nobody questioned the figure. "
        "Those who remembered the old ration said nothing, because remembering was itself a "
        "kind of crime. The posters on the walls watched them, as they always did, and the "
        "party was satisfied."
    ),
    "Sharma-style": (
        "Success grows from disciplined daily habits. The way you begin your morning shapes "
        "the quality of your entire day, and the quality of your days shapes your life. So "
        "rise early. Protect your first hour. Invest in your mindset, your heartset, your "
        "healthset and your soulset. Remember that every master was once a beginner, and that "
        "greatness is built through small, consistent improvements that, over time, deliver "
        "spectacular results. Lead without a title and make your life a masterpiece."
    ),
}

# ---------------------------
# Styling
# ---------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1150px; }
      .dq-hero h1 { font-size: 2.4rem; margin-bottom: 0.1rem; }
      .dq-hero p { color: #52514e; font-size: 1.05rem; margin-top: 0; }
      .dq-card {
        border: 1px solid rgba(0,0,0,0.08);
        border-left: 6px solid var(--accent);
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      }
      .dq-card .label { color: #52514e; font-size: 0.85rem; text-transform: uppercase;
                        letter-spacing: 0.06em; margin: 0; }
      .dq-card .name { color: #0b0b0b; font-size: 1.9rem; font-weight: 700; margin: 0.15rem 0; }
      .dq-card .meta { color: #52514e; font-size: 0.95rem; margin: 0; }
      .dq-badge { display: inline-block; padding: 0.1rem 0.6rem; border-radius: 999px;
                  font-size: 0.8rem; font-weight: 600; margin-left: 0.4rem;
                  background: #f0efec; color: #0b0b0b; }
      .dq-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px;
                   margin-right: 6px; vertical-align: middle; }
      .dq-footer { margin-top: 3.5rem; padding: 1.6rem 0 0.4rem;
                   border-top: 1px solid rgba(0,0,0,0.08); text-align: center; }
      .dq-footer .tagline { color: #52514e; font-size: 0.98rem; font-style: italic;
                            max-width: 640px; margin: 0 auto 0.9rem; line-height: 1.55; }
      .dq-footer .who { color: #0b0b0b; font-weight: 700; font-size: 1.05rem; margin: 0 0 0.5rem; }
      .dq-footer .links { display: flex; justify-content: center; gap: 1.4rem; flex-wrap: wrap; }
      .dq-footer .links a { display: inline-flex; align-items: center; gap: 0.4rem;
                            color: #52514e !important; text-decoration: none; font-size: 0.92rem; }
      .dq-footer .links a:hover { color: #2a78d6 !important; }
      .dq-footer svg { width: 16px; height: 16px; fill: currentColor; }
      .dq-footer .copy { color: #8a8984; font-size: 0.8rem; margin-top: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource
def get_artifacts():
    return predictor.load_artifacts()


@st.cache_data
def get_dataset_info():
    model, vectorizer = get_artifacts()
    df = predictor.load_dataset()
    return predictor.evaluate(model, vectorizer, df), predictor.author_profiles(df)


model, vectorizer = get_artifacts()
eval_stats, profiles = get_dataset_info()


# ---------------------------
# Callbacks
# ---------------------------
def load_sample(name):
    st.session_state.input_text = SAMPLES[name]


def clear_text():
    st.session_state.input_text = ""


def load_upload():
    file = st.session_state.get("uploaded_file")
    if file is not None:
        st.session_state.input_text = file.getvalue().decode("utf-8", errors="ignore")


st.session_state.setdefault("input_text", "")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("## 🪶 DeltaQuill")
    st.caption("Writing Analysis and Authorship Detection Engine")

    st.markdown("#### Model")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{eval_stats['accuracy']:.1%}")
    c2.metric("TF-IDF features", f"{eval_stats['features']:,}")
    c1.metric("Training chunks", eval_stats["train_chunks"])
    c2.metric("Test chunks", eval_stats["test_chunks"])
    st.caption("Logistic Regression on word unigrams + bigrams. Accuracy is measured "
               "at book level, on books the model never saw during training.")

    st.markdown("#### Known authors")
    for key, info in AUTHORS.items():
        st.markdown(
            f"<span class='dq-swatch' style='background:{AUTHOR_COLORS[key]}'></span>"
            f"**{info['name']}**<br>"
            f"<small>Train: <i>{info['train_book']}</i> · Test: <i>{info['test_book']}</i></small>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Limitations")
    st.caption("The model always picks one of the three known authors, so text by anyone "
               "else will still be assigned to one of them. Use 100+ words for reliable results.")

# ---------------------------
# Header
# ---------------------------
st.markdown(
    "<div class='dq-hero'><h1>Stylometric Fingerprinting</h1>"
    "<p>Every writer leaves a fingerprint. Paste a passage to find out whose style it matches.</p></div>",
    unsafe_allow_html=True,
)

# ---------------------------
# Input
# ---------------------------
with st.container(border=True):
    st.markdown("**Try a sample**")
    cols = st.columns([1, 1, 1, 0.6])
    for col, name in zip(cols, SAMPLES):
        col.button(name, on_click=load_sample, args=(name,), width="stretch")
    cols[3].button("Clear", on_click=clear_text, width="stretch", type="tertiary")

    text = st.text_area(
        "Input text",
        key="input_text",
        height=220,
        placeholder="Paste a paragraph or more of English prose here…",
    )

    with st.expander("Or upload a .txt file"):
        st.file_uploader(
            "Text file", type=["txt"], key="uploaded_file",
            on_change=load_upload, label_visibility="collapsed",
        )

    n_words = len(text.split())
    left, right = st.columns([3, 1], vertical_alignment="center")
    left.caption(f"{n_words:,} words" + ("" if n_words >= 50 or n_words == 0
                                          else " · 50+ recommended"))
    analyze = right.button("Analyze text", type="primary", width="stretch",
                           disabled=n_words == 0)

# ---------------------------
# Footer
# ---------------------------
GITHUB_ICON = (
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
    '2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94'
    '-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 '
    '2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36'
    '-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 '
    '2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73'
    '.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
)
LINKEDIN_ICON = (
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M0 1.15C0 .52.52 0 1.17 0h13.66C15.48 '
    '0 16 .52 16 1.15v13.7c0 .63-.52 1.15-1.17 1.15H1.17C.52 16 0 15.48 0 14.85V1.15zm4.94 '
    '12.24V6.17H2.54v7.22h2.4zM3.74 5.18c.84 0 1.36-.55 1.36-1.25-.02-.71-.52-1.25-1.35-1.25'
    '-.82 0-1.36.54-1.36 1.25 0 .7.52 1.25 1.33 1.25h.02zm4.9 8.21V9.36c0-.22.02-.43.08-.59.18'
    '-.43.57-.88 1.24-.88.87 0 1.22.66 1.22 1.64v3.86h2.4V9.25c0-2.22-1.18-3.25-2.76-3.25-1.28 '
    '0-1.85.7-2.17 1.2v.03h-.02l.02-.03V6.17h-2.4c.03.68 0 7.22 0 7.22h2.4z"/></svg>'
)
MAIL_ICON = (
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M0 4a2 2 0 012-2h12a2 2 0 012 2v8a2 '
    '2 0 01-2 2H2a2 2 0 01-2-2V4zm2-1a1 1 0 00-1 1v.22l7 4.2 7-4.2V4a1 1 0 00-1-1H2zm13 '
    '2.38-4.7 2.82L15 11.1V5.38zm-.03 6.85L9.4 8.8 8 9.62 6.6 8.8l-5.57 3.43A1 1 0 002 13h12a1 '
    '1 0 00.97-.77zM1 11.1l4.7-2.9L1 5.38v5.72z"/></svg>'
)


def render_footer():
    st.markdown(
        f"""
        <div class="dq-footer">
          <p class="tagline">A lifelong reader at heart: lost in Dostoevsky's St. Petersburg,
             haunted by Orwell's Oceania, and up at 5 AM with Robin Sharma.
             DeltaQuill is my love letter to the books that shaped me.</p>
          <p class="who">Soumika Devarakonda</p>
          <div class="links">
            <a href="https://github.com/soumikadevarakonda/" target="_blank" rel="noopener">{GITHUB_ICON}GitHub</a>
            <a href="https://www.linkedin.com/in/soumikadevarakonda" target="_blank" rel="noopener">{LINKEDIN_ICON}LinkedIn</a>
            <a href="mailto:soumikadevarakonda@gmail.com" title="Email" aria-label="Email">{MAIL_ICON}</a>
          </div>
          <p class="copy">🪶 DeltaQuill · Stylometric Fingerprinting</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# Results
# ---------------------------
if analyze:
    label, probs, vec = predictor.predict(model, vectorizer, text)
    top_prob = float(probs.loc[0, "probability"])
    info = AUTHORS.get(label, {"name": label.title(), "style": "", "train_book": ""})

    if vec.nnz == 0:
        st.warning("None of the words in this text appear in the model's vocabulary, "
                   "so no prediction can be made. Try a longer English passage.")
        render_footer()
        st.stop()

    if top_prob >= 0.75:
        level = "High confidence"
    elif top_prob >= 0.5:
        level = "Moderate confidence"
    else:
        level = "Low confidence"

    st.markdown("### Result")
    res_left, res_right = st.columns([1, 1.25], gap="large")

    with res_left:
        st.markdown(
            f"""
            <div class="dq-card" style="--accent:{AUTHOR_COLORS.get(label, '#52514e')}">
              <p class="label">Predicted author</p>
              <p class="name">{info['name']}</p>
              <p class="meta"><b>{top_prob:.0%}</b> probability
                 <span class="dq-badge">{level}</span></p>
              <p class="meta" style="margin-top:0.6rem">{info['style']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if top_prob < 0.5:
            st.info("No author clearly matches. This text may not be by any of the three "
                    "known authors, or it may be too short to judge.")
        elif n_words < 50:
            st.info("Short inputs give less reliable predictions. 50+ words recommended.")

    with res_right:
        st.markdown("**Probability by author**")
        probs["pct"] = probs["probability"].map(lambda p: f"{p:.1%}")
        base = alt.Chart(probs).encode(
            y=alt.Y("name:N", sort=list(probs["name"]), title=None,
                    axis=alt.Axis(labelFontSize=13, labelLimit=200, ticks=False, domain=False)),
            x=alt.X("probability:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", tickCount=5, grid=True, gridOpacity=0.4)),
            tooltip=[alt.Tooltip("name:N", title="Author"),
                     alt.Tooltip("probability:Q", title="Probability", format=".1%")],
        )
        bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
            color=alt.Color(
                "author:N", legend=None,
                scale=alt.Scale(domain=list(AUTHOR_COLORS), range=list(AUTHOR_COLORS.values())),
            )
        )
        labels = base.mark_text(align="left", dx=6, fontSize=13, color="#0b0b0b").encode(
            text="pct:N"
        )
        st.altair_chart((bars + labels).properties(height=150), width="stretch")

    tab_why, tab_style = st.tabs(["🔍 Why this author?", "📐 Style profile"])

    # --- Explanation ---
    with tab_why:
        terms = predictor.top_terms(model, vectorizer, vec, label, n=10)
        if terms.empty:
            st.caption("No known vocabulary in the input pushed the model toward this author.")
        else:
            st.caption(f"Words and phrases from your text that pushed the model most toward "
                       f"**{info['name']}** (TF-IDF weight × model coefficient). Common function "
                       f"words such as *the*, *was* or *of* often rank highly: how often a writer "
                       f"uses them is one of the classic stylometric fingerprints.")
            chart = alt.Chart(terms).mark_bar(
                cornerRadiusEnd=4, height=16, color=AUTHOR_COLORS.get(label, "#2a78d6")
            ).encode(
                y=alt.Y("term:N", sort=list(terms["term"]), title=None,
                        axis=alt.Axis(labelFontSize=13, labelLimit=220, ticks=False, domain=False)),
                x=alt.X("contribution:Q", title="Contribution to score",
                        axis=alt.Axis(grid=True, gridOpacity=0.4, format=".2f")),
                tooltip=[alt.Tooltip("term:N", title="Term"),
                         alt.Tooltip("contribution:Q", title="Contribution", format=".3f")],
            ).properties(height=max(120, 28 * len(terms)))
            st.altair_chart(chart, width="stretch")

    # --- Stylometric profile ---
    with tab_style:
        mine = predictor.style_metrics(text)

        m = st.columns(4)
        m[0].metric("Words", f"{n_words:,}", border=True)
        m[1].metric("Avg sentence length", f"{mine[1]:.1f} words", border=True)
        m[2].metric("Avg word length", f"{mine[0]:.2f} chars", border=True)
        m[3].metric("Vocabulary richness", f"{mine[7]:.2f}", border=True,
                    help="Type-token ratio: unique words / total words. Depends on text "
                         "length, so short inputs score higher than the 1,000-word "
                         "training chunks.")

        table = profiles.copy()
        table.columns = [AUTHORS.get(a, {}).get("name", a.title()) for a in table.columns]
        author_cols = list(table.columns)
        table.insert(0, "Your text", mine)
        table["Closest author"] = [
            min(author_cols, key=lambda c: abs(row[c] - row["Your text"]))
            for _, row in table.iterrows()
        ]
        table.index.name = "Metric"

        st.caption("Your text compared with each author's average over 1,000-word chunks "
                   "from their books. Shown for interpretation only; the prediction uses "
                   "TF-IDF features.")
        st.dataframe(
            table,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.2f")
                for c in ["Your text", *author_cols]
            },
        )
else:
    st.caption("Results will appear here after you click **Analyze text**.")

render_footer()
