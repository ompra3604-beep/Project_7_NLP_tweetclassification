from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import streamlit as st

from predictor import classify_tweet, load_classifier

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MAX_TWEET_LENGTH = 5_000
MAX_BATCH_ROWS = 2_000
MAX_HISTORY = 500
DEVELOPER_NAME = "Om Prakash"
DEVELOPER_TITLE = "Data Science & AI Engineer"

SAMPLE_TWEETS = {
    "Earthquake (literal)": "BREAKING: A 6.8 magnitude earthquake struck the coast, buildings collapsed and rescue teams are searching for survivors #earthquake",
    "Wildfire (literal)": "Evacuation orders issued as wildfire spreads rapidly across the hills, thousands forced to leave their homes.",
    "Flood (literal)": "Severe flooding in the city after record rainfall. Roads submerged, please avoid travel #flood",
    "Figurative use": "My plans collapsed after a long week, this day is a total disaster lol",
    "Casual / unrelated": "Just had the best coffee ever, this cafe is on fire today!",
    "Sports metaphor": "That goalkeeper was absolutely on fire, the crowd exploded after the final whistle.",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")
WORD_RE = re.compile(r"\b\w+\b")

DISASTER_HINTS = {
    "earthquake", "flood", "flooding", "wildfire", "fire", "tsunami", "hurricane",
    "cyclone", "tornado", "storm", "landslide", "evacuate", "evacuation", "collapsed",
    "explosion", "rescue", "survivors", "casualties", "emergency", "disaster",
    "drought", "volcano", "eruption", "typhoon", "blizzard", "famine",
}

st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon=":material/crisis_alert:",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 5rem;}
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 14px;
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 55%, #ea580c 100%);
        color: #fff; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2rem; color: #fff;}
    .hero p {margin: .4rem 0 0 0; opacity: .92;}
    .footer {
        position: fixed; left: 0; right: 0; bottom: 0; text-align: center;
        padding: .55rem; font-size: .85rem; z-index: 100;
        background: rgba(127,127,127,.12); backdrop-filter: blur(6px);
    }
    mark.hit {background: #fde68a; color: #111; padding: 0 .2rem; border-radius: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def get_classifier():
    """Load the serialized model once and reuse it across Streamlit reruns."""
    return load_classifier()


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("tweet_input", "")
    st.session_state.setdefault("last_result", None)


def analyze_text(text: str) -> dict:
    words = WORD_RE.findall(text)
    lowered = {w.lower() for w in words}
    return {
        "characters": len(text),
        "words": len(words),
        "hashtags": HASHTAG_RE.findall(text),
        "mentions": MENTION_RE.findall(text),
        "urls": URL_RE.findall(text),
        "hints": sorted(lowered & DISASTER_HINTS),
    }


def highlight_hints(text: str, hints: list[str]) -> str:
    safe = (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    if not hints:
        return safe
    pattern = re.compile(r"\b(" + "|".join(re.escape(h) for h in hints) + r")\b", re.I)
    return pattern.sub(r'<mark class="hit">\1</mark>', safe)


def decide_label(result: dict, threshold: float) -> int:
    """Apply the user threshold when a score exists, else fall back to the model label."""
    score = result.get("score")
    if score is None:
        return int(result["target"])
    return int(float(score) >= threshold)


def confidence_band(score: float | None, threshold: float) -> str:
    if score is None:
        return "n/a"
    distance = abs(score - threshold)
    if distance >= 0.35:
        return "High"
    if distance >= 0.15:
        return "Medium"
    return "Low"


def classify_and_record(text: str, threshold: float, source: str = "single") -> dict:
    raw = classify_tweet(get_classifier(), text)
    score = None if raw.get("score") is None else float(raw["score"])
    label = decide_label(raw, threshold)
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tweet": text,
        "label": "Disaster" if label == 1 else "Not disaster",
        "target": label,
        "score": score,
        "confidence": confidence_band(score, threshold),
        "source": source,
    }
    history = st.session_state["history"]
    history.append(record)
    del history[:-MAX_HISTORY]
    return record


def load_sample(name: str) -> None:
    st.session_state["tweet_input"] = SAMPLE_TWEETS[name]


def clear_input() -> None:
    st.session_state["tweet_input"] = ""
    st.session_state["last_result"] = None


def history_df() -> pd.DataFrame:
    cols = ["time", "tweet", "label", "target", "score", "confidence", "source"]
    return pd.DataFrame(st.session_state["history"], columns=cols)


def show_result_card(record: dict, threshold: float) -> None:
    with st.container(border=True):
        st.subheader("Classification result")
        c1, c2, c3 = st.columns(3)
        with c1:
            if record["target"] == 1:
                st.badge("Disaster-related", icon=":material/warning:", color="red")
            else:
                st.badge("Not disaster-related", icon=":material/check_circle:", color="green")
        c2.metric("Disaster score", "n/a" if record["score"] is None else f"{record['score']:.1%}")
        c3.metric("Decision confidence", record["confidence"])

        if record["score"] is not None:
            st.progress(
                min(max(record["score"], 0.0), 1.0),
                text=f"Score vs. threshold {threshold:.0%} · not a calibrated probability",
            )

        info = analyze_text(record["tweet"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Characters", info["characters"])
        m2.metric("Words", info["words"])
        m3.metric("Hashtags", len(info["hashtags"]))
        m4.metric("URLs / mentions", f"{len(info['urls'])} / {len(info['mentions'])}")

        with st.expander("Tweet review & keyword hints", expanded=True):
            st.markdown(highlight_hints(record["tweet"], info["hints"]), unsafe_allow_html=True)
            if info["hints"]:
                st.caption(
                    "Highlighted words are common disaster vocabulary shown for reference only; "
                    "the model decides from full context."
                )
            else:
                st.caption("No common disaster keywords detected in this tweet.")

        st.caption(
            "This automated estimate may be wrong, especially for sarcasm, slang, or unfamiliar events. "
            "Verify urgent reports with trusted sources before acting."
        )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
init_state()

with st.sidebar:
    st.header(":material/tune: Settings")
    threshold = st.slider(
        "Decision threshold",
        0.05, 0.95, 0.50, 0.05,
        help="Tweets with a score at or above this value are labelled disaster-related. "
             "Raise it to reduce false alarms, lower it to catch more possible disasters.",
    )
    st.caption("Applies only when the model returns a score.")
    st.divider()
    st.subheader("Quick examples")
    for sample_name in SAMPLE_TWEETS:
        st.button(sample_name, on_click=load_sample, args=(sample_name,), width="stretch", key=f"s_{sample_name}")
    st.divider()
    st.subheader("Session")
    st.metric("Tweets analyzed", len(st.session_state["history"]))
    if st.button("Reset session history", icon=":material/delete:", width="stretch"):
        st.session_state["history"] = []
        st.session_state["last_result"] = None
        st.rerun()
    st.divider()
    st.markdown(f"**Developed by {DEVELOPER_NAME}**  \n{DEVELOPER_TITLE}")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🚨 Disaster Tweet Classifier</h1>
        <p>Estimate whether a tweet describes a real disaster event, analyze batches, and track results on a live dashboard.</p>
        <p style="font-size:.85rem; opacity:.85;">Developed by <b>{DEVELOPER_NAME}</b> · {DEVELOPER_TITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_dash, tab_about = st.tabs(
    [":material/search: Single tweet", ":material/upload_file: Batch CSV", ":material/dashboard: Dashboard", ":material/info: About"]
)

# ----------------------------------------------------------------------------
# Tab 1: single tweet
# ----------------------------------------------------------------------------
with tab_single:
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Analyze a tweet")
            with st.form("tweet_classification", clear_on_submit=False):
                tweet = st.text_area(
                    "Tweet text",
                    key="tweet_input",
                    placeholder="Paste the tweet as it appeared, including hashtags if present…",
                    max_chars=MAX_TWEET_LENGTH,
                    height=160,
                    help=f"Enter up to {MAX_TWEET_LENGTH:,} characters. URLs and punctuation are handled by the model preprocessing.",
                )
                b1, b2 = st.columns([3, 1])
                submitted = b1.form_submit_button(
                    "Classify tweet", type="primary", icon=":material/search:", width="stretch"
                )
                b2.form_submit_button("Clear", on_click=clear_input, width="stretch")

        if submitted:
            text = tweet.strip()
            if not text:
                st.warning("Enter a tweet before classifying.", icon=":material/info:")
            else:
                try:
                    with st.spinner("Classifying…"):
                        st.session_state["last_result"] = classify_and_record(text, threshold)
                except FileNotFoundError:
                    st.error("The trained model file is missing. Restore the model artifact and try again.")
                    st.stop()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not classify this tweet: {exc}")
                    st.stop()

        if st.session_state["last_result"]:
            show_result_card(st.session_state["last_result"], threshold)

    with right:
        with st.container(border=True):
            st.subheader("Recent analyses")
            df = history_df()
            if df.empty:
                st.caption("Your classified tweets will appear here.")
            else:
                recent = df.tail(5).iloc[::-1]
                for _, row in recent.iterrows():
                    icon = "🔴" if row["target"] == 1 else "🟢"
                    preview = row["tweet"][:80] + ("…" if len(row["tweet"]) > 80 else "")
                    score_txt = "" if row["score"] is None or pd.isna(row["score"]) else f" · {row['score']:.0%}"
                    st.markdown(f"{icon} **{row['label']}**{score_txt}  \n<small>{preview}</small>", unsafe_allow_html=True)

        with st.expander("See how context can change meaning"):
            st.markdown("**Literal event:** “The bridge collapsed after the earthquake.”")
            st.markdown("**Figurative use:** “My plans collapsed after a long week.”")
            st.caption("These are examples only; the classifier can still make mistakes on either type.")

# ----------------------------------------------------------------------------
# Tab 2: batch CSV
# ----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Classify many tweets at once")
    st.write(f"Upload a CSV, choose the column containing tweet text, and classify up to {MAX_BATCH_ROWS:,} rows.")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the CSV: {exc}")
            data = None

        if data is not None and not data.empty:
            text_cols = [c for c in data.columns if data[c].dtype == object] or list(data.columns)
            default_idx = next((i for i, c in enumerate(text_cols) if str(c).lower() in {"text", "tweet", "content"}), 0)
            column = st.selectbox("Tweet text column", text_cols, index=default_idx)
            st.caption(f"{len(data):,} rows detected. Preview:")
            st.dataframe(data.head(5), width="stretch")

            if st.button("Run batch classification", type="primary", icon=":material/play_arrow:"):
                rows = data[column].dropna().astype(str).str.strip()
                rows = rows[rows != ""].head(MAX_BATCH_ROWS)
                if rows.empty:
                    st.warning("No usable text found in that column.")
                else:
                    out, bar = [], st.progress(0.0, text="Starting…")
                    try:
                        for i, text in enumerate(rows, start=1):
                            rec = classify_and_record(text[:MAX_TWEET_LENGTH], threshold, source="batch")
                            out.append(rec)
                            if i % 10 == 0 or i == len(rows):
                                bar.progress(i / len(rows), text=f"Classified {i:,} of {len(rows):,}")
                    except FileNotFoundError:
                        st.error("The trained model file is missing. Restore the model artifact and try again.")
                        st.stop()
                    bar.empty()
                    st.session_state["batch_out"] = pd.DataFrame(out)

    batch_out = st.session_state.get("batch_out")
    if isinstance(batch_out, pd.DataFrame) and not batch_out.empty:
        st.success(f"Classified {len(batch_out):,} tweets.")
        k1, k2, k3 = st.columns(3)
        k1.metric("Disaster-related", int((batch_out["target"] == 1).sum()))
        k2.metric("Not disaster-related", int((batch_out["target"] == 0).sum()))
        k3.metric("Share disaster", f"{(batch_out['target'] == 1).mean():.1%}")
        st.dataframe(
            batch_out[["tweet", "label", "score", "confidence"]],
            width="stretch",
            column_config={
                "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
        st.download_button(
            "Download results (CSV)",
            batch_out.to_csv(index=False).encode("utf-8"),
            file_name="disaster_tweet_predictions.csv",
            mime="text/csv",
            icon=":material/download:",
        )

# ----------------------------------------------------------------------------
# Tab 3: dashboard
# ----------------------------------------------------------------------------
with tab_dash:
    st.subheader("Session dashboard")
    df = history_df()

    if df.empty:
        st.info("No data yet. Classify a tweet or run a batch to populate the dashboard.", icon=":material/insights:")
    else:
        scored = df.dropna(subset=["score"])
        total = len(df)
        n_dis = int((df["target"] == 1).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total analyzed", f"{total:,}")
        m2.metric("Disaster-related", f"{n_dis:,}", f"{n_dis / total:.0%} of total", delta_color="off")
        m3.metric("Not disaster", f"{total - n_dis:,}")
        m4.metric("Average score", "n/a" if scored.empty else f"{scored['score'].mean():.1%}")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("**Label distribution**")
                st.bar_chart(df["label"].value_counts(), horizontal=True)
        with c2:
            with st.container(border=True):
                st.markdown("**Decision confidence**")
                st.bar_chart(df["confidence"].value_counts())

        c3, c4 = st.columns(2, gap="large")
        with c3:
            with st.container(border=True):
                st.markdown("**Score over time**")
                if scored.empty:
                    st.caption("The model did not return scores.")
                else:
                    st.line_chart(scored.reset_index(drop=True)["score"])
        with c4:
            with st.container(border=True):
                st.markdown("**Score distribution**")
                if scored.empty:
                    st.caption("The model did not return scores.")
                else:
                    bins = pd.cut(scored["score"], bins=[i / 10 for i in range(11)], include_lowest=True)
                    hist = bins.value_counts().sort_index()
                    hist.index = [f"{iv.left:.1f}-{iv.right:.1f}" for iv in hist.index]
                    st.bar_chart(hist)

        # Most frequent hashtags and keyword hints
        hashtags: list[str] = []
        hints: list[str] = []
        for t in df["tweet"]:
            info = analyze_text(t)
            hashtags += [h.lower() for h in info["hashtags"]]
            hints += info["hints"]

        c5, c6 = st.columns(2, gap="large")
        with c5:
            with st.container(border=True):
                st.markdown("**Top hashtags**")
                if hashtags:
                    st.bar_chart(pd.Series(hashtags).value_counts().head(8), horizontal=True)
                else:
                    st.caption("No hashtags found yet.")
        with c6:
            with st.container(border=True):
                st.markdown("**Top disaster keywords**")
                if hints:
                    st.bar_chart(pd.Series(hints).value_counts().head(8), horizontal=True)
                else:
                    st.caption("No disaster keywords found yet.")

        st.markdown("**History**")
        f1, f2 = st.columns([1, 3])
        label_filter = f1.selectbox("Filter", ["All", "Disaster", "Not disaster"])
        shown = df if label_filter == "All" else df[df["label"] == label_filter]
        st.dataframe(
            shown.iloc[::-1][["time", "tweet", "label", "score", "confidence", "source"]],
            width="stretch",
            column_config={
                "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
        st.download_button(
            "Export history (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="classification_history.csv",
            mime="text/csv",
            icon=":material/download:",
        )

# ----------------------------------------------------------------------------
# Tab 4: about
# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        This app uses a trained NLP classifier to estimate whether a tweet refers to a **real disaster event**
        or uses disaster-related words in an everyday or figurative way.

        **Features**
        - Single-tweet classification with adjustable decision threshold
        - Keyword highlighting and tweet statistics (words, hashtags, URLs, mentions)
        - Batch CSV classification with downloadable predictions
        - Live dashboard: label mix, score trends, confidence, top hashtags and keywords
        - Session history with filtering and CSV export

        **Limitations**
        - Scores are model estimates and are **not calibrated probabilities**.
        - Sarcasm, slang, and brand-new events can cause mistakes.
        - History lives only in the current browser session.
        """
    )
    with st.container(border=True):
        st.markdown(f"### 👤 {DEVELOPER_NAME}")
        st.write(DEVELOPER_TITLE)

st.markdown(
    f'<div class="footer">Developed by <b>{DEVELOPER_NAME}</b> · {DEVELOPER_TITLE} · '
    "For exploration and educational use. Not an emergency alert service.</div>",
    unsafe_allow_html=True,
)
