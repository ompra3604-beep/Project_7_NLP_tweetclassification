from __future__ import annotations

import os
import re
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "disaster_tweet_classifier.joblib"
MODEL_PATH_ENV = "DISASTER_MODEL_PATH"

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
URL_COUNT_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
POSITIVE_WORDS = set("good great excellent happy love loved best wonderful amazing thank thanks safe success successful hope hopeful beautiful brave calm joy joyful pleased proud relief relieved strong support supportive win wins welcome".split())
NEGATIVE_WORDS = set("bad terrible awful sad hate hated worst horrible danger dangerous injured dead death killed kill fear afraid tragic tragedy emergency disaster damage damaged destroy destroyed hurt pain fire flood storm earthquake explosion crash attack evacuate evacuation missing suffer suffering".split())


def load_classifier(model_path: str | Path | None = None):
    configured_path = model_path or os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
    resolved_path = Path(configured_path)
    if not resolved_path.is_absolute():
        resolved_path = BASE_DIR / resolved_path
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Trained classifier not found: {resolved_path}")
    return joblib.load(resolved_path)


def clean_tweet(text: str) -> str:
    """Apply the same normalization used when training the classifier."""
    text = URL_RE.sub(" ", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_model_input(raw_text: str) -> pd.DataFrame:
    """Build the text and numeric fields expected by the trained pipeline."""
    clean_text = clean_tweet(raw_text)
    tokens = clean_text.split()
    token_count = len(tokens)
    positive_count = sum(token in POSITIVE_WORDS for token in tokens)
    negative_count = sum(token in NEGATIVE_WORDS for token in tokens)
    return pd.DataFrame([{
        "clean_text": clean_text,
        "char_count": len(raw_text),
        "token_count": token_count,
        "mean_token_length": (sum(map(len, tokens)) / token_count) if token_count else 0.0,
        "hashtag_count": len(HASHTAG_RE.findall(raw_text)),
        "mention_count": len(MENTION_RE.findall(raw_text)),
        "url_count": len(URL_COUNT_RE.findall(raw_text)),
        "exclamation_count": raw_text.count("!"),
        "question_count": raw_text.count("?"),
        "uppercase_ratio": (sum(char.isupper() for char in raw_text) / len(raw_text)) if raw_text else 0.0,
        "positive_word_count": positive_count,
        "negative_word_count": negative_count,
        "sentiment_polarity": ((positive_count - negative_count) / token_count) if token_count else 0.0,
    }])


def classify_tweet(model, text: str) -> dict[str, object]:
    features = make_model_input(text)
    predicted_label = int(model.predict(features)[0])
    result: dict[str, object] = {
        "label": "Disaster-related" if predicted_label == 1 else "Not disaster-related",
        "target": predicted_label,
        "score": None,
    }
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        result["score"] = float(model.predict_proba(features)[0][classes.index(1)])
    return result
