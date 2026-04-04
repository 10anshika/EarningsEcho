"""
finbert_scorer.py
Sentence-level FinBERT sentiment scoring for earnings text.

Public API
----------
score_text(text: str, pipe=None) -> dict
    Returns per-section sentiment ratios and trajectory.
load_pipeline() -> transformers.Pipeline
    Lazy-load the FinBERT pipeline (cached after first call).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import nltk
from loguru import logger

# Download punkt tokenizer data silently on first use
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize

MODEL_NAME = "ProsusAI/finbert"
# FinBERT max token length; we truncate longer sentences
MAX_TOKENS = 512
# Minimum sentence length to bother classifying (avoids noise on short fragments)
MIN_SENT_CHARS = 20
# Batch size for pipeline inference
BATCH_SIZE = 16


@lru_cache(maxsize=1)
def load_pipeline():
    """Load and cache the FinBERT text-classification pipeline."""
    from transformers import pipeline as hf_pipeline

    logger.info(f"Loading FinBERT model: {MODEL_NAME} (CPU, first call only)")
    pipe = hf_pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=-1,          # CPU
        truncation=True,
        max_length=MAX_TOKENS,
        top_k=None,         # return all three label scores
    )
    logger.success("FinBERT pipeline ready")
    return pipe


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK punkt, filter short fragments."""
    raw_sents = sent_tokenize(text)
    return [s.strip() for s in raw_sents if len(s.strip()) >= MIN_SENT_CHARS]


def _classify_sentences(sentences: list[str], pipe) -> list[dict]:
    """
    Run FinBERT on a list of sentences in batches.
    Returns list of {positive, negative, neutral} score dicts.
    """
    if not sentences:
        return []

    results = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        raw = pipe(batch)
        for label_list in raw:
            # top_k=None returns list of {label, score} for all classes
            scores = {item["label"]: item["score"] for item in label_list}
            results.append(
                {
                    "positive": scores.get("positive", 0.0),
                    "negative": scores.get("negative", 0.0),
                    "neutral": scores.get("neutral", 0.0),
                }
            )
    return results


def _aggregate(classified: list[dict]) -> dict:
    """
    Compute ratios and dominant-label counts from a list of sentence scores.
    Uses the highest-scoring label per sentence as the classification.
    """
    if not classified:
        return {
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 0.0,
            "avg_positive": 0.0,
            "avg_negative": 0.0,
            "sentence_count": 0,
        }

    n = len(classified)
    pos_count = neg_count = neu_count = 0
    sum_pos = sum_neg = 0.0

    for s in classified:
        dominant = max(s, key=s.get)
        if dominant == "positive":
            pos_count += 1
        elif dominant == "negative":
            neg_count += 1
        else:
            neu_count += 1
        sum_pos += s["positive"]
        sum_neg += s["negative"]

    return {
        "positive_ratio": pos_count / n,
        "negative_ratio": neg_count / n,
        "neutral_ratio": neu_count / n,
        "avg_positive": sum_pos / n,
        "avg_negative": sum_neg / n,
        "sentence_count": n,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_text(
    opening_remarks: str,
    qa_section: str,
    pipe=None,
) -> dict:
    """
    Run FinBERT sentence-level classification on both text sections.

    Parameters
    ----------
    opening_remarks : str   — prepared remarks / narrative portion
    qa_section      : str   — financial tables / Q&A portion
    pipe            : optional pre-loaded pipeline (loads from cache if None)

    Returns
    -------
    dict with keys:
        opening_positive_ratio, opening_negative_ratio, opening_neutral_ratio,
        qa_positive_ratio, qa_negative_ratio, qa_neutral_ratio,
        overall_positive_ratio, overall_negative_ratio, overall_neutral_ratio,
        sentiment_trajectory   (opening avg_positive − qa avg_positive;
                                positive = sentiment fell in Q&A),
        opening_sentence_count, qa_sentence_count
    """
    if pipe is None:
        pipe = load_pipeline()

    opening_sents = _split_sentences(opening_remarks)
    qa_sents = _split_sentences(qa_section)

    logger.debug(
        f"Scoring {len(opening_sents)} opening + {len(qa_sents)} QA sentences"
    )

    opening_classified = _classify_sentences(opening_sents, pipe)
    qa_classified = _classify_sentences(qa_sents, pipe)

    opening_agg = _aggregate(opening_classified)
    qa_agg = _aggregate(qa_classified)

    # Combined across both sections
    all_classified = opening_classified + qa_classified
    overall_agg = _aggregate(all_classified)

    # Trajectory: positive = sentiment dropped from opening to Q&A
    # (higher in opening than in Q&A)
    trajectory = opening_agg["avg_positive"] - qa_agg["avg_positive"]

    return {
        # Opening section
        "opening_positive_ratio": round(opening_agg["positive_ratio"], 4),
        "opening_negative_ratio": round(opening_agg["negative_ratio"], 4),
        "opening_neutral_ratio": round(opening_agg["neutral_ratio"], 4),
        "opening_sentence_count": opening_agg["sentence_count"],
        # QA / tables section
        "qa_positive_ratio": round(qa_agg["positive_ratio"], 4),
        "qa_negative_ratio": round(qa_agg["negative_ratio"], 4),
        "qa_neutral_ratio": round(qa_agg["neutral_ratio"], 4),
        "qa_sentence_count": qa_agg["sentence_count"],
        # Overall
        "overall_positive_ratio": round(overall_agg["positive_ratio"], 4),
        "overall_negative_ratio": round(overall_agg["negative_ratio"], 4),
        "overall_neutral_ratio": round(overall_agg["neutral_ratio"], 4),
        # Key derived metric
        "sentiment_trajectory": round(trajectory, 4),
    }
