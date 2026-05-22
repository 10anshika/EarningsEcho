"""
app.py
EarningsEcho — Streamlit dashboard.

Panels
------
1. EW Risk Score gauge + SHAP hedge chart + component bars
2. Transcript with hedging highlights (amber) and negative sentences (light red)
3. Earnings-day price candlestick
4. Backtest summary (EDGAR vs Motley Fool accuracy split)
5. Corpus scatter (EW_Risk_Score vs actual 5d return)
6. Bilingual Summary (English + Hindi explanation via Claude)

Sidebar also shows last 10 MLflow runs.

Usage
-----
streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# ---------------------------------------------------------------------------
# Path bootstrap — ensure project root is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.edgar_fetcher import fetch_transcripts
from src.ingestion.transcript_parser import parse_transcript
from src.nlp.hedging_detector import _HEDGE_PATTERN
from src.nlp.finbert_scorer import load_pipeline, _split_sentences, _classify_sentences
from src.nlp.nlp_pipeline import analyze
from src.nlp.shap_explainer import explain_hedge_score
from src.backtest.stats import compute_stats
from src.tracking.mlflow_logger import log_run, get_recent_runs

SCORES_DIR      = ROOT / "data" / "scores"
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
BACKTEST_CSV    = ROOT / "data" / "backtest_results.csv"

RISK_RED = "#E24B4A"
RISK_GREEN = "#639922"
RISK_AMBER = "#BA7517"
INFO_BG = "#E6F1FB"
INFO_BORDER = "#185FA5"
INFO_TEXT = "#0C447C"

COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "ABBV": "AbbVie Inc.",
    "ADBE": "Adobe Inc.",
    "AMZN": "Amazon.com, Inc.",
    "AXP": "American Express Co.",
    "BAC": "Bank of America Corp.",
    "BLK": "BlackRock, Inc.",
    "C": "Citigroup Inc.",
    "COP": "ConocoPhillips",
    "COST": "Costco Wholesale Corp.",
    "CRM": "Salesforce, Inc.",
    "CVS": "CVS Health Corp.",
    "CVX": "Chevron Corp.",
    "EOG": "EOG Resources, Inc.",
    "GOOGL": "Alphabet Inc.",
    "GS": "Goldman Sachs Group, Inc.",
    "HD": "Home Depot, Inc.",
    "INTC": "Intel Corp.",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "LLY": "Eli Lilly and Co.",
    "MCD": "McDonald's Corp.",
    "MDT": "Medtronic plc",
    "META": "Meta Platforms, Inc.",
    "MPC": "Marathon Petroleum Corp.",
    "MRK": "Merck & Co., Inc.",
    "MS": "Morgan Stanley",
    "MSFT": "Microsoft Corp.",
    "NKE": "NIKE, Inc.",
    "NVDA": "NVIDIA Corp.",
    "PFE": "Pfizer Inc.",
    "PSX": "Phillips 66",
    "SBUX": "Starbucks Corp.",
    "SLB": "Schlumberger N.V.",
    "TGT": "Target Corp.",
    "TSLA": "Tesla, Inc.",
    "UNH": "UnitedHealth Group Inc.",
    "VLO": "Valero Energy Corp.",
    "WFC": "Wells Fargo & Co.",
    "XOM": "Exxon Mobil Corp.",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EarningsEcho",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stToolbar"]      { display: none !important; }
    [data-testid="stDecoration"]   { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    header[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; visibility: hidden !important; }
    #MainMenu                      { visibility: hidden !important; }
    footer                         { visibility: hidden !important; }
    .block-container               { padding-top: 1rem !important; }
    section[data-testid="stSidebar"] { top: 0 !important; }
    section[data-testid="stSidebar"] { display: block !important; min-width: 18rem; }
    section[data-testid="stSidebar"] > div { display: block !important; }
    section[data-testid="stSidebar"] { 
        display: flex !important; 
        width: 18rem !important;
        min-width: 18rem !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: 18rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading FinBERT model (one-time)...")
def _get_finbert():
    """Load and cache the FinBERT pipeline as a Streamlit resource (loaded once per session)."""
    return load_pipeline()


@st.cache_data(show_spinner="Fetching transcript from EDGAR...")
def _fetch_edgar_cached(ticker: str, n: int = 3) -> list[dict]:
    """Fetch up to n EDGAR 8-K transcript exhibits for ticker, cached by Streamlit."""
    return fetch_transcripts(ticker, n=n)


@st.cache_data(show_spinner="Running NLP pipeline...")
def _score_cached(parsed_path: str) -> dict:
    """Run full NLP pipeline on parsed_path, cached so re-selecting the same file is instant."""
    return analyze(parsed_path, pipe=_get_finbert())


@st.cache_data(show_spinner="Fetching price data...")
def _fetch_candles_cached(ticker: str, event_date_iso: str) -> Optional[pd.DataFrame]:
    """Fetch ~20 trading days of OHLCV around event_date_iso for the candlestick chart."""
    ed    = date.fromisoformat(event_date_iso)
    start = ed - timedelta(days=35)
    end   = ed + timedelta(days=12)
    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        event_ts    = pd.Timestamp(ed)
        idx         = df.index
        after       = idx[idx >= event_ts]
        before      = idx[idx < event_ts]
        keep_before = before[-15:] if len(before) >= 15 else before
        keep_after  = after[:7]    if len(after) >= 7   else after
        return df.loc[keep_before.union(keep_after)]
    except Exception:
        return None


@st.cache_data(show_spinner="Classifying sentences...")
def _classify_for_highlight(text: str) -> dict[str, bool]:
    """Return {sentence: is_negative} dict using FinBERT."""
    pipe  = _get_finbert()
    sents = _split_sentences(text)
    if not sents:
        return {}
    scores = _classify_sentences(sents, pipe)
    return {
        s: (max(sc, key=sc.get) == "negative")
        for s, sc in zip(sents, scores)
    }


@st.cache_data(show_spinner="Classifying sentences for highlights...")
def _classify_both_sections(opening: str, qa: str) -> tuple[dict[str, bool], dict[str, bool]]:
    """Classify opening + Q&A in a single batched FinBERT pass instead of two separate calls."""
    pipe = _get_finbert()
    opening_sents = _split_sentences(opening) if opening else []
    qa_sents      = _split_sentences(qa)      if qa      else []
    all_sents     = opening_sents + qa_sents
    if not all_sents:
        return {}, {}
    all_scores = _classify_sentences(all_sents, pipe)
    split = len(opening_sents)
    neg_map_opening = {s: (max(sc, key=sc.get) == "negative") for s, sc in zip(opening_sents, all_scores[:split])}
    neg_map_qa      = {s: (max(sc, key=sc.get) == "negative") for s, sc in zip(qa_sents,      all_scores[split:])}
    return neg_map_opening, neg_map_qa


@st.cache_data
def _load_backtest_df() -> Optional[pd.DataFrame]:
    """Load and cache the backtest results CSV; returns None if file does not exist."""
    if BACKTEST_CSV.exists():
        return pd.read_csv(BACKTEST_CSV)
    return None


# ---------------------------------------------------------------------------
# Helpers — corpus listing + source lookup
# ---------------------------------------------------------------------------

def _list_corpus_options() -> list[str]:
    """Return sorted list of 'TICKER  YYYY-MM-DD' labels for score files."""
    files  = sorted(SCORES_DIR.glob("*_score.json"))
    labels = []
    for fp in files:
        parts = fp.stem.replace("_score", "").rsplit("_", 1)
        labels.append(f"{parts[0]}  {parts[1]}" if len(parts) == 2 else fp.stem)
    return labels


def _label_to_score_path(label: str) -> Path:
    """Convert a corpus dropdown label ('AAPL  2024-10-31') to its score JSON path."""
    return SCORES_DIR / f"{label.strip().replace('  ', '_')}_score.json"


def _score_to_parsed_path(score_data: dict) -> Optional[Path]:
    """Return the parsed transcript JSON path for a score dict, or None if not on disk."""
    p = TRANSCRIPTS_DIR / f"{score_data.get('ticker','')}_{score_data.get('date','')}_parsed.json"
    return p if p.exists() else None


def _get_source(ticker: str, call_date: str) -> str:
    """Look up the transcript source ('edgar' or 'motleyfool') from the backtest CSV."""
    df = _load_backtest_df()
    if df is None:
        return "edgar"
    row = df[(df["ticker"] == ticker) & (df["call_date"] == call_date)]
    return str(row["source"].iloc[0]) if not row.empty else "edgar"


# ---------------------------------------------------------------------------
# Panel 1 — Score gauge + SHAP + component bars
# ---------------------------------------------------------------------------

def _risk_color(risk_class: str) -> str:
    """Return a hex color for a risk class string (red/orange/green)."""
    return {"High Risk": RISK_RED, "Medium Risk": RISK_AMBER, "Low Risk": RISK_GREEN}.get(
        risk_class, "#1f77b4"
    )


def render_score_panel(score_data: dict) -> None:
    """Render Tab 1: EW_Risk_Score gauge, component progress bars, and SHAP hedge chart."""
    ew  = score_data["EW_Risk_Score"]
    rc  = score_data["risk_class"]
    hedge_raw = score_data.get("hedging", {})
    sent_raw  = score_data.get("sentiment", {})
    vocab_raw = score_data.get("vocab", {})
    hedge_density = float(hedge_raw.get("hedging_density", 0.0))
    neg_sentiment = float(sent_raw.get("overall_negative_ratio", 0.0))
    backward_ratio = float(vocab_raw.get("backward_ratio", 0.0))
    hedge_norm = min(hedge_density / 3.0, 1.0)
    neg_norm = min(neg_sentiment / 0.20, 1.0)
    back_norm = min(backward_ratio / 1.0, 1.0)

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <div style="font-size:24px;font-weight:700;color:#18212B;">EW Risk Score</div>
          <div style="font-size:24px;font-weight:800;color:#18212B;">{ew:.2f}</div>
          <div style="background:{_risk_color(rc)};color:white;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:700;">
            {rc}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.metric("EW_Risk_Score", f"{ew:.2f}")
        st.progress(min(max(ew / 100.0, 0.0), 1.0))
        st.markdown(
            "<div style='display:flex;justify-content:space-between;font-size:12px;color:#5F7080;'>"
            "<span>0</span><span>35</span><span>65</span><span>100</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Signal Components**")
        st.caption("Hedging density")
        st.progress(float(hedge_norm), text=f"{hedge_density:.3f}")
        st.caption("Negative sentiment")
        st.progress(float(neg_norm), text=f"{neg_sentiment:.3f}")
        st.caption("Backward vocab ratio")
        st.progress(float(back_norm), text=f"{backward_ratio:.3f}")
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hedging density",  f"{hedge_density:.2f}")
        m2.metric("Neg. sentiment",   f"{neg_sentiment:.3f}")
        m3.metric("Backward ratio",   f"{backward_ratio:.3f}")
        m4.metric("Sent. trajectory", f"{sent_raw.get('sentiment_trajectory', 0):+.4f}")
        st.markdown(
            f"<div style='height:8px;border-radius:8px;background:linear-gradient(90deg,{RISK_GREEN} 0%,{RISK_GREEN} 35%,{RISK_AMBER} 35%,{RISK_AMBER} 65%,{RISK_RED} 65%,{RISK_RED} 100%);'></div>",
            unsafe_allow_html=True,
        )

    with col_side:
        risk_short = rc.replace(" Risk", "")
        st.markdown(
            f"""
            <div style="font-size:14px;color:#5F7080;margin-bottom:4px;">Risk Class</div>
            <div style="font-size:34px;font-weight:800;color:{_risk_color(rc)};margin-bottom:12px;">{risk_short}</div>
            """,
            unsafe_allow_html=True,
        )
        top_phrases = hedge_raw.get("top_phrases", [])
        st.markdown("**Top phrases detected:**")
        if isinstance(top_phrases, list) and top_phrases:
            for phrase, count in top_phrases[:5]:
                st.caption(f"- {phrase}: {count}")
        else:
            st.caption("- None detected")

    # ── SHAP-style hedge explanation ──────────────────────────────────────
    st.markdown("---")
    parsed_path = _score_to_parsed_path(score_data)
    if parsed_path is not None and hedge_raw:
        opening_text = json.loads(parsed_path.read_text(encoding="utf-8")).get("opening_remarks", "")
        if opening_text:
            shap_result = explain_hedge_score(hedge_raw, opening_text)
            st.plotly_chart(shap_result["fig"], use_container_width=True)
        else:
            st.caption("No opening remarks text available for hedge explanation.")
    else:
        st.caption("Parsed transcript not found — hedge explanation unavailable.")


# ---------------------------------------------------------------------------
# Panel 2 — Transcript with highlights
# ---------------------------------------------------------------------------

def _highlight_html(text: str, neg_map: dict[str, bool]) -> str:
    """Wrap hedging phrases in amber marks and negative sentences in light-red spans; return HTML."""
    if not text:
        return "<p><em>(No text available)</em></p>"

    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)

    html_parts: list[str] = []
    for sent in sentences:
        highlighted = _HEDGE_PATTERN.sub(
            lambda m: (
                f'<mark style="background:#FFA500;color:#000;'
                f'padding:1px 3px;border-radius:3px;font-weight:600">'
                f'{m.group()}</mark>'
            ),
            sent,
        )
        if neg_map.get(sent, False):
            html_parts.append(
                f'<span style="background:#FFCCCC;display:inline">{highlighted}</span> '
            )
        else:
            html_parts.append(highlighted + " ")

    return f'<div style="line-height:1.75;font-size:14px">{"".join(html_parts)}</div>'


def render_transcript_panel(score_data: dict) -> None:
    """Render Tab 2: highlighted transcript with hedging phrases (amber) and negative sentences (pink)."""
    parsed_path = _score_to_parsed_path(score_data)
    if parsed_path is None:
        st.info("Parsed transcript file not found — re-run the pipeline to generate it.")
        return

    parsed  = json.loads(parsed_path.read_text(encoding="utf-8"))
    opening = parsed.get("opening_remarks", "")
    qa      = parsed.get("qa_section", "")

    lc1, lc2, _ = st.columns([1, 1, 4])
    with lc1:
        st.markdown(
            '<span style="background:#FFA500;padding:3px 8px;border-radius:3px">'
            'Hedging phrase</span>',
            unsafe_allow_html=True,
        )
    with lc2:
        st.markdown(
            '<span style="background:#FFCCCC;padding:3px 8px;border-radius:3px">'
            'Negative sentence</span>',
            unsafe_allow_html=True,
        )

    with st.spinner("Classifying sentences for highlights..."):
        neg_map_opening, neg_map_qa = _classify_both_sections(opening or "", qa or "")

    tab_open, tab_qa = st.tabs(["Opening Remarks", "Q&A Session"])
    with tab_open:
        if opening:
            st.markdown(_highlight_html(opening, neg_map_opening), unsafe_allow_html=True)
        else:
            st.info("Opening remarks section not found.")
    with tab_qa:
        if qa:
            st.markdown(_highlight_html(qa, neg_map_qa), unsafe_allow_html=True)
        else:
            st.info("Q&A section not found (sections_found=False — possible press release format).")


# ---------------------------------------------------------------------------
# Panel 3 — Candlestick
# ---------------------------------------------------------------------------

def render_candlestick_panel(score_data: dict) -> None:
    """Render Tab 3: Plotly candlestick over 20 trading days with earnings date marked."""
    ticker    = score_data["ticker"]
    call_date = score_data["date"]

    df = _fetch_candles_cached(ticker, call_date)
    if df is None or df.empty:
        st.warning(f"Could not fetch price data for {ticker} around {call_date}.")
        return

    event_ts = pd.Timestamp(call_date)

    fig = go.Figure(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=ticker,
        increasing_line_color="#2ca02c",
        decreasing_line_color="#d62728",
    ))

    fig.add_vline(
        x=event_ts.timestamp() * 1000,
        line_width=2,
        line_dash="dash",
        line_color="#ff7f0e",
        annotation_text="Earnings",
        annotation_position="top left",
        annotation_font_size=12,
    )

    fig.update_layout(
        title=f"{ticker} — 20 trading days around earnings  ({call_date})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(t=50, b=30, l=10, r=10),
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Panel 4 — Backtest summary
# ---------------------------------------------------------------------------

_KEY_FINDING = (
    "**Key finding:** 71.4% NEGATIVE signal accuracy on full call transcripts (Motley Fool, n=7) "
    "vs 56.5% on press releases (EDGAR, n=46). "
    "Full transcripts capture Q&A hedging language absent from press releases — "
    "the richer signal source produces statistically stronger risk detection."
)


def render_backtest_panel(current_ticker: str, current_date: str) -> None:
    """Render Tab 4: backtest headline metrics, EDGAR vs MF accuracy split, top-5 calls."""
    if not BACKTEST_CSV.exists():
        st.warning("Backtest results CSV not found. Run `python -m src.backtest.engine` first.")
        return

    df    = pd.read_csv(BACKTEST_CSV)
    stats = compute_stats(df, primary_window="5d")

    # ── Key finding callout ───────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:{INFO_BG};border-left:3px solid {INFO_BORDER};padding:10px 12px;border-radius:8px;margin-bottom:8px;color:{INFO_TEXT};">
          {_KEY_FINDING.replace("**", "")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Headline metrics ──────────────────────────────────────────────────
    acc = stats["overall_accuracy"]
    pv  = stats["p_value"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Directional accuracy (5d)", f"{acc*100:.1f}%" if acc else "N/A",
              help="POSITIVE + NEGATIVE signals vs actual return direction")
    c2.metric("p-value (binomial)", f"{pv:.4f}" if pv else "N/A")
    c3.metric("Signal Sharpe",    str(stats["signal_sharpe"])    if stats["signal_sharpe"]    else "N/A")
    c4.metric("Buy-Hold Sharpe",  str(stats["buy_hold_sharpe"])  if stats["buy_hold_sharpe"]  else "N/A")

    st.markdown("---")

    # ── EDGAR vs Motley Fool breakdown ────────────────────────────────────
    src = stats.get("source_accuracy", {})
    if src:
        st.markdown("**Accuracy by source (directional signals)**")
        h1, h2, h3, h4 = st.columns([2, 2, 2, 2])
        h1.markdown("**Source**")
        h2.markdown("**Accuracy**")
        h3.markdown("**Correct/Total**")
        h4.markdown("**p-value**")
        for s, ss in sorted(src.items()):
            src_label = "Motley Fool" if s == "motleyfool" else "EDGAR"
            accuracy = float(ss["accuracy"]) * 100
            acc_color = RISK_GREEN if accuracy > 60 else (RISK_AMBER if accuracy >= 50 else RISK_RED)
            c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
            c1.write(src_label)
            c2.markdown(
                f"<span style='color:{acc_color};font-weight:700'>{accuracy:.1f}%</span>",
                unsafe_allow_html=True,
            )
            c3.write(f"{ss['n_correct']}/{ss['n']}")
            c4.write(f"{ss['p_value']:.4f}")
        st.caption(
            "p=0.31 is expected at n=106 — Cohen's h=0.076 (negligible effect size). "
            "Min-n for 80% power: 1,357 signals."
        )

    neg = stats.get("negative_signal_accuracy", {})
    if neg:
        st.markdown(
            f"**NEGATIVE signal accuracy (risk-detection):** "
            f"{neg['accuracy']*100:.1f}%  ({neg['n_correct']}/{neg['n']})  "
            f"p={neg['p_value']:.4f}"
        )

    st.markdown("---")

    # ── Top 5 highest-scoring calls ───────────────────────────────────────
    st.markdown("**Top 5 highest-risk calls in corpus**")
    top5 = (
        df.sort_values("ew_risk_score", ascending=False)
        .head(5)[["ticker", "call_date", "ew_risk_score", "signal", "source", "actual_5d"]]
    )
    th1, th2, th3, th4, th5, th6 = st.columns([1.2, 1.5, 2.0, 1.4, 1.5, 1.6])
    th1.markdown("**Ticker**")
    th2.markdown("**Date**")
    th3.markdown("**EW score**")
    th4.markdown("**Signal**")
    th5.markdown("**Source**")
    th6.markdown("**5d return**")

    for _, row in top5.iterrows():
        ticker = str(row["ticker"])
        date_val = str(row["call_date"])
        score_val = float(row["ew_risk_score"])
        signal = str(row["signal"])
        source = "Motley Fool" if str(row["source"]) == "motleyfool" else "EDGAR"
        ret_val = float(row["actual_5d"]) if pd.notna(row["actual_5d"]) else np.nan
        ret_color = RISK_GREEN if (pd.notna(ret_val) and ret_val > 0) else RISK_RED

        r1, r2, r3, r4, r5, r6 = st.columns([1.2, 1.5, 2.0, 1.4, 1.5, 1.6])
        r1.markdown(
            f"**{ticker}**" if (ticker == current_ticker and date_val == current_date) else ticker
        )
        r2.write(date_val)
        with r3:
            st.progress(min(score_val / 100.0, 1.0), text=f"{score_val:.2f}")
        r4.write(signal)
        r5.write(source)
        r6.markdown(
            f"<span style='color:{ret_color};font-weight:700'>{ret_val:+.3f}</span>" if pd.notna(ret_val) else "N/A",
            unsafe_allow_html=True,
        )

    # ── Window accuracy breakdown ─────────────────────────────────────────
    st.markdown("**Accuracy by return window**")
    wrows = []
    for w, ws in stats["window_accuracy"].items():
        a = ws["accuracy"]
        wrows.append({"Window": w, "Accuracy": f"{a*100:.1f}%" if a else "N/A", "n": ws["n"]})
    st.dataframe(pd.DataFrame(wrows), hide_index=True, use_container_width=False)


# ---------------------------------------------------------------------------
# Panel 5 — Corpus scatter
# ---------------------------------------------------------------------------

def render_scatter_panel(current_ticker: str, current_date: str) -> None:
    """Render Tab 5: corpus scatter of EW_Risk_Score vs actual 5d return, coloured by sector."""
    if not BACKTEST_CSV.exists():
        st.warning("Backtest results CSV not found.")
        return

    df = pd.read_csv(BACKTEST_CSV).dropna(subset=["actual_5d"])

    fig = px.scatter(
        df,
        x="ew_risk_score",
        y="actual_5d",
        color="sector",
        symbol="signal",
        color_discrete_map={
            "Consumer": "#1D9E75",
            "Healthcare": "#378ADD",
            "Financials": "#7F77DD",
            "Technology": "#BA7517",
            "Energy": "#D85A30",
        },
        symbol_map={
            "POSITIVE": "triangle-up",
            "NEGATIVE": "triangle-down",
            "NEUTRAL": "circle",
        },
        hover_data={
            "ticker": True, "call_date": True, "source": True,
            "ew_risk_score": ":.2f", "actual_5d": ":.4f",
        },
        labels={
            "ew_risk_score": "EW Risk Score",
            "actual_5d":     "Actual 5-day Return",
            "sector":        "Sector",
            "signal":        "Signal",
        },
        title="Corpus: EW Risk Score vs Actual 5-day Return",
        template="plotly_white",
        opacity=0.75,
        height=480,
    )

    # Highlight currently selected transcript
    cur = df[(df["ticker"] == current_ticker) & (df["call_date"] == current_date)]
    if not cur.empty:
        fig.add_trace(go.Scatter(
            x=cur["ew_risk_score"],
            y=cur["actual_5d"],
            mode="markers",
            marker=dict(size=18, color="gold", line=dict(color="black", width=2)),
            name=f"{current_ticker} (selected)",
            showlegend=True,
        ))

    # Reference lines: y=0 (break-even) and x=50 (mid-score)
    fig.add_hline(y=0,  line_dash="dash", line_color="#555", line_width=1.5,
                  annotation_text="Break-even", annotation_position="bottom right",
                  annotation_font_size=11)
    fig.add_vline(x=50, line_dash="dot",  line_color="#aaa", line_width=1)

    seen_sectors: set[str] = set()
    for trace in fig.data:
        name = str(trace.name)
        sector_name = name.split(",")[0].strip()
        trace.name = sector_name
        trace.legendgroup = sector_name
        if sector_name in seen_sectors:
            trace.showlegend = False
        else:
            trace.showlegend = True
            seen_sectors.add(sector_name)

    fig.update_layout(
        margin=dict(t=50, b=30, l=10, r=10),
        legend_title_text="Sector",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar — MLflow panel
# ---------------------------------------------------------------------------

def _check_mlflow_available() -> bool:
    """Return True if the local MLflow tracking server is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:5000", timeout=1)
        return True
    except Exception:
        return False


def render_mlflow_sidebar() -> None:
    """Render the MLflow recent-runs table and link button in the sidebar."""
    st.markdown("---")
    st.markdown("**Recent analysis runs**")

    if not _check_mlflow_available():
        st.caption("MLflow tracking available in local mode only.")
        return

    runs = get_recent_runs(n=10)
    if runs:
        run_rows = []
        for r in runs:
            ret = r["actual_5d_return"]
            run_rows.append({
                "Ticker":    r["ticker"],
                "Date":      r["filing_date"],
                "EW Score":  r["ew_risk_score"],
                "5d Return": f"{ret:+.3f}" if ret is not None else "—",
            })
        st.dataframe(
            pd.DataFrame(run_rows),
            hide_index=True,
            use_container_width=True,
            height=min(38 * len(run_rows) + 38, 320),
        )
    else:
        st.caption("No runs logged yet. Start MLflow with `mlflow ui` and analyze a ticker.")

    st.link_button("Open MLflow UI", "http://localhost:5000", use_container_width=True)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _inject_layout_styles() -> None:
    """Inject inline CSS for a cleaner two-column dashboard shell."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            width: 220px !important;
            min-width: 220px !important;
            background: #1D232A !important;
        }
        [data-testid="stSidebar"] * {
            color: #E8EDF2 !important;
        }
        [data-testid="stSidebar"] small, [data-testid="stSidebar"] .stCaption {
            color: #9FB0C1 !important;
        }
        [data-testid="stAppViewContainer"] {
            background: #FFFFFF !important;
        }
        [data-testid="stAppViewContainer"] * {
            color: #1F2A35;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 100% !important;
            padding-top: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_topbar(score_data: dict, source_label: str) -> None:
    """Render styled top bar with transcript context and risk summary."""
    ticker = str(score_data.get("ticker", "?"))
    call_date = str(score_data.get("date", "?"))
    company_name = COMPANY_NAMES.get(ticker, ticker)
    ew_score = float(score_data.get("EW_Risk_Score", 0.0))
    risk_class = str(score_data.get("risk_class", "Medium Risk"))
    badge_color = _risk_color(risk_class)
    risk_short = risk_class.replace(" Risk", "").upper()

    left_col, right_col = st.columns([3, 2], vertical_alignment="center")
    with left_col:
        st.markdown(
            f"""
            <div style="padding:10px 12px;border-radius:10px;background:#F4F7FB;border:1px solid #D8E2EC;">
              <div style="font-size:24px;font-weight:700;color:#18212B;">{ticker} · {company_name}</div>
              <div style="font-size:13px;color:#4B5A6A;">Call date: {call_date} · Source: {source_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(
            f"""
            <div style="padding:10px 12px;border-radius:10px;background:#F4F7FB;border:1px solid #D8E2EC;text-align:right;">
              <div style="display:inline-block;background:{badge_color};color:white;font-weight:700;font-size:12px;padding:3px 8px;border-radius:999px;">
                {risk_short}
              </div>
              <div style="font-size:33px;font-weight:800;color:#18212B;line-height:1.2;">{ew_score:.2f}</div>
              <div style="font-size:12px;color:#4B5A6A;">EW_Risk_Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_landing_page() -> None:
    """Render hero landing page when no transcript is selected."""
    st.markdown(
        """
        <div style="padding:4px 2px 8px 2px;">
          <div style="font-size:42px;font-weight:800;color:#18212B;">EarningsEcho</div>
          <div style="font-size:16px;color:#495B6D;">Language-native risk intelligence for post-earnings drift.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Directional accuracy (5d)", "53.8%", "n=106 signals")
    c2.metric("Signal Sharpe", "2.174", "vs B&H -0.896")
    c3.metric("Walk-forward accuracy", "55.5%", "no look-ahead bias")
    c4.metric("Best sector (Consumer)", "64.5%", "n=31 · p≈0.07")

    st.markdown(
        f"""
        <div style="background:{INFO_BG};border-left:3px solid {INFO_BORDER};padding:10px 12px;border-radius:8px;margin:8px 0 14px 0;color:{INFO_TEXT};">
          Full-call transcripts (Motley Fool) achieve 71.4% NEGATIVE signal accuracy vs 56.5% for EDGAR press releases — Q&A hedging language is the key differentiator.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_bt, col_sc = st.columns(2)
    with col_bt:
        st.markdown("#### Backtest overview")
        render_backtest_panel("", "")
    with col_sc:
        st.markdown("#### Corpus scatter")
        render_scatter_panel("", "")


# ---------------------------------------------------------------------------
# Live pipeline helper
# ---------------------------------------------------------------------------

def render_bilingual_explainer(score_data: dict) -> None:
    """Render Tab 6: bilingual (English + Hindi) explanation generated via Claude."""
    ew_risk_score = float(score_data.get("EW_Risk_Score", 0.0))
    risk_class = str(score_data.get("risk_class", "Unknown"))
    hedging_density = float(score_data.get("hedging", {}).get("hedging_density", 0.0))
    neg_sentiment = float(score_data.get("sentiment", {}).get("overall_negative_ratio", 0.0))
    backward_ratio = float(score_data.get("vocab", {}).get("backward_ratio", 0.0))
    ticker = str(score_data.get("ticker", "UNKNOWN"))
    call_date = str(score_data.get("date", "unknown"))

    hedging_data = score_data.get("hedging", {})
    hedge_counts = hedging_data.get("hedge_counts", {})
    if not isinstance(hedge_counts, dict) or not hedge_counts:
        top_phrases_raw = hedging_data.get("top_phrases", [])
        if isinstance(top_phrases_raw, list):
            hedge_counts = {
                str(item[0]): int(item[1])
                for item in top_phrases_raw
                if isinstance(item, (list, tuple)) and len(item) >= 2
            }
        else:
            hedge_counts = {}

    top_hedges = sorted(hedge_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_hedges_text = ", ".join([f"{phrase} ({count})" for phrase, count in top_hedges]) or "None detected"

    st.markdown(
        "Generate a balanced bilingual explanation of the current risk score and its three underlying signals."
    )

    if st.button("Generate Bilingual Summary", type="primary", key="generate_bilingual_summary"):
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            api_key = ""
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            st.error("Missing GROQ_API_KEY. Add it to .streamlit/secrets.toml")
            return

        prompt = (
            "You are explaining an earnings-call NLP risk score.\n\n"
            "Write exactly:\n"
            "1) A 4-6 sentence plain English paragraph explaining the score and signals.\n"
            "2) The same explanation in Hindi (Devanagari script), prefixed with 'Hindi Summary:'.\n\n"
            "Tone: informative, balanced, no buy/sell advice.\n\n"
            f"Data:\n"
            f"- Ticker: {ticker}\n"
            f"- Call Date: {call_date}\n"
            f"- EW_Risk_Score: {ew_risk_score:.2f}\n"
            f"- Risk Class: {risk_class}\n"
            f"- Hedging Density: {hedging_density:.4f}\n"
            f"- Negative Sentiment Ratio: {neg_sentiment:.4f}\n"
            f"- Backward Ratio: {backward_ratio:.4f}\n"
            f"- Top 5 hedging phrases: {top_hedges_text}\n"
        )

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            model_text = result.strip()


            english_text, hindi_text = model_text, ""
            if "Hindi Summary:" in model_text:
                english_text, hindi_text = model_text.split("Hindi Summary:", 1)
                english_text = english_text.strip()
                hindi_text = hindi_text.strip()

            col_en, col_hi = st.columns(2)
            with col_en:
                st.markdown("**English Summary**")
                st.info(english_text if english_text else "No English summary returned.")
            with col_hi:
                st.markdown("**Hindi Summary**")
                st.info(hindi_text if hindi_text else "Hindi Summary marker not found in response.")

        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to generate summary (request error): {exc}")
        except Exception as exc:
            st.error(f"Failed to generate summary: {exc}")


def _run_live_pipeline(ticker: str) -> Optional[dict]:
    """Fetch, parse, score, save, and MLflow-log a new EDGAR transcript for ticker."""
    with st.spinner(f"Fetching EDGAR transcripts for {ticker}..."):
        try:
            raw_list = _fetch_edgar_cached(ticker, n=3)
        except Exception as exc:
            st.error(f"EDGAR fetch failed: {exc}")
            return None

    if not raw_list:
        st.error(f"No transcripts found on EDGAR for {ticker}.")
        return None

    raw = raw_list[0]

    # Save raw JSON
    acc      = raw.get("accession_number", "unknown").replace("-", "")
    raw_path = TRANSCRIPTS_DIR / f"{ticker}_{raw.get('filed_date', 'unknown')}_{acc}.json"
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    # Parse
    with st.spinner("Parsing transcript..."):
        parsed = parse_transcript(raw_path)

    parsed_path = TRANSCRIPTS_DIR / f"{ticker}_{parsed.get('date', 'unknown')}_parsed.json"
    parsed_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    # Score
    score = _score_cached(str(parsed_path))

    # Save score JSON
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    score_path = SCORES_DIR / f"{ticker}_{score.get('date', 'unknown')}_score.json"
    score_path.write_text(
        json.dumps(score, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # Log to MLflow (non-fatal if server not running)
    with st.spinner("Logging run to MLflow..."):
        run_id = log_run(score, score_path=score_path)
        if run_id:
            st.success(f"Pipeline complete — {ticker} {score.get('date')}  |  MLflow run {run_id[:8]}")
        else:
            st.success(f"Pipeline complete — {ticker} {score.get('date')}  (MLflow unavailable)")

    return score


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: build sidebar, resolve selected transcript, render 6-tab dashboard."""
    _inject_layout_styles()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### **EarningsEcho**")
        st.caption("NLP risk signals from earnings calls")
        st.markdown("**From corpus**")

        corpus_options = _list_corpus_options()
        selected_label = st.selectbox(
            "Load transcript",
            options=["— select —"] + corpus_options,
            index=0,
            help="Instantly load any previously scored transcript (264 available)",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Live analysis**")
        new_ticker  = st.text_input("Ticker symbol", max_chars=10, placeholder="e.g. MSFT").strip().upper()
        analyze_btn = st.button("Analyze", type="primary", disabled=not new_ticker)

        st.markdown("---")
        st.caption("264 transcripts · 40 tickers · 5 sectors · 2023–2026")

    # ── Resolve which score_data to display ──────────────────────────────
    score_data: Optional[dict] = None

    if analyze_btn and new_ticker:
        score_data = _run_live_pipeline(new_ticker)

    elif selected_label and selected_label != "— select —":
        score_path = _label_to_score_path(selected_label)
        if score_path.exists():
            score_data = json.loads(score_path.read_text(encoding="utf-8"))
        else:
            st.error(f"Score file not found: {score_path}")

    if score_data is None:
        _render_landing_page()
        return

    ticker    = score_data.get("ticker", "?")
    call_date = score_data.get("date", "?")
    source    = _get_source(ticker, call_date)
    src_label = "Motley Fool" if source == "motleyfool" else "EDGAR"

    _render_topbar(score_data, src_label)

    # ── Six panels as tabs ────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Risk Score",
        "📄 Transcript",
        "📈 Price Chart",
        "🔁 Backtest",
        "🗂 Corpus",
        "🌐 Bilingual",
    ])

    with tabs[0]:
        render_score_panel(score_data)

    with tabs[1]:
        render_transcript_panel(score_data)

    with tabs[2]:
        render_candlestick_panel(score_data)

    with tabs[3]:
        render_backtest_panel(ticker, call_date)

    with tabs[4]:
        render_scatter_panel(ticker, call_date)

    with tabs[5]:
        render_bilingual_explainer(score_data)


if __name__ == "__main__":
    main()
