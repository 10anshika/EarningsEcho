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

Sidebar also shows last 10 MLflow runs.

Usage
-----
streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import pandas as pd
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EarningsEcho",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    return {"High Risk": "#d62728", "Medium Risk": "#ff7f0e", "Low Risk": "#2ca02c"}.get(
        risk_class, "#1f77b4"
    )


def _gauge_fig(score: float, risk_class: str) -> go.Figure:
    """Build a Plotly indicator gauge for EW_Risk_Score with colour-coded risk zones."""
    color = _risk_color(risk_class)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 38}},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "EW Risk Score", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0, 35],   "color": "#e8f5e9"},
                {"range": [35, 65],  "color": "#fff3e0"},
                {"range": [65, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=30, b=0, l=10, r=10))
    return fig


def render_score_panel(score_data: dict) -> None:
    """Render Tab 1: EW_Risk_Score gauge, component progress bars, and SHAP hedge chart."""
    ew  = score_data["EW_Risk_Score"]
    rc  = score_data["risk_class"]
    hn  = score_data.get("hedging_norm", 0.0)
    sn  = score_data.get("negative_sentiment_norm", 0.0)
    bn  = score_data.get("backward_ratio_norm", 0.0)

    col_gauge, col_detail = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(_gauge_fig(ew, rc), use_container_width=True)
        badge_color = _risk_color(rc)
        st.markdown(
            f'<div style="text-align:center;font-size:20px;font-weight:bold;'
            f'color:white;background:{badge_color};border-radius:8px;padding:6px 0">'
            f'{rc}</div>',
            unsafe_allow_html=True,
        )

    with col_detail:
        st.markdown("**Component Contributions**")
        st.caption("Hedging density  (weight 40%)")
        st.progress(float(hn), text=f"{hn*100:.1f} / 100")
        st.caption("Negative sentiment  (weight 35%)")
        st.progress(float(sn), text=f"{sn*100:.1f} / 100")
        st.caption("Backward-looking ratio  (weight 25%)")
        st.progress(float(bn), text=f"{bn*100:.1f} / 100")

        hedge_raw = score_data.get("hedging", {})
        sent_raw  = score_data.get("sentiment", {})
        vocab_raw = score_data.get("vocab", {})

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hedging density",  f"{hedge_raw.get('hedging_density', 0):.2f}")
        m2.metric("Neg. sentiment",   f"{sent_raw.get('overall_negative_ratio', 0):.3f}")
        m3.metric("Backward ratio",   f"{vocab_raw.get('backward_ratio', 0):.3f}")
        m4.metric("Sent. trajectory", f"{sent_raw.get('sentiment_trajectory', 0):+.4f}")

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
        neg_map_opening = _classify_for_highlight(opening) if opening else {}
        neg_map_qa      = _classify_for_highlight(qa)      if qa      else {}

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
    st.info(_KEY_FINDING)

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
        rows = []
        for s, ss in src.items():
            rows.append({
                "Source":   "Motley Fool" if s == "motleyfool" else "EDGAR",
                "Accuracy": f"{ss['accuracy']*100:.1f}%",
                "Correct":  ss["n_correct"],
                "Total":    ss["n"],
                "p-value":  f"{ss['p_value']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

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
        .rename(columns={
            "ticker":        "Ticker",
            "call_date":     "Date",
            "ew_risk_score": "EW Risk Score",
            "signal":        "Signal",
            "source":        "Source",
            "actual_5d":     "Actual 5d Return",
        })
    )

    def _style_row(row):
        """Highlight the currently selected ticker/date row in amber."""
        if row["Ticker"] == current_ticker and str(row["Date"]) == current_date:
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(top5.style.apply(_style_row, axis=1), use_container_width=True, hide_index=True)

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

    fig.update_layout(margin=dict(t=50, b=30, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar — MLflow panel
# ---------------------------------------------------------------------------

def render_mlflow_sidebar() -> None:
    """Render the MLflow recent-runs table and link button in the sidebar."""
    st.markdown("---")
    st.markdown("**Recent analysis runs**")

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
# Live pipeline helper
# ---------------------------------------------------------------------------

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
    """Entry point: build sidebar, resolve selected transcript, render 5-tab dashboard."""
    # ── Project header ────────────────────────────────────────────────────
    st.markdown(
        "# EarningsEcho\n"
        "**NLP signal from earnings call language** — "
        "hedging density · FinBERT sentiment · forward/backward vocabulary"
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("EarningsEcho")
        st.caption("Select a transcript to analyse")

        corpus_options = _list_corpus_options()
        selected_label = st.selectbox(
            "Load from corpus",
            options=["— select —"] + corpus_options,
            index=0,
            help="Instantly load any previously scored transcript (264 available)",
        )

        st.markdown("---")
        st.markdown("**Analyze a new ticker (live EDGAR fetch)**")
        new_ticker  = st.text_input("Ticker symbol", max_chars=10, placeholder="e.g. MSFT").strip().upper()
        analyze_btn = st.button("Analyze", type="primary", disabled=not new_ticker)

        render_mlflow_sidebar()

        st.markdown("---")
        st.caption("Sources: SEC EDGAR · Motley Fool · yfinance")

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
        st.info(
            "Select a transcript from the sidebar dropdown, "
            "or enter a ticker symbol and click **Analyze**."
        )
        col_bt, col_sc = st.columns(2)
        with col_bt:
            with st.expander("Backtest overview", expanded=True):
                render_backtest_panel("", "")
        with col_sc:
            with st.expander("Corpus scatter", expanded=True):
                render_scatter_panel("", "")
        return

    ticker    = score_data.get("ticker", "?")
    call_date = score_data.get("date", "?")
    source    = _get_source(ticker, call_date)
    src_label = "Motley Fool" if source == "motleyfool" else "EDGAR"

    st.markdown(f"### {ticker}  ·  {call_date}  ·  *{src_label}*")

    # ── Five panels as tabs ───────────────────────────────────────────────
    tabs = st.tabs([
        "Risk Score",
        "Transcript",
        "Price Chart",
        "Backtest",
        "Corpus",
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


if __name__ == "__main__":
    main()
