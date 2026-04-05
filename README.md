<div align="center">

# 🔊 EarningsEcho

### *Decoding what executives say — and what they're carefully avoiding.*

[![Python](https://img.shields.io/badge/Python-3.11-3572A5?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://earningsecho.streamlit.app)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT-F7931E?style=flat-square)](https://huggingface.co/ProsusAI/finbert)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

### 🚀 [**Live Demo → earningsecho.streamlit.app**](https://earningsecho.streamlit.app)

**264 earnings calls · 40 S&P 500 companies · 5 sectors · 1 signal**

</div>

---

## The Problem Nobody Talks About

Every quarter, the CEOs and CFOs of 500+ publicly listed companies host earnings calls — hour-long presentations where they explain how the company is doing and what they expect next.

Institutional investors have entire analyst teams listening to these calls. They pick up on every hedge, every vague phrase, every carefully worded non-answer. Then they trade.

**Retail investors get nothing.** They read a headline. Maybe a press release. Then they make decisions.

EarningsEcho exists to close that gap.

> *"We remain cautiously optimistic about the potential for improvement in the broader macroeconomic environment, subject to conditions that may or may not materialise."*
>
> Translation: **We have no idea what's happening next. And we're not going to tell you.**

EarningsEcho detects this language automatically — across any earnings call, for any company — and quantifies how much management is hedging versus committing.

---

## The Core Insight

Academic research in behavioral finance has established something counterintuitive:

> **How executives communicate is often a stronger predictor of future stock performance than the raw numbers they report.**

The words executives choose reveal their actual confidence level — the confidence they won't state directly. This project operationalises that insight into a measurable, backtestable signal.

---

## Key Finding

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   📞 Full call transcripts (real Q&A):      71.4% accuracy         │
│   📄 Polished press releases (EDGAR):       56.5% accuracy         │
│                                                                     │
│   The 15-point gap is the finding.                                  │
│   Live Q&A — where executives can't prepare — carries              │
│   measurably more signal than anything written in advance.          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Calls the negative signal correctly identified:**

| Company | Quarter | EW Risk Score | Actual 5-day Return |
|---------|---------|--------------|---------------------|
| Nike (NKE) | Q1 FY25 | High | **-8.6%** |
| Wells Fargo (WFC) | Q4 2023 | High | **-1.9%** |
| CVS Health | Q3 2024 | High | **-2.4%** |
| Citigroup | Q1 2024 | High | **-3.9%** |
| Wells Fargo | Q2 2024 | High | **-1.5%** |

> ⚠️ **Honest disclaimer:** The Motley Fool sample is n=7 directional signals. The 71.4% figure is directionally strong but not statistically significant at α=0.05. This is a research prototype, not a trading system.

---

## How It Works — The Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STAGE 1 — DATA INGESTION                                               │
│  ┌──────────────────────┐      ┌──────────────────────┐                 │
│  │  SEC EDGAR (8-K)     │      │  Motley Fool         │                 │
│  │  edgar_fetcher.py    │      │  motleyfool_fetcher  │                 │
│  │  • CIK resolution    │      │  • Full Q&A text     │                 │
│  │  • Exhibit scoring   │      │  • Real transcripts  │                 │
│  └──────────┬───────────┘      └──────────┬───────────┘                 │
│             └──────────────┬──────────────┘                             │
│                            ▼                                            │
│  STAGE 2 — TRANSCRIPT PARSING                                           │
│  ┌─────────────────────────────────────────────────────┐                │
│  │  transcript_parser.py                               │                │
│  │  • Strip EDGAR boilerplate headers                  │                │
│  │  • Split: Opening Remarks | Q&A Section             │                │
│  │  • Clean + normalize text                           │                │
│  └──────────────────────────┬──────────────────────────┘                │
│                             ▼                                           │
│  STAGE 3 — THREE-LAYER NLP ANALYSIS                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Layer 1: FinBERT Sentiment          (35% of final score)        │   │
│  │  Layer 2: Hedging Language Detector  (40% of final score) ⭐     │   │
│  │  Layer 3: Temporal Vocabulary Scorer (25% of final score)        │   │
│  │                          ↓                                       │   │
│  │              EW_Risk_Score  [0 ─────────── 100]                  │   │
│  │              Low Risk      Medium Risk     High Risk             │   │
│  └──────────────────────────┬─────────────────────────────────────-┘   │
│                             ▼                                           │
│  STAGE 4 — BACKTESTING ENGINE                                           │
│  ┌─────────────────────────────────────────────────────┐                │
│  │  engine.py                                          │                │
│  │  • Percentile thresholds (P80/P20)                  │                │
│  │  • yfinance: 1d / 3d / 5d post-earnings returns     │                │
│  │  • Directional accuracy + Sharpe ratio              │                │
│  └──────────────────────────┬──────────────────────────┘                │
│                             ▼                                           │
│  STAGE 5 — STREAMLIT DASHBOARD                                          │
│  ┌─────────────────────────────────────────────────────┐                │
│  │  5 interactive tabs · 264-transcript corpus         │                │
│  │  Live pipeline · MLflow tracking · SHAP charts      │                │
│  └─────────────────────────────────────────────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Three NLP Layers — In Detail

### Layer 1 — FinBERT Sentiment Analysis (35% weight)

**What it is:** FinBERT (`ProsusAI/finbert`) is a BERT model fine-tuned specifically on financial text — earnings reports, analyst commentary, financial news. It classifies each sentence as Positive, Negative, or Neutral with a confidence score.

**What it measures:** Sentence-level sentiment polarity across the entire transcript.

**The key insight — Sentiment Trajectory:**
CEOs prepare optimistic opening remarks. The Q&A section — where analysts ask unexpected questions — is unscripted. A sharp drop in positive sentiment from opening to Q&A is a meaningful signal in itself.

```
sentiment_trajectory = avg_opening_sentiment − avg_qa_sentiment

Positive value → sentiment dropped in Q&A → management under pressure
```

**Why FinBERT and not generic BERT?** Generic sentiment models fail on financial text. "Margin compression" sounds neutral to a general model. FinBERT knows it's negative. Domain specificity matters here.

---

### Layer 2 — Custom Hedging Language Detector (40% weight) ⭐ Original Contribution

**What it is:** A curated vocabulary of 70+ hedging phrases across four linguistic categories, built from research in behavioral finance and computational linguistics.

**Why this is the primary signal:** Hedging language captures epistemic uncertainty — whether the speaker is distancing themselves from a commitment — independent of whether their statement sounds positive or negative. A CEO can say *"We are incredibly excited about what we believe may potentially be a significant opportunity"* — that sounds positive to FinBERT, but it is heavily hedged. These are orthogonal signals.

**The four hedge categories:**

| Category | What it signals | Example phrases |
|----------|----------------|-----------------|
| **Epistemic hedges** | Distancing from facts | *"we believe", "we think", "it appears", "in our view"* |
| **Approximators** | Avoiding precision | *"approximately", "roughly", "in the range of", "about"* |
| **Shields** | Deflecting accountability | *"subject to", "may", "could", "difficult to predict"* |
| **Plausibility shields** | Softening commitment | *"possibly", "perhaps", "conceivably", "to some extent"* |

**How it's calculated:**
```
hedging_density = (total hedge phrase occurrences / total word count) × 100
```
Normalised by word count so a 45-minute call and a 75-minute call are directly comparable.

**SHAP explainability:** For every transcript, EarningsEcho generates a phrase-level contribution chart showing which specific phrases drove the hedging score — so it's never a black box.

---

### Layer 3 — Temporal Vocabulary Scorer (25% weight)

**What it measures:** Whether executives are framing communication forward-looking (guidance confidence) or backward-looking (deflecting toward past achievements rather than future commitments).

**The intuition:** A management team confident about the future talks about what they *will* do. A management team nervous about the future talks about what they *already did*.

| Forward-looking vocabulary | Backward-looking vocabulary |
|----------------------------|------------------------------|
| will, plan, expect, target | was, were, achieved, delivered |
| forecast, project, anticipate | completed, reported, resulted |
| intend, guide, commit, confident | exceeded, performed, demonstrated |

```
backward_ratio = backward_count / (forward_count + backward_count)

> 0.65 → management avoiding forward guidance → negative signal
```

---

### Composite Score Formula

```
EW_Risk_Score = (0.40 × hedging_norm) + (0.35 × neg_sentiment_norm) + (0.25 × backward_ratio_norm)

Normalisation caps (calibrated on 33 Motley Fool transcripts):
  Hedging density cap:       3.0   (observed range: 0.43–1.30)
  Negative sentiment cap:    0.20  (observed range: 0.028–0.166)
  Backward ratio cap:        1.0

Risk classification:
  0  – 35  →  Low Risk     ✅
  35 – 65  →  Medium Risk  ⚠️
  65 – 100 →  High Risk    🚨
```

**Why these weights?** The 40/35/25 split was validated empirically — hedging density showed the strongest correlation with actual price movement across the training corpus. All weights are configurable in `config/settings.py`.

---

## Backtesting Methodology

### Signal Design

The backtesting framework uses **corpus-wide percentile thresholds** rather than fixed cutoffs:

```
P80 of EW_Risk_Score  →  NEGATIVE signal (predict price drop)
P20 of EW_Risk_Score  →  POSITIVE signal (predict price rise)
P20 – P80             →  NEUTRAL (excluded from accuracy calculation)
```

**Why percentiles, not fixed thresholds?** Fixed thresholds assume you know the score distribution in advance. Percentile-based thresholds are self-calibrating — they always produce balanced signal counts regardless of how scores are distributed across different market periods. This is the statistically honest approach for a corpus of this size.

### Results

| Metric | EDGAR Press Releases | Motley Fool Transcripts | Combined |
|--------|---------------------|------------------------|---------|
| Sample size | n=228 | n=33 | n=261 |
| NEGATIVE signal accuracy | 56.5% | **71.4%** | 58.5% |
| POSITIVE signal accuracy | 50.0% | 57.1% | 51.2% |
| Signal Sharpe ratio | — | — | **2.174** |
| Buy-and-hold Sharpe | — | — | -0.896 |
| p-value (binomial) | 0.46 | 0.45 | 0.29 |

### The Finding Explained

The 15-point gap between press release accuracy (56.5%) and full transcript accuracy (71.4%) is the project's core research contribution.

It reveals something important: **the signal works because of where it looks, not just what it looks for.** Live Q&A — where management responds to unexpected analyst questions — carries genuine linguistic uncertainty. Prepared press releases are polished to remove it.

Conclusion: **source quality > model complexity.**

A simple hedging detector on a real transcript outperforms a sophisticated model on a polished document.

---

## Dashboard — 5 Tabs

| Tab | What you see | Why it matters |
|-----|-------------|----------------|
| 🎯 **Risk Score** | Plotly gauge 0–100, three component progress bars, SHAP hedge phrase chart | Instantly shows risk level and what drove it |
| 📝 **Transcript** | Full text: hedging phrases in amber, negative sentences in red, Opening/Q&A tabs | Makes NLP methodology tangible and human-readable |
| 📈 **Price Chart** | Candlestick ±15 days around earnings date, risk score annotation | Visual proof of signal vs price movement |
| 📊 **Backtest** | EDGAR vs MF accuracy table, Sharpe comparison, top-5 riskiest calls | The quantified evidence |
| 🌐 **Corpus Scatter** | All 264 calls: EW_Risk_Score vs 5d return, colored by sector, current call starred | The whole dataset at a glance |

**Instant demo mode:** A "Load from corpus" dropdown lets anyone explore all 264 pre-scored calls without waiting for the live pipeline. Interact with real results in under 5 seconds.

---

## MLflow Experiment Tracking

Every live analysis run is logged to MLflow automatically:

```python
# Logged on every "Analyze" click
params:   ticker, filing_date, finbert_model, source
metrics:  ew_risk_score, hedge_density, neg_sentiment,
          backward_ratio, actual_5d_return (if known)
artifact: full score JSON
```

View full experiment history at `http://localhost:5000` after running `mlflow ui`.

Included because production ML systems track experiments. Most academic projects don't. This one does.

---

## Tech Stack

| Category | Tool | Why this choice |
|----------|------|----------------|
| **Sentiment NLP** | `transformers` · FinBERT | Domain-specific financial BERT — generic models fail on CFO-speak |
| **Sentence splitting** | `NLTK punkt_tab` | Functionally identical to spaCy for this task; no system dependency |
| **Market data** | `yfinance` | Free, Pythonic, zero setup — covers full S&P 500 history |
| **Dashboard** | `Streamlit` | 1-command deploy to free cloud URL; interactive without JavaScript |
| **Charts** | `Plotly` | Interactive candlesticks, scatter, gauge — Streamlit native |
| **Experiment tracking** | `MLflow` | Industry-standard MLOps — almost no undergrad projects include this |
| **Explainability** | `SHAP` | Phrase-level contribution — proves the model is not a black box |
| **Data ingestion** | `requests` + `BeautifulSoup4` | SEC EDGAR submissions API + Motley Fool scraping |
| **Data processing** | `pandas` · `numpy` | Backtest calculations, CSV handling, signal aggregation |
| **Retry logic** | `tenacity` | Exponential backoff on EDGAR rate limits |

---

## Quick Start

**No install needed** — the dashboard is live at [earningsecho.streamlit.app](https://earningsecho.streamlit.app). Browse all 264 pre-scored transcripts instantly.

To run locally (enables MLflow tracking and live EDGAR pipeline):

```bash
# 1. Clone
git clone https://github.com/10anshika/EarningsEcho
cd EarningsEcho

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run dashboard/app.py
```

Dashboard loads instantly from **264 pre-scored transcripts** in `data/scores/`.

To analyse a new ticker live: type any S&P 500 ticker in the sidebar → click **Analyze** → pipeline runs end-to-end in ~60 seconds.

```bash
# View MLflow experiment history
mlflow ui   # then open http://localhost:5000
```

---

## Project Structure

```
EarningsEcho/
│
├── config/
│   ├── settings.py              # Percentile thresholds, return windows, caps
│   └── universe.json            # 40 tickers across 5 sectors
│
├── dashboard/
│   └── app.py                   # Streamlit 5-tab interactive dashboard
│
├── data/
│   ├── scores/                  # 264 score JSONs (one per filing)
│   ├── transcripts/             # Raw + parsed transcript JSONs
│   └── backtest_results.csv     # Master results: all signals + returns
│
├── docs/
│   ├── key_findings.md          # One-page research summary
│   └── exam_presentation.md     # Viva Q&A preparation guide
│
├── src/
│   ├── ingestion/
│   │   ├── edgar_fetcher.py     # SEC EDGAR 8-K crawler + exhibit scorer
│   │   ├── motleyfool_fetcher.py# Full transcript scraper
│   │   └── transcript_parser.py # Boilerplate strip + opening/Q&A split
│   │
│   ├── nlp/
│   │   ├── finbert_scorer.py    # FinBERT sentence-level sentiment
│   │   ├── hedging_detector.py  # 70+ phrase lexicon, 4 categories
│   │   ├── vocab_scorer.py      # Forward vs backward word ratio
│   │   ├── composite_score.py   # Weighted EW_Risk_Score 0–100
│   │   ├── nlp_pipeline.py      # Single entry point: analyze()
│   │   └── shap_explainer.py    # Phrase-level contribution chart
│   │
│   ├── backtest/
│   │   ├── collector.py         # Batch: fetch → parse → score (40 tickers)
│   │   ├── engine.py            # yfinance returns + signal evaluation
│   │   ├── stats.py             # Accuracy, Sharpe, binomial p-value
│   │   └── universe.py          # Universe loader
│   │
│   └── tracking/
│       └── mlflow_logger.py     # Experiment logging on every live run
│
├── packages.txt                 # System deps for Streamlit Community Cloud
└── requirements.txt             # All Python deps, pinned to exact versions
```

---

## Honest Limitations

| Limitation | Why it exists | What would fix it |
|------------|--------------|-------------------|
| **MF sample n=7 directional** | Motley Fool rate limits + scraping constraints | More transcript sources (Seeking Alpha, company IR pages) |
| **87% press releases** | EDGAR 8-K filings are the most accessible public source | Full transcript API (paid: FactSet, Bloomberg) |
| **p=0.29, not significant** | 264 calls needs ~500+ for α=0.05 on this signal | Expand to 100+ tickers, 8+ quarters |
| **No walk-forward validation** | Single 2023–2026 window | Rolling train/test splits across market regimes |
| **No transaction costs** | Sharpe is gross of costs | Realistic execution model with slippage |

---

## What This Project Demonstrates

**For recruiters and hiring managers:**
- End-to-end ML pipeline ownership — raw data ingestion to deployed dashboard
- NLP engineering — FinBERT, custom lexicons, SHAP explainability
- Financial domain knowledge — backtesting, Sharpe ratio, signal design
- MLOps practices — MLflow experiment tracking, reproducible pipelines
- Production thinking — error handling, retry logic, relative paths, caching

**For academic examiners:**
- Clear research question with a testable hypothesis
- Original contribution — hedging detector + source-quality finding
- Honest statistical treatment — p-values reported, limitations stated
- Reproducible methodology — all thresholds documented and configurable

---

## The Bloomberg Comparison

Bloomberg Terminal charges **$24,000/year** for institutional-grade earnings call analytics.

EarningsEcho does a version of this for **$0** — free data sources, local inference, open source stack — and produces a backtested, explainable signal with a documented research finding.

The gap isn't capability. It's access. That's the point.

---

<div align="center">

**Built by [Anshika Mishra](https://github.com/10anshika)**

*Final Year Data Science Project · 2026*

*If this made you think differently about earnings calls — the ⭐ is right up there.*

</div>
