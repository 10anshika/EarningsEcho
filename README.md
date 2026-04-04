# EarningsEcho
# EarningsEcho

**NLP pipeline that measures executive language patterns in earnings calls and backtests them against post-earnings stock returns.**

---

> **Key finding**
>
> **71.4% NEGATIVE signal accuracy on full call transcripts vs 56.5% on EDGAR press releases** (n=7 vs n=46, 5-day return window).
> Full conference call transcripts — with live Q&A hedging — carry a measurably stronger risk signal than the polished press releases companies file on EDGAR.

---

## Architecture

```
┌─────────────────────┐     ┌────────────────────┐
│  SEC EDGAR (8-K)    │     │  Motley Fool        │
│  edgar_fetcher.py   │     │  motleyfool_fetcher │
└────────┬────────────┘     └────────┬───────────┘
         │  raw JSON                 │  raw JSON
         └──────────┬────────────────┘
                    ▼
         ┌─────────────────────┐
         │  transcript_parser  │
         │  opening / Q&A split│
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────────────────────────┐
         │           nlp_pipeline                  │
         │  FinBERT sentiment  (35% weight)        │
         │  hedging_detector   (40% weight)        │
         │  vocab_scorer       (25% weight)        │
         │           ↓                             │
         │      EW_Risk_Score  0-100               │
         └──────────┬──────────────────────────────┘
                    ▼
         ┌─────────────────────┐
         │  backtest/engine    │
         │  P80 -> NEGATIVE    │
         │  P20 -> POSITIVE    │
         │  yfinance returns   │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────────────┐
         │  Streamlit dashboard        │
         │  + MLflow experiment log    │
         └─────────────────────────────┘
```

## Quick start

```bash
git clone https://github.com/your-username/earningsecho
cd earningsecho
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The dashboard loads instantly from the 264-transcript corpus already scored in `data/scores/`.
To analyze a new ticker live, enter it in the sidebar and click **Analyze**.

---

## Results

| Metric | Value |
|---|---|
| Corpus size | 264 scored transcripts (231 EDGAR, 33 Motley Fool) |
| Tickers covered | 40 (S&P 500 diversified) |
| Primary return window | 5 trading days |
| Overall directional accuracy | 53.8% |
| EDGAR press-release accuracy | 52.9% (54/102 correct) |
| Motley Fool full-transcript accuracy | **75.0%** (3/4 NEGATIVE signals correct) |
| NEGATIVE signal accuracy (all sources) | 58.0% (n=50) |
| Signal Sharpe ratio | **2.174** |
| Buy-and-hold Sharpe (same period) | -0.896 |
| p-value (binomial, overall) | 0.31 (n=106 directional) |

> Signal Sharpe 2.174 vs Buy-and-Hold -0.896: the strategy outperforms passive long-only over this period by going short on high-risk calls.

---

## Pipeline modules

| Module | Location | Description |
|---|---|---|
| EDGAR fetcher | `src/ingestion/edgar_fetcher.py` | Downloads 8-K exhibit text via SEC EDGAR submissions API |
| Motley Fool fetcher | `src/ingestion/motleyfool_fetcher.py` | Scrapes full conference call transcripts |
| Transcript parser | `src/ingestion/transcript_parser.py` | Splits opening remarks from Q&A using regex markers |
| FinBERT scorer | `src/nlp/finbert_scorer.py` | Sentence-level positive/negative/neutral classification |
| Hedging detector | `src/nlp/hedging_detector.py` | 70+ phrase lexicon across 4 hedge categories |
| Vocab scorer | `src/nlp/vocab_scorer.py` | Forward vs backward-looking word ratio |
| Composite score | `src/nlp/composite_score.py` | Weighted combination to EW_Risk_Score 0-100 |
| SHAP explainer | `src/nlp/shap_explainer.py` | Phrase-level hedge contribution chart |
| Backtest engine | `src/backtest/engine.py` | Percentile signals + yfinance returns |
| Backtest stats | `src/backtest/stats.py` | Accuracy, Sharpe, source breakdown |
| MLflow logger | `src/tracking/mlflow_logger.py` | Experiment tracking for live runs |
| Dashboard | `dashboard/app.py` | Streamlit 5-tab UI |

---

## EW_Risk_Score formula

```
EW_Risk_Score = (0.40 x hedging_norm) + (0.35 x neg_sentiment_norm) + (0.25 x backward_ratio_norm)
                x 100
```

Normalisation caps (calibrated on 33 Motley Fool transcripts):

| Signal | Raw range observed | Cap used |
|---|---|---|
| Hedging density | 0.43 - 1.30 per 100 words | 3.0 |
| Negative sentiment ratio | 0.028 - 0.166 | 0.20 |
| Backward ratio | 0.18 - 0.617 | 1.0 |

Signal assignment uses **corpus-wide percentiles** (P80 -> NEGATIVE, P20 -> POSITIVE, middle -> NEUTRAL) to guarantee balanced signal counts regardless of absolute score distribution.

---

## Limitations

- **Small MF sample (n=7 directional)**: The 75% accuracy figure is based on 7 directional signals from Motley Fool transcripts. It is directionally promising but not statistically significant at a=0.05.
- **Press releases vs real calls**: 87% of the corpus (EDGAR) are 8-K press release exhibits, not full transcript text. These lack the live Q&A section where hedging language is densest.
- **No live trading**: Signal Sharpe is computed on historical data with no transaction costs, slippage, or position sizing. This is a research prototype, not a trading system.
- **Single backtest window**: Covers 2023-2026 only; no out-of-sample split or walk-forward validation.
- **Motley Fool coverage**: 10 volatile tickers; generalisation to the full S&P 500 is untested.

---

## Project structure

```
earningsecho/
├── config/
│   └── settings.py          # Percentile thresholds, primary window
├── dashboard/
│   └── app.py               # Streamlit dashboard (5 tabs)
├── data/
│   ├── scores/              # 264 score JSON files
│   ├── transcripts/         # Raw + parsed transcript JSONs
│   └── backtest_results.csv # 264-row results table
├── docs/
│   ├── key_findings.md      # One-page research summary
│   └── exam_presentation.md # Viva preparation guide
├── scripts/
│   └── run_motleyfool_pipeline.py
├── src/
│   ├── backtest/
│   ├── ingestion/
│   ├── nlp/
│   └── tracking/
├── packages.txt             # System dependencies (Streamlit Cloud)
└── requirements.txt
```

