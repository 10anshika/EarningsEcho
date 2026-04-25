# README Rewrite Plan — EarningsEcho

## Information Gathered

### Corpus & Data
- 264 earnings transcripts across 40 S&P 500 tickers, 5 sectors (Technology, Financials, Healthcare, Consumer, Energy), 2023–2026
- Sources: SEC EDGAR 8-K press releases (231) + Motley Fool full call transcripts (33)
- Ablation study tested 8 weight configs on n=106 directional samples

### NLP Pipeline (3 Layers)
1. **Hedging Detector**: ~50+ phrases across 4 linguistic categories (epistemic, approximator, shield, plausibility)
2. **FinBERT Sentiment**: `ProsusAI/finbert`, sentence-level classification, sentiment trajectory (Opening vs Q&A)
3. **Temporal Vocab Scorer**: forward-looking vs backward-looking word ratio

### Signal & Weights
- EW_Risk_Score (0–100), P80/P20 percentile thresholds → NEGATIVE/POSITIVE/NEUTRAL
- V1 weights: 40/35/25 (hedging/sentiment/vocab)
- **V2 weights (current default): 33/33/33** — ablation-validated, +2.83pp over V1
- Normalization caps: hedging 3.0, neg_sentiment 0.20, backward_ratio 1.0

### Backtest Results (5-day window, honest)
- Overall directional accuracy: **53.8%** (n=106 directional signals)
- Signal Sharpe: **2.174** vs Buy-and-Hold **-0.896**
- EDGAR NEGATIVE accuracy: **56.5%** (26/46)
- Motley Fool NEGATIVE accuracy: **71.4%** (5/7) — key finding
- p-value (binomial): **0.31** — not significant at α=0.05

### Experiments
- **Ablation**: Equal weights wins (57.55%), hedging-only 51.89%, sentiment-only 49.06%, vocab-only 48.11%
- **ML Comparison**: Rule baseline 54.81% beats LogisticRegression (48.46%), RandomForest (47.58%), GradientBoosting (46.26%) on chronological split
- **Walk-forward**: 55.5% accuracy on 2025+ data, no look-ahead bias
- **Sector Analysis**: Consumer best at 64.5% (n=31), Technology weakest at 41.7% (n=12)
- **Power Analysis**: Cohen's h = 0.076 (negligible), need n≈1,357 for 80% power at 53.8%
- **Confidence Intervals**: Wilson 95% CIs computed for all grouped metrics

### Dashboard (6 Tabs)
1. Risk Score gauge + SHAP hedge chart + component bars
2. Transcript with hedging highlights (amber) + negative sentences (pink)
3. Earnings-day price candlestick (±15 days)
4. Backtest summary (EDGAR vs MF split, top-5 riskiest calls)
5. Corpus scatter (EW_Risk_Score vs 5d return, colored by sector)
6. Bilingual explainer (English + Hindi via Grok API)

### Engineering
- MLflow experiment tracking on every live run
- SHAP-style phrase-level contribution chart
- Modular architecture: `src/ingestion/`, `src/nlp/`, `src/backtest/`, `src/experiments/`, `src/analysis/`, `src/tracking/`
- Config-driven: `config/settings.py` + `config/universe.json`
- Caching in Streamlit (`@st.cache_resource`, `@st.cache_data`)
- Retry logic (`tenacity`) for EDGAR rate limits

---

## Rewrite Plan

### File to Edit
- `README.md` (complete replacement)

### Sections to Include (per task spec)

1. **Hero Section**
   - Headline: "EarningsEcho — NLP Risk Signals from Executive Language"
   - One-liner: "Quantify hedging, sentiment deterioration, and narrative backwardness in earnings calls to forecast post-earnings drift."
   - Badges: Python 3.11, NLP/FinBERT, Streamlit, MLflow, License MIT, Research Project
   - Buttons: Live Demo, Architecture, Results, Dataset

2. **Problem Statement**
   - Why earnings-call language matters (institutional vs retail asymmetry)
   - Inefficiency: linguistic uncertainty is systematically underpriced
   - Non-obvious alpha: the signal is in *how* executives communicate, not just *what* they say

3. **Why This Matters**
   - Novelty: orthogonal signal (hedging ⊥ sentiment)
   - Source-quality insight: live Q&A > polished press releases
   - Explainability by design, not post-hoc

4. **Feature Showcase** (visual cards)
   - NLP Signal Extraction
   - Hedging / Uncertainty Scoring
   - Earnings Risk Signal
   - Explainability (SHAP-style)
   - Backtesting Engine
   - Interactive Dashboard

5. **Architecture** (Mermaid diagrams)
   - High-level flow: Transcript Data → NLP Processing → Feature Engineering → Signal Model → Risk Scoring → Dashboard
   - Data pipeline
   - Modeling pipeline
   - Inference flow

6. **Quantitative Results** (publication-quality)
   - Model performance table
   - Key findings with honest disclaimers
   - Ablation table
   - ML comparison table
   - Walk-forward results
   - Limitations subsection (sample size, statistical power, transaction costs)

7. **Dashboard Showcase**
   - Screenshot placeholders with captions for all 6 tabs

8. **Tech Stack** (recruiter-friendly matrix)
   - NLP, Modeling, Data, App, Experiment Tracking, Explainability layers

9. **Repository Structure**
   - Polished tree with brief explanations

10. **Installation + Quick Start**
    - Clone, install, run locally, launch Streamlit, reproduce experiments

11. **What Makes This Different** (comparison table)
    - Typical Sentiment Project vs EarningsEcho

12. **Engineering Highlights**
    - Reproducibility, experiment tracking, modular design, explainability, robustness, production considerations

13. **Resume-Ready Impact**
    - Bullet points recruiters notice in <20 seconds

14. **Future Roadmap**
    - Checklist format

15. **Footer**
    - Citation block, license, author, contact

### Critical Constraints
- NO fake metrics — all numbers sourced from actual CSVs and code
- NO overclaiming production readiness — explicitly label as research prototype
- Preserve honest disclaimers about p-values, sample size, and statistical power
- Sound like a serious builder, not a student project

### Bonus Assets to Suggest
- Banner mockup concept
- Architecture diagram image
- Dashboard screenshots to capture
- GIF demo ideas
- Additional badges
- GitHub topics/tags

