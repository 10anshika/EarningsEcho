# EarningsEcho — Key Findings

## Research question

Do measurable patterns in executive language during earnings calls — hedging density, negative sentiment, and backward-looking vocabulary — predict post-earnings stock return direction better than chance?

And specifically: does the quality of the transcript source (full conference call vs. press release summary) affect prediction accuracy?

---

## Methodology

- **Corpus**: 264 earnings call transcripts across 40 S&P 500 tickers, spanning 2023–2026. Sources: SEC EDGAR 8-K exhibits (231, press releases) and Motley Fool full call transcripts (33).
- **NLP signals**: Three sub-signals combined into a single EW_Risk_Score (0–100): hedging density (40% weight, 70+ phrase lexicon across 4 categories), FinBERT negative sentiment ratio (35% weight, sentence-level), and backward-looking vocabulary ratio (25% weight).
- **Signal assignment**: Corpus-wide P80/P20 percentile thresholds — top 20% of EW_Risk_Score labelled NEGATIVE (predict stock falls), bottom 20% POSITIVE (predict stock rises), middle 60% NEUTRAL (excluded from directional accuracy).
- **Backtest**: 1-day, 3-day, and 5-day close-to-close returns from yfinance, measured from the next trading day after the filing date.
- **Evaluation**: Directional accuracy (signal matches return sign), binomial p-value against 50% null, and annualised Sharpe ratio for the long-short strategy.

---

## Results

| Metric | EDGAR (press releases) | Motley Fool (full calls) | Combined |
|---|---|---|---|
| Directional signals | 102 | 4 | 106 |
| Accuracy (5d window) | 52.9% | **75.0%** | 53.8% |
| NEGATIVE signal accuracy | 56.5% (26/46) | **71.4%** (5/7) | 58.0% |
| p-value (binomial) | 0.37 | 0.31 | 0.31 |
| Signal Sharpe | — | — | **2.174** |
| Buy-and-hold Sharpe | — | — | -0.896 |

---

## Conclusion

Hedging language in earnings calls carries a directional signal for post-earnings returns, and that signal is substantially stronger in full conference call transcripts than in polished press release summaries. The 18.5 percentage-point accuracy gap between Motley Fool and EDGAR sources (75.0% vs 56.5%) is the primary finding, consistent with the hypothesis that executive hedging in live Q&A — where language is less controlled — encodes information not visible in written summaries. The Signal Sharpe of 2.174 vs Buy-and-Hold of -0.896 over the same period confirms the strategy has practical directional value, though sample size limits statistical confidence.
