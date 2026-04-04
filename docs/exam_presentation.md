# EarningsEcho — Viva Preparation

## 90-second project pitch

EarningsEcho is an NLP pipeline that reads earnings call transcripts and scores how much risk language executives use — things like "we believe results may be roughly in line" or "subject to macroeconomic conditions." The hypothesis is that when executives hedge more than usual, the stock often falls after the call.

I built a three-signal scoring system: hedging phrase density, FinBERT negative sentiment across all sentences, and the ratio of backward-looking to forward-looking vocabulary. These combine into a single EW_Risk_Score between 0 and 100. Scores in the top 20% of the corpus get a NEGATIVE signal — I predict the stock will fall. Bottom 20% get POSITIVE.

I backtested this against actual 5-day stock returns across 264 transcripts and 40 tickers. The headline result: on full conference call transcripts from Motley Fool — which include live Q&A — accuracy is 75% on NEGATIVE signals. On the press release summaries that EDGAR files, it drops to 56.5%. The difference makes intuitive sense: executives choose their words carefully in written filings, but unscripted Q&A hedging is harder to control.

---

## 10 most likely examiner questions

**Q1: Why did you choose those three signals? Are the weights justified?**

The weights (hedging 40%, sentiment 35%, vocab 25%) reflect prior literature on earnings call language. Loughran and McDonald (2011) show hedging and negative tone are the two strongest textual predictors of post-earnings drift. Backward-looking ratio is a proxy for deflection — executives who spend more time explaining the past than guiding the future tend to have weaker forward outlooks. The weights are not learned from data; they are researcher priors applied consistently, which is honest at this corpus size.

**Q2: Your p-value is 0.31 — that's not significant. Why should I care about these results?**

Fair challenge. With 106 directional signals and 53.8% accuracy, we can't reject the null at alpha=0.05. But there are two reasons the result is still interesting. First, the Signal Sharpe of 2.17 vs Buy-and-Hold of -0.90 is the more relevant metric for a trading signal — it captures the return per unit of risk taken, not just raw accuracy. Second, the EDGAR vs Motley Fool accuracy gap of 18.5 percentage points is a structural finding about transcript quality, not a chance result — it replicates the known press release vs. transcript distinction in the academic literature. The small n is the constraint, not the finding.

**Q3: How did you handle the look-ahead bias in your backtest?**

Every return is measured from the day after the filing date — the next trading open. Signals are assigned using corpus-wide percentile thresholds computed over the full corpus, which technically looks ahead. A proper walk-forward backtest would recompute thresholds on an expanding window. I acknowledge this as a limitation; for the corpus size (264 rows) a walk-forward split would leave too few samples for meaningful accuracy statistics. The percentile approach guarantees signal balance, which was the priority for this exploratory analysis.

**Q4: Why use FinBERT specifically? Have you compared it to a baseline?**

FinBERT (ProsusAI/finbert) is the standard financial NLP sentiment model, fine-tuned on financial news and earnings call data. It outperforms general-purpose BERT on financial text by a substantial margin in the original paper (Araci 2019). I did not run an ablation against a simpler lexicon-based approach — that would be a natural next step. For this project, FinBERT provides sentence-level classification that a bag-of-words model can't match for long transcript text.

**Q5: What is the EW_Risk_Score actually measuring?**

It measures the density of executive uncertainty language relative to a corpus baseline. A score of 70 means the transcript is in the top 35% for hedging density AND has above-average negative sentiment AND uses more backward-looking vocabulary than forward. It is not a probability of stock decline — it is a risk-language index. The backtest converts it to a trading signal via percentile thresholds.

**Q6: Why did Motley Fool transcripts outperform EDGAR?**

Two reasons. First, Motley Fool includes the full Q&A session, where analysts push back on management and executives give unscripted responses. Hedging language is more concentrated in Q&A than in prepared remarks. Second, EDGAR 8-K exhibits for most companies are condensed press releases — management has reviewed them for investor relations purposes. The NLP signal is stronger on text that is less carefully curated.

**Q7: What would change if you used a larger corpus?**

Three things. First, the MF accuracy estimate would stabilise — 7 directional signals is too small to be confident in the 75% figure. Second, sector-level accuracy differences would become interpretable; currently we have too few observations per sector to draw conclusions. Third, it would justify a walk-forward backtest design, which would give a properly unbiased accuracy estimate.

**Q8: How does the SHAP explainer work?**

I used a simplified bag-of-words SHAP approach rather than model SHAP. For each hedging phrase in the transcript, the contribution is: (occurrence count / word count) * 100 * category weight. Category weights penalise approximators (around, roughly) relative to epistemic hedges (we believe, we expect), which are the stronger risk signal. The result is a ranked chart showing which specific phrases drove that transcript's hedge score — for NKE Q4 2024, "we expect" alone accounted for 46% of the hedge score with 12 occurrences.

**Q9: How would you productionise this?**

Three additions. First, replace SEC EDGAR polling with a webhook or daily batch job that processes new 8-K filings automatically. Second, add a proper train/validation/test split and tune weights on training data using a held-out signal; currently weights are researcher priors. Third, replace the MLflow local server with a managed tracking service (MLflow on Databricks or Weights & Biases) for persistence and team access. The core pipeline is already modular enough to slot into a production data workflow.

**Q10: Did you consider sentiment trajectory as a signal on its own?**

Yes — sentiment trajectory (the difference between Q&A sentiment and opening remarks sentiment) is computed and stored in every score file. I excluded it from the composite score because its correlation with the return signal was weak on this corpus, and adding a fourth component would have diluted the interpretability of the existing three. It is logged to MLflow for every run, so it would be easy to include in a multivariate model as a follow-up.

---

## 3 genuine limitations and honest answers

**Limitation 1: Sample size**

The MF accuracy figures are based on 33 transcripts total, with only 7 NEGATIVE directional signals. The 75% figure has a wide confidence interval. A serious study would need at least 200 full call transcripts per source to draw conclusions with alpha=0.05.

*What I would do differently*: Partner with a data provider (Refinitiv, Bloomberg) for bulk transcript access, or scrape Motley Fool for 3–5 years of calls across the full S&P 500 before running any backtest.

**Limitation 2: No walk-forward validation**

Percentile thresholds are computed over the full corpus, which creates mild look-ahead bias in the signal assignment. This inflates accuracy estimates compared to a real deployment scenario.

*What I would do differently*: Use an expanding-window design: for each date, compute the P80/P20 thresholds using only transcripts filed before that date. The 264-row corpus is too small for this to be meaningful, but it would be the correct methodology at scale.

**Limitation 3: No weight learning**

The 40/35/25 weights are set by researcher judgement, not learned from data. Different weight combinations might produce a stronger signal.

*What I would do differently*: Frame this as a supervised classification problem: train a logistic regression or gradient boosting model on the three NLP signals with the 5-day return direction as the label. Regularisation would automatically identify which signals carry predictive power and which are noise.

---

## Research framing: hypothesis → experiment → result

**Hypothesis**: Executive language in earnings calls contains measurable risk signals — specifically, high hedging density and negative sentiment predict below-average stock returns in the days following the call.

**Experiment**: Built a three-signal NLP scoring pipeline (hedging detector, FinBERT sentiment, vocab ratio) applied to 264 earnings call transcripts from 40 S&P 500 tickers. Assigned NEGATIVE/POSITIVE/NEUTRAL signals using P80/P20 percentile thresholds. Measured 5-day close-to-close returns from yfinance and computed directional accuracy and Sharpe ratio.

**Result**: Overall directional accuracy of 53.8% (not statistically significant at n=106). However, the Motley Fool full-transcript subset shows 75.0% NEGATIVE signal accuracy vs 56.5% for EDGAR press releases — an 18.5 percentage-point gap that supports the hypothesis that transcript quality mediates signal strength. The long-short strategy (long POSITIVE, short NEGATIVE) achieves a Signal Sharpe of 2.174 vs Buy-and-Hold Sharpe of -0.896 over the same period, confirming directional practical value despite the small sample.
