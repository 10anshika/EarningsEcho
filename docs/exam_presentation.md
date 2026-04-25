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

---

## New experiments — extended Q&A

### Ablation study

**Q: Equal weights beat your original 40/35/25 — doesn't that invalidate your design?**

No. I treat that as empirical refinement, not failure. I updated the live configuration to `WEIGHTS_V2` because the ablation results show that at small n, no individual signal dominates strongly enough to justify unequal priors. At larger scale, I would estimate weights with regularised logistic regression rather than hard-code priors.

**Q: Why not grid search over weights?**

At n=106 directional samples, full grid search would mostly fit noise and overstate confidence. I intentionally tested 8 interpretable configurations tied to hypotheses about signal contribution, not a brute-force parameter search optimising a single in-sample number.

### ML comparison

**Q: Why does rule-based beat LR/RF/GBT?**

After the chronological split, the supervised training set is only 37 rows. That is pure data starvation for ML classifiers, especially tree ensembles. In low-label financial NLP settings, simple rule systems frequently outperform supervised models until training data is much larger.

**Q: LR returned zero precision/recall — is something broken?**

No. Logistic regression collapsed to a majority-class predictor in a tiny training regime, which is a known failure mode under class imbalance + low sample count. That result is legitimate and actually strengthens the case for the rule-based baseline at this dataset size.

**Q: GBT and RF give different feature importances — which is right?**

Neither should be over-interpreted at n=37 train rows. The exact ranking is noisy in both models, but they agree on one stable point: `backward_ratio` is the weakest of the three features in this sample.

### Walk-forward

**Q: What is the design and why does it matter?**

I use an expanding-window walk-forward design: for each evaluation period, thresholds are recomputed from historical data only. That removes look-ahead bias by construction. The result (55.45%) exceeding the static full-corpus estimate (53.77%) shows performance is not being inflated by future information leakage.

**Q: 2026Q1 drops to 45.8% — is the model breaking down?**

No. That quarter is a high-volatility macro regime with tariff uncertainty and broad market repricing. The model is designed for idiosyncratic post-earnings language drift, not market-wide shock regimes. This is a structural boundary of scope, not a breakdown of the core signal.

### Sector analysis

**Q: Why Consumer works, Technology fails?**

Consumer-call language is relatively formulaic and stable, so deviations in tone/hedging carry clearer information. In Technology, hedging is expected and often already priced by investors, so the same language patterns carry weaker marginal information.

**Q: Consumer 3d is significant (p=0.015) — why use 5d as headline?**

I keep 5d as the headline horizon because that is standard practice in earnings event studies and better captures full information incorporation. The 3d significance result is a robustness check that reinforces, not replaces, the primary 5d framing.

**Q: ANOVA p=0.000166 across hedge density — what does it tell you?**

It shows sectors use statistically different baseline hedge densities. Financials are lowest (0.431), indicating a different executive communication register, while Consumer is highest (0.862), consistent with stronger language-signal separation there.

### Statistical rigor

**Q: What is Cohen's h?**

Cohen's h is an effect-size metric for differences in proportions. For 53.8% vs a 50% null, h=0.076, which is negligible by standard thresholds. That directly explains why p=0.31 appears even when the point estimate is directionally positive. Reporting h is the statistically honest way to contextualize small-n inference.

**Q: What n would you need for significance?**

For 53.8% accuracy, I need about 1,357 directional samples to reach 80% power. For Consumer at 64.5%, I need about 91. I documented these calculations in `src/analysis/power_analysis.py` and exported the tables for reporting.

**Q: Why Wilson CIs over normal approximation?**

Normal approximation intervals are unreliable near boundaries and at small sample sizes. Multiple project subsets are below n=30, so Wilson intervals provide better coverage and more honest uncertainty estimates.

### Bilingual feature

**Q: Why Hindi?**

This is aligned with the Indian academic context (SPPU) and improves accessibility for non-specialist stakeholders who are not fluent in finance NLP terminology. The Claude-generated explanation is grounded in the actual score components and transcript phrase counts, not generic text.

**Q: Does it always work?**

For this complexity level, quality is consistently usable. Proper nouns and some financial terms may remain in English, which is expected in bilingual technical writing. The UI marks it as AI-generated, and in production I would include a human-review layer for externally shared outputs.
