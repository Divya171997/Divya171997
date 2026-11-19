# Earnings Call Sentiment Analyzer

Measures the tone of company earnings calls with **FinBERT**, a finance-specific transformer, and tests whether that tone relates to, or predicts, the stock's market-adjusted price reaction. Methods are an event study, regressions, out-of-sample classification and a trading strategy.

**Notebook:** [`earnings_call_sentiment_analyzer.ipynb`](earnings_call_sentiment_analyzer.ipynb) (executed, with all outputs)

## Objectives

1. Score every sentence of 1,068 earnings calls as positive, negative or neutral with FinBERT, and aggregate the scores to a call-level tone.
2. Compare FinBERT with the Loughran–McDonald finance dictionary.
3. Test whether tone is associated with the announcement reaction (days 0 to +1) or with post-call drift (days +2 to +20).
4. Test whether tone predicts the direction of the reaction on a later period, and whether a long–short strategy built on it earns abnormal returns after costs.

## Data, models and licences

| Resource | Use | Licence |
|---|---|---|
| [MAEC](https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction) (Li et al., CIKM 2020): S&P 1500 earnings-call transcripts, 2015–2018, one sentence per line | Transcripts | CC BY-SA 4.0 |
| [S&P 500 stock data](https://www.kaggle.com/datasets/camnugent/sandp500) (Nugent, 2018), via the unchanged copy in [plotly/datasets](https://github.com/plotly/datasets) | Daily prices; equal-weighted market index | CC0 (data); MIT (mirror repository) |
| [FinBERT](https://huggingface.co/ProsusAI/finbert) (Araci, 2019; `ProsusAI/finbert`) | Sentence sentiment | CC BY-NC-SA 4.0: **non-commercial use only** |
| [Loughran–McDonald word lists](https://sraf.nd.edu/loughranmcdonald-master-dictionary/), via the `pysentiment2` package | Dictionary baseline | Free for academic research; commercial use requires a licence |

**Compliance.** This is a non-commercial research project. Transcripts, model weights and per-call scores are downloaded or computed at run time and kept only in the git-ignored `data/` folder; nothing derived from the transcripts or the model is redistributed here beyond the aggregate statistics and charts in the notebook. The data are public corporate disclosures and market prices: no personal or confidential information is used.

## Sample

- 3,443 MAEC calls → **1,068 calls from 408 companies** (February 2015 – January 2018). A call is kept when the company is in the S&P 500 price data, the full price window is available, and the transcript has at least 20 usable sentences. 83 transcripts were excluded as empty or truncated.
- **116,031 sentences** scored by FinBERT: 60% neutral, 33% positive, 8% negative.

## Method

| Step | Detail |
|---|---|
| Call tone | Mean over sentences of P(positive) − P(negative), plus shares of positive and negative sentences |
| Dictionary tone | (positive − negative) / (positive + negative) Loughran–McDonald word counts |
| Abnormal returns | Market model estimated on days −250 to −30, using an equal-weighted index of all 505 stocks as the market. Day 0 is the first trading day on or after the call. |
| Event windows | CAR(0,+1): announcement reaction. Day +1 is included because call timing (before the open or after the close) is not recorded. CAR(+2,+20): drift. |
| Regressions | CAR on standardised tone, controlling for the pre-call CAR(−20,−1), with year fixed effects and firm-clustered standard errors |
| Prediction | Logistic regression for the sign of CAR(0,+1); trained on 2015–2016 calls, tested on 2017–2018 calls |
| Strategy | After the close on day +1, long the top and short the bottom tone tercile (cut-offs from training data), held to day +20; 20 bp round-trip cost |

## Results

All figures are from the committed run.

**FinBERT and the dictionary agree only moderately** (correlation 0.54), so the transformer captures something the word lists do not.

**Tone is not significantly related to abnormal returns in this sample.**

| Tone tercile | Calls | CAR(0,+1), 95% CI | CAR(+2,+20), 95% CI |
|---|---:|---|---|
| Negative | 356 | +0.04% [−0.54, +0.62] | +0.06% [−0.50, +0.62] |
| Neutral | 356 | +0.06% [−0.49, +0.60] | +0.17% [−0.43, +0.77] |
| Positive | 356 | +0.15% [−0.39, +0.70] | +0.24% [−0.33, +0.81] |

- The positive-minus-negative gap is +0.12 pp at announcement (p = 0.78) and +0.17 pp over the drift window (p = 0.67).
- **Regressions:** one standard deviation of FinBERT tone is associated with +0.21 pp CAR(0,+1), with a firm-clustered standard error of 0.16 (p = 0.19). With both tone measures in the model, FinBERT's coefficient rises to +0.38 pp (p < 0.10), while the dictionary's is negative and insignificant. Tone has no relationship with drift (−0.05 pp, p = 0.76). The pre-call return does predict drift (+0.16 per 1 pp, p < 0.01), a short-term continuation effect unrelated to tone.

**Out-of-sample prediction** of the sign of CAR(0,+1), with 397 test calls of which 51.1% are positive:

| Features | Accuracy | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|
| Majority class | 0.511 | 0.500 | 0.500 |
| Loughran–McDonald | 0.511 | 0.500 | 0.480 |
| FinBERT | 0.506 | 0.497 | **0.532** (95% CI 0.470–0.590) |
| FinBERT + LM | 0.516 | 0.508 | 0.521 |

FinBERT ranks reactions slightly better than chance, but the bootstrap interval includes 0.5.

**Drift strategy (test period):** 317 trades, mean net abnormal return **+0.00%** per trade (t = 0.00). The strategy was long-heavy (255 longs vs 62 shorts) because 2017 calls were more positive than the training-period cut-offs.

### Interpretation

Within this sample of large US companies, the tone of an earnings call, measured by FinBERT or by a finance dictionary, carries little information about the market-adjusted price reaction that was not already priced. That is consistent with efficient pricing of public disclosures in liquid large caps. It is also a reminder that **text sentiment alone is a weak signal without controlling for the earnings news itself**. FinBERT's small edge over the dictionary in the joint regression and in ROC-AUC suggests that contextual models capture more than word counts, but the effect is not statistically reliable here.

## Limitations and next steps

- **No earnings-surprise control.** Adding analyst-forecast surprises is the most important extension, so that the effect of management's tone can be separated from the earnings news.
- Call timing is not recorded, so the announcement window is approximate.
- The sample is limited to S&P 500 companies (about a third of MAEC) and three years.
- FinBERT was fine-tuned on news sentences, not transcripts. Scoring prepared remarks and analyst Q&A separately, or fine-tuning on transcript data, could sharpen the measure.
- Prices exclude dividends, and the market factor is equal-weighted rather than value-weighted.
- `pysentiment2` matches stemmed word lists, so dictionary counts per call are modest.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook earnings_call_sentiment_analyzer.ipynb
```

The first run downloads the MAEC transcripts with a sparse git checkout (`git` must be installed) and the FinBERT weights from Hugging Face, then scores all sentences. That takes about 90 minutes on a 2-core CPU, or a few minutes on a GPU. Progress is checkpointed every 100 calls, and later runs reuse the scores cached in `data/`.

## Technologies

Python · Hugging Face Transformers · PyTorch · FinBERT · statsmodels · scikit-learn · pandas · NumPy · SciPy · Matplotlib

## References

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*.
- Li, J., Yang, L., Smyth, B. & Dong, R. (2020). MAEC: A Multimodal Aligned Earnings Conference Call Dataset for Financial Risk Prediction. *CIKM 2020*.
- Loughran, T. & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*.
- MacKinlay, A. C. (1997). Event Studies in Economics and Finance. *Journal of Economic Literature*.
- Malo, P. et al. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. *JASIST*.
