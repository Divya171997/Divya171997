# Portfolio Optimisation with Reinforcement Learning

A reinforcement-learning agent (PPO) that learns weekly allocations across 10 large US stocks and cash, rewarded by the differential Sharpe ratio. It is tested on an untouched final year against classical benchmarks, and results are reported for all five training seeds.

**Notebook:** [`portfolio_optimization_reinforcement_learning.ipynb`](portfolio_optimization_reinforcement_learning.ipynb) (executed, with all outputs)

## Objectives

1. Build a trading environment from historical prices with realistic frictions: transaction costs, long/short positions and cash.
2. Train an RL agent to maximise risk-adjusted rather than raw return.
3. Evaluate it out of sample against benchmarks that are hard to beat, and report the variation across random seeds instead of a single best run.

## Data and licences

| Source | Use | Licence |
|---|---|---|
| [S&P 500 stock data](https://www.kaggle.com/datasets/camnugent/sandp500) (Nugent, 2018): daily prices of the 505 S&P 500 constituents, 8 Feb 2013 – 7 Feb 2018 | Stock prices, and an equal-weighted market index built from all 505 stocks | CC0: Public Domain |
| Same file via [plotly/datasets](https://github.com/plotly/datasets) (unchanged copy, downloaded by the notebook) | Download mirror, since Kaggle requires a login | Repository: MIT |

The data contain only public market prices: no personal or confidential information. Prices are split-adjusted but exclude dividends.

**Universe without look-ahead.** The 10 stocks with the highest average daily dollar volume in the first 60 trading days (Feb–May 2013): AAPL, GOOGL, BAC, MSFT, C, XOM, JPM, FB, INTC, T. The rule uses only information available at the start of the sample.

**Chronological split.** Training: May 2013 – Jun 2016 (795 trading days). Validation: Jul – Dec 2016, used only to choose each seed's checkpoint. Test: Jan 2017 – Feb 2018 (276 trading days), evaluated once.

## Method

| Component | Choice |
|---|---|
| State | For each stock: last day's return, 20- and 60-day momentum, 20-day volatility, distance from the 50-day moving average (standardised with training-period statistics), plus current weights |
| Action | Weekly tilt of up to ±15 percentage points per stock around equal weight. Weights can turn negative (short), gross exposure is capped at 100%, and the remainder is cash. |
| Execution | Rebalance every 5 trading days, moving halfway toward the target, at a cost of 10 bp per unit of turnover |
| Reward | Differential Sharpe ratio (Moody & Saffell, 2001) on weekly returns net of costs |
| Agent | PPO (Stable-Baselines3), 2 × 128 MLP, 100,000 steps on random one-year episodes; 5 seeds |
| Model selection | Validation Sharpe evaluated every 5,000 steps; best checkpoint kept for each seed |
| Benchmarks | Buy-and-hold (equal start); daily-rebalanced 1/N; long-only minimum variance (252-day window, monthly); equal-weighted market index of all 505 stocks |

## Results

Test period: 3 January 2017 – 6 February 2018. Figures are from the committed run.

| Strategy | Total return | Annual volatility | Sharpe | Max drawdown | Avg daily turnover |
|---|---:|---:|---:|---:|---:|
| PPO seed 0 | 19.8% | 11.1% | 1.54 | −7.2% | 3.2% |
| PPO seed 1 | 24.1% | 11.3% | 1.80 | −7.5% | 1.1% |
| PPO seed 2 | 23.4% | 11.1% | 1.78 | −7.1% | 2.1% |
| PPO seed 3 | 24.2% | 11.5% | 1.78 | −7.4% | 1.9% |
| PPO seed 4 | 22.9% | 11.2% | 1.75 | −7.2% | 1.9% |
| **PPO, mean of 5 seeds** | **22.9%** | **11.2%** | **1.73** | **−7.3%** | **2.0%** |
| Buy-and-hold (equal start) | 25.0% | 11.6% | 1.81 | −7.6% | – |
| Equal weight 1/N (daily) | 24.0% | 11.4% | 1.78 | −7.6% | – |
| Minimum variance (monthly) | 24.8% | 11.4% | 1.83 | −7.6% | – |
| Market index, equal-weighted (505 stocks) | 15.6% | 8.5% | 1.60 | −7.6% | – |

**Interpretation**

- The agents learned a sensible, low-turnover policy close to the equal-weight portfolio: almost fully invested and long-only, with occasional moves into cash. **They did not outperform the simple benchmarks.** The mean Sharpe ratio of 1.73 is slightly below 1/N (1.78), buy-and-hold (1.81) and minimum variance (1.83). Three of five seeds beat 1/N on Sharpe ratio, by margins of 0.02 or less.
- All five seeds beat the broad market index on total return, but so does every equal-weight portfolio of these 10 stocks. That reflects the stocks in the universe, not the agent.
- During training, validation Sharpe peaked early (5,000–30,000 steps) and then drifted down, a sign of overfitting to the roughly 160 weekly decisions available for training.

This matches the wider evidence that 1/N is very hard to beat out of sample (DeMiguel, Garlappi & Uppal, 2009), especially with a short training history.

## Development history (disclosed for transparency)

The final design is the third version. Earlier versions were changed for the reasons below.

1. **Daily, unconstrained weights.** Agents turned over about 93% of the portfolio per day, and transaction costs destroyed performance: mean test Sharpe −1.49 across five seeds. *These test results were seen before the redesign, which is why this history is reported.*
2. **Weekly, partial rebalancing with unconstrained weights.** Costs fell, but on the **validation** period agents held net-short positions in a rising market (validation Sharpe about −2).
3. **Weekly tilts around equal weight, with a smaller feature set (final).** This was checked on validation only (Sharpe about 3.5 vs 3.5 for 1/N), then evaluated once on the test period.

Because the test period informed the decision to redesign after version 1, the final test results are not a fully untouched hold-out. A stricter evaluation would use walk-forward retraining on additional years.

## Limitations

- Short sample: about 3 years of training data and one test year in a single, calm bull market that ends with the February 2018 volatility spike.
- Prices exclude dividends; borrow costs, short availability and market impact are ignored beyond a flat 10 bp cost.
- PPO hyperparameters were set once and not tuned.
- A fixed 10-stock universe is not representative of the whole market.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook portfolio_optimization_reinforcement_learning.ipynb
```

A full run (5 seeds) takes about 5 minutes on a 2-core CPU.

## Technologies

Python · Stable-Baselines3 (PPO) · Gymnasium · PyTorch · pandas · NumPy · SciPy · Matplotlib

## References

- Moody, J. & Saffell, M. (2001). Learning to Trade via Direct Reinforcement. *IEEE Transactions on Neural Networks*.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy? *Review of Financial Studies*.
- Nugent, C. (2018). S&P 500 stock data. Kaggle. CC0.
