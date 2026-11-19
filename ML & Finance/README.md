# ML & Finance

Four machine-learning projects in quantitative finance. Each folder holds a fully executed Jupyter notebook, a README describing objectives, data, method and results, and a `requirements.txt`.

| Project | Question | Methods | Data |
|---|---|---|---|
| [Earnings Call Sentiment Analyzer](Earnings-Call-Sentiment-Analyzer/) | Does the tone of an earnings call relate to, and predict, the stock's price reaction? | FinBERT sentence-level sentiment, Loughran–McDonald dictionary, market-model event study, clustered regressions, out-of-sample classification, long–short backtest | MAEC transcripts (CC BY-SA 4.0); S&P 500 prices (CC0) |
| [Options Pricing with Neural Networks](Options-Pricing-Neural-Networks/) | Can a neural network learn option prices, and how does it compare with Black–Scholes when volatility is stochastic? | PyTorch MLP with a no-arbitrage output layer, Heston model (checked against QuantLib), implied-volatility surfaces, arbitrage checks | Synthetic, generated in the notebook |
| [Portfolio Optimisation with Reinforcement Learning](Portfolio-Optimization-Reinforcement-Learning/) | Can an RL agent learn a risk-adjusted allocation that beats simple benchmarks out of sample? | PPO (Stable-Baselines3), custom Gymnasium environment, differential Sharpe ratio reward, 5-seed evaluation vs 1/N and minimum variance | S&P 500 prices (CC0) |
| [Fraud Detection with Anomaly Detection](Fraud-Detection-Anomaly-Detection/) | How well do unsupervised detectors find card fraud without learning from fraud labels? | Isolation Forest, PyTorch autoencoder, ensemble, supervised reference, time-ordered evaluation, t-SNE | ULB credit-card fraud dataset (ODbL / DbCL) |

## Principles followed

- **Honest evaluation:** chronological train/validation/test splits; thresholds and checkpoints chosen on validation data; results reported for every random seed; negative or modest findings stated as they are.
- **Open, licensed data only:** every dataset and model is public and its licence is stated in the project README. No personal, confidential or proprietary data are used. Downloaded data and model weights are not redistributed in this repository.
- **Reproducibility:** fixed seeds, pinned minimum library versions, and notebooks that download their own data.

All four projects are non-commercial research work. Some components (FinBERT weights, the Loughran–McDonald word lists) are licensed for non-commercial or academic use only.
