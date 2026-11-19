# Options Pricing with Neural Networks

A neural network trained to price European call options, compared with the Black–Scholes model. The project runs in two stages. First, the network learns the Black–Scholes formula itself, a controlled test with a known answer. Then it learns the Heston stochastic-volatility model, which produces the volatility skew that Black–Scholes cannot.

**Notebook:** [`options_pricing_neural_networks.ipynb`](options_pricing_neural_networks.ipynb) (executed, with all outputs)

## Objectives

1. Measure how accurately a feed-forward network can approximate an option-pricing function across a wide range of contracts.
2. Quantify how much Black–Scholes misprices when volatility is stochastic, and whether a network trained on a richer model closes that gap.
3. Reconstruct the implied-volatility surface from network prices, and check it for arbitrage.
4. Measure the speed advantage that motivates neural pricers in practice.

## Data

All data are **synthetic**: generated inside the notebook from the Black–Scholes and Heston pricing models with fixed random seeds. No external dataset is downloaded, and no personal or proprietary data are used. QuantLib (open-source, BSD-style licence) is used only to check the Heston pricer.

| Dataset | Size | Inputs (sampled uniformly) |
|---|---|---|
| Black–Scholes | 200,000 contracts | S/K ∈ [0.7, 1.3], T ∈ [0.05, 2] years, r ∈ [0, 6%], σ ∈ [5%, 60%] |
| Heston | 200,000 contracts (18 removed by a bounds check) | S/K, T ∈ [0.1, 2], r, v₀ ∈ [0.01, 0.16], κ ∈ [0.5, 4], θ ∈ [0.01, 0.16], ξ ∈ [0.1, 0.8], ρ ∈ [−0.9, 0] |

Each dataset is split 80/10/10 into training, validation and test sets.

**Heston prices.** The pricer is a vectorised NumPy implementation of the characteristic-function formula in its numerically stable "little trap" form (Albrecher et al., 2007), integrated with 400-point Gauss–Legendre quadrature. Before any data are generated, it is checked against QuantLib's `AnalyticHestonEngine` on 300 random contracts. The maximum absolute difference is **3.5 × 10⁻¹⁰**.

## Method

- **Model:** a 4-layer, 128-unit fully connected network with SiLU activations, built in PyTorch. Prices are expressed per unit strike, since call prices are homogeneous of degree one in (S, K).
- **No-arbitrage lower bound by construction:** the network predicts the option's *time value* through a softplus, and that is added to max(S/K − e^(−rT), 0). Every prediction therefore respects the intrinsic-value bound.
- **Training:** MSE loss, Adam (learning rate 1e-3), learning-rate reduction on plateau, and early stopping on validation loss. The output layer is initialised to a small time value; with default initialisation, training on the Heston data stalled.
- **Benchmarks on Heston data:** Black–Scholes with σ = √v₀, and Black–Scholes with the contract's expected average volatility over its life. The second is the strongest constant-volatility benchmark available without seeing the price.
- **Implied-volatility surface:** network and exact Heston prices on a 31 × 25 strike–maturity grid, inverted to Black–Scholes implied volatility with Brent's method.
- **Arbitrage checks** on the network's price grid: monotonicity in strike, convexity in strike (butterfly spreads), and monotonicity in maturity (calendar spreads).

## Results

Figures are from the committed run of the notebook. Prices are in currency units for a strike of 100.

| Model, test set | RMSE | MAE | Max abs error |
|---|---:|---:|---:|
| Network trained on Black–Scholes | 0.027 | 0.017 | 0.24 |
| Network trained on Heston | **0.025** | **0.017** | **0.22** |
| Black–Scholes on Heston data, σ = √v₀ | 2.53 | 1.76 | 12.88 |
| Black–Scholes on Heston data, expected average vol | 0.66 | 0.44 | 5.78 |

- On Heston data, the network's RMSE is about **27 times lower** than the best constant-volatility Black–Scholes benchmark.
- **Implied-volatility surface:** the network reproduces the Heston skew with a mean absolute error of **22.5 basis points** across the grid. The largest error (484 bp) is at the shortest maturity and the highest strikes. There option prices are tiny, so a small price error becomes a large implied-volatility error.
- **Arbitrage:** there are no violations of monotonicity in strike or maturity. Convexity in strike fails at 12 of 725 grid points (1.7%), because this shape constraint is not enforced during training.
- **Speed:** the network prices an option in about **2.4 µs** in a batch, against roughly 210 µs for vectorised quadrature and 220 µs for QuantLib's analytic engine. That is about 90 times faster. Exact timings depend on the machine.

## Limitations

- All prices are model-generated, so the network learns the Heston model, not the market. Using it on real quotes would need calibrated parameters or training on market data, which brings bid–ask noise and sparse strikes.
- Accuracy is only established inside the sampled parameter ranges. Extrapolation is not tested.
- Convexity is not guaranteed. Shape-constrained architectures or penalty terms (for example Dugas et al., 2009; Ackerer et al., 2020) would address this.
- European calls only, with no dividends.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook options_pricing_neural_networks.ipynb
```

A full run takes about 3 minutes on a 2-core CPU; no GPU is needed.

## Technologies

Python · PyTorch · NumPy · SciPy · pandas · Matplotlib · QuantLib

## References

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*.
- Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility. *Review of Financial Studies*.
- Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). The Little Heston Trap. *Wilmott Magazine*.
- Hutchinson, J., Lo, A. & Poggio, T. (1994). A Nonparametric Approach to Pricing and Hedging Derivative Securities via Learning Networks. *Journal of Finance*.
- Culkin, R. & Das, S. (2017). Machine Learning in Finance: The Case of Deep Learning for Option Pricing. *Journal of Investment Management*.
