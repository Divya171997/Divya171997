# Fraud Detection with Anomaly Detection

Detecting fraudulent card transactions as anomalies: transactions that look unlike normal spending, found without relying on fraud labels to train the detector. The project compares an Isolation Forest, an autoencoder and their combination, evaluated on a later time period than the training data. A supervised model is reported as a reference point.

**Notebook:** [`fraud_detection_anomaly_detection.ipynb`](fraud_detection_anomaly_detection.ipynb) (executed, with all outputs)

## Objectives

1. Build unsupervised and deep-learning detectors that flag unusual transactions without learning from fraud labels.
2. Evaluate them the way they would be used: a fixed alert threshold set on past data and applied to new transactions.
3. Measure precision, recall, F1 and precision–recall AUC on a highly imbalanced problem, and compare against what labelled data makes possible.

## Dataset

**Credit Card Fraud Detection**, ULB Machine Learning Group and Worldline (Dal Pozzolo et al., 2015), published on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and [OpenML (ID 1597)](https://www.openml.org/d/1597). The database is released under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/), and its contents under the [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/). Attribution: Machine Learning Group, Université Libre de Bruxelles, and Worldline.

- 284,807 transactions by European cardholders over two days in September 2013. 492 of them (0.173%) are fraud.
- `V1`–`V28` are PCA components released in place of the confidential raw features. `Time` (seconds since the first transaction) and `Amount` are untransformed.
- **Privacy:** the data contain no personally identifiable information. Cardholder and transaction details were transformed by PCA before release, and only `Time`, `Amount` and the label are in original form.
- Kaggle requires a login, so the notebook downloads an unchanged copy from a public GitHub mirror ([stat432/credit-analysis](https://github.com/stat432/credit-analysis)). It verifies the file's SHA-256 checksum, the row count and the fraud count before use. The data are not stored in this repository (`data/` is git-ignored).

## Method

| Step | Detail |
|---|---|
| Features | `V1`–`V28`, log(1 + amount), and hour of day encoded as sine and cosine. Standardised using training-period statistics only. |
| Split | **Time-ordered**: first 60% of transactions for training, next 20% for validation (threshold selection), last 20% for testing. No random shuffling across time. |
| Isolation Forest | 400 trees, fitted on all training transactions without labels. Score = negative mean path length. |
| Autoencoder | PyTorch, 31 → 20 → 10 → 5 → 10 → 20 → 31 with ELU activations, trained only on normal training transactions (novelty detection). Score = mean squared reconstruction error. |
| Ensemble | Average of the two models' percentile ranks. |
| Supervised reference | Gradient-boosted trees (`HistGradientBoostingClassifier`, balanced class weights) trained with labels. Shown to put the unsupervised results in context; it is not an anomaly detector. |
| Threshold | Chosen to maximise F1 on the validation period, then applied unchanged to the test period. |

## Results

Test period: the final 20% of transactions (hours 40.3–48.0), containing 56,962 transactions and 75 frauds. Figures are from the committed run.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Frauds caught | False alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.063 | 0.954 | 0.092 | 0.533 | 0.157 | 40 / 75 | 394 |
| Autoencoder | **0.175** | 0.956 | 0.306 | 0.293 | 0.299 | 22 / 75 | 50 |
| Ensemble (IF + AE) | 0.133 | 0.960 | 0.207 | 0.560 | **0.302** | 42 / 75 | 161 |
| *Supervised GBM (reference)* | *0.748* | *0.957* | *0.879* | *0.680* | *0.767* | *51 / 75* | *7* |

All four models have similar ROC-AUC (about 0.95–0.96) yet very different PR-AUC. With so few positives, ROC-AUC barely separates useful detectors from weak ones, which is why PR-AUC and alert-level metrics are the main measures here.

**Interpretation**

- Every detector ranks fraud far above chance. The fraud rate in the test period is 0.13%, which is roughly the PR-AUC a random scorer would get. The autoencoder's PR-AUC is about 130 times higher.
- The two unsupervised models make different trade-offs at their validation-chosen thresholds. Isolation Forest catches more frauds but raises many false alarms. The autoencoder is more precise but catches fewer. The ensemble catches the most frauds of the three (42 of 75), with 161 false alarms.
- The reconstruction-error histogram shows the core difficulty: most frauds sit in the tail of normal behaviour, but they overlap with unusual legitimate transactions.
- The supervised model is far stronger, as expected once confirmed fraud cases are available to learn from. Unsupervised detection is most useful when labels are scarce or delayed, for fraud patterns not yet labelled, and as a source of cases for analysts to review and label.

The notebook also includes per-feature reconstruction errors (which components give fraud away), top-k alert precision, and a t-SNE visualisation of frauds and alerts.

## Limitations

- Two days of data from a single issuer in 2013. Fraud patterns drift, so results over longer periods would differ.
- The PCA-anonymised features rule out domain-level interpretation.
- With 57 frauds in validation and 75 in test, the metrics carry noticeable sampling uncertainty.
- The threshold maximises F1. A real deployment would set it from the cost of a missed fraud against the cost of an analyst review.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook fraud_detection_anomaly_detection.ipynb
```

The notebook downloads the dataset (69 MB) on first run. A full run takes about 1 minute on a 2-core CPU.

## Technologies

Python · PyTorch · scikit-learn · pandas · NumPy · Matplotlib

## References

- Dal Pozzolo, A., Caelen, O., Johnson, R. A. & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE Symposium on Computational Intelligence and Data Mining*.
- Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). Isolation Forest. *IEEE International Conference on Data Mining*.
- Saito, T. & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*.
