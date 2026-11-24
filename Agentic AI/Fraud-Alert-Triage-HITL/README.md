# Agentic Fraud Alert Triage with Human-in-the-Loop

An agent workflow that **investigates every fraud alert, decides what it safely can, and pauses for a human analyst on the rest**, recording a complete audit trail. The goal is to cut analyst workload without losing fraud catches.

Notebook: [`fraud_alert_triage_hitl.ipynb`](fraud_alert_triage_hitl.ipynb), fully executed.

## Objectives

- Turn raw anomaly alerts into decisions (`escalate` / `dismiss`) backed by evidence.
- Use the language model only where the evidence is ambiguous, and never let it make a high-stakes call alone.
- Demonstrate a real **pause-and-resume** human-in-the-loop step with LangGraph `interrupt`, and measure workload, catch rate and precision against honest baselines.

## Architecture

```mermaid
flowchart LR
    T[Transaction stream] --> D[Detector<br/>Isolation Forest]
    D -- "top 0.5% = alert" --> E[Evidence collector<br/>tools]
    E --> R{Risk router<br/>calibrated bands}
    R -- clear fraud --> P[Policy guard]
    R -- clearly legitimate --> P
    R -- ambiguous --> L[LLM investigator]
    L --> P
    P -- "high value or<br/>low confidence" --> H[[Human analyst<br/>interrupt / resume]]
    P -- automated decision --> C[Case reporter]
    H --> C
    C --> A[(JSONL audit log)]
```

| Agent / node | Type | Responsibility |
|---|---|---|
| Detector | Isolation Forest | Scores transactions; the threshold is fixed on historical data |
| Evidence collector | Tools | Anomaly percentile, the most unusual features (z-scores), the outcomes of the 25 most similar *historical* labelled cases, amount, hour |
| Risk router | Rules | Bands calibrated on historical alerts: auto-dismiss only where the past fraud rate was ≤ 0.5%, auto-escalate only where every evidence level had a fraud rate ≥ 50%, and send the middle band to the LLM |
| LLM investigator | LLM, Pydantic output | Decision, confidence and a one-sentence rationale, given the evidence and the calibrated base rate |
| Policy guard | Hard rules | No automated dismissal above €500, and LLM decisions below 0.6 confidence go to a human |
| Human analyst | LangGraph `interrupt` | The graph pauses with a case packet and resumes with the analyst's decision. **Simulated here with the true labels.** |
| Case reporter | Template | Writes a case note and appends a JSON line to the audit log for every alert |

## Data

| Dataset | Source | Licence |
|---|---|---|
| Credit-card fraud, Sept 2013 (284,807 transactions, 492 frauds; PCA-anonymised, no personal information) | Machine Learning Group, ULB / Worldline, via a [public GitHub mirror](https://github.com/stat432/credit-analysis); SHA-256-verified at download | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) (database), [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) (contents) |

The data are downloaded at run time and not redistributed here. Splits follow time: the first 60% of transactions is history (detector training, case library and threshold calibration), and the last 40% is the live period being evaluated.

## Results (from the committed run)

The detector raised **530 alerts** in the live period, of which 35 were fraud (6.6% precision). All 530 went through the triage graph.

| Policy | Frauds caught (of 35 alerted) | False escalations | Precision of escalations | Human reviews | Workload vs review-all |
|---|---:|---:|---:|---:|---:|
| Analyst reviews every alert | 35 | 0 | 100% | 530 | 100% |
| Auto-escalate every alert | 35 | 495 | 6.6% | 0 | 0% |
| **Agentic triage** | **35** | **1** | **97.2%** | **178** | **34%** |

- **Routing:** 491 alerts were auto-dismiss candidates, 36 auto-escalated and 3 investigated by the LLM. 175 of the would-be dismissals exceeded €500, so policy sent them to a human.
- **LLM investigator:** 3 of 3 proposals were correct (dismiss). All three were above €500, so the policy guard still required human sign-off.
  - In an earlier iteration the investigator did not see the calibrated base rate; it escalated all three, and all three were wrong. Adding the historical fraud rate to the evidence fixed this.
- **Rules:** 351 of 352 automated decisions were correct.
- **Detector ceiling:** 97 live frauds were never alerted. Triage cannot recover what the detector misses, so improving recall is a detector problem, out of scope here.

## Real-world applications

- Card and payment fraud operations: alert queues, case management and analyst productivity.
- AML transaction monitoring, insurance-claims triage, and security-alert (SOC) triage. These share the same "too many alerts, few true positives" structure and need audit trails.

## Limitations

- The **human analyst is simulated** with the true labels, which assumes a perfect reviewer. Real human accuracy is lower, so the human rows are an upper bound.
- Two days of 2013 data; thresholds are calibrated on one historical window, with no drift handling.
- The ambiguous band is small on this dataset, so the LLM decides very few cases. That is by design: clear cases do not need a model. But it means the LLM's measured accuracy rests on 3 cases.
- PCA-anonymised features limit how interpretable the evidence is to a human.

## Future improvements

- Analyst feedback loop: human decisions retrain the case library and recalibrate the bands.
- A cost-sensitive optimisation of the routing bands, using the fraud team's cost of a miss versus the cost of a review.
- Integration with a case-management queue, and SLA-aware prioritisation of the paused threads.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # optional: OPENAI_API_KEY switches the investigator to gpt-4o-mini
jupyter nbconvert --to notebook --execute --inplace fraud_alert_triage_hitl.ipynb
```

The whole notebook runs in about 2 minutes on CPU. Outputs include `outputs/audit_log.jsonl` (one record per alert) and `outputs/alerts.html` (interactive evidence plot).

## Models

| Model | Licence |
|---|---|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), the offline investigator | Apache-2.0 |

## Technologies

LangGraph (StateGraph, `interrupt`, `Command`, MemorySaver checkpointer) · LangChain (OpenAI backend) · Hugging Face Transformers · scikit-learn · Pydantic · pandas · matplotlib · Plotly

## References

- Dal Pozzolo, A., Caelen, O., Johnson, R. A. & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE SSCI*.
- Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). Isolation forest. *IEEE ICDM*.
- LangGraph documentation: human-in-the-loop with `interrupt`.
