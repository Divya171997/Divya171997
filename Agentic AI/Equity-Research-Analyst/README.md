# Multi-Agent Equity Research Analyst

A team of agents that writes short equity-research briefs. Every number is **computed from market data**, every qualitative claim is **checked against earnings-call transcripts**, and a deterministic verifier and publisher **withhold anything they cannot verify**.

Notebook: [`equity_research_analyst.ipynb`](equity_research_analyst.ipynb), fully executed.

## Objectives

- Combine **SQL analytics** (prices) with **retrieval-augmented generation** (transcripts in a vector database) in one agent workflow.
- Make fabrication structurally hard. The LLM plans and writes, but never computes numbers, and everything it writes is attributed or withheld.
- Defend against **prompt injection** arriving through retrieved documents.
- Measure what verification buys, compared with a no-verifier pipeline and a naive "critique and revise" loop.

## Architecture

```mermaid
flowchart LR
    Q[Question] --> P[Planner<br/>LLM + normaliser]
    P --> QA[Quant analyst<br/>SQLite + NumPy]
    P --> RA[Research analyst<br/>ChromaDB + injection screen]
    QA -- "facts [F#]" --> W[Writer<br/>LLM]
    RA -- "passages [E#]" --> W
    W --> V{Verifier<br/>deterministic + embeddings}
    V -- "specific feedback<br/>(max 2 revisions)" --> W
    V -- done --> PB[Publisher<br/>best draft, withhold failures]
    PB --> B[Published brief<br/>with citations]
```

| Agent | Type | Responsibility |
|---|---|---|
| Planner | LLM + Pydantic, normalised by code | Extracts tickers and dates. Tickers must exist in the database, company names map to tickers, and dates are clamped to the data. |
| Quant analyst | Tools: SQLite, NumPy | Total return, annualised volatility, Sharpe ratio, maximum drawdown and beta versus an equal-weighted market index, recorded as facts `[F#]` |
| Research analyst | Tools: ChromaDB, MiniLM | Retrieves transcript passages filtered by ticker and year, and drops passages that look like instructions (prompt-injection screen); kept passages become evidence `[E#]` |
| Writer | LLM | 120–200-word brief from the facts and evidence only |
| Verifier | Deterministic + embeddings | **Attribution:** each number must match, at the precision written, a fact for the company named in that sentence, and under the metric named before it. **Comparisons** ("higher volatility than…", "a larger drawdown") are checked against the facts. **Qualitative** sentences need a transcript sentence with cosine similarity ≥ 0.50. Supported claims get citations automatically. |
| Publisher | Deterministic gate | Publishes the draft with the fewest problems (not necessarily the latest) and withholds every sentence that still fails, with a note |

## Data

| Data | Source | Licence |
|---|---|---|
| Daily prices, 505 S&P 500 stocks, 2013–2018 | Kaggle *S&P 500 stock data* (C. Nugent), via Plotly's [datasets](https://github.com/plotly/datasets) mirror | CC0 1.0 (data); mirror repository MIT |
| Company names and sectors | [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies) | ODC-PDDL 1.0 |
| Earnings-call transcripts: 57 calls of 16 companies, 2015–2017, in 1,150 chunks | [MAEC](https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction) (Li et al., CIKM 2020) | CC BY-SA 4.0 |

All data are downloaded at run time and not redistributed here. For the injection test, the notebook adds one clearly labelled, fabricated "poisoned" passage to the vector store.

## Results (from the committed run)

**Demo.** "Compare Apple and Microsoft in 2017…":

- The poisoned passage was blocked at retrieval.
- The first draft passed verification: 11 of 11 numbers attributed to the correct company's facts, and its qualitative claim was supported by a transcript passage.

**Evaluation on six research questions:**

| | First draft (no verifier) | Last revision (naive revise loop) | Published (verifier + publisher) |
|---|---:|---:|---:|
| Verification problems across the six briefs | 9 | **25** | 7 in the selected drafts, all withheld |
| Unverified claims shown to the reader | 9 | 25 | **0** |
| Sentences published | all | all | 26 of 33 (7 withheld) |

Two of the six first drafts passed every check outright (AMZN vs WMT, and CAT).

- **Revision is not automatically improvement.** With a 1.5B-parameter model, rewriting to fix one problem often introduced several new ones; the AMD/Intel brief went from 1 problem to 18. Keeping the *best* draft and withholding failing sentences is what makes the output trustworthy.
- **Genuine errors the verifier caught in model drafts:**
  - "XOM faced a larger drawdown" (−26.41% vs CVX's −37.91%);
  - "AMD … lower risk than Intel" (AMD volatility 85.05% vs 22.65%);
  - KO's volatility written as −14.38% (the fact is +14.38%), and PEP's −9.94% drawdown presented as its volatility;
  - an unsourced "46.1% revenue growth" for NVIDIA;
  - a GPU-demand explanation for NVIDIA when no NVIDIA transcript was available for 2016.
- **Development finding:** an earlier verifier required the model to write `[F#]` citations itself. The small model never did, so every brief scored 0% even when its numbers were right. Attribution by code is stricter (it checks company, metric and precision) and measures truthfulness, not formatting.

## Real-world applications

- Research assistants for buy-side and sell-side analysts, investor relations and corporate development.
- Any regulated generation task (financial reporting, compliance summaries, clinical summaries) where every figure must be traceable.

## Limitations

- Qualitative support uses embedding similarity, which can accept a paraphrase that subtly changes the meaning; an NLI entailment model would be stricter.
- The comparison checker handles common phrasings, not arbitrary language.
- Coverage is 16 companies, 2015–2017; prices exclude dividends; the market index is an equal-weighted average of the dataset's stocks.
- The local model's planner output often fails validation; the deterministic normaliser (tickers and years from the question) then takes over.
- Withholding sentences can leave a brief that reads abruptly (e.g. beginning "The company also…"). That is the honest trade-off for zero unverified claims.

## Future improvements

- NLI-based entailment for qualitative claims, and claim-level rather than sentence-level withholding.
- More metrics (valuation, fundamentals from filings) as additional quant tools.
- Planner-driven multi-hop retrieval, and a larger model via `OPENAI_API_KEY`, expected to produce richer drafts that pass more often.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # optional: OPENAI_API_KEY to use gpt-4o-mini
jupyter nbconvert --to notebook --execute --inplace equity_research_analyst.ipynb
```

The first run downloads about 30 MB of prices and a sparse checkout of the transcript repository, builds `data/market.db` and the ChromaDB store in `data/chroma`, and takes about 20 minutes on CPU with the local model. LLM calls are cached in `outputs/llm_cache.json`.

## Models

| Model | Licence |
|---|---|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (offline LLM) | Apache-2.0 |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (embeddings) | Apache-2.0 |

## Technologies

LangGraph · LangChain (OpenAI backend) · ChromaDB · sentence-transformers · Hugging Face Transformers · SQLite · Pydantic · NumPy · pandas · matplotlib · NetworkX

## References

- Li, J., Yang, L., Smyth, B. & Dong, R. (2020). MAEC: a multimodal aligned earnings conference call dataset for financial risk prediction. *CIKM*.
- Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*.
- Greshake, K. et al. (2023). Not what you've signed up for: compromising real-world LLM-integrated applications with indirect prompt injection. *AISec*.
- Gao, L. et al. (2023). RARR: researching and revising what language models say, using language models. *ACL*.
