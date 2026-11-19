## Divya Arora

Junior Research Fellow (Project Associate-I) at the **Indian Institute of Information Technology Allahabad**, working on multi-agent decision-making for automated vehicles at dense, unsignalised intersections in the CARLA simulator.

My current question is a methodological one: when a learned intersection controller improves aggregate traffic statistics, is it actually completing the crossing task, or has it stopped attempting it? I build evaluation protocols that separate *participation* and *completion* from system-level outcomes, and test them with replicated training, clustered bootstrap inference and matched causal interventions.

M.Sc. Data Science, University of Essex (Merit) · B.Tech. Computer Science and Engineering, UPES Dehradun

---

### Current research

**DST–NSF India–USA project:** *Turning "Tragedy of the Commons" into "Emergent Cooperative Behavior" for Automated Vehicles at Intersections with Meta-Learning* (supervisor: Dr Surya Prakash)

**Study 1: Participation and completion in learned intersection control** (journal manuscript, under review)
- CARLA 0.9.15 (Town10HD_Opt, synchronous mode, 10 Hz). Twenty independently trained controllers, each evaluated frozen on 20 held-out traffic seeds: 400 controller–episode combinations and 1,960 focal-vehicle traces.
- A shared tabular multi-agent Q-learning controller (Yield / Creep / Proceed; 693-state gap–count–distance discretisation; Gipps-type longitudinal control), compared with a deterministic reservation-priority baseline under the same protocol.
- Evaluation reports eligible opportunities, conflict-zone entry, completion conditional on entry and unconditional success separately from aggregate statistics. Uncertainty comes from two-way clustered bootstrap intervals (20,000 replicates over controller and seed).
- Aggregate metrics can hide non-participation. The run with the lowest mean conflict-zone count made 0 entries in 48 eligible opportunities, and 15 of 20 trained instances entered repeatedly but completed no crossing. At a similar entry rate, the reservation baseline completed all 8 of its entries.
- The failure was traced through a five-stage mechanism audit, a reward-discounting analysis and a matched contact-removal intervention. The effective horizon is about 1 s at γ = 0.9, which strongly discounts the delayed completion bonus.

**Study 2: Image-based gridlock quantification and autonomy-stack audit** (iSES 2026)
- A fixed-parameter computer-vision pipeline: dual-band HSV segmentation with hue wrap-around, morphological cleaning and 8-connected components. It was applied without per-scenario tuning to 398 overhead frames from two runs of byte-identical traffic code, one ending in gridlock and one forming transient dense-flow channels.
- A source-level audit of a prototype autonomy stack showed that only a simplified longitudinal controller affected vehicle control, so the observed gridlock could not be attributed to the SLAM, LiDAR, A* or Q-learning modules.

Code and data for these studies are not currently public.

### Publications and manuscripts

- **D. Arora**, N. Sharma, R. Kala, X. Li, G. C. Nandi, S. Prakash. *Beyond Aggregate Traffic Metrics: Evaluating Participation and Completion in Adaptive Control at Unsignalized Intersections.* Journal of Intelligent Transportation Systems (Taylor & Francis), under review.
- **D. Arora**, N. Sharma, R. Kala, G. C. Nandi, S. Prakash. *Gridlock and Transient Channel Formation at a Dense CARLA Intersection: Image-Based Quantification and Autonomy-Stack Audit.* 12th International Symposium on Smart Electronic Systems (iSES 2026), NIT Goa. Accepted.
- *Analysis and Insights from ESG Disclosures.* Co-authored institutional volume, NIIF Chair in ESG, IIM Ahmedabad, 2025.
- H. K. Sharma, Shailendra, **D. Arora**. *User Activity Tracer: An Application to Record User Activities on Windows Platform.* IJRAR 5(3), 2018. Code: [Minor-2-User-Activity-Tracer-](https://github.com/Divya171997/Minor-2-User-Activity-Tracer-)
- H. K. Sharma, **D. Arora**, Shailendra. *Digital Cancer Diagnosis with Counts of Adenoma and Luminal Cells in Pleomorphic Adenoma Immunostained Healthcare System.* IJRAR 5(4), 2018.

### Projects in this account

Most repositories here are from my M.Sc. at Essex (2019–2020). Each README states the module, whether the work was individual or group, and what I contributed.

| Repository | What it is |
|---|---|
| [Predicting short-term stock movements](https://github.com/Divya171997/Predicting-Short-Term-Stock-Movements-with-Quantitative-Finance-and-Machine-Learning-in-Python.) | M.Sc. dissertation: LSTM vs. ARIMA, SVM, random forest and simple quantitative baselines on daily stock prices (Python, TensorFlow/Keras) |
| [Extreme multi-label text classification](https://github.com/Divya171997/Text-Analytics-eXtreme-Multi-Label-Classification-XMLC) | Subject indexing of PubMed and EconBiz titles with ML-kNN and Label Powerset (scikit-learn, scikit-multilearn) |
| [Forex time-series forecasting](https://github.com/Divya171997/Big-Data-for-Computational-Finance-Forex_Exchange_Rate) | 22 USD exchange rates, 2000–2019: cleaning, visualisation and walk-forward ARIMA compared against a persistence baseline |
| [Elasticsearch search engine and evaluation](https://github.com/Divya171997/Information-Retrieval_Elasticsearch-Evaluation) | Indexing and querying 8,000 news articles, with precision/recall reporting; tested Elasticsearch 8 version |
| [Residential properties: applied statistics](https://github.com/Divya171997/Residential_Properties-Applied_Statistics) | Group project: MICE imputation, multinomial logistic regression, LDA, random forest and boosting in R |
| [User Activity Tracer](https://github.com/Divya171997/Minor-2-User-Activity-Tracer-) | B.Tech. project behind the 2018 IJRAR paper: logs running and installed Windows applications to MySQL (Java) |

Other coursework: [classifier comparison for problematic-account detection](https://github.com/Divya171997/Machine-Learning-and-Data-Mining) · [indexing web pages with TF-IDF](https://github.com/Divya171997/Information-Retrieval_Indexing-for-Web-Search) · [life-expectancy regression](https://github.com/Divya171997/Life_Expectancy-Modelling_Experimental_Data) · [GDP factors dashboard](https://github.com/Divya171997/Analysing-Factor-Affecting-GDP) and [sector analysis](https://github.com/Divya171997/Agriculture-Industry-and-Service-Affecting-GDP-Of-Country) · [officer-injury EDA](https://github.com/Divya171997/Factors-Affecting-Officers-Injury) · [bank asset auction system (C)](https://github.com/Divya171997/Minor-1-Bank-Asset-Auctioning-System-)

### Tools

Python (primary), R, SQL, Java · PyTorch, TensorFlow/Keras, scikit-learn · tabular RL and reward design · CARLA 0.9.15 and Traffic Manager · OpenCV · NumPy/SciPy, pandas, Matplotlib · Git, Linux, LaTeX

### Experience

- **Junior Research Fellow**, IIIT Allahabad (Oct 2025 – present)
- **Business Intelligence and Operations Consultant**, Geneka Pharmaceuticals (Oct 2024 – Sep 2025)
- **Senior Data Scientist**, Softsensor AI, on the ESG project with IIM Ahmedabad (Sep 2021 – May 2024)

### Contact

divyaarora171997@gmail.com · [LinkedIn](https://www.linkedin.com/in/divya-arora-b91620195)
