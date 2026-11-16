# Question Bank: MLOps, Deployment and AI System Design

> **What this is:** 69 interview questions with model answers on versioning, reproducibility, deployment, inference APIs, monitoring, CI/CD, LLMOps, AI system design and production debugging, for rapid self-testing and last-week revision.
>
> **How to use it:** read the question, close your notes, answer out loud first (aim for about 2 minutes), and only then open the answer and compare it with the key points. Each section says which questions are core (asked in almost every senior AI/ML loop) and which are additional.
>
> **Source chapters:** MLOps: [Lifecycle and Experiment Tracking](../10-mlops/01-ml-lifecycle-and-experiment-tracking.md) · [Versioning and the Model Registry](../10-mlops/02-data-and-model-versioning-and-model-registry.md) · [Reproducibility and Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md) · [Deployment Patterns](../10-mlops/04-model-deployment-patterns.md) · [Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md) · [Monitoring](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md) · [CI/CD and Testing](../10-mlops/07-ci-cd-and-testing-for-ml-pipelines.md) · [LLMOps](../10-mlops/08-llmops-tracing-prompt-management-and-langfuse.md). System design: [Approach](../11-system-design/01-how-to-approach-ai-system-design-interviews.md) · [Enterprise RAG](../11-system-design/02-design-a-scalable-enterprise-rag-system.md) · [Multi-Tenant LLM Platform](../11-system-design/03-design-a-multi-tenant-llm-inference-platform.md) · [Support Chatbot and Evaluation](../11-system-design/04-design-an-enterprise-support-chatbot-with-evaluation.md) · [Fraud Detection](../11-system-design/05-design-a-real-time-fraud-detection-system.md) · [Quantized LLM Not Faster](../11-system-design/06-debugging-case-quantized-llm-not-faster.md) · [RAG Unsupported Answers](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md) · [Rising p95 Latency and Memory](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md).

## Table of contents

- [1. Versioning, lineage and the model registry](#1-versioning-lineage-and-the-model-registry)
- [2. Reproducibility and training-serving skew](#2-reproducibility-and-training-serving-skew)
- [3. Deployment, release strategies and promotion gates](#3-deployment-release-strategies-and-promotion-gates)
- [4. Building a scalable inference API](#4-building-a-scalable-inference-api)
- [5. Monitoring ML and LLM systems in production](#5-monitoring-ml-and-llm-systems-in-production)
- [6. CI/CD, testing and LLMOps](#6-cicd-testing-and-llmops)
- [7. Approaching AI system design interviews](#7-approaching-ai-system-design-interviews)
- [8. System design: a scalable enterprise RAG system](#8-system-design-a-scalable-enterprise-rag-system)
- [9. More design problems: inference platform, support chatbot, fraud detection](#9-more-design-problems-inference-platform-support-chatbot-fraud-detection)
- [10. Debugging: a quantized LLM is not faster](#10-debugging-a-quantized-llm-is-not-faster)
- [11. Debugging: a RAG app gives fluent but unsupported answers](#11-debugging-a-rag-app-gives-fluent-but-unsupported-answers)
- [12. Debugging: rising p95 latency and memory usage](#12-debugging-rising-p95-latency-and-memory-usage)
- [Rapid-fire round](#rapid-fire-round)
- [How to practice](#how-to-practice)

## 1. Versioning, lineage and the model registry

Core: Q1-Q2. Additional: Q3-Q4.

<details>
<summary><b>Q1. How do you version models, data, and experiments?</b></summary>

**Answer:** A model's behavior is a function of code, data, config, environment and random seed, so each gets an immutable ID and the pipeline records them automatically. Code lives in Git (and candidate runs from a dirty tree are refused); data is versioned as content-hashed snapshots (DVC or lakeFS for files, Delta Lake or Iceberg snapshot IDs for tables); the environment is a lock file plus a container image digest. Every training run is an experiment in a tracker such as MLflow or W&B that logs the commit, resolved config, data version, image digest, seeds, metrics (including per-slice) and the model with its preprocessing. Models we might ship are registered as immutable versions in a model registry linked back to that run. The chain registry version, then run ID, then data version, commit and config is the lineage that lets you answer "where did this model come from?" in minutes.

**Key points to mention:**
- Never log a dataset *name* like `latest`; log an immutable version or hash (a hash proves which data was used, only a snapshot can restore it).
- Most runs are never registered: the tracker records everything tried, the registry holds the curated few.
- LLM twist: also version prompts, model snapshot or weights revision, tokenizer and chat template, embedding model, index, retrieval config, eval set and judge prompt, pinned together in a release manifest.
- Measure it: pick a random production model and time how long it takes to name its commit, data, config and image; run periodic rebuild drills.
- Trade-off: keep everything behind production versions, let unregistered runs and intermediates expire under a retention policy.

**Deep dive:** [The ML Lifecycle, MLOps and Experiment Tracking](../10-mlops/01-ml-lifecycle-and-experiment-tracking.md#what-to-log-in-every-run), [Data Versioning, Model Versioning and the Model Registry](../10-mlops/02-data-and-model-versioning-and-model-registry.md#what-a-version-really-is-content-hashes-and-immutable-snapshots)

</details>

<details>
<summary><b>Q2. What is the role of a model registry?</b></summary>

**Answer:** A model registry is the system of record for models we intend to ship and the hand-off point between training and deployment. It stores immutable model versions with lineage and metadata (owner, signature, per-slice metrics, model card), tracks which version is in which state through aliases such as `champion` and `challenger`, records approvals and who promoted what, and enforces access control. Because deployment references an alias that resolves to a concrete version, promotion and rollback become metadata operations instead of rebuilds. A typical flow is: register, automated gates against the champion, challenger alias, shadow or canary, human approval, champion, with the previous version kept as the rollback target.

**Key points to mention:**
- Tracker vs registry: thousands of runs vs a curated set of candidates and production versions with governance.
- Stages vs aliases: MLflow deprecated fixed stages in 2.9 in favor of flexible aliases and tags.
- Separation of duties: trainers can register, only approved roles or pipelines can move `champion`; every alias move is audited.
- Failure modes: registry as a runtime dependency of serving (load and cache the pinned artifact at startup instead), garbage-collected old artifacts that break rollback, teams deploying straight from a bucket.
- "A registry with no metadata is just a bucket": registration should fail if required fields are missing.

**Deep dive:** [Data Versioning, Model Versioning and the Model Registry](../10-mlops/02-data-and-model-versioning-and-model-registry.md#the-model-registry)

</details>

<details>
<summary><b>Q3. How does the ML lifecycle differ from the software lifecycle, and what are the MLOps maturity levels?</b></summary>

**Answer:** In software, behavior is defined by code; in ML it is defined by code plus data plus hyperparameters plus randomness, so development is experimental, failures are often silent statistical degradation instead of crashes, tests use thresholds instead of exact assertions, and models decay as the world drifts. The lifecycle is therefore a loop: problem definition, data, training, evaluation, deployment, monitoring, retraining. Google's maturity model has level 0 (manual notebooks and hand-offs), level 1 (an automated training pipeline with continuous training and validation gates) and level 2 (CI/CD for the pipeline itself). From level 0, I would first add experiment tracking, data versioning and a registry, then turn the notebook into a parameterized pipeline, then add scheduled retraining with gates, and add monitoring early.

**Key points to mention:**
- Sculley et al. (2015): ML code is a small part of the system; "changing anything changes everything"; hidden feedback loops (a spam filter shapes its own future labels).
- Prioritize by pain: irreproducible models means tracking first; fast decay means CT and monitoring; many contributors breaking things means CI.
- Not every model needs level 2: match investment to change frequency, error cost and number of models.
- Pitfall: buying a platform before establishing practices, or automating a pipeline whose inputs are not versioned.

**Deep dive:** [The ML Lifecycle, MLOps and Experiment Tracking](../10-mlops/01-ml-lifecycle-and-experiment-tracking.md#mlops-maturity-levels)

</details>

<details>
<summary><b>Q4. A newly promoted model performs worse in production. Walk me through how versioning helps you respond.</b></summary>

**Answer:** Mitigate first: move the production alias back to the previous version, which is fast because its artifact and environment are retained and ideally still warm. Then confirm the new version is really the cause: the regression should start exactly at the promotion and the old version should recover on current traffic; if both degrade, the cause is upstream (input drift, a broken feature pipeline, a schema change) and rollback alone will not fix it. Next, diff the lineage of the two versions (data version, commit, config, image digest, eval set), which usually narrows the cause to one of data, code, config or environment immediately. Finally, ask why offline evaluation missed it (often a missing slice), add that slice as a gate, and use lineage to find other models that consumed the same bad data.

**Key points to mention:**
- Rollback should be one rehearsed action, not an improvised rebuild.
- Dashboards split by `model_version` show the regression boundary precisely.
- If data changed: compare snapshots for new partitions, label distribution, schema and null rates.
- Downstream impact analysis is the "what used this?" direction of lineage.

**Deep dive:** [Data Versioning, Model Versioning and the Model Registry](../10-mlops/02-data-and-model-versioning-and-model-registry.md#going-deeper-debugging-an-incident-with-lineage)

</details>

## 2. Reproducibility and training-serving skew

Core: Q5-Q7. Additional: Q8-Q10.

<details>
<summary><b>Q5. How do you make ML training reproducible?</b></summary>

**Answer:** Make every input to a run explicit and versioned: code by git SHA with dirty trees refused, data by an immutable snapshot and content hash (never "the table as of now", because of late data, backfills and deletions), config as code with the fully resolved config logged, environment by a lock file and container digest, and randomness by seeding every RNG including data loader workers. Log all of that, plus hardware and driver versions, in the tracker and attach it to the registered model. Distinguish three levels: lineage (always), statistical reproducibility across several seeds (for comparisons), and bitwise determinism (targeted). Bitwise determinism needs deterministic kernels (`torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG`), costs speed and only holds on identical hardware, software and GPU count, so I enable it for CI and debugging, not every run.

**Key points to mention:**
- Seed list: Python `random`, NumPy, `torch.manual_seed`, `worker_init_fn` plus a seeded DataLoader `generator`, `PYTHONHASHSEED` set before launch, library `random_state`.
- Resumability: checkpoints include optimizer, scheduler, scaler, RNG states and data position, written atomically.
- A container does not include the NVIDIA driver; cross-hardware bitwise reproduction is generally not achievable.
- Verify, do not assume: a CI job trains a tiny model twice in deterministic mode and asserts identical weights; real rebuilds must land within seed noise.

**Deep dive:** [Reproducible Training and Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md#making-training-reproducible-the-recipe)

</details>

<details>
<summary><b>Q6. How do you version datasets and preprocessing?</b></summary>

**Answer:** Datasets get immutable, content-addressed versions: DVC or lakeFS commits for files, snapshot IDs with time travel for Delta Lake or Iceberg tables. Preprocessing has two parts and both are versioned: the code (in Git, packaged in a pinned container) and the fitted state (scaler statistics, imputation values, vocabularies, tokenizers, chat templates), saved inside or next to the model artifact. The processed training set's identity is effectively a hash of the raw data version, the preprocessing commit and the preprocessing config, so changing any of them produces a new ID. The registry entry links the model to all of these, and the serving container verifies the preprocessing hashes at startup and refuses mismatches.

**Key points to mention:**
- Re-fitting a scaler on new data is different preprocessing even with identical code.
- Version the split and the label definition too: assign splits by a deterministic hash of the entity ID so one user never lands in two splits.
- Feature definitions as code (for example Feast): new logic means a new feature or version, not a silent change to history.
- Watch time-travel retention (VACUUM, snapshot expiry): pin or export snapshots behind production models.
- Trade-off: materialize the processed set (storage cost, exact reuse) vs recompute it (cheap, but only if preprocessing is deterministic and pinned).

**Deep dive:** [Data Versioning, Model Versioning and the Model Registry](../10-mlops/02-data-and-model-versioning-and-model-registry.md#versioning-preprocessing-code-and-feature-definitions-together-with-data), [Reproducible Training and Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md)

</details>

<details>
<summary><b>Q7. How do you prevent training-serving skew?</b></summary>

**Answer:** Training-serving skew is any difference between the features or preprocessing a model saw in training and what it receives at serving time, and the cure is "only one of everything". Use one shared transformation implementation (a bundled pipeline or preprocessing exported inside the model graph) instead of re-implementing features for serving; use a feature store so each feature has one definition feeding offline and online stores, with point-in-time correct joins for training; and version tokenizers, vocabularies and configs with the model. The strongest guarantee is log-and-train: log the exact features used at serving time and train future models on those logs. Then verify with row-level feature parity checks, distribution monitoring and golden-set replay tests in CI.

**Key points to mention:**
- Classic causes: pandas `std` uses n-1 while NumPy uses n, `NaN` vs `0` defaults, stale online vs fresh training features, RGB vs BGR, time zones, evaluating FP16 but serving INT4, a different chat template.
- Training must simulate serving freshness: join on availability time and apply the same TTL.
- Log-and-train has a cold-start ("log and wait") cost and privacy obligations.
- When a second implementation is unavoidable (on-device, C++ path), treat training code as the spec and gate deploys on a cross-language parity test.

**Deep dive:** [Reproducible Training and Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md#preventing-skew)

</details>

<details>
<summary><b>Q8. You re-ran a training job with the same seed and got a different validation metric. How do you debug it?</b></summary>

**Answer:** First establish what changed by diffing the run manifests: git SHA and dirty flag, data hash and row counts, resolved config, package versions, CUDA, cuDNN and driver, GPU type and count. Changed upstream data is the most common cause in practice, so check it first. If everything matches, rerun twice on the same machine with deterministic mode on: if that fixes it, the cause was GPU nondeterminism; if not, some RNG is unseeded or data order is unstable. To pinpoint it, compare initial weights, then first batch indices, then first-step loss; the first point of divergence names the culprit.

**Key points to mention:**
- `torch.use_deterministic_algorithms(True)` raises at the first nondeterministic op.
- Hidden sources: unsorted file listings, `set` iteration order, augmentation RNG in workers.
- Ask whether the difference matters at all: within seed-to-seed standard deviation, the issue is expectations, not a bug.

**Deep dive:** [Reproducible Training and Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md#sources-of-non-reproducibility)

</details>

<details>
<summary><b>Q9. Offline AUC is 0.92 but the model performs much worse in production. How do you tell skew, drift and leakage apart?</b></summary>

**Answer:** Skew is a pipeline difference: it shows up as a step change at a deploy and as row-level mismatches between online-logged features and offline recomputation for the same requests. Drift is the world changing: features match row by row, but the serving distribution moves away from training, usually gradually. Leakage means the offline number was never real: the model used future information, which I find by auditing feature timestamps and re-running evaluation with point-in-time joins on a time-based split; if offline AUC drops toward the online value, it was leakage. My order is: confirm the served artifacts are the evaluated ones, run feature parity, run prediction parity, check PSI or KS over time, then re-evaluate with as-of joins.

**Key points to mention:**
- Prediction parity (offline scoring of logged features vs logged predictions) isolates model-server problems.
- Concept drift can hurt performance with no visible input shift, so PSI can look clean; only delayed labels reveal it.
- Also check that label and metric definitions match between offline and online.

**Deep dive:** [Reproducible Training and Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md#detecting-skew)

</details>

<details>
<summary><b>Q10. What is a point-in-time correct join, and does a feature store eliminate training-serving skew?</b></summary>

**Answer:** A point-in-time correct join attaches, for each training example at time t, the most recent feature value that was available at or before t, never a later one. Example: sender reputation is 0.9 at 09:00, 0.4 at 12:00 and 0.1 at 15:00; an email at 11:00 must get 0.9, not 0.1, or the model learns from the future and fails online. Implement it with `pandas.merge_asof(direction="backward", by=entity)`, SQL `ASOF JOIN` or window functions, or the feature store's historical retrieval API, using the time the value became available in the online store. A feature store removes the biggest class of skew (inconsistent definitions and non point-in-time training data) but not all of it: online freshness, streaming vs batch computation differences and anything computed outside the store (tokenization, image preprocessing) can still diverge.

**Key points to mention:**
- Apply the same TTL as serving, so values that are too old become missing in training too.
- PIT joins fix temporal leakage, not a feature that is itself derived from the label.
- Feature stores are worth it when many models share entity features or real-time features matter; overkill for one batch model.

**Deep dive:** [Reproducible Training and Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md#point-in-time-correct-joins)

</details>

## 3. Deployment, release strategies and promotion gates

Core: Q11-Q13. Additional: Q14-Q16.

<details>
<summary><b>Q11. How do you deploy and monitor a production ML model?</b></summary>

**Answer:** I separate deployment (new version running) from release (users get its outputs). The model is trained by a reproducible pipeline, registered with lineage, and must pass CI promotion gates against the current champion; it is packaged as an immutable container or pinned artifact with its preprocessing and signature. It is deployed with zero traffic, shadowed on live traffic, then canaried in steps with automated metric gates and rollback, optionally followed by an A/B test for business impact. After full release I keep the old version warm and monitor four layers: system health (errors, p95/p99 latency, saturation), input data (schema, nulls, drift such as PSI), model outputs and quality (prediction distribution, accuracy once delayed labels arrive, calibration), and the business KPI. Every prediction is logged with request ID, `model_version`, features or a pointer and latency, so labels can be joined later for per-version accuracy.

**Key points to mention:**
- Choose the serving mode first: batch if staleness is fine (cheapest), online API, streaming, or edge.
- Serving stack by need: FastAPI for a small model, Triton, KServe, Ray Serve or BentoML for multi-model GPU serving, vLLM-class engines for LLMs.
- Page on SLO symptoms; input drift alone is usually a ticket, not a page.
- Close the loop: fast path is rollback via alias or routing; slow path is retraining through the same gated rollout.
- Pitfall: monitoring only infrastructure metrics misses silent model failures; delayed labels need proxy metrics.

**Deep dive:** [Model Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#deploying-and-monitoring-a-production-model-end-to-end), [Monitoring ML and LLM Systems in Production](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md)

</details>

<details>
<summary><b>Q12. How do you implement canary deployments and rollback?</b></summary>

**Answer:** Deploy the new version next to the stable one and route a small, sticky slice of traffic to it with `bucket = hash(salt + user_id) mod 100`, canary if `bucket < k`, then step k up (for example 1%, 5%, 25%, 50%, 100%) with a bake time at each step. At each step a controller compares canary metrics with stable traffic *in the same window*: errors, latency, prediction distribution, quality proxies and business guardrails. Hard failures roll back immediately; subtle differences are judged statistically (for example 90 vs 180 errors out of 18,000 each gives z of about 5.5, a clear failure). Rollback is a config change: set canary weight to zero or move the registry alias back, with the previous version kept warm for a soak period longer than the label delay.

**Key points to mention:**
- Tools: Argo Rollouts, Flagger, KServe `canaryTrafficPercent`, managed endpoint traffic weights.
- What rollback cannot fix: predictions already acted on, schema changes (use expand-contract), downstream models trained on new outputs.
- Canary replicas with cold caches or different load skew latency; per-request routing causes flip-flopping.
- A canary limits blast radius and detects large regressions; it is not an A/B test for small improvements.
- Measure the process: time to detect, time to roll back (rehearsed in game days) and false-abort rate of gates.

**Deep dive:** [Model Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#canary-releases-metric-gates-and-automated-rollback), [Implementing rollback](../10-mlops/04-model-deployment-patterns.md#implementing-rollback)

</details>

<details>
<summary><b>Q13. How do you validate a candidate model before promotion?</b></summary>

**Answer:** Through layered gates, always relative to the current champion scored on the same frozen, versioned, recent holdout in the same job. The gate rule is an absolute floor, non-regression versus production within a tolerance, and per-slice non-regression on critical segments, plus behavioral tests, latency, memory and cost budgets on target hardware, and a packaging check that the artifact loads in the serving image with the right signature. Significance matters: with 2,000 examples, a candidate that fixes 60 errors and introduces 40 moves accuracy from 0.900 to 0.910, but McNemar's statistic is (|60-40|-1)^2/100 = 3.61 < 3.84, so p is about 0.057 and the gain is not significant. A candidate that passes is registered as challenger with its gate report, then goes through shadow and canary before becoming champion.

**Key points to mention:**
- An aggregate gain can hide a large drop in a 10% language slice: gate per slice with a minimum slice size.
- Derive the tolerance from measured noise (champion score across seeds, bootstrap interval), not guesses.
- Pitfall: refreshing the eval set and comparing with the champion's old score; re-score both.
- For LLMs: golden prompts, format and tool-call validity, safety and refusal rates, pairwise judge vs current release, cost and latency.
- Overrides of a failed gate need a named approver and written justification stored with the version.

**Deep dive:** [Model Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#validating-a-candidate-model-before-promotion), [CI/CD and Testing for ML Pipelines](../10-mlops/07-ci-cd-and-testing-for-ml-pipelines.md#layer-5-evaluation-gates-against-the-production-model)

</details>

<details>
<summary><b>Q14. When would you use shadow deployment vs canary vs an A/B test?</b></summary>

**Answer:** Shadow runs the new model on a copy of live traffic and discards its outputs, so I use it to see real-traffic behavior with zero user risk (new architecture, new features, unknown performance). Canary exposes a small, growing share of real users to limit blast radius and catch regressions automatically. An A/B test randomizes users into arms for a pre-planned duration to get causal evidence that the new model improves a business metric. They are usually chained: shadow, then canary, then A/B if a business claim must be proven.

**Key points to mention:**
- Shadow cannot measure user reactions, costs extra compute and must have no side effects (never execute write tools in shadow).
- A/B needs a planned sample size: 5.0% to 5.6% conversion needs about 21,900 users per group at 80% power and alpha 0.05.
- For ranking, interleaving is far more sensitive than A/B but measures preference, not revenue.
- Low traffic: lean on shadow, offline replay and per-customer canaries.

**Deep dive:** [Model Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#shadow-deployment)

</details>

<details>
<summary><b>Q15. An A/B test shows the new model lifts conversion by 12% relative. Do you ship it?</b></summary>

**Answer:** Not on the lift alone. 500/10,000 vs 560/10,000 is a 12% relative lift but gives z of about 1.89 and p of about 0.058, and the test was underpowered (it needed about 21,900 users per group). The 95% interval for the difference is roughly -0.02 to +1.22 percentage points, consistent with no effect and with a large one. I would also check sample ratio mismatch (planned 50/50 but 10,200 vs 9,800 gives chi-square 8, p about 0.005, so assignment is broken), whether the sample size was fixed in advance or someone peeked, guardrail metrics, test duration relative to weekly cycles and novelty effects, and whether the lift holds in key segments.

**Key points to mention:**
- Peeking inflates false positives: use fixed horizons or sequential methods.
- Multiple metrics and slices need correction; pre-register the primary metric.
- Practical significance: is the plausible lift worth the extra cost or complexity?
- Long-term effects need holdouts.

**Deep dive:** [Model Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#ab-testing-with-statistical-significance)

</details>

<details>
<summary><b>Q16. Your canary passed every gate, but a week after full rollout the business metric dropped. What happened, and how do you prevent it?</b></summary>

**Answer:** Most likely the gates only covered fast operational signals while true quality depends on delayed labels or slow user behavior, or the regression hit a segment too small to see in a short canary. Other causes are novelty effects wearing off, a concurrent change (upstream data or another experiment), seasonality or feedback loops. I would roll back (the previous version should still be warm), check that the drop correlates with the rollout time and the treated population, and scope impact with per-prediction version logs. Prevention: slower quality gates, longer intermediate canary stages, segment-level monitoring, proxy metrics and a long-term holdout group.

**Key points to mention:**
- Keep the `previous` alias and warm capacity longer than the label delay.
- Per-version accuracy dashboards once labels arrive.
- Compare against a long-term holdout to separate the model's effect from the world's.

**Deep dive:** [Model Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#going-deeper-senior-level-trade-offs-and-failure-modes)

</details>

## 4. Building a scalable inference API

Core: Q17-Q20. Additional: Q21-Q23.

<details>
<summary><b>Q17. How do you build a scalable inference API?</b></summary>

**Answer:** I start from requirements (peak RPS, payload and output sizes, latency SLO, budget), then choose the API shape: synchronous for fast calls, SSE streaming for LLM output, asynchronous jobs with a durable queue for long tasks. A stateless async CPU tier (auth, validation, routing, deadlines) sits in front of a separate GPU tier where each GPU is owned by one process that loads the model once and batches requests (dynamic batching for classic models, continuous batching in an engine such as vLLM for LLMs). Everything is bounded: input size, sequence length, queue length with 503 and Retry-After, per-tenant rate limits. Calls carry end-to-end deadlines, retries are limited and jittered, dependencies sit behind circuit breakers, and I autoscale on queue depth or concurrency with a replica floor and monitor latency percentiles, throughput, errors, saturation and cost.

**Key points to mention:**
- Little's law L = lambda x W sizes concurrency: 2 chats/s x 8 s = 16 concurrent streams; load test the safe concurrency per replica.
- Readiness is true only after weights are loaded and warmed up; use a startup probe for slow loads.
- Load shedding instead of unbounded queues; fallbacks (smaller model, second provider, cached answer).
- Graceful shutdown that drains streams; model version in every response and metric.
- Pitfall: several worker processes each loading the model onto the same GPU.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md)

</details>

<details>
<summary><b>Q18. How do you handle concurrency, batching, and GPU memory limits?</b></summary>

**Answer:** Concurrency: the HTTP layer is async so thousands of mostly waiting requests are cheap, blocking model calls run off the event loop, CPU-heavy preprocessing goes to a process pool (the GIL blocks pure-Python threads), and one process owns each GPU. Batching: a batcher groups queued requests up to a max batch size and a max wait time, trading a little latency for much more throughput (illustratively, batch 1 in 20 ms gives 50 req/s while batch 16 in 40 ms gives 400 req/s). Memory: bound batch size and sequence length and admit requests only if their worst-case memory fits, which for LLMs is mostly KV cache. A 7B model in FP16 with 32 layers, 32 KV heads and head dim 128 needs 2 x 32 x 32 x 128 x 2 = 524,288 bytes per token, so 60 GB holds about 114k tokens, only 27 requests at 4,096 tokens: admit by tokens, not request count, and return 503 with Retry-After when full so overload becomes backpressure instead of an OOM crash.

**Key points to mention:**
- 429 means one client exceeded its own limit; 503 means the service as a whole is overloaded or not ready. Both carry Retry-After.
- Reserving worst case is safe but wasteful; paged engines allocate on demand and preempt under pressure (adds tail latency).
- Bucket by sequence length so one long input does not pad the whole batch.
- Measure: batch-size histogram, queue wait vs compute time, KV usage and preemption count, peak memory in a worst-case load test.
- Keep a memory margin: fragmentation can OOM below 100% reported usage.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#gpu-memory-limits-admission-control-and-backpressure), [Server-side dynamic batching](../10-mlops/05-building-a-scalable-inference-api.md#server-side-dynamic-batching)

</details>

<details>
<summary><b>Q19. How do you distinguish CPU-bound, memory-bound, and GPU-bound workloads?</b></summary>

**Answer:** I look at which resource saturates first as load rises, then change one variable and see whether the bottleneck moves. CPU-bound: cores pegged (often a single core because of the GIL), gaps in the GPU timeline, high event-loop lag, profiler time in tokenization or image decoding. Memory-bound has two flavors: capacity (GPU memory or KV cache full, OOMs, preemptions) and bandwidth (high DRAM activity, low tensor-core activity, per-token latency flat as batch grows, typical of LLM decode). GPU compute-bound: high SM and tensor-core activity and latency growing roughly linearly with batch size. A roofline check helps: decode at batch 1 is about 1 FLOP per byte, far below an A100's ridge point of about 156, so it is bandwidth-bound until batching raises intensity.

**Key points to mention:**
- Tools: `py-spy`, PyTorch profiler or Nsight Systems, and DCGM counters (`SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE`) instead of `nvidia-smi` utilization, which only says some kernel was running.
- Fixes differ: process pools and fast tokenizers for CPU; caps, quantization and paged KV for capacity; bigger batches, weight quantization and speculative decoding for bandwidth; lower precision, FlashAttention or a smaller model for compute.
- Prefill is compute-heavy, decode is bandwidth-heavy, so the answer can differ by phase.
- Do not forget I/O-bound: low CPU and GPU with high latency means you are waiting on downstream calls.
- Classic trap: adding GPUs to a CPU-bound service changes only the bill.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#cpu-bound-memory-bound-or-gpu-bound-diagnosing-the-bottleneck), [GPU Fundamentals: Compute-Bound vs Memory-Bound](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

</details>

<details>
<summary><b>Q20. How do you implement timeouts, retries, and circuit breakers?</b></summary>

**Answer:** Timeouts: set an end-to-end deadline at the edge and propagate the remaining budget; each downstream call gets the minimum of its own cap and the remaining time minus a reserve, and work is skipped if the deadline has already passed. Retries: only for idempotent requests (or with idempotency keys) and transient errors (timeouts, 429, 502, 503, 504), with exponential backoff and full jitter, honoring Retry-After, at a single layer, under a retry budget of about 10% extra traffic. Circuit breakers per dependency: closed counts failures, open fails fast for a reset timeout, half-open lets a trial call through and closes on success or reopens on failure, ideally paired with a fallback. Retry amplification is why this matters: 3 layers with 3 attempts each means up to 27 attempts at the bottom for one user request.

**Key points to mention:**
- Backoff: delay_n = uniform(0, min(cap, base x 2^n)); with base 0.1 s and cap 2 s the bounds are 0.1, 0.2, 0.4, 0.8, 1.6, 2.0 s.
- Trip breakers on failure rate over a sliding window with minimum volume; count timeouts and 5xx, not 4xx; one breaker per provider, region or model.
- Streaming: separate TTFT timeout and inter-token idle timeout; cancel the first generation before retrying an LLM call (otherwise you pay twice).
- Many HTTP clients and LLM SDKs retry by default; align them with your own layer.
- Bulkheads (bounded pool per dependency) stop a slow but not yet failing dependency from exhausting every worker; watch the retry ratio as an early warning.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#timeouts-and-deadline-propagation), [Retries](../10-mlops/05-building-a-scalable-inference-api.md#retries-backoff-jitter-idempotency-and-retry-budgets), [Circuit breakers](../10-mlops/05-building-a-scalable-inference-api.md#circuit-breakers)

</details>

<details>
<summary><b>Q21. When would you choose a synchronous API, an asynchronous job API, or streaming?</b></summary>

**Answer:** Synchronous when the work finishes in milliseconds to a few seconds and the caller needs the answer now, such as classification, embeddings or reranking. Streaming (SSE) when a human watches long generated output, because perceived latency drops to the time to first token and the user can cancel early. Asynchronous jobs with a durable queue when work takes minutes, arrives in bursts or must survive restarts, such as document processing, batch scoring or long agent runs: return 202 Accepted with a job ID, then poll or call a webhook. Hybrids are common, for example an agent API that streams progress events while the heavy work runs as a background job.

**Key points to mention:**
- Async jobs need idempotent workers, ack after completion, a dead-letter queue and job TTLs.
- Streaming costs: long-lived connections complicate load balancing and deploys (drain time), proxies may buffer, mid-stream retries are hard.
- Structured outputs are often better non-streamed, or streamed and validated at the end.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#api-design-choices-synchronous-asynchronous-jobs-and-streaming)

</details>

<details>
<summary><b>Q22. Why is CPU utilization a poor autoscaling signal for a GPU inference service, and how do you handle cold starts?</b></summary>

**Answer:** In a GPU service the CPU can sit at 20% while the GPU is saturated and the queue explodes, so CPU-based scaling never fires; GPU utilization is also poor because it reads near 100% under any meaningful load. I scale on leading indicators: queue depth or in-flight requests per replica, or LLM engine metrics such as waiting requests and KV cache usage, with targets found by load testing. The Kubernetes HPA formula is ceil(current x metric / target), so 4 replicas at queue depth 30 with target 10 gives 12. GPU cold starts take minutes (node provisioning, image pull, weight loading, warm-up), so I keep a replica floor and warm headroom, scale up aggressively and down slowly, pre-pull images and cache weights locally.

**Key points to mention:**
- Latency is a lagging signal: alert on it, scale on leading indicators.
- Weight loading alone: 14 GB at an illustrative 500 MB/s is 28 s before compilation or warm-up.
- Scale-to-zero suits rare, latency-tolerant workloads, not interactive chat.
- The queue absorbs the spike while replicas start only if it is bounded and deadlines are enforced.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#autoscaling-and-cold-starts)

</details>

<details>
<summary><b>Q23. You need to serve a 7B chat model at a peak of 5 new conversations per second, each streaming for about 10 seconds. How many replicas?</b></summary>

**Answer:** By Little's law, peak concurrency is L = lambda x W = 5 x 10 = 50 concurrent streams. If a realistic load test shows one GPU replica sustains 16 concurrent streams within the TTFT and TPOT SLOs (illustrative), I need ceil(50 / 16) = 4 replicas at peak. I add headroom for zone failure and bursts, for example 30% or one extra replica per zone, so about 5 to 6, and autoscale between a minimum floor and that peak. I also check the memory side: 16 streams times average tokens per stream must fit in the KV cache budget.

**Key points to mention:**
- The per-replica number must come from an open-loop load test with real prompt and output length distributions.
- If replicas span 2 zones and one fails, the survivors must still carry peak.
- Compute cost per request at average traffic, not just peak, to show the price of the replica floor.
- Levers to need fewer replicas: quantization, prefix caching, shorter outputs, routing easy requests to a smaller model.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#the-vocabulary-of-serving-latency-throughput-percentiles-and-littles-law)

</details>

## 5. Monitoring ML and LLM systems in production

Core: Q24-Q25. Additional: Q26-Q28.

<details>
<summary><b>Q24. How do you detect drift and delayed performance degradation?</b></summary>

**Answer:** I layer signals by speed because labels often arrive late. With no labels, I monitor data quality first, then input drift on important features (PSI or KS against a reference window) and prediction drift (score distribution, positive rate), which are leading indicators; a spam score mix moving from 70/20/10% to 50/25/25% across three bands gives PSI of about 0.22, a moderate shift. Next come proxy metrics (user corrections, "Not spam" clicks, escalations) and a small continuously human-labeled sample. The authoritative check is joining delayed labels back to predictions by ID and computing metrics on matured cohorts by prediction date and model version, plus backtests. I slice everything by segment, and treat drift alerts as triggers for investigation, not automatic retraining.

**Key points to mention:**
- PSI = sum of (a_i - e_i) x ln(a_i / e_i); weight drift by feature importance, since huge samples make tiny drifts "significant".
- Never treat missing labels as negatives early; choose the label window from the measured label-arrival curve.
- Size the labeled sample: n = 400 at 88% accuracy gives about a 3.2-point margin; about 4,000 labels for 1 point.
- Concept drift can hurt accuracy with no input shift; label-free estimates (average confidence) catch covariate shift only.
- Feedback loops: blocked transactions never get labels, so keep a small exploration or review slice.

**Deep dive:** [Monitoring ML and LLM Systems in Production](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md#layer-3-data-and-prediction-drift), [Detecting delayed performance degradation](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md#detecting-delayed-performance-degradation)

</details>

<details>
<summary><b>Q25. How do you monitor p95 latency, throughput, and cost?</b></summary>

**Answer:** Each replica records latency as a histogram (Prometheus) labeled by route and model version, plus counters by status and gauges for queue depth and in-flight requests. Fleet p95 comes from `histogram_quantile` over summed bucket rates, never from averaging per-replica percentiles, and bucket boundaries sit near the SLO threshold because the function interpolates within buckets. Throughput is requests per second and, for LLMs, input and output tokens per second, compared with load-tested capacity; I also track goodput (requests that met the SLO), which collapses under overload while raw throughput stays flat. Cost per request is replica-hours times price divided by requests (4 replicas at an illustrative 2.50 USD per GPU-hour serving 10 RPS is about 0.28 USD per 1,000 requests, and 5x more at 2 RPS), or tokens times price for APIs, attributed per tenant and feature. I alert on SLO burn rate and use saturation metrics (queue depth, KV usage, DCGM activity) for early warning.

**Key points to mention:**
- Why percentiles: a mean of 253.75 ms can coexist with p50 of 140 ms and p95 of 900 ms; the mean describes nobody.
- LLM latency = TTFT + (N_out - 1) x TPOT; rising TTFT with flat TPOT suggests queueing or longer prompts, rising TPOT suggests bigger batches or memory pressure.
- No user IDs, prompts or raw paths as metric labels (cardinality); put detail in traces.
- Streaming: HTTP middleware measures time to headers, so instrument inside the generator.
- Cost per successful outcome, not just per call; track tokens per request p95, cache hit rate and retries.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#monitoring-p95-latency-throughput-errors-saturation-and-cost), [Monitoring ML and LLM Systems in Production](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md#layer-1-system-health)

</details>

<details>
<summary><b>Q26. Explain SLOs, error budgets and burn-rate alerts. Why not just alert when the error rate exceeds a threshold?</b></summary>

**Answer:** An SLO is a target for a good-events ratio over a window, for example 99.9% of requests successful and under 300 ms over 30 days. The error budget is the allowed bad fraction, 0.1%, which is 43.2 minutes of full outage in 30 days or 10,000 bad requests out of 10 million. Burn rate is the observed error rate divided by the budget rate: 1.44% errors for an hour is burn rate 14.4 and consumes 14.4 / 720 = 2% of the monthly budget in that hour, a common paging threshold. Multi-window burn-rate alerts (a long and a short window) catch significant problems fast, ignore blips and stop firing soon after recovery, which a raw threshold cannot do: it is either too sensitive or too slow and ignores how much budget is left.

**Key points to mention:**
- SRE Workbook starting points: page at 14.4x over 1 h (5 min short window), page at 6x over 6 h, ticket at 1x over 3 days.
- Budgets drive decisions: an exhausted budget means a freeze on risky releases.
- ML quality SLIs (share of sampled answers judged faithful) are noisier and slower: usually tickets, not pages.
- Page only on actionable user-facing symptoms; give every alert an owner and a runbook.

**Deep dive:** [Monitoring ML and LLM Systems in Production](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md#alert-design-slos-error-budgets-and-burn-rates)

</details>

<details>
<summary><b>Q27. How would you monitor the quality of a RAG chatbot in production when you have no ground-truth labels?</b></summary>

**Answer:** I combine cheap signals on all traffic with expensive signals on a sample. On all traffic: retrieval scores and empty-retrieval rate, abstention and refusal rates, answer length, guardrail flags, and explicit and implicit feedback. On a deterministic 1-5% sample that oversamples risky slices: asynchronous LLM-judge scores for faithfulness, answer relevance and context relevance, reweighted when reporting overall rates. A smaller uniformly random human-reviewed sample calibrates the judge (Cohen's kappa) and gives an unbiased estimate. Everything is sliced by tenant and topic and versioned by prompt, model and index; judging 2% of 1M requests at about 0.012 USD each is about 240 USD per month instead of 12,000 USD for all traffic (illustrative).

**Key points to mention:**
- Low faithfulness with good retrieval points to generation; low context relevance points to retrieval.
- Abstention spikes often mean broken retrieval (stale index, permission filter, embedding mismatch); abstention dropping to zero can mean more hallucination.
- Pin the judge model and prompt; re-check agreement after any judge change.
- Thumbs-down traces become regression eval cases after PII removal.

**Deep dive:** [Monitoring ML and LLM Systems in Production](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md#llm-specific-monitoring), [LLMOps: continuous evaluation](../10-mlops/08-llmops-tracing-prompt-management-and-langfuse.md#continuous-evaluation-of-a-rag-app-with-ragas-and-langfuse-scores)

</details>

<details>
<summary><b>Q28. nvidia-smi shows 100% GPU utilization, but throughput is low and p95 latency is high. What do you check?</b></summary>

**Answer:** `nvidia-smi` utilization only means some kernel was running during the sample period, not that the chip's compute is saturated. I check DCGM counters for SM activity, tensor-core activity and memory bandwidth, the engine's batch size and running versus waiting requests, KV cache usage and preemptions, tokens per second, and whether requests are queueing. LLM decode is usually memory-bandwidth-bound, so small batches show high "utilization" with poor throughput. Little's law tells me whether arrival rate times latency exceeds the maximum concurrent sequences, in which case a queue forms and TTFT climbs.

**Key points to mention:**
- High KV usage leads to preemption or queueing: shorter contexts, prefix caching, KV quantization or more replicas.
- Also check thermal or power throttling and CPU-side bottlenecks (tokenization, Python overhead).
- Compare TTFT and TPOT trends separately to tell queueing or prefill problems from decode problems.
- The mirror case (GPU at 30% with high p95) means the GPU is starved: event-loop lag, batching not forming, a lock, or slow downstream calls.

**Deep dive:** [Monitoring ML and LLM Systems in Production](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md#layer-1-system-health), [Batching, Continuous Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md)

</details>

## 6. CI/CD, testing and LLMOps

Core: Q29. Additional: Q30-Q34.

<details>
<summary><b>Q29. How do you test ML pipelines in CI/CD?</b></summary>

**Answer:** With a layered ML testing pyramid, because code, data and model change independently and each layer catches a different failure at the cheapest point. Every push runs unit tests for feature transformations, preprocessing and metric code, plus data validation with schemas and expectations (pandera or Great Expectations) on committed sample data. PRs that touch the pipeline run a smoke training on tiny data (loss decreases, can overfit a tiny batch, artifacts reload with identical predictions), behavior tests on the resulting model and in-process API integration tests, including a training-serving parity test. On merge or nightly, the full pipeline trains on versioned data and the candidate must pass an evaluation gate against production plus performance tests on production-like hardware before registration; CD then deploys through staging, shadow or canary with automatic rollback.

**Key points to mention:**
- Name the three gates separately: fast and medium tiers block a **merge**, in-pipeline data checks block a **training run**, the evaluation gate and performance tests block a **promotion**.
- Tier by cost: under about 5 minutes per push, medium on relevant PRs, heavy on main or nightly on GPU runners.
- CI pins data version, lock file and seeds so a commit fully identifies a run.
- Never fix flaky quality tests with retries; fix seeds, use tolerances or assert on pass rates.
- Measure CI itself: duration, flake rate and escaped defects, each of which becomes a new test at the cheapest layer.

**Deep dive:** [CI/CD and Testing for ML Pipelines](../10-mlops/07-ci-cd-and-testing-for-ml-pipelines.md#the-ml-testing-pyramid)

</details>

<details>
<summary><b>Q30. What is the difference between CI, CD and CT in ML, and what should trigger retraining?</b></summary>

**Answer:** CI automatically tests every change (code, data schemas, configs, prompts) before merging. CD turns validated changes into deployable artifacts (images, registered models) and moves them to production safely; continuous deployment skips the manual approval. CT (continuous training) is ML-specific: the deployed training pipeline itself runs automatically and produces candidate models that must pass the same gates. Retraining triggers can be a schedule (simple, predictable), a data-volume threshold, a drift alert, or a measured performance drop; many teams combine a schedule with drift and performance alerts because performance triggers wait on delayed labels and drift triggers are only proxies.

**Key points to mention:**
- CI tests the pipeline code; CT runs the pipeline; gates test each model it produces.
- Serving code and model versions are separate artifacts with separate lifecycles.
- CT without data validation and eval gates turns a broken upstream feed into a broken production model overnight.
- Tag every automatic run with its trigger so you can audit why each model exists.

**Deep dive:** [CI/CD and Testing for ML Pipelines](../10-mlops/07-ci-cd-and-testing-for-ml-pipelines.md#continuous-delivery-and-continuous-training), [The ML Lifecycle, MLOps and Experiment Tracking](../10-mlops/01-ml-lifecycle-and-experiment-tracking.md)

</details>

<details>
<summary><b>Q31. Give examples of minimum functionality, invariance and directional tests for a model.</b></summary>

**Answer:** These behavior tests come from CheckList (Ribeiro et al., 2020), which showed that models with high benchmark accuracy still fail simple capability tests. For a spam filter: a minimum functionality test (MFT) checks that blatant spam like "WIN a FREE iPhone, click now" is labeled spam; an invariance test (INV) checks that changing the recipient's name does not move the spam score beyond a small tolerance; a directional test (DIR) checks that appending "click here to claim your prize" does not lower the score. For house prices: price is positive (MFT), changing the listing ID does not change the price (INV), and more square footage with everything else equal does not lower the price (DIR).

**Key points to mention:**
- Generate cases from templates and assert on pass rates for large suites; reserve 100% requirements for a small critical set.
- Invariance tests double as fairness tests (names, gender terms, dialect).
- Tree models are piecewise constant: test "not decreasing", and use monotone constraints if the business requires it.
- Every production incident becomes a new behavior test.

**Deep dive:** [CI/CD and Testing for ML Pipelines](../10-mlops/07-ci-cd-and-testing-for-ml-pipelines.md#layer-3-model-behavior-tests)

</details>

<details>
<summary><b>Q32. How do you test an LLM or RAG application in CI without making tests slow, flaky and expensive?</b></summary>

**Answer:** Split it into tiers. My own code (prompt templating, parsing, tool routing, retries) is tested deterministically with recorded LLM responses replayed from a cache keyed by a hash of model, prompt and parameters, so unit tests never call a live API. Retrieval is tested with deterministic metrics such as recall@k on golden queries. Quality is checked by a regression eval of about 100 curated cases on PRs that touch prompts, models or retrieval, scored by code checks and a pinned LLM judge with absolute and relative thresholds against the main-branch baseline, under a hard cost cap. The full golden set and safety suites run nightly, and scheduled runs also catch silent behavior changes in hosted models.

**Key points to mention:**
- Cost math: 200 cases at about 0.0051 USD is about 1.02 USD per run, but 90 runs a day is about 92 USD, hence path filters, content-hash caching and budget guards.
- Noise: with 100 cases one case is 1 point; do not gate on differences below the standard error.
- Critical cases (safety, refusals, legal) have zero tolerance.
- Post per-case diffs on the PR, not just aggregate scores.

**Deep dive:** [CI/CD and Testing for ML Pipelines](../10-mlops/07-ci-cd-and-testing-for-ml-pipelines.md#ci-for-llm-applications)

</details>

<details>
<summary><b>Q33. How do you trace tool calls and model decisions in an LLM application or agent, and how does LLMOps differ from MLOps?</b></summary>

**Answer:** Every request produces one trace: a root span with user, tenant, session and release IDs, a generation span per model call (model, parameters, prompt version, token usage, `finish_reason`, requested tool calls), and a span per tool call (name, arguments, result or a pointer to it, latency, errors, and the tool call ID that links it to the generation that requested it). Routing, planning and handoff decisions get their own spans or events, and agent loops get one span per iteration so a 15-step loop is readable. I instrument with OpenTelemetry and its GenAI semantic conventions (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.tool.name`) so the same spans can go to Langfuse or another backend. This reflects how LLMOps differs from MLOps: the things you change are prompts, retrieval and tools rather than weights, the model is often an external API, outputs are nondeterministic free text, cost is per token, quality is judged by rubrics and judges, and the unit of debugging is a multi-step trace.

**Key points to mention:**
- Always create your own root span; auto-instrumentation alone gives disconnected spans without business context.
- Keep 100% of error traces (tail sampling) even if you sample successes; export asynchronously so tracing never blocks the request.
- Mask PII before export: tool results and retrieved chunks contain customer data, not only user messages.
- Derive metrics from spans: tool error rate and p95 per tool, steps per task, share of traces hitting the iteration cap, loops.

**Deep dive:** [LLMOps: Tracing, Prompt Management, Langfuse and RAGAS](../10-mlops/08-llmops-tracing-prompt-management-and-langfuse.md#tracing-concepts-traces-spans-generations-and-sessions), [From MLOps to LLMOps](../10-mlops/08-llmops-tracing-prompt-management-and-langfuse.md#from-mlops-to-llmops-what-actually-changes)

</details>

<details>
<summary><b>Q34. Walk me through safely rolling out a new prompt or LLM version, and rolling it back.</b></summary>

**Answer:** Treat it like a model release. Create the new prompt as an immutable version (v8) with a `staging` label, with the model snapshot, decoding parameters, retrieval config and tool schemas pinned in the same bundle. Run it over the versioned eval dataset against v7 on quality, safety, format and tool-call validity, cost and latency, and gate in CI on no regressions. Then canary it to 5-10% of users with deterministic hash-based assignment per user or conversation, recording the version on every trace, and watch judge scores, feedback, guardrail and refusal rates, cost and latency. If everything holds, move the `production` label to v8; if anything regresses, move it back to v7, which takes effect within the client cache TTL without a deploy.

**Key points to mention:**
- Pin dated model snapshots; never point production at a floating "latest" alias; provider deprecations are releases.
- Route per conversation, not per message, to avoid mid-conversation switches.
- Significance: 72% vs 76% thumbs-up with 2,000 ratings per arm gives z of about 2.89, but thumbs are biased, so add sampled judge scores.
- A longer prompt multiplies input tokens on every request: watch cost regressions.
- Fallback prompts in code protect against the registry being unreachable at cold start.

**Deep dive:** [LLMOps: the prompt management lifecycle](../10-mlops/08-llmops-tracing-prompt-management-and-langfuse.md#the-prompt-management-lifecycle), [Model Deployment Patterns: LLM-specific rollouts](../10-mlops/04-model-deployment-patterns.md#going-deeper-llm-specific-rollouts)

</details>

## 7. Approaching AI system design interviews

Additional: Q35-Q38.

<details>
<summary><b>Q35. Walk me through how you approach an open-ended AI system design question.</b></summary>

**Answer:** I clarify the users, the core task and the constraints, then turn them into measurable SLOs for latency, quality, availability and cost. I do a quick back-of-the-envelope estimate of load, tokens, GPUs and storage because it decides the architecture regime (one node or sharded, hosted API or self-hosted). Then I draw the online and offline paths, deep dive into the one or two riskiest components, and harden the design with an evaluation plan, failure modes, security, monitoring and cost. I close with trade-offs and how the system evolves from v0 to scale, checking in with the interviewer at each transition.

**Key points to mention:**
- Rough time split: 10% clarifying, 15% metrics and estimates, 15% architecture, 35% deep dives, the rest hardening and summary.
- AI-specific parts never to skip: how we know answers are good, how prompts, models and indexes are versioned and gated, and GPU or token cost.
- Pick deep dives by likelihood of failure times impact (for enterprise RAG: permission-aware retrieval and retrieval quality), not by what you know best.
- Pitfall: over-engineering; at 2 QPS a hosted API and a single index is the right v0.

**Deep dive:** [How to Approach AI/ML System Design and Debugging Interviews](../11-system-design/01-how-to-approach-ai-system-design-interviews.md#the-framework-at-a-glance)

</details>

<details>
<summary><b>Q36. A product manager says "make the assistant fast and accurate". How do you turn that into SLOs?</b></summary>

**Answer:** I translate each adjective into an indicator, a target and a window, and tie each to an action. "Fast" becomes p95 TTFT and p95 TPOT targets for a streaming UI, for example TTFT at most 1.5 s and TPOT at most 50 ms. "Accurate" becomes faithfulness and retrieval recall on a versioned golden set (for example faithfulness at least 0.90) plus online signals like thumbs-up rate. I add availability (99.9% success over 30 days, a 43.2-minute error budget) and cost per request, because those implicit requirements bite later. All numbers are illustrative and negotiated with product.

**Key points to mention:**
- When the error budget is spent, risky releases pause.
- Quality SLOs need a judge calibrated against human labels, or the SLO measures the judge's bias.
- Separate retrieval and generation metrics so a failure points to the right component.
- "Why not average latency?": a 1 s average can coexist with a 6 s p99 that users hate.

**Deep dive:** [How to Approach AI/ML System Design Interviews](../11-system-design/01-how-to-approach-ai-system-design-interviews.md#step-2-define-success-metrics-and-slos)

</details>

<details>
<summary><b>Q37. Estimate how many GPUs you need to serve a 7B model to 2 million daily users making 5 requests each.</b></summary>

**Answer:** That is 10 million requests a day, about 116 QPS average and roughly 350 QPS at a 3x peak. With 300 output tokens per request, peak output is about 104,000 tokens per second; if a benchmark shows one 80 GB GPU sustains about 2,500 tokens per second at our latency SLO, that is about 42 GPUs, or 60 at 70% target utilization, plus redundancy. Cross-check with memory: KV per token is 2 x 32 x 32 x 128 x 2 = 524,288 bytes, so about 49 sequences of 2,300 tokens fit in 60 GB, and Little's law gives 347 x 6 s, about 2,083 concurrent sequences at peak, so about 43 GPUs, consistent with the throughput estimate.

**Key points to mention:**
- Per-GPU throughput must be measured at the target latency; peak throughput at huge batches violates TPOT SLOs.
- Levers: a GQA model (4x less KV with 8 KV heads), INT8 or INT4 weights, FP8 KV cache, shorter prompts, prefix caching, speculative decoding.
- Add N+1 or cross-zone capacity, and autoscale because average load is about a third of peak.

**Deep dive:** [How to Approach AI/ML System Design Interviews](../11-system-design/01-how-to-approach-ai-system-design-interviews.md#step-3-back-of-the-envelope-estimation), [The KV Cache Explained](../06-llm-inference/02-kv-cache-explained.md)

</details>

<details>
<summary><b>Q38. Hosted LLM API or self-hosted open model: how do you decide?</b></summary>

**Answer:** I decide on five factors: quality needed (measured on our golden set), volume and how steady it is, latency requirements, data residency and privacy constraints, and the team's capacity to operate GPUs. Hosted APIs win for v0, low or spiky volume and top-tier quality. Self-hosting wins at high steady volume, strict data constraints, or when a fine-tuned smaller model is good enough. With the chapter's illustrative numbers, 10 million requests a day costs about 14,500 USD per day on an API versus about 3,600 USD for 60 GPUs, but 20,000 requests a day costs about 29 USD on the API, less than one always-on GPU at about 60 USD. Often the answer is a hybrid behind an LLM gateway.

**Key points to mention:**
- Compare total cost: idle GPU time, engineering and on-call, not just the GPU-hour price.
- Self-hosting gives version control (no silent updates) and latency tuning, at the cost of capacity planning.
- A gateway keeps switching a config change and reduces lock-in.
- Re-evaluate periodically: prices and model quality change quickly.

**Deep dive:** [How to Approach AI/ML System Design Interviews](../11-system-design/01-how-to-approach-ai-system-design-interviews.md#step-10-cost)

</details>

## 8. System design: a scalable enterprise RAG system

Core: Q39-Q44. Additional: Q45.

<details>
<summary><b>Q39. Design a scalable enterprise RAG system.</b></summary>

**Answer:** I split it into an ingestion path and a query path over a tenant-partitioned storage layer. Ingestion: per-tenant connectors with webhooks plus reconciliation crawls, content-hash change detection, queues, parsers, chunkers, batched GPU embedding workers and idempotent index writers that flip a per-document version pointer only after all chunks are written. Storage: object store for raw files, a metadata DB as source of truth for versions, ACLs and tombstones, and a hybrid BM25 plus vector index sharded by tenant. Query: authenticate, resolve the user's principals, optionally rewrite the query, run hybrid retrieval with tenant and ACL pre-filters, fuse with RRF, rerank with a cross-encoder, build a cited context, generate with instructions to cite and abstain, and post-check citations. Around it: timeouts and degraded modes, tenant-scoped caches, SLOs for latency, availability, quality and freshness, tracing and a cost model.

**Key points to mention:**
- Numbers first: 20M docs x 5,000 tokens is 200M chunks of 500 tokens; at 1,024 dims that is 819.2 GB of float32 vectors (204.8 GB as int8).
- 5M queries a day is about 58 QPS average, about 174 at a 3x peak (design for about 200); at 5 s per answer Little's law gives 200 x 5 = 1,000 concurrent generations.
- The LLM is close to 90% of per-query cost, so context size and model routing are the main cost levers.
- Pitfalls: post-filtering ACLs, global caches, mixing embedding versions, incomplete deletion.
- Prove it works: SLO dashboards plus cross-tenant canary queries that must always return nothing.

**Deep dive:** [System Design: A Scalable Enterprise RAG System](../11-system-design/02-design-a-scalable-enterprise-rag-system.md#step-3-high-level-architecture)

</details>

<details>
<summary><b>Q40. How do you partition and index documents?</b></summary>

**Answer:** Partition by tenant first with a tiered model: small tenants share clusters with a namespace each, large or regulated tenants get dedicated shards, and a tenant directory routes queries. Inside a large tenant, shard by hash of `doc_id` so a document's chunks stay together. Index every chunk in both a BM25 inverted index and a dense ANN index (HNSW in RAM, or IVF-PQ or disk-based graphs when it does not fit), with metadata fields for tenant, ACL principals, doc ID, version, timestamps and embedding model version. Use versioned index names behind aliases so re-indexing or resharding is a pointer flip. Tenant-first limits fan-out: with 1% slow shards, touching 100 shards makes about 63% of queries slow, while touching 4 makes about 4%.

**Key points to mention:**
- Tenant-first also gives isolation (a filter bug cannot reach data outside the search space), per-tenant deletion, re-indexing and data residency.
- Thousands of tiny namespaces stress some engines: group small tenants with mandatory filters enforced in a data-access layer.
- Choose vector precision with data: compare recall@10 for float16, int8 and binary plus rescoring against the 4x-32x RAM savings.
- Hot tenant: add replicas or build a new index version with more shards and flip.
- Measure per-shard size and QPS skew, per-shard p99, and recall@k with and without filters.

**Deep dive:** [System Design: A Scalable Enterprise RAG System](../11-system-design/02-design-a-scalable-enterprise-rag-system.md#step-5-partitioning-and-indexing-documents), [Vector Search: ANN, HNSW, IVF and PQ](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md)

</details>

<details>
<summary><b>Q41. How do you scale ingestion and embedding generation?</b></summary>

**Answer:** Decouple every stage with queues so each scales independently, batch chunks for GPU embedding with length bucketing, and autoscale workers on queue lag rather than CPU. Respect source API limits with per-connector token buckets, keep separate priority lanes so ACL changes and live edits are never stuck behind backfills, and enforce per-tenant fairness. Skip work wherever possible: content hashes avoid re-embedding unchanged documents and ACL-only changes take a metadata-only path. Backfills and re-embeddings run as checkpointed jobs writing into a new index version; sizing is simple arithmetic, for example a 300M-token backlog at 20,000 tokens/s per worker drained in one hour needs ceil(4.17) = 5 workers.

**Key points to mention:**
- Steady-state churn is small (2% daily is about 1.16 GPUs in the example); backfills and onboarding drive capacity.
- Reliability: idempotent writes with deterministic chunk IDs, retries with backoff and jitter, a DLQ for poison documents, backpressure from index writers.
- Hosted embedding API is fast to start but rate-limited and per-token; self-hosted wins at high steady volume.
- Measure oldest-message age per lane and tenant (the autoscaling signal), canary edit-to-searchable time, DLQ inflow, tokens/s per worker.

**Deep dive:** [System Design: A Scalable Enterprise RAG System](../11-system-design/02-design-a-scalable-enterprise-rag-system.md#step-6-scaling-ingestion-and-embedding-generation), [Production Ingestion Pipelines at Scale](../08-rag/12-production-ingestion-pipelines-at-scale.md)

</details>

<details>
<summary><b>Q42. How do you enforce tenant permissions?</b></summary>

**Answer:** Authorization is enforced in code at retrieval time, never by asking the model, in two layers. Tenant isolation: the tenant comes only from the authenticated token and the query is routed to that tenant's namespace or shards. User ACLs: each chunk stores the principals allowed to read its document, a permission service expands the user into user ID plus groups, and retrieval applies a pre-filter so only readable chunks can ever become candidates. A late-binding check verifies the final cited documents against the authoritative permission source, and every cache, log and eval dataset is tenant-scoped, with cache keys including the tenant and an ACL fingerprint (a hash of sorted principals).

**Key points to mention:**
- Post-filtering leaks data into intermediate systems and returns empty results for users with narrow access.
- Very selective filters hurt filtered ANN recall: use filter-aware engines, wider search, or exact search over small allowed subsets.
- Revocation SLO of about a minute: high-priority metadata sync lag plus group-cache TTL must fit inside it.
- Deny overrides allow; expand nested groups; watch users in thousands of groups (huge filter clauses).
- Test with permission-trap golden questions and production cross-tenant canary queries.

**Deep dive:** [System Design: A Scalable Enterprise RAG System](../11-system-design/02-design-a-scalable-enterprise-rag-system.md#step-8-enforcing-tenant-permissions), [Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md)

</details>

<details>
<summary><b>Q43. How do you handle failures and document deletion?</b></summary>

**Answer:** In ingestion, every write is idempotent, transient errors are retried with exponential backoff and jitter, poison documents go to a monitored DLQ with a redrive tool, partial writes stay invisible until the version pointer flips, and reconciliation crawls catch missed events. On the query path, every stage has timeouts and circuit breakers and a defined degraded mode: skip rewrite, skip rerank, BM25-only or dense-only, fallback LLM, search results without synthesis, or cached answers, with each response tagged by mode. Deletion is two-phase: an immediate tombstone makes the document unretrievable and invalidates caches within seconds, then an asynchronous purge removes it from every index, replica, raw store, log and eval set, and an audit job verifies it is gone. Availability math justifies fallbacks: a gateway at 99.99% in series with retrieval, reranker and LLM at 99.9% each gives about 99.69%, while parallel fallbacks (1 - (1 - A1)(1 - A2)) and an optional reranker lift it to about 99.99% if failures are independent.

**Key points to mention:**
- ANN indexes often mark-delete: schedule compaction and include it in the purge SLO.
- Tenant offboarding: drop the namespace and destroy the tenant key (crypto-shredding).
- Backups need documented retention so "purged within 30 days" is true.
- Measure time-to-unretrievable, time-to-purge, DLQ size and age, degraded-mode rate.

**Deep dive:** [System Design: A Scalable Enterprise RAG System](../11-system-design/02-design-a-scalable-enterprise-rag-system.md#step-9-failures-degraded-modes-and-deletion-guarantees)

</details>

<details>
<summary><b>Q44. How do you define latency, quality, and availability SLOs?</b></summary>

**Answer:** Each SLO is an SLI, a target and a window. Latency: p95 TTFT under 2 s and p95 full answer under 8 s, measured at the gateway and also per tenant. Availability: the fraction of valid requests that return a good answer, 99.9% over 30 days, with an explicit rule for whether degraded answers count. Quality: retrieval recall@k on golden sets, faithfulness and citation precision from a calibrated LLM judge on sampled production traffic, and abstention correctness on unanswerable questions. I add freshness SLOs for edits, permission revocations and deletions, measured with synthetic canary documents; 99.9% over 30 days is a 43.2-minute budget, or 150,000 failed requests at 150M requests a month.

**Key points to mention:**
- Alert on burn rate: 14.4x for one hour spends 2% of a 30-day budget, a common paging threshold.
- Percentiles do not add across stages: budget per stage for engineering, but measure end to end with tracing.
- Report per tenant tier; a global average hides a large tenant's bad week.
- Customer SLAs are looser than internal SLOs.

**Deep dive:** [System Design: A Scalable Enterprise RAG System](../11-system-design/02-design-a-scalable-enterprise-rag-system.md#step-12-slos-for-latency-quality-availability-and-freshness)

</details>

<details>
<summary><b>Q45. How would you migrate all tenants to a new embedding model without downtime?</b></summary>

**Answer:** Vectors from different embedding models live in incompatible spaces, so you never mix them in one index or compare their scores. Build a new versioned index per tenant with a low-priority, checkpointed backfill, and start dual-writing live changes to both indexes before the backfill begins (or replay changes since its snapshot), otherwise edits, ACL changes and deletes made during the backfill are lost and a deleted document could reappear. Evaluate the new index offline on the golden set and in shadow, flip the alias tenant by tenant starting with small tenants, keep the old index for rollback through a bake period, then delete it. In the chapter's example the re-embed is 10^11 tokens, about 1,389 GPU-hours or about 3 days on 20 GPUs (illustrative).

**Key points to mention:**
- RAM roughly doubles during migration; quantizing the new index helps.
- Store `embedding_model_version` in index metadata and make the query path check it.
- Upgrading only the query encoder produces no errors, just quietly worse retrieval.

**Deep dive:** [System Design: A Scalable Enterprise RAG System](../11-system-design/02-design-a-scalable-enterprise-rag-system.md#going-deeper-trade-offs-and-how-the-design-evolves)

</details>

## 9. More design problems: inference platform, support chatbot, fraud detection

Additional: Q46-Q49.

<details>
<summary><b>Q46. In a multi-tenant LLM platform, one team launches a bulk job and everyone else's latency jumps. How do you prevent noisy neighbors?</b></summary>

**Answer:** Layer three defenses. First, per-tenant quotas at the gateway: token buckets on tokens per minute, requests per minute and concurrent requests, where a token limit reserves prompt tokens plus `max_tokens` at admission and refunds the unused part from the actual usage event. Rate limiting by tokens matters because GPU cost scales with tokens and requests differ by orders of magnitude in size. Second, priority classes so bulk work runs in a preemptible batch class with weighted shares (for example 60/30/10) rather than strict priority. Third, per-tenant fair queuing by tokens inside each class, backed by admission control that rejects early with 429 and Retry-After instead of queueing forever.

**Key points to mention:**
- Chunked prefill so one tenant's 30k-token prompts do not stall others' decode steps.
- Distributed enforcement: Redis with atomic scripts, or local leases refreshed from a central store.
- Escalate isolation to dedicated replicas for tenants whose SLO cannot tolerate sharing.
- Prove fairness with per-tenant goodput and queue wait metrics.

**Deep dive:** [System Design: A Multi-Tenant LLM Inference Platform](../11-system-design/03-design-a-multi-tenant-llm-inference-platform.md#step-7-priority-classes-fairness-and-admission-control), [Rate limiting and per-tenant quotas](../11-system-design/03-design-a-multi-tenant-llm-inference-platform.md#step-6-rate-limiting-and-per-tenant-quotas)

</details>

<details>
<summary><b>Q47. You need to serve 200 fine-tuned variants of the same 7B model. How?</b></summary>

**Answer:** If they are LoRA adapters, serve them all on shared base-model replicas with multi-LoRA serving: the base weights load once, and adapters (rank 16 on the four attention projections of a 7B GQA model with 8 KV heads is about 13.6M parameters, about 27 MB in FP16, versus 14 GB for the base) are paged between object storage, CPU memory and a few GPU slots. The mechanism is h = Wx + BAx: Wx is batched for everyone while BAx uses per-request adapters through special kernels (Punica SGMV, S-LoRA paging), so requests for different adapters share one batch. Route adapter-aware so requests land on replicas that already have their adapter loaded, and merge the few highest-traffic variants into dedicated replicas because unmerged adapters add some latency. If they are full fine-tunes, each is a separate 14 GB model, which needs bin-packing, scale-to-zero for the tail and a cold-start policy, so I would push teams toward LoRA.

**Key points to mention:**
- Per-tenant access control on adapters: tenant A cannot invoke tenant B's adapter.
- Latency overhead grows with rank and the number of distinct adapters per batch.
- Track adapter load misses as a cold-start metric.

**Deep dive:** [System Design: A Multi-Tenant LLM Inference Platform](../11-system-design/03-design-a-multi-tenant-llm-inference-platform.md#step-9-multi-lora-serving-for-many-fine-tuned-variants)

</details>

<details>
<summary><b>Q48. A support chatbot's answer is factually correct, but its citation points to the wrong document. Is that acceptable, and how do you score and fix it?</b></summary>

**Answer:** Not as a pass in an enterprise setting: the citation is a verifiability contract, and a citation that does not support the sentence means either the model answered from memory or there is a citation mapping bug, so the next answer may be wrong and still look verified. I score correctness and citation correctness as separate dimensions: correctness passes, citation precision and recall fail for that claim, and the case fails on high-risk slices. Then I diagnose from the trace: if the right document was never retrieved, it is a retrieval failure; if it was retrieved but missing from the prompt, it was lost at reranking or truncation; if the model emitted the right ID but the UI shows another, it is a mapping bug; if the model cited another ID, it is a generation attribution error. Fixes include stable chunk IDs (`doc#version#chunk`) end to end, citations constrained to IDs present in context, a per-sentence NLI or judge check with re-attribution, and fewer, deduplicated chunks with the strongest evidence first.

**Key points to mention:**
- Exception: if the cited document also genuinely supports the claim (duplicate, translation), add it to `acceptable_doc_ids`.
- Faithfulness against all retrieved context is the key diagnostic: pass means retrieval worked and attribution failed.
- A small batched NLI cross-encoder is usually cheap enough inline; otherwise verify synchronously only for high-risk intents.
- Add the case and variants to the golden set and gate on citation precision for high-risk slices.

**Deep dive:** [System Design: An Enterprise Support Chatbot and Its Evaluation Framework](../11-system-design/04-design-an-enterprise-support-chatbot-with-evaluation.md#going-deeper-correct-answer-wrong-citation)

</details>

<details>
<summary><b>Q49. Design a real-time fraud detection system for card payments. What are the hard parts?</b></summary>

**Answer:** I clarify the actions (approve, decline, step-up, review), volume, latency and who bears losses, then frame it as calibrated binary classification under heavy imbalance with amount-dependent costs. Features are transaction attributes, streaming velocity aggregates, user and merchant history and device or graph features, served from a feature store with point-in-time correct training data; a GBDT scores in-process after hard rules within a budget of about 50 ms. The hard parts are not the model: labels (chargebacks arrive weeks later, so train on matured windows plus fast proxy labels), selection bias (declined transactions never get labels, fixed with a small exploration slice and inverse propensity weights), and thresholds from business costs (decline when p > C_FP / (C_FP + C_FN), so with a 10 USD false-positive cost a 200 USD transaction is declined above about 4.8% while a 20 USD one needs about 33%). I evaluate out-of-time with PR-AUC, dollar recall and expected cost, roll out via shadow and canary, and explain decisions with SHAP reason codes.

**Key points to mention:**
- 10M transactions a day is about 116 TPS average; plan for roughly 10x peaks.
- PR-AUC over ROC-AUC: at 1,000 frauds in 1M, 80% recall at 1% FPR gives only about 7.4% precision.
- Negative downsampling inflates probabilities: correct in odds space before applying cost thresholds.
- Fallback when the feature store is slow: score with missing features or rules only, fail open below an amount limit.
- Two feedback loops: rules in minutes, retraining in weeks.

**Deep dive:** [System Design: A Classic ML System (Real-Time Fraud Detection)](../11-system-design/05-design-a-real-time-fraud-detection-system.md#step-2-labels-and-label-delay), [Thresholds from business costs](../11-system-design/05-design-a-real-time-fraud-detection-system.md#step-6-handling-imbalance-and-choosing-thresholds-from-business-costs)

</details>

## 10. Debugging: a quantized LLM is not faster

Core: Q50-Q55.

<details>
<summary><b>Q50. A quantized LLM uses less memory but has almost unchanged latency. Diagnose it.</b></summary>

**Answer:** Lower memory proves the weights are smaller; flat latency means reading weights was not what limited latency, or the smaller weights are not being read efficiently. I first clarify the method and library (bitsandbytes, GPTQ, AWQ, FP8), the GPU, the serving stack, prompt and output lengths, concurrency and which latency metric, and check the benchmark is fair (warm-up, `torch.cuda.synchronize()`, same prompts, same output lengths, percentiles). Then I split latency into outside-the-model time, TTFT and TPOT and test hypotheses: a compute-bound workload (prefill-heavy or large batches, where W4A16 still computes in FP16), missing fused low-bit kernels (dequantize then cuBLAS GEMM moves about 4.5 bytes per weight versus 2 for FP16), KV cache dominating the bytes, overhead outside the GPU math, and queueing with unchanged concurrency limits. The typical root cause is memory-oriented quantization served through an eager Python loop; the fix is an engine with fused W4A16 kernels and CUDA graphs, or FP8 W8A8 on H100, plus raised concurrency, verified with a 2 x 2 benchmark (engine x precision) and a quality eval.

**Key points to mention:**
- On an A100, W4A16 linear layers become compute-bound at roughly B > 39 tokens, much sooner than FP16 (about 156).
- MBU (model bandwidth utilization) falling sharply after quantization means the bytes went away but the time did not.
- Hypothesis 0 is always the measurement itself.
- Never claim a speedup without re-running the quality evaluation.

**Deep dive:** [Debugging Case: A Quantized LLM Uses Less Memory but Latency Is Unchanged](../11-system-design/06-debugging-case-quantized-llm-not-faster.md#1-the-case-and-how-to-attack-it-in-an-interview)

</details>

<details>
<summary><b>Q51. Is the workload compute-bound or memory-bandwidth-bound? How do you tell?</b></summary>

**Answer:** It depends on the phase and batch size. Prefill is compute-bound because hundreds or thousands of tokens share each weight read; decode at small batch is memory-bandwidth-bound because each weight is read to produce one token per sequence; large-batch decode moves toward compute-bound. The arithmetic intensity of a linear layer is about 2B / b_w FLOPs per byte, and an A100's ridge point is about 312 TFLOPS / 2.0 TB/s = 156, so FP16 crosses it at B of about 156, W8A16 at about 78 and W4A16 at about 39: weight-only quantization makes a workload compute-bound sooner. I measure the TTFT versus decode split, the running batch size and DCGM tensor-active versus DRAM-active, and place hot kernels on a roofline with Nsight Compute.

**Key points to mention:**
- Decode share: a chat with 200 output tokens is about 98% decode; a classifier with 6,000 input and 5 output tokens is about 85% prefill (illustrative).
- MBU for decode, MFU for prefill.
- If both tensor and DRAM activity are low, the GPU is neither: it is idle or launch-bound.

**Deep dive:** [Debugging Case: Quantized LLM](../11-system-design/06-debugging-case-quantized-llm-not-faster.md#5-hypothesis-1-is-the-workload-compute-bound-or-memory-bandwidth-bound)

</details>

<details>
<summary><b>Q52. Are optimized quantized kernels being used? How do you check?</b></summary>

**Answer:** Check what was loaded (module types, device map, quantization config), read the engine's startup log for the method and kernel it chose, then profile. A fused path shows a single low-bit GEMM or GEMV kernel per layer; a slow path shows dequantize kernels followed by standard cuBLAS GEMMs, FP32 kernels, or host-to-device copies from CPU offload. Dequantize-then-GEMM reads 0.5 byte, writes 2 bytes and reads 2 bytes per weight, about 4.5 bytes, which is worse than FP16's 2 bytes and can be slower than not quantizing. Finally I microbenchmark one layer at batch sizes 1, 8, 64 and 512 against FP16: a healthy fused 4-bit kernel beats FP16 at small token counts and converges to it near the compute-bound crossover.

**Key points to mention:**
- Known traps: bitsandbytes NF4 fast path mainly for single-token input, compute dtype left at FP32, LLM.int8() outlier decomposition, GPTQ or AWQ without kernel packages, FP8 on GPUs without FP8 tensor cores.
- MBU is the quick quantitative check (52% falling to 17% in the illustrative case).
- Fix: serve a pre-quantized checkpoint in a format with fused kernels (AWQ or GPTQ with Marlin-style kernels, FP8 on Hopper).

**Deep dive:** [Debugging Case: Quantized LLM](../11-system-design/06-debugging-case-quantized-llm-not-faster.md#6-hypothesis-2-are-optimized-quantized-kernels-actually-being-used), [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q53. Is the KV cache the dominant memory consumer? How would you find out, and what if it is?</b></summary>

**Answer:** Compute it: KV bytes per token = 2 x layers x KV heads x head dim x bytes per value, times context length, times concurrent sequences, and compare with weight bytes. For a Llama-2-7B-shaped model that is 0.5 MiB per token in FP16, so 8 sequences averaging about 8,025 tokens hold about 33.66 GB of KV against 14 GB of FP16 weights. When KV dominates, weight quantization attacks the minority of the bytes: in that long-document example the ideal decode speedup from 4-bit weights is only about 47.66 / 37.60, roughly 1.27x. The remedies target KV instead: FP8 or INT8 KV cache (halving KV bytes), GQA models, prefix caching and shorter contexts.

**Key points to mention:**
- Separate capacity (how many sequences fit) from traffic (bytes read per decode step); both scale with batch x length.
- At batch 1 and about 1,100 tokens, weights dominate, so the answer depends on the workload.
- Engines that pre-allocate memory make `nvidia-smi` useless here: read KV usage metrics and the startup log's block count.
- Profile attention versus linear kernel time.

**Deep dive:** [Debugging Case: Quantized LLM](../11-system-design/06-debugging-case-quantized-llm-not-faster.md#7-hypothesis-3-is-the-kv-cache-or-attention-the-dominant-cost), [The KV Cache Explained](../06-llm-inference/02-kv-cache-explained.md)

</details>

<details>
<summary><b>Q54. How would you profile inference?</b></summary>

**Answer:** Top-down, from cheap and broad to expensive and narrow. First request-level timers and tracing (outside the model vs TTFT vs decode), then serving engine metrics (queue, running batch, TPOT, KV usage), then device counters from DCGM (SM-active, tensor-active, DRAM-active rather than GPU-Util), then the PyTorch profiler for the dominant ops, then Nsight Systems for the timeline, and only finally Nsight Compute on the single hot kernel. Timing must be correct: warm-up, `torch.cuda.synchronize()` or CUDA events, fixed shapes, percentiles over many runs, and prefill and decode profiled separately because they use different shapes and kernels.

**Key points to mention:**
- Nsight Systems patterns: white gaps (launch- or CPU-bound), `cudaStreamSynchronize` every step, large host-to-device copies (CPU offload), alternating dequantize and GEMM kernels.
- `py-spy` when the Python process is suspected; NVTX ranges to label phases.
- Profilers add overhead (Nsight Compute replays kernels), so profile on a canary or offline replica with production-like lengths and concurrency.
- Keep always-on low-cost signals (TTFT, TPOT, queue time, batch size, DCGM) on dashboards.

**Deep dive:** [Debugging Case: Quantized LLM](../11-system-design/06-debugging-case-quantized-llm-not-faster.md#9-how-to-profile-inference-top-down)

</details>

<details>
<summary><b>Q55. Would batching, speculative decoding, or a smaller model help?</b></summary>

**Answer:** It depends on the diagnosis. Batching raises throughput and cuts queueing under load but does not make a lone request faster; quantization frees memory for more KV, so check that concurrency limits were raised to use it. Speculative decoding helps latency in memory-bound, low-batch decode with a good acceptance rate: expected tokens per target pass are (1 - alpha^(k+1)) / (1 - alpha), so alpha = 0.8 and k = 4 gives 3.36 tokens, about 2.4x ideal speedup if the draft costs 10% of the target, but gains shrink at high batch. A smaller or distilled model is the only option that cuts bytes, FLOPs and KV together, so it also helps prefill-bound workloads (3B vs 7B cuts about 14 to about 6 GFLOPs per token), at a quality cost that must be evaluated.

**Key points to mention:**
- Decision rule: idle-system TPOT too high and decode memory-bound means speculative decoding (after fixing kernels); p95 under load dominated by queue time means batching and concurrency; prefill-bound or everything slow means a smaller model.
- The draft model normally needs the target's tokenizer; acceptance drops with higher temperature and unpredictable text.
- Fewer output tokens is often the cheapest lever; FP8 W8A8 on H100 helps compute-bound phases.
- Evaluate a smaller model on the hardest slices separately.

**Deep dive:** [Debugging Case: Quantized LLM](../11-system-design/06-debugging-case-quantized-llm-not-faster.md#10-remedies-matched-to-the-diagnosis), [Speculative Decoding, Caching and Other Speedups](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md)

</details>

## 11. Debugging: a RAG app gives fluent but unsupported answers

Core: Q56-Q61.

<details>
<summary><b>Q56. A RAG application gives fluent but unsupported answers. Diagnose it.</b></summary>

**Answer:** I clarify what "unsupported" means and whether it started after a change (if so, diff deploys first: model version, prompt, embedding model, index rebuild). Then I reproduce and collect 30-100 failing examples with full traces and walk the pipeline in order, because each stage can only use what the previous one gave it: inspect the retrieved evidence, measure retrieval recall on a labeled set, check parsing, chunking, embeddings, filters and freshness, and only then examine generation with an oracle-context test. The decisive split is retrieval failure (the gold chunk never reached the prompt) versus generation failure (it was in the prompt and the answer is still unsupported). I bucket failures by root cause, fix the biggest bucket, add claim-level citations with verification and abstention, prove the gain with faithfulness before and after, and lock it in with CI evals and online monitoring.

**Key points to mention:**
- Typical findings: right chunk ranked too low or cut by the token budget, tables lost in parsing, stale and current docs both indexed, the model adding facts from parametric memory.
- Measure on a frozen labeled set and index snapshot: Recall@5 and @50, faithfulness, fully supported rate, correctness, abstention, p95 latency, per slice.
- Pitfall: answering "use a bigger model" or "tune the prompt" first; it hides the root cause.

**Deep dive:** [Debugging Case: A RAG Application Gives Fluent but Unsupported Answers](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md), [Debugging RAG Systems](../08-rag/11-debugging-rag-systems.md)

</details>

<details>
<summary><b>Q57. How would you inspect retrieved evidence?</b></summary>

**Answer:** For each failing query I replay the trace and walk a ladder: is the fact in the original document, in the parsed text, inside a single chunk, in the top-50 candidates, in the final top-k, and in the actual prompt? The first "no" names the broken stage, and automating the ladder over the whole failure set gives a histogram of verdicts. I log and inspect deep (top-50 with scores and ranks), not just the top-5 the model saw, along with the raw and rewritten query, filters, the exact final prompt, model and prompt versions and the index snapshot. Rank matters: gold at rank 3 when only 2 chunks fit in the budget is a context assembly bug, not a retrieval bug.

**Key points to mention:**
- Two kinds of truncation: a chunk boundary splitting the fact, and the token budget dropping a chunk.
- Look at what was retrieved instead: a stale or generic chunk outranking the right one points to missing metadata or ranking issues.
- In production, always log IDs, scores and versions; store full text only on an access-controlled, retention-limited sample.

**Deep dive:** [Debugging Case: RAG Unsupported Answers](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md#step-2-inspect-the-retrieved-evidence-for-each-failure)

</details>

<details>
<summary><b>Q58. How would you measure retrieval recall?</b></summary>

**Answer:** Build a labeled set of real, stratified queries with gold chunk or document IDs (plus unanswerable queries), run the retriever against a pinned index snapshot, and compute Recall@k = |relevant items in the top k| / |relevant items| averaged over queries, plus Hit@k and MRR. Report it at several k (for example 5 and 50) and at several stages (first stage, after reranking, final prompt), broken down by slice. In the chapter's example, four queries with gold at ranks 2, (1 and 31), 14 and 5 give Recall@5 = 0.625 but Recall@50 = 1.0: a ranking problem a reranker can fix, not a first-stage problem. Synthetic queries alone overestimate recall because they reuse document wording, so include expert-labeled production samples.

**Key points to mention:**
- Store gold as document ID plus character span so labels survive re-chunking.
- Label efficiently with pooling: label only the union of top results from several retrievers.
- With a few hundred queries small differences may be noise: use paired tests.
- Version the set and pin the index snapshot used for each measurement.

**Deep dive:** [Debugging Case: RAG Unsupported Answers](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md#step-3-measure-retrieval-recall-with-a-labeled-set), [RAG Evaluation Metrics and RAGAS](../08-rag/10-rag-evaluation-metrics-and-ragas.md)

</details>

<details>
<summary><b>Q59. Would you change chunking, embeddings, reranking, or generation?</b></summary>

**Answer:** Whichever the evidence points to; I do not decide before bucketing the failures. A fact split across chunks or headerless chunks means chunking; gold not in the top-50, especially for codes or jargon that BM25 finds, means embeddings or hybrid search; gold in the top-50 but not the top-5 (high Recall@50, low Recall@5) means reranking; gold in the prompt but the answer unsupported, confirmed by an oracle test, means generation. Content gaps are none of these and need abstention plus a report to content owners. I fix the biggest bucket first, one change at a time, and re-measure, keeping cost in mind: prompt and context assembly are cheapest, a reranker adds latency but no re-index, chunking needs re-ingestion, a new embedding model needs a full re-embed and migration, and fine-tuning the generator is the most expensive.

**Key points to mention:**
- Signals: Recall@50 low means first-stage retrieval (embeddings, hybrid, query rewriting, filters); recall fine and faithfulness low means generation.
- Changing several things at once makes attribution impossible.
- Re-run the full metric set after each change, including latency and abstention.

**Deep dive:** [Debugging Case: RAG Unsupported Answers](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md#step-6-decide-what-to-change-based-on-evidence)

</details>

<details>
<summary><b>Q60. How would you implement citations and abstention?</b></summary>

**Answer:** Give each chunk a stable ID in the prompt and require a structured answer where every claim lists its citation IDs and a verbatim quote. Then verify in code and with a model: IDs must exist in the context, quotes must appear in the cited chunk, and an NLI model or LLM judge must confirm the cited chunk entails the claim. Unsupported non-core claims are dropped or regenerated; if the core claim is unsupported or retrieval confidence is low, the system abstains with a helpful message. Abstention has three gates (before generation on reranker score and metadata, during generation via a fixed refusal instruction, after generation via the verifier), with thresholds calibrated on a labeled set of answerable and unanswerable queries; illustratively, abstaining on 15 of 100 queries cut unsupported answers shown from 22 to 10 at the cost of 3 false abstentions.

**Key points to mention:**
- Never ship quote matching alone: a correct quote with a wrong claim ("up to 12 months unpaid" restated as "12 months paid") only fails the entailment check.
- NLI limits: input length (often 512 tokens), multi-source claims, arithmetic; escalate borderline cases to an LLM judge.
- Measure citation precision and recall, false abstention rate and false answer rate on unanswerable queries.
- UX: link citations to highlighted spans; abstentions should suggest next steps.

**Deep dive:** [Debugging Case: RAG Unsupported Answers](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md#step-7-claim-level-citations-and-verification), [Abstention](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md#step-8-abstention-when-evidence-is-insufficient)

</details>

<details>
<summary><b>Q61. How would you prevent regression?</b></summary>

**Answer:** A versioned golden set with slices (answerable, tables, superseded policies, multi-hop, unanswerable, past incidents) runs in CI on every change to the prompt, model, embeddings, chunker, reranker, thresholds or judge, with per-slice thresholds against a stored baseline and per-query diffs. Tests are layered: deterministic unit tests, fast retrieval-only tests, then slower end-to-end LLM evals, with tolerances set from measured run-to-run noise and model and judge versions pinned. In production, sampled traces go through the verifier, and I monitor unsupported-claim rate, abstention rate, retrieval score distributions and freshness, review flagged answers weekly, and turn every confirmed failure into a permanent test. Risky changes still go through canary or shadow traffic.

**Key points to mention:**
- Test the tests: reintroduce a known bug (revert the table parser, shrink the token budget) and confirm the suite fails.
- Guardrail metrics alongside faithfulness (coverage, relevance, latency), since over-abstention can raise faithfulness while users complain more.
- Never use silently moving model aliases in either the app or the judge.

**Deep dive:** [Debugging Case: RAG Unsupported Answers](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md#step-10-prevent-regressions-with-ci-and-online-monitoring)

</details>

## 12. Debugging: rising p95 latency and memory usage

Core: Q62-Q67. Additional: Q68-Q69.

<details>
<summary><b>Q62. A production inference service has rising p95 latency and memory usage. How would you debug it?</b></summary>

**Answer:** First mitigate if users are hurting: roll back if it lines up with a deploy, scale out, or use rolling restarts as a stopgap, capturing heap snapshots and traces before restarting. Then clarify the symptom: gradual or sudden, correlated with deploys or traffic, which endpoints and tenants, host or GPU memory, and whether restarts reset it. Decompose latency into stages with metrics and traces to see whether time goes to waiting (queueing) or to work (prefill, decode, gateway). Form explicit hypotheses (capacity shortfall from longer prompts, a leak, KV not freed for cancelled requests, a retry storm, GC pauses), reproduce them with replayed load and a soak test, fix, and verify with identical before and after load tests and a long canary.

**Key points to mention:**
- The two symptoms can share a cause: an unbounded queue holds requests in memory; a leaking heap lengthens GC pauses.
- Look for feedback loops: timeouts cause retries cause more load; cancellations cause zombie generations and less capacity.
- First four graphs: p50 vs p95 over 30 days with deploy markers, per-stage latency share, prompt-token p95 over time, RSS and KV usage per replica against cumulative request count.
- Pitfall: adding GPUs without fixing the leak or retry policy leaves the system fragile.

**Deep dive:** [Debugging Case: Rising p95 Latency and Memory Usage](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md)

</details>

<details>
<summary><b>Q63. What metrics and traces would you collect?</b></summary>

**Answer:** End-to-end and per-stage latency percentiles (gateway, retrieval, tokenization, server queue, prefill, decode, post-processing) plus TTFT and TPOT; queue depth and queue wait; batch size distribution, running versus waiting sequences and preemptions; concurrency (in-flight requests, thread and connection pool usage, event-loop lag); GPU memory allocated versus reserved and KV cache usage; host CPU, RSS and GC pause time. Request shape is the most often missed: prompt and output token length distributions, because capacity is in tokens and longer prompts cut capacity while request rate looks flat. Add error, timeout, retry and cancellation rates, cache hit rates and deploy and config events, plus traces with tail-based sampling of the slowest requests.

**Key points to mention:**
- Histograms with buckets covering the slow range, merged across replicas for fleet percentiles.
- Group the slowest traces by attributes (prompt length bucket, tenant, replica, retry flag) to find what they share.
- Little's law on your own metrics reveals unmeasured waiting (for example in a load balancer or thread pool).
- High-cardinality fields (user, session) belong in traces and logs, not metric labels.

**Deep dive:** [Debugging Case: Rising p95 Latency](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#step-2-collect-the-right-metrics-and-traces)

</details>

<details>
<summary><b>Q64. How would you identify queue buildup and memory leaks?</b></summary>

**Answer:** Queue buildup: queue depth and queue wait rise while service time stays roughly flat, throughput plateaus as arrivals keep rising, and latency tracks traffic. I compare arrival rate lambda with capacity mu = c / S: two replicas with 16 slots each and 2.5 s service time give mu = 12.8 req/s, so at lambda = 14 the queue grows by 1.2 req/s, reaching 720 queued after 10 minutes with a new request waiting about 56 s. Memory leaks: memory grows linearly with request count at constant load, never plateaus and resets on restart (a sawtooth). I fit a slope from a soak test, convert it to bytes per request, then use `tracemalloc` snapshot diffs on the host or PyTorch memory stats and snapshots on the GPU to find the allocation site.

**Key points to mention:**
- Queueing explodes before 100% utilization: in M/M/1, W = (1/mu) / (1 - rho), so going from 90% to 99% busy is 10x worse.
- Queue grows while the GPU is idle: the bottleneck is CPU-side (event loop, thread pool, tokenization) or a scheduler limit.
- Leak sources: unbounded caches, tensors kept alive by references, KV not freed for cancelled requests, CUDA fragmentation (reserved much larger than allocated), native malloc fragmentation (RSS grows while `tracemalloc` is flat).
- In paged-attention servers GPU memory is flat by design; a leak shows as KV usage high relative to running requests.

**Deep dive:** [Debugging Case: Rising p95 Latency](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#step-3-identify-queue-buildup-with-littles-law), [Identify memory leaks](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#step-4-identify-memory-leaks-host-and-gpu)

</details>

<details>
<summary><b>Q65. How would you investigate batching and concurrency?</b></summary>

**Answer:** On the GPU side I look at the batch size distribution over time, running versus waiting sequences, preemptions, TPOT versus batch size, and whether long prefills stall decodes (head-of-line blocking). On the CPU side I check event-loop lag, thread pool and connection pool saturation, and blocking calls inside async handlers; thread pools obey Little's law too, so 40 threads each held for 5 s serve at most 8 req/s. I also check retries per user action and whether cancelled requests actually stop generating. Then I run a concurrency sweep to find the throughput knee and set limits there, since bigger batches trade latency for throughput (with an illustrative cost of 20 + 2B ms, B = 8 gives about 222 req/s and B = 64 gives about 432 req/s at 148 ms per batch).

**Key points to mention:**
- Chunked prefill reduces head-of-line blocking from long prompts.
- Retry amplification: 3 layers x 3 attempts can mean 27 attempts at the bottom.
- Admission control with bounded queues and load shedding stops overload from becoming unbounded latency and memory.

**Deep dive:** [Debugging Case: Rising p95 Latency](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#step-5-investigate-batching-and-concurrency)

</details>

<details>
<summary><b>Q66. How would you reproduce the issue through load testing?</b></summary>

**Answer:** Build a load test that replays production traffic shapes: real prompt and output length distributions (old and new), bursty peak arrival rates, endpoint and tenant mix, streaming with realistic cancellation rates and many distinct sessions. Use an open-loop generator that measures from the intended send time, because closed-loop generators slow down with the server and hide queueing (coordinated omission). Run a step load to find the knee and confirm the capacity hypothesis: with the new prompt distribution the knee should move left as mu = c / S predicts. Then run a multi-hour soak at constant load, discard warm-up, fit the memory slope and include a quiet period to see whether memory comes back down.

**Key points to mention:**
- No cancellations in the test means you never reproduce KV-not-freed bugs; few session IDs means no session-cache leaks.
- Match the production path (gateway, retrieval, GPU type, server flags) or document the differences.
- If the lab looks fine but production is slow, compare request-shape histograms and the measurement point, then use canaries and shadow traffic.

**Deep dive:** [Debugging Case: Rising p95 Latency](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#step-6-reproduce-the-problem-with-load-testing)

</details>

<details>
<summary><b>Q67. How would you verify the fix?</b></summary>

**Answer:** Write the pass criteria first, for example p95 under the SLO at the replayed peak rate, a post-warm-up memory slope whose interval includes zero, error and shed rates no worse, and eval scores unchanged. Run the identical replayed load before and after, on the same hardware and configuration except the change, several times each, and compare p50, p95 and p99, TTFT and TPOT, throughput, and error and shed rates with an interval on the p95 difference across independent runs. Run a long soak to confirm memory is flat, then canary in production against a concurrent baseline for a full daily cycle with predefined rollback criteria. Finally add alerts on leading indicators and a performance regression test so the problem cannot silently return.

**Key points to mention:**
- Comparing with last week is invalid: traffic, prompt mix and time of day differ.
- Bootstrap over requests within one run is overconfident because requests share queue state.
- Watch for shifted problems: load shedding lowers p95 but raises 429s; trimming context may cut quality.
- Verify each root cause separately (one change per run or a flag per fix).

**Deep dive:** [Debugging Case: Rising p95 Latency](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#step-7-fix-it-and-prove-the-fix-worked)

</details>

<details>
<summary><b>Q68. Why does p95 often rise much more than p50 when a system approaches capacity?</b></summary>

**Answer:** Waiting time grows nonlinearly with utilization, and bursts hit only some requests. In an M/M/1 model, average time in system is W = (1/mu) / (1 - rho): with 0.1 s service time it is 0.2 s at 50% busy, 1 s at 90% and 10 s at 99%. In multi-slot servers (continuous batching with many sequence slots), most requests at moderate load find a free slot and wait zero, so the median stays near the service time, while the minority arriving during bursts wait and form the tail. That is why latency-sensitive services keep peak utilization well below 100% (for example 60-70%) and shed load beyond it.

**Key points to mention:**
- Other tail-only causes: GC pauses, head-of-line blocking by long prompts, one slow replica, cold starts during autoscaling.
- The p95 is a few times the mean, so the same multiplicative factor adds many more seconds to it.
- Tail growth with a flat median is a classic early sign of queueing.

**Deep dive:** [Debugging Case: Rising p95 Latency](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#step-3-identify-queue-buildup-with-littles-law)

</details>

<details>
<summary><b>Q69. How do retries make an incident worse, and how do you design retries safely?</b></summary>

**Answer:** When a service is slow, timeouts trigger retries, which add load, which makes it slower: a positive feedback loop. Timed-out requests may also keep running on the server, so the same work is done twice, and with k layers each making a attempts the bottom can see up to a^k attempts (3 layers x 3 attempts is 27). Safe design: retry at one layer only, with exponential backoff and jitter, under a retry budget (for example at most 10% extra requests), only for idempotent and retryable errors, with deadline propagation so servers drop expired work and circuit breakers that fail fast. For LLM streaming, make sure the original generation is aborted when the client gives up.

**Key points to mention:**
- Load shedding with Retry-After tells clients to back off instead of hammering.
- Check SDK and HTTP client default retries; they stack on top of yours.
- Track the retry ratio (retries / first attempts) as an early warning signal.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#retries-backoff-jitter-idempotency-and-retry-budgets), [Debugging Case: Rising p95 Latency](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md)

</details>

## Rapid-fire round

| Question | Answer |
| --- | --- |
| What five things define a model's behavior? | Code, data, config, environment and random seed. |
| Tracker vs registry in one line? | The tracker records every run tried; the registry holds the curated versions you might ship, with governance. |
| Why aliases instead of stages in a registry? | Aliases (`champion`, `challenger`) are flexible pointers; MLflow deprecated fixed stages in 2.9. |
| DVC vs lakeFS vs Delta/Iceberg? | DVC for repo-level batch datasets, lakeFS for branchable data lakes, Delta/Iceberg for large tables with SQL time travel. |
| Three levels of reproducibility? | Lineage (always), statistical across seeds (for comparisons), bitwise determinism (targeted, CI and debugging). |
| What is training-serving skew? | Features or preprocessing at serving differ from training, for example `NaN` vs `0` or a different chat template. |
| Strongest guarantee against skew? | Log the exact served features and train future models on those logs. |
| Point-in-time join in one line? | Attach the latest feature value available at or before the label time, never a later one. |
| How do you route a sticky canary? | `hash(salt + user_id) mod 100 < k`, stepping k from 1 to 100 with bake times. |
| What can rollback not undo? | Predictions already acted on, schema changes and downstream models trained on new outputs. |
| How do you keep rollback possible with a new feature? | Expand-contract: produce both old and new features until the soak window ends. |
| Shadow, canary or A/B? | Shadow for zero-risk observation, canary to limit blast radius, A/B for causal business impact. |
| What does McNemar's test use? | Only the examples where the two models disagree (fixes vs breaks). |
| What is sample ratio mismatch? | Arms differ from the planned split (10,200 vs 9,800), signaling broken assignment. |
| Little's law? | L = lambda x W: concurrency equals arrival rate times time in system. |
| KV bytes per token, 7B MHA FP16? | 2 x 32 layers x 32 heads x 128 dims x 2 bytes = 524,288 bytes (0.5 MiB). |
| 429 vs 503? | 429: this client exceeded its own quota; 503: the service as a whole is overloaded or not ready. |
| Three circuit breaker states? | Closed (counting failures), open (fail fast), half-open (one trial call). |
| Why not autoscale a GPU service on CPU? | CPU can idle while the GPU saturates and the queue explodes; scale on queue depth or waiting requests. |
| Why is nvidia-smi utilization misleading? | It only says a kernel was running; use DCGM SM, tensor and DRAM activity. |
| How do you get fleet p95? | `histogram_quantile` over summed histogram buckets, never an average of per-replica p95s. |
| What is goodput? | Throughput counting only requests that met their latency SLO. |
| Error budget of 99.9% over 30 days? | 0.1%, about 43.2 minutes of full outage. |
| Burn rate 14.4 for one hour consumes? | 2% of the 30-day budget (14.4 / 720), a common paging threshold. |
| PSI of 0.22 means? | A moderate distribution shift worth investigating, not automatic retraining. |
| CI vs CD vs CT? | CI tests changes, CD ships validated artifacts, CT retrains automatically on new data or triggers. |
| What is an LLM gateway? | A proxy giving one API with auth, quotas, budgets, caching, routing, fallbacks and cost logging. |
| How is LLMOps different from MLOps? | You change prompts, retrieval and tools; outputs are free text; cost is per token; debugging is per trace. |
| Why rate limit an LLM platform by tokens? | GPU cost scales with tokens, and requests differ in size by orders of magnitude. |
| Pre-filter or post-filter ACLs in RAG? | Pre-filter: post-filtering leaks into intermediate systems and starves narrow-access users. |
| How do you delete a document from RAG? | Tombstone immediately (unretrievable in seconds), then purge everywhere and audit. |
| Can you mix embedding model versions in one index? | No: the vector spaces are incompatible; re-embed into a new index and flip an alias. |
| Prefill vs decode bottleneck? | Prefill is compute-bound; small-batch decode is memory-bandwidth-bound. |
| Why can W4A16 be no faster? | It still computes in FP16 and, without fused kernels, dequantizing moves about 4.5 bytes per weight vs 2. |
| What is MBU? | Achieved decode bytes per second (weights plus KV per step / TPOT) over peak memory bandwidth. |
| Retrieval or generation failure: the decisive test? | Is the gold chunk in the final prompt, then the oracle-context test. |
| Recall@50 high, Recall@5 low means? | A ranking problem: add or improve the reranker. |
| Why not ship quote matching alone for citations? | A correct quote can back a wrong claim; only an entailment check catches it. |
| Leak signature in a soak test? | Memory grows linearly with request count at constant load and resets on restart. |
| What is coordinated omission? | A closed-loop load generator slows with the server and hides queueing delay. |

## How to practice

1. **Answer out loud first.** Read only the question, then speak your answer before opening it. Silent "I know this" is not the same as a fluent spoken answer.
2. **Time yourself to 2 minutes.** Structure each answer as definition, mechanism, trade-off, one concrete number or example. For the core design and debugging questions, practice a 2-minute summary and a 10-minute version.
3. **Explain it to a non-expert.** If you cannot explain canary rollouts, burn rates or KV cache arithmetic in plain words, you do not own the idea yet.
4. **Then go deeper.** Open the answer, check which key points you missed, and practice the follow-ups. Interviewers grade senior candidates on trade-offs, failure modes and how you would measure success.
5. **Run mock rounds.** As in the reference guide, pick one system-design problem (Section 8 or 9) and one debugging problem (Sections 10-12) and run them back to back under time pressure.
6. **Use the rapid-fire table daily** in the last week: cover the answer column, say each answer, and revisit the chapters behind any you miss.
