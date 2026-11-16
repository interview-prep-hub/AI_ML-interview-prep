# Glossary

> **What this is:** an alphabetical list of the terms, acronyms and infrastructure words used across the course, each explained in one or two plain-English sentences with a link to the chapter that teaches it properly.
>
> **How to use it:** when a chapter (or an interviewer) uses a word you have not met yet, look it up here, read the short definition, and follow the link only if you need the full story. The [Commonly confused pairs](#commonly-confused-pairs) table and the interview questions at the end cover the distinctions interviewers like to probe.

## Table of contents

- [How to read an entry](#how-to-read-an-entry)
- [Commonly confused pairs](#commonly-confused-pairs)
- Terms: [A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [V](#v) · [W](#w) · [Y](#y) · [Z](#z)
- [Interview questions and model answers](#interview-questions-and-model-answers)
- [Key takeaways](#key-takeaways)

## How to read an entry

> **Analogy:** this page is the index at the back of a textbook. It does not teach the topic; it tells you what a word means in a sentence and which page to turn to.

The course defines every term where it first matters, but the [Fast track](../README.md#fast-track-about-4-weeks-2-to-3-hours-per-day) asks you to skim some chapters, so you will meet words such as *HBM*, *goodput* or *tombstone* before the chapter that explains them. Each entry has:

- **The term**, with the acronym first and its expansion in parentheses when the acronym is what people say out loud (for example "GQA (grouped-query attention)").
- **One or two sentences** of plain-English meaning. Where an entry gives a formula, it also gives a small worked example.
- **See also:** related or easily confused terms on this page.
- **Defined in:** the chapter and section that teach the idea in depth. **Used in:** appears instead for infrastructure words (Kubernetes, gRPC, Slurm and similar) that the course uses without a dedicated section.

Chapters are referred to as part/chapter, as in the README: 08/13 means Part 08, chapter 13. Entries are sorted alphabetically word by word, ignoring capitals, punctuation and anything in parentheses, so "LLM" comes before "LLM gateway" and "A/B test" before "A2A".

## Commonly confused pairs

Interviewers often ask "what is the difference between X and Y?" for these pairs. Say the one-line difference, then give the test that tells them apart.

| Pair | The difference in one line | How to tell them apart |
|---|---|---|
| [Latency](#latency) vs [throughput](#throughput) | Latency is how long one request takes; throughput is how much work the whole system finishes per second | Bigger batches usually raise throughput and raise latency too; judge the trade-off with [goodput](#goodput) |
| [Data drift](#data-drift-covariate-shift) vs [concept drift](#concept-drift) | Data drift: the inputs $P(X)$ change. Concept drift: the input-to-label rule $P(Y \mid X)$ changes | Data drift is visible without labels; concept drift needs labels or proxies for them |
| [Data contamination](#data-contamination) vs [benchmark leakage](#benchmark-leakage) | Contamination: evaluation data overlaps training data. Leakage: a benchmark's test information shaped how a model was trained or chosen | Tuning prompts or checkpoints on the test set leaks with zero overlap |
| [PTQ](#ptq-post-training-quantization) vs [QAT](#qat-quantization-aware-training) | PTQ quantizes a finished model using a calibration set; QAT trains with simulated quantization so the weights adapt | PTQ needs no training run; try it first and use QAT when PTQ loses too much accuracy |
| [Bi-encoder](#bi-encoder) vs [cross-encoder](#cross-encoder) | A bi-encoder embeds query and document separately; a cross-encoder reads them together and outputs one score | Bi-encoders search millions of precomputed vectors; cross-encoders rerank a shortlist |
| [MQA](#mqa-multi-query-attention) vs [GQA](#gqa-grouped-query-attention) vs [MLA](#mla-multi-head-latent-attention) | All shrink the KV cache: MQA keeps one key/value head, GQA one per group of query heads, MLA one compressed latent per token | With 32 query heads: MHA caches 32 key/value heads, GQA-8 caches 8, MQA caches 1 |
| [Authentication](#authentication-authn) vs [authorization](#authorization-authz) | Who you are vs what you may do or see | A logged-in user (authenticated) can still be denied a document (not authorized) |
| [Liveness probe](#liveness-probe) vs [readiness probe](#readiness-probe) | Failing liveness restarts the container; failing readiness only stops traffic to it | An LLM server that is still loading weights is alive but not ready |
| [Prefill](#prefill) vs [decode](#decode) | Prefill processes the whole prompt in one parallel pass; decode produces one token per step | Prefill is compute-bound and drives TTFT; small-batch decode is memory-bound and drives TPOT |
| [TTFT](#ttft-time-to-first-token) vs [TPOT](#tpot-time-per-output-token) | The wait before the first token vs the average gap between later tokens | Long prompts and queueing hurt TTFT; long outputs and big batches hurt total time through TPOT |
| [Prompt injection](#prompt-injection) vs [jailbreak](#jailbreak) | Injection hijacks the application's instructions; a jailbreak defeats the model's own safety training | Injection can come from a document or tool result the user never wrote |
| [Precision](#precision-and-recall) vs [recall](#precision-and-recall) | Precision: share of returned items that are relevant. Recall: share of relevant items that were returned | First-stage retrieval optimizes recall; reranking optimizes precision |

## A

### A/B test

Randomly split live users between the current system (A, control) and a candidate (B, treatment) for a fixed period, then compare a business metric with a significance test. It measures causal impact on users, unlike a shadow deployment, which only observes. **See also:** [Canary release](#canary-release), [Shadow deployment](#shadow-deployment). **Defined in:** [00/04 Probability and Statistics](../00-foundations/04-probability-and-statistics.md#ab-tests-and-comparing-two-models), [10/04 Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#ab-testing-with-statistical-significance).

### A2A (Agent2Agent protocol)

An open protocol that lets one agent hand tasks to another agent that has its own reasoning and tools, exchanging task messages and results (artifacts), originally over JSON-RPC 2.0 on HTTPS (later spec versions also define gRPC and REST bindings). MCP connects an agent to tools and data; A2A connects an agent to other agents. **See also:** [MCP](#mcp-model-context-protocol), [JSON-RPC 2.0](#json-rpc-20). **Defined in:** [09/05 Multi-Agent Systems](../09-agents/05-multi-agent-systems.md#inter-agent-protocols-a2a-in-brief).

### ABAC (attribute-based access control)

Access is decided by rules over attributes of the user, the resource and the context, for example "department = finance AND region = EU". More flexible than roles, but harder to audit. **See also:** [RBAC](#rbac-role-based-access-control), [ReBAC](#rebac-relationship-based-access-control), [ACL](#acl-access-control-list). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

### Abstention

The system deliberately says "I do not know" or asks a clarifying question when its evidence is insufficient, instead of guessing. In RAG it is triggered by weak retrieval or a failed support check on the draft answer. **See also:** [Hallucination](#hallucination). **Defined in:** [05/06 Hallucinations](../05-llms/06-hallucinations-and-grounding.md#8-abstention-when-to-say-i-do-not-know).

### ACL (access control list)

The list of principals allowed to access a resource, stored with that resource, for example "salary bands: allowed = `group:acme/hr`". In RAG every chunk inherits its document's ACL and retrieval filters on it. **See also:** [Principal](#principal), [RBAC](#rbac-role-based-access-control). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

### Adam and AdamW

Adam is an optimizer that combines momentum (a running average of gradients) with a per-parameter step size (from a running average of squared gradients), so it keeps two extra numbers per parameter. AdamW applies weight decay directly to the weights instead of through the gradient and is the default for training transformers. **See also:** [SGD](#sgd-stochastic-gradient-descent). **Defined in:** [02/03 Optimizers](../02-deep-learning/03-optimizers-sgd-momentum-adam-and-learning-rate-schedules.md#adam-momentum-plus-adaptive-rates).

### Admission control

A scheduler rule that lets a waiting request start only if the resources it will need (for an LLM server, enough KV cache blocks) are available, and otherwise queues or rejects it. Rejecting fast beats accepting work that will miss its SLO or crash the server. **See also:** [Backpressure](#backpressure), [Preemption](#preemption). **Defined in:** [06/04 Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md#admission-control).

### Agent (AI agent)

An LLM-driven system that decides its own next step in a loop: pick an action (usually a tool call), observe the result, and repeat until the goal is reached or a limit stops it. A **workflow**, by contrast, follows code paths fixed in advance and uses the LLM only inside steps. **See also:** [ReAct](#react), [Function calling](#function-calling-tool-calling). **Defined in:** [09/01 What Is an AI Agent](../09-agents/01-what-is-an-ai-agent.md#a-precise-definition-of-an-ai-agent).

### AI, ML, DL and GenAI

Artificial intelligence (AI) is the broad goal of machines doing tasks that seem to need intelligence; machine learning (ML) is the part of AI that learns patterns from data instead of hand-written rules; deep learning (DL) is ML with many-layer neural networks; generative AI (GenAI) is models that produce new content such as text, images or code. Each is a subset of the one before in the usual nested picture (strictly, generative models also exist outside deep learning, but modern GenAI is mostly deep learning). **Defined in:** [00/01 What AI, ML, DL and GenAI Are](../00-foundations/01-what-is-ai-ml-dl-and-genai.md#the-big-picture-four-nested-circles).

### ALiBi (Attention with Linear Biases)

A way to give a transformer position information without position embeddings: each attention score is reduced by a penalty proportional to the distance between the two tokens, with a different slope per head. It handles inputs longer than training better than learned positions. **See also:** [RoPE](#rope-rotary-position-embedding). **Defined in:** [04/06 Positional Encoding](../04-transformers/06-positional-encoding-sinusoidal-learned-rope-and-alibi.md#alibi-attention-with-linear-biases).

### All-gather

A collective operation in which each rank holds one slice of a tensor and every rank ends with the full concatenation. ZeRO-3 and FSDP use it to fetch a layer's full weights just before computing with them. **See also:** [All-reduce](#all-reduce), [Reduce-scatter](#reduce-scatter). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### All-reduce

A collective operation in which every rank contributes a tensor and every rank ends with the element-wise sum (or average); data-parallel training uses it to synchronize gradients after each backward pass. With ring all-reduce each rank sends about $2 \frac{N-1}{N} M$ bytes for a tensor of $M$ bytes on $N$ GPUs: for a 1 GB gradient on 4 GPUs that is $2 \times \frac{3}{4} \times 1 = 1.5$ GB, and it never exceeds 2 GB however many GPUs join. **See also:** [Reduce-scatter](#reduce-scatter), [All-gather](#all-gather), [NCCL](#nccl). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#51-ring-all-reduce).

### All-to-all

A collective operation in which each rank sends a different slice to every other rank. Mixture-of-experts layers use it to send each token to the GPU that holds its chosen expert. **See also:** [MoE](#moe-mixture-of-experts). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### ANN (approximate nearest neighbor) search

Finding vectors that are almost certainly among the closest to a query without comparing it with every stored vector, trading a little recall for large speedups. Quality is measured as recall@k against exact search. **See also:** [kNN](#knn-k-nearest-neighbors), [HNSW](#hnsw-hierarchical-navigable-small-world), [IVF](#ivf-inverted-file-index). **Defined in:** [08/06 Vector Search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#why-approximate-nearest-neighbor-search).

### Arithmetic intensity

The number of floating-point operations a computation performs per byte it moves from memory. Below the GPU's ratio of peak FLOP/s to memory bandwidth the work is memory-bound, above it compute-bound; small-batch LLM decode has very low arithmetic intensity. **See also:** [Roofline model](#roofline-model), [Compute-bound vs memory-bound](#compute-bound-vs-memory-bound). **Defined in:** [06/03 GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md#3-arithmetic-intensity-flops-per-byte-moved).

### Attention (scaled dot-product attention)

Each token compares its query vector with every token's key vector, turns the similarity scores into weights with softmax, and outputs the weighted average of the value vectors. It lets every token pull information from every other token in a single step. **See also:** [Queries, keys and values](#queries-keys-and-values-q-k-v), [Multi-head attention](#multi-head-attention-mha). **Defined in:** [04/03 Attention](../04-transformers/03-attention-mechanism-step-by-step.md#scaled-dot-product-attention-the-formula).

### Authentication (AuthN)

Proving who you are: a login, single sign-on or a signed token. **See also:** [Authorization](#authorization-authz). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

### Authorization (AuthZ)

Deciding what an authenticated principal may do or see. In LLM systems it must be enforced by deterministic code outside the model (retrieval filters, tool permission checks), never by asking the model to behave. **See also:** [Authentication](#authentication-authn), [Least privilege](#least-privilege). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization), [09/02 Tool Calling](../09-agents/02-tool-calling-in-depth.md#why-authorization-must-be-enforced-outside-the-llm).

### Autoregressive generation

Generating one token at a time, feeding each new token back in as input for the next step. It is why an LLM's response time grows with the length of its output. **See also:** [Decode](#decode). **Defined in:** [04/05 Full Transformer](../04-transformers/05-the-full-transformer-architecture.md#training-vs-inference-parallel-vs-autoregressive).

### AWQ (Activation-aware Weight Quantization)

A post-training 4-bit weight quantization method that uses activation statistics from calibration data to find the small fraction of weight channels that matter most and protects them by scaling before rounding. It needs no backpropagation. **See also:** [GPTQ](#gptq), [PTQ](#ptq-post-training-quantization). **Defined in:** [07/04 LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md#awq-protecting-salient-weights-with-activation-aware-scaling).

## B

### Backpressure

When a downstream stage is full, the signal "slow down" travels back upstream (to the queue, the API, the client) instead of work piling up until something crashes. Bounded queues, `429` and `503` responses, and admission control are all forms of it. **See also:** [Admission control](#admission-control), [Rate limiting and quotas](#rate-limiting-and-quotas). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#gpu-memory-limits-admission-control-and-backpressure).

### Beam search

A decoding strategy that keeps the $B$ most probable partial sequences at every step instead of only the single best. Useful for translation, rarely used for open-ended chat because it favors bland, repetitive text. **See also:** [Greedy decoding](#greedy-decoding), [Sampling](#sampling-temperature-top-k-and-top-p). **Defined in:** [05/03 Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md#beam-search).

### Benchmark leakage

Information about a specific (usually public) benchmark's test set reaching a model or the way it was built: test items crawled into training data, or checkpoints, prompts and hyperparameters selected by test score. It inflates leaderboard numbers even when no test item was trained on. **See also:** [Data contamination](#data-contamination), its most common cause. **Defined in:** [05/09 Forgetting and Contamination](../05-llms/09-catastrophic-forgetting-contamination-and-memorization.md#training-contamination-vs-benchmark-leakage).

### BERT

Bidirectional Encoder Representations from Transformers (2018): an encoder-only transformer pretrained by predicting masked-out tokens, so each token sees context on both sides. It is the base of many classifiers, embedding models and cross-encoder rerankers. **See also:** [Encoder-only model](#encoder-only-model). **Defined in:** [04/07 Encoder and Decoder Models](../04-transformers/07-encoder-only-encoder-decoder-and-decoder-only-models.md#encoder-only-models-bert-and-masked-language-modeling).

### BF16 (bfloat16)

A 16-bit floating-point format with the same 8 exponent bits (so the same range) as FP32 but only 7 mantissa bits. It rarely overflows, trains without loss scaling, and is the default for LLM training and inference on modern GPUs. **See also:** [FP32, FP16 and FP8](#fp32-fp16-and-fp8), [Mixed-precision training](#mixed-precision-training). **Defined in:** [07/01 Number Formats](../07-model-compression/01-number-formats-and-model-memory.md#bf16-brain-floating-point).

### Bi-encoder

A model that encodes the query and each document separately into one vector each and scores the pair by cosine similarity or dot product. Document vectors are computed once and indexed, so it scales to millions of documents; embedding models used for first-stage retrieval are bi-encoders. **See also:** [Cross-encoder](#cross-encoder), [Embedding model](#embedding-model). **Defined in:** [08/02 Text Embeddings](../08-rag/02-text-embeddings-and-similarity-search.md#the-bi-encoder-architecture), [08/08 Reranking](../08-rag/08-reranking-and-query-transformation.md#bi-encoders-encode-separately-compare-later).

### Blue-green deployment

Two complete production environments: blue serves all traffic while green receives the new version and is tested, then traffic switches over at once, and switching back is the rollback. The same pattern rebuilds a vector index without downtime by flipping an alias. **See also:** [Canary release](#canary-release). **Defined in:** [10/04 Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#blue-green-deployment).

### BM25

The classic keyword ranking function: a document scores higher when it contains the query's rare terms, with diminishing returns for repeated terms and a penalty for long documents. It is strong on exact names, codes and IDs where embeddings struggle. **See also:** [Sparse retrieval](#sparse-lexical-retrieval), [Hybrid search](#hybrid-search). **Defined in:** [08/05 Retrieval Methods](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#bm25-step-by-step).

### BPE (byte-pair encoding)

A subword tokenization algorithm that starts from characters (or bytes) and repeatedly merges the most frequent adjacent pair into a new token until the vocabulary reaches a target size. Byte-level BPE, used by GPT-style models, can encode any text without an "unknown" token. **See also:** [Token and tokenization](#token-and-tokenization). **Defined in:** [04/01 Tokenization](../04-transformers/01-tokenization.md#byte-pair-encoding-bpe-step-by-step).

### Broadcast

A collective operation in which one rank sends its tensor to all others, for example to give every GPU the same initial weights. **See also:** [Collective operation](#collective-operation). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### Burn rate

How fast the error budget is being spent compared with the pace that would use exactly all of it by the end of the SLO window. A burn rate of 14.4 sustained for 1 hour spends $14.4 \times 1 / 720 = 2\%$ of a 30-day (720-hour) budget, a common paging threshold. **See also:** [Error budget](#error-budget), [SLO, SLA and SLI](#slo-sla-and-sli). **Defined in:** [10/06 Monitoring](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md#burn-rate).

## C

### Caching allocator (allocated vs reserved GPU memory)

PyTorch keeps freed GPU memory in its own cache for reuse instead of returning it to the driver. "Allocated" is memory held by live tensors, "reserved" adds the cache, and `nvidia-smi` shows roughly reserved memory plus runtime overhead, so a GPU can look full while being fine. **See also:** [Memory leak](#memory-leak), [OOM kill](#oom-out-of-memory-kill). **Defined in:** [11/08 Rising p95 Latency Case](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#memory-words).

### Calibration (probability)

A model is calibrated when its confidence matches reality: of all predictions made with 80% confidence, about 80% are right. Not to be confused with calibration data for quantization. **See also:** [Calibration data](#calibration-data-quantization). **Defined in:** [01/04 Classification Metrics](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md#probability-calibration).

### Calibration data (quantization)

A small, representative sample of inputs (often a few hundred) run through a model during post-training quantization to measure activation ranges and, for GPTQ and AWQ, to decide how to round or scale weights. If it does not resemble production traffic, quantized accuracy suffers. **See also:** [PTQ](#ptq-post-training-quantization). **Defined in:** [07/03 PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md#calibration-data-the-quiet-make-or-break-input).

### Canary release

Send a small share of real traffic (for example 5%) to the new version, check metric gates, then widen step by step or roll back automatically. It limits the damage a bad release can do. **See also:** [Shadow deployment](#shadow-deployment), [A/B test](#ab-test), [Blue-green deployment](#blue-green-deployment). **Defined in:** [10/04 Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#canary-releases-metric-gates-and-automated-rollback).

### Catastrophic forgetting

A network abruptly loses ability on earlier tasks when it is trained on new data, because the same weights get overwritten. Parameter-efficient fine-tuning, lower learning rates, fewer epochs and mixing in general data reduce it. **See also:** [Fine-tuning](#fine-tuning), [PEFT](#peft-parameter-efficient-fine-tuning). **Defined in:** [05/09 Forgetting and Contamination](../05-llms/09-catastrophic-forgetting-contamination-and-memorization.md#catastrophic-forgetting-what-it-is-and-why-it-happens).

### Causal mask

A mask that lets token $i$ attend only to positions up to $i$, by setting the scores of future positions to minus infinity before softmax. Decoder-only LLMs need it so that training matches left-to-right generation. **See also:** [Decoder-only model](#decoder-only-model). **Defined in:** [04/04 Multi-Head Attention](../04-transformers/04-multi-head-attention-and-causal-masking.md#causal-masking-no-peeking-at-the-future).

### Chain-of-thought (CoT)

Having the model write intermediate reasoning steps before the final answer. Each written token is extra computation the model can build on, which helps multi-step problems at the cost of more output tokens. **See also:** [Reasoning models and GRPO](#grpo-group-relative-policy-optimization). **Defined in:** [05/04 Prompt Engineering](../05-llms/04-prompt-engineering-for-production.md#chain-of-thought-prompting-and-reasoning-models).

### Chat template

The exact rule that flattens a list of messages (system, user, assistant, tool) into one token sequence with the special tokens the model saw during training. Using the wrong template silently degrades quality. **Defined in:** [04/01 Tokenization](../04-transformers/01-tokenization.md#chat-templates).

### Chunk

A short piece of a document (typically a few hundred tokens) that is embedded, indexed and retrieved as one unit. Every chunk inherits its document's metadata, tenant and permissions. **See also:** [Corpus](#corpus), [Metadata](#metadata). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary).

### Chunked prefill

Splitting a long prompt's prefill into pieces and processing one piece per scheduler step alongside other requests' decode steps, under a token budget. It stops one long prompt from freezing token streaming for everyone else. **See also:** [Prefill](#prefill), [Continuous batching](#continuous-batching). **Defined in:** [06/04 Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md#chunked-prefill-mixing-prefill-with-decode).

### CI/CD and CT

Continuous integration (CI) automatically tests every change; continuous delivery or deployment (CD) automatically ships validated artifacts; continuous training (CT), specific to ML, retrains models automatically on new data or on a trigger. **See also:** [MLOps](#mlops). **Defined in:** [10/07 CI/CD for ML](../10-mlops/07-ci-cd-and-testing-for-ml-pipelines.md#ci-cd-and-ct).

### Circuit breaker

A guard around calls to a dependency: after many consecutive failures it opens and fails fast for a cooldown, then lets a trial call through (half-open) before closing again. It stops one failing dependency from tying up every worker. **See also:** [Backpressure](#backpressure). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#circuit-breakers).

### CLIP

Contrastive Language-Image Pretraining (2021): an image encoder and a text encoder trained so that matching image-caption pairs get similar vectors in one shared space. It enables zero-shot image classification and text-to-image search. **See also:** [ViT](#vit-vision-transformer). **Defined in:** [03/05 ViT and CLIP](../03-computer-vision/05-vision-transformers-clip-and-multimodal-models.md#clip-learning-a-shared-space-for-images-and-text).

### CNN (convolutional neural network)

A network that slides small learned filters over an image, reusing the same pattern detector at every position. That built-in assumption of local, position-independent patterns makes CNNs data-efficient for images. **See also:** [ViT](#vit-vision-transformer), [Transfer learning](#transfer-learning). **Defined in:** [02/07 CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md#2-the-convolution-operation).

### Cold start

The delay before a new replica can serve: pulling the container, downloading tens of GB of weights, loading them onto the GPU and warming up can take minutes for an LLM. It is why GPU autoscaling must react early or keep warm spares. **See also:** [HPA](#hpa-horizontal-pod-autoscaler), [Readiness probe](#readiness-probe). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#cold-starts-why-gpu-autoscaling-is-slow).

### Collective operation

A communication pattern in which every rank in a process group takes part, such as broadcast, all-reduce, reduce-scatter, all-gather and all-to-all. **See also:** [NCCL](#nccl), [Rank](#rank-distributed-training). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### Compute-bound vs memory-bound

A workload is compute-bound when the arithmetic units are the bottleneck and memory-bound (memory-bandwidth-bound) when the GPU mostly waits for bytes to arrive from memory. LLM prefill is usually compute-bound while small-batch decode is memory-bound, so decode speeds up from moving fewer bytes (quantization, GQA, batching), not from more FLOPs. **See also:** [Arithmetic intensity](#arithmetic-intensity), [Roofline model](#roofline-model). **Defined in:** [06/01 Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md#why-prefill-is-compute-bound-and-decode-is-memory-bandwidth-bound).

### Concept drift

The relationship between inputs and labels changes: $P(Y \mid X)$ shifts, so the same input now deserves a different answer, as when fraudsters change tactics. Only labels (or proxies for them) reveal it, and it usually requires retraining. **See also:** [Data drift](#data-drift-covariate-shift), where only the inputs move. **Defined in:** [01/08 Data Drift and Concept Drift](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md#concept-drift-and-its-four-shapes).

### Context

In RAG, the retrieved chunks that are placed in the prompt for the model to answer from. **See also:** [Context window](#context-window). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary).

### Context window

The maximum number of tokens (prompt plus generated output) a model can process in one call. Fitting in the window does not mean the model uses all of it well. **See also:** [Token and tokenization](#token-and-tokenization). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary), [04/06 Positional Encoding](../04-transformers/06-positional-encoding-sinusoidal-learned-rope-and-alibi.md#context-window-vs-long-context-use).

### Continuous batching

Also called iteration-level scheduling: the server re-forms the batch at every decode step, so finished requests leave immediately and waiting ones join, keeping the GPU busy. It is a main reason engines such as vLLM reach high throughput. **See also:** [Dynamic batching](#dynamic-batching), [PagedAttention](#pagedattention). **Defined in:** [06/04 Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md#continuous-batching-scheduling-at-every-decode-step).

### Corpus

The whole collection of documents a system searches. **See also:** [Chunk](#chunk). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary).

### Cosine similarity

The cosine of the angle between two vectors: their dot product divided by the product of their lengths, from -1 to 1. For $a = (1, 0)$ and $b = (1, 1)$ it is $1 / (1 \times \sqrt{2}) \approx 0.707$; for unit-length vectors it equals the plain dot product, which is why embedding pipelines normalize. **See also:** [Embedding](#embedding). **Defined in:** [00/02 Linear Algebra](../00-foundations/02-linear-algebra-for-ml.md#similarity-and-distance-cosine-dot-product-and-euclidean).

### Cross-encoder

A model that reads the query and one document together in a single forward pass and outputs a relevance score, so every query word can attend to every document word. It is more accurate than a bi-encoder, but nothing can be precomputed, so it only reranks a shortlist. **See also:** [Bi-encoder](#bi-encoder), [Reranker](#reranker). **Defined in:** [08/08 Reranking](../08-rag/08-reranking-and-query-transformation.md#cross-encoders-read-the-query-and-document-together).

### Cross-entropy

The standard loss for classification and next-token prediction: the negative log of the probability the model gave to the correct answer, averaged over examples. Giving the right token probability 0.9 costs $-\ln 0.9 \approx 0.105$; giving it 0.1 costs $-\ln 0.1 \approx 2.303$. **See also:** [Perplexity](#perplexity), [Softmax](#softmax). **Defined in:** [00/05 Softmax and Cross-Entropy](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md#cross-entropy-and-why-the-log).

### Cross-validation

Splitting the training data into $k$ folds, training $k$ times with each fold held out once, and averaging the scores, for a more stable estimate than a single validation split. **Defined in:** [01/01 ML Workflow and Splits](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md#cross-validation).

### CUDA

NVIDIA's platform for running general-purpose computation on its GPUs; a **kernel** is one function launched on the GPU. **See also:** [HBM](#hbm-high-bandwidth-memory). **Defined in:** [06/03 GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md#11-cpu-vs-gpu-in-one-picture).

## D

### Data contamination

Evaluation data, or something very close to it, was present in the data a model trained on, so the model scores well by recall rather than generalization. It is the LLM-scale form of data leakage and is detected with n-gram or hash overlap checks between training and evaluation sets. **See also:** [Benchmark leakage](#benchmark-leakage), [Data leakage](#data-leakage). **Defined in:** [05/09 Forgetting and Contamination](../05-llms/09-catastrophic-forgetting-contamination-and-memorization.md#data-contamination-and-benchmark-leakage).

### Data drift (covariate shift)

The input distribution $P(X)$ in production moves away from training while the input-to-label relationship stays the same, for example many more emails in Spanish. It is detectable without labels with tests such as PSI or KS, and it may or may not hurt accuracy. **See also:** [Concept drift](#concept-drift), [PSI](#psi-population-stability-index). **Defined in:** [01/08 Data Drift and Concept Drift](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md#covariate-shift-data-drift).

### Data leakage

Information that would not be available at prediction time (the label, future events, test rows) reaching the model or its preprocessing during training, so offline scores look better than production. **See also:** [Data contamination](#data-contamination). **Defined in:** [01/03 Overfitting and Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md#data-leakage-the-silent-killer).

### Data parallelism (DDP)

Every GPU holds a full copy of the model and processes a different slice of each batch, and gradients are averaged with all-reduce after the backward pass; PyTorch's DistributedDataParallel (DDP) is the standard implementation. For serving, data parallelism simply means independent replicas. **See also:** [Tensor parallelism](#tensor-parallelism), [Pipeline parallelism](#pipeline-parallelism), [ZeRO](#zero). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#6-data-parallelism-and-pytorch-ddp).

### Data residency

A legal or contractual requirement that data be stored and processed in a specific country or region. It often forces per-region deployments or a silo per tenant. **See also:** [Silo, bridge and pool](#silo-bridge-and-pool). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#tenant-isolation-options).

### Dead-letter queue (DLQ)

A side queue where jobs that keep failing after their retries are parked for inspection, so one bad message does not block the pipeline. **See also:** [Idempotency and idempotency key](#idempotency-and-idempotency-key). **Defined in:** [08/12 Ingestion Pipelines](../08-rag/12-production-ingestion-pipelines-at-scale.md#dead-letter-queues).

### Decode

The second phase of LLM generation: one new token per forward pass for each sequence, each step reading all the weights and the KV cache. At small batch sizes it is memory-bandwidth-bound, and it sets TPOT. **See also:** [Prefill](#prefill), [TPOT](#tpot-time-per-output-token). **Defined in:** [06/01 Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md#prefill-vs-decode-the-two-phases-of-generation).

### Decoder-only model

A stack of transformer blocks with causal self-attention, trained to predict the next token (GPT, Llama and most modern LLMs). **See also:** [Encoder-only model](#encoder-only-model), [Encoder-decoder model](#encoder-decoder-model), [Causal mask](#causal-mask). **Defined in:** [04/07 Encoder and Decoder Models](../04-transformers/07-encoder-only-encoder-decoder-and-decoder-only-models.md#decoder-only-models-gpt-and-causal-language-modeling).

### Defense in depth

Several independent layers of protection, so that one bug or bypass does not cause a breach. For an LLM app that means authorization in retrieval, least-privilege tools, output validation and audit logs, not one clever prompt. **See also:** [Least privilege](#least-privilege), [Guardrails](#guardrails). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization), [05/13 LLM Safety](../05-llms/13-llm-safety-guardrails-and-responsible-ai.md#defense-in-depth-the-big-picture).

### Dense retrieval

Retrieval by embedding similarity: the query and the chunks are embedded and the nearest vectors are returned. It is good at paraphrases and weaker on exact codes and rare terms. **See also:** [Sparse retrieval](#sparse-lexical-retrieval), [Hybrid search](#hybrid-search). **Defined in:** [08/05 Retrieval Methods](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#dense-retrieval).

### DETR (DEtection TRansformer)

An object detector (2020) that predicts a fixed-size set of boxes directly with a transformer and matches them one-to-one to the ground truth during training, removing hand-designed anchors and non-maximum suppression. **See also:** [YOLO](#yolo-you-only-look-once), [Faster R-CNN](#faster-r-cnn), [NMS](#nms-non-maximum-suppression). **Defined in:** [03/02 Object Detection](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md#detr-detection-as-set-prediction).

### Distillation (knowledge distillation)

Training a small student model to imitate a larger teacher, usually by matching the teacher's softened output probabilities or its generated outputs rather than only the hard labels. It cuts serving cost while keeping much of the quality. **See also:** [Quantization](#quantization), [Pruning](#pruning). **Defined in:** [07/05 Knowledge Distillation](../07-model-compression/05-knowledge-distillation-fundamentals.md#the-big-idea-a-teacher-trains-a-student).

### Document

One source item in a RAG system: a PDF, a web page, a wiki page, a support ticket or a database row. Documents are split into chunks, and each chunk keeps a pointer back to its document for citations, updates and deletions. **See also:** [Chunk](#chunk), [Corpus](#corpus). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary).

### DPO (Direct Preference Optimization)

An alignment method that trains directly on preference pairs (a chosen and a rejected answer) with a simple classification-style loss, without a separate reward model or reinforcement learning loop. It is simpler and usually more stable to train than PPO-based RLHF. **See also:** [RLHF](#rlhf-reinforcement-learning-from-human-feedback). **Defined in:** [05/02 Instruction Tuning, RLHF and DPO](../05-llms/02-instruction-tuning-rlhf-and-dpo.md#direct-preference-optimization-dpo).

### Dynamic batching

Server-side batching with a time window: requests wait in a queue until either a maximum batch size or a maximum wait (a few milliseconds) is reached, then run together. It is standard for classic models; LLM servers go further with continuous batching. **See also:** [Continuous batching](#continuous-batching). **Defined in:** [06/04 Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md#dynamic-batching-wait-a-little-to-batch-more), [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#server-side-dynamic-batching).

## E

### efConstruction and efSearch

The candidate-list sizes of an HNSW index. `efConstruction` is used while building the graph (higher gives a better graph and a slower build, and is fixed afterward); `efSearch` is used at query time (higher gives better recall and slower queries, and can change per query). **See also:** [HNSW](#hnsw-hierarchical-navigable-small-world), [nlist and nprobe](#nlist-and-nprobe). **Defined in:** [08/06 Vector Search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#the-three-knobs-m-efconstruction-efsearch).

### Embedding

A vector of numbers that represents the meaning of a token, text, image or document, so that similar meanings get vectors pointing in similar directions. **See also:** [Embedding model](#embedding-model), [Cosine similarity](#cosine-similarity). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary), [04/02 Embeddings](../04-transformers/02-embeddings-from-one-hot-to-contextual.md#the-embedding-table-a-learned-lookup-matrix).

### Embedding model

A model, usually an encoder trained contrastively, that turns text (or images) into embeddings. Queries and documents must be embedded with the same model and version, because vectors from different models are not comparable. **See also:** [Bi-encoder](#bi-encoder). **Defined in:** [08/02 Text Embeddings](../08-rag/02-text-embeddings-and-similarity-search.md#what-an-embedding-model-does).

### Encoder-decoder model

A transformer with an encoder that reads the full input with bidirectional attention and a decoder that generates the output while attending to the encoder through cross-attention (T5, the original 2017 Transformer). It suits translation and summarization. **See also:** [T5](#t5), [Seq2Seq](#seq2seq-sequence-to-sequence). **Defined in:** [04/07 Encoder and Decoder Models](../04-transformers/07-encoder-only-encoder-decoder-and-decoder-only-models.md#encoder-decoder-models-t5-and-span-corruption).

### Encoder-only model

A transformer stack with bidirectional attention that produces a contextual vector for every input token but does not generate text (BERT). Used for classification, embeddings and rerankers. **See also:** [BERT](#bert), [Decoder-only model](#decoder-only-model). **Defined in:** [04/07 Encoder and Decoder Models](../04-transformers/07-encoder-only-encoder-decoder-and-decoder-only-models.md#encoder-only-models-bert-and-masked-language-modeling).

### Error budget

The amount of failure an SLO allows over its window. A 99.9% availability SLO over 30 days allows $0.001 \times 43{,}200$ minutes $= 43.2$ minutes of full outage; teams spend it on releases and freeze risky changes when it runs out. **See also:** [SLO, SLA and SLI](#slo-sla-and-sli), [Burn rate](#burn-rate). **Defined in:** [10/06 Monitoring](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md#error-budget-math).

## F

### Fail closed

When a security check cannot be completed (a timeout, missing metadata), deny access rather than allow it. **See also:** [Defense in depth](#defense-in-depth). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

### Faithfulness (groundedness)

The share of an answer's claims that are supported by the context the model was given. A faithful answer can still be wrong if the retrieved context was wrong, which is a retrieval failure rather than a generation failure. **See also:** [Hallucination](#hallucination), [RAGAS](#ragas). **Defined in:** [08/10 RAG Evaluation](../08-rag/10-rag-evaluation-metrics-and-ragas.md#faithfulness-groundedness).

### Faster R-CNN

A two-stage object detector: a region proposal network suggests candidate boxes, then a second stage classifies and refines each one. It is usually more accurate on small objects and slower than single-stage detectors such as YOLO. **See also:** [YOLO](#yolo-you-only-look-once), [mAP](#map-mean-average-precision). **Defined in:** [03/02 Object Detection](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md#faster-r-cnn-end-to-end).

### Feature store

A system that computes feature definitions once and serves the same values offline for training (with point-in-time-correct joins) and online for low-latency inference, which prevents training-serving skew. **See also:** [Training-serving skew](#training-serving-skew). **Defined in:** [11/05 Fraud Detection Design](../11-system-design/05-design-a-real-time-fraud-detection-system.md#step-4-the-feature-store-and-onlineoffline-consistency).

### Few-shot prompting

Putting a few worked input-output examples in the prompt so the model copies their format and behavior; zero-shot means no examples. **See also:** [Chain-of-thought](#chain-of-thought-cot). **Defined in:** [05/04 Prompt Engineering](../05-llms/04-prompt-engineering-for-production.md#few-shot-examples).

### Fine-tuning

Continuing to train a pretrained model on task- or domain-specific data, either updating every weight (full fine-tuning) or only a small add-on such as LoRA. It changes behavior (format, style, skills) well and is a poor way to store facts that change often. **See also:** [LoRA](#lora-low-rank-adaptation), [RAG](#rag-retrieval-augmented-generation), [Catastrophic forgetting](#catastrophic-forgetting). **Defined in:** [05/07 Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md#what-fine-tuning-actually-changes).

### FlashAttention

An exact attention algorithm that computes attention in tiles held in fast on-chip SRAM and combines them with an online softmax, so the full sequence-by-sequence score matrix is never written to HBM. Same result, far less memory traffic, so it is faster and its extra memory grows linearly instead of quadratically with sequence length. **See also:** [SRAM](#sram-on-chip-memory), [HBM](#hbm-high-bandwidth-memory). **Defined in:** [04/08 Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md#flashattention-exact-attention-fewer-memory-trips).

### FLOPs

Floating-point operations (FLOP/s is the rate per second). A forward pass of an $N$-parameter transformer costs about $2N$ FLOPs per token (one multiply and one add per weight), so a 7B model needs about $2 \times 7 \times 10^9 = 14$ GFLOPs per generated token; training costs about $6N$ per token. **See also:** [MBU and MFU](#mbu-and-mfu). **Defined in:** [04/05 Full Transformer](../04-transformers/05-the-full-transformer-architecture.md#flops-per-token-why-inference-costs-about-2n), [05/01 Pretraining](../05-llms/01-pretraining-and-next-token-prediction.md#training-compute-the-6nd-rule-of-thumb).

### FP32, FP16 and FP8

Floating-point formats with 32, 16 and 8 bits. FP32 is the full-precision baseline; FP16 has a small range, so FP16 training needs loss scaling; FP8 (E4M3 or E5M2 variants) is used for inference and some training on recent NVIDIA GPUs (Ada, Hopper and newer). Weight memory is parameters times bytes: a 7B model is about 28 GB in FP32, 14 GB in FP16 or BF16 and 7 GB in FP8. **See also:** [BF16](#bf16-bfloat16), [INT8 and INT4](#int8-and-int4). **Defined in:** [07/01 Number Formats](../07-model-compression/01-number-formats-and-model-memory.md#the-formats-one-by-one).

### FSDP (Fully Sharded Data Parallel)

PyTorch's built-in version of the ZeRO-3 idea: parameters, gradients and optimizer state are sharded across the data-parallel GPUs, and each layer's full weights are gathered just in time (all-gather) and freed after use. **See also:** [ZeRO](#zero), [All-gather](#all-gather). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#96-pytorch-fsdp).

### Function calling (tool calling)

The model emits a structured request (a tool name plus JSON arguments that match a declared schema) instead of prose; your code validates it, runs the tool and returns the result as a new message. The model only proposes; your code decides and executes. **See also:** [Structured output](#structured-output), [Agent](#agent-ai-agent). **Defined in:** [05/05 Structured Outputs and Function Calling](../05-llms/05-structured-outputs-and-function-calling.md#function-calling-the-core-idea).

## G

### Garbage collection (GC)

The runtime's automatic reclamation of objects that nothing refers to. CPython uses reference counting plus a generational cycle collector, and in the classic three-generation collector a full (generation 2) collection scans every tracked object, so it can pause the process longer as the heap grows (recent CPython versions have been reworking this, so check the `gc` docs for your version). **See also:** [Memory leak](#memory-leak), [Tail latency](#tail-latency). **Defined in:** [11/08 Rising p95 Latency Case](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#memory-words).

### Generator

In RAG, the LLM that writes the final answer from the retrieved context. **See also:** [Retriever](#retriever). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary).

### GGUF

The single-file model format of llama.cpp, holding the weights in one of many quantization types (for example `Q4_K_M`) together with the tokenizer and metadata. It is the usual format for CPU, Apple Silicon and consumer-GPU inference with llama.cpp and Ollama. **See also:** [llama.cpp and Ollama](#llamacpp-and-ollama). **Defined in:** [07/04 LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md#gguf-and-llamacpp-k-quants-for-cpu-and-edge).

### Gloo

A collective communication backend for CPUs, used by PyTorch when the GPU backend NCCL is not available. **See also:** [NCCL](#nccl). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### Goodput

Throughput that counts only the requests that met their latency SLOs. If 1,000 requests finish in 100 s (10 req/s) but only 900 meet both the TTFT and TPOT targets, goodput is $900 / 100 = 9$ req/s; a more aggressive batch that finishes 1,200 requests in the same 100 s with only 700 inside the SLO has higher throughput (12 req/s) but lower goodput (7 req/s). **See also:** [Throughput](#throughput), [SLO, SLA and SLI](#slo-sla-and-sli). **Defined in:** [06/01 Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md#goodput).

### GPT

Generative Pre-trained Transformer: OpenAI's family of decoder-only language models, and the common name for the decoder-only, next-token-prediction design they made standard. **See also:** [Decoder-only model](#decoder-only-model). **Defined in:** [04/07 Encoder and Decoder Models](../04-transformers/07-encoder-only-encoder-decoder-and-decoder-only-models.md#decoder-only-models-gpt-and-causal-language-modeling).

### GPTQ

A post-training weight quantization method, usually to 4 bits, that quantizes one column at a time and uses approximate second-order information from calibration data to adjust the remaining weights so they compensate for each rounding error. **See also:** [AWQ](#awq-activation-aware-weight-quantization), [PTQ](#ptq-post-training-quantization). **Defined in:** [07/04 LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md#gptq-second-order-error-compensation).

### GQA (grouped-query attention)

Query heads are split into groups and each group shares one key head and one value head, shrinking the KV cache by the ratio of query heads to key/value heads with quality close to full multi-head attention. With 32 query heads and 8 key/value heads the cache is 4 times smaller: 128 KiB instead of 512 KiB per token for a 7B-class model in BF16. **See also:** [MQA](#mqa-multi-query-attention), [MLA](#mla-multi-head-latent-attention), [KV cache](#kv-cache). **Defined in:** [04/08 Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md#multi-query-and-grouped-query-attention).

### Gradient accumulation

Running several small forward and backward passes and summing their gradients before one optimizer step, to simulate a larger batch than fits in memory. Each micro-batch's loss must be divided by the number of accumulation steps. **See also:** [Gradient checkpointing](#gradient-activation-checkpointing). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#3-gradient-accumulation).

### Gradient (activation) checkpointing

Keeping only some layer activations during the forward pass and recomputing the rest during the backward pass. It trades roughly one extra forward pass of compute for a large cut in activation memory. **See also:** [Gradient accumulation](#gradient-accumulation). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#4-activation-gradient-checkpointing).

### GraphRAG

RAG over a knowledge graph built from the corpus: entities and relationships are extracted (often with community summaries), and retrieval walks the graph, which helps questions that connect many documents or need a global summary. **See also:** [RAG](#rag-retrieval-augmented-generation). **Defined in:** [08/09 Advanced RAG Patterns](../08-rag/09-advanced-rag-patterns.md#graphrag-and-knowledge-graphs).

### Greedy decoding

Always picking the single most probable next token. It is deterministic in principle but can loop or sound bland. **See also:** [Beam search](#beam-search), [Sampling](#sampling-temperature-top-k-and-top-p). **Defined in:** [05/03 Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md#greedy-decoding).

### gRPC

A remote procedure call framework (originally from Google) that runs over HTTP/2 and encodes messages with Protocol Buffers, with streaming in both directions. It is common between internal services, while browser-facing LLM output usually streams with SSE. **See also:** [SSE](#sse-server-sent-events), [WebSocket](#websocket). **Used in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#comparison), [06/06 Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md#nvidia-tensorrt-llm-and-triton-inference-server).

### GRPO (Group Relative Policy Optimization)

A reinforcement learning method that samples a group of answers per prompt and scores each answer relative to the group's average reward, removing the separate value model that PPO needs. It is widely used to train reasoning models on verifiable rewards such as passing tests or matching a known answer. **See also:** [PPO](#ppo-proximal-policy-optimization), [Chain-of-thought](#chain-of-thought-cot). **Defined in:** [05/12 Reasoning Models](../05-llms/12-reasoning-models-and-test-time-compute.md#grpo-at-a-conceptual-level).

### Guardrails

Checks around a model that block or modify unsafe inputs and outputs: input and output classifiers, schema validation, allowlists, PII redaction and policy checks in code. They reduce risk but do not replace authorization outside the model. **See also:** [Prompt injection](#prompt-injection), [Defense in depth](#defense-in-depth). **Defined in:** [05/13 LLM Safety](../05-llms/13-llm-safety-guardrails-and-responsible-ai.md#guardrail-frameworks).

## H

### Hallucination

Fluent, confident model output that is either false about the world or not supported by the source material the model was given. Low temperature does not prevent it; grounding, citations, verification and abstention reduce it. **See also:** [Faithfulness](#faithfulness-groundedness), [Abstention](#abstention). **Defined in:** [05/06 Hallucinations](../05-llms/06-hallucinations-and-grounding.md#1-what-a-hallucination-is).

### HBM (high-bandwidth memory)

The GPU's main memory: stacked DRAM chips beside the GPU die, tens of GB in size (80 GB on an A100 80GB or H100) and read at roughly 1.5 to 5 TB/s from the A100 to the H200 (newer GPUs go higher). Weights and the KV cache live here, and its bandwidth sets the speed limit for LLM decode. **See also:** [SRAM](#sram-on-chip-memory), [Compute-bound vs memory-bound](#compute-bound-vs-memory-bound). **Defined in:** [06/03 GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md#13-the-memory-hierarchy).

### HNSW (Hierarchical Navigable Small World)

A graph-based ANN index: each vector is a node linked to near neighbors, with sparse upper layers for long jumps and a dense bottom layer, and a query greedily walks toward closer nodes. It is fast and high-recall but memory-hungry and awkward with deletes. **See also:** [efConstruction and efSearch](#efconstruction-and-efsearch), [IVF](#ivf-inverted-file-index). **Defined in:** [08/06 Vector Search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#hnsw-graph-based-search).

### HPA (Horizontal Pod Autoscaler)

The Kubernetes controller that changes the number of pod replicas to keep a metric near a target: $\text{desired} = \lceil \text{current replicas} \times \text{current metric} / \text{target metric} \rceil$. With 4 replicas, an average queue depth of 30 and a target of 10, it asks for $\lceil 4 \times 30 / 10 \rceil = 12$ replicas; for GPU services, scale on queue depth or in-flight requests, not CPU. **See also:** [KEDA](#keda-kubernetes-event-driven-autoscaling), [Kubernetes](#kubernetes), [Cold start](#cold-start). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#autoscaling-and-cold-starts).

### Hybrid search

Running lexical (BM25) and dense retrieval side by side and fusing the two ranked lists, usually with reciprocal rank fusion. It catches both exact-term matches and paraphrases. **See also:** [RRF](#rrf-reciprocal-rank-fusion), [BM25](#bm25), [Dense retrieval](#dense-retrieval). **Defined in:** [08/05 Retrieval Methods](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#hybrid-search-combining-both-worlds).

## I

### Idempotency and idempotency key

An operation is idempotent if doing it twice has the same effect as doing it once. An idempotency key is a unique ID for one logical operation (for example `ticket-42:refund:A1001`): the receiver stores it with the first result and returns that result for any retry with the same key, so a retried payment or tool call runs at most once. **See also:** [Dead-letter queue](#dead-letter-queue-dlq). **Defined in:** [05/05 Structured Outputs and Function Calling](../05-llms/05-structured-outputs-and-function-calling.md#preventing-duplicate-side-effects-with-idempotency-keys), [09/02 Tool Calling](../09-agents/02-tool-calling-in-depth.md#preventing-duplicate-payments-and-other-irreversible-side-effects).

### Index (vector index and inverted index)

A data structure that makes search fast. A vector index stores embeddings and finds the nearest ones to a query vector; an inverted (keyword) index maps each term to the list of chunks that contain it, which is what BM25 searches. **See also:** [HNSW](#hnsw-hierarchical-navigable-small-world), [BM25](#bm25). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary), [08/05 Retrieval Methods](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#the-inverted-index).

### Ingestion (offline indexing)

The half of a RAG system that runs ahead of time or whenever documents change: load, parse, clean, chunk, embed and write to the index. In production it must be idempotent, handle updates and deletions without mixing versions, and version its embeddings. **See also:** [Tombstone](#tombstone), [Idempotency and idempotency key](#idempotency-and-idempotency-key). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#offline-indexing-step-by-step), [08/12 Ingestion Pipelines](../08-rag/12-production-ingestion-pipelines-at-scale.md#idempotent-ingestion).

### Instruction tuning (SFT)

Supervised fine-tuning (SFT) of a pretrained base model on (instruction, good response) examples so it follows instructions and holds a conversation instead of only continuing text. It is usually followed by preference alignment with RLHF or DPO. **See also:** [RLHF](#rlhf-reinforcement-learning-from-human-feedback), [DPO](#dpo-direct-preference-optimization). **Defined in:** [05/02 Instruction Tuning, RLHF and DPO](../05-llms/02-instruction-tuning-rlhf-and-dpo.md#supervised-fine-tuning-instruction-tuning).

### INT8 and INT4

8-bit and 4-bit integer formats for quantized weights (and sometimes activations): 1 byte and half a byte per value, plus small per-channel or per-group scale factors. A 7B model drops from about 14 GB in BF16 to about 7 GB in INT8 and about 3.5 GB in INT4, before overhead. **See also:** [Quantization](#quantization), [NF4](#nf4-4-bit-normalfloat). **Defined in:** [07/01 Number Formats](../07-model-compression/01-number-formats-and-model-memory.md#int8-and-int4-as-model-formats).

### IoU (intersection over union)

The overlap area of two boxes (or masks) divided by the area of their union, from 0 to 1. A detection usually counts as correct when its IoU with a ground-truth box is at least 0.5. **See also:** [mAP](#map-mean-average-precision), [NMS](#nms-non-maximum-suppression). **Defined in:** [03/02 Object Detection](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md#bounding-box-formats-and-iou).

### ITL (inter-token latency)

The time between two consecutive output tokens of one request. Each request yields many ITL samples, and their tail exposes stalls that an average TPOT hides. **See also:** [TPOT](#tpot-time-per-output-token), [TTFT](#ttft-time-to-first-token). **Defined in:** [06/01 Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md#inter-token-latency-itl-and-time-per-output-token-tpot).

### IVF (inverted file index)

An ANN index that clusters the vectors with k-means into `nlist` cells; a query is compared with the cell centers and searched exhaustively only inside the `nprobe` closest cells. It is often combined with product quantization (IVF-PQ) to save memory. **See also:** [nlist and nprobe](#nlist-and-nprobe), [Product quantization](#product-quantization-pq), [HNSW](#hnsw-hierarchical-navigable-small-world). **Defined in:** [08/06 Vector Search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#ivf-cluster-based-search).

## J

### Jailbreak

An input crafted to make a model violate its own safety training, for example to produce content it was trained to refuse. It differs from prompt injection, which hijacks the application's instructions. **See also:** [Prompt injection](#prompt-injection). **Defined in:** [05/13 LLM Safety](../05-llms/13-llm-safety-guardrails-and-responsible-ai.md#jailbreaks).

### JSON mode

A provider or engine setting that guarantees the output is syntactically valid JSON, but not that it matches your schema. Schema-constrained decoding plus validation in code is stronger. **See also:** [Structured output](#structured-output). **Defined in:** [05/05 Structured Outputs and Function Calling](../05-llms/05-structured-outputs-and-function-calling.md#level-2-json-mode).

### JSON-RPC 2.0

A small standard for remote procedure calls encoded as JSON: a request carries `jsonrpc`, `method`, `params` and an `id`; the response carries either `result` or `error` with the same `id`; a notification has no `id` and gets no reply. MCP and A2A use it as their message format. **See also:** [MCP](#mcp-model-context-protocol), [A2A](#a2a-agent2agent-protocol). **Defined in:** [09/07 MCP](../09-agents/07-model-context-protocol-and-tool-ecosystems.md#the-wire-format-json-rpc-20).

## K

### KEDA (Kubernetes Event-Driven Autoscaling)

An add-on that scales Kubernetes workloads on external signals such as queue length or Prometheus metrics: it drives the HPA with those metrics and handles scaling to and from zero replicas itself. It suits GPU and async-job workers, where CPU usage is a poor scaling signal. **See also:** [HPA](#hpa-horizontal-pod-autoscaler). **Used in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#which-signal-to-scale-on).

### KL divergence

Kullback-Leibler divergence $\text{KL}(P \,\|\, Q)$ measures how different a distribution $Q$ is from a reference $P$: the average extra cost of encoding samples from $P$ with a code built for $Q$. It is zero only when $P = Q$, never negative and not symmetric, and it appears in distillation, RLHF's penalty and drift detection. **See also:** [Cross-entropy](#cross-entropy), [PSI](#psi-population-stability-index). **Defined in:** [00/05 Softmax and Cross-Entropy](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md#kl-divergence).

### kNN (k-nearest neighbors)

Finding the $k$ stored points closest to a query. As a classifier it predicts the majority label of those neighbors; in vector search, exact kNN means comparing with every vector, which is the ground truth ANN indexes are measured against. **See also:** [ANN search](#ann-approximate-nearest-neighbor-search). **Defined in:** [01/09 Classic Algorithms](../01-classical-ml/09-knn-svm-naive-bayes-and-other-classic-algorithms.md#k-nearest-neighbors), [08/06 Vector Search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#exact-k-nearest-neighbor-search-brute-force).

### Kubernetes

An open-source container orchestrator: you declare which container images should run and how many copies, and it places them on machines (nodes), restarts failed ones, rolls out new versions, routes traffic to healthy copies and scales them. Most production model servers run on it. **See also:** [Pod](#pod), [HPA](#hpa-horizontal-pod-autoscaler), [Readiness probe](#readiness-probe). **Used in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#a-minimal-fastapi-service-load-once-health-and-readiness), [10/04 Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#serving-options-from-fastapi-to-model-servers-and-managed-endpoints).

### KV cache

GPU memory that stores the attention keys and values of every token already processed for every active request, so each decode step computes them only for the new token. Its size per token is $2 \times \text{layers} \times \text{KV heads} \times \text{head dim} \times \text{bytes}$; for a 7B model with 32 layers, 32 KV heads and head dimension 128 in FP16 that is $2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ bytes (0.5 MiB), and it often limits how many requests fit at once. **See also:** [PagedAttention](#pagedattention), [GQA](#gqa-grouped-query-attention). **Defined in:** [06/02 KV Cache](../06-llm-inference/02-kv-cache-explained.md#the-kv-cache-memory-formula).

## L

### Langfuse

An open-source LLM observability platform, self-hosted or managed, that records traces of every call (spans, generations, token counts and cost), manages versioned prompts and stores evaluation scores. **See also:** [LLMOps](#llmops). **Defined in:** [10/08 LLMOps](../10-mlops/08-llmops-tracing-prompt-management-and-langfuse.md#langfuse-in-depth).

### LangGraph

A library for building agents as explicit state graphs: nodes (functions or LLM calls), edges (including conditional ones), a shared typed state, and checkpointers that save the state after each step so runs can resume after a crash or pause for human approval. **See also:** [Agent](#agent-ai-agent). **Defined in:** [09/04 LangGraph](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#4-langgraph-core-concepts-state-nodes-edges-reducers).

### Latency

How long one request takes, from arrival to completed response; for streaming LLMs it is split into TTFT and TPOT. Report it as percentiles (p50, p95, p99), never only as an average. **See also:** [Throughput](#throughput), [p50, p95 and p99](#p50-p95-and-p99-percentiles). **Defined in:** [11/08 Rising p95 Latency Case](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#latency-and-throughput), [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#the-vocabulary-of-serving-latency-throughput-percentiles-and-littles-law).

### LayerNorm

A normalization layer that rescales each token's vector to zero mean and unit variance across its features, then applies a learned scale and shift. Transformers use it (or RMSNorm) in every block to keep training stable. **See also:** [RMSNorm](#rmsnorm). **Defined in:** [02/05 Gradients and Normalization](../02-deep-learning/05-vanishing-exploding-gradients-initialization-normalization-and-residuals.md#fix-5-layernorm-rmsnorm-and-groupnorm).

### Least privilege

Every component (user, service account, tool, agent) gets the minimum access it needs and nothing more. **See also:** [Defense in depth](#defense-in-depth). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

### Little's law

In any stable system, the average number of requests in flight $L$ equals the arrival rate $\lambda$ times the average time each spends in the system $W$: $L = \lambda W$. At 20 requests per second that each take 0.5 s, about $20 \times 0.5 = 10$ requests are in flight, so the service must handle at least 10 concurrent requests. **See also:** [Throughput](#throughput), [Latency](#latency). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#littles-law-the-one-formula-of-capacity-planning).

### Liveness probe

A cheap health check that asks "is the process alive and not stuck?"; if it fails, Kubernetes restarts the container. It should never run the model, or busy-but-healthy pods get killed under load. **See also:** [Readiness probe](#readiness-probe). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#a-minimal-fastapi-service-load-once-health-and-readiness).

### llama.cpp and Ollama

llama.cpp is a C/C++ inference engine for running quantized (GGUF) models on CPUs, Apple Silicon and consumer GPUs; Ollama is a developer-friendly wrapper that downloads and runs local models with one command. Both target local and edge use, not high-concurrency serving. **See also:** [GGUF](#gguf), [vLLM](#vllm). **Defined in:** [06/06 Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md#llamacpp).

### LLM (large language model)

A neural network, almost always a decoder-only transformer with billions of parameters, pretrained to predict the next token on huge text corpora and then usually instruction-tuned and aligned to follow requests. **See also:** [Next-token prediction](#next-token-prediction), [Transformer](#transformer). **Defined in:** [00/01 What AI, ML, DL and GenAI Are](../00-foundations/01-what-is-ai-ml-dl-and-genai.md#what-an-llm-is-in-plain-words).

### LLM-as-a-judge

Using a strong LLM, prompted with explicit criteria, to score or compare outputs. It is cheap and scalable but has biases (position, verbosity, preference for its own style) and must be checked against human labels. **See also:** [RAGAS](#ragas). **Defined in:** [05/10 LLM Evaluation](../05-llms/10-llm-evaluation-metrics-and-llm-as-a-judge.md#llm-as-a-judge).

### LLM gateway

A central proxy that all LLM traffic goes through, offering one API with authentication, quotas and budgets, routing and fallbacks between models or providers, caching and cost logging. **See also:** [Rate limiting and quotas](#rate-limiting-and-quotas). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#the-llm-gateway-pattern).

### LLM.int8()

An 8-bit quantization method for LLMs that multiplies most of each matrix in INT8 but keeps the few outlier feature dimensions (with very large activations) in FP16, which preserves accuracy. **See also:** [SmoothQuant](#smoothquant). **Defined in:** [07/04 LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md#llmint8-mixed-precision-decomposition).

### LLMOps

MLOps adapted to LLM applications: versioning prompts and model snapshots, tracing every call, evaluating free-text outputs, tracking cost per token and managing retrieval and tools. **See also:** [MLOps](#mlops). **Defined in:** [10/08 LLMOps](../10-mlops/08-llmops-tracing-prompt-management-and-langfuse.md#from-mlops-to-llmops-what-actually-changes).

### Local rank

A process's index among the GPUs of its own node (0 to 7 on an 8-GPU machine), used to pick which GPU it drives. **See also:** [Rank](#rank-distributed-training), [World size](#world-size). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### Logits

The raw, unnormalized scores a model outputs, one per class or vocabulary token; softmax turns them into probabilities. **See also:** [Softmax](#softmax). **Defined in:** [00/05 Softmax and Cross-Entropy](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md#logits-the-raw-scores-a-model-produces).

### LoRA (low-rank adaptation)

Fine-tuning that freezes the pretrained weights $W$ and learns a small low-rank update $BA$ for chosen matrices, with $B$ and $A$ of rank $r$. For a $4096 \times 4096$ matrix with $r = 16$, LoRA trains $2 \times 4096 \times 16 = 131{,}072$ parameters instead of about 16.8 million (under 1%), and adapters can be merged into the weights or served many per base model. **See also:** [QLoRA](#qlora), [PEFT](#peft-parameter-efficient-fine-tuning). **Defined in:** [05/07 Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md#lora-the-idea-and-the-math).

### LSTM (long short-term memory)

A recurrent cell with a separate memory cell and input, forget and output gates that control what is written, kept and read, which lets information and gradients survive over many more time steps than in a plain RNN. **See also:** [RNN](#rnn-recurrent-neural-network). **Defined in:** [02/08 RNNs and LSTMs](../02-deep-learning/08-rnns-lstms-and-the-road-to-transformers.md#lstm-a-memory-cell-with-gates).

## M

### mAP (mean average precision)

The standard object-detection metric: for each class, the area under the precision-recall curve of detections matched to ground truth at an IoU threshold, averaged over classes (COCO-style mAP also averages over IoU thresholds from 0.50 to 0.95). **See also:** [IoU](#iou-intersection-over-union), [Precision and recall](#precision-and-recall). **Defined in:** [03/02 Object Detection](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md#average-precision-and-map-step-by-step).

### MBU and MFU

Model bandwidth utilization (MBU) is the bytes per second a decode step actually moves (weights plus KV cache read per step, divided by the step time) over the GPU's peak memory bandwidth, the right yardstick for memory-bound decode. Model FLOPs utilization (MFU) is achieved useful FLOP/s over peak FLOP/s, the yardstick for compute-bound training and prefill. **See also:** [Compute-bound vs memory-bound](#compute-bound-vs-memory-bound). **Defined in:** [06/03 GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md#8-measuring-efficiency-mbu-and-mfu).

### MCP (Model Context Protocol)

An open standard for connecting LLM applications to tools and data: a **host** application runs one **client** per **server**, and servers expose tools, resources and prompts over JSON-RPC 2.0, through stdio for local servers or streamable HTTP for remote ones. It replaces $M \times N$ custom integrations between apps and tools with $M + N$ implementations of one protocol. **See also:** [JSON-RPC 2.0](#json-rpc-20), [A2A](#a2a-agent2agent-protocol), [Function calling](#function-calling-tool-calling). **Defined in:** [09/07 MCP](../09-agents/07-model-context-protocol-and-tool-ecosystems.md#what-mcp-is-hosts-clients-and-servers).

### Memory leak

Memory that keeps growing because the program keeps references to data it will never use again; in Python it is usually an ever-growing cache, list or global, not a missing `free`. Its signature is memory that grows with request count at constant load and resets on restart. **See also:** [RSS](#rss-resident-set-size), [Garbage collection](#garbage-collection-gc). **Defined in:** [11/08 Rising p95 Latency Case](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#memory-words).

### Metadata

Structured fields stored next to each chunk (source ID, title, URL, page, date, owner, permissions, tenant), used for filtering, citations and access control. **See also:** [Metadata filtering](#metadata-filtering-pre-filter-vs-post-filter). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary).

### Metadata filtering (pre-filter vs post-filter)

Restricting vector search by metadata such as tenant, ACL, date or product. A pre-filter (or in-graph filter) applies the condition during the search; a post-filter removes results after the search, which can leave too few results and must never be the only permission check. **See also:** [ACL](#acl-access-control-list), [Tenant](#tenant). **Defined in:** [08/07 Vector Databases](../08-rag/07-vector-databases-in-practice.md#metadata-filtering-pre-post-and-in-graph).

### MIG and MPS

Two NVIDIA ways to share one physical GPU between processes. MIG (Multi-Instance GPU) splits supported data-center GPUs such as the A100 and H100 into up to seven hardware-isolated slices, each with its own memory and compute; MPS (Multi-Process Service) lets several processes run kernels on the same GPU at once with weaker isolation. Both suit many small models, not one big model. **Used in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#the-gpu-is-a-shared-resource-not-a-thread-pool), [06/03 GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md#125-production-failure-modes-worth-knowing).

### Mixed-precision training

Running most forward and backward math in FP16 or BF16 for speed and memory while keeping an FP32 master copy of the weights for the updates; FP16 also needs loss scaling so small gradients do not underflow to zero. **See also:** [BF16](#bf16-bfloat16). **Defined in:** [02/09 Mixed Precision](../02-deep-learning/09-mixed-precision-training.md#5-the-mixed-precision-training-recipe).

### MLA (multi-head latent attention)

DeepSeek-V2's attention variant: it compresses each token's keys and values into one small latent vector, caches only that latent, and reconstructs per-head keys and values from it, aiming for an MQA-sized KV cache with MHA-like quality. It keeps a small separate component for RoPE, because the position rotation does not commute with the compression. **See also:** [MQA](#mqa-multi-query-attention), [GQA](#gqa-grouped-query-attention). **Defined in:** [04/08 Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md#going-further-multi-head-latent-attention-mla).

### MLOps

The practices and tools that take ML models from experiment to reliable production: experiment tracking, data and model versioning, a model registry, CI/CD and continuous training, deployment and monitoring. **See also:** [LLMOps](#llmops), [Model registry](#model-registry). **Defined in:** [10/01 ML Lifecycle](../10-mlops/01-ml-lifecycle-and-experiment-tracking.md#what-mlops-is).

### Model registry

The system of record for models you intend to ship: immutable versions with lineage and metadata, aliases such as `champion` and `challenger`, approvals and access control. Promotion and rollback become moving an alias. **See also:** [MLOps](#mlops). **Defined in:** [10/02 Model Registry](../10-mlops/02-data-and-model-versioning-and-model-registry.md#the-model-registry).

### MoE (mixture of experts)

A layer with many parallel feed-forward experts and a router that sends each token to only the top few (2 of 8 in Mixtral, 8 of 256 routed experts in DeepSeek-V3). Total parameters (memory) grow much faster than active parameters per token (compute), so the model is cheap per token but large to hold. **See also:** [All-to-all](#all-to-all). **Defined in:** [04/08 Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md#mixture-of-experts).

### MQA (multi-query attention)

All query heads share a single key head and a single value head, giving the smallest KV cache (with 32 query heads, 32 times smaller than standard multi-head attention) and the fastest decode, at some cost in quality. **See also:** [GQA](#gqa-grouped-query-attention), [MLA](#mla-multi-head-latent-attention). **Defined in:** [04/08 Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md#multi-query-and-grouped-query-attention).

### Multi-head attention (MHA)

Several attention operations (heads) run in parallel, each with its own query, key and value projections into a smaller subspace, and their outputs are concatenated and projected. Different heads can track different relationships between tokens. **See also:** [Attention](#attention-scaled-dot-product-attention), [GQA](#gqa-grouped-query-attention). **Defined in:** [04/04 Multi-Head Attention](../04-transformers/04-multi-head-attention-and-causal-masking.md#multi-head-attention-the-idea-and-the-formula).

### Multi-tenant system

One deployment of software serving many customer organizations (tenants) at once, sharing some infrastructure while keeping each tenant's data isolated. **See also:** [Tenant](#tenant), [Silo, bridge and pool](#silo-bridge-and-pool). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

## N

### NCCL

The NVIDIA Collective Communications Library (pronounced "nickel"), which implements all-reduce and the other collectives efficiently across NVIDIA GPUs over NVLink and the network; it is PyTorch's default GPU backend. **See also:** [Gloo](#gloo), [All-reduce](#all-reduce). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### Next-token prediction

The pretraining objective of LLMs: at every position, predict the next token given all the previous ones, trained with cross-entropy. **See also:** [Cross-entropy](#cross-entropy), [Perplexity](#perplexity). **Defined in:** [05/01 Pretraining](../05-llms/01-pretraining-and-next-token-prediction.md#next-token-prediction-the-intuition).

### NF4 (4-bit NormalFloat)

A 4-bit data type from the QLoRA paper whose 16 levels sit at quantiles of a normal distribution (dense near zero, sparse near the edges), matching the bell-shaped distribution of trained weights better than evenly spaced INT4. QLoRA stores the frozen base model in NF4 with per-block scales. **See also:** [QLoRA](#qlora), [INT8 and INT4](#int8-and-int4). **Defined in:** [07/01 Number Formats](../07-model-compression/01-number-formats-and-model-memory.md#nf4-4-bit-normalfloat).

### nlist and nprobe

The two knobs of an IVF index. `nlist` is the number of k-means cells the vectors are split into (fixed at build time); `nprobe` is how many of the closest cells each query searches (tunable per query), so raising it raises recall and latency, like `efSearch` for HNSW. With 10 million vectors, `nlist` = 10,000 (about 1,000 vectors per cell) and `nprobe` = 10, a query makes about $10{,}000 + 10 \times 1{,}000 = 20{,}000$ comparisons instead of 10 million. **See also:** [IVF](#ivf-inverted-file-index), [efConstruction and efSearch](#efconstruction-and-efsearch). **Defined in:** [08/06 Vector Search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#what-nprobe-controls-with-numbers).

### NMS (non-maximum suppression)

Detector post-processing that keeps the highest-scoring box and removes other boxes of the same class that overlap it above an IoU threshold, then repeats with the next-best box. **See also:** [IoU](#iou-intersection-over-union), [DETR](#detr-detection-transformer). **Defined in:** [03/02 Object Detection](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md#non-maximum-suppression-with-a-worked-example).

### Node

One physical machine in a cluster, often with 8 GPUs; in Kubernetes, a machine that pods are scheduled onto. **See also:** [Rank](#rank-distributed-training), [Pod](#pod). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

## O

### OOM (out of memory) kill

The operating system or container runtime terminates a process that exceeds its memory limit. On a GPU, running out of memory is a failed allocation (`CUDA out of memory`) raised as an exception rather than a kill. **See also:** [RSS](#rss-resident-set-size), [Memory leak](#memory-leak). **Defined in:** [11/08 Rising p95 Latency Case](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#memory-words).

### OpenAI-compatible endpoint

An HTTP API that accepts the same request and response format as OpenAI's chat completions API, so the same client code can call vLLM, SGLang, Ollama and many providers. Compatibility is often partial for advanced features. **See also:** [vLLM](#vllm). **Defined in:** [06/06 Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md#openai-compatible-endpoints).

## P

### p50, p95 and p99 (percentiles)

The latency that 50%, 95% and 99% of requests are at or below; p50 is the median and the region above p95 or p99 is the tail. With the nearest-rank method on 20 sorted latencies, p95 is the value at rank $\lceil 0.95 \times 20 \rceil = 19$; fleet-wide percentiles must come from merged histograms, never from averaging per-replica percentiles. **See also:** [Tail latency](#tail-latency), [Latency](#latency). **Defined in:** [11/08 Rising p95 Latency Case](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#percentiles-why-p95-and-not-the-average), [00/04 Probability and Statistics](../00-foundations/04-probability-and-statistics.md#percentiles-p50-p95-and-p99-by-hand).

### PagedAttention

vLLM's KV cache manager: the cache is split into fixed-size blocks that are allocated on demand and tracked by a per-request block table, like virtual-memory pages. Almost no memory is wasted on fragmentation or unused reservations, and blocks can be shared between requests with the same prefix. **See also:** [KV cache](#kv-cache), [Continuous batching](#continuous-batching). **Defined in:** [06/04 Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md#pagedattention-virtual-memory-for-the-kv-cache).

### PEFT (parameter-efficient fine-tuning)

The family of methods that train a small number of new or selected parameters while freezing the rest: LoRA, QLoRA, adapters, prefix and prompt tuning. They are cheap to train and store and less prone to catastrophic forgetting. **See also:** [LoRA](#lora-low-rank-adaptation). **Defined in:** [05/07 Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md#parameter-efficient-fine-tuning-peft-the-family).

### Perplexity

The exponential of the average cross-entropy per token, roughly the number of equally likely tokens the model is choosing between; an average cross-entropy of 2.0 nats gives perplexity $e^{2} \approx 7.39$. Lower is better, and it is comparable only between models with the same tokenizer and test set. **See also:** [Cross-entropy](#cross-entropy). **Defined in:** [00/05 Softmax and Cross-Entropy](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md#perplexity).

### Pipeline parallelism

Consecutive groups of layers (stages) live on different GPUs and micro-batches flow through them like an assembly line; the idle time at the start and end is the pipeline bubble. **See also:** [Tensor parallelism](#tensor-parallelism), [Data parallelism](#data-parallelism-ddp). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#8-pipeline-parallelism-splitting-layers-across-gpus).

### Pod

The smallest deployable unit in Kubernetes: one or more containers that share a network address and storage and are scheduled together onto one node. A model-serving replica is usually one pod, often owning one or more GPUs. **See also:** [Kubernetes](#kubernetes), [Replica](#replica). **Used in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#the-vocabulary-of-serving-latency-throughput-percentiles-and-littles-law).

### PPO (proximal policy optimization)

The reinforcement learning algorithm of classic RLHF: it updates the model to earn a higher reward-model score while clipping each update and adding a KL penalty that keeps it close to the starting model. **See also:** [RLHF](#rlhf-reinforcement-learning-from-human-feedback), [GRPO](#grpo-group-relative-policy-optimization). **Defined in:** [05/02 Instruction Tuning, RLHF and DPO](../05-llms/02-instruction-tuning-rlhf-and-dpo.md#rlhf-step-3-policy-optimization-with-ppo-and-a-kl-penalty).

### Pre-norm vs post-norm

Where a transformer block puts its normalization layer. Post-norm (the original 2017 Transformer) normalizes after the residual addition, $\text{norm}(x + f(x))$; pre-norm (modern LLMs) normalizes at the start of the branch, $x + f(\text{norm}(x))$, which keeps a clean residual path and trains deep stacks more stably, with less need for learning-rate warmup. **See also:** [LayerNorm](#layernorm), [Residual connection](#residual-connection). **Defined in:** [04/08 Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md#pre-normalization-moving-the-norm-inside-the-residual-branch).

### Precision and recall

Precision is the share of returned (or flagged) items that are truly relevant; recall is the share of truly relevant items that were returned. If a retriever returns 5 chunks of which 3 are relevant, out of 4 relevant chunks in total, precision is $3/5 = 0.6$ and recall is $3/4 = 0.75$; first-stage retrieval favors recall and reranking favors precision. **See also:** [Recall@k](#recallk), [Reranker](#reranker). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary), [01/04 Classification Metrics](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md#accuracy-precision-recall-specificity-and-f1).

### Preemption

When an LLM server runs out of KV cache blocks, it pauses some running requests and frees their memory, later recomputing their cache or swapping it back from CPU memory. It shows up as sudden latency spikes. **See also:** [Admission control](#admission-control), [KV cache](#kv-cache). **Defined in:** [06/04 Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md#preemption).

### Prefill

The first phase of LLM generation: the whole prompt is processed in one parallel forward pass, filling the KV cache and producing the first token. For prompts of more than a few hundred tokens it is usually compute-bound and, together with queueing, dominates TTFT on long prompts. **See also:** [Decode](#decode), [Chunked prefill](#chunked-prefill), [TTFT](#ttft-time-to-first-token). **Defined in:** [06/01 Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md#prefill-vs-decode-the-two-phases-of-generation).

### Prefix caching (prompt caching)

Keeping the KV cache of previously processed prompt prefixes and reusing it when a new request starts with exactly the same tokens, which skips most of the prefill for shared system prompts, documents or chat history. Put stable content first in prompts to get cache hits. **See also:** [Semantic caching](#semantic-caching). **Defined in:** [06/05 Speculative Decoding and Caching](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md#prefix-caching-inside-the-serving-engine).

### Principal

Anything that can be granted access: a user, a group or a service account, written like `user:acme/alice` or `group:acme/hr`. **See also:** [ACL](#acl-access-control-list). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

### Process group

A subset of ranks in a distributed job that communicate together, for example the 8 GPUs that share one tensor-parallel layer; collective operations run within a process group. **See also:** [Rank](#rank-distributed-training), [Collective operation](#collective-operation). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### Product quantization (PQ)

Compressing vectors by splitting each into $m$ sub-vectors and replacing each sub-vector with the ID of its nearest center in a small learned codebook. With $d = 768$, $m = 96$ and 8-bit codes, a 3,072-byte float32 vector becomes 96 bytes (32 times smaller); distances are approximated with lookup tables and the top results can be rescored with full vectors. **See also:** [IVF](#ivf-inverted-file-index). **Defined in:** [08/06 Vector Search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#product-quantization-compressing-vectors).

### Prompt injection

Attacker-written text that the model reads and obeys as if it were instructions, either typed by the user (direct) or hidden in documents, web pages, emails or tool results the model processes (indirect). No prompt fully prevents it, so limit what a hijacked model can do: authorization outside the model, least-privilege tools and human approval for risky actions. **See also:** [Jailbreak](#jailbreak), [Guardrails](#guardrails). **Defined in:** [05/13 LLM Safety](../05-llms/13-llm-safety-guardrails-and-responsible-ai.md#prompt-injection-the-root-problem), [09/08 Agent Reliability and Security](../09-agents/08-agent-reliability-security-and-prompt-injection.md#prompt-injection-from-tools-and-retrieved-documents).

### Pruning

Removing weights (unstructured pruning) or whole structures such as heads, channels or layers (structured pruning) from a trained model. Only structured or 2:4 semi-structured sparsity reliably speeds up GPUs. **See also:** [Quantization](#quantization), [Distillation](#distillation-knowledge-distillation). **Defined in:** [07/07 Pruning](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md#what-pruning-is).

### PSI (Population Stability Index)

A drift score that compares a feature's binned distribution in a current window ($a_i$) with a reference ($e_i$): $\text{PSI} = \sum_i (a_i - e_i) \ln(a_i / e_i)$. With 4 equal reference bins (0.25 each) and current shares 0.10, 0.20, 0.30 and 0.40, PSI $\approx 0.137 + 0.011 + 0.009 + 0.071 \approx 0.228$, a moderate shift under the usual rule of thumb (below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 significant). **See also:** [Data drift](#data-drift-covariate-shift), [KL divergence](#kl-divergence). **Defined in:** [01/08 Data Drift and Concept Drift](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md#population-stability-index-psi).

### PTQ (post-training quantization)

Quantizing an already-trained model without further training, using a small calibration set to choose scales, in minutes to hours. It is usually close to lossless at 8 bits and works well for 4-bit LLM weights with methods such as GPTQ and AWQ. **See also:** [QAT](#qat-quantization-aware-training), [Calibration data](#calibration-data-quantization). **Defined in:** [07/03 PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md#the-post-training-quantization-ptq-workflow).

## Q

### QAT (quantization-aware training)

Training or fine-tuning with simulated ("fake") quantization in the forward pass, using the straight-through estimator to pass gradients through the rounding, so the model learns weights that survive it. It recovers more accuracy at low bit widths than PTQ but costs a training run and data. **See also:** [PTQ](#ptq-post-training-quantization). **Defined in:** [07/03 PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md#quantization-aware-training-qat).

### QLoRA

LoRA training on top of a frozen base model stored in 4-bit NF4 (with double quantization of the scales and paged optimizers), computing in BF16. The paper's headline result was fine-tuning a 65B model on a single 48 GB GPU. **See also:** [LoRA](#lora-low-rank-adaptation), [NF4](#nf4-4-bit-normalfloat). **Defined in:** [05/07 Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md#qlora-fine-tuning-on-top-of-a-4-bit-base).

### Quantization

Storing weights (and possibly activations or the KV cache) in fewer bits by mapping real values onto a small integer or low-precision grid with a scale factor, plus a zero point for asymmetric schemes. It cuts memory and, for memory-bound decode, latency, as long as fast kernels exist for the format. **See also:** [Zero point and scale](#zero-point-and-scale), [PTQ](#ptq-post-training-quantization), [W4A16 and W8A8](#w4a16-and-w8a8). **Defined in:** [07/02 Quantization Fundamentals](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md#1-the-big-idea-snapping-real-numbers-onto-a-small-integer-grid).

### Queries, keys and values (Q, K, V)

The three vectors each token produces in attention through learned projections: the query says what the token is looking for, the keys say what each token offers to be matched against, and the values are the content that gets averaged. **See also:** [Attention](#attention-scaled-dot-product-attention), [KV cache](#kv-cache). **Defined in:** [04/03 Attention](../04-transformers/03-attention-mechanism-step-by-step.md#intuition-queries-keys-and-values-as-a-library-search).

## R

### RAG (retrieval-augmented generation)

At question time, retrieve relevant chunks from your own documents and put them in the prompt, so the LLM answers from current, citable sources instead of only what it memorized in training. **See also:** [Retriever](#retriever), [Generator](#generator), [Fine-tuning](#fine-tuning). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#the-problems-rag-solves).

### RAGAS

An open-source Python library of RAG evaluation metrics (faithfulness, response relevancy, context precision and context recall) computed with LLM judges and embeddings. **See also:** [Faithfulness](#faithfulness-groundedness), [LLM-as-a-judge](#llm-as-a-judge). **Defined in:** [08/10 RAG Evaluation](../08-rag/10-rag-evaluation-metrics-and-ragas.md#using-ragas).

### Rank (distributed training)

A process's global integer ID, from 0 to $N - 1$, in a distributed job that usually runs one process per GPU; rank 0 usually handles logging and checkpoints. Not to be confused with the rank of a matrix or LoRA's rank $r$. **See also:** [Local rank](#local-rank), [World size](#world-size). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### Rate limiting and quotas

Rate limiting caps how fast a client may send requests or tokens (per second or minute), usually with a token bucket; quotas cap total consumption over a longer period such as a day or month. LLM platforms limit by tokens, because cost scales with tokens rather than with requests. **See also:** [Token bucket](#token-bucket), [LLM gateway](#llm-gateway). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#rate-limiting-and-quotas).

### RBAC (role-based access control)

Permissions are granted to roles (such as "HR" or "admin") and users get roles. It is simple and auditable but coarse. **See also:** [ABAC](#abac-attribute-based-access-control), [ReBAC](#rebac-relationship-based-access-control). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

### ReAct

An agent pattern that interleaves reasoning (a thought), an action (a tool call) and an observation (the tool's result) in a loop until the task is done. **See also:** [Agent](#agent-ai-agent). **Defined in:** [09/03 Agent Design Patterns](../09-agents/03-agent-design-patterns.md#agent-pattern-1-react-reason-and-act-interleaved).

### Readiness probe

A health check that asks "can this replica serve traffic right now?". While it fails, Kubernetes stops routing requests to the pod but does not restart it, so for an LLM server it should pass only after the weights are loaded and the engine has warmed up; a separate startup probe gives slow-loading models extra time before liveness checks begin. **See also:** [Liveness probe](#liveness-probe), [Cold start](#cold-start). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#a-minimal-fastapi-service-load-once-health-and-readiness).

### ReBAC (relationship-based access control)

Access follows relationships in a graph ("Alice is in a team that edits the folder that contains this document"); Google's Zanzibar system is the best-known example. **See also:** [RBAC](#rbac-role-based-access-control), [ABAC](#abac-attribute-based-access-control). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

### Recall@k

The share of all relevant items that appear in the top $k$ results: if 2 relevant chunks exist and 1 of them is in the top 5, Recall@5 is $1/2 = 0.5$. In ANN benchmarking it means the overlap between the approximate top $k$ and the exact top $k$. **See also:** [Precision and recall](#precision-and-recall), [ANN search](#ann-approximate-nearest-neighbor-search). **Defined in:** [08/10 RAG Evaluation](../08-rag/10-rag-evaluation-metrics-and-ragas.md#recallk), [08/06 Vector Search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#measuring-ann-quality-recallk).

### Reduce-scatter

A collective operation that sums tensors across ranks but leaves each rank with only its $1/N$ slice of the result. ZeRO-2/3 and FSDP use it for gradients, and all-reduce equals a reduce-scatter followed by an all-gather. **See also:** [All-reduce](#all-reduce), [All-gather](#all-gather). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

### Replica

One running copy of a service behind a load balancer; in Kubernetes, usually one pod. Scaling out means adding replicas. **See also:** [Pod](#pod), [HPA](#hpa-horizontal-pod-autoscaler). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#the-vocabulary-of-serving-latency-throughput-percentiles-and-littles-law).

### Reranker

A slower, more accurate model (usually a cross-encoder or an LLM) that re-scores the shortlist from first-stage retrieval and reorders it, so the best few chunks go into the prompt. **See also:** [Cross-encoder](#cross-encoder), [Precision and recall](#precision-and-recall). **Defined in:** [08/08 Reranking](../08-rag/08-reranking-and-query-transformation.md#two-stage-retrieval-recall-first-precision-second).

### Residual connection

Adding a block's input to its output ($x + f(x)$), which gives gradients a direct path through deep networks and makes very deep ResNets and transformers trainable. **See also:** [LayerNorm](#layernorm). **Defined in:** [02/05 Gradients and Normalization](../02-deep-learning/05-vanishing-exploding-gradients-initialization-normalization-and-residuals.md#fix-6-residual-connections).

### Retriever

The component that takes a query and returns the top-$k$ most relevant chunks, using BM25, dense search or both. **See also:** [Top-k (retrieval)](#top-k-retrieval), [Generator](#generator). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary).

### RLHF (reinforcement learning from human feedback)

Alignment in three steps: collect human preferences between pairs of answers, train a reward model on them, then optimize the LLM with reinforcement learning (classically PPO) to raise its reward while a KL penalty keeps it close to the SFT model. **See also:** [DPO](#dpo-direct-preference-optimization), [PPO](#ppo-proximal-policy-optimization). **Defined in:** [05/02 Instruction Tuning, RLHF and DPO](../05-llms/02-instruction-tuning-rlhf-and-dpo.md#rlhf-step-1-collecting-preference-data).

### RMSNorm

A normalization layer that divides each vector by its root mean square and applies a learned gain, without subtracting the mean as LayerNorm does. It is a little cheaper, usually works as well, and is standard in Llama-style models. **See also:** [LayerNorm](#layernorm). **Defined in:** [04/08 Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md#rmsnorm-layernorm-without-the-mean).

### RMSProp

An optimizer that divides each parameter's step by the square root of a running average of that parameter's squared gradients, so parameters with consistently large gradients take smaller steps. Adam is essentially RMSProp plus momentum, with a bias correction for the first steps. **See also:** [Adam and AdamW](#adam-and-adamw), [SGD](#sgd-stochastic-gradient-descent). **Defined in:** [02/03 Optimizers](../02-deep-learning/03-optimizers-sgd-momentum-adam-and-learning-rate-schedules.md#per-parameter-adaptive-rates-adagrad-and-rmsprop).

### RNN (recurrent neural network)

A network that processes a sequence one step at a time, carrying a hidden state forward. It cannot be parallelized across time steps and struggles with long-range dependencies, the two problems transformers fixed. **See also:** [LSTM](#lstm-long-short-term-memory), [Seq2Seq](#seq2seq-sequence-to-sequence). **Defined in:** [02/08 RNNs and LSTMs](../02-deep-learning/08-rnns-lstms-and-the-road-to-transformers.md#recurrent-neural-networks-and-the-hidden-state).

### Roofline model

A chart of achievable FLOP/s against arithmetic intensity: a sloped memory-bandwidth roof on the left and a flat compute roof on the right, meeting at the ridge point (peak FLOP/s divided by bandwidth). Where a kernel sits on it tells you which resource limits it. **See also:** [Arithmetic intensity](#arithmetic-intensity). **Defined in:** [06/03 GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md#4-the-roofline-model).

### RoPE (rotary position embedding)

Encodes position by rotating pairs of query and key dimensions by angles proportional to the token's position, so the attention score between two tokens depends on their relative distance. Most modern LLMs use it, and context-extension methods rescale its frequencies. **See also:** [ALiBi](#alibi-attention-with-linear-biases). **Defined in:** [04/06 Positional Encoding](../04-transformers/06-positional-encoding-sinusoidal-learned-rope-and-alibi.md#rope-rotary-position-embedding).

### Row-level security (RLS)

A database feature (for example in Postgres) that applies a policy such as `tenant_id = current tenant` to every query, so the database itself refuses to return other tenants' rows. **See also:** [Silo, bridge and pool](#silo-bridge-and-pool). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#tenant-isolation-options).

### RRF (reciprocal rank fusion)

Merges ranked lists using only positions: $\text{RRF}(d) = \sum_{\text{lists}} \frac{1}{k + \text{rank}(d)}$, with $k = 60$ by convention. A chunk ranked 1st by BM25 and 2nd by dense search scores $1/61 + 1/62 \approx 0.016393 + 0.016129 = 0.032522$, beating a chunk that is 1st in only one list ($0.016393$), and no score normalization is needed. **See also:** [Hybrid search](#hybrid-search). **Defined in:** [08/05 Retrieval Methods](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#family-2-reciprocal-rank-fusion-rrf).

### RSS (resident set size)

The physical RAM a process is using right now, as seen by the operating system; it is what container memory limits and the OOM killer mainly react to. **See also:** [OOM kill](#oom-out-of-memory-kill), [Memory leak](#memory-leak). **Defined in:** [11/08 Rising p95 Latency Case](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#memory-words).

## S

### Sampling (temperature, top-k and top-p)

Instead of always taking the most likely token, the decoder draws from the predicted distribution. Temperature divides the logits before softmax (below 1 sharpens, above 1 flattens): logits $[2, 1]$ give probabilities $[0.73, 0.27]$ at temperature 1 and $[0.88, 0.12]$ at 0.5; top-k keeps only the $k$ most likely tokens, and top-p (nucleus) keeps the smallest set whose probabilities add up to at least $p$. **See also:** [Greedy decoding](#greedy-decoding), [Softmax](#softmax). **Defined in:** [05/03 Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md#sampling), [05/03 Temperature](../05-llms/03-text-generation-and-decoding-strategies.md#temperature).

### Semantic caching

Caching whole LLM responses keyed by an embedding of the question and serving the cached answer when a new question is similar enough. It skips the model entirely, but it can return a wrong answer for a similar-but-different question and must be scoped per tenant and permission. **See also:** [Prefix caching](#prefix-caching-prompt-caching). **Defined in:** [06/05 Speculative Decoding and Caching](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md#semantic-caching-of-whole-responses).

### Seq2Seq (sequence-to-sequence)

An encoder reads an input sequence into a representation and a decoder generates an output sequence from it, originally with RNNs for machine translation. Squeezing the whole input into one fixed-size vector was the bottleneck that attention was invented to remove. **See also:** [Encoder-decoder model](#encoder-decoder-model), [RNN](#rnn-recurrent-neural-network). **Defined in:** [02/08 RNNs and LSTMs](../02-deep-learning/08-rnns-lstms-and-the-road-to-transformers.md#sequence-to-sequence-encoder-decoder-models-for-translation).

### SGD (stochastic gradient descent)

Updating the weights with a step against the gradient computed on a small random mini-batch instead of the whole dataset. With momentum it remains a strong optimizer for vision models. **See also:** [Adam and AdamW](#adam-and-adamw). **Defined in:** [02/03 Optimizers](../02-deep-learning/03-optimizers-sgd-momentum-adam-and-learning-rate-schedules.md#vanilla-sgd).

### SGLang

An open-source LLM serving engine known for RadixAttention (automatic reuse of the KV cache for any shared prefix, stored in a prefix tree) and fast constrained decoding, which suits agents, multi-turn chat and structured output. **See also:** [vLLM](#vllm), [Prefix caching](#prefix-caching-prompt-caching). **Defined in:** [06/06 Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md#sglang).

### Shadow deployment

The new model receives a copy of live traffic and its outputs are logged and compared but never shown to users or acted on. It has zero user risk and measures behavior and latency, not user impact. **See also:** [Canary release](#canary-release), [A/B test](#ab-test). **Defined in:** [10/04 Deployment Patterns](../10-mlops/04-model-deployment-patterns.md#shadow-deployment).

### Silo, bridge and pool

Three ways to isolate tenants' data: silo gives each tenant its own index or database; bridge shares one system with a separate namespace or partition per tenant; pool keeps every tenant in one shared index where every query must carry a mandatory tenant filter. Isolation gets weaker and efficiency better from silo to pool. **See also:** [Multi-tenant system](#multi-tenant-system), [Row-level security](#row-level-security-rls). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#tenant-isolation-options).

### SLO, SLA and SLI

A service level indicator (SLI) is the measured quantity, such as the share of requests answered within 3 s; a service level objective (SLO) is the internal target for it, such as "p95 of `/chat` under 3 s over 30 days"; a service level agreement (SLA) is a contractual promise to customers, usually looser than the SLO. **See also:** [Error budget](#error-budget), [Goodput](#goodput). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#the-vocabulary-of-serving-latency-throughput-percentiles-and-littles-law), [10/06 Monitoring](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md#alert-design-slos-error-budgets-and-burn-rates).

### Slurm

An open-source job scheduler for high-performance computing and training clusters: users submit batch jobs that request nodes and GPUs (for example with `sbatch`), and Slurm queues them and starts the processes on the allocated machines. It is common for training; Kubernetes is more common for serving. **See also:** [Kubernetes](#kubernetes), [Node](#node). **Used in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#141-torchrun).

### SmoothQuant

A W8A8 quantization method that divides each activation channel by a smoothing factor and multiplies the matching weight row by the same factor (the product is unchanged), moving the outlier difficulty from activations to weights so both fit in INT8. **See also:** [LLM.int8()](#llmint8), [W4A16 and W8A8](#w4a16-and-w8a8). **Defined in:** [07/04 LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md#smoothquant-moving-the-difficulty-from-activations-to-weights).

### Softmax

Turns a vector of logits into probabilities that sum to 1: exponentiate each value and divide by the sum. Logits $[2, 1, 0]$ become about $[0.665, 0.245, 0.090]$. **See also:** [Logits](#logits), [Cross-entropy](#cross-entropy). **Defined in:** [00/05 Softmax and Cross-Entropy](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md#softmax-many-scores-to-a-probability-distribution).

### Sparse (lexical) retrieval

Retrieval by keyword matching over an inverted index, usually scored with BM25; "sparse" because each document is a vector over the whole vocabulary that is mostly zeros. **See also:** [Dense retrieval](#dense-retrieval), [BM25](#bm25). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary), [08/05 Retrieval Methods](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#the-inverted-index).

### Speculative decoding

A cheap drafter (a small model, extra heads or n-gram lookup) proposes several tokens, and the large model checks them all in one forward pass, accepting draft tokens up to the first rejection (then sampling one corrected token) with a rule that leaves the output distribution unchanged. It speeds up memory-bound decode and helps little when the server is already compute-bound at large batch sizes. **See also:** [Decode](#decode). **Defined in:** [06/05 Speculative Decoding and Caching](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md#speculative-decoding-the-core-idea).

### SRAM (on-chip memory)

The GPU's small, very fast on-chip memory (shared memory and L1, about 192 KB per streaming multiprocessor on an A100 and 256 KB on an H100), where kernels such as FlashAttention keep tiles of data to avoid round trips to HBM. **See also:** [HBM](#hbm-high-bandwidth-memory), [FlashAttention](#flashattention). **Defined in:** [06/03 GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md#13-the-memory-hierarchy).

### SSE (server-sent events)

A simple HTTP standard in which the server keeps one response open (content type `text/event-stream`) and pushes events, each written as `data: ...` followed by a blank line. It is one-way, server to client, and is how most LLM APIs stream tokens. **See also:** [WebSocket](#websocket), [gRPC](#grpc). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#3-streaming-with-server-sent-events-sse).

### Structured output

Getting machine-readable output that matches a schema, from prompt-only instructions through JSON mode to schema-constrained decoding (the engine blocks tokens that would break the schema), always followed by validation in code, for example with Pydantic. **See also:** [JSON mode](#json-mode), [Function calling](#function-calling-tool-calling). **Defined in:** [05/05 Structured Outputs and Function Calling](../05-llms/05-structured-outputs-and-function-calling.md#four-levels-of-enforcement).

### SVM (support vector machine)

A classifier that finds the boundary with the widest margin between classes, defined by the training points closest to it (the support vectors); kernels let it draw non-linear boundaries. **Defined in:** [01/09 Classic Algorithms](../01-classical-ml/09-knn-svm-naive-bayes-and-other-classic-algorithms.md#support-vector-machines).

### SwiGLU

A gated feed-forward layer used in Llama-style models: one projection goes through the SiLU (Swish) activation and is multiplied element-wise with a second projection, then a third projection maps back to the model width. Because it has three weight matrices instead of two, its hidden size is cut to about $\frac{8}{3}$ of the model width to keep the parameter count the same. **See also:** [RMSNorm](#rmsnorm). **Defined in:** [04/08 Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md#swiglu-a-gated-feed-forward-network).

## T

### T5

Text-to-Text Transfer Transformer: an encoder-decoder model that casts every task as text in, text out, pretrained by reconstructing masked-out spans of text. **See also:** [Encoder-decoder model](#encoder-decoder-model). **Defined in:** [04/07 Encoder and Decoder Models](../04-transformers/07-encoder-only-encoder-decoder-and-decoder-only-models.md#encoder-decoder-models-t5-and-span-corruption).

### Tail latency

The slow end of the latency distribution, above p95 or p99. It grows with queueing, garbage-collection pauses and fan-out to many backends, and it is invisible in averages and medians. **See also:** [p50, p95 and p99](#p50-p95-and-p99-percentiles). **Defined in:** [11/08 Rising p95 Latency Case](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md#percentiles-why-p95-and-not-the-average).

### Tenant

A customer organization whose data must be isolated from all other customers; usually one company account. **See also:** [Multi-tenant system](#multi-tenant-system), [Silo, bridge and pool](#silo-bridge-and-pool). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#core-vocabulary-tenants-principals-and-authorization).

### Tensor parallelism

Splitting the weight matrices inside each layer across GPUs (for attention, typically by heads), so every GPU computes a slice and the partial results are combined with all-reduce after each attention and feed-forward block. It needs a fast interconnect such as NVLink and is used to fit and speed up large models within one node. **See also:** [Pipeline parallelism](#pipeline-parallelism), [Data parallelism](#data-parallelism-ddp). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#7-tensor-parallelism-splitting-matrices-inside-a-layer).

### TensorRT-LLM

NVIDIA's library for building and serving LLMs with highly optimized kernels (in-flight batching, paged KV cache, FP8 and INT4 paths). It gives top performance on NVIDIA GPUs at the cost of more complexity and vendor lock-in. **See also:** [vLLM](#vllm). **Defined in:** [06/06 Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md#nvidia-tensorrt-llm-and-triton-inference-server).

### TGI (Text Generation Inference)

Hugging Face's LLM serving engine, with a Rust router, continuous batching and an OpenAI-compatible Messages API. In late 2025 Hugging Face announced it was moving to maintenance mode, so check its status before choosing it. **See also:** [vLLM](#vllm), [SGLang](#sglang). **Defined in:** [06/06 Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md#hugging-face-text-generation-inference-tgi).

### Threat model

A structured list of what you protect (assets), from whom (adversaries) and how it could go wrong (threats), written before you choose defenses. **See also:** [Defense in depth](#defense-in-depth). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#the-threat-model).

### Throughput

Completed work per unit time across all users: requests per second or output tokens per second. Larger batches raise throughput but slow each request, so always report it together with latency, ideally as goodput. **See also:** [Latency](#latency), [Goodput](#goodput), [Little's law](#littles-law). **Defined in:** [06/01 Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md#throughput-and-why-it-is-not-the-same-as-latency).

### Token and tokenization

A token is the unit a language model reads and writes: a word, part of a word, a character or a byte sequence. Tokenization splits text into tokens from a fixed vocabulary and maps them to integer IDs; in English prose one token is roughly 4 characters or 0.75 words, and cost, latency and context limits are all counted in tokens. **See also:** [BPE](#bpe-byte-pair-encoding), [Context window](#context-window). **Defined in:** [04/01 Tokenization](../04-transformers/01-tokenization.md#the-big-picture-from-text-to-token-ids-and-back).

### Token bucket

A rate-limiting algorithm: a bucket holds up to $B$ tokens and refills at rate $r$, each request spends tokens, and a request waits or is rejected when the bucket is empty. It allows short bursts up to $B$ while capping the long-run rate at $r$. **See also:** [Rate limiting and quotas](#rate-limiting-and-quotas). **Defined in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#the-token-bucket).

### Tombstone

A marker that says "this record or document is deleted as of version N". Retrieval checks it immediately, so deleted content stops appearing within seconds and delayed update events cannot bring the document back, while the physical removal (purge, compaction) happens later. **See also:** [Idempotency and idempotency key](#idempotency-and-idempotency-key). **Defined in:** [08/12 Ingestion Pipelines](../08-rag/12-production-ingestion-pipelines-at-scale.md#guaranteeing-deleted-documents-are-no-longer-retrievable), [08/07 Vector Databases](../08-rag/07-vector-databases-in-practice.md#the-write-path-upserts-deletes-tombstones-and-compaction).

### Top-k (retrieval)

The $k$ highest-scoring chunks a retriever returns, where $k$ is a number you choose (for example 5 for the prompt, 50 for a reranker). Not the same as top-k sampling in decoding. **See also:** [Retriever](#retriever), [Sampling](#sampling-temperature-top-k-and-top-p). **Defined in:** [08/01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md#core-vocabulary).

### TPOT (time per output token)

The average gap between a request's output tokens after the first: $\text{TPOT} = (\text{end-to-end latency} - \text{TTFT}) / (N_{\text{out}} - 1)$. A 101-token reply that finishes 5.2 s after sending, with a TTFT of 0.2 s, has TPOT $= (5.2 - 0.2) / 100 = 50$ ms, or 20 tokens per second for that user. **See also:** [TTFT](#ttft-time-to-first-token), [ITL](#itl-inter-token-latency), [Decode](#decode). **Defined in:** [06/01 Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md#inter-token-latency-itl-and-time-per-output-token-tpot).

### Training-serving skew

Any difference between how features or inputs are computed in training and in production (different code, library versions, preprocessing or data freshness), which makes a model that tested well behave worse live. **See also:** [Feature store](#feature-store), [Data drift](#data-drift-covariate-shift). **Defined in:** [10/03 Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md#training-serving-skew-what-it-is).

### Transfer learning

Reusing a model trained on a large dataset as the starting point for a new task, either as a frozen feature extractor or by fine-tuning some or all of its layers. **See also:** [Fine-tuning](#fine-tuning), [CNN](#cnn-convolutional-neural-network). **Defined in:** [02/07 CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md#11-transfer-learning-feature-extraction-vs-fine-tuning).

### Transformer

The neural network architecture behind modern LLMs: stacked blocks of self-attention (mixing information across tokens) and feed-forward layers (processing each token on its own), with residual connections and normalization, processing all positions in parallel. **See also:** [Attention](#attention-scaled-dot-product-attention), [Decoder-only model](#decoder-only-model). **Defined in:** [04/05 Full Transformer](../04-transformers/05-the-full-transformer-architecture.md#the-big-picture-from-text-to-next-token).

### TTFT (time to first token)

The time from sending a request to receiving the first output token: network, tokenization, queueing, prefill and the first sample. On a quiet server that might be $20 + 3 + 0 + 150 + 7 = 180$ ms, but about 2.18 s if the same request waits 2 s in the queue, which is why queueing often dominates TTFT under load. **See also:** [TPOT](#tpot-time-per-output-token), [Prefill](#prefill). **Defined in:** [06/01 Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md#time-to-first-token-ttft).

### TTL (time to live)

How long a cached entry may be served before it expires. Short TTLs bound how stale a cached answer or permission can get. **See also:** [Semantic caching](#semantic-caching). **Defined in:** [08/13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md#the-threat-model).

## V

### Vector

An ordered list of numbers, such as $[0.2, -1.3, 0.7]$. In ML a vector represents one data point, one embedding or one row of weights, and its length is called its dimension. **See also:** [Embedding](#embedding), [Cosine similarity](#cosine-similarity). **Defined in:** [00/02 Linear Algebra](../00-foundations/02-linear-algebra-for-ml.md#scalars-vectors-matrices-and-tensors).

### Vector database

A database built around a vector index that adds what an ANN library lacks: inserts, updates and deletes, persistence, metadata filtering, hybrid search, replication, access control and backups (for example pgvector, Qdrant, Weaviate, Milvus and Pinecone). **See also:** [HNSW](#hnsw-hierarchical-navigable-small-world), [Tombstone](#tombstone). **Defined in:** [08/07 Vector Databases](../08-rag/07-vector-databases-in-practice.md#from-an-ann-library-to-a-vector-database).

### ViT (Vision Transformer)

Splits an image into fixed-size patches (for example 16 x 16 pixels), embeds each patch as a token and runs a standard transformer encoder over them. It needs more data than a CNN to learn image structure but scales better. **See also:** [CNN](#cnn-convolutional-neural-network), [CLIP](#clip). **Defined in:** [03/05 ViT and CLIP](../03-computer-vision/05-vision-transformers-clip-and-multimodal-models.md#the-vision-transformer-vit-images-as-patch-tokens).

### vLLM

An open-source, high-throughput LLM serving engine built around PagedAttention and continuous batching, with prefix caching, many quantized formats and an OpenAI-compatible server; the usual default for self-hosting open models on GPUs. **See also:** [SGLang](#sglang), [TensorRT-LLM](#tensorrt-llm), [PagedAttention](#pagedattention). **Defined in:** [06/06 Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md#vllm).

## W

### W4A16 and W8A8

Shorthand for quantization schemes, where W is the weight bit width and A the activation bit width. W4A16 is weight-only 4-bit quantization (weights are expanded to 16-bit before the multiply, saving memory and bandwidth), while W8A8 quantizes both, so the multiply itself runs on INT8 or FP8 tensor cores. **See also:** [Quantization](#quantization), [Compute-bound vs memory-bound](#compute-bound-vs-memory-bound). **Defined in:** [07/03 PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md#the-naming-convention).

### WebSocket

A protocol that upgrades an HTTP connection into a persistent two-way channel over one TCP connection, so client and server can both send messages at any time. It suits cases where the client also streams, such as audio for voice agents; one-way token streaming usually uses SSE. **See also:** [SSE](#sse-server-sent-events), [gRPC](#grpc). **Used in:** [10/05 Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#comparison).

### World size

The total number of processes in a distributed job, usually the total number of GPUs. **See also:** [Rank](#rank-distributed-training). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#5-distributed-vocabulary-ranks-collectives-and-all-reduce).

## Y

### YOLO (You Only Look Once)

A family of single-stage object detectors that predict boxes and classes in one pass over the image, built for real-time speed. **See also:** [Faster R-CNN](#faster-r-cnn), [DETR](#detr-detection-transformer). **Defined in:** [03/02 Object Detection](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md#yolo-you-only-look-once).

## Z

### ZeRO

The Zero Redundancy Optimizer (from DeepSpeed) removes the duplicate copies that plain data parallelism keeps on every GPU by sharding optimizer state (stage 1), then also gradients (stage 2), then also parameters (stage 3) across the data-parallel GPUs. **See also:** [FSDP](#fsdp-fully-sharded-data-parallel), [Data parallelism](#data-parallelism-ddp). **Defined in:** [02/10 Distributed Training](../02-deep-learning/10-distributed-training-and-training-memory.md#9-zero-stages-1-2-3-and-pytorch-fsdp).

### Zero point and scale

The two numbers of affine quantization: real value $\approx$ scale $\times$ (integer $-$ zero point). Mapping the range $[-1.0, 1.55]$ to unsigned 8-bit integers (0 to 255) gives scale $= 2.55 / 255 = 0.01$ and zero point $= \text{round}(0 - (-1.0) / 0.01) = 100$, so real 0.0 is stored as 100; symmetric quantization fixes the zero point at 0. **See also:** [Quantization](#quantization). **Defined in:** [07/02 Quantization Fundamentals](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md#3-computing-scale-and-zero-point-from-min-and-max-worked-example).

## Interview questions and model answers

<details>
<summary><b>Q1. What is the difference between latency and throughput, and why can improving one hurt the other?</b></summary>

**Short answer:** Latency is how long one request takes (for an LLM, TTFT and TPOT, reported as p50, p95 and p99); throughput is how much work the whole system completes per second (requests or tokens per second). They are linked by Little's law: requests in flight equal throughput times latency. Batching more requests together raises throughput, because each weight read from GPU memory serves more sequences, but each decode step gets slower and requests wait longer to be scheduled, so per-request latency rises. The number to optimize is goodput: the highest throughput at which the latency SLO still holds.

**Deeper answer and follow-ups:**

- **Analogy:** a coffee shop can hand out many coffees per hour (throughput) while each customer still waits a long time (latency), for example with a huge queue that moves fast.
- **Worked numbers:** a moderate batch finishes 10 req/s with 90% inside the SLO (goodput 9 req/s); an aggressive batch finishes 12 req/s with about 58% inside it (goodput 7 req/s). Raw throughput picked the wrong configuration.
- **Little's law:** 20 req/s at 0.5 s each means about 10 requests in flight, which tells you how much concurrency the server must hold.
- **How to measure:** sweep concurrency with an open-loop load generator, plot p95 TTFT and TPOT against throughput, and pick the operating point just before the SLO breaks.
- **Pitfalls:** reporting average latency (it hides the tail), and closed-loop load tests that slow down with the server and hide queueing delay.

**Deep dive:** [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md#the-latency-throughput-trade-off-as-batch-size-grows), [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#the-vocabulary-of-serving-latency-throughput-percentiles-and-littles-law)

</details>

<details>
<summary><b>Q2. Explain data drift versus concept drift. How would you detect each?</b></summary>

**Short answer:** Data drift (covariate shift) means the input distribution $P(X)$ changed while the rule $P(Y \mid X)$ stayed the same, for example far more listings for large houses after a new suburb opens. Concept drift means $P(Y \mid X)$ itself changed, so the same input now deserves a different label, for example a word that used to signal spam now appearing in legitimate newsletters, or fraudsters changing tactics. Data drift is visible without labels through per-feature tests (PSI, KS, chi-square) or a domain classifier that tries to tell old data from new. Concept drift needs labels or proxies: delayed label backfill, windowed performance with confidence intervals, and sampled human review. Data drift may not hurt if the model already generalizes to the new region; concept drift almost always requires retraining on recent labeled data.

**Deeper answer and follow-ups:**

- **The third kind:** prior (label) shift, where $P(Y)$ changes, as in a spam wave; it can often be corrected by re-weighting outputs without retraining.
- **Worked PSI:** shares 0.10, 0.20, 0.30, 0.40 against 0.25 each give PSI about 0.228, a moderate shift: investigate, do not retrain automatically.
- **Shapes of concept drift:** sudden, gradual, incremental and recurring (seasonal); each suggests a different retraining cadence.
- **Alert design:** with huge samples every test is "significant", so alert on practical effect sizes and on performance, not on p-values alone.
- **LLM version:** users asking about new topics is data drift; the right answer changing because a policy was updated is closer to concept drift, and is usually fixed by refreshing the RAG index rather than retraining.

**Deep dive:** [Data Drift, Concept Drift and When to Retrain](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md#a-vocabulary-for-change-px-py-and-py--x)

</details>

<details>
<summary><b>Q3. What is the difference between data contamination and benchmark leakage?</b></summary>

**Short answer:** Contamination is overlap: evaluation items, or near-duplicates of them, are present in the data the model was trained on, so its score measures memory rather than generalization. Benchmark leakage is broader and concerns the ecosystem: information about a benchmark's test set reaches the way a model is built or chosen, either by being trained on (contamination is the most common route) or by selecting checkpoints, prompts or hyperparameters using test scores. So you can leak without contaminating: tuning a prompt on the test set never puts a test item in the weights, yet the reported score is inflated.

**Deeper answer and follow-ups:**

- **Effects:** contamination in your own pipeline inflates internal metrics, so you ship a worse model than you think; leakage on public benchmarks makes leaderboard comparisons wrong.
- **Detection:** n-gram or hash overlap between training and evaluation sets, canary strings, asking the model to complete test items verbatim, and score drops on freshly written equivalent items.
- **Input-only vs input-and-label contamination:** seeing the question alone inflates scores somewhat; seeing question and answer together inflates them much more.
- **Prevention:** deduplicate across splits, keep a private held-out set that is used rarely, version evaluation sets, and prefer items created after the model's training cutoff.

**Deep dive:** [Catastrophic Forgetting, Data Contamination and Memorization](../05-llms/09-catastrophic-forgetting-contamination-and-memorization.md#training-contamination-vs-benchmark-leakage)

</details>

<details>
<summary><b>Q4. When would you use post-training quantization (PTQ) and when quantization-aware training (QAT)?</b></summary>

**Short answer:** PTQ quantizes a finished model using a small unlabeled calibration set (often a few hundred samples) to choose scales: no training run, minutes to hours of work, usually near-lossless at 8 bits, and good at 4-bit weights for large LLMs with GPTQ or AWQ. QAT inserts simulated quantization (quantize then dequantize) into the forward pass during training or fine-tuning, with the straight-through estimator for gradients, so the weights learn to tolerate rounding. I start with PTQ and move to QAT only when PTQ's accuracy loss on my own evaluation set is unacceptable, which typically happens at very low bit widths, with activations quantized aggressively, or with small models that have less redundancy.

**Deeper answer and follow-ups:**

- **QAT costs:** training data, a training run (hours to days, multi-GPU for LLMs), and a pipeline whose exported numerics must match the runtime.
- **Calibration matters:** calibration data that does not match production (language, domain, prompt length) is a common cause of bad PTQ results; never calibrate on the test set.
- **How to measure:** perplexity, task benchmarks, your own evaluation set, per-slice and worst-case analysis, plus long-context and structured-output checks.
- **Speed is separate:** neither method makes inference faster unless kernels exist for the format; weight-only 4-bit helps memory-bound decode much more than compute-bound prefill.

**Deep dive:** [Post-Training Quantization vs Quantization-Aware Training](../07-model-compression/03-ptq-vs-qat-and-calibration.md#ptq-vs-qat-how-to-choose)

</details>

<details>
<summary><b>Q5. Bi-encoder vs cross-encoder: why do production RAG systems use both?</b></summary>

**Short answer:** A bi-encoder encodes the query and each document separately into one vector; document vectors are computed once at indexing time, so a query costs one encoder pass plus an ANN lookup over millions of vectors. A cross-encoder feeds the query and a document together through one transformer, so every query token can attend to every document token, which captures negation, exact constraints and word interactions much better, but it needs one forward pass per (query, document) pair at query time and nothing can be precomputed. Production RAG therefore uses the bi-encoder (often with BM25) for high-recall first-stage retrieval of tens to a few hundred candidates, and the cross-encoder to rerank them and keep the best handful.

**Deeper answer and follow-ups:**

- **The ceiling:** the reranker can only reorder what the first stage found, so first-stage Recall@50 caps final quality.
- **Middle ground:** late-interaction models such as ColBERT store one vector per token and score with cheap token-level matching.
- **How to measure:** Recall@k of the first stage, nDCG@10 or Recall@5 after reranking, end-to-end answer quality, and the added p95 latency, which grows with the number of candidates.
- **Training:** bi-encoders are trained contrastively with in-batch negatives; cross-encoders are trained as relevance classifiers on (query, document) pairs.

**Deep dive:** [Reranking and Query Transformation](../08-rag/08-reranking-and-query-transformation.md#bi-encoder-vs-cross-encoder-side-by-side)

</details>

<details>
<summary><b>Q6. Compare multi-head, multi-query, grouped-query and multi-head latent attention.</b></summary>

**Short answer:** All four differ in how much each token must store in the KV cache. Multi-head attention (MHA) gives every query head its own key and value head. Multi-query attention (MQA) shares one key/value head across all query heads: smallest cache, some quality loss. Grouped-query attention (GQA) is the middle ground: groups of query heads share a key/value head (Llama 3 uses 8 key/value heads), giving close to MHA quality with a cache 4 times smaller when there are 32 query heads. Multi-head latent attention (MLA, from DeepSeek-V2) caches one small compressed latent per token and reconstructs per-head keys and values from it, aiming for an MQA-sized cache with MHA-like quality. Because decode reads the whole cache every step, a smaller cache means more concurrent users and faster decode.

**Deeper answer and follow-ups:**

- **Formula:** KV bytes per token $= 2 \times L \times H_{kv} \times d_h \times b$. For a 7B-class model ($L = 32$, $d_h = 128$, BF16 so $b = 2$): MHA (32 KV heads) is 524,288 bytes (512 KiB), GQA-8 is 131,072 bytes (128 KiB), MQA is 16,384 bytes (16 KiB). For 16 sequences of 4,096 tokens that is 32 GiB vs 8 GiB vs 1 GiB.
- **Tensor parallelism:** 8 KV heads split cleanly across 8 GPUs, while MQA's single KV head must be copied to every GPU, which erodes its saving.
- **Converting checkpoints:** an MHA model can become GQA by mean-pooling each group's key and value projections and then training a little more (the GQA paper used about 5% of the original pretraining compute).
- **MLA wrinkle:** RoPE does not commute with the compression, so MLA keeps a small separate positional component.
- **What they do not fix:** prefill compute barely changes; these are decode-memory optimizations.

**Deep dive:** [Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md#multi-query-and-grouped-query-attention), [The KV Cache](../06-llm-inference/02-kv-cache-explained.md#shrinking-the-cache-part-1-multi-query-and-grouped-query-attention)

</details>

<details>
<summary><b>Q7. What is the difference between liveness and readiness probes for a model server?</b></summary>

**Short answer:** Both are periodic health checks that Kubernetes runs against each pod. Liveness asks "is the process alive and not stuck?", and when it fails the container is restarted. Readiness asks "can this pod take traffic right now?", and when it fails the pod is removed from load balancing but not restarted. An LLM server is alive long before it is ready, because downloading and loading tens of GB of weights and warming up takes minutes, so readiness should pass only after the model is loaded and a warm-up request succeeds, and a startup probe should give slow loads time before liveness checks start.

**Deeper answer and follow-ups:**

- **Never run inference in the liveness probe:** under heavy load it times out, healthy-but-busy pods get killed, the rest get more load, and the fleet restarts in a cascade.
- **Graceful shutdown:** during a deploy, stop advertising readiness first so new requests go elsewhere while in-flight streams finish.
- **Autoscaling interplay:** a new GPU replica takes minutes to become ready (cold start), so scale early on queue depth or in-flight requests rather than waiting for p95 latency to rise.
- **How to measure:** time from pod start to ready, restart counts, and error rates during deploys and scale-ups.

**Deep dive:** [Building a Scalable Inference API](../10-mlops/05-building-a-scalable-inference-api.md#a-minimal-fastapi-service-load-once-health-and-readiness)

</details>

## Key takeaways

- Use this page to decode a word quickly, then follow the **Defined in** link when you need the intuition, math and code behind it.
- For every confusable pair, be ready to give a one-line difference plus the test that tells them apart: latency vs throughput (judge with goodput), data vs concept drift (labels needed or not), contamination vs benchmark leakage (overlap vs any test information), PTQ vs QAT (calibrate vs train), bi-encoder vs cross-encoder (precompute vs read together), MQA vs GQA vs MLA (how much each token caches).
- Many terms come in families: collectives (broadcast, all-reduce, reduce-scatter, all-gather, all-to-all), KV-cache shrinkers (MQA, GQA, MLA, KV quantization), release strategies (shadow, canary, blue-green, A/B) and access models (ACL, RBAC, ABAC, ReBAC). Learn each family together.
- Infrastructure words such as Kubernetes, pod, HPA, KEDA, readiness probe, SSE and gRPC appear in almost every serving design answer; be able to say what each does in one sentence.
- When a definition includes a formula, practice its worked example until you can do it without notes: Little's law, RRF, PSI, KV cache bytes per token, TPOT and the error budget come up again and again.
