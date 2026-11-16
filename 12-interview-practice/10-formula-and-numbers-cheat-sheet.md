# Formula and Numbers Cheat Sheet

> **Level:** Beginner to Advanced · **Study time:** ~60 min for a first read, then 10 to 15 min a day in the final week · **Prerequisites:** a first pass through Parts 00 to 08; every entry links back to the chapter that derives it, so start there if a formula feels unfamiliar

## Table of contents

- [Why this matters](#why-this-matters)
- [How to use this cheat sheet](#how-to-use-this-cheat-sheet)
- [1. Math: softmax, cross-entropy, KL and perplexity](#1-math-softmax-cross-entropy-kl-and-perplexity)
- [2. Classical metrics: precision, recall, F-scores, ROC-AUC and PSI](#2-classical-metrics-precision-recall-f-scores-roc-auc-and-psi)
- [3. Transformer size, compute and scaling laws](#3-transformer-size-compute-and-scaling-laws)
- [4. Memory: weights, training and fine-tuning](#4-memory-weights-training-and-fine-tuning)
- [5. The KV cache](#5-the-kv-cache)
- [6. Inference speed, latency and queueing](#6-inference-speed-latency-and-queueing)
- [7. Quantization](#7-quantization)
- [8. Retrieval and vector search](#8-retrieval-and-vector-search)
- [9. Evaluation statistics](#9-evaluation-statistics)
- [10. Cost](#10-cost)
- [Numbers to know](#numbers-to-know)
- [Common mistakes and misconceptions](#common-mistakes-and-misconceptions)
- [Interview questions and model answers](#interview-questions-and-model-answers)
- [Key takeaways](#key-takeaways)
- [Further reading](#further-reading)
- [Next](#next)

## Why this matters

Senior AI/ML interviews are full of **back-of-envelope questions**: "How much memory does a 70B model need?", "How fast can this GPU decode?", "Is a 3-point drop on 200 eval cases real?", "When does self-hosting beat the API?". Interviewers are not testing whether you memorized a datasheet. They are testing whether you can turn a vague question into a formula, plug in sensible numbers, and sanity-check the answer in under two minutes. A candidate who says "KV cache is $2 \times$ layers $\times$ KV heads $\times$ head size $\times$ bytes per token, so about 320 KiB per token for a Llama-3-70B shape" sounds like someone who has run real systems.

The trouble is that these formulas are spread across more than 100 chapters of this course. This chapter collects them on one sheet, grouped by topic. Each entry gives the formula, what every symbol means, one worked number you can check by hand, and a link to the chapter that derives it. Nothing here is new material: if an entry does not make sense, follow the link and re-read the derivation before memorizing the result.

## How to use this cheat sheet

- **First pass (about an hour):** read each entry and redo the worked number on paper. If you cannot reproduce it, open the linked chapter.
- **Final week (10 to 15 minutes a day):** cover the formulas, read only the headings, and say each formula and its worked number out loud. Then do one [estimation drill](#interview-questions-and-model-answers) end to end.
- **In the interview:** state the formula first, then your assumptions (model shape, precision, GPU), then the arithmetic, then a sanity check ("that is about the weight size, so plausible"). Round aggressively and say so.
- **Units:** in this chapter GB means $10^9$ bytes and GiB means $2^{30}$ bytes (1 GiB is about 1.074 GB). Vendors quote GB; tools such as nvidia-smi report MiB. Say which one you mean.
- **Hardware and prices are approximate.** GPU numbers are rounded vendor datasheet figures as of 2026, and every price is illustrative. Learn the method; look up current numbers before a real decision.

## 1. Math: softmax, cross-entropy, KL and perplexity

These four ideas are the loss function of almost every classifier and language model. One running example: a model choosing the next word among `mat`, `floor` and `moon` with logits $[2.0,\ 1.0,\ 0.1]$.

### Softmax (with temperature)

Turns a vector of raw scores (**logits**) into probabilities that are positive and sum to 1. Only differences between logits matter.

$$\text{softmax}(z)_i = \frac{e^{z_i / T}}{\sum_{j=1}^{K} e^{z_j / T}}$$

- **Symbols:** $z_i$: logit of class $i$; $K$: number of classes; $T$: **temperature** (1 by default; $T < 1$ sharpens the distribution, $T > 1$ flattens it).
- **Worked:** $[2.0, 1.0, 0.1]$ gives $[0.659, 0.242, 0.099]$ at $T = 1$, $[0.864, 0.117, 0.019]$ at $T = 0.5$ and $[0.502, 0.304, 0.194]$ at $T = 2$.
- **Remember:** implementations subtract $\max_j z_j$ before exponentiating to avoid overflow; adding a constant to every logit does not change the output.
- **Derived in:** [Softmax, Entropy, Cross-Entropy and KL Divergence](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md)

### Sigmoid and binary cross-entropy

For one yes/no output, the **sigmoid** squashes a single logit into a probability, and **binary cross-entropy (BCE)** is the loss.

$$\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \text{BCE} = -\big[y \ln p + (1 - y) \ln(1 - p)\big]$$

- **Symbols:** $z$: the logit; $p = \sigma(z)$: predicted probability of the positive class; $y \in \{0, 1\}$: the label.
- **Worked:** a spam score of $z = 2.0$ gives $p = 0.881$. If the email is spam ($y = 1$), BCE $= -\ln 0.881 = 0.127$; if it is not ($y = 0$), BCE $= -\ln 0.119 = 2.127$.
- **Derived in:** [Softmax, Entropy, Cross-Entropy and KL Divergence](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md)

### Entropy and cross-entropy

**Entropy** $H(P)$ is the average surprise of outcomes drawn from $P$. **Cross-entropy** $H(P, Q)$ is the average surprise when reality is $P$ but you predict with $Q$. With a one-hot label it collapses to "minus the log of the probability given to the right answer".

$$H(P) = -\sum_x P(x) \log P(x) \qquad H(P, Q) = -\sum_x P(x) \log Q(x) \qquad \mathcal{L}_{\text{one-hot}} = -\log Q(y)$$

- **Symbols:** $P$: true distribution (labels); $Q$: model's predicted distribution; $y$: the correct class. With natural logs the unit is **nats**; 1 nat $= 1.4427$ bits.
- **Worked:** with $P = [0.7, 0.2, 0.1]$ and $Q = [0.5, 0.3, 0.2]$: $H(P) = 0.802$ and $H(P, Q) = 0.887$ nats. With a one-hot label `mat` and $Q(\texttt{mat}) = 0.659$: loss $= -\ln 0.659 = 0.417$ nats.
- **Derived in:** [Softmax, Entropy, Cross-Entropy and KL Divergence](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md)

### KL divergence

The extra cost of using $Q$ when the truth is $P$: cross-entropy minus the unavoidable entropy.

$$D_{KL}(P \,\|\, Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P) \;\ge\; 0$$

- **Symbols:** $P$: reference distribution (data or teacher); $Q$: approximation (model or student). The expectation is taken under $P$.
- **Worked:** same $P$ and $Q$: $0.887 - 0.802 = 0.085$ nats. Swapping the arguments gives $D_{KL}(Q \,\|\, P) = 0.092$: **KL is not symmetric**.
- **Remember:** forward KL $D_{KL}(P \| Q)$ is mode-covering, reverse KL $D_{KL}(Q \| P)$ is mode-seeking. KL appears in distillation, in the RLHF penalty that keeps a policy near its reference model, and in drift metrics (PSI is a symmetric KL).
- **Derived in:** [Softmax, Entropy, Cross-Entropy and KL Divergence](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md)

### Perplexity

"How many tokens the model was effectively choosing between": the exponential of the mean per-token cross-entropy.

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{t=1}^{N} \ln p_\theta(x_t \mid x_{<t})\right) = \left(\prod_{t=1}^{N} p_t\right)^{-1/N}$$

- **Symbols:** $N$: number of predicted tokens; $p_t = p_\theta(x_t \mid x_{<t})$: probability the model gave to the actual token $t$.
- **Worked:** token probabilities $0.5, 0.25, 0.125$ give mean NLL $= 1.386$ nats and $\text{PPL} = e^{1.386} = 4.0$. A uniform model over a vocabulary of $V$ tokens has $\text{PPL} = V$, so a fresh model should start near loss $\ln V$: $\ln 32{,}000 = 10.37$ (Llama 2), $\ln 128{,}256 = 11.76$ (Llama 3).
- **Remember:** perplexity is per token, so it is not comparable across different tokenizers; use **bits per byte** for that. Low perplexity does not mean a helpful chat model.
- **Derived in:** [Softmax, Entropy, Cross-Entropy and KL Divergence](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md), [How LLMs Are Trained, Part 1: Pretraining and Next-Token Prediction](../05-llms/01-pretraining-and-next-token-prediction.md)

## 2. Classical metrics: precision, recall, F-scores, ROC-AUC and PSI

One running example: a spam filter tested on 1,000 emails (100 spam, 900 legitimate) gives **TP = 80** (spam caught), **FN = 20** (spam missed), **FP = 30** (good email flagged) and **TN = 870**.

### Precision, recall, specificity and accuracy

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall (TPR)} = \frac{TP}{TP + FN} \qquad \text{Specificity} = \frac{TN}{TN + FP} \qquad \text{FPR} = \frac{FP}{FP + TN}$$

- **Symbols:** TP, FP, FN, TN: true positives, false positives, false negatives, true negatives. **TPR** is the true positive rate (recall); **FPR** is the false positive rate ($1 -$ specificity).
- **Worked:** precision $= 80/110 = 0.727$, recall $= 80/100 = 0.80$, specificity $= 870/900 = 0.967$, FPR $= 0.033$. Accuracy $= 950/1{,}000 = 0.95$, but a filter that never flags anything already scores 0.90, which is why accuracy misleads on imbalanced data.
- **Derived in:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

### F1 and F-beta

**F1** is the harmonic mean of precision and recall, so it is low if either is low. **F-beta** weights recall $\beta$ times as much as precision.

$$F_1 = \frac{2PR}{P + R} \qquad F_\beta = (1 + \beta^2)\frac{P R}{\beta^2 P + R} = \frac{(1 + \beta^2) TP}{(1 + \beta^2) TP + \beta^2 FN + FP}$$

- **Symbols:** $P$: precision; $R$: recall; $\beta > 1$ favors recall (for example $F_2$), $\beta < 1$ favors precision (for example $F_{0.5}$).
- **Worked:** $F_1 = 2 \times 0.727 \times 0.8 / 1.527 = 0.762$. $F_2 = 400/510 = 0.784$. $F_{0.5} = 100/135 = 0.741$. Recall is higher than precision here, so $F_2 > F_1 > F_{0.5}$.
- **Derived in:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

### ROC-AUC and what it means

The area under the ROC curve (TPR against FPR over all thresholds) has a probabilistic meaning that interviewers want to hear.

$$\text{ROC-AUC} = P\big(s(x^+) > s(x^-)\big) \quad \text{(ties count one half)}$$

- **Symbols:** $s(\cdot)$: the model's score; $x^+$, $x^-$: a randomly chosen positive and negative example.
- **Worked:** positives scored $0.9, 0.6$ and negatives $0.7, 0.3, 0.2$ form $2 \times 3 = 6$ pairs. The positive wins 5 of them (only $0.6 < 0.7$ loses), so AUC $= 5/6 = 0.833$.
- **Remember:** 0.5 is random ranking; below 0.5 usually means flipped labels or a sign bug. AUC is threshold-free and insensitive to class balance, which can make it look great under heavy imbalance; the **PR-AUC** (average precision) of a random model equals the positive rate (0.10 here), so it shows the difficulty more honestly.
- **Derived in:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

### Population Stability Index (PSI)

One number per feature that says how far the current distribution has moved from a reference distribution.

$$\text{PSI} = \sum_{i=1}^{B} (a_i - e_i) \ln\frac{a_i}{e_i}$$

- **Symbols:** $B$: number of bins (often 10, cut at reference quantiles); $e_i$: share of reference data in bin $i$; $a_i$: share of current data in bin $i$. PSI equals $\text{KL}(a \| e) + \text{KL}(e \| a)$.
- **Thresholds (rules of thumb, not laws):** below 0.10 no significant shift; 0.10 to 0.25 moderate, investigate; above 0.25 significant, act.
- **Worked:** reference $e = (0.25, 0.25, 0.25, 0.25)$, current $a = (0.10, 0.20, 0.30, 0.40)$: terms $0.137 + 0.011 + 0.009 + 0.071 = 0.228$, a moderate shift close to the upper edge.
- **Derived in:** [Data Drift, Concept Drift and When to Retrain](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md)

## 3. Transformer size, compute and scaling laws

### Scaled dot-product attention and why $\sqrt{d_k}$

Each query scores every key, softmax turns the scores into weights, and the output is the weighted sum of values.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

- **Symbols:** $Q$, $K$, $V$: query, key and value matrices (one row per token); $d_k$: dimension of each query and key vector (the head size).
- **Why $\sqrt{d_k}$:** if query and key entries have mean 0 and variance 1, the dot product $q \cdot k$ has variance $d_k$. Dividing by $\sqrt{d_k}$ brings the variance back to 1, keeping softmax out of its saturated, near-one-hot region where gradients vanish. It is not about overflow (the max-subtraction trick handles that).
- **Worked:** with $d_k = 128$, raw scores have a standard deviation of about $\sqrt{128} = 11.3$; after scaling, about 1.
- **Cost:** the score matrix is $n \times n$ per head (quadratic in sequence length $n$); per generated token, attention adds about $4 \cdot L \cdot t \cdot d$ FLOPs for $t$ context tokens on top of the $2N$ below (Kaplan et al. write $2 L\, n_{\text{ctx}}\, d$, which is the same term averaged over a causal sequence; state your convention).
- **Derived in:** [The Attention Mechanism, Step by Step](../04-transformers/03-attention-mechanism-step-by-step.md)

### Parameter count: about $12 L d^2$

Per block: attention has four $d \times d$ matrices ($4d^2$) and a 4x-wide MLP has two $d \times 4d$ matrices ($8d^2$).

$$N \approx 12\, L\, d^2 + V d \quad (\text{add another } V d \text{ if the LM head is untied})$$

- **Symbols:** $N$: parameters; $L$: number of layers (blocks); $d$: model width (hidden size); $V$: vocabulary size.
- **Worked (Llama 2 7B, $L = 32$, $d = 4{,}096$, $V = 32{,}000$, untied):** $12 \times 32 \times 4{,}096^2 = 6.44$B plus $2 \times 32{,}000 \times 4{,}096 = 0.26$B gives $6.70$B, against the published 6.74B. SwiGLU MLPs and GQA change the constants: Llama 2 sized its MLP to stay near $8d^2$, while Llama 3 8B's wider MLP (14,336) makes the formula about 7% low (7.49B vs 8.03B), so read the config when precision matters.
- **Reverse use:** a "7B" model with $d = 4{,}096$ must have about $7 \times 10^9 / (12 \times 4{,}096^2) \approx 35$ layers, minus room for embeddings, so about 32.
- **Derived in:** [The Full Transformer Architecture](../04-transformers/05-the-full-transformer-architecture.md)

### FLOPs per token: about $2N$ for inference, $6N$ for training

Every parameter takes part in one multiply-add (2 FLOPs) per token in the forward pass; the backward pass costs about twice the forward.

$$\text{FLOPs per token (forward)} \approx 2N \qquad \text{FLOPs per token (training)} \approx 6N$$

- **Symbols:** $N$: parameters used in matrix multiplies; a **FLOP** is one floating-point add or multiply.
- **Worked:** a 7B model needs about $1.4 \times 10^{10}$ FLOPs (14 GFLOPs) per generated token, so 1,000 output tokens cost about $1.4 \times 10^{13}$ FLOPs. At a 4,096-token context the attention term adds about 2.1 GFLOPs per token (about 15% more; about 16% against Llama 2 7B's exact $2N = 13.5$ GFLOPs); at 32K context it exceeds $2N$.
- **Remember:** FLOPs are not latency. Decode at small batch is limited by memory bandwidth, not FLOPs ([Section 6](#6-inference-speed-latency-and-queueing)).
- **Derived in:** [The Full Transformer Architecture](../04-transformers/05-the-full-transformer-architecture.md)

### Training compute: $C \approx 6ND$

$$C \approx 6 N D \qquad \text{GPU-hours} = \frac{C}{P_{\text{peak}} \times \text{MFU} \times 3{,}600}$$

- **Symbols:** $C$: total training FLOPs; $N$: parameters; $D$: training tokens; $P_{\text{peak}}$: peak FLOP/s per GPU; **MFU** (model FLOPs utilization): the fraction of peak spent on useful model math, often 35% to 55% in well-tuned runs.
- **Worked:** 7B parameters on 2T tokens: $6 \times 7 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{22}$ FLOPs. On A100s ($3.12 \times 10^{14}$ FLOP/s) at 40% MFU: about 187,000 GPU-hours (Llama 2 reported about 184,000), so about 7.6 days on 1,024 GPUs.
- **Derived in:** [How LLMs Are Trained, Part 1: Pretraining and Next-Token Prediction](../05-llms/01-pretraining-and-next-token-prediction.md)

### Chinchilla: about 20 tokens per parameter

For a fixed training budget, parameters and tokens should grow together, at roughly 20 training tokens per parameter.

$$D_{\text{opt}} \approx 20\, N \qquad C = 6 N \cdot 20 N = 120 N^2 \;\Rightarrow\; N_{\text{opt}} = \sqrt{C / 120}$$

- **Symbols:** $D_{\text{opt}}$: compute-optimal number of training tokens; $N_{\text{opt}}$: compute-optimal parameter count for budget $C$.
- **Worked:** Chinchilla itself was 70B parameters on $20 \times 70\text{B} = 1.4$T tokens. A budget of $10^{23}$ FLOPs gives $N_{\text{opt}} = \sqrt{10^{23}/120} \approx 29$B parameters and about 580B tokens.
- **Remember:** Chinchilla optimizes **training** compute only. Models that will serve a lot of traffic are deliberately "overtrained": Llama 3 8B saw over 15T tokens, about 1,875 tokens per parameter, because a smaller model is cheaper to serve forever.
- **Derived in:** [How LLMs Are Trained, Part 1: Pretraining and Next-Token Prediction](../05-llms/01-pretraining-and-next-token-prediction.md)

## 4. Memory: weights, training and fine-tuning

### Weight memory: parameters times bytes

$$M_{\text{weights}} = N \times b$$

| Format | Bytes per parameter ($b$) | 7B model | 70B model |
|---|---|---|---|
| FP32 | 4 | 28 GB | 280 GB |
| FP16 / BF16 | 2 | 14 GB | 140 GB |
| FP8 / INT8 | 1 | 7 GB | 70 GB |
| INT4 / NF4 | 0.5 (plus scales) | 3.5 GB | 35 GB |

- **Symbols:** $N$: parameters; $b$: bytes per parameter.
- **Shortcut:** billions of parameters times bytes per parameter equals GB. A 70B model in BF16 needs at least two 80 GB GPUs; in INT4 the weights fit on one.
- **Remember:** real 4-bit checkpoints are a little larger (scales, 16-bit embeddings and LM head; see [Section 7](#7-quantization)), and serving also needs the KV cache, activations and runtime overhead: $M_{\text{GPU}} \ge M_{\text{weights}} + M_{\text{KV}} + M_{\text{activations}} + M_{\text{overhead}}$.
- **Derived in:** [Number Formats and Model Memory](../07-model-compression/01-number-formats-and-model-memory.md), [GPU Fundamentals for LLMs](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

### Full fine-tuning: about 16 bytes per parameter with Adam

Mixed-precision AdamW keeps, per parameter: BF16 weights (2) + BF16 gradients (2) + FP32 master weights (4) + two FP32 Adam moments (4 + 4).

$$M_{\text{states}} = N \times (2 + 2 + 4 + 4 + 4) = 16 N \text{ bytes (activations extra)}$$

- **Worked:** 7B: $16 \times 7 = 112$ GB before activations, about 8 times the 14 GB needed just to run it. 70B: about 1.12 TB.
- **Remember:** that is why full fine-tuning is a multi-GPU job: ZeRO or FSDP shard these states (112 GB over 8 GPUs is 14 GB each), and gradient checkpointing trades recompute for activation memory. "Training memory = 2x model size" is a classic wrong answer.
- **Derived in:** [Fine-Tuning LLMs: Full Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md), [Distributed Training and Training Memory](../02-deep-learning/10-distributed-training-and-training-memory.md)

### LoRA parameters: $r(d_{\text{in}} + d_{\text{out}})$

LoRA freezes $W_0$ and learns a low-rank update $\Delta W = \frac{\alpha}{r} B A$, with $A$ of shape $r \times d_{\text{in}}$ and $B$ of shape $d_{\text{out}} \times r$.

$$P_{\text{LoRA}} = r\,(d_{\text{in}} + d_{\text{out}}) \text{ per adapted matrix} \qquad M_{\text{LoRA}} \approx 2N + 16\,P_{\text{trainable}} \text{ bytes}$$

- **Symbols:** $r$: rank; $d_{\text{in}}, d_{\text{out}}$: the matrix's input and output sizes; $\alpha$: scaling hyperparameter (often $r$ or $2r$); $P_{\text{trainable}}$: total LoRA parameters (weights, gradients and Adam moments kept in 32-bit, about 16 bytes each).
- **Worked:** a $4{,}096 \times 4{,}096$ projection at $r = 16$ trains $16 \times 8{,}192 = 131{,}072$ parameters, 0.78% of the 16.8M in the full matrix. All linear layers of Llama 2 7B at $r = 16$: about 40M trainable (0.6%), an adapter file of about 80 MB in BF16. Memory: $14 + 40\text{M} \times 16 \text{ bytes} = 14 + 0.64 \approx 14.6$ GB before activations.
- **Derived in:** [Fine-Tuning LLMs: Full Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md)

### QLoRA memory

QLoRA stores the frozen base in 4-bit NF4 (0.5 bytes per parameter), quantizes the block scales too (double quantization, about 0.127 bits per parameter instead of 0.5), and trains 16-bit LoRA adapters on top (the estimate below still budgets 16 bytes per trainable parameter, because PEFT keeps adapter weights and optimizer states in 32-bit by default).

$$M_{\text{QLoRA}} \approx 0.5\,N + \frac{0.127\,N}{8} + 16\,P_{\text{trainable}} \text{ bytes (activations extra)}$$

- **Worked (7B, $r = 16$ on all linear layers):** $3.5 + 0.11 + 0.64 \approx 4.3$ GB, against about 14.6 GB for LoRA and 112 GB for full fine-tuning. With gradient checkpointing, a 7B QLoRA run at moderate sequence lengths fits on one 16 to 24 GB GPU.
- **Worked (Llama-2-70B shape, $r = 16$ on all linear layers, about 207M trainable):** $35 + 1.1 + 3.3 \approx 39$ GB before activations, which is why 65B to 70B QLoRA fits on a single 48 GB or 80 GB GPU (the QLoRA paper's headline was a 65B model on one 48 GB GPU).
- **Derived in:** [Fine-Tuning LLMs: Full Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md)

## 5. The KV cache

During generation, each layer stores the key and value vector of every previous token so they are not recomputed. That cache, not the weights, usually decides how many users fit on a GPU.

### The formula

$$M_{\text{KV}} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times b \times S \times B$$

- **Symbols:** 2: one K and one V; $L$: layers; $n_{\text{kv}}$: key/value heads per layer (equal to the attention heads for multi-head attention, fewer with **grouped-query attention (GQA)**); $d_{\text{head}}$: head dimension (often 128); $b$: bytes per element (2 for FP16/BF16, 1 for an FP8 cache); $S$: tokens cached per sequence (prompt plus output so far); $B$: concurrent sequences.
- **Per-token form:** $m_{\text{token}} = 2 L\, n_{\text{kv}}\, d_{\text{head}}\, b$, so $M_{\text{KV}} = m_{\text{token}} \times$ total tokens held. For plain multi-head attention, $n_{\text{kv}} d_{\text{head}} = d$, so $m_{\text{token}} = 2 L d b$.
- **Derived in:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md)

### Worked: Llama-style 8B and 70B shapes (BF16 cache)

| Shape | $L$ | $n_{\text{kv}}$ | $d_{\text{head}}$ | Per token | One 8K sequence | 32 users x 8K | One 128K sequence |
|---|---|---|---|---|---|---|---|
| Llama-3-8B (GQA) | 32 | 8 | 128 | 131,072 B = 128 KiB | 1 GiB | 32 GiB | 16 GiB |
| Llama-3-70B (GQA) | 80 | 8 | 128 | 327,680 B = 320 KiB | 2.5 GiB | 80 GiB (about 86 GB) | 40 GiB |
| Llama-2-7B (no GQA) | 32 | 32 | 128 | 524,288 B = 512 KiB | 4 GiB | 128 GiB | 64 GiB |

- **Step by step for 70B:** $2 \times 80 = 160$; $\times 8 = 1{,}280$; $\times 128 = 163{,}840$ numbers per token; $\times 2$ bytes $= 327{,}680$ bytes. Times 8,192 tokens $= 2.5$ GiB per sequence.
- **Read the table:** for the 8B model, one 128K-token conversation needs about as much memory (16 GiB) as the weights (16 GB). Without GQA (64 KV heads), the 70B shape would need 2.5 MiB per token, 8 times more. An FP8 KV cache halves every number.
- (The original Llama 2 7B supports only 4,096 tokens, so its 8K and 128K entries are hypothetical, shown to illustrate the cost of full multi-head attention.)

### Capacity: how many sequences fit

$$\text{token capacity} = \frac{u \times M_{\text{GPU}} - M_{\text{weights}} - M_{\text{reserve}}}{m_{\text{token}}}$$

- **Symbols:** $u$: fraction of GPU memory the engine may use (vLLM's `gpu_memory_utilization` defaults to 0.9 in many versions); $M_{\text{reserve}}$: activations, CUDA graphs and runtime overhead (a few GB, measured at startup).
- **Worked (8B, BF16, one H100 80 GB, 2 GB reserve):** $72 - 16 - 2 = 54$ GB, divided by 131,072 B is about 412,000 tokens, about 50 concurrent 8K-token sequences.
- **Worked (70B, BF16, four H100s with tensor parallelism, 8 GB reserve):** $288 - 140 - 8 = 140$ GB, divided by 327,680 B is about 427,000 tokens, about 52 sequences at 8K.
- **Derived in:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md), [GPU Fundamentals for LLMs](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

## 6. Inference speed, latency and queueing

### Decode speed upper bound: bandwidth divided by bytes

At small batch, each decode step streams every weight (and the sequence's KV cache) from GPU memory (**HBM**, high-bandwidth memory) to do about 2 FLOPs per weight, so memory bandwidth sets the speed.

$$\text{tokens/s per sequence (batch 1)} \;\lesssim\; \frac{BW}{N \cdot b + M_{\text{KV}}}$$

- **Symbols:** $BW$: memory bandwidth in bytes/s; $N \cdot b$: weight bytes; $M_{\text{KV}}$: KV bytes read for this sequence per step. Real kernels reach roughly 60% to 80% of peak bandwidth.
- **Worked:** 7B FP16 (14 GB) on an A100 (2.0 TB/s): $2{,}000 / 14 \approx 143$ tokens/s, about 7 ms per token. 8B BF16 (16 GB) on an H100 (3.35 TB/s): about 209 tokens/s (4.8 ms). 70B INT4 (35 GB) on an A100: about 57 tokens/s. 70B BF16 split over four H100s: at most about $4 \times 3{,}350 / 140 \approx 96$ tokens/s, before communication overhead.
- **Remember:** this is why weight-only quantization speeds up batch-1 decode (fewer bytes per step) but barely helps prefill.
- **Derived in:** [GPU Fundamentals for LLMs: Compute-Bound vs Memory-Bound](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

### Batching: same weight read, more tokens

$$t_{\text{step}}(B) \approx \frac{N b + B \cdot M_{\text{KV,seq}}}{BW} \qquad \text{system tokens/s} \approx \frac{B}{t_{\text{step}}}$$

- **Symbols:** $B$: sequences decoding together; $M_{\text{KV,seq}}$: KV bytes per sequence.
- **Worked (7B FP16 on an A100, every sequence at 1,000 tokens of context, 0.52 GB of KV each):** batch 1 is 7.3 ms per step, about 138 tokens/s. Batch 32 is $(14 + 16.8)/2{,}000 \approx 15.4$ ms per step, about 2,080 tokens/s in total but about 65 tokens/s per user. Batching multiplied throughput about 15x and halved per-user speed.
- **Derived in:** [GPU Fundamentals for LLMs](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md), [Batching, Continuous Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md)

### Arithmetic intensity and the ridge point

**Arithmetic intensity (AI)** is FLOPs performed per byte moved. The **ridge point** of a GPU is the AI at which compute and memory take equal time: below it a kernel is memory-bound, above it compute-bound.

$$AI = \frac{F}{M} \qquad \text{ridge} = \frac{P_{\text{peak}}}{BW} \qquad AI_{\text{linear}}(B) \approx \frac{2B}{b} \text{ for } B \ll d$$

- **Symbols:** $F$: FLOPs; $M$: bytes moved to and from HBM; $P_{\text{peak}}$: peak FLOP/s; $B$: tokens processed together; $b$: bytes per weight.
- **Worked:** A100 ridge $= 312 / 2.0 = 156$ FLOPs per byte; H100 $= 989 / 3.35 \approx 295$. Batch-1 FP16 decode has $AI \approx 1$ (deeply memory-bound); a 2,000-token prefill has $AI$ in the hundreds to thousands (compute-bound). FP16 weight matmuls in decode become compute-bound only around batch 150 on an A100. Decode attention stays near $AI \approx 1$ at any batch size, because each sequence reads its own KV cache (with GQA it rises to roughly the number of query heads per KV head, for example about 8, still far below the ridge).
- **Derived in:** [GPU Fundamentals for LLMs](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

### Prefill time, MFU and MBU

$$t_{\text{prefill}} \approx \frac{2 N T}{P_{\text{peak}} \times \text{MFU}} \qquad \text{MFU} = \frac{\text{achieved FLOP/s}}{\text{peak FLOP/s}} \qquad \text{MBU} = \frac{\text{achieved bytes/s}}{\text{peak bytes/s}}$$

- **Symbols:** $T$: prompt tokens; **MFU**: model FLOPs utilization (use for prefill and training); **MBU**: model bandwidth utilization (use for decode).
- **Worked:** a 2,000-token prompt on a 7B model on an A100 at 50% MFU: $2.8 \times 10^{13} / 1.56 \times 10^{14} \approx 180$ ms of prefill, the floor for TTFT before queueing. A 7B FP16 decode step at 1,000 tokens of context reads about 14.52 GB (14 GB of weights plus 0.52 GB of KV); doing that in 10 ms achieves 1,452 GB/s, an MBU of about 73% on an A100 (2.0 TB/s) (60% to 80% is healthy; 20% means overhead is eating the time).
- **Derived in:** [GPU Fundamentals for LLMs](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

### TTFT, ITL, TPOT and end-to-end latency

$$\text{TTFT} = T_{\text{network}} + T_{\text{queue}} + T_{\text{prefill}} + \dots \qquad \text{TPOT} = \frac{\text{E2E} - \text{TTFT}}{N_{\text{out}} - 1} \qquad \text{E2E} \approx \text{TTFT} + (N_{\text{out}} - 1)\cdot\text{TPOT}$$

- **Symbols:** **TTFT**: time to first token; **ITL**: inter-token latency (each gap between tokens); **TPOT**: time per output token (a request's average gap); **E2E**: end-to-end latency; $N_{\text{out}}$: output tokens. Per-user speed $= 1/\text{TPOT}$.
- **Worked:** TTFT 180 ms, TPOT 22 ms, 250 output tokens: $180 + 249 \times 22 = 5{,}658$ ms, about 5.7 s, at about 45 tokens/s per user. Decode is 97% of the wait, which is why cutting output tokens is usually the biggest latency win.
- **Derived in:** [LLM Inference Fundamentals: Prefill, Decode, TTFT and Throughput](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md)

### Little's law, queueing and replica count

$$L = \lambda W \qquad W_{\text{M/M/1}} = \frac{1}{\mu - \lambda} \qquad R = \left\lceil \frac{\lambda_{\text{peak}}}{c \times u_{\text{target}}} \right\rceil$$

- **Symbols:** $L$: requests in flight; $\lambda$: arrival rate; $W$: time each request spends in the system; $\mu$: service rate of a single-server queue; $R$: replicas; $c$: per-replica capacity at SLO; $u_{\text{target}}$: target utilization (for example 0.7).
- **Worked:** 10 requests/s at 5.66 s each means about 57 requests in flight, so the KV cache must hold about 57 sequences. With $\mu = 10$/s, the M/M/1 time in system (queueing plus service) goes from 0.2 s at $\lambda = 5$ to 1 s at 9 to 10 s at 9.9: latency explodes near saturation. Peak 30 requests/s at 4 per replica and 70% target utilization: $\lceil 30 / 2.8 \rceil = 11$ replicas.
- **Derived in:** [LLM Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md), [Optimizing and Benchmarking LLM Latency and Cost](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

### Speculative decoding expected speedup

A small draft model proposes $k$ tokens; the target model verifies them in one pass and keeps the accepted prefix plus one token of its own, so the output distribution is unchanged.

$$E[\text{tokens per iteration}] = \frac{1 - \alpha^{k+1}}{1 - \alpha} \qquad \text{speedup} = \frac{1 - \alpha^{k+1}}{(1 - \alpha)(k c + 1)}$$

- **Symbols:** $\alpha$: probability each draft token is accepted (assumed independent); $k$: draft tokens per iteration; $c$: cost of one draft pass relative to one target pass.
- **Worked:** $\alpha = 0.8$, $k = 4$, $c = 0.05$: $(1 - 0.8^5)/0.2 = 3.36$ tokens per iteration, at a cost of $4 \times 0.05 + 1 = 1.2$ target passes, so about **2.8x**. Tokens per iteration can never exceed $1/(1 - \alpha)$ (5 here), and with $\alpha = 0.3$, $c = 0.2$ (same $k = 4$) it is slower than not speculating (0.79x).
- **Remember:** it helps memory-bound, small-batch decode; at large batch the GPU is already busy and gains shrink.
- **Derived in:** [Speculative Decoding, Prompt Caching and Other Inference Speedups](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md)

### Amdahl's law: the cap on any single fix

$$S = \frac{1}{(1 - p) + p/s}$$

- **Symbols:** $S$: overall speedup; $p$: fraction of time spent in the part you improve; $s$: speedup of that part.
- **Worked:** decode is 80% of latency and you make it 2x faster: $1 / (0.2 + 0.4) = 1.67$x overall. Profile first, then optimize the biggest $p$.
- **Derived in:** [Optimizing and Benchmarking LLM Latency and Cost](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

## 7. Quantization

Quantization snaps real numbers onto a small integer grid. Every scheme is defined by a **scale** (the step size) and a **zero point** (the integer that represents real 0).

### Affine (asymmetric) quantization

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}} \qquad z = \text{round}\!\left(q_{\min} - \frac{x_{\min}}{s}\right) \qquad q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right) + z,\ q_{\min},\ q_{\max}\right) \qquad \hat{x} = s\,(q - z)$$

- **Symbols:** $x$: real value; $s$: scale; $z$: integer zero point; $q$: stored integer; $q_{\min}, q_{\max}$: integer range (0 and 255 for UINT8); $\hat{x}$: dequantized value; the range is extended to include 0 so that 0 is exact.
- **Worked (UINT8, data in $[-1.0, 1.55]$):** $s = 2.55/255 = 0.01$, $z = \text{round}(0 + 100) = 100$. Then $x = 0.3142 \to 31.42 \to 31 + 100 = 131$, and $\hat{x} = 0.01 \times 31 = 0.31$ (error 0.0042).
- **Derived in:** [Quantization Fundamentals: Scale, Zero Point, Symmetric and Asymmetric](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md)

### Symmetric quantization

$$s = \frac{\max \lvert x \rvert}{q_{\max}} \qquad q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right),\ -q_{\max},\ q_{\max}\right) \qquad \hat{x} = s\,q$$

- **Symbols:** $\max\lvert x\rvert$: the **absmax**; $q_{\max}$: 127 for INT8 or 7 for INT4 (narrow range); the zero point is fixed at 0.
- **Worked (INT8 weights with absmax 0.635):** $s = 0.635 / 127 = 0.005$; $w = 0.0874 \to 17.48 \to q = 17 \to \hat{w} = 0.085$ (error 0.0024).
- **When to use which:** symmetric for weights (centered on zero, simpler integer math); asymmetric for one-sided data such as post-ReLU activations, where symmetric would waste half the grid.
- **Derived in:** [Quantization Fundamentals](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md)

### Error bound and metadata overhead

$$\lvert x - \hat{x} \rvert \le \frac{s}{2} \text{ (inside the range)} \qquad \text{effective bits per weight} = b + \frac{b_s + b_z}{g}$$

- **Symbols:** $b$: bits per stored integer; $b_s$, $b_z$: bits per scale and per zero point; $g$: group size (weights sharing one scale).
- **Worked:** the UINT8 example above has $s/2 = 0.005$, and its actual error (0.0042) stays inside that bound. Values outside the range are **clipped**, and that error can be large, which is why one outlier that stretches the range hurts everything else. INT4 with $g = 128$ and an FP16 scale: $4 + 16/128 = 4.125$ bits, so a 7B model is about $7 \times 10^9 \times 4.125 / 8 \approx 3.61$ GB; with an FP16 zero point too, 4.25 bits; NF4 with double quantization, about 4.127 bits.
- **Derived in:** [Quantization Fundamentals](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md), [Number Formats and Model Memory](../07-model-compression/01-number-formats-and-model-memory.md)

## 8. Retrieval and vector search

### BM25

The default keyword ranking function: rare terms count more (IDF), repeated terms saturate ($k_1$), and long documents are penalized ($b$).

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d)\,(k_1 + 1)}{f(t, d) + k_1\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)} \qquad \text{IDF}(t) = \ln\!\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right)$$

- **Symbols:** $f(t, d)$: count of term $t$ in document $d$; $|d|$: document length; avgdl: average document length; $N$: number of documents; $\text{df}(t)$: documents containing $t$; $k_1$: saturation, typically 1.2 to 2.0 (Lucene default 1.2); $b$: length normalization, default 0.75.
- **Worked:** for a document of average length with $k_1 = 1.2$, the term-frequency part is $2.2f/(f + 1.2)$: 1.0 for one mention, 1.375 for two, 1.571 for three, and never above $k_1 + 1 = 2.2$. A term in 10 of 1,000 documents has $\text{IDF} = \ln(1 + 990.5/10.5) = 4.56$.
- **Derived in:** [Retrieval Methods: BM25, Dense Retrieval and Hybrid Search](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md)

### Cosine similarity

$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert\,\lVert b \rVert} \qquad \lVert a - b \rVert^2 = 2 - 2\cos(a, b) \text{ for unit vectors}$$

- **Worked:** $a = [1, 2, 2]$, $b = [2, 0, 1]$: $a \cdot b = 4$, $\lVert a \rVert = 3$, $\lVert b \rVert = \sqrt{5} = 2.236$, so $\cos = 4 / 6.708 = 0.596$.
- **Remember:** on normalized embeddings, cosine, dot product and L2 distance give the same ranking, so normalize once and use the fastest one.
- **Derived in:** [Text Embeddings and Similarity Search](../08-rag/02-text-embeddings-and-similarity-search.md)

### Reciprocal Rank Fusion (RRF)

Merges ranked lists (for example BM25 and dense) using ranks only, so no score normalization is needed.

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)} \qquad k = 60 \text{ by convention}$$

- **Symbols:** $R$: the set of ranked lists; $\text{rank}_r(d)$: position of $d$ in list $r$ (lists where $d$ is missing contribute 0).
- **Worked:** a document ranked 1st by BM25 only scores $1/61 = 0.0164$. A document ranked 5th by BM25 and 4th by dense scores $1/65 + 1/64 = 0.0310$ and wins: agreement between retrievers beats being first in one list.
- **Derived in:** [Retrieval Methods: BM25, Dense Retrieval and Hybrid Search](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md)

### Recall@K, Precision@K, MRR and nDCG

$$\text{Recall@}K = \frac{|\text{Rel} \cap \text{Top}_K|}{|\text{Rel}|} \qquad \text{Precision@}K = \frac{|\text{Rel} \cap \text{Top}_K|}{K} \qquad \text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q}$$

$$\text{DCG@}K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)} \qquad \text{nDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$

- **Symbols:** Rel: the labeled relevant items for a query; $\text{Top}_K$: the first $K$ retrieved items; $\text{rank}_q$: position of the first relevant item for query $q$ (contributes 0 if none is retrieved); $\text{rel}_i$: graded relevance at position $i$; IDCG: DCG of the ideal ordering of **all** labeled relevant items, including ones the system missed.
- **Worked:** relevant $\{C3, C4, C8\}$, top 3 $= (C3, C8, C1)$: Recall@3 $= 2/3$, Precision@3 $= 2/3$. First-relevant ranks 1, 4 and 1 over three queries: MRR $= (1 + 0.25 + 1)/3 = 0.75$. Grades $(1, 0, 2, 0, 0)$ in ranked order: DCG $= 1 + 3 \times 0.5 = 2.5$, IDCG $= 3 + 1 \times 0.631 = 3.631$, nDCG@5 $= 0.689$.
- **Remember:** always state K and match it to the pipeline: Recall@50 for the first stage feeding a reranker, nDCG@5 for what goes into the prompt.
- **Derived in:** [Evaluating RAG: Retrieval Metrics, Faithfulness and RAGAS](../08-rag/10-rag-evaluation-metrics-and-ragas.md)

### Index memory, HNSW and IVF knobs

$$M_{\text{flat}} = N \times d \times 4 \text{ bytes} \qquad M_{\text{HNSW}} \approx N\,(4d + 8M) \qquad M_{\text{PQ}} \approx N \times m \text{ bytes} \qquad \text{scanned}_{\text{IVF}} \approx N \cdot \frac{n_{\text{probe}}}{n_{\text{list}}}$$

- **Symbols:** $N$: vectors; $d$: dimension; $M$: HNSW links per node (up to $2M$ on the base layer, 4 bytes each); $m$: product-quantization sub-vectors (1 byte each with 8-bit codes); $n_{\text{list}}$: IVF clusters; $n_{\text{probe}}$: clusters searched per query.
- **Worked (10M vectors of 768 dimensions):** flat float32 is 30.7 GB; HNSW with $M = 32$ adds about 2.6 GB of links (about 33.3 GB); HNSW over int8 vectors is about 10.2 GB; PQ with $m = 96$ is about 0.96 GB (32x smaller than 3,072 bytes per vector). IVF with $n_{\text{list}} = 4{,}096$ and $n_{\text{probe}} = 16$ scans about 39,000 vectors (0.39%) per query.

| Knob | Typical starting range | What it trades | Change without rebuild? |
|---|---|---|---|
| HNSW `M` | 16 to 64 | Recall ceiling vs memory and build time | No |
| HNSW `efConstruction` | 100 to 500 | Graph quality vs build time | No |
| HNSW `efSearch` | 50 to 500 (at least $k$) | Recall vs query latency | Yes, per query |
| IVF `nlist` | about $\sqrt{N}$ to $16\sqrt{N}$ | Cell size vs centroid cost | No (retrain) |
| IVF `nprobe` | A small fraction of `nlist` (for example 16 of 4,096), tuned to the recall target | Recall vs vectors scanned | Yes, per query |

- **Derived in:** [Vector Search Algorithms: Exact kNN, HNSW, IVF and Product Quantization](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md)

## 9. Evaluation statistics

### Standard error of a pass rate

$$\text{SE} = \sqrt{\frac{\hat{p}\,(1 - \hat{p})}{n}} \qquad \text{95\% interval} = \hat{p} \pm 1.96\,\text{SE}$$

- **Symbols:** $\hat{p}$: observed pass rate; $n$: independent test cases; 1.96: the normal quantile for 95%.
- **Worked:** 170 of 200 pass ($\hat{p} = 0.85$): SE $= 0.025$, margin $\pm 4.9$ points, so about 80% to 90%. At $n = 100$, $\hat{p} = 0.8$: $\pm 7.8$ points. Quadrupling $n$ halves the margin. Near 0 or 1, use a Wilson or exact interval.
- **Derived in:** [Building an LLM Evaluation Framework and Regression Tests](../05-llms/11-building-an-llm-evaluation-framework-and-regression-tests.md)

### Eval-set size for a target margin

$$n = \frac{z^2\,p\,(1 - p)}{E^2}$$

- **Symbols:** $z$: 1.96 for 95% confidence; $p$: expected pass rate; $E$: desired margin (half-width).
- **Worked:** $p = 0.85$, $E = 0.03$: $3.8416 \times 0.1275 / 0.0009 \approx 544$ cases. Worst case $p = 0.5$: about 1,068 cases for $\pm 3$ points, about 385 for $\pm 5$ points. Each slice you report on needs its own $n$.
- **Derived in:** [Building an LLM Evaluation Framework and Regression Tests](../05-llms/11-building-an-llm-evaluation-framework-and-regression-tests.md)

### McNemar's test for paired comparisons

When baseline and candidate run on the same cases, only the cases where they disagree carry information.

$$\chi^2 = \frac{(\lvert b - c \rvert - 1)^2}{b + c} \quad \text{compare with } 3.84 \text{ (5\% level, 1 degree of freedom)}$$

- **Symbols:** $b$: cases the baseline passed and the candidate failed (regressions); $c$: the reverse (improvements); the $-1$ is a continuity correction. With small $b + c$, use the exact binomial test instead.
- **Worked:** 85.0% vs 82.0% on 200 cases with $b = 14$, $c = 8$: $(6 - 1)^2 / 22 = 1.14 < 3.84$ ($p \approx 0.29$). Not significant: read the 14 regressions, but do not call it a proven drop.
- **Derived in:** [Building an LLM Evaluation Framework and Regression Tests](../05-llms/11-building-an-llm-evaluation-framework-and-regression-tests.md), [Evaluating LLMs: Metrics, Benchmarks and LLM-as-a-Judge](../05-llms/10-llm-evaluation-metrics-and-llm-as-a-judge.md)

### pass@k and Cohen's kappa

$$\text{pass@}k = \mathbb{E}_{\text{problems}}\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right] \qquad \kappa = \frac{p_o - p_e}{1 - p_e}$$

- **Symbols (pass@k):** $n$: samples generated per problem ($n \ge k$); $c$: samples that pass the unit tests; $k$: attempts allowed.
- **Symbols (kappa):** $p_o$: observed agreement between two raters (for example a judge model and humans); $p_e$: agreement expected by chance from each rater's label frequencies.
- **Worked:** $n = 10$, $c = 3$: pass@1 $= 0.3$; pass@5 $= 1 - 21/252 = 0.917$, while the plug-in $1 - 0.7^5 = 0.832$ is biased low. A judge that agrees with humans 75% of the time when chance agreement is 60% has $\kappa = 0.15/0.40 = 0.375$, only "fair".
- **Derived in:** [Evaluating LLMs: Metrics, Benchmarks and LLM-as-a-Judge](../05-llms/10-llm-evaluation-metrics-and-llm-as-a-judge.md)

## 10. Cost

All prices below are illustrative, not any vendor's list price. Restate your assumed prices in the interview.

### API cost per request and per day

$$\text{cost per request} = \frac{T_{\text{in}}\, p_{\text{in}} + T_{\text{out}}\, p_{\text{out}}}{10^6} \qquad C_{\text{API}} \text{ per million output tokens} = p_{\text{out}} + k\, p_{\text{in}}$$

- **Symbols:** $T_{\text{in}}, T_{\text{out}}$: input and output tokens per request; $p_{\text{in}}, p_{\text{out}}$: price per million input and output tokens; $k$: input tokens per output token.
- **Worked:** 2,000 input and 500 output tokens at \$0.50 and \$1.50 per million: \$0.001 + \$0.00075 = \$0.00175 per request, or \$1,750 per million requests. At a 5:1 input to output ratio, $C_{\text{API}} = 1.50 + 5 \times 0.50 = 4.00$, so \$4.00 per million output tokens.
- **Derived in:** [Optimizing and Benchmarking LLM Latency and Cost](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

### Prompt-cache discount

Cached input tokens (an exact repeated prefix such as a system prompt or a shared document) are billed at a fraction $d$ of the normal input price.

$$\text{input cost} = p_{\text{in}}\,(T_{\text{uncached}} + d\, T_{\text{cached}}) \qquad \text{effective input price} = p_{\text{in}}\,(1 - h + d\,h)$$

- **Symbols:** $d$: cached price as a fraction of the normal price; $h$: fraction of input tokens served from cache.
- **Worked:** a 10,000-token shared prefix plus 500 new tokens, 1,000 requests per hour, \$2.00 per million input tokens, $d = 0.25$: without caching, 10.5M tokens at \$2.00 per million is \$21.00 per hour; with caching, 10M cached tokens at \$0.50 plus 0.5M uncached at \$2.00 is \$5.00 + \$1.00 = \$6.00 per hour, a 71% saving (and lower TTFT). Here $h = 0.952$, so the effective input price is \$0.57 per million.
- **Remember:** caches match exact prefixes only, so put static content first, and log the cached-token count as your hit rate.
- **Derived in:** [Speculative Decoding, Prompt Caching and Other Inference Speedups](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md)

### Self-hosting cost and break-even

$$C_{\text{self}} = \frac{P_{\text{GPU}} \times G \times 10^6}{T \times 3{,}600 \times U} \qquad V^{*} = \frac{G_{\text{total}} \times P_{\text{GPU}} \times 24}{C_{\text{API}}}$$

- **Symbols:** $C_{\text{self}}$: dollars per million output tokens; $P_{\text{GPU}}$: price per GPU-hour; $G$: GPUs per replica; $T$: sustained output tokens/s per replica at your SLO (measured); $U$: average utilization; $V^{*}$: daily output tokens (in millions) at which a fixed GPU fleet of $G_{\text{total}}$ GPUs costs the same as the API.
- **Worked:** \$2.50 per GPU-hour, one GPU, 1,500 tokens/s: \$0.46 per million at $U = 1$, but \$1.16 at a realistic $U = 0.4$. Two GPUs (one is a spare) cost \$120 per day, so against a \$4.00 API the break-even is $120 / 4 = 30$M output tokens per day. Those two GPUs, sized for a 3x peak-to-average ratio, can absorb up to $1{,}500 / 3 \times 86{,}400 \approx 43$M per day; above that you add GPUs and the comparison resets.
- **Remember:** self-hosted cost moves in GPU-sized steps while API cost is linear; add engineer time, redundancy and the quality difference between models before deciding.
- **Derived in:** [Optimizing and Benchmarking LLM Latency and Cost](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md), [LLM Serving Frameworks: vLLM, SGLang, TGI, TensorRT-LLM, llama.cpp and Ollama](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md)

## Numbers to know

### GPUs (approximate, as of 2026)

These are rounded vendor datasheet figures for dense (non-sparse) math on SXM-class parts. They change with every product generation and differ between SKUs (PCIe vs SXM, different memory configurations), so treat them as interview-grade estimates and check the current datasheet before sizing a real deployment.

| GPU | Memory (HBM) | Memory bandwidth | Dense BF16 TFLOPS | Low-precision tensor cores | Ridge point (BF16, FLOPs/byte) | GPU-to-GPU link (total per GPU) |
|---|---|---|---|---|---|---|
| A100 80 GB SXM | 80 GB HBM2e | ~2.0 TB/s | ~312 | INT8 (no FP8) | ~156 | NVLink ~600 GB/s |
| H100 SXM | 80 GB HBM3 | ~3.35 TB/s | ~989 | FP8 (~1,979 dense) | ~295 | NVLink ~900 GB/s |
| H200 SXM | 141 GB HBM3e | ~4.8 TB/s | ~989 (H100 compute) | FP8 | ~206 | NVLink ~900 GB/s |
| B200 | ~180 to 192 GB HBM3e (varies by system) | ~8 TB/s | ~2,250 | FP8 (~4,500 dense) and FP4 | ~281 | NVLink ~1.8 TB/s |

Patterns worth saying out loud: memory capacity decides **how many** users and how large a model fit; memory bandwidth decides **how fast** small-batch decode runs; FLOPs decide prefill and training speed. From A100 to H100, compute grew about 3.2x but bandwidth only about 1.7x, and the H200 kept H100 compute while adding memory and bandwidth. PCIe Gen4 x16 moves only about 32 GB/s per direction (Gen5 about 64), roughly 60 times slower than A100 HBM, which is why offloading weights to CPU memory makes decode slow.

### Everyday numbers

| Quantity | Value to remember | Use it for |
|---|---|---|
| Characters per token (English) | About 4 characters, about 0.75 words; 1,000 tokens is about 750 words | Converting documents and prompts to tokens and cost. Code and non-English text differ, so measure with the real tokenizer. |
| Human reading speed | About 4 to 7 tokens/s | Beyond this, faster streaming matters less than TTFT |
| Typical interactive chat SLO | p95 TTFT about 0.5 to 1 s; p95 TPOT/ITL about 30 to 60 ms (roughly 17 to 33 tokens/s) | Starting targets for chat; code completion and voice need tighter budgets, batch jobs only care about throughput |
| Batch-1 decode, 7B FP16 on A100 | At best about 7 ms per token (at most about 143 tokens/s); about 9 to 12 ms at 60% to 80% of peak bandwidth | Sanity-checking measured latency |
| Bytes per parameter | 4 / 2 / 1 / 0.5 for FP32 / BF16 / INT8 or FP8 / INT4; 16 for mixed-precision Adam training | Memory estimates |
| KV cache per token (BF16) | 128 KiB (Llama-3-8B shape), 320 KiB (Llama-3-70B shape), 512 KiB (Llama-2-7B, no GQA) | Concurrency and context-length planning |
| Model shapes | Llama 3 8B: 32 layers, $d = 4{,}096$, 32 heads, 8 KV heads. Llama 3 70B: 80 layers, $d = 8{,}192$, 64 heads, 8 KV heads. Both: head size 128, vocabulary 128,256 | Plugging into the KV and parameter formulas |
| Compute rules | $2N$ FLOPs per token (inference), $6N$ (training), $C = 6ND$, about 20 tokens per parameter (Chinchilla) | Training time and scaling questions |
| Utilization | Training MFU about 35% to 55%; decode MBU about 60% to 80% is healthy | Turning peak specs into realistic estimates |
| Statistics constants | $z = 1.96$ for 95%; $\chi^2_1$ 5% cutoff $= 3.84$; $\pm 5$ points needs about 385 cases, $\pm 3$ points about 1,068 (worst case) | Eval sizing and significance |
| Retrieval defaults | BM25 $k_1 = 1.2$, $b = 0.75$; RRF $k = 60$; HNSW `M` 16 to 64 | Configuring and explaining hybrid search |
| Drift thresholds | PSI below 0.10 stable, 0.10 to 0.25 investigate, above 0.25 act | Monitoring answers |
| Embedding storage | 768-d float32 is 3,072 bytes per vector; 1M vectors is about 3.1 GB | Vector database sizing |
| Units | 1 GiB = 1.074 GB, so "80 GB" is about 74.5 GiB; 86,400 s per day; about 730 hours per month | Avoiding 7% errors and converting rates |

## Common mistakes and misconceptions

- **Mixing GB and GiB.** The 7% difference matters when planning tightly. Pick one and say it.
- **"Training needs 2 bytes per parameter."** That is inference in BF16. Mixed-precision Adam needs about 16 bytes per parameter plus activations.
- **Using the attention-head count for the KV cache of a GQA model.** Use `num_key_value_heads`, not `num_attention_heads`; for a Llama-3-70B shape that is a factor of 8.
- **Quoting decode tokens/s per sequence as system throughput (or the reverse).** Bandwidth over bytes gives the per-sequence ceiling at small batch; batching multiplies system throughput while per-user speed falls.
- **"Inference is memory-bound."** It depends on the phase and batch: prefill is compute-bound, small-batch decode is memory-bound, large-batch decode moves toward compute-bound for the weight matmuls.
- **Comparing perplexity across tokenizers.** Use bits per byte instead.
- **Quoting a pass rate without a margin.** 82% vs 85% on 200 cases is within noise; use a paired test.
- **Computing IDCG only from retrieved items.** It must come from all labeled relevant items, or missed documents vanish from the metric.
- **Treating RRF's $k = 60$, PSI thresholds or Chinchilla's 20 as laws.** They are conventions and empirical rules; say so.
- **Forgetting utilization in self-hosting cost.** A GPU idle at night still costs money; $U = 0.4$ makes the per-token cost 2.5 times higher than $U = 1$.
- **Reciting hardware numbers as exact.** Say "about 3.35 TB/s on an H100 SXM" and note that numbers vary by SKU and generation.

## Interview questions and model answers

These are estimation drills. Answer out loud, write the formula first, state assumptions, then do the arithmetic.

<details>
<summary><b>Q1. How much GPU memory do you need to serve a Llama-3-70B-class model to 32 concurrent users at an 8K context?</b></summary>

**Answer:** Weights first: 70B parameters at 2 bytes in BF16 is 140 GB. Then the KV cache: $2 \times 80 \text{ layers} \times 8 \text{ KV heads} \times 128 \times 2 \text{ bytes} = 327{,}680$ bytes, or 320 KiB per token. For 32 users at 8,192 tokens that is $320 \text{ KiB} \times 8{,}192 \times 32 = 80$ GiB, about 86 GB. Add a few GB per GPU for activations and runtime, so roughly 230 to 240 GB in total. In BF16 that means four 80 GB H100s with tensor parallelism (288 GB usable at 90%), or two 141 GB H200s (about 254 GB usable), which is tight. With FP8 weights (70 GB) and an FP8 KV cache (about 43 GB) the total is about 115 GB, which fits on two H100s, if quality holds on your eval.

**Key points to mention:**

- Use the KV heads (8), not the attention heads (64): without GQA the cache would be 8 times larger.
- KV grows linearly with users and context, so the answer changes completely at 128K context (40 GiB per user).
- Leave headroom: engines reserve memory up front, and fragmentation and CUDA graphs take more.
- Little's law turns a traffic forecast into the "32 concurrent users" number in the first place.

**Deep dive:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md), [Number Formats and Model Memory](../07-model-compression/01-number-formats-and-model-memory.md)

</details>

<details>
<summary><b>Q2. What decode speed would you expect from an 8B model on one H100, and what does batching do to it?</b></summary>

**Answer:** Decode at batch 1 is memory-bandwidth-bound: each step reads all 16 GB of BF16 weights to do about 2 FLOPs per weight. So the ceiling is $3.35 \text{ TB/s} / 16 \text{ GB} \approx 209$ tokens/s, about 4.8 ms per token; at a realistic 60% to 80% of peak bandwidth that is roughly 125 to 170 tokens/s. With 32 sequences at 2K context, each step reads the weights once plus $32 \times 0.27 \text{ GB} \approx 8.6$ GB of KV cache, about 24.6 GB, so about 7.3 ms per step: about 4,350 tokens/s for the system but about 136 tokens/s per user. The compute for that step is only about 0.5 ms at H100 peak, so it is still memory-bound.

**Key points to mention:**

- Formula: tokens/s per sequence $\lesssim BW / (\text{weight bytes} + \text{KV bytes})$.
- Batching amortizes the weight read, raising throughput far more than it hurts per-user speed, until the KV cache dominates the bytes or the SLO is hit.
- INT4 weights cut weight bytes 4x and raise the batch-1 ceiling; they do little for prefill.
- Speculative decoding helps most exactly in this small-batch, memory-bound regime.

**Deep dive:** [GPU Fundamentals for LLMs: Compute-Bound vs Memory-Bound](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

</details>

<details>
<summary><b>Q3. How long would it take to train a 7B model on 2 trillion tokens with 1,024 A100s, and is that compute-optimal?</b></summary>

**Answer:** Training compute is about $6ND = 6 \times 7 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{22}$ FLOPs. An A100 peaks at about 312 TFLOPS in BF16; at 40% MFU that is $1.25 \times 10^{14}$ useful FLOP/s, so $8.4 \times 10^{22} / 1.25 \times 10^{14} / 3{,}600 \approx 187{,}000$ GPU-hours, about 7.6 days on 1,024 GPUs before failures and restarts. Chinchilla's compute-optimal data for 7B is only about $20 \times 7\text{B} = 140$B tokens, so this run is about 14 times past it. That is deliberate: Chinchilla optimizes training compute only, and a smaller, overtrained model is cheaper to serve for its whole life.

**Key points to mention:**

- The chain: $C = 6ND$, divide by peak times MFU, divide by 3,600, divide by GPU count.
- The Llama 2 paper reported about 184,000 A100-hours for its 7B model, so the estimate is within a few percent.
- $6ND$ answers "how long", not "how many GPUs to fit it": memory is about 16 bytes per parameter plus activations.
- Real budgets are several times the final run once ablations and failed runs are included.

**Deep dive:** [How LLMs Are Trained, Part 1: Pretraining and Next-Token Prediction](../05-llms/01-pretraining-and-next-token-prediction.md)

</details>

<details>
<summary><b>Q4. Can you fine-tune a 7B model on a single 24 GB GPU?</b></summary>

**Answer:** Not with full fine-tuning: mixed-precision Adam needs about 16 bytes per parameter, so about 112 GB before activations. LoRA keeps the 14 GB BF16 base frozen and trains about 40M adapter parameters at $r = 16$ on all linear layers, about 0.64 GB of weights, gradients and optimizer states, so about 14.6 GB plus activations: possible with gradient checkpointing and short sequences, but tight. QLoRA stores the base in 4-bit NF4 (3.5 GB plus about 0.11 GB of quantization constants), so about 4.3 GB plus activations, which fits comfortably and leaves room for longer sequences or bigger batches. The trade-off is that QLoRA steps are slower because weights are dequantized on the fly.

**Key points to mention:**

- LoRA parameters per matrix are $r(d_{\text{in}} + d_{\text{out}})$; memory drops far less than the parameter count suggests because the frozen base still sits in memory.
- Activations scale with batch size times sequence length and often dominate what is left.
- Compare quality on your own eval: LoRA and QLoRA usually come close to full fine-tuning on narrow tasks, not always.

**Deep dive:** [Fine-Tuning LLMs: Full Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md)

</details>

<details>
<summary><b>Q5. A prompt change moved your eval from 85% to 82% on 200 cases. Is it a regression?</b></summary>

**Answer:** Not provably. The standard error at $\hat{p} = 0.85$ and $n = 200$ is $\sqrt{0.85 \times 0.15 / 200} \approx 0.025$, so each score has a 95% margin of about $\pm 5$ points. A better test is paired, because both versions ran on the same cases: if 14 cases regressed and 8 improved, McNemar's $\chi^2 = (|14 - 8| - 1)^2 / 22 \approx 1.14$, well below the 3.84 cutoff ($p \approx 0.29$). So I would not block on the aggregate, but I would read the 14 regressions, because they are often concentrated in one slice. To shrink each score's margin to $\pm 3$ points I would need about $1.96^2 \times 0.85 \times 0.15 / 0.03^2 \approx 544$ cases; reliably detecting a 3-point drop takes more than that (size it with a power calculation, where the paired design helps), or I would agree a pre-registered non-inferiority rule.

**Key points to mention:**

- Paired designs cancel per-case difficulty and are far more sensitive than comparing two independent rates.
- More samples per case reduce sampling noise; only more cases reduce case-selection noise.
- Checking many slices at 5% significance produces false alarms; pre-choose the gated metrics.

**Deep dive:** [Building an LLM Evaluation Framework and Regression Tests](../05-llms/11-building-an-llm-evaluation-framework-and-regression-tests.md)

</details>

<details>
<summary><b>Q6. At what volume does self-hosting an open model beat paying for an API?</b></summary>

**Answer:** Convert both to the same unit. The API costs $p_{\text{out}} + k\,p_{\text{in}}$ per million output tokens; with illustrative prices of \$1.50 and \$0.50 and 5 input tokens per output token, that is \$4.00. Self-hosting costs GPU-hours whether used or not: two GPUs (one as a spare) at \$2.50 per hour is \$120 per day. The break-even is $120 / 4 = 30$M output tokens per day. Those two GPUs, at a measured 1,500 tokens/s and sized for a 3x peak, handle up to about 43M per day; beyond that you add GPUs and redo the comparison. Below about 30M per day, or with spiky traffic and low utilization, the API wins on infrastructure cost alone.

**Key points to mention:**

- Utilization dominates: \$0.46 per million at 100% busy becomes \$1.16 at 40%.
- Add people cost, redundancy, upgrades and the quality gap between the API model and the open model.
- Privacy, data residency or customization (many LoRA adapters) can decide it regardless of cost.
- Prompt caching and batch APIs change the API side of the equation.

**Deep dive:** [Optimizing and Benchmarking LLM Latency and Cost](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

</details>

<details>
<summary><b>Q7. How much memory does an HNSW index over 10 million 768-dimensional embeddings need, and how would you shrink it?</b></summary>

**Answer:** The raw vectors are $10^7 \times 768 \times 4$ bytes $= 30.7$ GB in float32. HNSW with $M = 32$ stores up to $2M = 64$ neighbor IDs of 4 bytes on the base layer, about 256 bytes per vector or 2.6 GB, so about 33.3 GB in total. To shrink it: scalar-quantize vectors to int8 (about 10.2 GB with the graph, small recall loss), or use IVF-PQ with 96 one-byte codes per vector (about 0.96 GB of codes) and re-rank the top candidates with full vectors. Tune `efSearch` (or `nprobe` for IVF) at query time to hit the recall target at the latency SLO.

**Key points to mention:**

- Measure recall@k against exact search on your own queries; recall depends on the data.
- `M` and `efConstruction` need a rebuild to change; `efSearch` and `nprobe` are per-query dials.
- Under about 100K to 1M vectors after filtering, exact search may be simplest.

**Deep dive:** [Vector Search Algorithms: Exact kNN, HNSW, IVF and Product Quantization](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md)

</details>

## Key takeaways

- Always answer an estimation question in the same order: formula, assumptions, arithmetic, sanity check.
- Cross-entropy is $-\log$ of the probability of the right answer; KL $= H(P, Q) - H(P)$; perplexity $= e^{\text{loss}}$ and a uniform model scores $V$.
- Precision $= TP/(TP + FP)$, recall $= TP/(TP + FN)$; ROC-AUC is the probability a random positive outscores a random negative; PSI above 0.25 means significant drift.
- Parameters $\approx 12 L d^2$; FLOPs per token $\approx 2N$ (inference) and $6N$ (training); $C \approx 6ND$; Chinchilla is about 20 tokens per parameter, and modern models deliberately go far beyond it.
- Weights $= N \times$ bytes (4 / 2 / 1 / 0.5); full fine-tuning with Adam is about 16 bytes per parameter; LoRA trains $r(d_{\text{in}} + d_{\text{out}})$ per matrix; QLoRA makes the frozen base about 0.5 bytes per parameter.
- KV cache $= 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times$ bytes per token: 128 KiB for a Llama-3-8B shape, 320 KiB for a Llama-3-70B shape in BF16.
- Small-batch decode speed $\lesssim$ bandwidth / bytes read per step; ridge point $=$ peak FLOP/s / bandwidth (about 156 on an A100, 295 on an H100).
- E2E $\approx$ TTFT $+ (N_{\text{out}} - 1) \times$ TPOT; Little's law $L = \lambda W$ gives concurrency; speculative decoding speedup $= (1 - \alpha^{k+1}) / ((1 - \alpha)(kc + 1))$.
- Quantization: $s = (x_{\max} - x_{\min})/(q_{\max} - q_{\min})$, $\hat{x} = s(q - z)$, rounding error at most $s/2$; INT4 at group 128 is about 4.125 bits per weight.
- Retrieval: BM25 ($k_1 = 1.2$, $b = 0.75$), RRF $= \sum 1/(60 + \text{rank})$, Recall@K for the first stage, nDCG@K for the final list; `efSearch` and `nprobe` are the runtime recall-latency dials.
- Eval statistics: SE $= \sqrt{p(1-p)/n}$, $n = z^2 p(1-p)/E^2$, McNemar on discordant pairs, unbiased pass@k.
- Cost: tokens times price, cached tokens at a discount, self-hosting divided by utilization; break-even is a curve, not a point. Hardware numbers are approximate and dated: say so.

## Further reading

- Vaswani et al., 2017. "Attention Is All You Need": scaled dot-product attention.
- Kaplan et al., 2020. "Scaling Laws for Neural Language Models": the $2N$ and $6N$ FLOP estimates.
- Hoffmann et al., 2022. "Training Compute-Optimal Large Language Models" (Chinchilla).
- Williams, Waterman and Patterson, 2009. "Roofline: An Insightful Visual Performance Model for Multicore Architectures".
- Kwon et al., 2023. "Efficient Memory Management for Large Language Model Serving with PagedAttention".
- Leviathan, Kalman and Matias, 2023. "Fast Inference from Transformers via Speculative Decoding".
- Hu et al., 2021. "LoRA: Low-Rank Adaptation of Large Language Models", and Dettmers et al., 2023. "QLoRA: Efficient Finetuning of Quantized LLMs".
- Jacob et al., 2018. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference".
- Robertson and Zaragoza, 2009. "The Probabilistic Relevance Framework: BM25 and Beyond", and Cormack, Clarke and Buettcher, 2009. "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods".
- Chen et al., 2021. "Evaluating Large Language Models Trained on Code" (pass@k), and Miller, 2024. "Adding Error Bars to Evals".
- NVIDIA's official datasheets for the A100, H100, H200 and B200: check the exact SKU, because SXM and PCIe variants differ.

## Next

- **Next chapter:** [Glossary](./11-glossary.md), an A-Z reference for every term used in these formulas.
- **Practice using these numbers:**
  - [Question Bank: LLM Inference, Serving, Quantization and Distillation](./06-question-bank-llm-inference-and-model-compression.md)
  - [Question Bank: Transformers, LLMs, Prompting, Fine-Tuning and Evaluation](./05-question-bank-transformers-and-llms.md)
  - [Question Bank: RAG, Embeddings and Vector Databases](./07-question-bank-rag-and-vector-databases.md)
  - [Mock Interviews with Model Answers](./12-mock-interviews.md)
- **Revisit the derivations that come up most:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md), [GPU Fundamentals for LLMs](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md), [Fine-Tuning LLMs: Full Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md)
- Back to the [course overview](../README.md)
