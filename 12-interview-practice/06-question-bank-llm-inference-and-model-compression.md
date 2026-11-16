# Question Bank: LLM Inference, Serving, Quantization and Distillation

> A revision question bank for LLM inference, decoding, serving, quantization, distillation and pruning, built for rapid self-testing in the last week before a Senior Software Engineer (AI/ML) interview.
>
> **How to use it:** read each question, answer it out loud first (aim for about 2 minutes), and only then open the answer to compare. Mark the questions you fumbled and come back to them the next day.
>
> **Source chapters:** [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md) · [The KV Cache](../06-llm-inference/02-kv-cache-explained.md) · [GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md) · [Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md) · [Speculative Decoding and Caching](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md) · [Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md) · [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md) · [Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md) · [Number Formats](../07-model-compression/01-number-formats-and-model-memory.md) · [Quantization Fundamentals](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md) · [PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md) · [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md) · [Distillation Fundamentals](../07-model-compression/05-knowledge-distillation-fundamentals.md) · [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md) · [Pruning and Choosing a Method](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md)

## Table of contents

- [1. Token generation, prefill, decode and latency metrics](#1-token-generation-prefill-decode-and-latency-metrics)
- [2. Decoding strategies](#2-decoding-strategies)
- [3. The KV cache and GPU memory](#3-the-kv-cache-and-gpu-memory)
- [4. Batching, PagedAttention and scheduling](#4-batching-pagedattention-and-scheduling)
- [5. Speculative decoding and caching](#5-speculative-decoding-and-caching)
- [6. Optimizing and benchmarking latency and cost](#6-optimizing-and-benchmarking-latency-and-cost)
- [7. Serving frameworks and deployment choices](#7-serving-frameworks-and-deployment-choices)
- [8. Number formats and quantization fundamentals](#8-number-formats-and-quantization-fundamentals)
- [9. PTQ, QAT and calibration](#9-ptq-qat-and-calibration)
- [10. Practical LLM quantization: GPTQ, AWQ and outliers](#10-practical-llm-quantization-gptq-awq-and-outliers)
- [11. Knowledge distillation fundamentals](#11-knowledge-distillation-fundamentals)
- [12. Distilling large language models](#12-distilling-large-language-models)
- [13. Pruning, low-rank factorization and choosing a method](#13-pruning-low-rank-factorization-and-choosing-a-method)
- [Rapid-fire round](#rapid-fire-round)
- [How to practice](#how-to-practice)

## 1. Token generation, prefill, decode and latency metrics

<details>
<summary><b>Q1. Explain token generation during inference.</b></summary>

**Answer:** An LLM generates text autoregressively, one token at a time. The prompt is tokenized and run through the transformer; the last position produces a logit for every vocabulary token, softmax turns those into probabilities, and a decoding strategy (greedy, beam search, or sampling with temperature, top-k, top-p) picks one token. That token is appended and the loop repeats until an end-of-sequence token, a stop sequence or `max_tokens`. In practice this splits into two phases: prefill processes the whole prompt in one parallel pass and fills the KV cache, and decode then runs one forward pass per new token, attending to the cached keys and values. The consequence for systems is that output tokens are produced sequentially, so a 500-token answer costs 500 decode steps no matter how fast the GPU is.

**Key points to mention:**

- Name both phases (prefill and decode) and the KV cache, not just "it predicts the next token".
- The sampler is cheap; the forward passes dominate cost.
- Stop conditions matter in production: a missing stop token or wrong chat template leads to runaway outputs.
- Follow-up hook: why output tokens cannot be computed in parallel (they depend on previously sampled tokens), and how speculative decoding approximates it.

**Deep dive:** [Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md), [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md)

</details>

<details>
<summary><b>Q2. What are the prefill and decode phases, and why do they behave so differently?</b></summary>

**Answer:** Prefill processes all prompt tokens in one parallel forward pass, builds the KV cache and produces the first output token. Because every loaded weight is reused across hundreds or thousands of prompt tokens, its arithmetic intensity is high and it is compute-bound. Decode generates one token per forward pass, and each step must stream all the weights (and the KV cache) from HBM to do roughly 2 FLOPs per weight, so at small batch it is memory-bandwidth-bound. For a 7B FP16 model on an A100, a 1,000-token prefill is about 1.4 x 10^13 FLOPs (roughly 90 ms at 50% MFU), while each decode step must read 14 GB, at least about 7 ms. Prefill drives time to first token; decode drives inter-token latency and usually most of the end-to-end time.

**Key points to mention:**

- Different bottlenecks mean different fixes: batching and weight quantization help decode; prefix caching and chunked prefill help prefill.
- A very short prompt (under roughly 150 tokens on an A100) can still be memory-bound in prefill.
- Under continuous batching both phases share the GPU, so a long prefill can stall everyone's decode.
- Quote the A100 ridge point (about 153 FLOPs per byte in FP16) to justify the regimes.

**Deep dive:** [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md), [GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

</details>

<details>
<summary><b>Q3. What are time to first token (TTFT) and inter-token latency (ITL)?</b></summary>

**Answer:** TTFT is the time from sending a request to receiving the first output token; it includes network, tokenization, queueing and prefill. ITL is the time between consecutive output tokens of one request and reflects decode step time. TPOT (time per output token) is the per-request average of those gaps, (E2E - TTFT) / (N_out - 1), and 1/TPOT is the user's tokens per second. End-to-end latency is roughly TTFT + (N_out - 1) x TPOT, which is why long answers are slow even with a fast TTFT. Typical chat SLOs look like TTFT under 500 ms and TPOT under 50 ms at p95.

**Key points to mention:**

- TTFT is what the user feels as "responsiveness"; TPOT is reading speed (a typical adult reads roughly 4 to 7 tokens per second, so beyond that faster ITL matters less than TTFT).
- Measure both at percentiles, not averages.
- A large TTFT tail usually means queueing or very long prompts, not a slow model.
- Streaming improves perceived latency (user sees text after TTFT) but does not reduce E2E.

**Deep dive:** [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md)

</details>

<details>
<summary><b>Q4. How would you distinguish prefill latency from decode latency?</b></summary>

**Answer:** Start with streaming timestamps per request: TTFT minus queue and network time approximates prefill, and the gaps between later tokens measure decode. Then run controlled experiments: sweep prompt length with `max_tokens=1` to isolate prefill cost, and sweep output length with a short fixed prompt to get the per-token decode slope. Cross-check with server-side metrics (queue time, prefill time, decode time per request, batch size) that engines such as vLLM expose. If the numbers still do not add up, use a profiler timeline to see which kernels dominate. The result is a latency budget that tells you whether to attack the prompt (caching, trimming, chunked prefill) or the output (shorter answers, quantization, speculative decoding).

**Key points to mention:**

- Separate queueing from prefill: high TTFT under load is often queueing.
- Use controlled sweeps, not just production averages.
- Prefill time grows with prompt length; decode time grows with output length and slightly with context length.
- Report both at p50 and p95/p99.

**Deep dive:** [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md), [GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

</details>

<details>
<summary><b>Q5. How do context (prompt) length and output length affect latency?</b></summary>

**Answer:** Prompt length is paid mostly once, during prefill, so it mainly drives TTFT: roughly linear at moderate lengths and superlinear at very long lengths because attention is quadratic in sequence length. Output length is paid per token during decode, so it multiplies end-to-end latency and is usually the dominant factor: 500 output tokens at 20 ms each is 10 s regardless of prompt size. Long context also slows every decode step a little, because each step reads a larger KV cache, and it reduces how many requests fit in GPU memory, which lowers throughput. So the cheapest latency win is usually cutting output tokens (concise instructions, structured outputs, tight `max_tokens`), then trimming prompts and caching shared prefixes.

**Key points to mention:**

- Prompt length to TTFT, output length to E2E, context length to per-step ITL and capacity.
- Long contexts hurt concurrency via KV cache memory, not just latency.
- Workload shape matters: RAG is prefill-heavy, chat and reasoning are decode-heavy.
- Give a back-of-envelope number to show you can reason quantitatively.

**Deep dive:** [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md), [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

</details>

<details>
<summary><b>Q6. What is the difference between throughput and latency?</b></summary>

**Answer:** Latency is how long one request takes from the user's point of view (TTFT, ITL, E2E). Throughput is how much work the whole system completes per unit time (output tokens per second or requests per second), and it sets cost per token. They trade off: batching more requests raises throughput because decode reuses each weight read across the batch, but each step gets slower, so per-user latency rises. In a toy model of a 7B decode, batch 1 gives about 97 tokens/s per user and system, while batch 64 gives about 34 tokens/s per user but about 2,200 tokens/s for the system. The goal is maximum throughput that still meets the latency SLO, which is goodput.

**Key points to mention:**

- Little's law connects them: concurrency = arrival rate x time in system.
- The latency SLO picks the batch size, not the other way round.
- Batching helps decode a lot and prefill little.
- Never report throughput without the latency it was achieved at.

**Deep dive:** [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md), [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

</details>

<details>
<summary><b>Q7. What is goodput, and why is it a better capacity metric than raw throughput?</b></summary>

**Answer:** Goodput is the rate of requests (or tokens) completed within their latency SLOs, for example TTFT under 500 ms and TPOT under 50 ms. Raw throughput keeps rising as you push more load, even while latency becomes unacceptable, because huge batches are efficient for the GPU but terrible for users. Goodput peaks and then falls as SLO violations grow, so its peak identifies the real usable capacity of a replica. Capacity planning should therefore ask "how many requests per second can one replica serve while 90% or 99% of requests meet the SLO?" and size the fleet from that number plus headroom.

**Key points to mention:**

- Define the SLO first (metric, percentile, threshold).
- Measure with a rate sweep on realistic traffic.
- Use goodput to compare configurations, engines and compressed models fairly.
- Mention SLO attainment as the related percentage metric.

**Deep dive:** [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md), [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

</details>

## 2. Decoding strategies

<details>
<summary><b>Q8. What is the difference between greedy decoding and beam search?</b></summary>

**Answer:** Greedy decoding picks the single most probable token at each step and keeps one sequence. Beam search keeps the B most probable partial sequences at each step, scored by summed log probabilities, so it can find a sequence whose first token was not the top choice but whose total probability is higher; greedy is beam search with B = 1. Beam search costs about B times the decode compute and KV cache memory and cannot stream reliably because the leading beam can change. For open-ended chat, the highest-probability text tends to be bland and repetitive, so beam search is rarely used there. It remains a good fit for tasks with roughly one correct output, such as translation, speech recognition and some constrained generation.

**Key points to mention:**

- Both are deterministic search methods, unlike sampling.
- Beam search needs length normalization or it favors short outputs.
- Cost: B x KV cache and compute, lower throughput.
- Maximizing probability is not the same as maximizing quality for open-ended text.

**Deep dive:** [Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md)

</details>

<details>
<summary><b>Q9. Compare greedy decoding, temperature, top-k and top-p sampling.</b></summary>

**Answer:** Greedy always takes the argmax: deterministic, good for factual and structured tasks, but bland and loop-prone on long outputs. Temperature divides the logits by T before softmax, sharpening the distribution when T < 1 and flattening it when T > 1; it controls randomness but does not remove the long tail of bad tokens. Top-k samples only from the k most likely tokens, a fixed-size cut that ignores how confident the model is. Top-p (nucleus) samples from the smallest set whose cumulative probability reaches p, an adaptive cut that keeps few tokens when the model is confident and many when it is not. In practice they are combined, for example T = 0.7 with top-p = 0.9 for chat, and near-zero temperature for extraction or code.

**Key points to mention:**

- Order of operations: penalties, temperature, then truncation (top-k, top-p, min-p), then sampling.
- Top-k = 1 or T approaching 0 both reduce to greedy.
- Min-p scales the threshold with the top token's probability and behaves better at high temperature.
- Pick settings per task and validate them on an eval, do not copy defaults blindly.

**Deep dive:** [Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md)

</details>

<details>
<summary><b>Q10. What does repetition penalty do?</b></summary>

**Answer:** A repetition penalty lowers the logits of tokens that have already appeared, so the model is less likely to repeat itself. In the CTRL-style version used by Hugging Face and vLLM, positive logits of seen tokens are divided by a penalty greater than 1 and negative logits are multiplied by it, so both move toward "less likely". OpenAI-style APIs instead offer additive penalties: a frequency penalty that subtracts per occurrence and a presence penalty that subtracts once if the token appeared at all. A mild value (around 1.05 to 1.2) can stop loops; strong values hurt quality because legitimate repetition (variable names in code, entity names, JSON keys) gets punished.

**Key points to mention:**

- It treats a symptom; check the root cause too (greedy on long outputs, wrong chat template, missing stop token).
- Strong penalties break code and structured output.
- Distinguish multiplicative (CTRL-style) from additive (frequency and presence) penalties.
- Monitor `finish_reason == "length"` and distinct-n to detect loops in production.

**Deep dive:** [Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md)

</details>

<details>
<summary><b>Q11. How do these decoding parameters affect diversity and factual reliability?</b></summary>

**Answer:** Higher temperature, larger top-p or top-k and stronger penalties increase diversity, but they raise the probability of picking low-probability tokens, which can introduce factual errors, topic drift or broken formats. Lower temperature, tighter truncation and no penalties make output more consistent and usually more reliable for factual, extraction and structured tasks, at the cost of blandness, repetition and no variety across samples. The important nuance is that conservative settings only make the output match the model's own beliefs more closely; they do not make those beliefs correct. So decoding settings are a variance knob, while factual reliability comes from grounding (retrieval, citations), better models and evaluation.

**Key points to mention:**

- Match settings to task type: extraction and code near 0, chat moderate, creative higher.
- Diversity is useful for self-consistency and sampling multiple candidates.
- Validate settings with multiple samples per prompt, measuring quality and variance.
- Format validity is a separate concern best solved with constrained decoding.

**Deep dive:** [Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md)

</details>

<details>
<summary><b>Q12. Why can low temperature not guarantee factual correctness?</b></summary>

**Answer:** Temperature only reshapes the model's existing probability distribution; it cannot add knowledge the model does not have. Low temperature makes the model pick its most likely token more consistently, but the most likely token can be wrong, because a language model learns what text is likely, not what is true. If the model has learned a wrong fact, or the question is outside its knowledge, it will be confidently and consistently wrong at temperature 0. In short, temperature controls variance, not bias. Reducing hallucinations needs grounding (RAG with citations), abstention behavior, better or fine-tuned models, and faithfulness evaluation.

**Key points to mention:**

- "Variance, not bias" is the crisp one-liner.
- Greedy can still hallucinate fluent, plausible text.
- Even temperature 0 is not fully deterministic on GPUs.
- Mention evaluation of faithfulness and answer correctness as the real control.

**Deep dive:** [Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md)

</details>

<details>
<summary><b>Q13. You set temperature to 0, but the same prompt returns different answers in production. Why?</b></summary>

**Answer:** Greedy decoding is only deterministic in exact arithmetic. Floating-point addition is not associative, and GPU kernels sum values in orders that depend on batch size, batch composition, sequence length, kernel choice, hardware and parallelism. Under continuous batching the batch changes every step, so logits can differ in the last bits; when two tokens are nearly tied, the argmax flips, and after one different token the rest of the output diverges. Hosted APIs add more variation: different hardware pools, load-dependent batching and model updates behind aliases. If you need reproducibility, pin model versions, use batch-invariant or deterministic kernels where available, cache responses, and design evals to tolerate small variation.

**Key points to mention:**

- Non-associativity of floating-point sums is the root cause.
- Near-ties plus autoregression amplify a tiny difference into a different answer.
- Seeds do not fix it when batch composition changes.
- Test and eval pipelines should compare semantics, not exact strings.

**Deep dive:** [Decoding Strategies](../05-llms/03-text-generation-and-decoding-strategies.md), [Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md)

</details>

## 3. The KV cache and GPU memory

<details>
<summary><b>Q14. What is a KV cache, and why is it important?</b></summary>

**Answer:** During autoregressive generation the KV cache stores the key and value vectors of every previous token at every layer, so each decode step only computes K and V for the newest token and reuses the rest. Without it, every step would recompute the whole sequence, making generation cost grow quadratically with output length instead of linearly. It matters for two reasons: it is what makes generation fast enough to be practical, and it is the main variable memory cost in serving, so it determines how many concurrent requests and how much context a GPU can handle. For a 7B-class multi-head attention model in FP16 it costs 512 KiB per token, so one 4K-token sequence needs 2 GiB.

**Key points to mention:**

- Trade compute for memory.
- The cache grows with every token of every request; weights are a fixed cost.
- It drives capacity planning, PagedAttention, GQA, KV quantization and prefix caching.
- Each decode step also has to read the whole cache, so it affects ITL too.

**Deep dive:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md)

</details>

<details>
<summary><b>Q15. Which tensors are stored in the KV cache?</b></summary>

**Answer:** Only keys and values: for every layer, every KV head and every past token (prompt and generated tokens alike). Per layer, K and V are each typically shaped `[batch, n_kv_heads, seq_len, head_dim]`. Queries are not stored because each query is only used once, at its own step, and attention outputs and MLP activations are not stored either because later tokens never need them. With grouped-query attention, the stored head count is the number of KV heads (for example 8), not the number of query heads (for example 32), which is exactly why GQA shrinks the cache.

**Key points to mention:**

- Why no queries: the new token's query attends to old keys, old queries are never reused.
- Shape and per-layer storage.
- Under tensor parallelism each GPU stores only its share of KV heads.
- Serving engines store it in blocks (PagedAttention), not one contiguous tensor.

**Deep dive:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md)

</details>

<details>
<summary><b>Q16. How does the KV cache reduce repeated computation during autoregressive generation?</b></summary>

**Answer:** Without a cache, step t runs all P + t - 1 tokens through every layer even though only the last position's output is used to pick the next token. With a cache, the model processes the prompt once during prefill, then each decode step feeds in only the newest token: it computes that token's q, k and v, appends k and v to the cache, and attends over all cached keys and values. Total token passes drop from N x P + N(N - 1)/2 to P + N - 1 for a prompt of P tokens and N generated tokens. The attention step itself still scales with the current context length, but the expensive projections and MLPs run on one token per step.

**Key points to mention:**

- Causal masking makes this valid: past tokens' K and V never change.
- Quadratic to linear total cost in output length.
- The cost moves to memory and memory bandwidth.
- Speculative decoding and prefix caching both build on the same reuse idea.

**Deep dive:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md)

</details>

<details>
<summary><b>Q17. How does KV-cache memory scale with layers, heads, sequence length and batch size?</b></summary>

**Answer:** Linearly in each: M = 2 x n_layers x n_kv_heads x head_dim x seq_len x batch x bytes_per_element, where the 2 is for keys and values. For a 7B-class MHA model (32 layers, 32 KV heads, head_dim 128, FP16) that is 524,288 bytes, or 512 KiB per token, so a 4K sequence is 2 GiB and a batch of 26 such sequences fills about 56 GB. An 8B-class GQA model with 8 KV heads needs only 128 KiB per token, so 1 GiB per 8K sequence. Levers to shrink it follow directly from the formula: fewer KV heads (GQA, MQA), fewer bytes (FP8 or INT8 KV cache), shorter contexts or sliding windows, and smaller batches.

**Key points to mention:**

- Write the formula and compute one example out loud.
- Note it uses KV heads, not query heads.
- Each factor maps to an optimization (GQA, KV quantization, sliding window, admission control).
- Total memory = weights + KV cache + activations + framework overhead.

**Deep dive:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md)

</details>

<details>
<summary><b>Q18. What are multi-query attention and grouped-query attention?</b></summary>

**Answer:** They are attention variants that keep many query heads but reduce the number of key and value heads. In MQA (Shazeer, 2019) all query heads share a single K and V head; in GQA (Ainslie et al., 2023) query heads are split into groups that each share one K and V head, sitting between full multi-head attention and MQA. Because the KV cache is proportional to the number of KV heads, they shrink cache memory and the bytes read per decode step by the ratio of query heads to KV heads: 32 query heads with 8 KV heads is a 4x reduction. GQA keeps quality close to MHA while capturing most of MQA's savings, which is why most modern open models use it.

**Key points to mention:**

- Memory and bandwidth savings both matter: more concurrent requests and faster decode.
- MQA can lose quality; GQA is the practical compromise.
- Existing MHA checkpoints can be uptrained into GQA.
- Interaction with tensor parallelism: KV heads are split across GPUs.

**Deep dive:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md), [Modern LLM Architecture](../04-transformers/08-modern-llm-architecture-improvements.md)

</details>

<details>
<summary><b>Q19. What happens when the context window or concurrency increases substantially?</b></summary>

**Answer:** Total tokens in flight grow, so the KV cache grows linearly and hits the memory budget sooner. A naive server simply runs out of memory. Modern engines cap concurrency instead, so throughput falls and requests queue, which shows up as rising TTFT; if they over-admit, they must preempt running requests by recomputing or swapping their cache, causing latency spikes. Even before limits are hit, each decode step reads more KV bytes, so inter-token latency rises, and longer prompts raise prefill time and TTFT. Mitigations include GQA models, FP8 KV cache, prefix caching, chunked prefill, admission control on `max_model_len`, more GPUs or tensor parallelism, and routing long-context requests to dedicated replicas.

**Key points to mention:**

- Symptoms: queueing, preemptions, ITL growth, OOM in naive stacks.
- Monitor KV cache utilization, waiting requests and preemption counters.
- Long-context and short-context traffic may need separate pools.
- Validate any KV quantization on long-context evals.

**Deep dive:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md), [Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md)

</details>

<details>
<summary><b>Q20. Walk me through the GPU memory budget for serving. Is the KV cache the dominant consumer, and how many concurrent requests fit?</b></summary>

**Answer:** GPU memory holds weights, the KV cache, activation workspace and framework overhead (CUDA context, CUDA graphs). Weights are fixed; the KV cache grows with every token of every request, so for one short request weights dominate, but in production with long contexts or many users the KV cache usually dominates and sets capacity. Example: an 8B-class GQA model on an 80 GB GPU: about 74.5 GiB total, about 67 GiB usable at 90% utilization, minus about 15 GiB of FP16 weights and a few GiB of overhead leaves about 48 GiB of KV. At 128 KiB per token a 16K-token request needs 2 GiB, so about 24 concurrent requests in FP16, or about 48 with an FP8 cache. The crossover where the KV cache exceeds the weights is around 122K cached tokens in total for that model, but only about 27K for a 7B-class MHA model.

**Key points to mention:**

- Separate fixed costs (weights) from variable costs (KV per token).
- Quantizing weights frees memory for KV, which often matters more than the speedup.
- State assumptions (utilization cap, overhead) explicitly; they are illustrative.
- Concurrency limit = KV budget / (tokens per request x bytes per token).

**Deep dive:** [The KV Cache](../06-llm-inference/02-kv-cache-explained.md), [GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

</details>

<details>
<summary><b>Q21. Is LLM inference compute-bound or memory-bandwidth-bound? Explain with the roofline model.</b></summary>

**Answer:** Arithmetic intensity is FLOPs per byte moved from memory, and the roofline says attainable performance is min(peak FLOPs, intensity x bandwidth). The ridge point, peak FLOPs divided by bandwidth, separates the regimes: about 153 FLOPs per byte for an A100 in FP16 and about 295 for an H100 in dense BF16. Prefill reuses each weight across many prompt tokens, so its intensity is roughly the number of tokens processed and, beyond a few hundred tokens, it is compute-bound. Decode at small batch is a matrix-vector product, about 1 FLOP per byte in FP16, far below any ridge point, so it is memory-bandwidth-bound and step time is about weight bytes divided by bandwidth. Batching raises decode intensity to roughly the batch size, which is why it is nearly free throughput until the batch approaches the ridge point.

**Key points to mention:**

- Answer "it depends on the phase and the batch size", then prove it with arithmetic.
- Confirm by measurement: step time scaling with bytes versus with FLOPs.
- Use MBU for decode and MFU for prefill and training.
- Lower precision helps whichever resource is the bottleneck only if kernels exploit it.

**Deep dive:** [GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

</details>

<details>
<summary><b>Q22. Estimate the best possible decode speed for a 7B model in FP16 on one A100 at batch 1. How do INT4 weights and a long context change it?</b></summary>

**Answer:** Batch-1 decode is bandwidth-bound, so time per token is at least weight bytes divided by bandwidth: 14 GB / about 2 TB/s is about 7 ms, so roughly 143 tokens per second at best. INT4 weights are about 3.5 GB, so the bound drops to about 1.7 ms per token (about 571 tokens per second), up to 4x faster in theory, although real gains are smaller because of scales, dequantization and kernels that do not reach peak bandwidth. A 4,096-token context with full multi-head attention adds about 2.15 GB of KV reads per step, lowering the FP16 bound to about 124 tokens per second. This is the core argument for why weight-only quantization speeds up small-batch decode and why long contexts slow it down.

**Key points to mention:**

- Show the formula: bytes read per step / bandwidth.
- Real systems reach maybe 60-80% of peak bandwidth.
- KV cache reads add to weight reads, more so at long context and large batch.
- CPU offloading over PCIe (about 32 GB/s) would be about 0.44 s per token, which is why it is so slow.

**Deep dive:** [GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md), [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md)

</details>

## 4. Batching, PagedAttention and scheduling

<details>
<summary><b>Q23. How do batching and continuous batching differ?</b></summary>

**Answer:** Classic static or dynamic batching groups requests at the request level: a batch is formed, runs until every member finishes, and only then can new requests start. With variable output lengths this wastes slots, because short requests finish early and their slots sit idle while the longest one keeps going, and new requests suffer head-of-line blocking. Continuous batching (from Orca, 2022) schedules at the iteration level: after every decode step, finished sequences leave and waiting requests join. That keeps the batch full, gives much higher throughput and lowers queueing delay for generative workloads. It is the default in vLLM, SGLang, TensorRT-LLM and TGI.

**Key points to mention:**

- Static batching is fine for fixed-shape models (classifiers), not for generation.
- Iteration-level scheduling is the key phrase.
- Side effect: prefill and decode share iterations, causing interference (fixed by chunked prefill).
- Batch composition changes every step, which affects determinism.

**Deep dive:** [Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md)

</details>

<details>
<summary><b>Q24. What are KV cache, paged attention and speculative decoding?</b></summary>

**Answer:** The KV cache stores each processed token's attention keys and values so decode does not recompute them; it grows with every token and dominates serving memory. PagedAttention (Kwon et al., 2023, the core of vLLM) stores that cache in fixed-size blocks, for example 16 tokens, addressed through a per-sequence block table, like virtual memory pages in an operating system, so memory is allocated on demand, waste drops to under one block per sequence, and blocks can be shared across sequences. Speculative decoding is a decode-latency optimization: a cheap draft proposes k tokens, the large target model verifies them all in one forward pass, and a rejection-sampling acceptance rule keeps the agreeing prefix. Because small-batch decode is memory-bound, verifying several tokens costs about the same as generating one, and the output distribution is exactly the target model's.

**Key points to mention:**

- Three different problems: recomputation (KV cache), memory waste (PagedAttention), sequential latency (speculative decoding).
- PagedAttention enables larger batches and prefix sharing.
- Speculative decoding helps at low load and can hurt at high load.
- Typical speculative speedups are 1.5x to 3x depending on acceptance rate.

**Deep dive:** [Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md), [Speculative Decoding and Caching](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md)

</details>

<details>
<summary><b>Q25. How do paged attention and continuous batching improve serving efficiency?</b></summary>

**Answer:** Decode is memory-bound, so throughput comes from large batches that amortize each weight read across many sequences. Continuous batching keeps the batch full at every step by letting requests join and leave per iteration. PagedAttention makes the batch able to be large: older systems reserved contiguous max-length buffers per request, wasting most KV memory through reservation and fragmentation, while paging allocates blocks on demand, nearly eliminates waste, allows overcommit with preemption, and lets sequences share blocks for common prefixes and parallel sampling. One keeps the slots occupied; the other creates more slots. Together they gave vLLM multi-fold throughput gains over earlier serving systems on the same hardware.

**Key points to mention:**

- "One fills the slots, the other creates more slots."
- Copy-on-write block sharing for beam search and `n > 1` sampling.
- Preemption becomes necessary when overcommitted.
- Both are about raising effective batch size under a fixed memory budget.

**Deep dive:** [Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md), [The KV Cache](../06-llm-inference/02-kv-cache-explained.md)

</details>

<details>
<summary><b>Q26. A long-prompt request arrives and other users' streams freeze for a moment. What is happening and how do you fix it?</b></summary>

**Answer:** That is prefill-decode interference. With continuous batching, the new request's prefill runs in the same iteration as everyone else's decode step, so one huge prefill (say 30K tokens) makes that iteration take hundreds of milliseconds and every stream stalls. Confirm it by correlating ITL spikes with the arrival of long prompts in traces. Chunked prefill splits the long prompt into pieces processed across several iterations under a per-iteration token budget (`max_num_batched_tokens` in vLLM), mixed with decode tokens, which smooths ITL and pairs compute-bound prefill with memory-bound decode; it slightly increases TTFT for the long request. At large scale with strict SLOs, disaggregated serving puts prefill and decode on separate GPU pools, removing interference at the cost of KV transfer and operational complexity.

**Key points to mention:**

- Symptom is periodic ITL spikes, not slow TTFT.
- Chunked prefill is the first fix; disaggregation is for large fleets.
- Tune the token budget against the ITL SLO.
- Disaggregation lets you scale and tune each phase separately.

**Deep dive:** [Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md), [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md)

</details>

<details>
<summary><b>Q27. When the KV cache is full, what does the engine do? Compare recompute and swap preemption.</b></summary>

**Answer:** The scheduler preempts one or more running sequences, often the most recently admitted, to free blocks for the others. With recompute, it drops their KV blocks and later re-runs prefill over the prompt plus already-generated tokens, which costs GPU compute but no extra memory. With swap, it copies their KV blocks to CPU memory and back later, which costs PCIe bandwidth and CPU RAM. Either way, the preempted requests see a latency spike, so frequent preemption is a sign of over-admission: too high `max_num_seqs`, too long `max_model_len`, or too little KV memory. The fix is to reduce concurrency caps, add KV capacity (quantize weights or KV, use more GPUs), or route long requests elsewhere.

**Key points to mention:**

- Preemption counters are a key serving metric.
- Recompute is usually preferred for short sequences, swap for long ones.
- Preemption shows up as tail latency, not average latency.
- Relates directly to the p99 debugging story.

**Deep dive:** [Batching and PagedAttention](../06-llm-inference/04-batching-continuous-batching-and-paged-attention.md)

</details>

## 5. Speculative decoding and caching

<details>
<summary><b>Q28. Why does speculative sampling produce exactly the target model's distribution, and when does speculative decoding hurt?</b></summary>

**Answer:** It is rejection sampling. A draft token x proposed with draft probability q(x) is accepted with probability min(1, p(x)/q(x)), where p is the target model's probability; on rejection, a token is resampled from the normalized residual max(0, p - q). The probability of emitting x is min(p(x), q(x)) from acceptance plus max(0, p(x) - q(x)) from resampling, which sums to p(x), so quality is unchanged. The speedup comes from using idle compute while the GPU waits on memory: with acceptance rate 0.6, k = 5 and a draft costing 10% of a target pass, expected tokens per iteration are (1 - 0.6^6)/(1 - 0.6), about 2.38, for a cost of 1.5 target passes, so about 1.59x. At large batch the GPU is already compute-bound from other users' tokens, so verifying k + 1 tokens per sequence multiplies FLOPs, rejected drafts are pure waste, and the draft model takes memory away from the KV cache; speculative decoding then helps little or hurts.

**Key points to mention:**

- Lossless in distribution (greedy output identical up to floating-point effects).
- Helps per-user latency at low to moderate load and with predictable text.
- TTFT is unchanged or slightly worse (draft must prefill too).
- Variants: separate draft model, Medusa heads, EAGLE, prompt lookup (n-gram) decoding.

**Deep dive:** [Speculative Decoding and Caching](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md)

</details>

<details>
<summary><b>Q29. What is prefix caching, when does it help, and why might the hit rate be low?</b></summary>

**Answer:** Prefix caching keeps the KV blocks of previously processed prompt prefixes, keyed by chained hashes of their tokens, and reuses them when a new request starts with exactly the same tokens, skipping that part of prefill. It helps when many requests share long system prompts, tool definitions, few-shot examples, documents or chat history, cutting TTFT and compute substantially. SGLang's RadixAttention generalizes it with a radix tree and a cache-aware scheduler. Low hit rates usually come from dynamic content near the start of the prompt (a timestamp or user name in the system prompt), non-deterministic serialization (JSON key order), requests for the same prefix landing on different replicas, or eviction under memory pressure.

**Key points to mention:**

- Put static content first and dynamic content last.
- Use prefix-aware or session-affinity routing across replicas.
- It is exact-match on tokens, unlike semantic caching.
- Provider prompt caching in hosted APIs is the same idea, priced per cached token.

**Deep dive:** [Speculative Decoding and Caching](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md), [The KV Cache](../06-llm-inference/02-kv-cache-explained.md)

</details>

<details>
<summary><b>Q30. How would you use response caching and model routing to cut cost, and what can go wrong?</b></summary>

**Answer:** Put an exact-match response cache first, then a semantic cache that returns a stored answer when a new query's embedding is close enough to a previous one. Partition it by tenant, model version and prompt version, tune the similarity threshold on labeled pairs, add TTLs and invalidation when source documents change, and cache only non-personalized, single-turn, side-effect-free answers. For routing, a router picks a model before generating so each query pays for one model, while a cascade tries the small model first and escalates on low confidence, so escalated queries pay twice but the decision uses the actual answer. The main risks are wrong semantic matches, stale answers, cross-tenant leakage and routers that silently send hard queries to the weak model.

**Key points to mention:**

- Evaluate routers and caches by plotting quality against cost across thresholds.
- Never share a semantic cache across tenants.
- Routing easy traffic to a small model is often the single biggest cost lever.
- Track cache hit rate and the quality of cache hits separately.

**Deep dive:** [Speculative Decoding and Caching](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md)

</details>

## 6. Optimizing and benchmarking latency and cost

<details>
<summary><b>Q31. How would you optimize LLM inference latency and cost?</b></summary>

**Answer:** First define the target: which metric (TTFT, ITL, E2E, cost per token), which percentile, at what load, and with what quality bar, then build a realistic benchmark plus tracing to see where time and money go. Then apply fixes from cheapest to most invasive: reduce tokens (shorter prompts, much shorter outputs, structured outputs), add caching (prefix KV caching, then exact or semantic response caching), tune the serving engine and continuous batching, quantize (FP8 or INT4 weight-only), add speculative decoding for low-load latency, and consider a smaller, distilled or routed model; change hardware last. After each step re-run both the latency benchmark and the quality eval, because a speedup that breaks accuracy is not a win. For example, cutting a 600-token answer to 200 tokens cuts decode time, usually most of E2E, by about 3x before touching the model.

**Key points to mention:**

- Measure first, then optimize the dominant term in the latency budget.
- Distinguish latency goals (speculative decoding, smaller model) from cost goals (batching, utilization).
- Output tokens are the most expensive tokens.
- Quality gate after every change.

**Deep dive:** [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

</details>

<details>
<summary><b>Q32. A model is producing accurate answers, but latency is too high. How would you optimize it?</b></summary>

**Answer:** Clarify which latency is too high (TTFT, ITL or E2E) and the target, then trace a slow request to build a latency budget across network, retrieval, queueing, prefill and decode. Usually decode dominates, so I first stream the output and cut output length, then trim the prompt and retrieved context, enable prefix caching for the static prompt, and fix queueing with more capacity or chunked prefill. Next I reduce per-token time with weight quantization or speculative decoding. If that is still not enough, I evaluate a smaller fine-tuned or distilled model, or route easy requests to one. Because "accurate" is the constraint we must not break, every change is re-checked on the eval set.

**Key points to mention:**

- Ask clarifying questions first: which metric, which percentile, what load.
- Streaming fixes perceived latency cheaply.
- Look outside the model too: retrieval, tool calls, network, tokenization.
- Keep the eval set as a regression gate.

**Deep dive:** [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md), [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md)

</details>

<details>
<summary><b>Q33. Would you use batching, a smaller model, speculative decoding, quantization or caching?</b></summary>

**Answer:** It depends on the failing metric and the regime, and they are often combined. Batching (continuous batching) raises throughput and lowers cost when the GPU is underutilized, but it does not make a single user faster and slightly raises per-request latency. Caching helps when prefixes or whole questions repeat. Quantization helps memory-bound decode and frees memory for more KV cache. Speculative decoding cuts per-token latency at low to moderate load without changing the output distribution, but helps little at high load. A smaller model helps latency, throughput and cost at once and is the biggest lever, but carries the most quality risk, so it needs an evaluation gate.

**Key points to mention:**

- Map each technique to the metric it moves.
- Measure TTFT vs TPOT, queue time, GPU utilization and batch size before choosing.
- Techniques stack: quantized model + prefix caching + continuous batching is standard.
- Speculative decoding and large batches compete for the same idle compute.

**Deep dive:** [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md), [Speculative Decoding and Caching](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md)

</details>

<details>
<summary><b>Q34. When would you use a smaller model instead of optimizing a larger one?</b></summary>

**Answer:** When the smaller model, possibly fine-tuned or distilled for the task, stays within the quality tolerance on my own eval set, and the latency or cost target cannot be reached by optimizing the large model with the cheap techniques. This is most common for narrow, high-volume tasks such as classification, extraction, routing and fixed-format summarization, where a 1-8B model fine-tuned on task data often matches a much larger general model. Going from a 70B to an 8B model cuts weight bytes by about 9x, which directly cuts decode time and GPU count. The decision should be made with a cost-per-successful-task comparison, not on benchmark scores alone. Routing is the middle ground: send easy traffic to the small model and hard traffic to the large one.

**Key points to mention:**

- The eval set decides, not intuition.
- Fine-tuning or distillation closes much of the gap on narrow tasks.
- Consider routing and cascades as hybrids.
- Watch for slice regressions hidden by averages.

**Deep dive:** [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md), [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md)

</details>

<details>
<summary><b>Q35. How would you benchmark p50, p95 and p99 latency under concurrent requests?</b></summary>

**Answer:** Replay a realistic workload: input and output length distributions from production, realistic prefix sharing and the same sampling settings. Use an open-loop load generator at a fixed arrival rate (for example Poisson arrivals) and measure from the scheduled send time to avoid coordinated omission. Discard warmup, collect enough samples for the tail (thousands of requests for a stable p99), and sweep the arrival rate to find the knee where latency explodes. Report TTFT, ITL and E2E percentiles separately, along with throughput, goodput, hardware, engine version and configuration, and repeat runs to show variance.

**Key points to mention:**

- Open-loop vs closed-loop load, and coordinated omission.
- Percentiles need sample size: p99 of 100 requests is basically the max.
- Make sure the client is not the bottleneck.
- Benchmark tools: vLLM's benchmark scripts, GenAI-Perf, or a custom harness.

**Deep dive:** [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

</details>

<details>
<summary><b>Q36. Your p50 latency is fine but p99 is 10x worse. How do you investigate?</b></summary>

**Answer:** First confirm the measurement is trustworthy: enough samples, open-loop load, no client-side bottleneck. Then break slow requests into network, queue, prefill and decode time, and correlate them with request attributes: input length, output length, tenant, replica, time of day and engine events. Common causes are very long prompts or answers, long prefills interrupting other streams, KV cache preemption or swapping under memory pressure, queueing during bursts, retries, cold replicas and noisy neighbors. Fixes follow the cause: chunked prefill, admission limits on context length, more KV capacity, separate pools for long requests, or better autoscaling on queue depth.

**Key points to mention:**

- Tail latency is usually a queueing or memory-pressure problem, not a model-speed problem.
- Look at preemption counters and waiting-request gauges.
- Segment by request length before blaming the engine.
- Reproduce with a load test before and after the fix.

**Deep dive:** [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md), [Debugging Case: Rising p95 Latency](../11-system-design/08-debugging-case-rising-p95-latency-and-memory.md)

</details>

<details>
<summary><b>Q37. How would you do capacity planning for an LLM service with a latency SLO?</b></summary>

**Answer:** Forecast peak requests per second and the request length mix. Run a rate sweep on one replica with that traffic to find the highest rate that still meets all SLOs (its goodput). Divide peak load by that capacity times a target utilization (for example 70%) for burst headroom, then add N+1 redundancy for failures and deploys. Check KV memory with Little's law (concurrent requests = rate x average time in system, times tokens per request) to make sure the cache fits. Autoscale on queue depth, KV utilization or p95 TTFT rather than CPU, and account for cold-start time of GPU replicas (minutes for image pull, weight loading and CUDA graph capture).

**Key points to mention:**

- Capacity per replica comes from measurement, not vendor numbers.
- Headroom for bursts and failures.
- LLM-specific autoscaling signals.
- Re-plan when the prompt or output length mix changes.

**Deep dive:** [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md), [Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md)

</details>

## 7. Serving frameworks and deployment choices

<details>
<summary><b>Q38. What does an LLM serving engine like vLLM do that `model.generate()` behind a web server does not?</b></summary>

**Answer:** It turns a model into a high-throughput, multi-user service. It schedules many concurrent requests and batches them continuously at every decode step, manages KV-cache memory in pages so far more sequences fit in a batch, reuses KV for shared prefixes, and runs optimized kernels (FlashAttention-style attention, quantized GEMMs, CUDA graphs). It also splits models across GPUs with tensor or pipeline parallelism and exposes a streaming OpenAI-compatible API with metrics. A naive `generate()` server processes one request (or one static batch) at a time, so the GPU spends most of its time reading weights for a single token. The core win is that decode is memory-bound, so a bigger batch amortizes each weight read over more tokens.

**Key points to mention:**

- Continuous batching + PagedAttention + optimized kernels are the big three.
- Observability: `/metrics` with KV usage, running and waiting requests, preemptions.
- Multi-LoRA serving and structured outputs are additional features.
- Often 10x or more throughput versus naive serving on the same GPU.

**Deep dive:** [Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md)

</details>

<details>
<summary><b>Q39. Compare vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp and Ollama. When would you pick each?</b></summary>

**Answer:** vLLM is the default for high-throughput GPU serving, with the broadest model and hardware support. SGLang shines on prefix-heavy and structured workloads such as agents, multi-turn chat and JSON output, thanks to RadixAttention and fast constrained decoding. TensorRT-LLM gives top performance on NVIDIA GPUs at the cost of build complexity and vendor lock-in. TGI is Hugging Face's server, common in existing deployments but now in maintenance mode. llama.cpp runs GGUF models on CPUs, Apple Silicon and edge devices, and Ollama wraps it for one-command local development. The right answer is to shortlist two engines and benchmark them on your own traffic and SLOs.

**Key points to mention:**

- Choose by workload shape, hardware, team skills and operational cost.
- Benchmark on your own traffic; published numbers rarely transfer.
- Local and edge (llama.cpp, Ollama) vs multi-user GPU serving (vLLM, SGLang, TensorRT-LLM).
- OpenAI-compatible APIs make switching engines cheaper, but long-tail features differ.

**Deep dive:** [Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md)

</details>

<details>
<summary><b>Q40. Should we use a managed LLM API or self-host an open model? Walk me through the decision.</b></summary>

**Answer:** Start with quality: if only a frontier closed model meets the bar on our evals, use the API. Otherwise compare cost at our real utilization, privacy and data-residency requirements, the need for control (pinned versions, fine-tunes, LoRA adapters, logprobs), latency predictability, and the operational burden we can absorb. To compare cost, benchmark tokens per second per GPU at our SLO and real input-to-output mix, convert to cost per million tokens at realistic utilization, and compare with the API's effective price, then add people cost and redundancy. Self-hosting tends to win at high, steady volume or under strict privacy rules; APIs win for low or spiky volume, speed to market and frontier quality. Often the answer is a hybrid behind one gateway.

**Key points to mention:**

- Utilization is the hidden variable: an idle GPU is expensive.
- Include engineering and on-call cost, not just GPU hours.
- Data governance can decide it regardless of cost.
- A gateway lets you switch or mix providers later.

**Deep dive:** [Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md), [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

</details>

## 8. Number formats and quantization fundamentals

<details>
<summary><b>Q41. Explain quantization and compare FP32, FP16, BF16, INT8 and INT4.</b></summary>

**Answer:** Quantization maps model numbers from a high-precision format to a lower-precision one, using scale factors so the small format covers the data's range, to save memory, bandwidth and sometimes compute at the cost of a small rounding error. FP32 is 4 bytes with huge range and about 7 decimal digits: the reference. FP16 and BF16 are 2 bytes; FP16 has 10 mantissa bits but a maximum of 65,504, while BF16 keeps FP32's 8-bit exponent (same range) with only 7 mantissa bits, so it is the default for LLM training. INT8 is 1 byte and INT4 half a byte; they are uniform integer grids that need a scale per tensor, channel or group, and are used mainly for inference, INT8 often for weights and activations and INT4 typically weight-only. A 70B model is 140 GB in BF16, about 70 GB in INT8 or FP8 and about 35-40 GB in INT4.

**Key points to mention:**

- Floats have exponent (range) and mantissa (precision); integers are a uniform grid plus a scale.
- FP8 (E4M3, E5M2) sits between BF16 and INT8 and is native on Hopper-class GPUs.
- 8-bit is usually near-lossless; 4-bit needs good methods (GPTQ, AWQ) and evaluation.
- Training uses BF16 plus FP32 master weights; inference goes lower.

**Deep dive:** [Number Formats](../07-model-compression/01-number-formats-and-model-memory.md)

</details>

<details>
<summary><b>Q42. How does quantization reduce memory consumption?</b></summary>

**Answer:** Weight memory is parameters times bytes per parameter, and quantization lowers the bytes per parameter: from 4 (FP32) or 2 (BF16) to 1 (INT8 or FP8) or about 0.5 (INT4). A 7B model goes from 14 GB in FP16 to about 3.6 GB at INT4 with group size 128. The small extra cost is the per-group scales and zero points, plus layers usually kept in 16-bit such as embeddings and the output head, which is why a "4-bit" 7B file is around 3.9-4.5 GB rather than 3.5 GB. Because decoding reads every weight for every token, fewer bytes also means less bandwidth per token, which speeds up bandwidth-bound decode. The freed memory is often more valuable as KV cache, allowing more concurrent requests.

**Key points to mention:**

- Separate weights from KV cache and activations: weight quantization does not shrink the KV cache.
- Count scale overhead: 16/128 = 0.125 extra bits per weight with FP16 scales at group size 128.
- Memory saving is guaranteed; speed saving is not.
- KV cache can be quantized separately (FP8 KV).

**Deep dive:** [Number Formats](../07-model-compression/01-number-formats-and-model-memory.md), [Quantization Fundamentals](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md)

</details>

<details>
<summary><b>Q43. Explain scale and zero point.</b></summary>

**Answer:** Uniform quantization maps a real value to an integer with q = clamp(round(x / s) + z) and back with x_hat = s x (q - z). The scale s is the step size, the real distance between two adjacent integer codes, computed as real range divided by the number of integer steps, for example (x_max - x_min) / 255 for 8 bits. The zero point z is the integer code that represents real 0 exactly, which shifts the grid so an asymmetric real range fits the integer range. Within the range the rounding error is at most s/2; values outside the range are clipped and can have much larger errors. Example: for a range of [-1, 3] in UINT8, s = 4/255, about 0.0157, and z = round(1/s) = 64.

**Key points to mention:**

- Real zero must be exactly representable (padding, ReLU outputs, masks, pruned weights).
- Clipping range trades rounding error against clipping error.
- Range choice: min-max, percentile, MSE search or KL (entropy) search.
- Integer matmuls accumulate in INT32 to avoid overflow.

**Deep dive:** [Quantization Fundamentals](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md)

</details>

<details>
<summary><b>Q44. Compare symmetric and asymmetric quantization.</b></summary>

**Answer:** Symmetric quantization forces z = 0 and uses a range centered on zero, [-a, a], with s = a/127 for INT8. Asymmetric (affine) quantization allows any range [b, a] by adding an integer zero point, so it can use the full grid for skewed data. Symmetric is simpler and makes integer matmuls cheaper, because there are no zero-point cross terms to correct, so it is the default for weights, which are roughly centered on zero. Asymmetric suits skewed data such as post-ReLU or post-GELU activations: with symmetric quantization, all-positive data would waste half of the integer levels. Many INT4 weight schemes still store zero points per group, trading a little memory for better use of only 16 levels.

**Key points to mention:**

- Symmetric for weights, asymmetric for skewed activations.
- Zero-point cross terms add compute in integer GEMMs.
- With 4 bits, every level matters, so asymmetric groups can help.
- Mention the effective bits cost of storing zero points.

**Deep dive:** [Quantization Fundamentals](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md)

</details>

<details>
<summary><b>Q45. What are per-channel and group-wise quantization?</b></summary>

**Answer:** They are granularities: how many values share one scale and zero point. Per-tensor uses one pair for the whole tensor; per-channel uses one pair per output channel, for example per row of a linear layer's weight matrix; group-wise splits each row into contiguous blocks of, say, 128 input weights and gives each block its own scale. Finer granularity lets each part of the tensor use a range that fits it, so one large row or one outlier weight no longer forces a coarse step on everything else. The cost is storage and kernel work: an FP16 scale per 128 weights adds 0.125 bits per weight, so INT4 with group size 128 is about 4.125 effective bits, and group size 32 is about 4.5 bits. Weights are usually per-channel or per-group, while activations are per-tensor or per-token, because a per-input-channel activation scale would not factor out of the integer dot product.

**Key points to mention:**

- Granularity is an accuracy versus overhead trade-off.
- Group size 128 is the common default for 4-bit LLM weights.
- Explain why activation scales are per-token, not per-input-channel.
- Weights are static, so fine-grained scales are computed once offline.

**Deep dive:** [Quantization Fundamentals](../07-model-compression/02-quantization-fundamentals-scale-and-zero-point.md), [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q46. Why does lower precision not always produce faster inference?</b></summary>

**Answer:** Lower precision reduces bytes, but time only drops if bytes were the bottleneck and an efficient kernel exploits them. If the workload is compute-bound (large-batch decode or long prefill), weight-only formats such as W4A16 still do 16-bit math, so there is little gain. Without a fused low-bit kernel, weights may be dequantized to FP16 first, and dequantization overhead can even make things slower. Other causes are KV cache reads or attention dominating at long context, CPU and framework overhead, time spent outside the model (queueing, tokenization, retrieval), and hardware without native support, for example FP8 on an A100. So the right claim is "quantization always saves memory; it saves time only when the workload is memory-bound or the hardware has faster low-precision math".

**Key points to mention:**

- Identify the bottleneck first (memory vs compute vs overhead).
- Kernel support is as important as the format.
- Weight-only helps decode at small batch; W8A8 helps compute-bound phases.
- Verify speedups end to end in the real engine.

**Deep dive:** [GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md), [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q47. Why is BF16 preferred over FP16 for training, and why might a model produce NaNs in FP16 but not in BF16?</b></summary>

**Answer:** BF16 has the same 8-bit exponent as FP32, so it has the same dynamic range, and activations and gradients almost never overflow or underflow. FP16's maximum is 65,504 and small values underflow, so FP16 training needs loss scaling and still risks NaN spikes. Networks tolerate BF16's lower precision well, especially since matmuls accumulate in FP32 and master weights are kept in FP32. A model that produces NaNs in FP16 but works in BF16 almost certainly overflows somewhere, often in attention logits, MLP outputs or the residual stream: a value exceeds 65,504, becomes infinity, and then NaN (for example infinity minus infinity in a softmax). The fix is to run in BF16 if the hardware supports it, or keep the overflowing operations in FP32.

**Key points to mention:**

- Range (exponent bits) vs precision (mantissa bits).
- Loss scaling exists because of FP16 underflow.
- BF16 needs Ampere or newer GPUs.
- Overflow is a common gotcha when porting BF16-trained models to older FP16-only hardware.

**Deep dive:** [Number Formats](../07-model-compression/01-number-formats-and-model-memory.md)

</details>

<details>
<summary><b>Q48. INT8 vs FP8: when would you choose each, and what are E4M3 and E5M2?</b></summary>

**Answer:** Both are 1 byte. INT8 is a uniform grid with broad hardware support (GPUs since Turing, CPUs, mobile NPUs) and fast integer math, but it needs careful scale choice and suffers when data has outliers. FP8 has a logarithmic grid that handles wide distributions more gracefully, so it usually needs less calibration and loses less accuracy for LLM activations, but it requires newer accelerators such as Hopper. E4M3 (4 exponent, 3 mantissa bits, max 448) has more precision and is used for weights and activations in inference; E5M2 (5 exponent, 2 mantissa bits, max 57,344) has more range and is classically used for gradients in FP8 training. On Hopper-class GPUs FP8 W8A8 is often the default first step for serving; on older GPUs, CPUs and edge devices INT8 is the practical choice.

**Key points to mention:**

- Both rely on per-tensor or per-block scaling factors.
- FP8 W8A8 speeds up compute-bound phases via FP8 tensor cores.
- Some recent recipes use E4M3 everywhere with fine-grained block scaling.
- Hardware support decides the format as much as accuracy does.

**Deep dive:** [Number Formats](../07-model-compression/01-number-formats-and-model-memory.md), [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

## 9. PTQ, QAT and calibration

<details>
<summary><b>Q49. Compare post-training quantization (PTQ) and quantization-aware training (QAT).</b></summary>

**Answer:** PTQ quantizes an already-trained model: it computes scales from the weights and from a small calibration set (often a few hundred samples), optionally optimizes rounding per layer (GPTQ, AWQ, AdaRound), and converts. It is cheap, fast, needs no labels and works on third-party models. QAT inserts fake quantization (quantize then dequantize) into the forward pass during fine-tuning, uses the straight-through estimator to backpropagate through rounding, and lets the weights adapt to quantization noise. It costs a training run and data, but usually recovers more accuracy, especially at very low bit widths (4-bit and below) or for small, sensitive models. For most LLM teams, PTQ is the default and QAT is the escalation path.

**Key points to mention:**

- STE: identity gradient through rounding, zero outside the clipping range.
- PTQ is near-lossless at 8-bit and usually good at 4-bit weight-only for large models.
- QAT is how vendors produce better low-bit checkpoints.
- QLoRA is not QAT: its 4-bit base is frozen.

**Deep dive:** [PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md)

</details>

<details>
<summary><b>Q50. When would you choose PTQ over QAT?</b></summary>

**Answer:** Almost always first, and permanently whenever PTQ meets the quality budget. Specifically: 8-bit targets (INT8 or FP8), 4-bit weight-only quantization of large LLMs, third-party models where I lack the training data or pipeline, when I need to iterate quickly across many schemes or models, and when there is no compute budget for training. QAT becomes worth it when PTQ misses the quality budget on important slices, at very low bit widths, for small models with little redundancy, or for edge deployments where INT8 activations are mandatory. Even then I would first try cheaper PTQ fixes: better calibration data, finer granularity, keeping sensitive layers in higher precision, or switching methods.

**Key points to mention:**

- Large models are more robust to PTQ than small ones.
- Exhaust PTQ knobs before paying for QAT.
- QAT needs data close to the training distribution.
- Decide against a degradation budget set in advance.

**Deep dive:** [PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md)

</details>

<details>
<summary><b>Q51. How does calibration data affect quantization quality?</b></summary>

**Answer:** Calibration data is what PTQ uses to observe the model's internal values: activation ranges for static scales, KV cache ranges, and, for GPTQ, AWQ and SmoothQuant, the activation statistics that drive weight adjustments. If it does not represent production traffic, static ranges are wrong (clipping or wasted resolution) and error-compensation methods optimize for the wrong distribution, so quality drops, often only on specific slices such as another language, code or long documents. A few hundred representative samples, with realistic lengths and formatted with the real chat template, usually beat thousands of generic web samples. Treat the calibration set as a versioned artifact, and when a customer reports a regression, check whether their traffic was underrepresented.

**Key points to mention:**

- Representativeness beats size.
- Use the production chat template and realistic sequence lengths.
- Mismatch shows up as slice regressions, not average drops.
- Dynamic quantization reduces calibration sensitivity for activations.

**Deep dive:** [PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md)

</details>

<details>
<summary><b>Q52. How do you evaluate the accuracy degradation from quantization?</b></summary>

**Answer:** Compare the quantized model against its own full-precision baseline under identical conditions (same prompts, template, decoding settings and engine), in layers. Start with cheap smoke tests: perplexity, KL divergence from the baseline's token distributions, and top-1 agreement. Then run standard task benchmarks, and as the real deployment gate, my own production-derived eval set, with per-slice and worst-case analysis including "flips" (examples the baseline got right and the quantized model gets wrong). Add targeted tests for fragile capabilities: long context, code, math, structured output and tool calling. Finally, measure latency, throughput and memory to confirm the benefit, all against a degradation budget defined in advance, for example "no more than 1 point drop overall and no slice down more than 3".

**Key points to mention:**

- Perplexity alone can hide task-specific damage (code, reasoning).
- Report flips, not just net accuracy.
- Use paired comparisons and confidence intervals.
- Quality and speed are reported together.

**Deep dive:** [PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md), [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q53. What are the trade-offs of weight-only versus weight-and-activation quantization?</b></summary>

**Answer:** Weight-only quantization (W8A16, W4A16) stores weights in low precision and dequantizes them to 16-bit inside the kernel before multiplying. It shrinks the model and the bytes read per decode step, so it speeds up memory-bound, small-batch decoding and lets big models fit on smaller GPUs, with relatively low accuracy risk since weights are static and well-behaved. But the math still runs in 16-bit, so it does not speed up compute-bound prefill or large-batch serving and can even be slower there. Weight-and-activation quantization (W8A8 in INT8 or FP8) runs the GEMM on low-precision tensor cores, giving real compute speedups, but activations are input-dependent and have outliers, so accuracy is harder to keep and it needs calibration or dynamic scaling and hardware support. Rule of thumb: W4A16 for memory-limited, low-batch serving; FP8 W8A8 for high-throughput serving on Hopper-class GPUs.

**Key points to mention:**

- Weight-only saves bytes; W8A8 saves FLOPs.
- Activation outliers are the main obstacle for W8A8 (SmoothQuant addresses this).
- Static vs dynamic activation scales.
- Choose by workload regime and hardware.

**Deep dive:** [PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md), [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q54. Why and how would you quantize the KV cache, and what are the risks?</b></summary>

**Answer:** At long context and high concurrency the KV cache can exceed the weights in memory, so quantizing it (commonly to FP8 or INT8) roughly halves its size, doubling the tokens that fit and reducing the bytes attention reads per decode step. In the 8B GQA example, that moves concurrency from about 24 to about 48 requests at 16K tokens on one 80 GB GPU. Unlike weights, the KV cache is produced at runtime, so its scales must be calibrated (static) or computed dynamically, and any error affects attention over every future token. The risks are clipping of outlier channels, especially in keys, and error that accumulates over very long contexts. Validate with task evals plus long-context retrieval tests (needle-in-a-haystack style) at production context lengths, against an FP16 cache.

**Key points to mention:**

- It is independent of weight quantization; you can combine them.
- Keys tend to have more outlier channels than values.
- Evaluate specifically at long context.
- Big capacity win for little effort in engines like vLLM.

**Deep dive:** [PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md), [The KV Cache](../06-llm-inference/02-kv-cache-explained.md)

</details>

## 10. Practical LLM quantization: GPTQ, AWQ and outliers

<details>
<summary><b>Q55. Explain GPTQ, AWQ and practical LLM quantization.</b></summary>

**Answer:** Practical LLM quantization usually means post-training, either weight-only 4-bit (W4A16) or 8-bit weight-and-activation (W8A8, often FP8). GPTQ and AWQ are the two main calibration-based 4-bit weight methods. GPTQ quantizes each layer column by column and uses inverse-Hessian information from calibration activations to adjust the not-yet-quantized weights so the layer's output error stays small. AWQ observes that about 1% of input channels, those with large activations, matter most, and scales those channels' weights up before quantization (folding the inverse scale into the activations), choosing the scale by a small grid search. Both produce group-wise (typically group size 128) INT4 weights that run with fused dequantize-and-multiply kernels in vLLM, SGLang or TensorRT-LLM, cutting a 70B model from 140 GB to about 35-40 GB.

**Key points to mention:**

- Both minimize layer output error, not weight error.
- Both need a small calibration set and no gradient training.
- Kernel support matters as much as the algorithm.
- Alternatives by context: FP8 on Hopper, GGUF for CPU and edge.

**Deep dive:** [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q56. What is the key idea behind GPTQ?</b></summary>

**Answer:** GPTQ quantizes a layer's weights one column at a time, and after each column it updates the remaining, not-yet-quantized weights to compensate for the rounding error just introduced. The compensation uses the inverse Hessian of the layer-wise reconstruction loss ||WX - W_hat X||^2, where the Hessian H = 2 X X^T is computed from calibration inputs. Intuitively, the Hessian tells GPTQ which other input dimensions are correlated with the one just rounded, so their weights can be nudged to cancel its error on real data. It builds on Optimal Brain Surgeon ideas but makes them efficient (lazy batched updates, Cholesky decomposition), so a 175B-class model can be quantized in a few GPU hours. Options such as act-order (quantize the most important columns first) and group size trade accuracy against speed and storage.

**Key points to mention:**

- Error compensation is the key idea, not just rounding.
- Hessian from calibration activations, hence sensitivity to calibration data.
- One-shot, layer by layer, no backpropagation through the full model.
- SparseGPT uses the same machinery for pruning.

**Deep dive:** [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q57. How does AWQ use activation information?</b></summary>

**Answer:** AWQ (activation-aware weight quantization) runs calibration data through the model and measures the average activation magnitude of each input channel. Channels with large activations are "salient": their weights get multiplied by large numbers, so rounding errors there are amplified in the output. AWQ protects them by multiplying each input channel's weights by a per-channel scale s that grows with its activation magnitude before quantization, and dividing the activations by s (folded into the previous operation), which reduces the relative rounding error on salient weights without mixed precision. The scale is s = s_X^alpha, with alpha chosen by a grid search that minimizes the layer's output error. Because it does not do GPTQ-style weight updates, it is fast, less prone to overfitting the calibration set, and generalizes well across domains.

**Key points to mention:**

- Saliency is determined by activations, not weight magnitude.
- The scaling is mathematically equivalent before quantization, so it is free at inference.
- About 1% of channels matter most.
- Compare: GPTQ compensates errors, AWQ prevents them on important channels.

**Deep dive:** [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q58. How do outlier activations affect quantization?</b></summary>

**Answer:** Quantization uses a shared scale set by the largest absolute value in the group. LLMs above a few billion parameters develop systematic outlier features: a few hidden dimensions whose values are often an order of magnitude or more larger than the rest, across most tokens and layers. One outlier forces a huge scale, so all normal values collapse onto a handful of integer levels and lose precision, which is why naive per-tensor INT8 activation quantization breaks large models. Clipping the outliers is not an option because they carry important information. The fixes are to isolate them (LLM.int8() keeps outlier dimensions in FP16), migrate them into the weights (SmoothQuant), protect their weights (AWQ), use finer granularity (per-token, per-group), or use FP8, whose logarithmic grid tolerates wide ranges better.

**Key points to mention:**

- Outliers are concentrated in specific channels, which is what makes them fixable.
- Weights are well-behaved; activations are the hard part.
- This is why W4A16 is easy and W4A4 is hard.
- Connect to AWQ and Wanda, which both use activation magnitude.

**Deep dive:** [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q59. How would you choose between a 4-bit and an 8-bit model?</b></summary>

**Answer:** I decide from five factors: memory budget (including KV cache at target concurrency), quality budget on my actual tasks, hardware and kernel support, batch size and workload shape, and task sensitivity. 8-bit, especially FP8 W8A8 on Hopper-class GPUs, is usually near-lossless and speeds up compute-bound, high-throughput serving. 4-bit weight-only (AWQ or GPTQ) is the choice when memory is the binding constraint, for example fitting a 70B model on one 80 GB GPU, or for low-batch, latency-sensitive decoding, as long as evals on sensitive tasks such as code, math and long context hold up. A useful comparison: a 4-bit larger model often beats an 8-bit smaller model at the same memory, so compare at equal memory, not equal parameter count.

**Key points to mention:**

- Run both through the same eval and benchmark before deciding.
- Sensitive tasks (code, math, reasoning) degrade first at 4-bit.
- Memory freed by 4-bit can buy more KV cache and concurrency.
- Hardware without fast 4-bit kernels erases the speed benefit.

**Deep dive:** [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q60. How would you verify real inference speedups from quantization?</b></summary>

**Answer:** Benchmark end to end in the real serving engine against an unquantized baseline with identical settings, using production-like prompt and output lengths and concurrency. Measure TTFT, time per output token, throughput and p95/p99 latency across a sweep of batch sizes or arrival rates and sequence lengths, plus peak memory. Then confirm with engine logs and a profiler (for example Nsight Systems) that the optimized quantized kernels are actually running, and quantify dequantization overhead. Report quality alongside speed and translate both into cost per token or requests per second per GPU at the SLO. Expect the speedup to vary by regime: large at batch 1 decode, small or negative for long prefill or large batches with weight-only formats.

**Key points to mention:**

- Microbenchmarks of one layer are not enough.
- Sweep regimes, since speedups depend on batch and length.
- Check the kernel path in logs, not just the config.
- Compare at equal quality and at the latency SLO.

**Deep dive:** [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md), [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

</details>

<details>
<summary><b>Q61. A quantized LLM uses much less memory but latency is almost unchanged. Diagnose it.</b></summary>

**Answer:** Work through the likely causes in order. First, is the workload compute-bound or memory-bandwidth-bound? Weight-only INT4 only speeds up bandwidth-bound work, so if the benchmark is dominated by long prefills or large-batch decode, the math still runs in FP16 and little changes. Second, are optimized quantized kernels used? If the engine dequantizes weights to FP16 before a standard GEMM, or the GPU lacks fast kernels for that format, savings are eaten by overhead. Third, is the KV cache the dominant memory and bandwidth consumer at this context length, so weight bytes were never the bottleneck? Fourth, profile from request metrics (queue, TTFT, TPOT, batch size) down to a kernel timeline, and check for time outside the model such as queueing, tokenization or retrieval. Depending on the finding, batching (if underutilized), speculative decoding (if low-batch decode dominates), FP8 W8A8 (if compute-bound) or a smaller model are the next levers.

**Key points to mention:**

- Memory saving does not imply latency saving.
- Classic causes: compute-bound regime, missing fused kernel, KV-dominated reads, non-model overhead.
- Use the roofline argument with numbers.
- Turn the freed memory into value: larger batches, more KV cache, fewer GPUs.

**Deep dive:** [Debugging Case: Quantized LLM Not Faster](../11-system-design/06-debugging-case-quantized-llm-not-faster.md), [GPU Fundamentals](../06-llm-inference/03-gpu-fundamentals-compute-bound-vs-memory-bound.md)

</details>

<details>
<summary><b>Q62. Explain SmoothQuant. Why is the transformation free at inference time?</b></summary>

**Answer:** SmoothQuant targets W8A8, where activation outliers make INT8 activations inaccurate. It rescales each input channel j: activations are divided by s_j and the corresponding weight rows are multiplied by s_j, so the product XW is mathematically unchanged, but activation outliers shrink and the weights absorb some of the range. With s_j = max|X_j|^alpha / max|W_j|^(1 - alpha) and alpha around 0.5, both activations and weights become easy to quantize to INT8, enabling true INT8 GEMMs. It is free at inference because the division of activations is folded offline into the preceding LayerNorm or linear layer, and the multiplication is baked into the stored weights. Alpha controls how much difficulty migrates from activations to weights.

**Key points to mention:**

- "Migrate quantization difficulty from activations to weights."
- Enables real compute speedups, unlike weight-only methods.
- Same channel-scaling idea as AWQ, used for a different goal.
- Calibration data sets the per-channel maxima.

**Deep dive:** [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q63. Which quantized format would you choose for H100 serving, a small GPU where the model barely fits, and a laptop?</b></summary>

**Answer:** On an H100, FP8 W8A8: it halves memory, uses Hopper's FP8 tensor cores to speed up both prefill and decode, and usually loses very little quality. On an A10G or L4 where the model barely fits, a 4-bit weight-only format such as AWQ or GPTQ with fast kernels (or FP8 on the L4, which supports it) frees memory for the KV cache and speeds up low-batch decode. On a laptop, CPU or Apple Silicon, GGUF (typically Q4_K_M or Q5_K_M) with llama.cpp or Ollama. The general rule: GGUF with llama.cpp for local, single-user and edge; AWQ, GPTQ or FP8 on vLLM, SGLang or TensorRT-LLM for multi-user GPU serving where continuous batching and paged attention matter.

**Key points to mention:**

- Hardware support decides the format.
- Fit is only step one; leave room for KV cache.
- Validate the chosen format on your evals.
- Mention that GGUF k-quants mix bit widths across tensors.

**Deep dive:** [Serving Frameworks](../06-llm-inference/06-serving-frameworks-vllm-sglang-tgi-tensorrt-llm-and-ollama.md), [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q64. Your 4-bit model matches the original on perplexity, but users report worse code generation. What do you do?</b></summary>

**Answer:** Treat it as a real regression: perplexity is an average over generic text and can hide task-specific damage, and code is precision-sensitive because one wrong token breaks a program. Build or run a code-specific eval (pass rate on unit-tested problems) comparing FP16 and 4-bit with identical prompts and decoding, and inspect failures. Then try mitigations in order of cost: recalibrate on code-heavy data, change method or settings (AWQ vs GPTQ, smaller group size, act-order), keep the most sensitive layers in higher precision, or move to 8-bit or FP8 for that workload. Add the code eval to the release gate so the next quantized checkpoint is checked before launch.

**Key points to mention:**

- Perplexity is a smoke test, not a release gate.
- Calibration data mismatch is a likely cause.
- Per-task evaluation with execution-based metrics.
- Consider routing code traffic to a higher-precision deployment.

**Deep dive:** [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md), [PTQ vs QAT](../07-model-compression/03-ptq-vs-qat-and-calibration.md)

</details>

## 11. Knowledge distillation fundamentals

<details>
<summary><b>Q65. Explain knowledge distillation.</b></summary>

**Answer:** Knowledge distillation is a training technique where a small student model learns to imitate a large, accurate teacher model, so we can deploy the small model with most of the big model's quality. Instead of training the student only on hard labels, we also train it to match the teacher's output distribution, softened with a temperature, using a KL divergence loss; the classic formulation is Hinton, Vinyals and Dean (2015). The usual loss is alpha x T^2 x KL(teacher_T || student_T) + (1 - alpha) x CE(labels, student). The student is cheaper and faster at inference because it has fewer or narrower layers, and the gains are real on any hardware. DistilBERT is the textbook example: about 40% smaller and 60% faster than BERT-base while keeping about 97% of its language-understanding performance.

**Key points to mention:**

- Soft targets carry "dark knowledge" about class similarity.
- Costs a real training run, unlike PTQ.
- Stacks with quantization: distill, then quantize the student.
- For LLMs, distillation often means training on teacher-generated text.

**Deep dive:** [Distillation Fundamentals](../07-model-compression/05-knowledge-distillation-fundamentals.md)

</details>

<details>
<summary><b>Q66. What are teacher and student models?</b></summary>

**Answer:** The teacher is a large, already-trained, accurate model (or an ensemble) that provides training targets; it is usually frozen and only run forward, in eval mode. The student is the smaller model we actually want to deploy, trained to reproduce the teacher's behavior as well as the true labels. Typically the student has fewer or narrower layers, and it is often initialized from a pretrained small model or from a subset of the teacher's layers, which speeds up convergence. The student can even have a different architecture entirely, for example a transformer teacher and a small CNN or BiLSTM student, as long as the outputs are comparable.

**Key points to mention:**

- Teacher must actually be better than the student on the target data.
- A huge teacher can be a worse teacher (capacity gap); teacher assistants help.
- Teacher logits can be precomputed and cached to save compute.
- For LLMs, sharing a tokenizer enables logit-level distillation.

**Deep dive:** [Distillation Fundamentals](../07-model-compression/05-knowledge-distillation-fundamentals.md)

</details>

<details>
<summary><b>Q67. How do soft targets differ from hard labels?</b></summary>

**Answer:** A hard label is a one-hot vector: it only says which class is correct. A soft target is the teacher's full probability distribution, for example [0.879, 0.119, 0.002] for cat, dog and car. The small probabilities on wrong classes, called dark knowledge, tell the student how similar the classes are (this cat looks a bit like a dog and nothing like a car), so each example carries much more information and gradients have lower variance. That lets students learn from fewer examples, generalize better and even learn from unlabeled data labeled by the teacher. Soft targets also act as a form of label smoothing tailored to each example.

**Key points to mention:**

- More bits of information per example.
- Enables distillation on unlabeled transfer sets.
- Black-box APIs that do not expose probabilities limit you to hard (sequence-level) distillation.
- Temperature makes the dark knowledge visible to the loss.

**Deep dive:** [Distillation Fundamentals](../07-model-compression/05-knowledge-distillation-fundamentals.md)

</details>

<details>
<summary><b>Q68. What is the role of temperature in distillation, and why multiply the loss by T squared?</b></summary>

**Answer:** Temperature divides the logits before the softmax, softmax(z / T). With T > 1 the distribution gets softer, so tiny probabilities on wrong classes become large enough to influence the loss and the student can learn the teacher's similarity structure. Both teacher and student use the same T in the distillation term, the argmax never changes, and the student is deployed at T = 1. Typical values are 2 to 5 for classification. The gradient of the soft term with respect to a student logit is (q_i - p_i) / T, and at high temperature q_i - p_i itself shrinks roughly like 1/T, so the total gradient scales as 1/T^2; multiplying the soft loss by T^2 keeps soft and hard terms balanced, so changing T does not silently change how much the student listens to the teacher.

**Key points to mention:**

- Same T on both sides of the KL term.
- Too high a T makes everything uniform and uninformative.
- The T^2 factor is a common implementation bug when omitted.
- Tune T and alpha together.

**Deep dive:** [Distillation Fundamentals](../07-model-compression/05-knowledge-distillation-fundamentals.md)

</details>

<details>
<summary><b>Q69. Why is KL divergence commonly used as the distillation loss?</b></summary>

**Answer:** KL divergence measures how one probability distribution differs from another across all classes, weighted by the teacher's probabilities, so it captures the whole distribution, including the dark knowledge. Since KL(p || q) = CE(p, q) - H(p), and the teacher's entropy H(p) is a constant with respect to the student, minimizing KL is the same as minimizing cross-entropy against the teacher's soft targets; KL just has a natural zero when the distributions match. Its gradient with respect to the student's logits is simply (q - p) / T, which is clean and stable. In PyTorch the common pitfall is that `kl_div` expects student log-probabilities and teacher probabilities, with `reduction="batchmean"`.

**Key points to mention:**

- KL vs cross-entropy equivalence up to a constant.
- Forward KL is mode-covering; reverse KL is mode-seeking.
- Implementation details are a frequent source of silent bugs.
- MSE on logits is an alternative, closely related at high temperature.

**Deep dive:** [Distillation Fundamentals](../07-model-compression/05-knowledge-distillation-fundamentals.md)

</details>

<details>
<summary><b>Q70. How do response-based and feature-based distillation differ?</b></summary>

**Answer:** Response-based distillation matches only the teacher's final outputs, typically temperature-softened logits with a KL loss. Feature-based distillation also matches intermediate representations such as hidden states, feature maps or attention matrices, usually with an MSE loss after a learnable projection when dimensions differ (as in FitNets or TinyBERT). Response-based is simpler and works with a black-box teacher, as long as the API exposes probabilities (otherwise you are limited to hard, sequence-level distillation). Feature-based gives richer layer-by-layer guidance and can speed up convergence, but it needs access to the teacher's internals and a sensible layer mapping between teacher and student. A third family, relation-based distillation, matches relationships between examples (distances or angles in embedding space), which suits embedding and retrieval models.

**Key points to mention:**

- Response-based: outputs only; feature-based: internals too.
- Layer mapping and projection heads are design choices.
- White-box access is required for feature-based.
- Relation-based for embedding geometry.

**Deep dive:** [Distillation Fundamentals](../07-model-compression/05-knowledge-distillation-fundamentals.md)

</details>

## 12. Distilling large language models

<details>
<summary><b>Q71. How would you distill a large LLM into a smaller model?</b></summary>

**Answer:** Start from the goal: define the tasks, a latency and memory budget and a fixed eval suite. Pick a strong small student, ideally from the teacher's family so tokenizers and chat templates match. Build a diverse prompt set from scrubbed real traffic plus synthetic gap-filling, have the teacher generate several responses per prompt with a rich prompt (policies, examples, retrieval), and filter hard with verifiers, schema checks, LLM judges, deduplication and decontamination. Then fine-tune the student on the survivors (sequence-level KD); if I own the teacher and share a tokenizer, I add a logit-level KL term and optionally finish with on-policy distillation on the student's own samples. Finally, measure capability loss per slice and benchmark cost and latency fairly before a shadow and canary rollout.

**Key points to mention:**

- Three approaches: sequence-level, logit-level, on-policy.
- Data quality and filtering matter more than data volume.
- Decontaminate against eval sets.
- Check licensing of teacher outputs and the student base model.

**Deep dive:** [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md)

</details>

<details>
<summary><b>Q72. How do you select the student architecture?</b></summary>

**Answer:** Work backwards from deployment constraints. Compute the memory budget (weights plus KV cache at the target context length and concurrency) and the latency budget; since decode is memory-bandwidth-bound, per-token time scales roughly with weight bytes, so model size maps directly to ITL. Then choose the largest strong pretrained small model that fits, preferably from the teacher's family for a shared tokenizer and chat template, with enough native context length and a GQA design for a smaller KV cache. Pilot two or three sizes (for example 1B, 3B and 8B) on a subset of the data and pick the smallest that passes the eval bar at the SLO. Starting from a good pretrained base is almost always better than training a student from scratch.

**Key points to mention:**

- Constraints first, then candidates.
- Shared tokenizer enables logit KD.
- Check license, context length and quantization friendliness.
- Plan to quantize the student too.

**Deep dive:** [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md)

</details>

<details>
<summary><b>Q73. How would you create and filter teacher-generated data?</b></summary>

**Answer:** To create: gather prompts from scrubbed production logs, a task taxonomy covering every intent and edge case, and synthetic expansion to fill gaps; keep a deliberate difficulty mix; and sample the teacher several times per prompt with a rich prompt (system policies, examples, retrieved context) that the student will not get at inference. To filter: validate schema and format, verify by execution, tests or reference answers where possible, score with an LLM judge using a rubric, use rejection sampling to keep the best passing sample, remove exact and near duplicates, and decontaminate against every eval set. Track counts at each stage (for example 200K generated to 80K kept) and spot-check by hand, because filters can silently remove whole intents.

**Key points to mention:**

- Teacher outputs are not ground truth; verify them.
- Include refusals, "I don't know" and error-recovery cases.
- Decontamination protects the integrity of your evals.
- Balance the final dataset across intents and difficulty.

**Deep dive:** [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md)

</details>

<details>
<summary><b>Q74. How do you preserve reasoning, structured output and tool calling when distilling?</b></summary>

**Answer:** Put each capability in the training data explicitly and verify it, because the student only learns what it sees. For reasoning, distill rationales (step-by-step traces) and keep only traces whose final answers are verified correct, preferring shorter traces for latency. For structured output, train only on schema-validated outputs and also use constrained decoding at inference as a safety net. For tool calling, train on full validated trajectories (call, result, next call, answer) rendered in the student's own chat template, mask the loss on tool results, and include negative cases (when not to call a tool) and error-recovery cases. Then measure each capability separately: reasoning accuracy, schema validity rate and tool-call accuracy.

**Key points to mention:**

- Verified rationales, not just any rationales.
- Chat template mismatches silently break tool calling.
- Loss masking on tool outputs.
- Per-capability metrics in the eval suite.

**Deep dive:** [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md)

</details>

<details>
<summary><b>Q75. How do you measure capability loss after distillation?</b></summary>

**Answer:** Run teacher and student on the same fixed eval suite with identical prompts, templates and decoding settings: task evals, a few general benchmarks to detect forgetting, tool-call accuracy, schema validity rate, long-tail slices and safety tests. Report a retention ratio (student score over teacher score) overall and per slice, for example "96% overall but 81% on multi-step refunds". Use confidence intervals and paired significance tests, because many eval sets are small and averages hide tail failures. Also compare against simple baselines, such as the student base model with good prompting, to show that distillation actually added value.

**Key points to mention:**

- Per-slice retention, not just the average.
- Paired comparisons on the same examples.
- Check forgetting of general skills.
- Include safety and refusal behavior.

**Deep dive:** [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md)

</details>

<details>
<summary><b>Q76. How would you compare latency, memory, throughput and answer quality between teacher and student?</b></summary>

**Answer:** Hold everything constant except the model: same hardware, serving framework and version, same precision (or report each combination), same prompts and output limits, and the same decoding settings. Sweep concurrency or arrival rate and measure TTFT, inter-token latency and end-to-end latency at p50, p95 and p99, throughput, and peak memory, and run the quality eval on the same serving stack, not in a notebook. Then compare at the latency SLO: requests per second per GPU and quality. Summarize as cost per successful task and a Pareto plot of quality against cost, so stakeholders can pick the operating point.

**Key points to mention:**

- Same stack for quality and speed measurements.
- Compare at the SLO, not at arbitrary batch sizes.
- Cost per successful task is the business metric.
- Include variance across repeated runs.

**Deep dive:** [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md), [Pruning and Choosing a Method](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md)

</details>

<details>
<summary><b>Q77. What is the difference between forward and reverse KL, and why does on-policy distillation often use reverse KL?</b></summary>

**Answer:** Forward KL, KL(p_teacher || q_student), is averaged under the teacher, so it heavily punishes the student for missing anything the teacher considers plausible: it is mode-covering, and a small student spreads its probability mass thinly. Reverse KL, KL(q_student || p_teacher), is averaged under the student, so it punishes the student for putting mass where the teacher puts little: it is mode-seeking and concentrates on a few high-quality modes. A small student cannot cover all of a big teacher's modes, so mode seeking avoids generating low-quality text "in between" them. On-policy methods (such as GKD or MiniLLM) also train on the student's own samples, scored by the teacher, which reduces exposure bias: the student learns to recover from its own mistakes rather than only seeing perfect teacher prefixes.

**Key points to mention:**

- Forward KL for classification and as a default.
- Reverse KL or mixed divergences for generative students.
- Exposure bias is the main motivation for on-policy training.
- On-policy needs teacher logits on student samples, so it needs white-box access.

**Deep dive:** [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md), [Distillation Fundamentals](../07-model-compression/05-knowledge-distillation-fundamentals.md)

</details>

<details>
<summary><b>Q78. When would you choose logit-level distillation over sequence-level distillation?</b></summary>

**Answer:** Sequence-level distillation simply fine-tunes the student on text the teacher generated; logit-level distillation trains the student to match the teacher's full next-token distribution at every position. Logit KD gives far more signal per training token (a whole distribution instead of one sampled token), so it reaches better quality for the same data. I would use it when I own the teacher weights, the student shares the teacher's tokenizer, and the compute for running the teacher on the training data is available. Otherwise, especially with an API teacher or a cross-family student with a different tokenizer, sequence-level distillation is the practical choice, and it works well when combined with strong filtering.

**Key points to mention:**

- Tokenizer match is the hard requirement for logit KD.
- Store top-k teacher logits to save storage.
- Sequence-level works with black-box API teachers.
- Both can be combined (SFT term plus KL term).

**Deep dive:** [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md)

</details>

<details>
<summary><b>Q79. Your distilled student matches the teacher's tone perfectly but hallucinates more. Why, and what do you do?</b></summary>

**Answer:** Imitation transfers style much more easily than knowledge. The teacher answered from knowledge the much smaller student does not have, and SFT on those confident answers taught the student to always answer confidently, even when it does not know. The fixes are to narrow the task to what the student can learn, ground the student with retrieval so knowledge comes from context instead of parameters, add training examples where the right behavior is to look something up or say it does not know, and filter training data for faithfulness. If that is not enough, consider a bigger student or route uncertain cases back to the teacher. Measure faithfulness and abstention explicitly, since style metrics and LLM judges can be fooled by fluent, confident text.

**Key points to mention:**

- "Style is cheap, knowledge is expensive."
- Grounding plus abstention training.
- Judges that reward fluency can hide the problem.
- Routing as a safety valve.

**Deep dive:** [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md)

</details>

## 13. Pruning, low-rank factorization and choosing a method

<details>
<summary><b>Q80. Compare quantization, pruning, distillation and low-rank factorization.</b></summary>

**Answer:** Quantization stores the same weights in fewer bits: 2-4x smaller, often faster decode, low effort, low risk at 8-bit and moderate at 4-bit, and usually no retraining. Pruning removes weights or structures: unstructured pruning keeps quality well but rarely gives speed on GPUs, while structured pruning (removing heads, neurons or layers) gives real speed on any hardware but needs recovery training. Distillation trains a smaller student to mimic a teacher: you choose the size, gains are real everywhere, but it costs the most data and compute. Low-rank factorization replaces a matrix with two thin ones: dense and hardware-friendly, but it only helps when the weights are actually low rank and the rank is well below break-even. In practice: quantize first, distill or structurally prune when you need a fundamentally smaller model, and combine them.

**Key points to mention:**

- Compare on effort, risk, hardware friendliness and size reduction.
- Unstructured sparsity rarely speeds up GPUs.
- Methods stack: prune, distill to heal, then quantize.
- The fair benchmark decides, not the method's reputation.

**Deep dive:** [Pruning and Choosing a Method](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md)

</details>

<details>
<summary><b>Q81. Which would you try first for a memory-constrained deployment, and why?</b></summary>

**Answer:** Weight-only post-training quantization: INT8, or 4-bit with AWQ or GPTQ. It gives the largest memory reduction for the least effort and risk (about 4x at 4-bit versus FP16), needs no training, is supported by mainstream serving stacks, and often speeds up decode because decode is memory-bound. For example, a 70B model drops from 140 GB to about 35-40 GB and fits on one 80 GB GPU with room for KV cache. If that is not enough or quality drops too much, add FP8 KV cache quantization, then move to a smaller or distilled model (also quantized).

**Key points to mention:**

- Cheapest, fastest, least risky lever first.
- Remember the KV cache in the memory budget.
- Evaluate quality on your own tasks before shipping.
- Escalate to distillation only if needed.

**Deep dive:** [Pruning and Choosing a Method](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md), [Practical LLM Quantization](../07-model-compression/04-llm-quantization-gptq-awq-and-outliers.md)

</details>

<details>
<summary><b>Q82. Why does unstructured sparsity not necessarily improve latency?</b></summary>

**Answer:** GPUs and their tensor cores multiply dense tiles and do not skip zeros, so a dense tensor with 50% zeros runs at the same speed and uses the same memory as before. To skip zeros you need a sparse format, which adds index overhead, and sparse kernels, which suffer from irregular memory access and load imbalance and cannot use tensor cores well. These only beat dense kernels at very high sparsity (often 90% or more), where quality usually suffers, and your serving stack must actually use them. The exceptions are structured patterns: 2:4 semi-structured sparsity on Ampere and newer GPUs (up to 2x matmul throughput in theory), or structured pruning that builds genuinely smaller dense layers.

**Key points to mention:**

- Zeros in a dense tensor are still stored and multiplied.
- PyTorch pruning masks even make checkpoints bigger.
- 2:4 sparsity is the hardware-friendly middle ground.
- Structured pruning gives real speed but needs recovery training.

**Deep dive:** [Pruning and Choosing a Method](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md)

</details>

<details>
<summary><b>Q83. Can these compression methods be combined? How, and in what order?</b></summary>

**Answer:** Yes, and the best results usually come from combinations. The typical order is structural changes first (prune or choose a smaller architecture), then recovery training (fine-tuning or distillation from the original model), and quantization last, because quantization error is easiest to control on the final weights. Examples: prune then heal (NVIDIA's Minitron heals with distillation, Sheared LLaMA with continued pretraining), distill then quantize, prune then quantize (Deep Compression, SparseGPT with quantization), 2:4 sparsity plus INT8, and QLoRA, which combines a 4-bit quantized frozen base with low-rank adapters for fine-tuning. Evaluate after each stage so you know which step caused any regression.

**Key points to mention:**

- Structure, then heal, then quantize.
- Distillation is the healing tool after pruning.
- Errors compound, so evaluate at each step.
- QLoRA as a familiar example of quantization plus low rank.

**Deep dive:** [Pruning and Choosing a Method](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md), [Fine-Tuning LLMs](../05-llms/07-fine-tuning-full-lora-and-qlora.md)

</details>

<details>
<summary><b>Q84. How would you design a fair compression benchmark?</b></summary>

**Answer:** Change only the model variant: same hardware, drivers, serving stack and version, kernels and decoding settings. Drive it with production-shaped load (realistic concurrency and prompt and output length distributions), warm up, and repeat runs. Measure TTFT and per-token latency at p50, p95 and p99, throughput at a latency SLO, and peak memory. Measure quality on task evals plus general benchmarks, perplexity and behavior checks such as format compliance and safety, on the same stack. Report variance and confidence intervals and plot quality against cost, so the comparison shows the trade-off rather than a single cherry-picked number.

**Key points to mention:**

- One variable at a time.
- Both quality and efficiency, measured on the same stack.
- Compare at the SLO and at equal memory where relevant.
- Report configuration in full for reproducibility.

**Deep dive:** [Pruning and Choosing a Method](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md), [Optimizing and Benchmarking](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md)

</details>

<details>
<summary><b>Q85. You must shrink a 7B model to about 4B parameters for a device. Walk me through your plan.</b></summary>

**Answer:** First check the cheap options: quantizing the 7B to 4-bit (about 3.6-3.7 GB of quantized weights, around 3.9 GB or more as a real file) may already meet the memory constraint, and an existing 3-4B open model fine-tuned on the task may meet the quality bar. If we really need a custom 4B model, I would use structured pruning plus distillation, the Minitron-style recipe. Estimate the importance of layers, attention heads and MLP neurons on calibration data, prune depth and/or width to the target shape, then heal by distilling from the original 7B on a curated dataset, which needs far fewer tokens than pretraining from scratch. Finally quantize the result for the device, evaluate quality per slice against the 7B, and measure latency and memory on the actual target hardware.

**Key points to mention:**

- Cheapest options first: quantize, or pick an existing small model.
- Structured pruning produces real dense speedups.
- Distillation heals the pruned model.
- Benchmark on the real device, not a server GPU.

**Deep dive:** [Pruning and Choosing a Method](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md), [Distilling LLMs](../07-model-compression/06-distilling-large-language-models.md)

</details>

## Rapid-fire round

| Question | Answer |
| --- | --- |
| Which inference phase is compute-bound? | Prefill (long prompts); small-batch decode is memory-bandwidth-bound. |
| Lower bound on batch-1 decode time per token? | Bytes read per step (weights plus KV) divided by memory bandwidth. |
| KV cache per token for a 7B MHA model in FP16? | 512 KiB (2 x 32 layers x 32 heads x 128 dim x 2 bytes). |
| Does GQA shrink the KV cache by query heads or KV heads? | KV heads: 32 query heads with 8 KV heads is 4x smaller. |
| Are queries stored in the KV cache? | No, only keys and values. |
| What does PagedAttention borrow from operating systems? | Virtual memory paging: fixed-size blocks plus a per-sequence block table. |
| Typical PagedAttention waste per sequence? | Less than one partially filled block. |
| Continuous batching schedules at which level? | Iteration (decode step) level, not request level. |
| What does chunked prefill trade? | Smoother ITL for everyone in exchange for slightly higher TTFT on long prompts. |
| Recompute vs swap preemption cost? | Recompute costs GPU compute; swap costs PCIe bandwidth and CPU memory. |
| Does streaming reduce end-to-end latency? | No, it reduces perceived latency by showing tokens after TTFT. |
| What is coordinated omission? | A closed-loop load generator sends less during slow periods and under-records the tail. |
| Little's law for LLM serving? | Concurrent requests = arrival rate x average time in system. |
| MFU vs MBU? | MFU (FLOPs utilization) for prefill and training; MBU (bandwidth utilization) for decode. |
| Does nvidia-smi 100% GPU-Util mean saturated? | No, it only means some kernel was running during the sample period. |
| H100 BF16 ridge point? | About 295 FLOPs per byte (989 TFLOP/s / 3.35 TB/s). |
| Why is CPU offloading of weights so slow? | PCIe at about 32 GB/s means about 0.44 s per token for 14 GB of weights. |
| Speculative speedup with acceptance 0.6, k = 5, draft cost 10%? | About 2.38 tokens per 1.5 target passes, roughly 1.59x. |
| Does speculative decoding change the outputs? | No, rejection sampling preserves the target distribution exactly. |
| Best speculative method for code editing? | Prompt lookup (n-gram) decoding, since outputs copy the input. |
| Why can prefix cache hit rates be low? | Dynamic content early in the prompt, unstable serialization, replica spread, eviction. |
| Router vs cascade? | A router picks one model up front; a cascade tries small first and escalates. |
| How to serve 40 LoRA fine-tunes of one base model? | Multi-LoRA serving: one base model, adapters swapped per request in the same batch. |
| How to fit a 70B FP16 model on 8 x 80 GB GPUs? | Tensor parallelism over NVLink: TP = 4 for two replicas or TP = 8 for one. |
| Autoscale LLM replicas on which signals? | Queue depth, KV cache utilization, preemptions and p95 TTFT, not CPU. |
| Why is beam search rare for chat? | Bland output, B x compute and KV memory, and it cannot stream reliably. |
| What does min-p do? | Keeps tokens with probability at least p_min times the top token's probability. |
| FP16 maximum value? | 65,504, which is why FP16 overflows where BF16 does not. |
| What is NF4? | A 4-bit format with levels at normal-distribution quantiles, from QLoRA. |
| Effective bits for INT4 with group size 128 and FP16 scales? | 4.125 bits per weight, about 3.6 GB for a 7B model. |
| Why is a "4-bit" 7B file bigger than 3.5 GB? | Scales, zero points, 16-bit embeddings and output head, and metadata. |
| What is the straight-through estimator? | Treat rounding as identity in the backward pass so QAT gradients flow. |
| Static vs dynamic activation quantization? | Static calibrates scales offline; dynamic computes them per batch or token at runtime. |
| Why do INT8 matmuls accumulate in INT32? | INT8 products reach about 16,000 and thousands are summed, so 32 bits avoid overflow. |
| What is dark knowledge? | The teacher's small probabilities on wrong classes that encode class similarity. |
| Is a bigger teacher always better? | No, a large capacity gap can hurt; teacher assistants or mid-size teachers help. |
| Can a student beat its teacher? | Sometimes (Born-Again Networks), but rarely when the student is much smaller. |
| Can we train on a commercial API's outputs? | Check the provider's terms, the student's license and data-sharing rules with legal. |
| What is 2:4 sparsity? | Two zeros in every four weights; sparse tensor cores give up to 2x matmul throughput. |
| Low-rank break-even for a 4096 x 4096 layer? | Rank must be below mn/(m+n) = 2048, and well below it for real speedups. |

## How to practice

- **Answer out loud first.** Read the question, close your eyes or look away, and answer as if the interviewer is waiting. Only then open the answer and compare.
- **Time yourself to about 2 minutes.** A senior answer gives the definition, the mechanism, one trade-off and one concrete number within that window, then stops and lets the interviewer pick the follow-up.
- **Explain it to a non-expert.** If you cannot explain the KV cache or prefill vs decode to a product manager in three sentences, you do not own it yet.
- **Then go deeper.** For each question, rehearse the "Key points to mention" as likely follow-ups, and practice the back-of-envelope numbers (KV per token, decode bound, model memory) until you can do them without notes.
- **Use the rapid-fire table daily** in the last week: cover the answer column and go down the list in under ten minutes.
- **Revisit weak spots in the chapters.** Every answer links to its deep-dive chapter; reread only the sections behind the questions you missed.
