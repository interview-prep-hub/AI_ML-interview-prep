# Question Bank: RAG, Embeddings and Vector Databases

> **What this is:** 121 interview questions with model answers on embeddings, retrieval, vector search, RAG evaluation, debugging, production ingestion and secure multi-tenant RAG, for rapid self-testing and last-week revision before a Senior Software Engineer (AI/ML) interview.
>
> **How to use it:** read the question, answer it out loud first (aim for about 2 minutes), then open the answer and compare. Mark the questions you fumbled and revisit them, and use the "Deep dive" links to go back to the chapter that teaches the topic in full.
>
> **Source chapters:** [01 What Is RAG](../08-rag/01-what-is-rag-and-why-it-matters.md) · [02 Embeddings](../08-rag/02-text-embeddings-and-similarity-search.md) · [03 Ingestion and Parsing](../08-rag/03-document-ingestion-and-parsing.md) · [04 Chunking](../08-rag/04-chunking-strategies.md) · [05 BM25, Dense and Hybrid](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md) · [06 ANN, HNSW, IVF and PQ](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md) · [07 Vector Databases](../08-rag/07-vector-databases-in-practice.md) · [08 Reranking and Query Transformation](../08-rag/08-reranking-and-query-transformation.md) · [09 Advanced RAG Patterns](../08-rag/09-advanced-rag-patterns.md) · [10 Evaluating RAG](../08-rag/10-rag-evaluation-metrics-and-ragas.md) · [11 Debugging RAG](../08-rag/11-debugging-rag-systems.md) · [12 Production Ingestion](../08-rag/12-production-ingestion-pipelines-at-scale.md) · [13 Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md)

## Table of contents

- [Embeddings and similarity search](#embeddings-and-similarity-search) (Q1-Q11)
- [The RAG pipeline end to end](#the-rag-pipeline-end-to-end) (Q12-Q20)
- [Document ingestion and parsing](#document-ingestion-and-parsing) (Q21-Q23)
- [Chunking](#chunking) (Q24-Q30)
- [Retrieval: BM25, dense and hybrid search](#retrieval-bm25-dense-and-hybrid-search) (Q31-Q39)
- [Approximate nearest-neighbor search and vector databases](#approximate-nearest-neighbor-search-and-vector-databases) (Q40-Q52)
- [Reranking and query transformation](#reranking-and-query-transformation) (Q53-Q62)
- [Advanced RAG: filters, parent-child, contextual retrieval and multi-hop](#advanced-rag-filters-parent-child-contextual-retrieval-and-multi-hop) (Q63-Q76)
- [Evaluating RAG](#evaluating-rag) (Q77-Q90)
- [Debugging RAG systems](#debugging-rag-systems) (Q91-Q99)
- [Production ingestion: idempotency, updates, deletions and re-embedding](#production-ingestion-idempotency-updates-deletions-and-re-embedding) (Q100-Q110)
- [Secure multi-tenant RAG](#secure-multi-tenant-rag) (Q111-Q121)
- [Rapid-fire round](#rapid-fire-round)
- [How to practice](#how-to-practice)

## Embeddings and similarity search

<details>
<summary><b>Q1. Explain how an embedding model represents text.</b></summary>

**Answer:** An embedding model maps a piece of text to a fixed-length vector of real numbers (for example 768 floats) so that texts the model was trained to treat as related end up close together in that space. Internally a transformer produces a contextual vector for every token, and a pooling step (mean, CLS or last-token) combines them into one vector for the whole text. Contrastive training then shapes the space so that angle or distance reflects relevance, which turns search into a nearest-neighbor lookup. Because the length is fixed regardless of input length, a 5-word query is directly comparable with a 300-word passage. No single dimension is interpretable: meaning lives in relative directions, and rotating the whole space would leave every similarity unchanged. The trade-off is compression: one vector per chunk is weak on exact identifiers, numbers, negation and chunks that mix several topics.

**Key points to mention:**
- "Similar" is defined by the training data: paraphrase pairs, question-answer pairs and topic data produce different notions of similarity.
- Inputs longer than the model's `max_seq_length` are silently truncated, so the tail of a long chunk is invisible.
- Mitigate single-vector weaknesses with hybrid search (BM25), a reranker and good chunking.
- Check a model's notion of similarity by reading nearest neighbors of known queries, then measuring recall@k on your own labeled queries.

**Deep dive:** [What an embedding model does](../08-rag/02-text-embeddings-and-similarity-search.md#what-an-embedding-model-does)

</details>

<details>
<summary><b>Q2. What is the difference between token embeddings and sentence embeddings?</b></summary>

**Answer:** Token embeddings are one vector per token: either the static lookup-table vector at the input or, more usefully here, the contextual vector the transformer outputs for each token, giving a $T \times d$ matrix for $T$ tokens. A sentence embedding is a single $d$-dimensional vector for the whole text, produced by pooling the token vectors and trained so whole texts can be compared. Retrieval indexes store sentence embeddings, one per chunk. The pooling must match how the model was trained: CLS pooling for models trained that way (several BGE models), mean pooling for many sentence-transformers and E5 models, and last-token pooling for decoder-based embedders. Raw pretrained BERT pooled outputs make poor sentence embeddings because masked-language-model pretraining never optimized whole-text comparability. Late-interaction models such as ColBERT keep the token-level vectors for retrieval, trading much more storage for finer matching.

**Key points to mention:**
- Mean pooling must mask padding tokens, or the same text gets different vectors depending on batch composition.
- Sentence-BERT (Reimers and Gurevych, 2019) showed that raw BERT embeddings perform poorly for similarity and need fine-tuning on sentence pairs; Ethayarajh (2019) showed they are anisotropic (crowded into a narrow cone).
- Shapes: token level $T \times d$, sentence level $d$.
- ColBERT is the exception that stores token vectors (see Q60).

**Deep dive:** [From token embeddings to sentence embeddings: pooling](../08-rag/02-text-embeddings-and-similarity-search.md#from-token-embeddings-to-sentence-embeddings-pooling)

</details>

<details>
<summary><b>Q3. How are contrastive embedding models trained?</b></summary>

**Answer:** You collect positive pairs, such as a query and the passage that answers it, and train a bi-encoder so each pair scores higher than the anchor scores with negatives. In a batch of $B$ pairs, each anchor's other $B - 1$ positives act as free in-batch negatives, often supplemented with mined hard negatives. The loss is InfoNCE: a softmax cross-entropy over similarities divided by a temperature $\tau$, where the correct class is the true positive. Worked example: cosines of 0.8 for the positive and 0.3 and 0.1 for negatives at $\tau = 0.1$ give logits 8, 3 and 1, a positive probability of about 0.992 and a loss of about 0.0076; replacing the 0.3 negative with a hard negative at 0.75 (logits 8, 7.5 and 1) raises the loss to about 0.475, which is why hard negatives drive learning. A typical recipe is weakly supervised pretraining on huge mined pairs, then supervised fine-tuning with hard negatives, sometimes with distillation from a cross-encoder.

**Key points to mention:**
- Formula: $\mathcal{L}_i = -\log \frac{\exp(s(q_i, p_i)/\tau)}{\sum_j \exp(s(q_i, p_j)/\tau)}$, averaged over the batch.
- Temperature around 0.01 to 0.1 is common; sentence-transformers' default scale of 20 means $\tau = 0.05$.
- Larger batches give more negatives; scale them with cross-device negatives or gradient caching.
- False negatives (duplicates in the batch, unlabeled relevant passages mined as negatives) push relevant texts apart; filter them with a cross-encoder and use a no-duplicates sampler.

**Deep dive:** [How contrastive embedding models are trained](../08-rag/02-text-embeddings-and-similarity-search.md#how-contrastive-embedding-models-are-trained)

</details>

<details>
<summary><b>Q4. Why might the same text produce different embeddings with different models?</b></summary>

**Answer:** Each model learns its own coordinate system. Architecture, dimension, tokenizer, pooling, training data and objective all change the vector, and even two identically trained models can differ by an arbitrary rotation, because the contrastive loss only constrains similarities between vectors, not absolute coordinates. Think of two maps of the same city with different orientations and units: distances within one map are meaningful, coordinates across maps are not. The consequences are practical: never mix models (or versions of one model) in one index, never compare a query from model B with documents from model A, and re-calibrate similarity thresholds when you switch, since score distributions differ. Fine-tuning also creates a new space, so the corpus must be re-embedded.

**Key points to mention:**
- Store model name, version, dimension, normalization and prefixes as index metadata and validate them on every write and query.
- Same-dimension models are not interchangeable; mixing them fails silently.
- Tiny numeric differences from hardware, FP16 vs FP32 or batching rarely change rankings but break exact-equality tests.
- To compare two models, compare their behavior (neighbor lists, recall@k on your eval set), never raw coordinates.

**Deep dive:** [Why the same text gets different embeddings in different models](../08-rag/02-text-embeddings-and-similarity-search.md#why-the-same-text-gets-different-embeddings-in-different-models)

</details>

<details>
<summary><b>Q5. Why should query and document embeddings be compatible?</b></summary>

**Answer:** Retrieval only works because training placed a query and its relevant passage near each other in one specific vector space. If queries and documents are embedded with different models, versions, prefixes, pooling or normalization, similarity scores stop reflecting relevance, and the system returns plausible-looking but wrong results without raising any error. Many asymmetric models also need side-specific inputs: E5 expects `"query: "` and `"passage: "` prefixes, Nomic uses `"search_query: "` and `"search_document: "`, and hosted APIs expose an input or task type parameter. The engineering fix is a single `embed_query()` and `embed_document()` wrapper so nobody calls the raw model inconsistently, plus metadata checks that the query encoder matches the index. A classic debugging symptom is a sharp recall drop after a "harmless" refactor that dropped a prefix.

**Key points to mention:**
- Same model (or jointly trained towers, as in DPR), same version, same preprocessing on both sides.
- Symmetric tasks (duplicate detection) treat both sides identically; asymmetric tasks (RAG) often need prefixes.
- Prefix strings are version-specific: check the model card.
- Add a canary test on every deploy: embed known (query, relevant chunk) pairs and fail if their ranks drop.

**Deep dive:** [Query and document compatibility](../08-rag/02-text-embeddings-and-similarity-search.md#query-and-document-compatibility)

</details>

<details>
<summary><b>Q6. When should you normalize embeddings?</b></summary>

**Answer:** Normalize (scale each vector to unit L2 length, $\hat{v} = v / \lVert v \rVert$) when the model was trained with cosine similarity, which covers most general-purpose text embedders. After normalizing, dot product equals cosine and Euclidean distance gives the same ranking, so you can use the fastest inner-product indexes. Do not normalize when the model was trained for raw dot product and its vector length carries signal, as with some MS MARCO models whose cards say "use dot product". Whatever you choose, apply it identically to queries and documents, and re-normalize after truncating Matryoshka embeddings. Example: $(3, 4)$ has norm 5 and becomes $(0.6, 0.8)$.

**Key points to mention:**
- Normalizing at write time avoids computing norms at query time and enables `IndexFlatIP` or an `ip` metric.
- Re-normalizing an already unit-length vector is harmless; many hosted APIs return unit vectors.
- Quick check: compute norms of stored vectors and live query vectors; a mix of 1.0 and other values means one side skipped normalization.
- Measure recall after quantization or truncation, not just before.

**Deep dive:** [Normalization: what it is and when to do it](../08-rag/02-text-embeddings-and-similarity-search.md#normalization-what-it-is-and-when-to-do-it)

</details>

<details>
<summary><b>Q7. What is the difference between cosine similarity, dot product, and Euclidean distance? (Also asked as: what are cosine similarity and dot product?)</b></summary>

**Answer:** The dot product $\sum_i a_i b_i$ depends on both the angle between vectors and their lengths. Cosine similarity divides the dot product by both lengths, so it measures only the angle and ranges from -1 to 1. Euclidean distance is the straight-line gap between the points, where lower is better. For unit vectors all three give the same ranking, because $\lVert a - b \rVert^2 = 2 - 2\cos(a, b)$ and the dot product equals the cosine. Without normalization they can disagree: for query $(1, 0)$ and documents $a = (1, 0)$ and $b = (3, 1)$, the dot product ranks $b$ first (3 vs 1) while cosine ranks $a$ first (1.0 vs 0.949). The rule is to use the metric the model was trained with, and for cosine-trained models to normalize once and use the cheaper inner product.

**Key points to mention:**
- Example: $[1,2,2] \cdot [2,1,2] = 8$, both lengths are 3, cosine $= 8/9 \approx 0.889$.
- Databases often return cosine distance $1 - \cos$, and FAISS `IndexFlatL2` returns squared L2: sort and threshold accordingly.
- An inner-product index fed unnormalized vectors from a cosine-trained model gives subtly wrong rankings.
- Cosine values are not probabilities; any threshold must be calibrated per model on labeled data.

**Deep dive:** [Cosine similarity vs dot product vs Euclidean distance](../08-rag/02-text-embeddings-and-similarity-search.md#cosine-similarity-vs-dot-product-vs-euclidean-distance)

</details>

<details>
<summary><b>Q8. How would you estimate and reduce the storage cost of a vector index?</b></summary>

**Answer:** Raw storage is number of vectors times dimensions times bytes per value: 10 million chunks at 1,024 float32 dimensions is about 41 GB before any index overhead. The same corpus is about 20.5 GB in float16, 10.2 GB in int8 and 1.3 GB as binary vectors. A Matryoshka-trained model lets you truncate to, say, 256 dimensions (about 10.2 GB in float32) because it was trained so that prefixes of the vector remain useful. Product quantization goes further, compressing a 768-dimensional float32 vector from 3,072 bytes to 96 bytes. Each step costs some recall, so the standard pattern is a funnel: search with small or quantized vectors, then rescore a shortlist with full-precision vectors kept on SSD. Verify recall@k on your own eval set after every compression step.

**Key points to mention:**
- HNSW adds link overhead per vector (about $2M \times 4$ bytes), and RAM rather than disk is usually the binding constraint.
- Matryoshka truncation only works for models trained with that objective; re-normalize after truncating.
- Smaller dimensions also speed up every comparison, since cost is linear in $d$.
- Budget headroom for rebuilds and migrations (two copies during a blue-green swap).

**Deep dive:** [Embedding dimensions, storage cost and Matryoshka embeddings](../08-rag/02-text-embeddings-and-similarity-search.md#embedding-dimensions-storage-cost-and-matryoshka-embeddings)

</details>

<details>
<summary><b>Q9. How would you choose an embedding model for a new RAG system?</b></summary>

**Answer:** Use the MTEB retrieval scores (not the overall average) only to build a shortlist, filtered by language, domain, maximum input length, dimension, license and hosting constraints. Then build a labeled eval set of real queries with the chunk IDs that answer them, embed the corpus with each candidate using its correct prefixes and normalization, and compare recall@k, MRR and nDCG@k. Evaluate both the embedder alone and the full pipeline with hybrid search and reranking, because a reranker can erase small embedder differences. Slice results by query type (identifiers, long questions, each language, each document type), and include latency, throughput, cost and storage in the decision. Public leaderboard gaps of a point or two rarely transfer to a specific domain.

**Key points to mention:**
- Eval set sources: logs, tickets, FAQs, experts, or LLM-generated questions reviewed by humans; split by document to avoid leakage.
- Check `max_seq_length` against your chunk sizes (256 tokens for `all-MiniLM-L6-v2`, 8,191 for OpenAI's text-embedding-3).
- Be wary of benchmark overlap with training data.
- Keep the eval set as a regression test for future model or chunking changes.

**Deep dive:** [Choosing an embedding model](../08-rag/02-text-embeddings-and-similarity-search.md#choosing-an-embedding-model)

</details>

<details>
<summary><b>Q10. How would you fine-tune an embedding model for a specialized domain, and what can go wrong?</b></summary>

**Answer:** First try cheaper levers (better chunking, hybrid search, a reranker, a different base model), because fine-tuning defines a new vector space and forces a full re-embed. If it is still needed, build (query, relevant passage) pairs from logs, FAQs or LLM-generated questions per chunk, mine hard negatives with the base model and filter false negatives with a cross-encoder. Train with an in-batch-negatives contrastive loss such as `MultipleNegativesRankingLoss`, large batches, a no-duplicates sampler, about one epoch and a low learning rate, keeping the base model's prefix conventions. Evaluate on held-out documents before and after, on both the domain slice and general queries. Risks include forgetting general ability, overfitting to synthetic question style, false negatives and leakage when the same document appears in train and test.

**Key points to mention:**
- Synthetic questions tend to copy chunk wording; review samples and encourage paraphrased questions.
- Optionally wrap the loss in a Matryoshka loss so truncation keeps working.
- Version the model, training data and eval results in a registry; the fine-tuned model defines a new index.
- Plan the migration (Q107) before you start training.

**Deep dive:** [Fine-tuning an embedding model on domain data](../08-rag/02-text-embeddings-and-similarity-search.md#fine-tuning-an-embedding-model-on-domain-data)

</details>

<details>
<summary><b>Q11. All your cosine similarities fall between 0.7 and 0.9, even for unrelated pairs. Is the model broken?</b></summary>

**Answer:** Not necessarily. Many models produce anisotropic embeddings that occupy a narrow cone, so absolute scores are compressed and only the relative ranking carries meaning. A very low training temperature compresses scores too: the E5 model card notes its cosines cluster around 0.7 to 1.0 because it was trained with $\tau = 0.01$. The test is whether relevant documents still rank above irrelevant ones on a labeled eval set; if they do, the model is fine and any threshold simply has to be calibrated to this model's score distribution. If rankings are also bad, look for the usual bugs: wrong pooling, unmasked padding, missing prefixes, a raw pretrained model, or mixed model versions.

**Key points to mention:**
- Plot score histograms for known relevant vs irrelevant pairs and pick thresholds from the overlap.
- Cosine scores are not probabilities; reranker scores are usually better for "no good answer" cutoffs.
- Thresholds must be re-calibrated whenever the model changes.
- Do not "fix" it by rescaling scores; fix ranking problems at their cause.

**Deep dive:** [Scores are not probabilities](../08-rag/02-text-embeddings-and-similarity-search.md#scores-are-not-probabilities)

</details>

## The RAG pipeline end to end

<details>
<summary><b>Q12. Explain the complete RAG pipeline.</b></summary>

**Answer:** RAG has an offline indexing half and an online query half. Offline, you load documents from source systems together with their permissions and metadata, parse them into clean structured text, clean and deduplicate, split into chunks, embed each chunk, and store vectors, text and metadata in an index (usually vector plus keyword). Online, you authenticate the user, optionally rewrite the query, retrieve 20-100 candidates with permission filters applied inside the search, rerank them, assemble the best 3-8 into a numbered-source prompt within a token budget, generate, map citations back to sources, and post-check the answer for faithfulness before returning it. Every stage is a trade-off knob (chunk size, first-stage k, dense vs hybrid, reranker on or off, final k) that should be settled on a golden set, not by intuition. Measure recall@k for retrieval, faithfulness and correctness for generation, citation validity, abstention on unanswerable questions, and p95 latency and cost per query.

**Key points to mention:**
- Offline concerns: incremental sync, deterministic chunk IDs for idempotent upserts, deletion propagation, embedding-model versioning.
- Online concerns: latency budget (generation usually dominates), recall-oriented first stage, precision-oriented reranking.
- Always include an abstention path so failed retrieval yields "I don't know" instead of a guess.
- Pitfall: describing only "embed, search, prompt" signals a junior understanding.

**Deep dive:** [The complete pipeline at a glance](../08-rag/01-what-is-rag-and-why-it-matters.md#the-complete-pipeline-at-a-glance), [System Design: A Scalable Enterprise RAG System](../11-system-design/02-design-a-scalable-enterprise-rag-system.md)

</details>

<details>
<summary><b>Q13. How do document parsing, chunking, embeddings, retrieval, reranking, and generation interact?</b></summary>

**Answer:** They form a chain where each stage can only work with what the previous one produced, so errors compound. Parsing sets the upper bound on available information, chunking defines the retrievable units, embeddings define what "similar" means, retrieval sets the recall ceiling, reranking decides which few chunks enter the limited context, and generation turns that context into a faithful answer or not. A rough model is $P(\text{correct}) \approx \prod_i p_i$: four stages at 0.95, 0.90, 0.85 and 0.90 give only about 0.65 end to end, and the weakest stage limits the total. The stages are coupled: chunk size interacts with the embedding model's max input length and with how many chunks fit in the prompt, and reranker cost scales with first-stage k. So a change in one stage, such as chunk size, forces re-embedding and re-tuning of k and thresholds, and re-running the full evaluation.

**Key points to mention:**
- Error propagation examples: a scrambled table leads to wrong numbers in a fluent answer; a stale index leads to outdated citations.
- Debugging rule: check first whether the gold chunk was retrieved at all; that splits upstream from downstream problems.
- Measure each stage: recall@k, rank of the gold chunk after reranking, faithfulness.
- Changing the embedding model invalidates the whole index.

**Deep dive:** [How the stages interact and how errors propagate](../08-rag/01-what-is-rag-and-why-it-matters.md#how-the-stages-interact-and-how-errors-propagate)

</details>

<details>
<summary><b>Q14. How do you retrieve relevant chunks?</b></summary>

**Answer:** Start from the user's authenticated identity and a standalone version of the query (rewritten if it is a chat follow-up). Run a hybrid first stage: BM25 for exact terms and identifiers and dense vector search for paraphrases, both with tenant and permission filters applied inside the query, taking roughly 50-100 candidates per leg. Fuse the lists with reciprocal rank fusion, deduplicate, and send the candidates to a cross-encoder reranker that scores each (query, chunk) pair, keeping the top 3-8 or those above a calibrated score threshold. The first stage optimizes recall (the reranker can only reorder what it receives), and the second stage optimizes precision for the small context. Tune per-leg depth, k and the final cutoff on a labeled set while watching p95 latency.

**Key points to mention:**
- Metadata filters (version, date, product) remove look-alike chunks from the wrong context.
- Run both legs in parallel so latency is the slower leg, not the sum.
- Keep a fallback (fused order) if the reranker times out.
- Route identifier-heavy queries toward BM25 or exact-match fields.

**Deep dive:** [Online querying, step by step](../08-rag/01-what-is-rag-and-why-it-matters.md#online-querying-step-by-step), [Combining reranking with hybrid retrieval](../08-rag/08-reranking-and-query-transformation.md#combining-reranking-with-hybrid-retrieval)

</details>

<details>
<summary><b>Q15. How is the context passed to the LLM?</b></summary>

**Answer:** As plain text in the prompt. The selected chunks are formatted into a template with clear delimiters and short numeric source IDs, ordered with the most relevant first, under an explicit token budget, together with instructions to answer only from the sources, cite them and abstain when they do not contain the answer. The budget is the window minus system prompt, history, query, output reserve and a margin: for example $8{,}192 - 400 - 0 - 50 - 1{,}000 - 200 = 6{,}542$ tokens, so at most 19 sources of 330 tokens fit, though 3-8 reranked chunks usually work better. The budget is a cap, not a target, because extra chunks add cost, prefill latency and distractors. Place sources in the user turn rather than the system prompt, since retrieved text is untrusted, and keep static instructions first so prompt caching can reuse the prefix.

**Key points to mention:**
- Ordering: best first, because models use mid-context information less well ("Lost in the Middle", Liu et al., 2023); keep same-document chunks in original order and deduplicate.
- Include light metadata (title, date) to help resolve conflicts between versions.
- Multi-turn: condense history into a standalone query for retrieval and keep only a summary in the prompt.
- Enforce the budget in your own code: some serving setups silently truncate an oversized prompt, possibly cutting instructions.

**Deep dive:** [How context is passed to the LLM](../08-rag/01-what-is-rag-and-why-it-matters.md#how-context-is-passed-to-the-llm)

</details>

<details>
<summary><b>Q16. How do you preserve source attribution?</b></summary>

**Answer:** By carrying stable IDs and metadata through every stage. At ingestion each document and chunk gets a deterministic ID (for example a hash of doc ID, version and text) plus locators: URL, page, section path, character offsets and version. Those travel with the chunk through retrieval, reranking and context assembly; the prompt shows short numbered sources while code keeps the number-to-chunk mapping. Any transformation (merging chunks, compression, deduplication) must keep the list of source IDs it came from. After generation, the citation markers are mapped back to chunks and rendered as links, and everything is logged with index, prompt and model versions so any answer can be reproduced later. When a document is updated, old versions (or their hashes and text) stay addressable for as long as audit logs reference them.

**Key points to mention:**
- Deterministic IDs make re-ingestion idempotent and keep old logs meaningful; random IDs per run break both.
- Offsets point into the original parsed document, never into transformed text.
- Store page and bounding box for PDFs and row and column for tables so the UI can highlight the exact source.
- See Q68 for provenance through advanced transformations.

**Deep dive:** [Preserving source attribution end to end](../08-rag/01-what-is-rag-and-why-it-matters.md#preserving-source-attribution-end-to-end)

</details>

<details>
<summary><b>Q17. How do you generate citations?</b></summary>

**Answer:** Present each chunk in the prompt with a short stable label, instruct the model to cite the supporting label after every factual sentence (not as a list at the end), and parse the markers from the output, or use structured JSON output with `source_ids` per sentence. Then verify: reject any citation that points to a source not provided in the context, and check that each cited chunk actually supports its sentence with string matching on required quotes, an NLI entailment model or a validated LLM judge. Unsupported sentences are removed, flagged or trigger a regeneration. Track citation precision (share of citations that support their sentence) and citation recall (share of factual sentences with at least one supporting citation) as release metrics. A common bug is plumbing: positional numbers that shift after reordering make citations point to the wrong chunk, so use stable chunk IDs in the prompt.

**Key points to mention:**
- Never trust citations blindly: models cite a real source number for a claim that source does not support.
- Provider-native citation features exist but are vendor-specific.
- Citation validity (cited ID was in the context) is a cheap deterministic check to run on every request.
- A correct answer with a wrong citation is still a failure for audit-sensitive products.

**Deep dive:** [Citation precision and citation recall](../08-rag/10-rag-evaluation-metrics-and-ragas.md#citation-precision-and-citation-recall), [Preserving source attribution end to end](../08-rag/01-what-is-rag-and-why-it-matters.md#preserving-source-attribution-end-to-end)

</details>

<details>
<summary><b>Q18. Why does a RAG system still hallucinate, and how do you reduce it?</b></summary>

**Answer:** Because the generator is still a language model: it can ignore the context, mix in prior knowledge, overgeneralize from a partially relevant chunk, or answer when retrieval returned nothing useful. Many RAG "hallucinations" start as retrieval failures: the right chunk was missing, so the model filled the gap fluently. Reduce it in order: improve retrieval recall and precision, pass fewer and better chunks, instruct and explicitly allow abstention, gate on a relevance threshold (if the best reranker score is low, abstain), require per-claim citations, use low temperature for factual answers, and run a post-generation faithfulness check with NLI or an LLM judge. In high-risk domains, block or route low-support answers to a human. Track faithfulness on a golden set and on sampled production traffic.

**Key points to mention:**
- Check recall first; swapping the LLM cannot fix missing evidence.
- A stricter "answer only from sources" prompt can raise false abstentions; measure both directions.
- Faithfulness alone can be gamed by terse answers; pair it with correctness and abstention rate.
- See the practical exercise in Q96.

**Deep dive:** [Debugging Case: A RAG Application Gives Fluent but Unsupported Answers](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md), [Hallucinations and grounding](../05-llms/06-hallucinations-and-grounding.md)

</details>

<details>
<summary><b>Q19. With context windows of hundreds of thousands of tokens, do we still need RAG?</b></summary>

**Answer:** Yes, for most production knowledge systems: long context changes how much you can pass after retrieval, not whether you need retrieval. Corpora are usually far larger than any window, and paying for and waiting on hundreds of thousands of input tokens on every request is expensive and slow, since cost scales with input tokens and prefill latency grows with prompt length. Models also still under-use information buried in very long contexts, so precision keeps mattering. Retrieval additionally gives per-user permission filtering, freshness without retraining, and precise chunk-level attribution. Long context shines when the relevant material is small and known in advance (one contract, one report), and a common hybrid is to retrieve generously, for example whole parent sections, because the window allows it.

**Key points to mention:**
- Prompt caching only helps if the same long prefix is reused across requests.
- KV cache memory for very long prompts limits concurrency on the serving side.
- Decide with an evaluation on your own questions comparing quality, cost and latency.
- Fine-tuning is for behavior and style, not for fresh or permissioned facts.

**Deep dive:** [RAG vs long-context prompting vs fine-tuning](../08-rag/01-what-is-rag-and-why-it-matters.md#rag-vs-long-context-prompting-vs-fine-tuning), [Prompting vs RAG vs fine-tuning decision guide](../05-llms/08-prompting-vs-rag-vs-fine-tuning-decision-guide.md)

</details>

<details>
<summary><b>Q20. How do you choose k for retrieval and for the final context?</b></summary>

**Answer:** Use two different values. The first stage uses a large k (for example 20-100) to maximize recall, because the reranker can only reorder what it receives; plot recall@k against k and pick the knee where recall stops improving much. After reranking, keep a small number (3-8) or apply a calibrated score threshold to maximize precision inside the token budget. Reranker latency grows roughly linearly with the first-stage k, so the knee also bounds cost. For the final context, measure faithfulness and answer quality as you vary k, because too many chunks add distractors, tokens and latency. Broad or multi-part questions may need more chunks or diversity (MMR), while simple lookups need one or two, so a dynamic k based on reranker scores often beats a fixed number.

**Key points to mention:**
- First stage is judged by Recall@K at the depth the next stage consumes; final list by nDCG@n and answer quality.
- Chunk size and final k compete for the same token budget.
- Re-tune after changing chunk size, the reranker or the embedding model.
- Log how many chunks were dropped by the budget.

**Deep dive:** [When to rerank and how many candidates](../08-rag/08-reranking-and-query-transformation.md#when-to-rerank-and-how-many-candidates)

</details>

## Document ingestion and parsing

<details>
<summary><b>Q21. How do you ingest documents?</b></summary>

**Answer:** As a pipeline: connectors pull content plus permissions and change events from each source, a router sends each file (or each PDF page) to the right parser, and parsing produces a common element model with structure (headings, tables, lists, figures) and rich metadata. Then come cleaning, Unicode normalization, deduplication and cross-reference linking, automated quality checks with fallbacks to heavier parsers, and only then chunking, embedding and indexing. Intermediate outputs are stored so you can re-chunk without re-parsing, and everything is incremental and idempotent. Metadata captured here (doc ID, URL, title, section path, page, dates, doc type, ACL, parser version, content hash) drives filtering, citations, freshness and security later, and cannot be added later without re-ingesting. Measure parse pass rate per source, ingestion lag, coverage (source vs index counts, including deletions) and cost per page.

**Key points to mention:**
- Connectors: least-privilege service account, rate-limit backoff, webhooks or delta APIs first with polling as fallback, ACL capture at ingestion.
- Route by MIME type sniffed from content, not just the file extension.
- Quality checks: characters per page, garbage ratio, OCR confidence, table consistency.
- The most common real failure is a parser that silently scrambles multi-column text or tables, so sample and read extracted text early.

**Deep dive:** [The ingestion pipeline at a glance](../08-rag/03-document-ingestion-and-parsing.md#the-ingestion-pipeline-at-a-glance)

</details>

<details>
<summary><b>Q22. How do you parse PDFs, tables, and structured data?</b></summary>

**Answer:** A PDF is positioned glyphs, not paragraphs, so reading order must be reconstructed. First detect per page whether there is a usable text layer: born-digital pages go to a fast extractor with layout-aware reading order, scanned pages go to OCR at about 300 DPI with deskewing, and complex pages go to a layout or table-structure model or a vision-language model. Tables are extracted as real grids and represented as Markdown (or HTML when merged cells matter), with the header row repeated in every chunk and a short natural-language summary for retrieval. Truly structured data (SQL tables, large spreadsheets) stays in a database and is answered by a text-to-SQL tool, because top-k retrieval cannot sum or average 400 rows. Routing per page buys most of the accuracy at a fraction of the cost: in the course's 100,000-page example, \$150 of VLM calls instead of \$1,000.

**Key points to mention:**
- Remove repeated headers and footers, fix hyphenation, normalize ligatures.
- Evaluate with CER on text, cell accuracy or TEDS on tables, and end-to-end recall on table questions.
- Silent failure modes: interleaved columns, broken character maps, merged cells returned as empty, tables split across pages.
- VLM transcription can produce clean, plausible, wrong numbers; cross-check numbers on high-stakes documents.

**Deep dive:** [Text-based vs scanned PDFs and OCR](../08-rag/03-document-ingestion-and-parsing.md#text-based-vs-scanned-pdfs-and-ocr), [Tables](../08-rag/03-document-ingestion-and-parsing.md#tables), [Structured data: SQL tables and spreadsheets](../08-rag/03-document-ingestion-and-parsing.md#structured-data-sql-tables-and-spreadsheets)

</details>

<details>
<summary><b>Q23. Users ask "what was our total revenue by region last quarter?" over a folder of spreadsheets. Would you use RAG?</b></summary>

**Answer:** Not plain vector retrieval. The question needs exact aggregation over many rows, and top-k chunk retrieval sees maybe 5 of 400 rows, so an LLM "averaging" them gives a confident wrong answer (in the course example, 39.5 instead of the true 50). Instead, load the spreadsheets into a database or DataFrames, describe the schema with column descriptions and example values, and give the LLM a text-to-SQL (or code execution) tool, using retrieval only to find which table is relevant. The folder rarely shares one schema, so harmonize column names, currencies and region labels and deduplicate copies, or the SQL sums the same revenue twice. Resolve "last quarter" to explicit dates (fiscal vs calendar), and show the generated SQL or result table to the user.

**Key points to mention:**
- Text-to-SQL safety: read-only user, row and column restrictions per user, timeouts, row limits, query logging.
- Evaluate with question-to-expected-result pairs, not text similarity.
- Keep row serialization for fuzzy lookups ("that cheap USB hub").
- Route by question type: aggregation to SQL, explanation to RAG.

**Deep dive:** [Structured data: SQL tables and spreadsheets](../08-rag/03-document-ingestion-and-parsing.md#structured-data-sql-tables-and-spreadsheets)

</details>

## Chunking

<details>
<summary><b>Q24. How do you choose chunk size and overlap?</b></summary>

**Answer:** Start from the constraints and the content, then decide with data. The hard ceiling is the embedding model's max sequence length (including prepended headers and special tokens), and the soft budget is how many chunks fit in the LLM context. Within that, respect document structure first (sections, tables, functions), start around 256-512 tokens with about 10-20% overlap for prose, then sweep a few sizes and overlaps on a fixed evaluation set. Compare span-level Recall@K and nDCG at an equal token budget plus answer correctness, and pick the cheapest configuration within noise of the best. Overlap protects facts that straddle a boundary at a storage cost of about $S/(S-O)$, for example $512/448 \approx 1.14$ for 64 tokens of overlap, so size it to about one typical sentence.

**Key points to mention:**
- Size trades context against focus: small chunks are sharp but fragmentary, large ones are diluted.
- Fact lookups favor smaller chunks; synthesis questions favor larger chunks or parent-child retrieval.
- Too much overlap creates near-duplicate chunks that crowd the top-K.
- Pitfalls: measuring in characters, ignoring header tokens, comparing configs at equal K instead of equal token budget.

**Deep dive:** [Fixed-size token chunking with overlap](../08-rag/04-chunking-strategies.md#fixed-size-token-chunking-with-overlap), [Typical starting ranges](../08-rag/04-chunking-strategies.md#typical-starting-ranges)

</details>

<details>
<summary><b>Q25. What happens with chunks that are too small or too large?</b></summary>

**Answer:** Too small, and chunks lose context ("it must be unused": unused what?), answers fragment across many chunks that may not all be retrieved, facts get split, and the index fills with low-information near-duplicate vectors. Too large, and the embedding becomes an average of several topics, so the relevant passage is diluted and ranks lower; each retrieved chunk wastes context tokens, fewer distinct chunks fit, and text beyond the embedding model's limit is silently truncated. Illustrative dilution: with mean pooling, a chunk with one refund sentence and three shipping sentences averages to $(0.25, 0.75)$, giving cosine $\approx 0.32$ with a refund query versus 1.0 for the focused chunk. In metrics, too small often shows decent Recall@K but poor answer correctness, and too large shows lower recall at a fixed token budget and higher latency.

**Key points to mention:**
- Fixes beyond resizing: prepend section headers, or use parent-child retrieval (small for matching, large for context).
- Large chunks also hurt generation through "lost in the middle".
- Spot problems without labels: a chunk-length histogram (a spike at the limit means truncation), truncation rate, reading random chunks.
- Orphan heading or footer chunks are a sign of chunks that are too small or structure-blind.

**Deep dive:** [The Goldilocks problem: too small vs too large](../08-rag/04-chunking-strategies.md#the-goldilocks-problem-too-small-vs-too-large)

</details>

<details>
<summary><b>Q26. How do semantic chunking and fixed-size chunking differ?</b></summary>

**Answer:** Fixed-size chunking cuts every $S - O$ tokens regardless of meaning: it is cheap, predictable and always within limits, but it can split ideas and mix topics. Semantic chunking embeds sentences (or small windows) and cuts where the cosine distance between neighbors, $d_i = 1 - \cos(e_i, e_{i+1})$, exceeds a threshold, often a percentile of the document's distances, so chunks follow topic shifts. It costs an extra embedding pass at ingestion, produces variable sizes, adds hyperparameters, still needs min and max size guards, and is harder to debug. Published evidence ("Is Semantic Chunking Worth the Computational Cost?", Qu, Tu and Bao, 2024) found its gains over good baselines inconsistent. If documents have headings, structure-aware chunking captures most topic boundaries for free, so semantic chunking is most useful inside long unstructured stretches and should be adopted only if it wins on your eval set.

**Key points to mention:**
- Percentile thresholds always create some boundaries, even in a single-topic document.
- Semantic boundaries depend on the embedding model, so upgrading the model also moves the boundaries: version them together.
- Compare against a structure-aware baseline at an equal token budget, including ingestion cost.
- Single-sentence embeddings are noisy; use sentence windows.

**Deep dive:** [Semantic chunking vs fixed-size chunking](../08-rag/04-chunking-strategies.md#semantic-chunking-vs-fixed-size-chunking)

</details>

<details>
<summary><b>Q27. How do you preserve tables, headings, and document hierarchy?</b></summary>

**Answer:** In three layers. First, parse structure at ingestion (layout-aware PDF parsing, HTML elements, Markdown headings) so it is not lost before chunking. Second, chunk along that structure: split at sections, keep small tables whole, split large tables by row groups with the caption and header row repeated, and never cut inside a row or a code function. Third, make each chunk self-describing: prepend the document title and section path to the embedded text (for example "Document: Acme Help Center / Section: Returns > Refunds") and store doc ID, section path, page, offsets and parent ID as metadata for filtering, citations and expansion. For retrieval, tables can be linearized row by row ("Product: Phone. Warranty: 1 year.") or summarized, while the LLM receives the original table.

**Key points to mention:**
- Merge headings into the following content to avoid orphan heading chunks.
- Keep headers short and budget their tokens against the embedding limit.
- Hierarchy enables parent-child retrieval and neighbor expansion.
- CI tests: every table chunk contains its header row; no chunk is heading-only.

**Deep dive:** [Preserving tables, headings and hierarchy](../08-rag/04-chunking-strategies.md#preserving-tables-headings-and-hierarchy)

</details>

<details>
<summary><b>Q28. How do token limits and embedding-model limits affect chunk size?</b></summary>

**Answer:** The embedding model's maximum sequence length is a hard ceiling: header plus body plus special tokens must fit, or the tail is silently truncated and becomes unretrievable. For example a 512-token model with a 40-token header leaves $512 - 40 - 2 = 470$ body tokens. The LLM's context window is a budget: the number of chunks you can include is roughly $\lfloor (C - P - Q - A)/(S + h) \rfloor$, so with an 8,192-token window, 500 system, 200 question and 1,000 answer tokens you have 6,492 tokens, which fits 11 chunks of 552 tokens or 21 of 296. Even when both limits are large, retrieval quality, cost and latency usually favor chunks far below the maximum, because one vector for thousands of tokens is diluted and long prompts pay prefill and "lost in the middle" costs.

**Key points to mention:**
- Count with the right tokenizer for each limit; embedding and LLM tokenizers differ.
- Limits vary widely: 256 tokens for `all-MiniLM-L6-v2`, 8,191 for OpenAI's text-embedding-3.
- Log each chunk's embedding-token length at ingestion and alert on any truncation.
- Re-check limits whenever you swap embedding models.

**Deep dive:** [How token limits shape chunk size](../08-rag/04-chunking-strategies.md#how-token-limits-shape-chunk-size)

</details>

<details>
<summary><b>Q29. How would you tune chunking using retrieval and answer-quality metrics?</b></summary>

**Answer:** Build a fixed evaluation set of real questions with gold evidence labeled as character spans in the source documents, so labels survive re-chunking. For each candidate configuration (strategy, size, overlap, headers), re-chunk and re-embed a representative corpus sample that still contains distractor documents, then measure span-level Recall@K and nDCG@K at an equal context-token budget. At 2,048 tokens that is $K = 16$ for 128-token chunks but $K = 4$ for 512-token chunks, which stops bigger chunks from winning trivially. Shortlist the best, run the full RAG pipeline, and score answer correctness, faithfulness, context precision, tokens and latency. Pick the cheapest configuration within statistical noise of the best (paired bootstrap), check per-segment results, and freeze it as a versioned config with regression tests.

**Key points to mention:**
- Hold everything else fixed: embedding model, retriever, prompt, LLM, token budget.
- Credit each gold span once when overlapping chunks both contain it.
- If retrieval improves but answers worsen, chunks are likely too fragmentary; read examples.
- Re-tune after adding a reranker or hybrid search, since they shift the optimum.

**Deep dive:** [Tuning chunking systematically with metrics](../08-rag/04-chunking-strategies.md#tuning-chunking-systematically-with-metrics)

</details>

<details>
<summary><b>Q30. What is late chunking and when would you use it?</b></summary>

**Answer:** Instead of embedding each chunk in isolation, you run the whole document (up to the model's limit) through a long-context embedding model to get contextual token vectors, then mean-pool the token vectors inside each chunk's span (Günther et al., 2024). Each chunk vector then reflects document context, which helps when chunks contain pronouns or references like "it" or "the above". The index and query side are unchanged: still one vector per chunk. It needs a long-context embedding model and access to token-level outputs, which many hosted APIs do not expose. Alternatives with similar goals are prepending section headers (free) and contextual retrieval (an LLM-written context per chunk, which also helps BM25 but costs an LLM call per chunk).

**Key points to mention:**
- It improves the dense leg only; BM25 does not benefit.
- Cost is roughly one long-document forward pass instead of many short ones.
- Measure recall@k with and without it on chunks that depend on context.
- Compare against cheap header prepending first.

**Deep dive:** [Late chunking (briefly)](../08-rag/04-chunking-strategies.md#late-chunking-briefly)

</details>

## Retrieval: BM25, dense and hybrid search

<details>
<summary><b>Q31. Compare BM25, dense retrieval, and hybrid search.</b></summary>

**Answer:** BM25 is lexical: it scores documents by exact term overlap, weighting rare terms higher (IDF), with saturating term frequency and length normalization, served from an inverted index. It excels at exact keywords and identifiers and is cheap and explainable, but misses synonyms and paraphrases. Dense retrieval embeds queries and documents with a bi-encoder and ranks by cosine or dot product; it captures meaning, paraphrases and even other languages, but is weak on exact IDs, numbers, rare jargon and out-of-domain text, and costs an embedding model plus vector memory. Hybrid runs both and fuses the results, usually with reciprocal rank fusion, covering both failure modes at the cost of two indexes and some tuning. A standard production pipeline is a hybrid first stage with 50-100 candidates per leg, RRF, then a cross-encoder reranker.

**Key points to mention:**
- Failure modes: BM25 scores 0 for "I forgot my login" against "reset password steps"; dense returns `ERR-1024` for an `ERR-1042` query.
- Choose by reading query logs: mostly IDs means BM25 plus exact fields, conversational means dense, mixed traffic means hybrid.
- Hybrid means two indexing paths that must stay in sync for inserts, deletes and permissions.
- Pitfall: claiming hybrid always wins; measure it on your queries.

**Deep dive:** [The big picture: lexical vs semantic retrieval](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#the-big-picture-lexical-vs-semantic-retrieval), [Hybrid search: combining both worlds](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#hybrid-search-combining-both-worlds)

</details>

<details>
<summary><b>Q32. When is keyword search better than semantic search?</b></summary>

**Answer:** When the query contains exact strings whose identity matters: error codes, SKUs, ticket IDs, function names, legal citations, people and product names, version numbers, and rare domain terms the embedding model never learned. Example: an embedding sees `ERR-1042` and `ERR-1024` as two nearly identical "error code" strings, while BM25 sees two different rare tokens and matches only the right one. Brand-new terms (a product launched yesterday) work in BM25 immediately, and users who paste from logs or forms produce keyword-style queries. BEIR (Thakur et al., 2021) showed BM25 is a strong zero-shot baseline across domains; modern embedders now beat it on average, but BM25 still wins or stays close on several datasets. Keyword search also wins on explainability, cost and latency, while semantic search wins when users paraphrase, ask long questions or use other languages.

**Key points to mention:**
- Measure it: tag identifier queries in the eval set and compare BM25 vs dense Recall@K and MRR on that slice.
- The best answer is often an exact-match keyword field for detected ID patterns plus hybrid for everything else.
- Query routing (a regex for ID patterns boosts BM25) gets the best of both.
- Say the mirror image out loud to show balance.

**Deep dive:** [Where BM25 shines and where it fails](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#where-bm25-shines-and-where-it-fails)

</details>

<details>
<summary><b>Q33. Why does BM25 work well for exact keywords and identifiers?</b></summary>

**Answer:** Because it matches exact tokens through an inverted index, and its IDF term gives rare tokens very large weight. Identifiers are rare by nature, so a single match dominates the score, and there is no learned model that could blur `E401` into `E410`. In the course's 4-document example, IDF("e401") $= \ln(1 + 3.5/1.5) = 1.204$ versus IDF("password") $= 0.357$, and the document containing "e401" scored 1.56 against 0.51 for the next. Term-frequency saturation ($k_1$) stops a document that repeats a common word many times from outscoring the one with the rare ID. The advantage disappears if the analyzer tokenizes `E-401` into `e` and `401`, so ID-like strings need custom tokenization or keyword sub-fields, with the same normalization at index and query time.

**Key points to mention:**
- BM25 formula: $\sum_t \text{IDF}(t) \cdot \frac{f(k_1+1)}{f + k_1(1 - b + b\,|d|/\text{avgdl})}$; defaults around $k_1 = 1.2$ to 1.5 and $b = 0.75$.
- An "identifier" that appears on every page gets low IDF and stops being decisive.
- Partial IDs (`88213` without `SKU-`) need a numeric sub-field or prefix matching.
- Analyzer changes require a full reindex: version the analyzer and test real IDs with the engine's analyze API.

**Deep dive:** [BM25 step by step](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#bm25-step-by-step)

</details>

<details>
<summary><b>Q34. What are the strengths of dense retrieval?</b></summary>

**Answer:** Semantic matching: it retrieves passages that mean the same thing as the query even with no shared words, covering synonyms, paraphrases, conversational questions and, with multilingual models, queries and documents in different languages. The mechanism is a bi-encoder trained contrastively on (query, relevant passage) pairs that maps both into a shared space where relevance becomes geometric closeness, for example "I forgot my login" retrieving "reset password steps" where BM25 scores 0. Documents are embedded offline, so query time is one embedding call plus an ANN search. It needs no hand-built synonym lists, can be fine-tuned to a domain, and extends to multimodal search (CLIP-style text-image retrieval). The costs are an embedding model at index and query time, vector memory (1 million chunks at 768 float32 dimensions is about 3 GB before index overhead) and a small recall loss from approximate search.

**Key points to mention:**
- It is less tolerant of typos than often claimed unless trained with typo augmentation.
- Volunteer the weaknesses: exact IDs and numbers, negation, out-of-domain jargon, opacity, re-embedding on model change.
- Measure Recall@K on the paraphrase slice and per language.
- Pin the model version and required prefixes.

**Deep dive:** [Dense retrieval](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#dense-retrieval)

</details>

<details>
<summary><b>Q35. What problems arise with domain-specific terminology?</b></summary>

**Answer:** General embedding models have not seen your jargon, codes or internal names, so subword tokenization breaks a term like "Falcon-X" or `E11.9` into fragments the model has only seen elsewhere, and its vector lands near generic or wrong concepts. Ambiguous acronyms get the common-web meaning ("PO" is a purchase order in procurement but "by mouth" in medicine), and near-identical codes look the same. On the BM25 side, analyzers can mangle identifiers, users write the same term in several forms, and vocabulary mismatch between how users ask and how documents are written remains. Fixes, cheapest first: custom analyzers and ID normalization, synonym and acronym dictionaries, query expansion, document expansion (doc2query), hybrid search, fine-tuning embeddings with domain pairs and hard negatives, and cross-encoder reranking.

**Key points to mention:**
- Fine-tuning requires re-embedding the corpus and can regress general queries; evaluate both slices.
- Jargon keeps changing: monitor zero-result and low-score queries to discover new vocabulary.
- Keep synonym and acronym files under version control and review them like code.
- Measure on a jargon-heavy slice of the eval set.

**Deep dive:** [Domain-specific terminology: the hardest case](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#domain-specific-terminology-the-hardest-case)

</details>

<details>
<summary><b>Q36. What is reciprocal rank fusion, and how does it combine rankings?</b></summary>

**Answer:** RRF merges several ranked lists using only ranks: each document's fused score is $\sum_r \frac{1}{k + \text{rank}_r(d)}$ summed over the lists it appears in, with $k$ conventionally 60 (Cormack, Clarke and Buettcher, 2009). Documents ranked well by several retrievers rise to the top, and because no raw scores are used, it needs no normalization between BM25 and cosine scales, which makes it a robust default. Worked example with $k = 60$: BM25 list [D2, D5, D1, D3] and dense list [D4, D2, D6, D1] give D2 $= 1/61 + 1/62 = 0.0325$ and D1 $= 1/63 + 1/64 = 0.0315$, both above D4 at $1/61 = 0.0164$. A large $k$ flattens the difference between adjacent ranks so consensus matters more; with $k = 1$, D4 would overtake D1. The limitation is that it discards margins: a dominant BM25 hit on an exact ID counts the same as a marginal rank 1.

**Key points to mention:**
- Weighted RRF ($w_r/(k + \text{rank})$) lets you trust one retriever more.
- Fuse 50-100 candidates per leg, not the top 5.
- Adding a third leg (SPLADE, a title-only field) needs no recalibration.
- Many search engines and vector databases implement RRF natively.

**Deep dive:** [Family 2: Reciprocal Rank Fusion (RRF)](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#family-2-reciprocal-rank-fusion-rrf)

</details>

<details>
<summary><b>Q37. How would you determine whether hybrid retrieval actually improves performance?</b></summary>

**Answer:** Build a labeled set of real queries with known relevant chunks, sampled from logs and stratified by type (identifier, keyword, natural language, multilingual), a few hundred as a start, with separate dev and test splits. Pool candidates from all systems before labeling so no system is favored. Run BM25, dense and hybrid on exactly the same index and queries and compare Recall@K at the depth the next stage consumes (Recall@50 before a reranker) and nDCG@K, overall and per query type. Check significance with per-query paired differences, a bootstrap 95% confidence interval or a Wilcoxon signed-rank test. Then weigh the gain against p95 latency, extra memory and operational burden, confirm end-to-end answer quality, and finish with an online A/B test.

**Key points to mention:**
- Classic trap: an overall gain hiding a regression on ID queries.
- Correct for multiple comparisons when sweeping many configurations.
- Keep the set as a retrieval regression test for embedding, analyzer, chunking or fusion changes.
- Tune fusion parameters on dev, report on test.

**Deep dive:** [Does hybrid actually help? Measuring it properly](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#does-hybrid-actually-help-measuring-it-properly)

</details>

<details>
<summary><b>Q38. What is SPLADE and when would you use it?</b></summary>

**Answer:** SPLADE (Formal, Piwowarski and Clinchant, 2021) is a learned sparse retriever: a BERT-style model outputs weights over the whole vocabulary, including expansion terms not present in the text, and the result is stored in an ordinary inverted index. It keeps exact-term matching and interpretability while reducing vocabulary mismatch, for example giving "change login credentials" some weight on "password" so a "password" query finds it. Use it when BM25 is your workhorse but vocabulary mismatch hurts, or as a third leg in a hybrid setup, which RRF makes easy. The costs are neural inference at both indexing and query time, larger postings lists than BM25, and possibly domain adaptation. Contrast it with ColBERT, which keeps one dense vector per token and costs much more storage.

**Key points to mention:**
- Sparse output means existing inverted-index infrastructure can serve it.
- Measure on the paraphrase slice where BM25 fails.
- It still needs re-encoding when the model changes, unlike BM25.
- Good answer to "how do you get semantic matching without a vector database?"

**Deep dive:** [Learned sparse retrieval: SPLADE](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#learned-sparse-retrieval-splade)

</details>

<details>
<summary><b>Q39. Users report that queries containing product SKUs return the wrong products. Your system is dense-only. How do you debug and fix it?</b></summary>

**Answer:** Confirm with a slice of SKU queries and their correct products, measure Recall@K on that slice, and inspect what is retrieved. Dense models embed similar-looking codes almost identically, so wrong neighboring SKUs are expected behavior, not a bug. Check tokenization on both paths: how the embedding tokenizer splits `SKU-88213`, and how a BM25 analyzer would. The fix is to add a lexical leg (BM25 with an analyzer that keeps SKUs as single normalized tokens, or an exact-match filter on a structured SKU field) and fuse, with routing that boosts the lexical leg when a SKU pattern is detected. Then re-evaluate all query types to make sure natural-language queries did not regress.

**Key points to mention:**
- Structured data belongs in structured fields: a keyword field enables exact filtering before semantic ranking.
- Normalize variants (`SKU-88213`, `sku 88213`, `88213`) at index and query time.
- Longer term: fine-tune embeddings with neighboring SKUs as hard negatives if codes appear inside natural-language queries.
- Watch for a reranker demoting the exact match (Q59).

**Deep dive:** [Where BM25 shines and where it fails](../08-rag/05-retrieval-bm25-dense-and-hybrid-search.md#where-bm25-shines-and-where-it-fails)

</details>

## Approximate nearest-neighbor search and vector databases

<details>
<summary><b>Q40. Explain approximate nearest-neighbor search, including HNSW and IVF.</b></summary>

**Answer:** Exact search compares the query with every vector: $N \cdot d$ multiply-adds per query, 7.68 billion for 10 million 768-dimensional vectors, reading about 30.7 GB, so it is memory-bandwidth bound. Approximate nearest-neighbor (ANN) search finds most of the true nearest vectors while examining a small fraction of the data, trading a little recall for orders-of-magnitude speedups. HNSW does it with a layered proximity graph: start at a sparse top layer, greedily walk toward the query, descend layer by layer, and finish with a beam search on the full bottom layer. IVF does it with clustering: k-means splits the data into nlist cells, and a query scans only the nprobe nearest cells. HNSW gives the best recall-latency trade-off in RAM but uses more memory, builds slowly and handles deletes awkwardly; IVF uses less memory, builds fast, deletes easily and pairs naturally with PQ, but needs training and retraining on drift.

**Key points to mention:**
- ANN quality is recall@k against exact search: always quote recall at a specific operating point.
- HNSW node levels are drawn with probability about $1/M$ per level, giving about $\log_M N$ layers.
- IVF misses happen when a true neighbor sits across a cell border.
- IVF work is about $n_{\text{list}} + N \cdot n_{\text{probe}}/n_{\text{list}}$ comparisons.

**Deep dive:** [HNSW: graph-based search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#hnsw-graph-based-search), [IVF: cluster-based search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#ivf-cluster-based-search)

</details>

<details>
<summary><b>Q41. What are HNSW, IVF, and product quantization?</b></summary>

**Answer:** HNSW and IVF are search structures that reduce how many vectors you compare against, through graph traversal and cluster probing respectively. Product quantization (PQ) reduces how big each vector is: it splits each vector into $m$ subvectors and replaces each with the ID of its nearest centroid in a 256-entry codebook, so with $m = 96$ and 8 bits a 768-dimensional float32 vector shrinks from 3,072 bytes to 96 bytes (32x), and 10 million vectors fit in 960 MB. Distances are computed with asymmetric distance computation: the query stays exact, a per-query $m \times 256$ lookup table is built, and each database vector costs $m$ table lookups instead of $d$ multiply-adds. They compose: IVF-PQ is the classic billion-scale index, and HNSW can sit over scalar- or PQ-compressed vectors. PQ loses accuracy and can flip rankings near the top, so fetch more candidates and re-rank them with full vectors.

**Key points to mention:**
- PQ bytes per vector $= m \cdot n_{\text{bits}}/8$.
- "Asymmetric" means only the database side carries quantization error.
- OPQ adds a learned rotation that balances information across subvectors.
- In IVF-PQ, codes encode the residual from the cell centroid.

**Deep dive:** [Product quantization: compressing vectors](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#product-quantization-compressing-vectors)

</details>

<details>
<summary><b>Q42. What are the trade-offs between search latency, recall, and memory?</b></summary>

**Answer:** You cannot maximize all three at once. Higher recall costs latency (larger efSearch or nprobe, more vectors scanned) or memory (more graph links, less compression), and lower memory via quantization costs recall unless you pay latency for re-ranking with full vectors; build time and update cost are a fourth axis. Runtime dials such as efSearch and nprobe trade recall for latency with no memory change, and recall saturates while latency keeps growing, so the last 1% of recall is the most expensive. For 10 million 768-dimensional vectors: float32 HNSW is about 33 GB, HNSW over int8 vectors about 10 GB, and IVF-PQ about 1 GB. The way to choose is to fix the recall target and latency SLO from product requirements, then pick the cheapest point on the recall-versus-QPS Pareto frontier.

**Key points to mention:**
- HNSW's M raises the recall ceiling and memory ($2M \times 4$ bytes per vector of links: 2.56 GB per 10 million at M = 32).
- efConstruction raises recall and build time, not memory.
- Compression ladder: int8 4x, PQ or binary 32x, with re-ranking to recover recall.
- A setting tuned at today's size silently loses recall as the index grows or accumulates tombstones, so re-measure on a schedule.

**Deep dive:** [The trade-off table and choosing an index](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#the-trade-off-table-and-choosing-an-index)

</details>

<details>
<summary><b>Q43. What does nprobe control in IVF?</b></summary>

**Answer:** nprobe is the number of IVF cells (clusters) searched per query: the query is compared with all nlist centroids, the nprobe nearest cells are scanned, and everything else is skipped. Higher nprobe means higher recall and roughly proportionally higher latency; with IVF-Flat, nprobe equal to nlist is exact search (IVF-PQ stays approximate because the codes are compressed). The fraction scanned is about $n_{\text{probe}}/n_{\text{list}}$: with 10 million vectors, nlist 4,096 and nprobe 16, you scan about 0.39% of the data (about 39,000 vectors plus 4,096 centroid comparisons), roughly 230x less work than brute force. You need more than one probe because neighbors of queries near cell borders live in adjacent cells. It is a query-time parameter, so it can be raised per request, for example for selective filters.

**Key points to mention:**
- Tuned jointly with nlist: finer cells need larger nprobe; total work is minimized near $n_{\text{list}} = \sqrt{N \cdot n_{\text{probe}}}$.
- Set it by sweeping 1, 2, 4, 8, ... against exact ground truth and taking the smallest value meeting the recall target within the p99 budget.
- Unbalanced cells after data drift make latency vary by query.
- Retrain the coarse quantizer when the data distribution shifts.

**Deep dive:** [What nprobe controls, with numbers](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#what-nprobe-controls-with-numbers)

</details>

<details>
<summary><b>Q44. How does HNSW's search configuration affect performance?</b></summary>

**Answer:** efSearch, the size of the candidate list kept during the bottom-layer beam search, is the main runtime knob: a larger efSearch visits more nodes, raises recall with diminishing returns and increases latency roughly linearly, with no memory cost. It must be at least k, and should exceed k when a reranker needs a deeper candidate list. M (links per node) and efConstruction (candidate list during insertion) are build-time settings that set the recall ceiling, memory and build time: higher M helps on high-dimensional or clustered data but increases memory linearly and per-hop cost, while efConstruction only costs build time. Typical starting ranges are M 16-64, efConstruction 100-500 and efSearch 50-500, tuned on a recall-versus-QPS sweep. As the index grows, the same efSearch yields lower recall, so monitor recall over time.

**Key points to mention:**
- efSearch can be set per query: raise it when a filter discards many nodes.
- Thread count and batching change measured QPS; benchmark the deployed access pattern.
- M cannot be changed without a rebuild.
- Tombstones from deletes also eat into effective efSearch.

**Deep dive:** [The three knobs: M, efConstruction, efSearch](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#the-three-knobs-m-efconstruction-efsearch)

</details>

<details>
<summary><b>Q45. When would you use exact search instead of approximate search?</b></summary>

**Answer:** When the collection, or the filtered subset a query actually searches, is small enough that brute force meets the latency budget: 100,000 vectors at 768 dimensions is 76.8 million multiply-adds, typically milliseconds with optimized kernels, with no build, no tuning and no recall loss. Also when recall must be perfect (legal, compliance, deduplication, safety matching), and when generating ground truth, since every ANN benchmark needs exact top-k for a query sample. Heavily filtered queries are a big case: a tenant with 5,000 chunks out of 10 million is best served by pre-filtering and brute force, while filtered graph search can degrade. Offline batch jobs (clustering, dedup) and tiny, rapidly changing collections such as per-session uploads also favor exact search. The decision rule: measure brute-force p99 on deployed hardware at current and projected sizes, and move to ANN only when exact search breaks the SLO.

**Key points to mention:**
- Many systems keep both paths and fall back to exact search for small or heavily filtered subsets automatically.
- A GPU matrix multiply is the simplest exact search for batched offline work.
- Exact search is also the debugging tool to separate index loss from embedding problems (Q48).
- Brute force cost grows linearly with N, so re-check as the corpus grows.

**Deep dive:** [When to use exact search](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#when-to-use-exact-search)

</details>

<details>
<summary><b>Q46. How would you scale retrieval to millions of vectors?</b></summary>

**Answer:** Start with the memory math: 1 million 768-dimensional float32 vectors is about 3 GB, which a single well-configured node with HNSW and a replica handles easily, so do not distribute prematurely. For 10 million vectors, float32 HNSW is about 33 GB; int8 scalar quantization cuts that to about 10 GB, and 100 million vectors is 307 GB in float32, 77 GB in int8 or about 10 GB as binary. As you grow, add quantization with full-precision rescoring, then disk-based indexes (DiskANN-style) or IVF-PQ, then sharding for data size and replication for throughput and availability. Partition by tenant so most queries hit one shard, and design filtering in from the start. Watch fan-out tail latency: with 10 shards each slow 1% of the time, about 9.6% of queries hit a slow shard, so use hedged requests, timeouts and fewer, larger shards.

**Key points to mention:**
- Replicas = target QPS / measured per-replica QPS, plus spares for failures and rolling rebuilds.
- Sharding: random balances load; semantic reduces fan-out but risks hotspots; tenant sharding helps isolation.
- At scale, near-duplicates and distractors grow, so hybrid search and reranking matter more than raw ANN speed.
- Measure recall@k against exact search with filters, and p95/p99 latency while ingestion is running.

**Deep dive:** [Going deeper: scaling to millions and billions of vectors](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#going-deeper-scaling-to-millions-and-billions-of-vectors), [Scaling to millions and billions of vectors](../08-rag/07-vector-databases-in-practice.md#scaling-to-millions-and-billions-of-vectors)

</details>

<details>
<summary><b>Q47. How would you benchmark retrieval at millions of vectors?</b></summary>

**Answer:** Use real corpus vectors and 1,000-10,000 held-out real queries (never database vectors as queries), compute exact ground truth once with GPU brute force, then sweep index parameters. Plot recall@k against QPS and p50/p95/p99 latency under realistic concurrency, filters and concurrent writes, and record memory and build time. Compare indexes at equal recall (for example recall@10 = 0.95), never at different recall levels, and run separate experiments for batched throughput and single-query latency. Warm up first and use an open-loop load generator to avoid coordinated omission. Choose the cheapest configuration that meets the recall target and the latency SLO, then confirm end-to-end retrieval and answer quality, since ANN recall only measures agreement with exact search.

**Key points to mention:**
- Track two recalls: ANN recall vs exact, and retrieval recall or nDCG vs relevance labels.
- Methodology follows ann-benchmarks (recall-vs-QPS Pareto curves) and big-ann-benchmarks (filtered and out-of-distribution tracks).
- Include rebuild time and behavior with deletes if production has them.
- Keep the suite as a regression test for model or library upgrades.

**Deep dive:** [Going deeper: benchmarking ANN properly](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#going-deeper-benchmarking-ann-properly)

</details>

<details>
<summary><b>Q48. ANN recall is 0.98 but users say retrieval is bad. What is going on?</b></summary>

**Answer:** ANN recall only measures agreement with exact search over the same embeddings. At 0.98 the index is nearly perfect, so the problem is upstream: the embedding model, chunking, parsing, query formulation, missing keyword matching for identifiers, or ranking. Confirm by running exact search on the failing queries: if exact search is also wrong, tuning efSearch or nprobe will not help. Then measure retrieval quality properly, with recall, hit rate or nDCG against labeled relevant chunks, and read the failures. Typical fixes are hybrid BM25 plus dense for identifiers, a better or domain-tuned embedding model, better chunking, a cross-encoder reranker and query rewriting.

**Key points to mention:**
- Two different "recalls": index fidelity vs relevance to labels; interviewers test whether you separate them.
- Check for metric mismatch (inner product on unnormalized vectors) and mixed model versions, which make exact and ANN results both wrong.
- Collect 20-50 failures and bucket them before changing anything.
- Low ANN recall would instead point at index parameters, filters or tombstones.

**Deep dive:** [Going deeper: production failure modes and debugging](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md#going-deeper-production-failure-modes-and-debugging)

</details>

<details>
<summary><b>Q49. What does a vector database give you that FAISS does not?</b></summary>

**Answer:** FAISS is an in-process ANN library: it maps vectors to the nearest row IDs, very fast, on CPU or GPU. A vector database adds the system around that: CRUD by stable IDs, durable storage with a write-ahead log, a payload store with filtered search, hybrid keyword plus vector queries, replication and sharding, access control, backups and snapshots, concurrent reads and writes, and monitoring. FAISS does have custom 64-bit IDs (`IndexIDMap`), search-time ID filtering through `IDSelector` and index serialization, but it lacks a metadata query language, a WAL, replication and access control, and some index types (HNSW) do not support removal at all. FAISS remains a great choice for offline batch jobs (dedup, clustering, ground truth), research, GPU-heavy workloads, and static corpora rebuilt on a schedule and loaded into a stateless service.

**Key points to mention:**
- Many vector databases use FAISS-like or hnswlib-like algorithms inside; differentiation is filtering, writes and distribution.
- If you use FAISS directly in production, you build all the surrounding system yourself.
- The real question is operational: who owns durability, deletes and filters.
- Mention when you would still pick FAISS to show judgment.

**Deep dive:** [From an ANN library to a vector database](../08-rag/07-vector-databases-in-practice.md#from-an-ann-library-to-a-vector-database)

</details>

<details>
<summary><b>Q50. pgvector or a dedicated vector database: how do you decide?</b></summary>

**Answer:** Choose pgvector if you already run Postgres, the corpus is up to roughly tens of millions of vectors, and you value transactions, joins with application data, row-level security and one operational system. Choose a dedicated vector database when vector search is the core workload at large scale, or when you need filter-aware HNSW with a planner, native multi-tenancy, quantization with rescoring, disk indexes or horizontal sharding out of the box. pgvector strengths include replacing a document's chunks in one ACID transaction, mature backups and replication, and SQL hybrid search with full-text. Its watch-outs: HNSW index scans post-filter by default (mitigate with `hnsw.iterative_scan` in 0.8.0+, partial indexes or partitioning), builds need `maintenance_work_mem`, vector search competes with OLTP traffic, and the `vector` type indexes only up to 2,000 dimensions (use `halfvec` for 3,072-dimensional embeddings).

**Key points to mention:**
- A new datastore has real costs: security review, backups, on-call, skills.
- Do not add one without a measured need.
- Use a read replica for vector queries, accepting replication lag.
- Either way, validate filtered recall and p99 latency at your target size.

**Deep dive:** [Choosing a vector store: a decision guide](../08-rag/07-vector-databases-in-practice.md#choosing-a-vector-store-a-decision-guide)

</details>

<details>
<summary><b>Q51. A filter on a small tenant returns zero results, but the tenant has relevant documents. What is happening and how do you fix it?</b></summary>

**Answer:** Almost certainly post-filtering: the ANN index returns the top candidates from the whole corpus, they all belong to bigger tenants, and the filter then removes every one of them. The math is brutal: with selectivity $s$ and $k'$ candidates, expected survivors are $k' \cdot s$ and $P(\text{zero}) = (1-s)^{k'}$, so at $s = 0.001$ and $k' = 100$ the chance of an empty result is about 0.905. Diagnose by running exact search restricted to the tenant: if results exist, the index path is the culprit, and logging candidate counts before and after filtering confirms it. Fixes, from quickest to best: raise the candidate count or enable iterative scanning, use a filter-aware engine with a cardinality-based planner, brute-force the filtered subset for small tenants, or partition by tenant so no selective filter is needed at all.

**Key points to mention:**
- Correlation makes it worse: when the topic is dominated by other tenants' data, the global top-k is all theirs.
- Add a monitor on "results returned < k" so it never fails silently again.
- In pgvector: `hnsw.iterative_scan`, partial indexes for large tenants, or table partitioning.
- For permissions, plain post-filtering is also a security problem (Q114).

**Deep dive:** [Metadata filtering: pre, post and in-graph](../08-rag/07-vector-databases-in-practice.md#metadata-filtering-pre-post-and-in-graph)

</details>

<details>
<summary><b>Q52. How do updates and deletes work in an HNSW-based vector database, and what are the consequences?</b></summary>

**Answer:** Physically removing a node from an HNSW graph would break its neighbors' links and could leave regions unreachable, so systems mark deleted nodes with a tombstone: they stay in the graph for navigation but are never returned. An update is usually delete plus insert, and background compaction later rewrites segments without dead records and rebuilds their graphs. Consequences: tombstones waste search effort and reduce recall until compaction (a candidate list of 10 with 30% tombstoned yields only 7 live results), compaction consumes CPU, memory and I/O and can cause latency spikes, and deleted data still physically exists until compaction and backup expiry, which matters for GDPR erasure. IVF is simpler: deletes remove entries from an inverted list and inserts go to the nearest centroid, but centroids go stale as data drifts.

**Key points to mention:**
- FAISS `IndexHNSWFlat` cannot remove at all; hnswlib supports `mark_deleted` and slot reuse.
- Postgres differs: MVCC dead tuples are cleaned by VACUUM.
- Monitor deleted ratio and recall drift; schedule rebuilds after heavy churn.
- When a document shrinks, delete by `doc_id` to remove orphan chunks.

**Deep dive:** [The write path: upserts, deletes, tombstones and compaction](../08-rag/07-vector-databases-in-practice.md#the-write-path-upserts-deletes-tombstones-and-compaction)

</details>

## Reranking and query transformation

<details>
<summary><b>Q53. Explain reranking and query rewriting.</b></summary>

**Answer:** Both sit around first-stage retrieval. Reranking happens after retrieval: a fast retriever returns, say, 50-100 candidates optimized for recall, then a more accurate but more expensive model, usually a cross-encoder, rescores each candidate against the query so the best 3-10 go to the LLM. Query rewriting happens before retrieval: it turns the user's message into a better search query, for example making a chat follow-up like "and how long does it take?" standalone, generating paraphrases, or expanding terms. Reranking fixes ordering (precision at the top) while rewriting fixes matching (recall and search intent), so a common pattern is to rewrite or expand to widen the pool, then rerank against the user's original intent. Both add latency (tens to hundreds of milliseconds for reranking, an LLM call for rewriting), so each must be justified by measured gains.

**Key points to mention:**
- A reranker cannot recover documents retrieval missed; stage-1 Recall@K is its ceiling.
- A rewriter can inject hallucinated terms; keep the original query as a fallback leg.
- Rewriting family: conversational rewriting, multi-query, HyDE, expansion, step-back, routing, decomposition.
- Measure the reranker by nDCG@n at the n the LLM sees; the rewriter by Recall@n vs the raw query and a gold standalone query.

**Deep dive:** [Two-stage retrieval: recall first, precision second](../08-rag/08-reranking-and-query-transformation.md#two-stage-retrieval-recall-first-precision-second), [Query transformation: fixing the question before searching](../08-rag/08-reranking-and-query-transformation.md#query-transformation-fixing-the-question-before-searching)

</details>

<details>
<summary><b>Q54. How does a cross-encoder differ from a bi-encoder?</b></summary>

**Answer:** A bi-encoder encodes the query and each document separately into one vector each and scores them with a dot product or cosine, so document vectors can be precomputed and searched with an ANN index over millions of items in milliseconds. A cross-encoder concatenates query and document into one input ("[CLS] query [SEP] document [SEP]"), runs a transformer with full attention across both, and outputs a single relevance score. Because every query token attends to every document token in every layer, it is more accurate, but nothing can be precomputed, so it is only used to rerank a short candidate list. Bi-encoders are usually trained contrastively with in-batch negatives, while cross-encoders are trained as relevance classifiers or with ranking losses (Nogueira and Cho, 2019). Late interaction (ColBERT) sits between them.

**Key points to mention:**
- In a bi-encoder, the document vector is a query-independent bottleneck; interaction happens only at the final dot product.
- Cross-encoder failure modes: truncation at max input length (often 512 tokens), domain shift, scores not calibrated across queries.
- Cross-encoders can also label or filter hard negatives and distill into bi-encoders.
- Using a cross-encoder for first-stage retrieval would need one forward pass per corpus document per query.

**Deep dive:** [Bi-encoder vs cross-encoder side by side](../08-rag/08-reranking-and-query-transformation.md#bi-encoder-vs-cross-encoder-side-by-side)

</details>

<details>
<summary><b>Q55. Why is a cross-encoder generally more expensive?</b></summary>

**Answer:** Its input is the (query, document) pair, so it must run a full transformer forward pass for every candidate at query time, and nothing can be computed before the query arrives. A bi-encoder runs the encoder once on the short query and compares against document vectors computed offline. Using FLOPs $\approx 2 \times P \times T$: a 22M-parameter reranker on 320-token pairs costs about 14 GFLOPs per pair, so 100 candidates is about 1.4 TFLOPs, roughly 70 ms on a GPU at an illustrative 20 TFLOP/s but about 1.4 s on a CPU at 1 TFLOP/s. The bi-encoder's query encoding is about 0.88 GFLOPs, 1,600 times less than reranking 100 candidates. Cost scales linearly with the number of candidates, with model size and with sequence length (plus attention's quadratic term).

**Key points to mention:**
- Capacity: 50 QPS with K = 100 is 5,000 pair evaluations per second, several GPUs in this example.
- Mitigations: smaller K, distilled or smaller models, INT8, batching, caching repeated pairs, conditional reranking.
- Batches pad to the longest pair: cap chunk length or sort by length to keep p95 stable.
- A better top-4 can cut LLM prompt tokens and partly pay for the reranker.

**Deep dive:** [Why a cross-encoder is more accurate and more expensive](../08-rag/08-reranking-and-query-transformation.md#why-a-cross-encoder-is-more-accurate-and-more-expensive)

</details>

<details>
<summary><b>Q56. When should you rerank results, and when should you use a cross-encoder reranker?</b></summary>

**Answer:** Rerank when the relevant chunk is usually retrieved but not ranked at the very top, which shows up as a big gap between Recall@50 and Recall@5 on a labeled set, and when the LLM only reads a few chunks. It pays off most for natural-language, specific or nuanced queries, corpora with many near-duplicates, and when fusing several retrievers whose scores are not comparable. Choose the rerank depth K where stage-1 Recall@K plateaus, typically 50-100 for a cross-encoder (20-30 for an LLM reranker), and pass 3-10 chunks to the LLM. Skip it or make it conditional when the latency budget is very tight, when first-stage precision is already near the ceiling, for exact identifier lookups, or when the real problem is stage-1 recall. Its cost can be partly offset downstream: cutting context from 10 chunks to 4 saves about 1,800 prompt tokens at 300 tokens per chunk.

**Key points to mention:**
- Signals it is not working: low rank-shift rate, no answer-quality gain, regressions on segments such as non-English queries with an English-only reranker.
- Alternatives under tight latency: ColBERT, a smaller distilled reranker, reranking only some routes.
- Reranker scores are better than raw cosines for "no good answer" thresholds.
- Always report stage-1 recall next to reranked metrics.

**Deep dive:** [When to rerank and how many candidates](../08-rag/08-reranking-and-query-transformation.md#when-to-rerank-and-how-many-candidates)

</details>

<details>
<summary><b>Q57. What is multi-query retrieval?</b></summary>

**Answer:** An LLM generates several paraphrases or alternative formulations of the user's query, you run retrieval for each one plus the original, and fuse the result lists, typically with reciprocal rank fusion, so documents found by several variants rise to the top. It improves recall for vague, short or ambiguous queries and for vocabulary mismatch, at the cost of an extra LLM call and N times the retrieval load. Example: a document at ranks 1, 2 and 1 across three variants scores $1/61 + 1/62 + 1/61 \approx 0.0489$, beating one found once at rank 1 ($\approx 0.0164$). Three to five variants is a common range with diminishing returns beyond that, and retrievals should run in parallel so wall-clock cost is roughly one retrieval plus the generation. Always include the original query and rerank the fused pool against it to filter noise from bad paraphrases.

**Key points to mention:**
- Related techniques: RAG-Fusion (multi-query plus RRF), HyDE (generate an answer), decomposition (generate different sub-questions).
- On precise queries, paraphrases add little and can drift; route those around multi-query.
- Measure Recall@K of the fused pool vs original-only, then final nDCG@n and added p95 latency.
- It is a recall tool, so evaluate it as one.

**Deep dive:** [Multi-query retrieval](../08-rag/08-reranking-and-query-transformation.md#multi-query-retrieval)

</details>

<details>
<summary><b>Q58. When does query expansion hurt retrieval?</b></summary>

**Answer:** When the added terms change what the query is about or dilute what made it precise. Typical cases: topic drift, where expanded or pseudo-relevance-feedback terms pull results toward a related but different topic (especially if the initial results were already wrong); diluting exact identifiers; already specific long queries; strong retrievers that do not need help; and hallucinated or wrongly disambiguated terms from an LLM. Example: "error E401 on vpn client" expanded with "authentication failure, login problem, access denied" makes BM25 match many generic chunks, and the decisive rare token E401 carries less of the score. Research on LLM-based expansion (Weller et al., 2024) found it tends to help weaker retrievers and can hurt stronger ones. Mitigations: boost original terms over expansion terms, never expand identifier-like tokens, expand only the BM25 leg, expand conditionally on low first-pass scores, and rerank against the original query.

**Key points to mention:**
- Pseudo-relevance feedback amplifies early errors.
- Evaluate per segment: expansion can help short queries and hurt ID queries at the same time.
- Every LLM expansion adds latency on the critical path.
- Keep the raw query as one fused leg as a safety net.

**Deep dive:** [Query expansion and when it hurts](../08-rag/08-reranking-and-query-transformation.md#query-expansion-and-when-it-hurts)

</details>

<details>
<summary><b>Q59. How would you combine reranking with hybrid retrieval?</b></summary>

**Answer:** Run BM25 and dense retrieval in parallel with permission filters applied inside both queries, take the top 25-100 from each, fuse with RRF and deduplicate, then send the fused candidates (about 50-100) to a cross-encoder that scores each (query, chunk) pair against the original or standalone-rewritten query. Keep the top 3-8 by reranker score, optionally with a calibrated cutoff, and pass them to the LLM. Hybrid maximizes recall (BM25 for identifiers, dense for paraphrases), RRF merges without score calibration, and the reranker gives one consistent relevance judgment so the final order no longer depends on mixing BM25 and cosine scales; strong candidates often have nearly tied RRF scores (0.0323 vs 0.0320) that the reranker separates. Include title or section metadata in the reranker input, respect its max length, and fall back to the fused order on timeout.

**Key points to mention:**
- Ablation on one labeled set: BM25 only, dense only, hybrid, hybrid plus rerank, to see whether gains come from recall or ordering.
- Failure mode: the reranker demotes an exact-identifier match that BM25 ranked first; route ID queries or blend in a small first-stage prior.
- Tune per-leg depth, K and n with a sweep watching Recall@K, nDCG@n and p95 latency.
- Cache frequent (query, chunk) pairs.

**Deep dive:** [Combining reranking with hybrid retrieval](../08-rag/08-reranking-and-query-transformation.md#combining-reranking-with-hybrid-retrieval)

</details>

<details>
<summary><b>Q60. What is late interaction (ColBERT), and when would you choose it over a cross-encoder?</b></summary>

**Answer:** ColBERT encodes queries and documents separately but keeps one vector per token, and scores relevance with MaxSim: for each query token, take its best dot product with any document token, then sum, $S(q, d) = \sum_i \max_j \mathbf{q}_i \cdot \mathbf{d}_j$. Document token vectors are precomputed, so it is much cheaper at query time than a cross-encoder while capturing token-level matches better than a single-vector bi-encoder. Choose it when you need better-than-bi-encoder quality over large candidate sets, or even as a first-stage retriever, and can afford the storage. That storage is the catch: a 300-token chunk at 128 dimensions in FP16 is 76,800 bytes versus 1,536 bytes for one 768-dimensional FP16 vector, about 50 times more. Accuracy usually sits between bi-encoders and cross-encoders, so a cross-encoder on top can still help.

**Key points to mention:**
- ColBERTv2 residual compression and the PLAID engine reduce the cost.
- Infrastructure is more specialized than a standard vector index, though support is growing.
- Good middle ground when cross-encoder latency is too high.
- Measure quality and storage against bi-encoder plus reranker.

**Deep dive:** [Late interaction: ColBERT as a middle ground](../08-rag/08-reranking-and-query-transformation.md#late-interaction-colbert-as-a-middle-ground)

</details>

<details>
<summary><b>Q61. How would you evaluate whether a reranker justifies its latency?</b></summary>

**Answer:** Compare the pipeline with and without the reranker on four axes. Retrieval quality: nDCG@n, MRR and Recall@n where n is what the LLM actually sees, reported next to stage-1 Recall@K as the ceiling. Answer quality: correctness, faithfulness and citation accuracy end to end. Cost: added p95 and p99 latency (not the mean, since long chunks dominate the tail) and dollars per query, minus LLM token savings from passing fewer chunks. And per-segment results by query type and language. Use paired bootstrap or permutation tests over queries, sweep K and model size to find the best quality-latency point, then confirm with an online A/B or canary test using user signals and latency guardrails. Ship if important segments gain significantly within the SLO; otherwise try a smaller model, smaller K, quantization or conditional reranking.

**Key points to mention:**
- Example: relevant chunk moves from rank 3 to 1, nDCG@5 goes from 0.5 to 1.0; a query whose chunk is outside the top 50 stays at 0.
- Count token savings: 10 to 4 chunks at 300 tokens saves 1,800 prompt tokens per query.
- An offline nDCG@10 gain can be invisible if the LLM reads only the top 3-5.
- Online signals: thumbs, escalations, reformulations.

**Deep dive:** [Evaluating whether a reranker justifies its latency](../08-rag/08-reranking-and-query-transformation.md#evaluating-whether-a-reranker-justifies-its-latency)

</details>

<details>
<summary><b>Q62. What is HyDE and when does it fail?</b></summary>

**Answer:** HyDE (Hypothetical Document Embeddings, Gao et al., 2022) asks an LLM to write a hypothetical answer document for the query, embeds that document (often averaged with the query embedding and other generations), and uses it for dense search. The fake answer is written in document style, so it lands near real answer passages even when its details are wrong, which helps in zero-shot settings, with weak embedders, and when questions and documents look very different. It fails when the LLM does not know the domain, for identifier or exact-match queries, when hallucinated specifics bias retrieval toward the wrong passages, and when the generation latency (a paragraph of tokens before retrieval can even start) is unacceptable. Modern embedders trained on question-passage pairs already handle much of the asymmetry, so measure the gain before adopting it.

**Key points to mention:**
- Keep BM25 on the raw query in a hybrid setup so identifiers are not lost.
- Generate a few hypotheses at moderate temperature and average them with the query to dampen one bad generation.
- It sits on the critical path; a short rewrite is cheaper.
- Evaluate per segment like any query transformation.

**Deep dive:** [HyDE: hypothetical document embeddings](../08-rag/08-reranking-and-query-transformation.md#hyde-hypothetical-document-embeddings)

</details>

## Advanced RAG: filters, parent-child, contextual retrieval and multi-hop

<details>
<summary><b>Q63. Explain metadata filtering, parent-child retrieval, and contextual retrieval.</b></summary>

**Answer:** They fix three different retrieval problems. Metadata filtering restricts the search to chunks whose structured attributes match (version, date, department, language, permissions), because embeddings capture topic but not applicability. Parent-child retrieval embeds small child chunks for precise matching but sends the larger parent section to the LLM, so the answer has full context. Contextual retrieval prepends a short LLM-generated description of where a chunk sits in its document before embedding and BM25 indexing, so chunks that rely on implicit context ("it grew 3%") become findable. Put differently, filtering acts on the candidate set, parent-child on the context unit the LLM sees, and contextual retrieval on the representation that gets indexed. They compose: filter to v3.2, search contextualized children, expand to parents, rerank, generate.

**Key points to mention:**
- Costs differ: filtering needs clean metadata; parent-child costs prompt tokens; contextual retrieval costs an LLM call per chunk at ingestion.
- Measure each on its own slice: wrong-version rate, whether the needed fact reaches the context, recall@k before and after contextualization.
- Adopt each only when its slice improves.
- Name which problem each pattern fixes, not just what it is.

**Deep dive:** [The map: which problem each pattern fixes](../08-rag/09-advanced-rag-patterns.md#the-map-which-problem-each-pattern-fixes)

</details>

<details>
<summary><b>Q64. When are metadata filters useful?</b></summary>

**Answer:** Whenever relevance depends on an attribute that similarity cannot see: product version, time range, department, document type, language, tenant and permissions. The typical signal is that retrieved chunks are on the right topic but from the wrong version, year or audience: a 2019 and a 2026 version of the same policy embed almost identically, so a version or date filter is the only reliable way to pick the right one. Security filters (tenant and ACL) are mandatory and must be derived server-side from the authenticated user, never from the LLM or the query text. An LLM can auto-extract filters from natural language ("Q3 2025 EMEA sales"), but validate values against allowed lists, log them, and fall back to unfiltered search with disclosure if the filter returns nothing. Implementation matters: selective filters done as post-filtering can return zero results (Q51), so use pre-filtering or filter-aware search.

**Key points to mention:**
- Hard filters only for rules that must hold (security, explicit version); preferences like "recent" work better as a soft boost.
- Dirty metadata silently hides documents: validate at ingestion and monitor null filter fields.
- Filters need payload indexes, or every filtered query scans payloads.
- Track zero-result and fewer-than-k rates and extraction accuracy.

**Deep dive:** [Metadata filtering](../08-rag/09-advanced-rag-patterns.md#metadata-filtering), [Metadata filtering: pre, post and in-graph](../08-rag/07-vector-databases-in-practice.md#metadata-filtering-pre-post-and-in-graph)

</details>

<details>
<summary><b>Q65. How can small child chunks be retrieved while returning larger parent context?</b></summary>

**Answer:** Split each document into parents (sections of around 1,000 tokens) and children (sentences or 100-200 tokens) that store a `parent_id`, and embed and index only the children. At query time, retrieve the top-k children, map them to parent IDs, deduplicate while keeping the best child order, fetch the parents from a document store, fit them into the token budget and send them to the LLM. Context size is the sum over unique parents: with 128-token children, 1,024-token parents and 8 retrieved children from 3 parents, that is 3,072 tokens instead of 1,024 for children alone. Auto-merging is a smarter middle ground: swap to the parent only when a threshold fraction of its children are retrieved (for example 5 of 8 = 0.625 over a 0.5 threshold). For reranking, either rerank children and aggregate by max per parent, or rerank parents if they fit the reranker's input limit.

**Key points to mention:**
- Pitfalls: parents that are too large (token cost, lost in the middle) and stale children after a parent changes (delete by `parent_id`).
- Measure whether the needed fact reaches the final context, not just whether a child matched.
- Sentence-window retrieval is a lighter variant (expand to neighboring sentences).
- Compare tokens per query against a plain-chunk baseline.

**Deep dive:** [Parent-child (small-to-big) retrieval](../08-rag/09-advanced-rag-patterns.md#parent-child-small-to-big-retrieval)

</details>

<details>
<summary><b>Q66. What are the costs of adding contextual information to chunks?</b></summary>

**Answer:** Ingestion cost: one LLM call per chunk, each reading the whole document, so cost grows roughly with document length squared. Illustrative: an 8,000-token document with 40 chunks and a 100-token instruction needs $40 \times 8{,}300 = 332{,}000$ input tokens, 41.5 times the document size; with prompt caching of the document prefix at an illustrative 10% price, that drops to about 51,200 token-equivalents, roughly 6.5 times cheaper. Storage cost: each chunk's indexed text grows by the context length (for example 200 to 280 tokens, about 40% more), and embedding input grows too. Maintenance cost: when a document, the prompt or the model changes, contexts must be regenerated, which can be a full backfill. Quality risk: the generated context can be wrong, so spot-check it and store it separately from the original text so citations still point to source spans.

**Key points to mention:**
- Process a document's chunks consecutively with the document as the exact cached prefix.
- Key the context cache on doc version, chunk text, prompt version and model.
- Cheaper alternatives first: prepend heading path and metadata (free), late chunking (dense only).
- Measure recall@k with and without contexts on your own eval set; published gains vary a lot.

**Deep dive:** [Contextual retrieval](../08-rag/09-advanced-rag-patterns.md#contextual-retrieval)

</details>

<details>
<summary><b>Q67. How would you handle tables and cross-references?</b></summary>

**Answer:** Preserve structure at ingestion, then use it at query time. For tables, extract real grids instead of flattening them into a word stream, keep small tables whole with caption and headers and embed a short description, and for large tables create row-group or row-level chunks that repeat the caption and header row (with a 400-token chunk, 20-token header and 15-token rows, about 25 rows fit per chunk). Route numeric or aggregate questions to SQL or pandas over the extracted table, with the answer citing the table ID and row. For cross-references, detect "see Section 4.2", "Appendix B" and footnote markers, resolve them to element IDs at ingestion, inline short targets such as footnotes and definitions, and for long targets store links and expand one hop at query time with a cap. Otherwise retrieval finds section 4.2 and the LLM answers "see Appendix B".

**Key points to mention:**
- Trade-off: inlining adds tokens and dilutes the embedding; link expansion adds a retrieval hop and latency.
- Forward-fill merged cells and stitch tables split across pages.
- Cite the target section, not the referring one.
- Add eval questions that need a table cell or a referenced target, and count unresolved references as a quality signal.

**Deep dive:** [Tables and cross-references at query time](../08-rag/09-advanced-rag-patterns.md#tables-and-cross-references-at-query-time), [Handling cross-references, footnotes and appendices](../08-rag/03-document-ingestion-and-parsing.md#handling-cross-references-footnotes-and-appendices)

</details>

<details>
<summary><b>Q68. How would you preserve provenance through every transformation?</b></summary>

**Answer:** Give every source a stable `doc_id` and version, give every chunk a deterministic ID and character offsets into the parsed original, and make every derived artifact (generated contexts, summaries, propositions, graph edges, table rows) store the source spans it came from. Offsets always point to the original, never to transformed text, and generated text is never cited as evidence: you cite the spans behind it. Carry the IDs into the prompt as citation labels, map them back after generation, validate that each citation was in context and supports its claim, and log the index version, document versions, chunk IDs and prompt version per answer so any answer can be reproduced. Deterministic IDs (path or content hash) also make updates idempotent and keep citation history stable.

**Key points to mention:**
- Include page and bounding box for PDFs, row and column for tables, timestamps for media.
- Measure citation validity rate (cited ID was in context and offsets resolve) and citation precision as release metrics.
- Lineage is also what lets you delete or re-permission derived artifacts later.
- Random IDs per ingestion run break both idempotency and audits.

**Deep dive:** [Preserving provenance end to end](../08-rag/09-advanced-rag-patterns.md#preserving-provenance-end-to-end)

</details>

<details>
<summary><b>Q69. How do you handle multi-hop or complex questions in RAG?</b></summary>

**Answer:** First classify the question: single-hop, bridge, comparison, aggregation, temporal or global. Single retrieval fails on bridge questions because the bridge entity is not in the question: for "Who leads the team that owns billing-api?", the team name ("Payments") never appears in the query. For multi-hop questions, decompose into sub-questions with dependencies, retrieve for each (in parallel when independent, in sequence when a later query needs an earlier answer, iteratively when you cannot plan ahead), check that every required fact has supporting and consistent evidence, then answer, answer partially, clarify or abstain. Known methods include Least-to-Most prompting, Self-Ask and IRCoT (interleaved retrieval and reasoning). Cap the number of steps and total time, and parallelize independent hops to control latency.

**Key points to mention:**
- Route by complexity so simple questions skip the expensive path.
- Evaluate per hop (did hop 1 find the right bridge?) and on known-unanswerable questions.
- Validate each hop's answer before substituting it into the next query.
- Track false-join rate (Q73) separately from accuracy.

**Deep dive:** [Multi-hop and complex questions](../08-rag/09-advanced-rag-patterns.md#multi-hop-and-complex-questions)

</details>

<details>
<summary><b>Q70. What is query decomposition?</b></summary>

**Answer:** Using an LLM (or rules) to split a complex question into simpler sub-questions that can each be answered by one retrieval, together with an explicit plan of how they depend on each other and how to combine the answers. "Who leads the team that owns billing-api?" becomes "Which team owns billing-api?" followed by "Who leads {that team}?". Ask for structured output with sub-question IDs, text, `depends_on` and a combination step, and use parallel decomposition for comparisons ("v2.0 vs v3.2 retention") and sequential decomposition for bridge questions. It differs from query rewriting and multi-query, which produce alternative phrasings of the same question, while decomposition produces different questions. Failure modes are over-decomposing simple questions (latency and drift), wrong decompositions and errors propagating from hop 1.

**Key points to mention:**
- Mitigate with complexity routing, per-hop validation and a cap on sub-questions.
- Measure decomposition quality against hand-labeled sub-questions and per-hop recall.
- Track the share of simple questions needlessly decomposed (wasted latency).
- Decomposition doubles as the checklist for fact coverage (Q72).

**Deep dive:** [Query decomposition](../08-rag/09-advanced-rag-patterns.md#query-decomposition)

</details>

<details>
<summary><b>Q71. How would you retrieve information from multiple sources?</b></summary>

**Answer:** Route each sub-question to the relevant sources using source descriptions, then fan out in parallel with per-source timeouts, querying each source in its own language: embeddings, keywords, SQL or APIs. Enforce each source's permissions with the end user's identity, merge with rank-based fusion such as RRF or a single reranker over the union (scores from different engines are not comparable), deduplicate, and prefer the authoritative source for each fact type, for example the HR directory over a newsletter for "who leads a team". With RRF and $k = 60$, a chunk ranked 1 and 3 in two sources scores about 0.0323, beating one ranked 2 in a single source at about 0.0161. On partial failure, continue without the timed-out source and tell the user which source was unavailable. Structured sources (warehouse, CMDB) are often better than text for aggregations and current-state facts.

**Key points to mention:**
- Measure router accuracy (were the right sources chosen?).
- Track per-source latency, timeout rate and answer quality when one source is down.
- No all-powerful service account: each source checks the user's own permissions.
- Authority rules per fact type prevent mixing stale and current facts.

**Deep dive:** [Retrieving from multiple sources](../08-rag/09-advanced-rag-patterns.md#retrieving-from-multiple-sources)

</details>

<details>
<summary><b>Q72. How would you verify that all required facts are present?</b></summary>

**Answer:** Turn the decomposition into a checklist of required facts, and require each one to have an answer plus at least one evidence passage that supports it, checked claim by claim with an NLI model or an LLM judge rather than "some document mentions the topic". Compute coverage as supported facts over required facts: for owner, lead and runbook with no source for the lead, coverage is $2/3 \approx 0.67$. Only full coverage permits a full answer; otherwise issue a targeted retrieval for each missing fact (with a cap) and then answer partially or abstain. After generation, run a groundedness check so every sentence of the answer maps to evidence, removing or flagging unsupported sentences. To test it, build questions where you deliberately removed one required document from the corpus: the system should notice the gap instead of filling it from memory.

**Key points to mention:**
- Coverage is a per-question metric you can log and alert on.
- Iterative retrieval for missing facts should be bounded in steps and time.
- A fluent answer with a missing fact is the most dangerous failure.
- Pair with the partial-answer policy in Q74.

**Deep dive:** [Verifying evidence and deciding how to respond](../08-rag/09-advanced-rag-patterns.md#verifying-evidence-and-deciding-how-to-respond)

</details>

<details>
<summary><b>Q73. How would you avoid joining unrelated evidence?</b></summary>

**Answer:** Before combining facts, check that they are about the same entity (by canonical ID, not just by name), valid at the same time or at the time the question asks about, from the same version and scope, and in compatible units. Example false join: "Payments owns billing-api" (2026) plus "Daniel Kim leads Payments" (2024) gives a fluent, cited, wrong answer when the current lead is in the HR directory. Prevent it with entity linking and temporal metadata at ingestion, authoritative sources per fact type, freshness limits per fact type (a team lead goes stale in about a year, a founding date never does), and by requiring hop 2's evidence to mention the exact bridge entity from hop 1. Interpret time words correctly: "currently" means as of today, "when X happened" means as of X's date. When facts conflict, explain the conflict by version or date instead of choosing silently.

**Key points to mention:**
- Build a trap set with stale facts and same-name entities (two "Payments" teams).
- Track false-join rate as its own metric; overall accuracy hides it.
- Same-name entities across tenants or departments are a classic source of false joins.
- Citations do not protect you: each fact is cited, the join is what is wrong.

**Deep dive:** [Avoiding joining unrelated evidence](../08-rag/09-advanced-rag-patterns.md#avoiding-joining-unrelated-evidence)

</details>

<details>
<summary><b>Q74. When should the system answer partially, ask a clarifying question, or abstain?</b></summary>

**Answer:** Answer fully when all required facts are supported and consistent. Answer partially when some facts are supported and useful on their own, clearly stating what is missing or uncertain, and never fill the gaps silently from model memory. Ask a clarifying question when the question has interpretations that lead to different answers and you cannot cheaply cover them all; if you can cover them briefly, prefer that ("Basic: 30 days; Pro and Enterprise: 90 days"). Abstain when there is no supporting evidence, when authoritative sources conflict unresolvably, or when stakes are high and confidence is low. Thresholds depend on domain risk and should be stricter for legal, medical, finance and security use cases.

**Key points to mention:**
- Measure the policy: accuracy of answered questions, abstention rate on answerable questions, correct abstention on unanswerable ones.
- Tune so the system is neither reckless nor useless.
- Partial answers should label caveats such as staleness or a missing source.
- Retrieval-confidence gating (low reranker score means abstain) is a simple first implementation.

**Deep dive:** [Answer partially, ask a clarifying question, or abstain](../08-rag/09-advanced-rag-patterns.md#answer-partially-ask-a-clarifying-question-or-abstain)

</details>

<details>
<summary><b>Q75. What is GraphRAG, and when is it worth its cost?</b></summary>

**Answer:** GraphRAG builds a knowledge graph from the corpus with an LLM (extracting entities and relationships per chunk and resolving duplicate entities), detects communities of related entities (Leiden), and writes summaries for them. It answers relationship questions by traversing the graph (local search) and corpus-wide questions by map-reducing over community summaries (global search). The reference paper (Edge et al., 2024) showed wins on global "sensemaking" questions judged by an LLM on comprehensiveness and diversity, not on factoid accuracy; for single-fact lookups plain hybrid RAG is usually as good and far cheaper. Costs include LLM extraction over every chunk, community summarization, hard incremental updates, expensive global queries and entity-resolution errors. It is worth it only when relationship or global questions are a real, measured share of traffic and vector RAG fails on them.

**Key points to mention:**
- Cheaper alternatives: an existing structured graph (service catalog, org chart, CMDB), a summary index, or SQL over extracted entities.
- Community summaries mix ACLs, a common security trap (see Q113).
- Many teams rebuild communities on a schedule rather than incrementally.
- Evaluate on a labeled slice of relationship and global questions.

**Deep dive:** [GraphRAG and knowledge graphs](../08-rag/09-advanced-rag-patterns.md#graphrag-and-knowledge-graphs)

</details>

<details>
<summary><b>Q76. What is agentic RAG, and how do Self-RAG and corrective RAG relate to it?</b></summary>

**Answer:** Agentic RAG turns the fixed pipeline into a loop: retrieval becomes a tool the LLM can call zero or more times, choosing the source, query and filters, grading what comes back and deciding when it has enough. Self-RAG trains a model to decide when to retrieve and to critique relevance and support with special reflection tokens. Corrective RAG (CRAG) adds a retrieval evaluator that, depending on confidence, refines the retrieved documents, falls back to another source such as web search, or combines both; FLARE retrieves mid-generation on low-confidence tokens, and Adaptive-RAG routes by predicted complexity. The durable ideas are: retrieve on demand, grade retrieval before generating, verify the answer against evidence, and have a fallback. The pragmatic default is a fixed pipeline plus a router plus one corrective retry, using free-form agents only where needed, because loops cost latency, tokens and predictability.

**Key points to mention:**
- Controls: step and token budgets, tracing, tools that enforce the user's permissions.
- Evaluate trajectories (tool calls, steps), not just final answers.
- Agents multiply prompt-injection risk from retrieved content.
- Know when not to use an agent: simple lookups.

**Deep dive:** [Agentic RAG](../08-rag/09-advanced-rag-patterns.md#agentic-rag), [What Is an AI Agent?](../09-agents/01-what-is-an-ai-agent.md)

</details>

## Evaluating RAG

<details>
<summary><b>Q77. How do you evaluate the pipeline (a RAG system as a whole)?</b></summary>

**Answer:** Separately for retrieval and generation, on a golden set of realistic questions with known supporting evidence, then end to end and in production. Retrieval: Recall@K at the depth each stage consumes and nDCG or MRR for ordering, which need no LLM and are cheap and deterministic. Generation: faithfulness to the context, answer relevance, correctness against a reference, and citation precision and recall, usually scored by an LLM judge that has itself been calibrated against human labels. Include unanswerable questions to test abstention, and slice everything by question type (factual, multi-hop, numeric, ID lookup, tables). Run the suite as a regression test on every change to the chunker, embedding model, reranker, prompt or LLM, and close the loop with online signals: sampled faithfulness, thumbs, escalations, reformulations, latency and cost.

**Key points to mention:**
- Example: 1 of 2 relevant chunks in the top 5 gives Recall@5 = 0.5.
- Hold the generator fixed (same model, prompt, temperature 0) when comparing retrievers.
- Report confidence intervals; small eval sets hide or invent differences.
- Every debugged production failure becomes a regression case.

**Deep dive:** [The big picture: evaluate retrieval, generation and end to end](../08-rag/10-rag-evaluation-metrics-and-ragas.md#the-big-picture-evaluate-retrieval-generation-and-end-to-end), [Building an LLM Evaluation Framework and Regression Tests](../05-llms/11-building-an-llm-evaluation-framework-and-regression-tests.md)

</details>

<details>
<summary><b>Q78. How do you evaluate and debug retrieval quality?</b></summary>

**Answer:** Build a labeled test set of real queries with known relevant chunks (ideally graded), run the retriever, and compute Recall@K at the depth the next stage consumes (Recall@50 for a first stage feeding a reranker, Recall@5 and nDCG@5 for the list that enters the prompt), overall and per slice. Then debug by reading failures: sort queries by worst recall, compare retrieved vs gold chunks, and look up the gold chunk's rank (is it at 12, or not retrieved at all?). Compare BM25 vs dense on the same query, check metadata filters and permissions, and check chunk boundaries and parsing. Common root causes are parsing (tables, PDFs), chunking that splits the answer, jargon or IDs that dense models miss, a stale index and wrong filters. Validate each fix with a paired comparison on the same queries and confirm end-to-end answer quality.

**Key points to mention:**
- Log a retrieval trace per request (queries, filters, top-K IDs and scores, what entered the prompt) so any bad answer can be replayed.
- Test set: real log queries plus SME and reviewed synthetic queries, hard and unanswerable cases.
- Fix the largest failure bucket first.
- Turn every investigated failure into a test case.

**Deep dive:** [Retrieval metrics, worked by hand](../08-rag/10-rag-evaluation-metrics-and-ragas.md#retrieval-metrics-worked-by-hand), [How to inspect retrieval results](../08-rag/11-debugging-rag-systems.md#how-to-inspect-retrieval-results)

</details>

<details>
<summary><b>Q79. Explain Recall@K, Precision@K, MRR, and nDCG.</b></summary>

**Answer:** Recall@K is the fraction of all relevant items that appear in the top K. Precision@K is the fraction of the top K that is relevant. MRR (mean reciprocal rank) averages 1/rank of the first relevant item over queries, so it only cares about the first hit. nDCG@K uses graded relevance with a logarithmic position discount, $\text{DCG} = \sum (2^{rel_i}-1)/\log_2(i+1)$, normalized by the DCG of the ideal ranking, so it rewards putting the most relevant items at the very top. Example: relevant C2 (grade 2) and C5 (grade 1), ranking C5, C1, C2 gives Recall@3 = 1.0, Precision@3 = 0.667, reciprocal rank 1, and nDCG = 2.5/3.631 = 0.689. For RAG, recall matters most for the first stage, and nDCG matters for the final ordering because LLMs use the start of a long context better than the middle.

**Key points to mention:**
- Recall and precision ignore order within K; MRR sees only the first hit; nDCG sees all positions and grades.
- Precision@K is capped when a query has fewer than K relevant items.
- MRR suits single-answer lookups; nDCG suits graded, multi-answer ranking.
- Always state K, and choose it to match what the next stage reads.

**Deep dive:** [Retrieval metrics, worked by hand](../08-rag/10-rag-evaluation-metrics-and-ragas.md#retrieval-metrics-worked-by-hand)

</details>

<details>
<summary><b>Q80. How would you measure Recall@K, Precision@K, MRR, and nDCG in practice?</b></summary>

**Answer:** Create qrels (query to relevant chunk, document or answer span, with relevance grades), run the retriever on every query to get ranked lists, compute each metric per query, then macro-average across queries. Use a tested library (pytrec_eval, ranx) or a small verified function, and document conventions such as the nDCG gain formula (scikit-learn uses linear gain). Compute IDCG from all labeled relevant items, not just the retrieved ones, and label at span level so a chunk counts as relevant if it contains a labeled answer span, which keeps metrics valid when chunking changes. Measure on what actually entered the prompt (after truncation and deduplication) at the depth each stage reads. Report confidence intervals: on 100 queries with mean recall 0.8, the 95% interval is at most about plus or minus 0.08.

**Key points to mention:**
- Pool candidates from several retrievers before labeling so new systems are not penalized for finding unlabeled relevant items.
- A bootstrap over queries gives the exact interval.
- Per-query results enable paired comparisons between systems.
- Version qrels with the corpus snapshot.

**Deep dive:** [Retrieval metrics in code](../08-rag/10-rag-evaluation-metrics-and-ragas.md#retrieval-metrics-in-code), [Measuring retrieval recall in practice](../08-rag/10-rag-evaluation-metrics-and-ragas.md#measuring-retrieval-recall-in-practice)

</details>

<details>
<summary><b>Q81. What is retrieval recall?</b></summary>

**Answer:** The fraction of the relevant information for a query that the retriever actually returns within the top K, usually measured as Recall@K $= |\text{Rel}_q \cap \text{Top}_K(q)| / |\text{Rel}_q|$ against labeled relevant chunks, averaged over queries. Example: relevant chunks C3, C4 and C8, and the top 3 contains C3 and C8, so Recall@3 = 2/3 = 0.667. It is the ceiling on RAG quality, because the LLM cannot use what it never sees and a reranker cannot recover what the first stage missed. When you have reference answers but no chunk labels, a label-free variant is RAGAS context recall: the share of reference-answer claims supported by the retrieved context. Raising K raises recall but adds noise, tokens, cost and latency, which is why a reranker sits between a high-recall first stage and a small final K.

**Key points to mention:**
- Always state K and measure at the depth the next stage consumes.
- It depends on label completeness: unlabeled relevant chunks make recall look worse than it is.
- Distinguish it from ANN recall (agreement with exact search).
- Measure on what entered the prompt, after truncation.

**Deep dive:** [Measuring retrieval recall in practice](../08-rag/10-rag-evaluation-metrics-and-ragas.md#measuring-retrieval-recall-in-practice)

</details>

<details>
<summary><b>Q82. How do you create a reliable retrieval test set?</b></summary>

**Answer:** Start from real user queries sampled from logs, stratified by type and topic, deduplicated and PII-scrubbed, then add SME-written and human-reviewed synthetic questions to cover rare documents, plus hard cases and unanswerable questions. Have experts label relevant documents or answer spans with graded relevance using written guidelines, pool candidates from several retrievers so labels are not biased toward one system, and measure inter-annotator agreement. Label at span level, because chunk-ID labels break whenever chunking changes. Size it for the precision you need: confidence intervals shrink with $1/\sqrt{n}$, so 100 queries give roughly plus or minus 0.08 on a 0.8 hit rate and 400 give about plus or minus 0.04. Version the set with the corpus snapshot, keep a held-out split, refresh it with production failures, and retire queries about deleted documents.

**Key points to mention:**
- Synthetic questions copy source wording and are too easy; review and rewrite them.
- Tag queries by type for slice metrics; a slice of 10 queries supports no conclusion.
- Split by document to avoid leakage when the set is also used for fine-tuning.
- LLM-assisted labeling with human audit is acceptable at scale.

**Deep dive:** [Building a reliable retrieval test set](../08-rag/10-rag-evaluation-metrics-and-ragas.md#building-a-reliable-retrieval-test-set)

</details>

<details>
<summary><b>Q83. How do you evaluate RAG independently from the LLM?</b></summary>

**Answer:** Evaluate the retrieval stage on its own with a labeled test set and IR metrics (Recall@K, nDCG@K, MRR). No LLM is involved: you only compare ranked chunk IDs with relevance labels, so it is cheap, fast, deterministic and can run in CI on every change, isolating retriever, chunking, embedding and reranker changes from prompt or model changes. Conversely, evaluate the generator independently by fixing the context: the oracle-context test feeds gold chunks directly, so generator quality is measured without retrieval noise. RAGAS context recall and context precision also target retrieval, but they use an LLM judge, so they are not LLM-free. If "independently" means without trusting an LLM judge, add deterministic checks: exact match on short answers, numbers and cited IDs present in the context, a small NLI classifier for claims, and periodic human review.

**Key points to mention:**
- When comparing retrievers end to end, hold the generator fixed (same model, prompt, temperature 0).
- Retrieval metrics are necessary but not sufficient: confirm gains translate into better answers.
- Use label-based metrics as the CI gate, LLM-judged metrics nightly or before release.
- Clarify which meaning of "independently" the interviewer intends.

**Deep dive:** [The big picture: evaluate retrieval, generation and end to end](../08-rag/10-rag-evaluation-metrics-and-ragas.md#the-big-picture-evaluate-retrieval-generation-and-end-to-end)

</details>

<details>
<summary><b>Q84. What is faithfulness?</b></summary>

**Answer:** Faithfulness (groundedness) measures whether the answer's claims are supported by the retrieved context. You split the answer into atomic claims, check each claim against the context with an NLI-style judgment, and report supported claims divided by total claims. Example: three claims, two supported by chunk C2, and "unused days roll over" not in any chunk gives faithfulness 2/3. Faithfulness is not correctness: a faithful answer from an outdated document is still wrong, and a correct answer from model memory is unfaithful. Because it needs no reference answer, it can run on live traffic, which makes it a key production monitoring metric; at scale, a trained NLI classifier is a cheaper alternative to an LLM judge.

**Key points to mention:**
- Judge pitfalls: using its own knowledge, inconsistent claim splitting, very long contexts.
- Mitigate with "sources only" instructions, required evidence quotes and calibration against humans.
- Terse answers and abstentions can inflate it; pair with correctness and abstention rate.
- Claim granularity changes the score, so freeze the judge prompt and model.

**Deep dive:** [Faithfulness (groundedness)](../08-rag/10-rag-evaluation-metrics-and-ragas.md#faithfulness-groundedness)

</details>

<details>
<summary><b>Q85. What is answer relevance?</b></summary>

**Answer:** Answer relevance measures whether the answer directly addresses the user's question, regardless of whether it is correct. RAGAS computes it by having an LLM generate several questions the answer would respond to and averaging their embedding cosine similarity with the original question, $\frac{1}{N}\sum_i \cos(e_{g_i}, e_q)$; illustrative similarities of 0.95, 0.88 and 0.80 give 0.877. A direct judge rubric also works. It penalizes evasive, incomplete or padded answers, and some implementations zero out noncommittal ones. It does not check facts, so an on-topic wrong answer scores high, which is why it must be paired with faithfulness and correctness.

**Key points to mention:**
- Scores depend on the embedding model and judge; compare only with the same setup.
- Gaming risk: an answer that restates the question can still score well.
- Never use it alone as a release gate.
- Useful for spotting answers that drift from the question in multi-turn chat.

**Deep dive:** [Answer relevance](../08-rag/10-rag-evaluation-metrics-and-ragas.md#answer-relevance)

</details>

<details>
<summary><b>Q86. How do you distinguish a retrieval failure from a generation failure?</b></summary>

**Answer:** Check whether the answer-bearing chunk was in the context the LLM actually saw. Walk the chain: is the answer in the corpus at all, is the chunk indexed and correctly parsed, is it in the top K at any rank, did it survive truncation into the final prompt, and is the answer faithful to it? If the chunk never reached the prompt, it is a retrieval, ranking, ingestion or assembly failure; if it was there and the answer is still wrong or unsupported, it is a generation failure. The decisive second test is the oracle-context test: give the model the gold evidence with the same prompt; if it now answers correctly, the bottleneck is upstream, and if it still fails, it is a prompt or model problem. Aggregate the verdicts into a failure breakdown over the test set to decide where to invest.

**Key points to mention:**
- Variant: gold chunk plus the usual distractors; right alone but wrong with distractors means noisy or conflicting context.
- Metric pairing: low context recall with high faithfulness means retrieval problems; high context recall with low faithfulness means generation problems.
- A closed-book test shows whether correctness comes from memory rather than RAG.
- Double-check the gold label: a wrong label looks like a generation error.

**Deep dive:** [Telling a retrieval failure from a generation failure](../08-rag/10-rag-evaluation-metrics-and-ragas.md#telling-a-retrieval-failure-from-a-generation-failure), [Retrieval error or generation error? Two decisive tests](../08-rag/11-debugging-rag-systems.md#retrieval-error-or-generation-error-two-decisive-tests)

</details>

<details>
<summary><b>Q87. How would you use RAGAS or a custom evaluator?</b></summary>

**Answer:** Run the RAG pipeline on the test set and log, per query, the question, retrieved contexts, answer and reference, then compute RAGAS metrics such as faithfulness, response relevancy, context precision and context recall with a pinned judge model at temperature 0. Read the worst-scoring samples, not just the averages. Pin the RAGAS version, because its API changes between releases (0.1 used `question/answer/contexts/ground_truth`, 0.2 uses `user_input/response/retrieved_contexts/reference`). Where generic metrics miss what matters in your domain, write a custom evaluator: a versioned judge prompt with a narrow rubric, binary verdicts, required evidence quotes, JSON output and few-shot borderline examples, plus deterministic checks. Calibrate any evaluator on a human-labeled sample before using it as a release gate, and cache judge calls by (sample, metric, judge version) to control cost.

**Key points to mention:**
- Each LLM metric makes one or more judge calls per sample; budget for it.
- Write regression tests for the evaluator itself.
- Combine with label-based retrieval metrics that need no judge.
- Judge models drift when providers update them; pin versions.

**Deep dive:** [Using RAGAS](../08-rag/10-rag-evaluation-metrics-and-ragas.md#using-ragas), [Building a custom evaluator](../08-rag/10-rag-evaluation-metrics-and-ragas.md#building-a-custom-evaluator)

</details>

<details>
<summary><b>Q88. Why can an automated evaluation score disagree with human judgment?</b></summary>

**Answer:** Because the judge is a biased, noisy model applying a rubric that may not match what humans value, often against an imperfect reference. Typical causes are verbosity, position and self-preference bias, the judge using its own knowledge instead of the sources, rubric mismatch, bad or outdated references, claim-splitting granularity, and partial credit the metric cannot express. Judge models also drift when providers update them, and outputs are not perfectly deterministic. Calibrate: label an illustrative 100-300 samples, compute agreement and Cohen's kappa, read the confusion matrix to see whether the judge is lenient or strict, refine the rubric, and freeze it. Example: 85% raw agreement with 0.60 chance agreement gives $\kappa = 0.25/0.40 = 0.625$, and the confusion matrix might show the judge passing 10 answers humans failed.

**Key points to mention:**
- Humans disagree too; measure human-human agreement as the ceiling.
- Prediction-powered inference (used in ARES) corrects judge bias with a small human-labeled set.
- Re-calibrate after changing the judge model or prompt.
- Report judge-human agreement next to any automated score.

**Deep dive:** [When automated scores disagree with humans](../08-rag/10-rag-evaluation-metrics-and-ragas.md#when-automated-scores-disagree-with-humans), [LLM evaluation metrics and LLM-as-a-judge](../05-llms/10-llm-evaluation-metrics-and-llm-as-a-judge.md)

</details>

<details>
<summary><b>Q89. Your faithfulness scores are high, but users say the answers are wrong. What is going on?</b></summary>

**Answer:** Faithfulness only checks that the answer matches the retrieved context. If the context itself is wrong (an outdated policy, the wrong product version, another tenant's variant, a superseded document ranking above the current one), the model faithfully repeats wrong information. It can also mean the right document is missing and the model answers faithfully from a near-miss, or that the judge is too lenient. Investigate with answer correctness against references and context precision (is a stale chunk ranked above the right one?), look for duplicate or conflicting document versions, and check retrieval recall. Fixes are usually version and recency metadata, filters on current versions, deletion handling, and an audit of the judge with a human-labeled sample.

**Key points to mention:**
- "Faithful but wrong" is its own failure category.
- Track faithfulness together with correctness, context recall and user feedback.
- Check whether the judge uses outside knowledge.
- Stale-index problems often trace back to ingestion (Q99).

**Deep dive:** [Generation metrics: faithfulness, relevance and correctness](../08-rag/10-rag-evaluation-metrics-and-ragas.md#generation-metrics-faithfulness-relevance-and-correctness)

</details>

<details>
<summary><b>Q90. How many test queries do you need, and how do you know an improvement is real?</b></summary>

**Answer:** Enough that the confidence interval is narrower than the change you care about. For a proportion, $SE = \sqrt{p(1-p)/n}$: at $p = 0.8$, 100 queries give about plus or minus 0.08 at 95% and 400 give about plus or minus 0.04. Compare systems with paired tests on the same queries, because per-query difficulty cancels out and paired comparisons are far more sensitive than comparing two independent averages. Paired bootstrap: resample queries with replacement, recompute the metric difference each time, and check whether the 95% interval of the difference excludes zero. Example: a 2-point recall@5 gain on 200 queries is about 4 queries, within the noise (each system's recall has a standard error of about 2.8 points), so inspect which queries flipped in each direction before shipping.

**Key points to mention:**
- A change that fixes 10 queries and breaks 6 deserves a look at the 6.
- Each important slice needs enough queries on its own.
- Keep a held-out set to avoid overfitting the eval set through repeated tuning.
- If the change is cheap and the interval mostly positive, a canary rollout can decide.

**Deep dive:** [Size, confidence and versioning](../08-rag/10-rag-evaluation-metrics-and-ragas.md#size-confidence-and-versioning), [Tuning retrieval parameters systematically](../08-rag/11-debugging-rag-systems.md#tuning-retrieval-parameters-systematically)

</details>

## Debugging RAG systems

<details>
<summary><b>Q91. Your RAG system retrieves the wrong documents. How do you debug it?</b></summary>

**Answer:** Reproduce with a set of 20-50 failing queries rather than one, and for each, read the retrieval report: original and rewritten query, filters, top results with scores, metadata and full chunk text. Label the gold chunk, find its rank in a deep retrieval (say the top 100), and use that to localize the failure: not retrievable at all (parsing, chunking, embedding, freshness, filters, query understanding), retrieved but ranked low (ranking), or retrieved but dropped (context assembly). Cheap checks first: look up the gold document by ID in the index (missing means ingestion), search an exact phrase from it (not found means garbled parsing or a bad split), rerun without filters in a debug environment (found means a wrong filter or ACL), and compare BM25 vs dense. Also inspect why the wrong documents win: superseded versions, near-duplicates, boilerplate-heavy chunks or unnormalized score fusion. Fix the largest bucket first and confirm with recall@k on a fixed eval set.

**Key points to mention:**
- Pitfall: debugging on a single query and generalizing.
- Work in pipeline order, because upstream bugs masquerade as downstream ones.
- If BM25 wins on identifiers and jargon, add hybrid search.
- Add the diagnosed cases to the regression set.

**Deep dive:** [The golden rule: inspect the evidence before blaming the LLM](../08-rag/11-debugging-rag-systems.md#the-golden-rule-inspect-the-evidence-before-blaming-the-llm), [The failure taxonomy: symptoms and checks per stage](../08-rag/11-debugging-rag-systems.md#the-failure-taxonomy-symptoms-and-checks-per-stage)

</details>

<details>
<summary><b>Q92. How would you inspect retrieval results before looking at generated answers?</b></summary>

**Answer:** Make sure every request is traced, then for a failing query review the retrieval stage in isolation: the original and rewritten query, the filters with their values, the top-N chunk IDs with scores and ranks before and after reranking, each chunk's source, version and date, the full chunk text (not a truncated preview), and the exact final prompt. Compare that against the gold chunk a human says contains the answer. Inspect at several depths (5, 20, 100) to separate retrieval problems from ranking problems, and check the final prompt for the presence and position of the gold text, since context assembly can drop or bury it. Summarize what you see with recall@k (is the gold in the top k?), MRR (how high?) and context precision (how much of the retrieved context is relevant). Inspect the stored trace of the original request, not only a fresh re-run, because the index, caches or model may have changed since.

**Key points to mention:**
- Previews hide parsing errors and mid-sentence boundaries; read full text.
- Build a small inspection tool or use a tracing UI so this takes seconds.
- A re-run that "passes" does not explain the user's failure.
- Retrieval inspection comes before any prompt or model change.

**Deep dive:** [How to inspect retrieval results](../08-rag/11-debugging-rag-systems.md#how-to-inspect-retrieval-results)

</details>

<details>
<summary><b>Q93. Could the problem be caused by parsing, chunking, embeddings, filters, or indexing? How would you tell which?</b></summary>

**Answer:** Yes, any of them, and each leaves a distinct fingerprint. Parsing: the text itself is wrong when you read it next to the original (interleaved columns, gibberish, flattened tables). Chunking: the text is fine but the fact is split across chunks or lacks its heading. Embeddings: text and chunks are fine but semantic matches score low and BM25 often beats dense; re-embed a stored chunk and compare with its stored vector (cosine about 1.0 confirms the same model and preprocessing), and check prefixes, normalization and max length. Filters: the gold appears when filters are removed, or result counts collapse after filtering. Indexing: the document is missing, outdated or duplicated in the index compared with the source, visible in source vs index counts, freshness lag and ingestion job failures.

**Key points to mention:**
- Run the cheap checks first: ID lookups and exact-phrase searches take seconds, embedding experiments take hours.
- Flag pages with few characters or high symbol ratios as parsing suspects.
- Watch for post-filtering that empties result lists.
- Work in pipeline order.

**Deep dive:** [The failure taxonomy: symptoms and checks per stage](../08-rag/11-debugging-rag-systems.md#the-failure-taxonomy-symptoms-and-checks-per-stage)

</details>

<details>
<summary><b>Q94. When would you change the embedding model?</b></summary>

**Answer:** Only after fixing parsing, chunking and embedding code-path bugs (prefixes, normalization, truncation, mixed versions), and only when a labeled eval on real domain queries shows poor recall that cheaper fixes (hybrid search, a reranker, query rewriting, fine-tuning the current model) do not close. Hard requirement changes also justify it: new languages, much longer chunks than the model's max input, cost or latency targets, hosting or compliance constraints, licensing, or vendor deprecation. Compare candidates on your own data with correct prefixes and normalization, per slice, including dimension and storage (10 million chunks at 1,024 float32 dimensions is about 41 GB), latency and license. Decide the minimum gain that justifies migration before seeing the numbers, because a change means a full re-embed and re-index with blue-green migration.

**Key points to mention:**
- Signals: semantic misses on domain vocabulary, BM25 consistently beating dense on paraphrased queries.
- Re-tune similarity thresholds and hybrid weights after switching; they do not transfer.
- Cost estimate: chunks times tokens per chunk times price, plus double storage during migration.
- Do not switch for a small leaderboard improvement.

**Deep dive:** [When to change the embedding model](../08-rag/11-debugging-rag-systems.md#when-to-change-the-embedding-model), [Choosing an embedding model](../08-rag/02-text-embeddings-and-similarity-search.md#choosing-an-embedding-model)

</details>

<details>
<summary><b>Q95. How do you tune retrieval parameters systematically?</b></summary>

**Answer:** Fix a versioned eval set with gold chunks, record a baseline, then change one parameter at a time across a range: chunk size, overlap, first-stage top-k, hybrid weight or RRF k, reranker depth, final k and ANN search parameters. Track retrieval metrics, end-to-end metrics, latency and token cost for every run in an experiment tracker. Pick the knee of the quality-versus-latency curve rather than the maximum, confirm with a small grid around it because parameters interact (chunk size and final k compete for the same token budget, reranker depth caps what it can recover), check significance with paired comparisons, and validate on a held-out set. Watch the noise: with 200 queries at recall 0.8, the standard error is about $\sqrt{0.8 \times 0.2 / 200} \approx 0.028$, so a 2-point gain may be nothing.

**Key points to mention:**
- Chunk-size sweeps need re-chunking and re-embedding, so sweep on a representative sample.
- Inspect the queries that flipped in each direction.
- Keep a held-out set to avoid overfitting the eval set.
- Re-tune after adding a reranker or changing the embedding model.

**Deep dive:** [Tuning retrieval parameters systematically](../08-rag/11-debugging-rag-systems.md#tuning-retrieval-parameters-systematically)

</details>

<details>
<summary><b>Q96. Practical exercise: a RAG system has strong answer fluency but low faithfulness. Diagnose the pipeline in order, starting with retrieval and evidence inspection rather than immediately changing the LLM.</b></summary>

**Answer:** Fluent but unfaithful usually means the model is filling gaps, so start with the evidence, not the LLM. Step 0: trust the measurement, by validating the faithfulness judge against a human-labeled sample. Step 1: pull about 50 unfaithful traces and read the retrieved chunks and final prompts. Step 2: find the gold chunk's rank to split failures into not retrieved, ranked too low and dropped in assembly. Step 3: run the oracle-context test to separate retrieval from generation. Step 4: bucket and count (no relevant evidence, stale evidence, garbled tables, truncated or buried evidence, genuine over-reach, judge errors). Then fix in pipeline order (parsing, chunking, index freshness, filters, query understanding, ranking, assembly) and only then prompt and model: "answer only from context", explicit abstention, per-claim citations, low temperature, retrieval-confidence gating and a post-generation verification pass.

**Key points to mention:**
- Why not swap the LLM first: it cannot fix missing evidence, costs more, and can make unsupported answers more convincing.
- Measurement trap: terse answers and abstentions can inflate faithfulness; track correctness, context recall and abstention rate alongside.
- Re-run the eval after each fix and add diagnosed cases to the regression set.
- Monitor faithfulness on sampled production traffic afterwards.

**Deep dive:** [Full walkthrough: fluent answers, low faithfulness](../08-rag/11-debugging-rag-systems.md#full-walkthrough-fluent-answers-low-faithfulness), [Debugging Case: A RAG Application Gives Fluent but Unsupported Answers](../11-system-design/07-debugging-case-rag-fluent-but-unsupported-answers.md)

</details>

<details>
<summary><b>Q97. What would you log in a per-request trace for a RAG system?</b></summary>

**Answer:** One trace per request with a span per stage: the original and rewritten query, filters and their values, retrieval results with IDs, scores and ranks per leg, reranker inputs and outputs, the final context IDs in order with token counts and anything dropped by the budget, the full prompt (or a reference to it), model and decoding parameters, the output, citations and their validity, and latency per stage. Add the app, index, prompt and model versions, which are what let you tie a regression to a change, and user feedback. Store enough inputs to replay the request against a new configuration. Privacy matters because traces contain retrieved confidential text: redact PII, restrict access, set retention, and store full payloads for a sample plus all errors and negative feedback.

**Key points to mention:**
- Use OpenTelemetry-style spans; LLM observability tools (Langfuse, Phoenix, LangSmith and similar) build on this.
- Traces must be tenant-scoped like the data they contain.
- Per-leg ranks make hybrid failures attributable.
- Traces feed both debugging and new eval cases.

**Deep dive:** [Per-request tracing of every stage](../08-rag/11-debugging-rag-systems.md#per-request-tracing-of-every-stage)

</details>

<details>
<summary><b>Q98. What is "lost in the middle", and how do you handle context assembly?</b></summary>

**Answer:** Liu et al. (2023) found that models used relevant information best when it sat at the beginning or end of a long context and worse when it was in the middle. So keep contexts focused, place the strongest evidence first, deduplicate, keep same-document chunks in their original order, label each chunk with its source, and manage the token budget explicitly so nothing is silently truncated. Budget arithmetic example: $8{,}192 - 600 - 1{,}500 - 100 - 1{,}000 = 4{,}992$ tokens for chunks fits nine 512-token chunks, not ten. More context is not free: it raises cost and time to first token and adds distractors. The effect varies by model, so measure accuracy against gold position on your own model rather than assuming it.

**Key points to mention:**
- Log dropped chunks; never truncate the system prompt.
- Reranking plus a smaller final k is the main mitigation.
- Use MMR for diversity on broad questions.
- This is why nDCG (ordering) matters for the final list.

**Deep dive:** [How context is passed to the LLM](../08-rag/01-what-is-rag-and-why-it-matters.md#how-context-is-passed-to-the-llm), [Debugging RAG Systems](../08-rag/11-debugging-rag-systems.md)

</details>

<details>
<summary><b>Q99. Users report answers based on an outdated policy. How do you debug it?</b></summary>

**Answer:** This is most often an indexing and versioning problem. First confirm from the trace that the old figure actually appears in a retrieved chunk; if it does not, the model is answering from parametric memory (a generation and abstention problem), or the source system itself was never updated. Then look up both document IDs in the index and compare source and index timestamps: was the new document ingested, was the old one removed or marked superseded, and do both versions coexist and compete? A shorter or more keyword-dense old version can outrank the new one. Also check answer and retrieval caches and the freshness lag of the ingestion pipeline. Fixes are version and status metadata, deletion handling, a filter on current versions, recency signals in ranking and cache keys that include the index version.

**Key points to mention:**
- Monitor freshness lag and ingestion failures against an SLO.
- Add a canary query for critical policies.
- Dedup must keep the canonical latest version, not delete blindly.
- Turn the case into a regression test.

**Deep dive:** [The failure taxonomy: symptoms and checks per stage](../08-rag/11-debugging-rag-systems.md#the-failure-taxonomy-symptoms-and-checks-per-stage), [Handling document updates without mixed versions](../08-rag/12-production-ingestion-pipelines-at-scale.md#handling-document-updates-without-mixed-versions)

</details>

## Production ingestion: idempotency, updates, deletions and re-embedding

<details>
<summary><b>Q100. How would you handle document ingestion at scale?</b></summary>

**Answer:** Build an event-driven pipeline: connectors that detect changes per source (webhooks, CDC, delta APIs, ETags), a durable queue partitioned by document ID, separate worker pools for parsing, chunking and embedding (with batching and rate limiting), and index writers that upsert into the vector and keyword indexes using deterministic IDs. A registry tracks every document's version, content hash, chunk IDs and state, and a document only becomes visible when all its chunks are indexed. Size it first: 10 million chunks at 400 tokens is 4 billion tokens, about \$80 at an illustrative \$0.02 per million tokens, but about 66.7 hours at 1 million tokens per minute, so the bottleneck is quota, not money. Prioritize deletes over live updates over backfill, run periodic reconciliation between source and index, and monitor lag, failures and counts against a freshness SLO.

**Key points to mention:**
- Queues decouple stages with different resource profiles and give backpressure and retries.
- Correctness tools: idempotent writes, versioned swaps, DLQ, checkpoints, reconciliation.
- For large backfills, build a new collection offline and switch an alias.
- Interviewers will ask "what if the embedding API fails halfway?" next (Q104).

**Deep dive:** [Ingestion architecture: the big picture](../08-rag/12-production-ingestion-pipelines-at-scale.md#ingestion-architecture-the-big-picture), [Throughput and cost math](../08-rag/12-production-ingestion-pipelines-at-scale.md#throughput-and-cost-math)

</details>

<details>
<summary><b>Q101. How do you make ingestion idempotent?</b></summary>

**Answer:** Deterministic IDs plus upserts plus orphan cleanup plus version guards. Each chunk's ID is derived from `doc_id + version + chunk_index` (or a content hash), using UUIDv5 when the store requires UUIDs, so reprocessing the same input overwrites the same records instead of creating duplicates. After writing a document, delete any chunk of that document not in the new ID set, which removes stale trailing chunks when a document shrinks. A version check in the registry ignores events older than what is already applied, so retries and reordering cannot roll a document back. Partition the queue by `doc_id` (Kafka message key, SQS FIFO message group) so one document is processed in order by one consumer. This is needed because queues are at-least-once, workers crash after writing but before acknowledging, and operators replay dead-letter queues.

**Key points to mention:**
- Expensive non-idempotent side effects (LLM summaries) get an idempotency key and a cache.
- Test: ingest the same batch twice and assert identical index state.
- Exactly-once delivery does not solve it across external systems (Q109).
- Random IDs per run make idempotency impossible.

**Deep dive:** [Idempotent ingestion](../08-rag/12-production-ingestion-pipelines-at-scale.md#idempotent-ingestion)

</details>

<details>
<summary><b>Q102. How do you avoid duplicate chunks?</b></summary>

**Answer:** Remove the causes at each layer. Deterministic IDs and upserts stop retry duplicates. Canonical URLs stop the same page being ingested under several URLs. Exact hashes of normalized text (after Unicode NFKC, whitespace and zero-width-character cleanup) catch copies across systems. MinHash with LSH, or SimHash, catches near-duplicates such as templated pages: the probability that two min-hashes match equals the Jaccard similarity, and LSH banding with $b = 20$, $r = 5$ makes pairs at similarity 0.8 candidates with probability about 0.9996 and pairs at 0.3 only about 0.047. Boilerplate stripping during parsing removes repeated headers and footers inside documents. Be conservative with thresholds, because two versions of a policy can look 95% similar and mean opposite things.

**Key points to mention:**
- Keep one canonical copy but record all source locations and merge permissions correctly, or dedup becomes a security bug.
- Dedup at document level (cheap, before embedding) and chunk level (shared boilerplate across documents).
- Similar chunks from different product versions should stay distinct with version metadata.
- Measure duplicate rate in top-k results on a query sample.

**Deep dive:** [Avoiding duplicate chunks](../08-rag/12-production-ingestion-pipelines-at-scale.md#avoiding-duplicate-chunks)

</details>

<details>
<summary><b>Q103. How do you detect changed documents?</b></summary>

**Answer:** Use cheap signals to trigger work and content hashes to confirm it. Triggers come from CDC on databases, webhooks or delta APIs from SaaS tools, and conditional HTTP requests with ETags or Last-Modified for web content. When a trigger fires, fetch, parse and hash the normalized text: if the hash matches what is stored, skip; if it differs, re-chunk and re-embed only the chunks whose hashes changed. Store a pipeline fingerprint (parser, chunker and model versions) next to the hash, so internal changes also trigger reprocessing. Deletes are the hardest change to detect, so combine explicit delete events with a periodic listing diff, and use reconciliation as the backstop for anything the events missed.

**Key points to mention:**
- Timestamps suffer from clock skew and "touched but unchanged" edits; webhooks can be lost, duplicated or reordered.
- Structure-based chunking makes per-chunk hashing effective; fixed windows shift after an insertion.
- Persist delta cursors only after the batch is safely processed.
- Measure the share of triggers with unchanged hashes, ingestion lag, and reconciliation stale and orphaned counts.

**Deep dive:** [Detecting changed documents](../08-rag/12-production-ingestion-pipelines-at-scale.md#detecting-changed-documents)

</details>

<details>
<summary><b>Q104. How do you recover from partial embedding failures?</b></summary>

**Answer:** Classify errors first: transient ones (rate limits, timeouts, 5xx) are retried with exponential backoff and full jitter, respecting `Retry-After`; systemic ones (expired key, exhausted quota) trip a circuit breaker and alert instead of retrying; poison inputs are isolated by bisecting the failing batch and sent to a dead-letter queue. With full jitter, $d_n = \text{uniform}(0, \min(c, a \cdot 2^n))$, so with $a = 1$ s and $c = 60$ s the upper bounds are 1, 2, 4, 8, 16, 32 and 60 seconds. Checkpoint long backfills so they resume instead of restarting. Keep each document invisible until all of its chunks are indexed: the registry stores `expected_chunks` and flips the active-version pointer only when the written count matches. A reconciliation job finds anything still missing or stale and re-enqueues it, and because the pipeline is idempotent, every retry and replay is safe.

**Key points to mention:**
- The DLQ needs alerting on depth and age, triage and a replay tool.
- Pitfall: partial success with immediate visibility, so users see half a document.
- Separate priority lanes keep live updates flowing during a backfill.
- Idempotency is what makes aggressive retrying safe.

**Deep dive:** [Recovering from partial embedding failures](../08-rag/12-production-ingestion-pipelines-at-scale.md#recovering-from-partial-embedding-failures)

</details>

<details>
<summary><b>Q105. How do you generate and version embeddings?</b></summary>

**Answer:** Generate them in batched embedding workers with exactly the same model, prefixes and normalization used at query time, enforcing the model's max input length and logging any truncation. Batch by length on GPUs to reduce padding, size API concurrency with Little's Law ($L = \lambda W$), and cache vectors by (embedding fingerprint, chunk content hash) so unchanged content is never re-embedded. Version them by attaching a fingerprint (model name, pinned revision, dimension, normalization, prefixes, chunker and parser versions) to every index and document, keeping one index per fingerprint, and binding the query encoder to the index in one config record. Pin hosted model versions, because an unannounced upgrade is an unplanned re-embed. At startup, check that the query encoder's fingerprint matches the index's stored fingerprint and fail loudly on mismatch.

**Key points to mention:**
- Same-dimension models are not interchangeable.
- Track embedding cache hit rate, tokens embedded per day and truncation rate.
- Store embedding lineage in a registry like any model artifact.
- Migrations use side-by-side indexes and an alias switch (Q107).

**Deep dive:** [Generating and versioning embeddings](../08-rag/12-production-ingestion-pipelines-at-scale.md#generating-and-versioning-embeddings)

</details>

<details>
<summary><b>Q106. How do you handle document updates and deletions?</b></summary>

**Answer:** Updates use a versioned swap: write the new version's chunks with version-tagged IDs while the old version still serves, verify that all are present, flip the document's active-version pointer in one atomic write, then garbage-collect the old chunks. That avoids both the mixed-version window of upserting in place and the missing-document window of delete-then-insert, at the cost of briefly storing two versions of that document. The pointer is enforced by filtering retrieved chunks against the registry, or by a transactional flag flip in stores with transactions such as pgvector. Deletions use a tombstone with immediate hiding, then physical removal from every store and derived artifact, then verification. Permission-only changes update metadata without re-embedding, and version guards plus partitioning by `doc_id` keep stale events from rolling anything back.

**Key points to mention:**
- If cleanup fails, old chunks are invisible anyway and a GC job removes them later.
- Measure update and delete lag against separate SLOs, docs stuck in a non-visible state, and GC backlog.
- Assert in tests or audits that no result set contains two versions of one document.
- Remember caches and derived artifacts (summaries, contexts, graph nodes).

**Deep dive:** [Handling document updates without mixed versions](../08-rag/12-production-ingestion-pipelines-at-scale.md#handling-document-updates-without-mixed-versions)

</details>

<details>
<summary><b>Q107. How would you version embedding models without corrupting existing indexes?</b></summary>

**Answer:** One index per embedding fingerprint, never mixed, because vectors from different models live in different coordinate systems and cross-model cosine is meaningless and fails silently, even at the same dimension. For a new model, create a new index side by side, start dual-writing new changes to both indexes before the backfill begins, backfill the new one with the idempotent pipeline, and evaluate it offline (recall@k and nDCG on a golden set) and in shadow mode (overlap@k on real traffic with judged samples), plus latency and cost. Then atomically switch an alias that bundles the index and its matching query encoder. Keep the old index and dual-writes during a rollback window, then delete it. Budget for a full re-embed (the whole token count again, 4 billion tokens for 10 million 400-token chunks) and double storage during the overlap.

**Key points to mention:**
- Index writers reject any vector whose fingerprint differs from the collection's stored fingerprint.
- Version guards resolve races between backfill and live writes.
- The same playbook applies to chunker or parser changes.
- Keep live updates fresh during the backfill with priority lanes and reserved quota.

**Deep dive:** [Migrating to a new embedding model safely](../08-rag/12-production-ingestion-pipelines-at-scale.md#migrating-to-a-new-embedding-model-safely), [Index rebuilds and reindexing](../08-rag/07-vector-databases-in-practice.md#index-rebuilds-and-reindexing)

</details>

<details>
<summary><b>Q108. How do you guarantee deleted documents are no longer retrievable?</b></summary>

**Answer:** Three layers. Hide fast: a tombstone and visibility flag in the registry that the query path filters on, so the document disappears immediately, with versions on tombstones so late update events cannot resurrect it. Remove everywhere: hard delete from the vector index, keyword index, chunk store, object store and caches, plus every derived artifact found through lineage (summaries, contextual chunks, graph entities), with compaction for soft-deleting engines and retention limits or crypto-shredding for backups. Prove it: a scheduled verification job queries by deleted IDs and probes with distinctive content from the deleted document, alerting on any hit; content probes catch copies under other IDs that ID checks cannot. Many ANN and Lucene-style engines only mark deletes until segments merge, so legal deletion must account for compaction timing.

**Key points to mention:**
- Track delete lag with a stricter SLO than update lag.
- Logs, traces and eval sets containing retrieved text need retention limits too.
- Deletes get top priority in the queue.
- Per-tenant keys make crypto-shredding practical.

**Deep dive:** [Guaranteeing deleted documents are no longer retrievable](../08-rag/12-production-ingestion-pipelines-at-scale.md#guaranteeing-deleted-documents-are-no-longer-retrievable)

</details>

<details>
<summary><b>Q109. Can you get exactly-once processing in an ingestion pipeline?</b></summary>

**Answer:** Not end to end across a queue, an external embedding API and a vector database. What you get is at-least-once delivery combined with idempotent processing, so duplicates have no effect: "effectively once", which is the standard design. Kafka transactions give exactly-once semantics for read-process-write within Kafka, not for side effects in other systems. Stream processors such as Flink reach end-to-end exactly-once only with transactional (two-phase commit) or idempotent sinks, and a paid embedding API is neither, while most vector databases offer no distributed transactions. So the practical tools are deterministic IDs, upserts, version guards and idempotency keys for paid side effects.

**Key points to mention:**
- The transactional outbox pattern avoids losing events from your own application database.
- Saying "we use exactly-once delivery" is a red flag answer.
- Design for duplicates and reordering, then test both.
- Caching paid calls by idempotency key avoids paying twice.

**Deep dive:** [Idempotent ingestion](../08-rag/12-production-ingestion-pipelines-at-scale.md#idempotent-ingestion)

</details>

<details>
<summary><b>Q110. A user reports the assistant quoted a document that was deleted last week. How do you debug it?</b></summary>

**Answer:** Trace the answer back to the exact retrieved chunk IDs and their source, then check in order. Did the delete event arrive (is there a registry tombstone)? If not, the connector missed it, and listing-diff reconciliation is the fix. Was it hidden (visibility flag)? A tombstone that is still retrievable means the query path is not filtering, or one index (keyword vs vector) missed the delete. Is the text coming from somewhere else: a duplicate under another chunk ID, a derived summary, an answer cache hit, or agent memory? A different chunk ID with the same content points to a dedup or lineage gap, fixed with content probes in verification; a cache hit calls for invalidation by document tag and shorter TTLs. Finish with a regression test and a probe for this document.

**Key points to mention:**
- Check every store: vector, keyword, cache, derived artifacts, backups.
- Hybrid search means two indexes that must both honor deletes.
- This can be a compliance incident; treat it with urgency.
- Measure delete lag continuously, not only after incidents.

**Deep dive:** [Guaranteeing deleted documents are no longer retrievable](../08-rag/12-production-ingestion-pipelines-at-scale.md#guaranteeing-deleted-documents-are-no-longer-retrievable)

</details>

## Secure multi-tenant RAG

<details>
<summary><b>Q111. How would you build (implement) secure, multi-tenant RAG?</b></summary>

**Answer:** Start from a threat model (cross-tenant leakage, intra-tenant permission bypass, prompt injection in documents, leakage through caches, logs, embeddings and summaries, stale ACLs), then enforce authorization in the retrieval layer, not in the LLM. Every chunk carries `tenant_id` and the document's allowed principals synced from the source system. The server builds the user's identity from a verified token, resolves their groups, and every search runs with a tenant scope and an ACL pre-filter inside the query, through one retriever service that is the only path to the index and does not accept tenant or filter arguments from callers. Isolation is tiered: pooled index with mandatory filters or namespaces for small tenants, dedicated indexes for large or regulated ones, backed by row-level security or namespaces at the storage layer. Add defense-in-depth checks at prompt-building and citation time, permission-scoped caches, ACL sync with a bounded revocation window, tamper-evident audit logs, PII handling and per-tenant encryption keys.

**Key points to mention:**
- Agents: every tool enforces the end user's identity; no all-powerful service account.
- Test with canary documents per tenant, bypass tests and revocation-latency tests; the leakage target is zero.
- Caches and traces are the most common real-world leak paths.
- Fail closed when any check cannot complete.

**Deep dive:** [The threat model](../08-rag/13-secure-multi-tenant-rag.md#the-threat-model), [Going deeper: reference architecture, testing and scale](../08-rag/13-secure-multi-tenant-rag.md#going-deeper-reference-architecture-testing-and-scale)

</details>

<details>
<summary><b>Q112. How do you isolate tenant data?</b></summary>

**Answer:** There are three options. Silo: a separate index or database per tenant, with the strongest isolation and easiest deletion but the highest cost and operational load. Namespaces or partitions: one cluster with a logical container per tenant, giving good isolation at moderate cost and making offboarding a simple drop. Shared index with a mandatory tenant filter: cheapest and simplest to operate, but one missing filter leaks everything, so the filter must be injected automatically by a central component and backed by database-level controls like row-level security. Most platforms tier: pool the long tail and silo enterprise or regulated tenants, with a tenant router hiding physical placement so tenants can move between tiers. Isolation applies to everything, not just the index: caches, logs, traces, queues, eval datasets, backups and encryption keys.

**Key points to mention:**
- Decision inputs: number and size distribution of tenants, compliance (residency, customer-managed keys), budget, noisy-neighbor tolerance.
- Per-tenant keys plus crypto-shredding make deletion auditable, if no plaintext copies live elsewhere.
- Huge tenants get a dedicated index, quotas and their own scaling.
- Measure cross-tenant canary leakage (must be zero) and per-tenant p95 latency.

**Deep dive:** [Tenant isolation options](../08-rag/13-secure-multi-tenant-rag.md#tenant-isolation-options), [Multi-tenancy: collections, namespaces and partitions](../08-rag/07-vector-databases-in-practice.md#multi-tenancy-collections-namespaces-and-partitions)

</details>

<details>
<summary><b>Q113. How do you enforce document-level access permissions, and how do you handle document-level and user-level permissions?</b></summary>

**Answer:** Document level: sync each document's ACL (including inherited permissions) from the source system at ingestion and store the allowed principals, usually groups rather than expanded user lists, as metadata on every chunk. User level: at query time, resolve the user's principal set (user ID plus nested groups) from the directory based on the authenticated session. Retrieval then filters inside the search query on `tenant_id = session tenant AND allowed_principals overlaps user principals`, with no deny entry matching: formally, $A(c) \cap P(u) \neq \varnothing$. Storing groups on chunks means a user removed from a group loses access at the next query without reindexing. Chunks with missing or unparseable ACL metadata are excluded (fail closed), never treated as public.

**Key points to mention:**
- Group explosion: keep only groups that appear in some ACL; use a ReBAC service (Zanzibar-style) for complex hierarchies.
- Do not trust token group claims as complete; some identity providers truncate them.
- Derived artifacts over several documents get the intersection of source ACLs.
- For restricted documents, add a late-binding check against the source of truth on the final top-k.

**Deep dive:** [Document-level ACLs and user-level permissions](../08-rag/13-secure-multi-tenant-rag.md#document-level-acls-and-user-level-permissions)

</details>

<details>
<summary><b>Q114. Why is filtering only after retrieval unsafe?</b></summary>

**Answer:** Because by the time you filter, unauthorized content has already been read into memory and may have passed through the reranker, query rewriting, the prompt, logs or traces. It also opens side channels (result counts, timing, "some results were hidden" messages), depends on a separate step that a bug can skip, and destroys recall because forbidden documents use up the top-k. The recall math: with visible fraction $p$ and top-$k$, expected survivors are $kp$ and $P(\text{none}) = (1-p)^k$, so at $p = 0.05$ and $k = 10$ you expect 0.5 survivors and have about a 60% chance of zero results. Compensating with a huge k means loading even more forbidden data and paying more latency. Pre-filtering searches only the visible subset, so forbidden data never leaves storage.

**Key points to mention:**
- A post-retrieval check is fine as a redundant second layer on already pre-filtered results.
- Use engines with filter-aware traversal, or exact search on small filtered sets.
- Measure filtered recall for users with narrow permissions.
- The same math explains the "small tenant gets zero results" bug (Q51).

**Deep dive:** [Why filtering only after retrieval is unsafe](../08-rag/13-secure-multi-tenant-rag.md#why-filtering-only-after-retrieval-is-unsafe)

</details>

<details>
<summary><b>Q115. Where should authorization be enforced?</b></summary>

**Answer:** Primarily in the retrieval service and the tool layer, using identity from the authenticated session, so forbidden data is never fetched. Then defense in depth: database-level isolation (row-level security, namespaces), a re-check of tenant and ACL for every chunk in the prompt builder, and citation verification at output time. Never in the LLM's instructions and never only in the UI. Use many policy enforcement points but one consistent decision point (a policy engine or authorization service) so rules do not drift between components. For agents, the model only supplies the query, while identity and scope are bound server-side, which prevents confused-deputy attacks. The redundant checks should raise security alerts when they fire, and everything should fail closed if a check cannot complete.

**Key points to mention:**
- Late-binding checks against the source system for sensitive documents.
- Tool arguments from the model are untrusted input.
- Measure invariant-violation alerts (expected zero) and bypass-test pass rate.
- Track the latency each enforcement point adds.

**Deep dive:** [Where authorization should be enforced](../08-rag/13-secure-multi-tenant-rag.md#where-authorization-should-be-enforced)

</details>

<details>
<summary><b>Q116. How do you handle revoked permissions and deleted documents? What happens when a user loses access to a document?</b></summary>

**Answer:** Keep the exposure window small, measured and stated as an SLO per sensitivity level. Sync ACLs and group memberships with webhooks or change feeds plus periodic full reconciliation, store groups on chunks so removing a user from a group takes effect at the next query without reindexing, use short cache TTLs and an ACL epoch in cache keys, and for sensitive documents re-check the final results against the source of truth. The window is roughly $T_{\text{detect}} + T_{\text{propagate}} + T_{\text{cache}}$: illustratively 15 + 2 + 10 = 27 minutes with polling versus 1 + 2 + 2 = 5 minutes with webhooks and short TTLs. When a user loses access, the directory or ACL update and an epoch bump make old cache entries unreachable, and the next retrieval cannot return the document; past answers cannot be recalled, but the audit log shows what they saw. Deletes write a tombstone that retrieval honors immediately, then chunks, embeddings, caches and derived artifacts are purged.

**Key points to mention:**
- Do not forget copies: summaries, graph nodes, eval sets, fine-tuning data, backups (crypto-shredding helps), logs.
- Fail closed if the source-of-truth check times out.
- Test revocation latency continuously.
- Permission-only changes must not require re-embedding.

**Deep dive:** [Revoked permissions and deleted documents](../08-rag/13-secure-multi-tenant-rag.md#revoked-permissions-and-deleted-documents)

</details>

<details>
<summary><b>Q117. How do you prevent cross-tenant cache leakage?</b></summary>

**Answer:** Scope every cache layer (retrieval, semantic, response) by tenant and by a permission fingerprint: a hash of the user's sorted effective principal set, restricted to principals that some ACL actually names, so users with identical access can share entries. The key is roughly $H(\text{tenant} \,\|\, \text{fp}(P(u)) \,\|\, \text{acl\_epoch} \,\|\, \text{normalize}(q) \,\|\, \text{versions})$, including index, prompt and model versions. Semantic caches search only within the (tenant, fingerprint) partition, and an entry is never shared across different permission sets. Bump a per-tenant ACL epoch on permission changes to invalidate instantly without scanning, and use short TTLs to bound the revocation window. A cache hit bypasses late-binding and tombstone checks, so do not cache restricted answers, or re-check cited chunks on every hit.

**Key points to mention:**
- Fingerprinting the raw principal set (including the user ID) makes every key per-user and kills hit rate.
- Shared LLM KV prefix caches can leak through timing; isolate per tenant or use per-tenant cache salts.
- Test with canary queries across tenants and permission sets.
- Caches are one of the most common real-world leak paths.

**Deep dive:** [Preventing cache leakage across tenants](../08-rag/13-secure-multi-tenant-rag.md#preventing-cache-leakage-across-tenants)

</details>

<details>
<summary><b>Q118. How would you audit which sources supported a response?</b></summary>

**Answer:** Log per request: tenant, user (or a pseudonym), query, permission fingerprint and ACL epoch, retrieved candidate chunk IDs with document versions and content hashes, the chunks actually placed in the prompt, citations (and any dropped citations), reranker, prompt template and model versions, and authorization decisions. Versions and hashes let you prove exactly what text was read even after documents change, and enable replay: re-running retrieval with the same index version to debug or answer a customer complaint. Store logs tenant-scoped, access-controlled and tamper-evident, using a hash chain $h_i = H(h_{i-1} \| r_i)$ with the head hash anchored externally, or write-once object-lock storage. Prefer IDs and hashes over copying raw text into logs, redact PII, and balance retention between audit obligations and deletion rights.

**Key points to mention:**
- Measure audit completeness: the share of answers whose every citation resolves to a logged chunk ID, version and hash (target 100%).
- Verify the hash chain periodically.
- Keep old document versions addressable as long as logs reference them.
- Audit logs also show which chunk was in context when a suspicious action happened.

**Deep dive:** [Auditing which sources supported a response](../08-rag/13-secure-multi-tenant-rag.md#auditing-which-sources-supported-a-response)

</details>

<details>
<summary><b>Q119. Isn't it enough to tell the model in the system prompt to only use documents the user may see?</b></summary>

**Answer:** No. The model cannot verify identity, has no reliable way to know each document's ACL, and can be manipulated by prompt injection, so anything placed in the context window may appear in the output. Instructions reduce accidents, but they are not a security boundary. The correct mental model is that the LLM is an untrusted component that only ever sees pre-authorized data: authorization must happen before data reaches the model, in the retrieval and tool layers. The same logic applies to tool calls, where arguments produced by the model are untrusted input to be validated and authorized server-side.

**Key points to mention:**
- "Defense in the prompt" fails silently and is impossible to audit.
- Pair with pre-filtering (Q114) and enforcement points (Q115).
- The output can still be checked (citation verification) as a second layer.
- A good answer names prompt injection as the reason, not just "models make mistakes".

**Deep dive:** [Where authorization should be enforced](../08-rag/13-secure-multi-tenant-rag.md#where-authorization-should-be-enforced)

</details>

<details>
<summary><b>Q120. A document in the corpus contains a prompt injection. What can it do, and how do you limit the damage?</b></summary>

**Answer:** This is indirect prompt injection (Greshake et al., 2023): instructions hidden in retrieved content rather than in the user's message. If identity is bound to the session and retrieval is pre-filtered, it cannot make retrieval return data the user cannot already see. It can still try to exfiltrate the user's visible data (for example through a rendered Markdown image URL carrying data to an attacker's server), mislead the user with false instructions, or trigger tool actions. Limit it by disabling or allowlisting external links and images in output, requiring confirmation for side-effecting tools, giving tools least privilege with no ability to send data to model-chosen destinations, labeling retrieved content as untrusted data in the prompt, tracking source trust, and scanning content at ingestion.

**Key points to mention:**
- Put retrieved text in the user turn, not the system prompt.
- Audit logs show which chunk was in context when a suspicious action happened.
- Agentic RAG widens the blast radius because the model can act.
- No mitigation is perfect, so design for containment.

**Deep dive:** [Prompt injection inside documents](../08-rag/13-secure-multi-tenant-rag.md#prompt-injection-inside-documents), [Agent reliability, security and prompt injection](../09-agents/08-agent-reliability-security-and-prompt-injection.md)

</details>

<details>
<summary><b>Q121. Are embeddings sensitive data? Can we share an embedding index across tenants since vectors "are just numbers"?</b></summary>

**Answer:** Yes, embeddings are sensitive. Research (Morris et al., 2023, Vec2Text) showed that text can be substantially reconstructed from embeddings, near-exactly for short passages, when the attacker can query the same embedding model, which is often a public or widely used one. So vectors must be treated like the source text: same isolation, encryption, access control, retention and deletion. Sharing a physical index across tenants is fine only with enforced tenant scoping, exactly as in any pooled design. Deleting a document must delete its vectors too, including in backups or via crypto-shredding, and vectors sent to a third-party embedding or vector service fall under the same data processing agreements as the text.

**Key points to mention:**
- "Just numbers" is the misconception the interviewer is testing.
- The same applies to derived artifacts such as summaries and graph nodes, which need intersection ACLs.
- Logs that contain vectors need the same protection.
- Mention residency rules for hosted vector services.

**Deep dive:** [PII handling and encryption](../08-rag/13-secure-multi-tenant-rag.md#pii-handling-and-encryption)

</details>

## Rapid-fire round

| Question | Answer |
|---|---|
| What does RAG stand for, and what does it fix? | Retrieval-augmented generation: knowledge cutoff, private data, freshness, attribution, and some hallucination. |
| Cosine vs dot product on unit vectors? | Identical; and $\lVert a-b \rVert^2 = 2 - 2\cos$, so Euclidean gives the same ranking. |
| Raw size of 10M x 1,024 float32 vectors? | About 41 GB before index overhead. |
| Raw size of 1M x 768 float32 vectors? | About 3 GB. |
| InfoNCE in one line? | Softmax cross-entropy over similarities divided by temperature, with the true positive as the correct class. |
| What are in-batch negatives? | The other B - 1 positives in a batch, reused as free negatives for each anchor. |
| What is a Matryoshka embedding? | A vector trained so its prefixes (for example the first 256 dims) remain useful; re-normalize after truncating. |
| Typical starting chunk size and overlap for prose? | About 256-512 tokens with 10-20% overlap, then tune on an eval set. |
| Storage overhead of overlap O on chunk size S? | About S/(S - O), for example 512/448 = 1.14. |
| Why label eval data at span level? | Chunk-ID labels break whenever chunking changes; spans survive re-chunking. |
| BM25 default parameters? | k1 around 1.2 to 1.5 (term-frequency saturation), b = 0.75 (length normalization). |
| What does k1 = 0 do in BM25? | Turns it into a binary model: a sum of IDFs of matched terms. |
| RRF formula and default k? | Sum over lists of 1/(k + rank), with k = 60. |
| Main weakness of RRF? | It discards score margins, so a dominant exact match counts like a marginal rank 1. |
| How many candidates per leg before fusion? | Roughly 50-100, not the top 5. |
| What is SPLADE? | Learned sparse retrieval: vocabulary-wide term weights with expansion, served from an inverted index. |
| HNSW runtime knob? | efSearch: higher means more recall and more latency, with no memory cost. |
| HNSW build-time knobs? | M (links per node, memory and recall ceiling) and efConstruction (build quality and time). |
| HNSW link memory at M = 32 for 10M vectors? | About 2M x 4 bytes per vector, 2.56 GB. |
| What does nprobe control? | The number of IVF cells scanned per query; nprobe = nlist with IVF-Flat is exact search. |
| Fraction scanned with nlist 4,096 and nprobe 16? | About 0.39% of the vectors. |
| PQ size with m = 96 and 8 bits? | 96 bytes per vector, 32x smaller than 3,072-byte float32 768-dim vectors. |
| Why "asymmetric" distance in PQ? | Only database vectors are quantized; the query stays at full precision. |
| ANN recall vs retrieval recall? | ANN recall is agreement with exact search; retrieval recall is agreement with relevance labels. |
| Post-filter zero-result probability? | (1 - s)^k; for s = 0.01 and k = 100 about 37%, for s = 0.001 about 90%. |
| Why deletes are hard in HNSW? | Removing a node breaks neighbors' links, so systems tombstone and compact later. |
| Bi-encoder vs cross-encoder in one line? | Bi-encoder: separate encodings, precomputable, fast; cross-encoder: joint encoding, accurate, one pass per pair. |
| Typical rerank depth and final k? | Rerank 50-100 candidates, keep 3-8 for the prompt. |
| What is MaxSim? | ColBERT scoring: for each query token, its best dot product with any document token, summed. |
| What is HyDE? | Embed an LLM-written hypothetical answer instead of the query to search for real answers. |
| Recall@K vs Precision@K? | Recall: share of relevant items found in the top K; precision: share of the top K that is relevant. |
| MRR in one line? | Mean over queries of 1/rank of the first relevant result. |
| Why nDCG for the final list? | It rewards the best chunks at the top, which matters because LLMs under-use mid-context information. |
| Faithfulness vs correctness? | Faithfulness: supported by the context; correctness: true against a reference. Faithful answers from stale docs are wrong. |
| Oracle-context test? | Feed the gold chunk directly; if the answer becomes correct, the failure is upstream of generation. |
| 95% margin on 100 queries at 0.8 recall? | About plus or minus 0.08; 400 queries give about plus or minus 0.04. |
| Exactly-once ingestion? | Not end to end; use at-least-once delivery plus idempotent processing. |
| Deterministic chunk ID recipe? | Hash or UUIDv5 of doc ID, version and chunk index or content. |
| Safe update pattern? | Versioned swap: write new chunks, verify, flip the active-version pointer atomically, then garbage-collect. |
| Can you embed queries with a new model against an old index? | Never; different models live in different coordinate systems, even at the same dimension. |

## How to practice

- **Answer out loud first.** Read only the question, then speak your answer before opening the collapsed model answer. Saying it exposes gaps that silent reading hides.
- **Time yourself to 2 minutes.** That is roughly the length of a good first answer in an interview: definition, mechanism, one trade-off and one concrete number. Stop when the timer ends and compare.
- **Explain it to a non-expert.** If you cannot explain HNSW, RRF or faithfulness to a friend without jargon, you do not own it yet. Use the analogies from the chapters (maps with different orientations, open-book exams).
- **Then go deeper.** Use the "Key points to mention" bullets as the interviewer's likely follow-ups and answer each one in a sentence or two. For anything you miss, open the "Deep dive" chapter section.
- **Work in rounds.** Day 1: all sections, mark weak questions. Day 2: only the marked ones plus the rapid-fire table. Final days: rapid-fire round daily, and the practical exercise (Q96) and debugging questions as mock walkthroughs.
- **Practice the numbers.** Memory math, post-filter probabilities, RRF scores and confidence intervals come up constantly; recompute a few by hand rather than memorizing them. The [coding exercises for LLMs, RAG and agents](02-coding-exercises-llm-rag-and-agents.md) let you implement BM25, RRF and retrieval metrics yourself.
