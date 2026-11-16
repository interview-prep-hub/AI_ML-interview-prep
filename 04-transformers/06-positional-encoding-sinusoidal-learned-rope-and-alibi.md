# Positional Encoding: Sinusoidal, Learned, RoPE and ALiBi

> **Level:** Beginner to Advanced · **Study time:** ~110 min · **Prerequisites:** [Linear Algebra for ML](../00-foundations/02-linear-algebra-for-ml.md), [The Attention Mechanism, Step by Step](03-attention-mechanism-step-by-step.md), [Multi-Head Attention and Causal Masking](04-multi-head-attention-and-causal-masking.md), [The Full Transformer Architecture](05-the-full-transformer-architecture.md)

## Table of contents

- [Why this matters](#why-this-matters)
- [The problem: attention cannot see order](#the-problem-attention-cannot-see-order)
- [The design space: absolute vs relative, and where position enters](#the-design-space-absolute-vs-relative-and-where-position-enters)
- [Sinusoidal positional encoding](#sinusoidal-positional-encoding)
- [Learned absolute position embeddings](#learned-absolute-position-embeddings)
- [Sinusoidal vs learned, side by side](#sinusoidal-vs-learned-side-by-side)
- [Relative position ideas](#relative-position-ideas)
- [RoPE: Rotary Position Embedding](#rope-rotary-position-embedding)
- [ALiBi: Attention with Linear Biases](#alibi-attention-with-linear-biases)
- [All methods compared](#all-methods-compared)
- [Why models struggle beyond their training length](#why-models-struggle-beyond-their-training-length)
- [Extending the context window](#extending-the-context-window)
- [Context window vs long-context use](#context-window-vs-long-context-use)
- [Going deeper: production pitfalls and debugging](#going-deeper-production-pitfalls-and-debugging)
- [Common mistakes and misconceptions](#common-mistakes-and-misconceptions)
- [Interview questions and model answers](#interview-questions-and-model-answers)
- [Key takeaways](#key-takeaways)
- [Further reading](#further-reading)
- [Next](#next)

## Why this matters

Every Transformer needs some way to know that "dog bites man" and "man bites dog" are different sentences, and the method a model uses for this decides how long its inputs can be. When a vendor announces that a model now supports 128k or 1M tokens, the engineering underneath is almost always a change to positional encoding (a new RoPE base, interpolation, NTK or YaRN scaling, all explained later in this chapter) plus continued training on long documents. In production you will meet these ideas as config fields like `rope_theta` and `rope_scaling`, as garbage output when someone serves a model past its trained length, and as RAG answers that miss facts buried in the middle of a long prompt. In interviews, "why does attention need positions?", "explain RoPE" and "how would you extend a model's context window?" are standard questions, and senior candidates are expected to separate "the model accepts 128k tokens" from "the model can actually use 128k tokens".

## The problem: attention cannot see order

### Intuition first

> **Analogy:** Imagine dumping a sentence written on Scrabble tiles into a bag. The bag contains exactly the same tiles for "dog bites man" and "man bites dog". Anyone who only looks inside the bag cannot tell who bit whom. Self-attention, on its own, is that bag.

Recall from [The Attention Mechanism, Step by Step](03-attention-mechanism-step-by-step.md) that each token's output is a weighted average of all tokens' **value** vectors, with weights computed from how well its **query** matches each token's **key**:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $Q = XW_Q$, $K = XW_K$, $V = XW_V$: queries, keys and values, one row per token.
- $X$: the input matrix, one row per token (its embedding vector).
- $d_k$: the length of each query and key vector.

Look at what the score between token $i$ and token $j$ depends on: only the vectors $x_i$ and $x_j$. Nothing in the formula says *where* in the sentence those tokens sit. The token "dog" gets the same query whether it is the first word or the last.

### The precise statement: permutation equivariance

A **permutation** is a reordering of items. A function $f$ is **permutation-equivariant** if reordering its inputs reorders its outputs in exactly the same way and changes nothing else:

$$f(PX) = P\,f(X)$$

- $X$: the $n \times d$ input matrix ($n$ tokens, each a $d$-dimensional vector).
- $P$: a **permutation matrix**, an $n \times n$ matrix of 0s and 1s that reorders the rows of whatever it multiplies.
- $f$: self-attention (without any position information and without a mask).

Why it holds, in three steps:

1. Shuffling the rows of $X$ shuffles the rows of $Q$, $K$ and $V$ the same way (each row is computed independently: $q_i = x_i W_Q$).
2. The score matrix $S = QK^\top$ gets both its rows and its columns shuffled, but every individual score $s_{ij} = q_i \cdot k_j$ is unchanged.
3. Softmax is applied row by row, so each token's attention weights are the same numbers in a shuffled order, and the weighted sum of value vectors gives exactly the same output vector for that token.

So "dog" in "dog bites man" and "dog" in "man bites dog" get **identical** output vectors. The outputs are just listed in a different order.

### See it in code

```python
# pip install numpy
import numpy as np

rng = np.random.default_rng(0)
d = 4
emb = {w: rng.normal(size=d) for w in ["dog", "bites", "man"]}   # toy word embeddings
W_q, W_k, W_v = (rng.normal(size=(d, d)) for _ in range(3))       # toy projections

def self_attention(X):
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    S = Q @ K.T / np.sqrt(d)
    A = np.exp(S - S.max(axis=1, keepdims=True))   # numerically stable softmax
    A /= A.sum(axis=1, keepdims=True)
    return A @ V

X1 = np.stack([emb[w] for w in ["dog", "bites", "man"]])
X2 = np.stack([emb[w] for w in ["man", "bites", "dog"]])
out1, out2 = self_attention(X1), self_attention(X2)

print(np.allclose(out1[0], out2[2]))  # True: "dog" gets the same vector in both sentences
print(np.allclose(out1[2], out2[0]))  # True: so does "man"
print(np.allclose(out1[1], out2[1]))  # True: and "bites"
```

The model literally cannot tell who bit whom. Stacking more layers does not help, because every layer has the same property (the feed-forward sublayer, residual connections and layer normalization all act on each token independently, so they preserve equivariance too).

### Why is this a feature and a bug?

- **Feature:** equivariance is why attention is so parallel and general. The same weights work for any length, and the computation does not need to walk the sequence step by step like an RNN (recap: [RNNs, LSTMs and the Road to Transformers](../02-deep-learning/08-rnns-lstms-and-the-road-to-transformers.md)). An RNN gets order for free from processing tokens one at a time; a Transformer gives that up for speed.
- **Bug:** language, code, audio and time series are all ordered. Meaning depends on order. So we must inject position information somehow.

> **Senior insight:** A **causal mask** (each token may only attend to earlier tokens, used by GPT-style decoders, see [Multi-Head Attention and Causal Masking](04-multi-head-attention-and-causal-masking.md)) already breaks full equivariance: token 1 sees one token, token 100 sees 100. Researchers have shown decoder-only models trained with **no positional encoding at all** ("NoPE") can still learn position information from this asymmetry (Haviv et al., 2022, "Transformer Language Models without Positional Encodings Still Learn Positional Information"; Kazemnejad et al., 2023, "The Impact of Positional Encoding on Length Generalization in Transformers"). Encoders like BERT, which use no causal mask, have no such escape and absolutely need explicit positions. In practice nearly every production LLM still uses an explicit method, most often RoPE.

## The design space: absolute vs relative, and where position enters

Before looking at specific methods, here are the two questions every method answers.

**Question 1: absolute or relative?**

- An **absolute positional encoding** tells each token "you are at position 7". The analogy is a seat number in a theater.
- A **relative positional encoding** tells a pair of tokens "you are 3 apart". The analogy is "the person two seats to my left".

Language mostly cares about relative position: "the adjective right before this noun" matters regardless of whether the phrase starts at word 5 or word 500. Absolute position matters a little (the start of a document is special), but relative structure dominates. This is the main reason the field moved from absolute to relative methods.

**Question 2: where does position information enter the network?**

```mermaid
flowchart LR
    T[Token IDs] --> E[Token embeddings]
    E --> ADD["+ position vector<br/>(sinusoidal or learned)"]
    ADD --> PROJ["Project to Q, K, V<br/>in every layer"]
    PROJ --> ROT["Rotate Q and K by position<br/>(RoPE, every layer)"]
    ROT --> SC["Scores QK^T / sqrt(d_k)"]
    SC --> BIAS["+ distance bias<br/>(ALiBi, T5 bias, every layer)"]
    BIAS --> SM[Softmax, weighted sum of V]
```

| Where | Methods | Effect |
|---|---|---|
| Added to the input embeddings, once, before layer 1 | Sinusoidal, learned absolute | Position is mixed into the token vector and must survive through every layer |
| Applied to queries and keys, in every attention layer | RoPE | Position shows up directly in every attention score as a relative offset |
| Added to attention scores, in every attention layer | ALiBi, T5 relative bias, Shaw et al. relative | Scores are nudged up or down based on distance |

Keep this table in mind; the rest of the chapter walks down it.

## Sinusoidal positional encoding

This is the method from the original Transformer paper ("Attention Is All You Need", Vaswani et al., 2017).

### Intuition: a clock with many hands

> **Analogy:** A clock shows time with several hands moving at different speeds. The second hand spins fast and tells you fine detail; the hour hand creeps and tells you the big picture. Read all hands together and you know the exact time. Sinusoidal encoding gives each position a "reading" of many hands spinning at different speeds. Fast hands distinguish neighbors (position 5 vs 6). Slow hands distinguish far-apart regions (position 50 vs 5,000).

Another way to see it: binary counting. The lowest bit flips every step, the next bit every 2 steps, the next every 4 steps. Sinusoids are a smooth, continuous version of that idea.

### The formula

For position $pos$ and dimension index $i$, the encoding is a vector of length $d_{\text{model}}$ whose even entries use sine and odd entries use cosine:

$$PE_{(pos,\,2i)} = \sin\!\left(\omega_i \cdot pos\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\omega_i \cdot pos\right), \qquad \omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}$$

- $pos$: the token's position in the sequence, starting at 0.
- $d_{\text{model}}$: the embedding size (the length of every token vector).
- $i$: the index of a sine/cosine **pair**, from $0$ to $d_{\text{model}}/2 - 1$. Dimensions $2i$ and $2i+1$ form pair $i$.
- $\omega_i$: the **angular frequency** of pair $i$, in radians per position. Pair 0 has $\omega_0 = 1$ (fastest); the last pair has $\omega \approx 1/10000$ (slowest).
- $10000$: the **base**, a hand-picked constant that sets how slow the slowest hand is.
- Each pair's **wavelength** (how many positions until it repeats) is $2\pi / \omega_i$, ranging from $2\pi \approx 6.28$ positions to $2\pi \times 10000 \approx 62{,}832$ positions (approximately, for large $d_{\text{model}}$).

The resulting vector is **added** to the token embedding: $x_{pos} = \text{embed}(\text{token}) + PE_{pos}$. (The original paper also multiplies the token embedding weights by $\sqrt{d_{\text{model}}}$; a common explanation is that this keeps the token signal from being drowned out by the position signal.)

### Worked example by hand: $d_{\text{model}} = 4$

With 4 dimensions there are 2 pairs ($i = 0, 1$).

Step 1, the frequencies:

- $\omega_0 = 1 / 10000^{0/4} = 1 / 1 = 1$
- $\omega_1 = 1 / 10000^{2/4} = 1 / 10000^{0.5} = 1/100 = 0.01$

Step 2, so $PE_{pos} = [\sin(pos),\ \cos(pos),\ \sin(0.01\,pos),\ \cos(0.01\,pos)]$ (angles in radians).

Step 3, plug in positions 0 to 3:

| pos | dim 0: $\sin(pos)$ | dim 1: $\cos(pos)$ | dim 2: $\sin(0.01\,pos)$ | dim 3: $\cos(0.01\,pos)$ |
|---|---|---|---|---|
| 0 | $\sin 0 = 0$ | $\cos 0 = 1$ | $0$ | $1$ |
| 1 | $\sin 1 = 0.8415$ | $\cos 1 = 0.5403$ | $\sin 0.01 = 0.0100$ | $\cos 0.01 = 1.0000$ |
| 2 | $\sin 2 = 0.9093$ | $\cos 2 = -0.4161$ | $\sin 0.02 = 0.0200$ | $\cos 0.02 = 0.9998$ |
| 3 | $\sin 3 = 0.1411$ | $\cos 3 = -0.9900$ | $\sin 0.03 = 0.0300$ | $\cos 0.03 = 0.9996$ |

Notice:

- Dims 0 and 1 (the fast pair) change a lot from one position to the next. They tell neighbors apart.
- Dims 2 and 3 (the slow pair) barely move over 4 positions. They would only become useful for telling position 10 apart from position 300.
- Every value stays between -1 and 1, no matter how large $pos$ gets. Compare that with the naive idea of just appending the number $pos$ itself, which would blow up to 100,000 for long documents and dwarf the word meaning.

### A wider example: $d_{\text{model}} = 8$

With 8 dimensions the four frequencies are $1,\ 0.1,\ 0.01,\ 0.001$ (each pair is 10 times slower than the previous one). Values rounded to 4 decimals:

| pos | pair 0 ($\omega=1$) | pair 1 ($\omega=0.1$) | pair 2 ($\omega=0.01$) | pair 3 ($\omega=0.001$) |
|---|---|---|---|---|
| 0 | 0, 1 | 0, 1 | 0, 1 | 0, 1 |
| 1 | 0.8415, 0.5403 | 0.0998, 0.9950 | 0.0100, 1.0000 | 0.0010, 1.0000 |
| 2 | 0.9093, -0.4161 | 0.1987, 0.9801 | 0.0200, 0.9998 | 0.0020, 1.0000 |
| 10 | -0.5440, -0.8391 | 0.8415, 0.5403 | 0.0998, 0.9950 | 0.0100, 1.0000 |
| 50 | -0.2624, 0.9650 | -0.9589, 0.2837 | 0.4794, 0.8776 | 0.0500, 0.9988 |

A cute pattern: pair 1 at position 10 equals pair 0 at position 1 (both are $\sin 1, \cos 1$). Each slower pair is a "zoomed out" copy of the faster one. Fast pairs have already wrapped around by position 10, which is why you need slow pairs to keep far positions distinct.

### The hidden gift: relative offsets are rotations

For any pair $i$ and any offset $k$, the encoding at $pos + k$ is a fixed rotation of the encoding at $pos$:

$$\begin{bmatrix} \sin(\omega_i (pos+k)) \\ \cos(\omega_i (pos+k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} \sin(\omega_i\, pos) \\ \cos(\omega_i\, pos) \end{bmatrix}$$

This follows from the angle-addition identities $\sin(a+b) = \sin a \cos b + \cos a \sin b$ and $\cos(a+b) = \cos a \cos b - \sin a \sin b$. The rotation matrix depends only on $k$, not on $pos$. Vaswani et al. chose sinusoids partly because they hoped this would let the model "easily learn to attend by relative positions".

A consequence: the dot product of two position vectors depends only on their distance.

$$PE_{p} \cdot PE_{p+k} = \sum_{i} \left[\sin(\omega_i p)\sin(\omega_i (p+k)) + \cos(\omega_i p)\cos(\omega_i (p+k))\right] = \sum_i \cos(\omega_i k)$$

Check with our $d_{\text{model}} = 4$ table, positions 1 and 2 ($k = 1$):

- Direct: $0.8415 \times 0.9093 + 0.5403 \times (-0.4161) + 0.0100 \times 0.0200 + 1.0000 \times 0.9998$
- $= 0.7652 - 0.2248 + 0.0002 + 0.9998 = 1.5404$
- Formula: $\cos(1) + \cos(0.01) = 0.5403 + 1.0000 = 1.5403$ (equal up to rounding).

And for $k = 2$: $\cos 2 + \cos 0.02 = -0.4161 + 0.9998 = 0.5837$. Closer positions have more similar encodings.

> **Common mistake:** Concluding that sinusoidal encoding makes attention relative. It does not. Inside attention the score is $(x_i + p_i) W_Q W_K^\top (x_j + p_j)^\top$. Expanding gives four terms (content-content, content-position, position-content, position-position), and the learned matrices sit between the position vectors, so even the position-position term is not guaranteed to depend only on $i - j$. The relative structure is available, but the model has to learn to exploit it. RoPE was designed to make it exact.

### Why add instead of concatenate?

Adding the position vector to the token vector seems to "mix" two kinds of information destructively. In practice it works because high-dimensional vectors have lots of room: the model can learn to keep position information in a subspace that word meaning mostly does not use. Concatenating would also work but would grow every vector (and every weight matrix), costing parameters and compute. Adding is the standard choice.

### Code: sinusoidal encoding from scratch

```python
import numpy as np

def sinusoidal_pe(max_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Return a (max_len, d_model) matrix of sinusoidal position encodings."""
    pos = np.arange(max_len)[:, None]                  # (max_len, 1)
    i = np.arange(d_model // 2)[None, :]               # (1, d_model/2)
    omega = base ** (-2 * i / d_model)                 # frequencies, fastest first
    angles = pos * omega                               # (max_len, d_model/2)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles)                       # even dims: sine
    pe[:, 1::2] = np.cos(angles)                       # odd dims: cosine
    return pe

print(np.round(sinusoidal_pe(4, 4), 4))
# [[ 0.      1.      0.      1.    ]
#  [ 0.8415  0.5403  0.01    1.    ]
#  [ 0.9093 -0.4161  0.02    0.9998]
#  [ 0.1411 -0.99    0.03    0.9996]]

pe = sinusoidal_pe(1000, 64)
print(round(pe[10] @ pe[13], 4), round(pe[500] @ pe[503], 4))  # same value: depends only on the offset 3
```

Adding this to our "dog bites man" example breaks the symmetry: `self_attention(X1 + sinusoidal_pe(3, 4))` no longer gives "dog" the same vector as in "man bites dog", because "dog" now carries a different position vector in each sentence.

## Learned absolute position embeddings

### Intuition

> **Analogy:** Instead of computing a seat's "reading" from a clock formula, give every seat number its own name tag, and let the theater staff (gradient descent) write on the tags whatever turns out to be useful.

A **learned position embedding** is a trainable lookup table with one vector per position, exactly like a token embedding table but indexed by position instead of token ID (recap: [Embeddings](02-embeddings-from-one-hot-to-contextual.md)).

$$x_{pos} = E_{\text{token}}[\text{id}_{pos}] + E_{\text{pos}}[pos]$$

- $E_{\text{token}}$: the token embedding table, shape $V \times d_{\text{model}}$ ($V$ = vocabulary size).
- $E_{\text{pos}}$: the position embedding table, shape $L_{\max} \times d_{\text{model}}$.
- $L_{\max}$: the **maximum sequence length**, fixed before training.
- $\text{id}_{pos}$: the token ID at position $pos$.

BERT (Devlin et al., 2019) uses 512 learned positions; GPT-2 uses 1,024; GPT-3 uses 2,048. Vision Transformers also use learned position embeddings for image patches (see [Vision Transformers, CLIP and Multimodal LLMs](../03-computer-vision/05-vision-transformers-clip-and-multimodal-models.md)).

Parameter cost is small: for BERT-base, $512 \times 768 = 393{,}216$ parameters, a fraction of a percent of its roughly 110M.

### Code: learned positions in PyTorch

```python
# pip install torch
import torch
import torch.nn as nn

class LearnedPositions(nn.Module):
    def __init__(self, max_len: int = 512, d_model: int = 768):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)    # one trainable vector per position

    def forward(self, tok_emb: torch.Tensor) -> torch.Tensor:
        # tok_emb: (batch, seq, d_model)
        seq = tok_emb.shape[1]
        idx = torch.arange(seq, device=tok_emb.device)
        return tok_emb + self.pos(idx)               # fails if seq > max_len: no row 512 exists

x = torch.randn(2, 10, 768)
print(LearnedPositions()(x).shape)  # torch.Size([2, 10, 768])
```

And to see the limit in a real model's config:

```python
# pip install transformers
from transformers import AutoConfig

print(AutoConfig.from_pretrained("bert-base-uncased").max_position_embeddings)  # 512
print(AutoConfig.from_pretrained("gpt2").n_positions)                            # 1024
```

### What learned embeddings end up looking like

When people visualize trained position tables (for example from GPT-2 or BERT), nearby positions usually end up with similar vectors and smooth, wave-like patterns appear on their own. Gradient descent rediscovers something sinusoid-like. Two practical gotchas:

- **Rare positions are undertrained.** If most training sequences are short, the rows for high positions get few gradient updates and stay noisy.
- **Hard ceiling.** Position 513 in a 512-position BERT has no row. The code above raises an index error on CPU (and a device-side assert on GPU). This is why "BERT can only take 512 tokens" is a real constraint, and why long documents must be chunked for BERT-style models.

## Sinusoidal vs learned, side by side

| | Sinusoidal (fixed) | Learned absolute |
|---|---|---|
| What it is | A formula of sines and cosines at many frequencies | A trainable table, one vector per position |
| Extra parameters | 0 | $L_{\max} \times d_{\text{model}}$ |
| Works beyond training length? | Defined for any position, but quality degrades in practice | No: positions beyond $L_{\max}$ do not exist |
| Relative structure | Built in mathematically (offsets are rotations) | Only what the model happens to learn |
| Flexibility | Fixed; cannot adapt to data | Can learn task-specific patterns (for example, "position 0 is special") |
| Quality at trained lengths | Vaswani et al. 2017 reported nearly identical results to learned | About the same |
| Used in | Original Transformer, some early seq2seq models | BERT, RoBERTa, GPT-2, GPT-3, OPT, ViT |
| Main weakness | Poor length extrapolation despite the hope | Hard length limit, undertrained tail positions |

> **Interview tip:** The crisp answer is: "At trained lengths they perform about the same. Sinusoidal costs no parameters and is defined for any position, which was the original hope for extrapolation, but in practice it does not extrapolate well either. Learned embeddings are more flexible but have a hard maximum length. Both are absolute and both were largely replaced by relative methods, mainly RoPE."

## Relative position ideas

Since what matters for language is mostly distance, why not give attention the distance directly? Several influential designs did exactly that, and they set the stage for RoPE and ALiBi.

1. **Shaw et al., 2018** ("Self-Attention with Relative Position Representations"): learn a vector for each clipped relative distance (for example, -16 to +16, with anything farther clipped to the edge value) and add it into the key (and value) when computing attention between tokens $i$ and $j$.
2. **Transformer-XL** (Dai et al., 2019): rewrote the attention score into content and relative-position terms using sinusoidal relative encodings plus learned global bias vectors, so the model could process long text in segments.
3. **T5 relative bias** (Raffel et al., 2020, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"): add a learned **scalar** to each attention score, chosen by the distance $j - i$. Distances are grouped into **buckets**: small distances get their own bucket, larger ones share logarithmically wider buckets (in the standard T5 config, 32 buckets, split between the two directions in the bidirectional encoder, with everything 128 or more tokens away sharing the last bucket). Each head has its own bias values, shared across layers.
4. **DeBERTa** (He et al., 2020): "disentangled attention" keeps separate content and relative-position vectors and computes content-to-position and position-to-content terms explicitly.

The common pattern: position stops being a property of a single token and becomes a property of a **pair** of tokens. Two problems held these methods back at scale:

- Some (Shaw, Transformer-XL, DeBERTa) need extra per-pair computation that fights with highly optimized attention kernels.
- Learned relative tables or buckets still have a range: distances never seen in training have nothing meaningful learned for them.

RoPE and ALiBi solve this more elegantly: relative position with **zero learned parameters** and minimal extra compute.

## RoPE: Rotary Position Embedding

RoPE was introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su, Lu, Pan, Murtadha, Wen and Liu, 2021). It is used by GPT-J, GPT-NeoX and PaLM, and is the default in most open LLMs as of 2026: the Llama family, Mistral, Qwen, Gemma, DeepSeek and many others.

### Intuition: rotating arrows

> **Analogy:** Give every token an arrow on a dial. Before comparing two tokens, turn each arrow by an amount proportional to the token's position: token 1 turns 30°, token 2 turns 60°, token 5 turns 150°. Now compare two arrows by the angle between them. If you shift the whole sentence by 100 positions, both arrows turn by an extra 3,000°, and **the angle between them does not change**. That is RoPE's trick: absolute turning, relative comparison.

### Primer: rotating a 2D vector

A **rotation matrix** turns a 2D vector by angle $\alpha$ counterclockwise without changing its length:

$$R(\alpha) = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix}, \qquad R(\alpha)\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} x_1\cos\alpha - x_2\sin\alpha \\ x_1\sin\alpha + x_2\cos\alpha \end{bmatrix}$$

- $\alpha$: the rotation angle.
- $x_1, x_2$: the two coordinates of the vector being rotated.

Two facts we need (recap: [Linear Algebra for ML](../00-foundations/02-linear-algebra-for-ml.md)):

1. **Rotations compose by adding angles:** $R(a)\,R(b) = R(a+b)$.
2. **Undoing a rotation is its transpose:** $R(a)^\top = R(-a)$.

### The RoPE rule in 2D

After computing a query $q$ for the token at position $m$ and a key $k$ for the token at position $n$ (the usual $q = xW_Q$, $k = xW_K$), rotate each by its position times a fixed angle $\theta$:

$$\tilde{q}_m = R(m\theta)\,q, \qquad \tilde{k}_n = R(n\theta)\,k$$

The attention score uses the rotated versions:

$$\tilde{q}_m^\top \tilde{k}_n = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R(-m\theta) R(n\theta)\, k = q^\top R\big((n-m)\theta\big)\, k$$

- $m, n$: positions of the query token and key token.
- $\theta$: the rotation speed (radians per position).
- The last expression contains only the **difference** $n - m$. The absolute positions have vanished.

That one line is the whole reason RoPE works: **the score between two tokens depends on their content ($q$, $k$) and their relative distance, and on nothing else about position.**

### Worked example by hand

Let us use an illustrative $\theta = 30°$ per position (round numbers for hand arithmetic; in real RoPE the fastest pair turns 1 radian, about 57.3°, per position).

Query $q = (1, 2)$ belongs to the token at position $m = 2$. Key $k = (3, 1)$ belongs to the token at position $n = 1$.

Useful values: $\cos 30° = 0.86603$, $\sin 30° = 0.5$, $\cos 60° = 0.5$, $\sin 60° = 0.86603$.

**Step 0, the unrotated score:** $q \cdot k = 1 \times 3 + 2 \times 1 = 5$.

**Step 1, rotate the query by $m\theta = 60°$:**

- first coordinate: $1 \times 0.5 - 2 \times 0.86603 = 0.5 - 1.73205 = -1.23205$
- second coordinate: $1 \times 0.86603 + 2 \times 0.5 = 0.86603 + 1 = 1.86603$
- $\tilde{q}_2 = (-1.23205,\ 1.86603)$

**Step 2, rotate the key by $n\theta = 30°$:**

- first coordinate: $3 \times 0.86603 - 1 \times 0.5 = 2.59808 - 0.5 = 2.09808$
- second coordinate: $3 \times 0.5 + 1 \times 0.86603 = 1.5 + 0.86603 = 2.36603$
- $\tilde{k}_1 = (2.09808,\ 2.36603)$

**Step 3, the rotated score:**

$$\tilde{q}_2 \cdot \tilde{k}_1 = (-1.23205)(2.09808) + (1.86603)(2.36603) = -2.58494 + 4.41508 = 1.83014 \approx 1.8301$$

**Step 4, check it only depends on $n - m = -1$:** rotate $k$ by $-30°$ and dot with the *unrotated* $q$:

- $R(-30°)\,k = (3 \times 0.86603 + 1 \times 0.5,\ -3 \times 0.5 + 1 \times 0.86603) = (3.09808,\ -0.63397)$
- $q \cdot R(-30°)k = 1 \times 3.09808 + 2 \times (-0.63397) = 3.09808 - 1.26795 = 1.83013 \approx 1.8301$. Same score.

**Step 5, shift the whole pair by 3 positions** ($m = 5$, $n = 4$, angles 150° and 120°):

- $\tilde{q}_5 = R(150°)\,q = (1 \times (-0.86603) - 2 \times 0.5,\ 1 \times 0.5 + 2 \times (-0.86603)) = (-1.86603,\ -1.23205)$
- $\tilde{k}_4 = R(120°)\,k = (3 \times (-0.5) - 1 \times 0.86603,\ 3 \times 0.86603 + 1 \times (-0.5)) = (-2.36603,\ 2.09808)$
- Score: $(-1.86603)(-2.36603) + (-1.23205)(2.09808) = 4.41508 - 2.58494 = 1.83014 \approx 1.8301$. Unchanged.

The vectors themselves are completely different from Step 1 and Step 2, but the score is identical because the gap is still 1. Here is how the score changes with the gap (same $q$ and $k$):

| Query pos $m$ | Key pos $n$ | Gap $n - m$ | Score |
|---|---|---|---|
| 2 | 2 | 0 | 5.0000 (rotations cancel, equals $q \cdot k$) |
| 2 | 1 | -1 | 1.8301 |
| 5 | 4 | -1 | 1.8301 |
| 2 | 0 | -2 | -1.8301 |
| 0 | 2 | +2 | 6.8301 |

Note that $+2$ and $-2$ give different scores: RoPE knows direction ("before" vs "after"), not just distance.

### From 2D to $d$ dimensions

Real query and key vectors have $d$ dimensions per head (for example, 128 in Llama-style 7B models). RoPE splits them into $d/2$ pairs and rotates **each pair at its own speed**, using the same frequency ladder as sinusoidal encoding:

$$\theta_i = b^{-2i/d}, \qquad i = 0, 1, \ldots, \tfrac{d}{2} - 1$$

- $\theta_i$: rotation speed of pair $i$ in radians per position.
- $b$: the **RoPE base**, 10,000 in the original paper (exposed as `rope_theta` in Hugging Face configs).
- $d$: the head dimension.

The full rotation is a **block-diagonal** matrix (a big matrix that is zero everywhere except small blocks on its diagonal), with one $2 \times 2$ rotation block per pair:

$$R_m = \begin{bmatrix} R(m\theta_0) & & \\ & \ddots & \\ & & R(m\theta_{d/2-1}) \end{bmatrix}$$

Since every block preserves the relative property, so does the whole thing: $\tilde{q}_m^\top \tilde{k}_n = q^\top R_{n-m}\, k$.

For $d = 128$ and $b = 10{,}000$:

- Pair 0: $\theta_0 = 1$ radian per token. It completes a full turn every $2\pi \approx 6.3$ tokens: a "second hand" for local word order.
- Pair 63: $\theta_{63} = 10000^{-126/128} \approx 0.000115$ radians per token. A full turn takes about 54,400 tokens: an "hour hand" for coarse distance.

### Key properties (know these for interviews)

1. **Relative by construction.** Shown above.
2. **No parameters.** Just cosines and sines of fixed angles.
3. **Applied to Q and K only, in every layer, after projection.** Values $V$ are not rotated: position should affect *which* tokens you attend to, not *what content* you copy. Applying it in every layer means position is never "washed out" by depth, unlike an encoding added once at the input.
4. **Norm-preserving.** Rotation does not change vector length, so it does not disturb the scale of attention scores.
5. **Long-term decay (soft).** The RoFormer paper argues that an upper bound on the score shrinks as relative distance grows, because many pairs rotate out of alignment. It is a tendency, not a hard rule.
6. **Compatible with fast kernels and the KV cache.** It is an elementwise operation on Q and K before attention, so FlashAttention (a fused GPU kernel that computes exact attention without materializing the full score matrix) and other kernels need no changes. Keys are rotated **before** they are stored in the KV cache (the stored keys and values of earlier tokens that generation reuses instead of recomputing; see [The KV Cache](../06-llm-inference/02-kv-cache-explained.md)), so each cached key is rotated once and never again.

### Implementation: the efficient form

Nobody builds the block-diagonal matrix. For a pair $(x_{2i}, x_{2i+1})$ with angle $\alpha$, the rotation is just:

$$\text{out}_{2i} = x_{2i}\cos\alpha - x_{2i+1}\sin\alpha, \qquad \text{out}_{2i+1} = x_{2i}\sin\alpha + x_{2i+1}\cos\alpha$$

That is two elementwise multiplications and an add, $O(n \cdot d)$ work, which is negligible next to attention itself.

```python
import numpy as np

def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotate each pair (x[2i], x[2i+1]) of every row by angle position * theta_i.
    x: (seq, d) with d even; positions: (seq,)"""
    seq, d = x.shape
    theta = base ** (-np.arange(0, d, 2) / d)           # (d/2,) speeds, fastest first
    angles = positions[:, None] * theta[None, :]        # (seq, d/2)
    cos, sin = np.cos(angles), np.sin(angles)
    x_even, x_odd = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out

rng = np.random.default_rng(0)
q = rng.normal(size=(1, 64))
k = rng.normal(size=(1, 64))

def score(m: int, n: int) -> float:
    return (rope(q, np.array([m])) @ rope(k, np.array([n])).T).item()

# Same gap of 3 at very different absolute positions -> same score
print(np.isclose(score(5, 2), score(105, 102)), np.isclose(score(5, 2), score(1005, 1002)))  # True True
# Rotation preserves length
print(np.isclose(np.linalg.norm(rope(q, np.array([7]))), np.linalg.norm(q)))  # True
```

### The PyTorch / Hugging Face style ("rotate half")

Production code (for example the Llama implementation in Hugging Face `transformers`) pairs dimension $j$ with dimension $j + d/2$ instead of neighbors $2i, 2i+1$. It is the same math with a different pairing, written with a `rotate_half` helper:

```python
import torch
import torch.nn.functional as F

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rope_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0, scale: float = 1.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
    pos = torch.arange(seq_len).float() / scale     # scale > 1 gives position interpolation (later)
    freqs = torch.outer(pos, inv_freq)              # (seq, head_dim/2), computed in FP32
    emb = torch.cat((freqs, freqs), dim=-1)         # (seq, head_dim): pairs are (j, j + head_dim/2)
    return emb.cos(), emb.sin()

def apply_rope(q, k, cos, sin):
    # q, k: (batch, heads, seq, head_dim); cos, sin broadcast over batch and heads
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

batch, heads, seq, head_dim = 2, 4, 16, 64
q, k, v = (torch.randn(batch, heads, seq, head_dim) for _ in range(3))
cos, sin = rope_cos_sin(seq, head_dim)
q_rot, k_rot = apply_rope(q, k, cos, sin)
out = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True)  # V is NOT rotated
print(out.shape)  # torch.Size([2, 4, 16, 64])
```

> **Senior insight:** The "interleaved" layout (pairs $2i, 2i+1$, used in the RoFormer paper, GPT-J and Meta's original Llama code) and the "half-split" layout (pairs $j, j + d/2$, used by Hugging Face Llama) are both valid, but a checkpoint trained with one will produce garbage if loaded with the other unless the $W_Q$ and $W_K$ columns are permuted to match. Conversion scripts do exactly this permutation. If you ever port weights between frameworks and get fluent-looking nonsense or very high **perplexity** (the exponential of the average next-token cross-entropy loss: roughly how many tokens the model is torn between at each step, so lower is better; see [Perplexity](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md#perplexity)), check the RoPE layout first.

### Variants you may hear about

- **Partial rotary:** rotate only a fraction of each head's dimensions and leave the rest position-free (GPT-J and GPT-NeoX do this). Some models also interleave layers with RoPE and layers with no positional encoding.
- **Different bases per layer type:** some recent models use a small base on sliding-window (local) attention layers and a large base on global layers (Gemma 3 reports this).
- **Multimodal RoPE:** for images and video, rotate different pairs by row, column and time indices (for example Qwen2-VL's "M-RoPE"; 2D RoPE is also used in some vision transformers).
- **Decoupled RoPE:** DeepSeek-V2's Multi-head Latent Attention compresses keys into a small latent vector to shrink the KV cache; because a position-dependent rotation would block that compression trick, it carries position in a small separate RoPE component. See [Modern LLM Architecture](08-modern-llm-architecture-improvements.md).

## ALiBi: Attention with Linear Biases

ALiBi comes from "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (Press, Smith and Lewis, ICLR 2022). It is used in BLOOM and MPT.

### Intuition: hearing fades with distance

> **Analogy:** In a crowded room, you hear people near you clearly and people far away faintly. ALiBi tells each attention head: "penalize every token by how far away it is, and the penalty grows in a straight line". Some heads have sharp ears for nearby voices only (steep penalty); others can hear across the room (gentle penalty).

### The formula

ALiBi adds **no position vectors at all**. Instead, for a causal model, it subtracts a distance penalty from each attention score before softmax:

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m_h \cdot (i - j), \qquad j \le i$$

- $i$: the query's position; $j$: the key's position (only $j \le i$ because of the causal mask).
- $i - j$: the distance from the query back to the key (0 for the token itself).
- $m_h$: a fixed, **head-specific slope**. It is not learned.
- The penalty is a **bias**: a number added to the score, not multiplied.

**The slopes.** For $h$ heads, the paper uses a geometric sequence starting at $2^{-8/h}$ with that same ratio. For 8 heads that is:

$$m = \tfrac{1}{2},\ \tfrac{1}{4},\ \tfrac{1}{8},\ \tfrac{1}{16},\ \tfrac{1}{32},\ \tfrac{1}{64},\ \tfrac{1}{128},\ \tfrac{1}{256}$$

The steep head ($m = 1/2$) is strongly local; the gentle head ($m = 1/256$) barely penalizes even 100 tokens back. Together they cover many ranges, like RoPE's ladder of frequencies. (For head counts that are not powers of 2, the reference code combines slopes from the two nearest powers of 2.)

### Worked example by hand

Four tokens, query at position $i = 3$, head with slope $m = 1/2$. Suppose (illustratively) all four raw scores $q_3 \cdot k_j / \sqrt{d_k}$ equal 1.0, so content alone would give uniform attention of 0.25 each.

| Key $j$ | Distance $3 - j$ | Bias $-0.5 \times$ distance | Score $1.0 + $ bias | $e^{\text{score}}$ | Weight |
|---|---|---|---|---|---|
| 0 | 3 | -1.5 | -0.5 | 0.6065 | 0.6065 / 5.9735 = 0.1015 |
| 1 | 2 | -1.0 | 0.0 | 1.0000 | 1.0000 / 5.9735 = 0.1674 |
| 2 | 1 | -0.5 | 0.5 | 1.6487 | 1.6487 / 5.9735 = 0.2760 |
| 3 | 0 | 0 | 1.0 | 2.7183 | 2.7183 / 5.9735 = 0.4551 |

The exponentials sum to $0.6065 + 1.0000 + 1.6487 + 2.7183 = 5.9735$, and the weights sum to $1.0000$. With identical content, the nearest token now gets 45.5% of the attention and the farthest 10.2%. With slope $1/256$ instead, the biases would be at most $-3/256 \approx -0.012$ and the weights would stay almost uniform.

### Code: ALiBi bias

```python
import numpy as np

def alibi_slopes(n_heads: int) -> np.ndarray:
    """Geometric slopes from the ALiBi paper (exact for power-of-2 head counts)."""
    start = 2 ** (-8 / n_heads)
    return np.array([start ** (h + 1) for h in range(n_heads)])

def alibi_bias(n_heads: int, seq_len: int) -> np.ndarray:
    """(heads, seq, seq) bias to add to attention scores, with the causal mask included."""
    pos = np.arange(seq_len)
    rel = pos[None, :] - pos[:, None]                  # j - i: 0 on diagonal, negative for past keys
    bias = alibi_slopes(n_heads)[:, None, None] * rel  # = -m * (i - j)
    return np.where(rel[None] <= 0, bias, -np.inf)     # future keys masked out

print(alibi_slopes(8))
# [0.5        0.25       0.125      0.0625     0.03125    0.015625
#  0.0078125  0.00390625]
print(alibi_bias(8, 4)[0])  # head 0, slope 0.5
# [[ 0.  -inf -inf -inf]
#  [-0.5  0.  -inf -inf]
#  [-1.  -0.5  0.  -inf]
#  [-1.5 -1.  -0.5  0. ]]
```

### Strengths and weaknesses

- **Strength: extrapolation of perplexity.** The paper showed models trained on short sequences keep stable perplexity on much longer sequences, while sinusoidal models degrade. Nothing in ALiBi is ever "unseen": a distance of 5,000 just means a bigger penalty.
- **Strength: zero parameters, trivial to implement.** It is one extra additive bias matrix.
- **Weakness: built-in recency bias.** The linear penalty makes far tokens progressively harder to attend to. Stable perplexity on long text does not mean the model can retrieve a fact from 50,000 tokens back; ALiBi "extrapolates" partly by effectively ignoring distant context. Perplexity is dominated by nearby tokens, so it hides this.
- **Weakness: kernel support.** It needs a per-head bias inside the attention kernel. Modern kernels (for example FlashAttention 2) support ALiBi, but it is less universal than RoPE.
- **Market reality:** most frontier and popular open models (2023 onwards) chose RoPE plus context-extension techniques over ALiBi.

## All methods compared

| Method | Type | Where applied | Extra params | Beyond training length | Pros | Cons | Notable users |
|---|---|---|---|---|---|---|---|
| Sinusoidal | Absolute (relative-friendly) | Added to input once | 0 | Defined but degrades | Simple, no params | Weak extrapolation, position can fade through layers | Original Transformer |
| Learned absolute | Absolute | Added to input once | $L_{\max} \times d$ | Impossible | Flexible, simple | Hard length cap | BERT, GPT-2/3, OPT, ViT |
| Shaw / T5 relative | Relative | Scores or keys, every layer | Small table | Clipped or bucketed | Captures distance directly | Extra compute or limited range | T5, DeBERTa (variant) |
| RoPE | Relative (via absolute rotation) | Q and K, every layer | 0 | Degrades without scaling; good with PI/NTK/YaRN plus tuning | Exact relative scores, kernel-friendly, cache-friendly | Needs scaling tricks to extend | Llama, Mistral, Qwen, Gemma, DeepSeek, PaLM |
| ALiBi | Relative | Score bias, every layer | 0 | Perplexity stays stable | Simplest extrapolation | Recency bias, weaker long retrieval | BLOOM, MPT |
| NoPE | Implicit via causal mask | None | 0 | Mixed results | No design choices | Encoders cannot use it; less proven at scale | Research; some layers in hybrid designs |

(Model lists reflect public descriptions up to 2026; new models may differ.)

## Why models struggle beyond their training length

Suppose our running example, a 7B Llama-style model with RoPE (base 10,000, head dimension 128), was trained on sequences of up to 4,096 tokens. You send it 16,000 tokens. Typically perplexity explodes and the output turns into repetition or nonsense. Why? There are four distinct reasons, and a senior answer names more than one.

### Reason 1: positions or angles the model has never seen

- **Learned absolute:** position 4,097 has no embedding. Hard failure.
- **Sinusoidal:** the vectors exist, but the network was never trained on them; the combination of slow-pair values at position 16,000 is out of distribution.
- **RoPE:** this is subtle and important. The fast pairs have spun around hundreds of times during training, so they have seen every angle. The slow pairs have not.

Let us compute it. The slowest pair turns $\theta_{63} = 10000^{-126/128} \approx 0.000115$ radians per token.

- Largest angle seen in training: $4096 \times 0.000115 \approx 0.473$ radians (about 27°).
- Angle at position 16,384: $16384 \times 0.000115 \approx 1.892$ radians (about 108°).

The model's weights were only ever trained with that pair's angle between 0° and 27°. At 108° the query-key interaction for those dimensions is something it has never seen. How many pairs are affected? A pair completes a full turn during training only if its wavelength $2\pi / \theta_i$ is at most 4,096 tokens. With $d = 128$, $b = 10{,}000$ that holds for pairs 0 to 45; pairs 46 to 63 (18 of the 64 pairs) never complete a full rotation in 4,096 tokens, so for them, longer inputs produce genuinely new angles. These low-frequency pairs are exactly the ones carrying long-range position information.

### Reason 2: attention distribution shift

Even with perfectly "valid" positions, more tokens change the statistics of softmax. With $n$ keys, attention weight is spread over more candidates, so the distribution tends to get flatter (its **entropy**, a measure of spread defined in [Softmax, Entropy, Cross-Entropy and KL Divergence](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md), grows roughly with $\log n$). Heads that learned to focus sharply at 4k tokens become blurrier at 32k. This is why YaRN adds a temperature correction (below), and why some research scales logits by $\log n$.

A related phenomenon: trained LLMs dump a lot of attention on the first few tokens, called **attention sinks** (Xiao et al., 2023, "Efficient Streaming Language Models with Attention Sinks"). Long-context behavior depends on these patterns staying intact.

### Reason 3: the model never learned to use long-range dependencies

Even if position math were perfect, a model trained only on 4k windows has never had to connect a question at token 15,000 with a fact at token 200. Skills like "find the definition mentioned 20 pages ago" are learned from data, and long natural documents are relatively rare in pretraining corpora. This is why every serious context-extension recipe includes training on long data.

### Reason 4 (practical, not quality): cost

Attention compute grows with $n^2$ and the KV cache grows with $n$. For Llama 2 7B (32 layers, hidden size 4,096, FP16, no grouped-query attention, i.e. every query head has its own key and value head), the KV cache is $2 \times 32 \times 4096 \times 2$ bytes $= 524{,}288$ bytes $= 0.5$ MiB **per token**, so a single 32,768-token sequence needs 16 GiB of cache on top of roughly 13 GB of FP16 weights. See [The KV Cache](../06-llm-inference/02-kv-cache-explained.md). A model can be capable of long context and still be too expensive to serve that way.

```text
Training window (4,096 tokens)
|<------------------ seen ----------------->|<------ unseen ------->|
pair 0  (fast):  ~650 full turns: all angles seen, no problem
pair 63 (slow):  0 to 27 degrees only        | 27 to 108 degrees: NEW -> out of distribution
attention:       sharp, trained statistics   | flatter, shifted statistics
data:            dependencies up to 4k       | no practice linking far tokens
```

## Extending the context window

The goal: take a model pretrained at length $L$ (say 4,096) and make it work at $L' = sL$ (say 16,384, so the **scale factor** $s = 4$) without pretraining from scratch. All the main techniques target RoPE, because that is what most modern LLMs use (a learned table would need new rows trained from nothing, and ALiBi already accepts any length).

### Method 1: Position Interpolation (PI)

"Extending Context Window of Large Language Models via Positional Interpolation" (Chen, Wong, Chen and Tian, 2023).

> **Analogy:** If your ruler only goes to 4,096 marks, do not extend the ruler; squeeze the longer document onto it. Position 16,000 is read as position 4,000.

Instead of extrapolating to new angles, scale every position down:

$$m' = \frac{m}{s} = m \cdot \frac{L}{L'}$$

- $m$: the real position; $m'$: the position fed into RoPE (no longer an integer).
- $s = L'/L$: the scale factor.

Worked example: $L = 4096$, $L' = 16384$, $s = 4$. Real position 10,000 becomes $10000 / 4 = 2500$, and the last position 16,383 becomes $4095.75$, inside the trained range. Every angle is now one the model has seen (or lies between angles it has seen).

- **Pro:** no unseen angles. The paper found that a short fine-tune (on the order of 1,000 steps) was enough to adapt LLaMA models to much longer windows, while direct extrapolation without interpolation failed badly.
- **Con:** it squeezes **all** frequencies equally, including the fast pairs. Neighboring tokens now differ by only 0.25 radians in pair 0 instead of 1 radian, so the model's fine-grained sense of local order gets blurrier. That is why PI needs fine-tuning, and why quality on short inputs can drop slightly.

### Method 2: NTK-aware scaling (change the base)

Proposed in 2023 in open-source community discussions (motivated by **Neural Tangent Kernel** theory, which suggests networks struggle to learn high-frequency details when inputs are compressed). The idea: do not squeeze the fast pairs (they are fine already); only stretch the slow pairs. A clean way to do that is to **increase the RoPE base**:

$$b' = b \cdot s^{\,d/(d-2)}$$

- $b$: original base (10,000); $b'$: new base.
- $s$: scale factor; $d$: head dimension.

Why this exponent? It makes the slowest pair slow down by exactly $s$ (the same as PI for that pair) while the fastest pair is untouched:

- Fastest pair: $\theta_0 = b'^{\,0} = 1$, unchanged.
- Slowest pair: $\theta'_{d/2-1} = b'^{-(d-2)/d} = b^{-(d-2)/d} \cdot s^{-1} = \theta_{d/2-1} / s$.

Worked example: $d = 128$, $s = 4$.

- Exponent: $128 / 126 = 1.01587$
- $s^{1.01587} = 4^{1.01587} \approx 4.089$
- $b' = 10000 \times 4.089 \approx 40{,}890$
- Fastest pair: still 1.0 radian per token (no local blurring).
- Slowest pair: max angle at 16,384 tokens $= 16384 \times 0.0000289 \approx 0.473$ radians, exactly the training maximum.

In between, the pairs are stretched by a smoothly increasing amount (pair $i$ slows by a factor $s^{2i/(d-2)}$). One catch, pointed out in the YaRN paper: those in-between slow pairs are stretched by **less** than $s$, so in our example pairs 46 to 62 still reach angles somewhat beyond their training range at 16,384 tokens (only pair 63 lands exactly on its old maximum). NTK-aware scaling works surprisingly well even **without** fine-tuning and improves with it, but because of this slight extrapolation the YaRN authors found it fine-tunes less well than PI.

**Dynamic NTK** recomputes the scale factor from the current sequence length at inference: short prompts use the original base (no quality loss), long prompts get scaled.

### Method 3: YaRN

"YaRN: Efficient Context Window Extension of Large Language Models" (Peng, Quesnelle, Fan and Shippole, 2023). YaRN combines two ideas, conceptually:

1. **Treat pairs by wavelength ("NTK-by-parts"):** compare each pair's wavelength with the original context length $L$.
   - Pairs that rotate many times within $L$ (short wavelength): **do not interpolate**. They encode local order and have seen all angles.
   - Pairs whose wavelength is longer than $L$: **interpolate fully** (divide by $s$, as in PI). These are the ones that would see new angles.
   - Pairs in between: blend smoothly with a ramp.
   - Worked example for our running model ($L = 4096$, $d = 128$, base 10,000), using the thresholds the paper suggests for Llama models (leave a pair alone if it turns more than 32 times within $L$, i.e. wavelength under 128 tokens; fully interpolate if it turns less than once, i.e. wavelength over 4,096): pairs 0 to 20 are untouched, pairs 46 to 63 (the same 18 "unseen angle" pairs from Reason 1) are divided by $s$, and pairs 21 to 45 are blended.
2. **Attention temperature:** multiply the attention logits (the raw scores $q \cdot k / \sqrt{d_k}$ before softmax) by a factor slightly above 1 to counter the softmax flattening from Reason 2. The paper's recommended factor is $\sqrt{1/t} = 0.1 \ln s + 1$ applied to both $q$ and $k$ (so the logits are scaled by $1/t$). For $s = 4$: $0.1 \times 1.386 + 1 = 1.139$, so logits are scaled by $1.139^2 \approx 1.297$.

YaRN reached good long-context quality with a small fraction of the original pretraining compute, and it is supported by mainstream inference engines. Several model families publish YaRN settings for extending their context (for example Qwen model cards describe enabling YaRN via `rope_scaling`).

### Method 4: raise the base and keep training on long data

The most robust recipe used by model builders is not a trick but continued pretraining:

- **Increase the RoPE base** (sometimes called adjusted base frequency) so all positions up to the target length fall in a well-behaved angle range, then **continue training on long sequences**. Meta's "Effective Long-Context Scaling of Foundation Models" (Xiong et al., 2023) raised the base from 10,000 to 500,000 for Llama 2 Long; Code Llama used a base of 1,000,000; Llama 3 ships with `rope_theta = 500000`.
- **Data matters as much as positions.** Upsample naturally long documents (books, long code repositories, long papers) while keeping the domain mix balanced, and train in stages (for example 8k, then 32k, then 128k) to control cost. Research such as "Data Engineering for Scaling Language Models to 128K Context" (Fu et al., 2024) found that a modest amount of well-mixed long data goes a long way.
- **Then instruction-tune with long examples**, otherwise the chat model may still ignore long inputs.
- Related methods you may see: LongRoPE (Ding et al., 2024) searches for per-dimension rescale factors; Llama 3.1 uses its own frequency-dependent RoPE rescaling (`rope_type: "llama3"` in Hugging Face configs).

### Side-by-side

| Method | What changes | Fine-tuning needed | Short-context quality | Typical use |
|---|---|---|---|---|
| Naive extrapolation | Nothing | n/a | Unchanged | Do not do it: fails beyond trained length |
| Position Interpolation | Positions divided by $s$ | Yes, short fine-tune | Slightly worse (local blur) | Simple, early 2023 extensions |
| NTK-aware (static) | Base raised, slow pairs stretched most | Optional, better with | Mostly preserved | Quick zero-shot extension |
| Dynamic NTK | Base raised only when prompt is long | No | Preserved for short prompts | Inference-time stretch |
| YaRN | Per-wavelength interpolation plus temperature | Small fine-tune recommended | Preserved | Common in open models and engines |
| Bigger base plus continued pretraining | Base, weights, data | Yes, substantial | Preserved | How vendors build 128k+ models |

### Code: how each method changes the angles

```python
import numpy as np

d, base, L_train, L_new = 128, 10000.0, 4096, 16384
s = L_new / L_train                                   # scale factor = 4
i = np.arange(d // 2)

theta = base ** (-2 * i / d)                          # original speeds
theta_pi = theta / s                                  # Position Interpolation
base_ntk = base * s ** (d / (d - 2))                  # NTK-aware: raise the base
theta_ntk = base_ntk ** (-2 * i / d)

print(f"train max angle, slowest pair: {L_train * theta[-1]:.3f} rad")
for name, th in [("original", theta), ("PI", theta_pi), ("NTK-aware", theta_ntk)]:
    print(f"{name:10s} fastest pair: {th[0]:.4f} rad/token | "
          f"slowest pair angle at {L_new}: {L_new * th[-1]:.3f} rad")
# train max angle, slowest pair: 0.473 rad
# original   fastest pair: 1.0000 rad/token | slowest pair angle at 16384: 1.892 rad
# PI         fastest pair: 0.2500 rad/token | slowest pair angle at 16384: 0.473 rad
# NTK-aware  fastest pair: 1.0000 rad/token | slowest pair angle at 16384: 0.473 rad
```

The output tells the whole story: the original model sees an unseen angle (1.892 vs 0.473), PI fixes it but slows the fastest pair 4x, NTK fixes it while leaving the fastest pair alone.

### Using scaling in practice

In Hugging Face `transformers`, RoPE scaling is read from the model config. The exact field names have changed across versions (for example `type` vs `rope_type` in 4.x, and in v5 `rope_theta` and the scaling settings were merged into a single `rope_parameters` dict, so `cfg.rope_theta` no longer exists and overwriting `cfg.rope_scaling` replaces the whole dict, dropping `rope_theta`), so treat this as a sketch and check the docs for your version:

```python
# Version-fragile sketch: check your transformers version's docs for exact field names.
from transformers import AutoConfig, AutoModelForCausalLM

name = "your-org/your-rope-model"          # placeholder
cfg = AutoConfig.from_pretrained(name)
orig_len = cfg.max_position_embeddings
yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": orig_len}
if getattr(cfg, "rope_parameters", None) is not None:      # transformers v5+
    cfg.rope_parameters = {**cfg.rope_parameters, **yarn}  # merge, so rope_theta is kept
else:                                                      # transformers 4.x
    cfg.rope_scaling = yarn
cfg.max_position_embeddings = 4 * orig_len                 # the new target length
model = AutoModelForCausalLM.from_pretrained(name, config=cfg)
```

Serving engines such as vLLM read the same config fields (and offer flags to override them); flags differ by version.

> **Senior insight:** Static scaling applies to **every** request, including short ones. If a model card says "only enable YaRN when you need long context", that is because static scaling can slightly hurt short-prompt quality. In a multi-tenant service, you may run two deployments (or two configs) rather than penalize every short request for the rare long one.

## Context window vs long-context use

This is the distinction interviewers love, because it separates people who read announcements from people who ship systems.

### Two different properties

- **Context window (capacity):** the maximum number of tokens the model can accept and still produce coherent, low-perplexity text. Extension methods above raise this.
- **Effective long-context use (capability):** whether the model can actually **find, combine and reason over** information anywhere in that window. This depends on training data, instruction tuning and architecture, and it is usually lower than the advertised window.

> **Analogy:** Extending the context window is giving someone a bigger desk. Improving long-context use is teaching them to find and connect the right papers on that desk. A bigger desk covered in papers does not make anyone a better researcher.

### Evidence that they differ

- **Perplexity can mislead.** Low perplexity on long documents mostly reflects predicting each token from its nearby context. A model can score well while ignoring everything more than a few thousand tokens back (the ALiBi recency effect is an example).
- **"Lost in the middle"** (Liu et al., 2023, "Lost in the Middle: How Language Models Use Long Contexts"): on multi-document question answering, accuracy was highest when the relevant document was at the beginning or end of the context and dropped when it was in the middle, a U-shaped curve. Longer windows did not remove this effect.
- **Needle in a haystack (NIAH):** insert one out-of-place fact ("the secret code is 7421") at various depths of a long filler document and ask for it. Many models now score near-perfectly. But NIAH is the easiest possible long-context task: a single fact, often with literal word overlap between question and needle, and no reasoning.
- **Harder benchmarks** such as RULER (Hsieh et al., 2024, "RULER: What's the Real Context Size of Your Long-Context Language Models?") add multiple needles, multi-hop tracing of variables, aggregation (count the most frequent words) and question answering. Models that pass NIAH often degrade well before their advertised length on these; the RULER authors describe an "effective" context length that is frequently much shorter than the claimed one. LongBench (Bai et al., 2023) evaluates realistic long-document tasks.

| | Needle in a haystack | Real multi-fact long-context reasoning |
|---|---|---|
| Facts needed | 1 | Several, scattered |
| Matching | Often literal keyword match | Paraphrased, implicit |
| Reasoning | Copy the fact | Combine, compare, count, track changes |
| What it proves | The model can attend to any position | The model can use the whole context |
| Typical result | Near-perfect for modern long-context models | Degrades with length and with position in the middle |

### What improves real long-context use

- **Training data with long-range dependencies:** long documents, synthetic tasks that require combining distant facts, and long instruction-tuning examples.
- **Prompt structure:** put the question and the most important material at the start or end; label documents clearly; ask the model to quote supporting passages first.
- **Retrieval instead of stuffing:** for most production question answering, RAG with good ranking beats putting everything in the window: it is cheaper, faster and often more accurate. See [What Is RAG and Why It Matters](../08-rag/01-what-is-rag-and-why-it-matters.md) and [Reranking and Query Transformation](../08-rag/08-reranking-and-query-transformation.md).
- **Decomposition:** map-reduce summarization, iterative reading, or agents that read selectively.

### How to measure it for your product

1. Build an eval set from **your** documents and **your** question types at the lengths you plan to use (see [Building an LLM Evaluation Framework](../05-llms/11-building-an-llm-evaluation-framework-and-regression-tests.md)).
2. Vary the **depth** of the key evidence (start, 25%, middle, 75%, end) and the **total length** (8k, 32k, 128k), and plot accuracy as a heatmap.
3. Include multi-fact questions, paraphrased evidence and distractors, not just a single literal needle.
4. Track cost and latency alongside accuracy: time to first token grows with prompt length (see [Inference Fundamentals](../06-llm-inference/01-inference-fundamentals-prefill-decode-and-latency-metrics.md)).

A minimal harness sketch (the `generate` function is whatever client you use):

```python
import random

def make_haystack(filler_paragraphs: list[str], needle: str, depth: float, n_paragraphs: int) -> str:
    """Place the needle at a relative depth in [0, 1] of the document."""
    body = random.sample(filler_paragraphs, n_paragraphs)
    body.insert(int(depth * len(body)), needle)
    return "\n\n".join(body)

def run_grid(generate, filler, lengths=(20, 80, 320), depths=(0.0, 0.25, 0.5, 0.75, 1.0)):
    needle = "The access code for the archive room is 7421."
    question = "What is the access code for the archive room? Answer with the number only."
    results = {}
    for n in lengths:
        for depth in depths:
            doc = make_haystack(filler, needle, depth, n)
            answer = generate(f"{doc}\n\nQuestion: {question}")
            results[(n, depth)] = "7421" in answer
    return results  # extend with multi-fact and paraphrased questions for a realistic picture
```

## Going deeper: production pitfalls and debugging

Positional encoding bugs are nasty because the model still produces fluent-looking text; it is just subtly (or badly) worse. Here is the checklist experienced engineers use.

### Common production bugs

| Symptom | Likely cause | Check |
|---|---|---|
| Fluent nonsense after porting weights | RoPE layout mismatch (interleaved vs half-split) | Compare logits with the reference implementation on the same input |
| Quality drops only on long prompts | Serving past the trained length without scaling, or wrong `rope_scaling` | Read `max_position_embeddings`, `rope_theta`, `rope_scaling` in the deployed config |
| Short prompts got worse after enabling long context | Static scaling applied to all requests | Compare short-prompt evals with scaling on and off |
| Batched generation differs from single-request generation | Wrong position IDs with **left padding** (positions must start at 0 at the first real token, not at the pad tokens) | Recompute position IDs from the attention mask |
| Training on packed sequences (several short documents concatenated into one training row) behaves oddly | Position IDs not reset per document (or masks letting documents see each other) | Decide explicitly: reset positions and use per-document masks, or deliberately continue them |
| Generation degrades over very long outputs | Positions computed in low precision | Compute angles and cos/sin in FP32 (mainstream implementations do this); `position * theta` in BF16 loses integer precision at large positions |
| Cached-prefix outputs differ from uncached ones | KV cache keys rotated with a different scaling than new queries (for example a dynamic scale that changed as the sequence grew) | Make scaling deterministic per request, or invalidate prefix caches when it changes |

### Interactions with the rest of the stack

- **KV cache and prefix caching:** cached keys are already rotated for their absolute positions. A shared prefix (system prompt) can be reused only if it sits at the same positions with the same RoPE settings in every request. This is one reason prefix caching works on exact prefixes from position 0 (see [Speculative Decoding, Prompt Caching and Other Inference Speedups](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md)).
- **Sliding-window attention** (each token attends only to the last $W$ tokens, used in Mistral 7B and in local layers of some models) caps per-token cost and cache size, but it is not a way to extend usable context on its own: information beyond the window must flow indirectly through layers.
- **Parameter-efficient context extension:** LongLoRA (Chen et al., 2023) showed LoRA-style fine-tuning can extend context, but needed the embedding and normalization layers trainable too (see [Fine-Tuning LLMs: Full Fine-Tuning, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md)).

### A debugging recipe for "long prompts are worse"

1. **Confirm the numbers:** what length was the model trained on, what length are requests, what scaling is configured, and is it the same in training, eval and serving?
2. **Measure perplexity vs position** on held-out long documents: plot loss per token index. A cliff at the trained length means positional failure; a smooth plateau means positions are fine.
3. **Run a depth-by-length retrieval grid** (above) to see whether the model fails to *use* context even when perplexity is fine.
4. **Compare against a reference implementation** for a few prompts at long lengths to rule out framework bugs.
5. **Only then** decide between: enabling or tuning scaling, continued training on long data, or changing the application to retrieve and shorten (RAG, summarization).

> **Senior insight:** "Supports 1M tokens" on a model card is a capacity claim. Before committing a product design to it, budget the cost (prefill compute is roughly quadratic in prompt length for the attention part, the KV cache grows linearly) and validate effective use on your task. Often the winning design is a 16k-32k prompt built by good retrieval, not a 500k prompt built by concatenation.

## Common mistakes and misconceptions

| Misconception | Reality |
|---|---|
| "Attention knows word order because the inputs come in order." | Self-attention is permutation-equivariant. The order of rows in $X$ is invisible to it without positional information. |
| "Sinusoidal encoding makes attention relative." | It makes relative structure available, but the learned $W_Q W_K^\top$ sits between the encodings. RoPE makes the score exactly relative. |
| "Sinusoidal encodings extrapolate to any length because the formula works for any position." | Defined is not the same as trained. Empirically they degrade beyond the training length. |
| "RoPE adds a vector to the embedding." | RoPE **rotates** queries and keys in every layer. Nothing is added to the token embedding, and values are not touched. |
| "RoPE is purely relative, so it extrapolates." | The score is relative, but unseen relative distances produce unseen angles for slow pairs. RoPE needs scaling or long training to extend. |
| "ALiBi solves long context." | ALiBi keeps perplexity stable on longer inputs, partly by down-weighting distant tokens. That is not the same as retrieving distant facts. |
| "A 128k context window means the model uses all 128k tokens well." | Effective use is usually shorter; check "lost in the middle" and multi-fact benchmarks like RULER. |
| "Passing needle-in-a-haystack proves long-context reasoning." | NIAH tests single-fact lookup, often with literal overlap. Real tasks need combining multiple paraphrased facts. |
| "Position interpolation is free." | It blurs local positional resolution and needs fine-tuning to recover quality. |
| "Increasing `max_position_embeddings` in the config extends the context." | For RoPE models that only changes a limit check; without scaling and training the model still sees unseen angles. For learned embeddings it adds untrained rows. |

## Interview questions and model answers

<details>
<summary><b>Q1. Explain positional encoding and modern alternatives.</b></summary>

**Short answer:** Self-attention has no built-in notion of order, so Transformers inject position information. The original Transformer added fixed sinusoidal vectors to the token embeddings; BERT and GPT-2 added learned position embeddings. Modern LLMs mostly use relative methods applied inside attention: RoPE rotates queries and keys by position-dependent angles so scores depend on relative distance, and ALiBi adds a per-head linear distance penalty to the scores. RoPE plus context-extension techniques (interpolation, NTK, YaRN) is the dominant choice today.

**Deeper answer and follow-ups:**
- Two axes: absolute vs relative, and where position enters (input embedding once, Q/K rotation every layer, or score bias every layer).
- Sinusoidal: $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$, cosine for odd dims; zero parameters; multiple frequencies like clock hands.
- Learned: a $L_{\max} \times d$ table; flexible but a hard length cap (BERT 512, GPT-2 1,024).
- Relative precursors: Shaw et al. 2018, Transformer-XL, T5 bucketed bias, DeBERTa.
- RoPE: $\tilde q_m^\top \tilde k_n = q^\top R_{n-m} k$, zero parameters, kernel and KV-cache friendly.
- ALiBi: score minus $m_h (i-j)$, geometric slopes, extrapolates perplexity but has recency bias.
- Pitfall to avoid in the answer: saying RoPE "adds" something to embeddings.
- Production angle: the choice shows up in the model config (`max_position_embeddings`, `rope_theta`, `rope_scaling`), and it decides how (and whether) you can extend context later, so read those fields before serving a model at a new length.

</details>

<details>
<summary><b>Q2. Why does attention need information about token positions?</b></summary>

**Short answer:** Because self-attention is permutation-equivariant: each score $q_i \cdot k_j$ depends only on the two tokens' vectors, so if you shuffle the input tokens, each token gets exactly the same output vector, just listed in a different order. "Dog bites man" and "man bites dog" would produce identical per-word representations. Language meaning depends on order, so position must be injected.

**Deeper answer and follow-ups:**
- Formally $f(PX) = P f(X)$ for any permutation matrix $P$; FFN, residuals and LayerNorm are per-token, so the whole stack is equivariant.
- RNNs get order from sequential processing; Transformers gave that up for parallelism and must add it back.
- Follow-up: "Does a causal mask fix it?" Partly: it breaks symmetry, and decoder-only NoPE models can infer positions, but encoders cannot, and explicit methods still work better in practice.
- Quick test you can mention: permute inputs in a unit test and assert outputs permute identically when positions are disabled.

</details>

<details>
<summary><b>Q3. Compare sinusoidal and learned positional embeddings.</b></summary>

**Short answer:** Both are absolute and are added to token embeddings once at the input. Sinusoidal uses a fixed formula with no parameters and is defined for any position; learned embeddings are a trainable table with one vector per position up to a fixed maximum. At trained lengths they perform about the same (Vaswani et al. reported nearly identical results). Sinusoidal was hoped to extrapolate but does so poorly in practice; learned embeddings cannot extrapolate at all.

**Deeper answer and follow-ups:**
- Sinusoidal has built-in relative structure: $PE_{p+k}$ is a rotation of $PE_p$ per pair, and $PE_p \cdot PE_{p+k} = \sum_i \cos(\omega_i k)$.
- Learned embeddings can capture data-specific patterns (start-of-document effects) but tail positions are undertrained if long sequences are rare.
- Parameter cost of learned: $L_{\max} \times d$, for BERT-base $512 \times 768 \approx 393$k, negligible.
- Both are added once, so position information must survive many layers; RoPE and ALiBi re-inject it in every layer.
- Learned embeddings are still standard in ViTs (fixed number of patches), with interpolation when resolution changes.
- How to compare them fairly: same model and data, then plot held-out loss per token position, both inside the trained length and (for sinusoidal only) beyond it; the gap past $L_{\max}$ is the extrapolation story, the gap inside it is usually negligible.

</details>

<details>
<summary><b>Q4. What are RoPE and ALiBi? What are RoPE and positional embeddings, and how do they differ?</b></summary>

**Short answer:** A classic positional embedding is one vector per position (fixed sinusoidal or a learned table) added to the token embedding once, before the first layer. RoPE (Rotary Position Embedding) rotates each query and key, pair of dimensions by pair, by an angle proportional to the token's position, using a ladder of frequencies $\theta_i = b^{-2i/d}$. Because rotations compose, the dot product between a rotated query at $m$ and a rotated key at $n$ depends only on $n - m$. So despite its name, RoPE embeds nothing into the input: it acts on Q and K inside every attention layer and makes scores relative by construction, whereas classic positional embeddings are absolute and only make relative structure learnable. ALiBi adds no embeddings at all either; it subtracts $m_h \times$ distance from each attention score, with a fixed slope per head.

**Deeper answer and follow-ups:**
- RoPE math: $\tilde q_m^\top \tilde k_n = q^\top R(m\theta)^\top R(n\theta) k = q^\top R((n-m)\theta) k$.
- RoPE properties: zero parameters, norm-preserving, applied to Q and K but not V, keys rotated before entering the KV cache, compatible with FlashAttention.
- ALiBi slopes for 8 heads: $1/2, 1/4, \ldots, 1/256$; steep heads are local, gentle heads see far.
- Trade-off: ALiBi extrapolates perplexity out of the box but has a recency bias; RoPE needs scaling (PI, NTK, YaRN) to extend but is the more widely adopted choice.
- Implementation gotcha: interleaved vs half-split RoPE layouts are not interchangeable without permuting $W_Q$, $W_K$.
- How to measure the trade-off: ALiBi tends to look good on perplexity-vs-length curves and worse on a depth-by-length retrieval grid, so evaluate both before claiming one "extrapolates better".

</details>

<details>
<summary><b>Q5. Why can a model struggle with sequences longer than those seen during training?</b></summary>

**Short answer:** Several reasons stack up. Positions beyond training are either missing (learned embeddings) or out of distribution: with RoPE, the slow-rotating dimension pairs reach angles they never saw in training. Attention statistics also shift: with more tokens the softmax spreads out and heads become less sharp. And the model never learned from data to connect information across such distances. Finally, compute and KV-cache memory grow with length.

**Deeper answer and follow-ups:**
- Concrete RoPE numbers: base 10,000, head dim 128, trained at 4,096 tokens: the slowest pair only reached about 0.47 rad (27°); at 16k it hits about 1.89 rad (108°), never seen. 18 of 64 pairs never complete a full rotation within 4,096 tokens.
- Fast pairs are fine because they wrapped around hundreds of times.
- Entropy of attention grows roughly with $\log n$; YaRN's temperature term corrects for this.
- Attention-sink patterns can break at new lengths.
- Symptom: perplexity cliff exactly at the trained length; repetitive or incoherent outputs.
- Practical cost: KV cache for Llama 2 7B in FP16 is 0.5 MiB per token, 16 GiB at 32k tokens.

</details>

<details>
<summary><b>Q6. What is the difference between extending a context window and improving long-context retrieval?</b></summary>

**Short answer:** Extending the window is about capacity: making the model accept more tokens without breaking, mostly by fixing positional encoding (interpolation, NTK, YaRN, larger RoPE base) plus some long-sequence training. Improving long-context retrieval or use is about capability: making the model actually find and combine relevant information anywhere in that window. A model can have a 128k window and still miss facts in the middle ("lost in the middle") or fail multi-fact reasoning, even while passing needle-in-a-haystack.

**Deeper answer and follow-ups:**
- Perplexity mainly measures local prediction, so it can look fine while long-range use is poor.
- NIAH is a single literal fact; RULER-style tasks (multi-needle, variable tracking, aggregation) show effective length is often far below claimed length.
- Improving use: long-dependency training data, long instruction tuning, prompt ordering (key info at start or end), retrieval and reranking instead of stuffing, decomposition.
- Measure with a depth-by-length grid on your own tasks, including paraphrased and multi-fact questions.
- Production framing: long prompts also raise time to first token and cost; often RAG at 16k-32k beats a 500k stuffed prompt.

</details>

<details>
<summary><b>Q7. Prove that RoPE makes the attention score depend only on relative position. Why does it rotate Q and K but not V?</b></summary>

**Short answer:** Rotate $q$ by $m\theta$ and $k$ by $n\theta$. Then $\tilde q^\top \tilde k = q^\top R(m\theta)^\top R(n\theta) k$. Because a rotation's transpose is its inverse and rotations add, this equals $q^\top R((n-m)\theta) k$, which depends only on $n - m$. V is not rotated because position should decide where to attend, not change the content being copied; rotating V would also make outputs depend on absolute position.

**Deeper answer and follow-ups:**
- In $d$ dimensions it is a block-diagonal rotation, one 2x2 block per pair, each with its own $\theta_i$, so the argument holds per block.
- Worked check: $q = (1,2)$, $k = (3,1)$, $\theta = 30°$: positions (2,1) and (5,4) both give 1.8301; same positions give $q \cdot k = 5$.
- Direction matters: gap +2 and -2 give different scores, so RoPE encodes order, not just distance.
- Follow-up: "Can you precompute $W_Q W_K^\top$?" Not with RoPE, since a position-dependent rotation sits between them.

</details>

<details>
<summary><b>Q8. You have a 7B RoPE model trained at 4k tokens and need it to work at 32k. Walk me through your options.</b></summary>

**Short answer:** First confirm the need: can retrieval keep prompts under 4k-8k? If not, the options in increasing cost are: dynamic NTK or YaRN at inference with no training (quick, some quality loss); position interpolation, NTK-aware or YaRN scaling plus a short fine-tune on long data; or raising the RoPE base and doing continued pretraining on long documents, then long instruction tuning. I would evaluate each on a depth-by-length grid from our own tasks, plus short-prompt regression tests.

**Deeper answer and follow-ups:**
- Scale factor $s = 32768/4096 = 8$.
- PI: positions divided by 8; simple, but the fastest pair drops from 1 to 0.125 rad per token, blurring local order; needs fine-tuning.
- NTK-aware: base $b' = b \cdot s^{d/(d-2)}$; with $d=128$, $s=8$: $8^{1.0159} \approx 8.27$, so $b' \approx 82{,}700$; high-frequency pairs untouched.
- YaRN: interpolate only long-wavelength pairs, ramp in between, plus attention temperature; usually the best training-efficient option.
- Infrastructure: KV cache at 32k for Llama 2 7B in FP16 is 16 GiB per sequence, so plan batching, GQA or KV quantization, and TTFT budgets.
- Keep the config identical across training, eval and serving; static scaling may hurt short prompts.

</details>

<details>
<summary><b>Q9. What is the RoPE base (rope_theta), and why do long-context models increase it?</b></summary>

**Short answer:** The base $b$ sets the frequency ladder $\theta_i = b^{-2i/d}$. A bigger base makes the slow pairs rotate even slower, so their angles stay within a small, well-trained range over much longer distances. Long-context models raise it (Llama 2 Long to 500,000, Code Llama to 1,000,000, Llama 3 ships with 500,000) and then train on long sequences so the model adapts.

**Deeper answer and follow-ups:**
- With $b = 10{,}000$, $d = 128$, the slowest wavelength is about 54k tokens; increasing $b$ lengthens all but the fastest wavelengths.
- NTK-aware scaling is exactly a base change chosen so the slowest pair slows by the scale factor.
- Too large a base reduces positional resolution between distant tokens (angles barely differ), so it is a trade-off, not "bigger is always better".
- Changing the base of a trained model without training shifts all angles and hurts quality; it is paired with continued training (or with careful scaling formulas).

</details>

<details>
<summary><b>Q10. Does a decoder-only model need positional encoding at all?</b></summary>

**Short answer:** Not strictly. The causal mask breaks permutation symmetry, so a decoder can infer position (for example by how many tokens it can see), and NoPE models have been shown to learn language modeling and sometimes length-generalize decently. But explicit methods, especially RoPE, remain the standard because they give reliable, precise position information. Encoders without a causal mask absolutely need explicit positions.

**Deeper answer and follow-ups:**
- References: Haviv et al., 2022; Kazemnejad et al., 2023.
- Some recent architectures mix layers with RoPE and layers without positional encoding to balance local precision and long-range generalization.
- Interview point: knowing why it works (causal asymmetry) matters more than the result.

</details>

<details>
<summary><b>Q11. How would you evaluate whether a model really uses its long context before shipping a long-document feature?</b></summary>

**Short answer:** Build a task-specific eval set at the target lengths, vary where the key evidence sits (start, middle, end) and how long the context is, and include multi-fact, paraphrased and distractor-heavy questions, not just a single needle. Compare against a RAG baseline with shorter prompts, and track latency and cost too.

**Deeper answer and follow-ups:**
- Plot a depth-by-length heatmap; look for the "lost in the middle" dip.
- Use public suites (RULER, LongBench) as sanity checks, but trust your own tasks most.
- Score with exact match where possible; use LLM-as-a-judge with care for open-ended answers (see [Evaluating LLMs](../05-llms/10-llm-evaluation-metrics-and-llm-as-a-judge.md)).
- Also plot per-token loss vs position on long documents to detect positional failure.
- Add regression tests so a config change (for example a new `rope_scaling`) cannot silently degrade results.

</details>

<details>
<summary><b>Q12. What production bugs have you seen (or would you watch for) related to positional encoding?</b></summary>

**Short answer:** Mismatched RoPE layout when converting checkpoints (interleaved vs half-split), wrong or missing `rope_scaling` at serving time, wrong position IDs with left-padded batches, position IDs not reset for packed sequences, computing RoPE angles in low precision, and prefix or KV caches built with different scaling than the current request.

**Deeper answer and follow-ups:**
- Symptom of layout mismatch: fluent nonsense, very high perplexity; fix by permuting $W_Q$, $W_K$ or using the right implementation.
- Left padding: position IDs should be derived from the attention mask so the first real token is position 0.
- Precision: compute $pos \times \theta$ and cos/sin in FP32; BF16 has only about 8 bits of mantissa precision.
- Debug method: compare logits to a reference implementation; plot loss vs position; run a short-prompt regression after any scaling change.

</details>

## Key takeaways

- Self-attention is **permutation-equivariant**: without position information, shuffling tokens just shuffles the outputs, so "dog bites man" equals "man bites dog".
- Position methods differ in **absolute vs relative** and in **where** they act: input embedding once, Q/K rotation every layer, or score bias every layer.
- **Sinusoidal:** $\sin/\cos(pos \cdot 10000^{-2i/d})$, many frequencies like clock hands, zero parameters; offsets are rotations, but attention is not made exactly relative.
- **Learned absolute:** a trainable table; about as good at trained lengths, hard cap at $L_{\max}$ (BERT 512, GPT-2 1,024).
- **RoPE:** rotate each Q/K dimension pair by $m\theta_i$; scores become $q^\top R_{n-m} k$, exactly relative; no parameters; V is not rotated; keys are cached already rotated.
- **ALiBi:** subtract $m_h \times$ distance from scores with geometric per-head slopes; perplexity extrapolates, but it has a recency bias.
- Long inputs fail because of **unseen angles** (slow RoPE pairs), **attention distribution shift**, **no long-range training**, and **cost** (quadratic attention, linear KV cache).
- **Position Interpolation** divides positions by $s$ (needs fine-tuning, blurs local order); **NTK-aware** raises the base, $b' = b \cdot s^{d/(d-2)}$, sparing fast pairs; **YaRN** interpolates by wavelength plus a temperature fix.
- Vendors build long-context models by **raising the RoPE base and continuing training on long data**, then long instruction tuning.
- **Context window is capacity; long-context use is capability.** "Lost in the middle" and RULER show effective length is often far below the advertised one.
- Needle-in-a-haystack is necessary but nowhere near sufficient; test multi-fact, paraphrased questions at multiple depths on your own data.
- Production checklist: RoPE layout, `rope_theta`, `rope_scaling`, position IDs with padding and packing, FP32 angles, consistent configs across train, eval and serve.

## Further reading

- Vaswani et al., 2017, "Attention Is All You Need" (sinusoidal encoding, comparison with learned).
- Devlin et al., 2019, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (learned absolute positions).
- Shaw, Uszkoreit and Vaswani, 2018, "Self-Attention with Relative Position Representations".
- Dai et al., 2019, "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context".
- Raffel et al., 2020, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5 relative bias).
- Su et al., 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding".
- Press, Smith and Lewis, 2022, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ALiBi).
- Chen et al., 2023, "Extending Context Window of Large Language Models via Positional Interpolation".
- Peng et al., 2023, "YaRN: Efficient Context Window Extension of Large Language Models".
- Xiong et al., 2023, "Effective Long-Context Scaling of Foundation Models".
- Liu et al., 2023, "Lost in the Middle: How Language Models Use Long Contexts".
- Hsieh et al., 2024, "RULER: What's the Real Context Size of Your Long-Context Language Models?"
- Xiao et al., 2023, "Efficient Streaming Language Models with Attention Sinks".
- Kazemnejad et al., 2023, "The Impact of Positional Encoding on Length Generalization in Transformers".
- Hugging Face `transformers` documentation on RoPE scaling configuration (check the version you use).

## Next

Continue with [Encoder-Only, Encoder-Decoder and Decoder-Only Models (BERT, T5, GPT)](07-encoder-only-encoder-decoder-and-decoder-only-models.md), where you will see which positional methods each model family adopted and why.

Related chapters:

- [The Attention Mechanism, Step by Step](03-attention-mechanism-step-by-step.md) and [Multi-Head Attention and Causal Masking](04-multi-head-attention-and-causal-masking.md): the scores that positional methods modify.
- [Modern LLM Architecture: Pre-Norm, RMSNorm, SwiGLU, GQA, MoE and FlashAttention](08-modern-llm-architecture-improvements.md): the rest of a modern RoPE-based decoder.
- [The KV Cache: What It Is and Why It Dominates Memory](../06-llm-inference/02-kv-cache-explained.md): why long context is expensive to serve.
- [Chunking Strategies](../08-rag/04-chunking-strategies.md) and [Advanced RAG Patterns](../08-rag/09-advanced-rag-patterns.md): the retrieval alternative to stuffing long contexts.
