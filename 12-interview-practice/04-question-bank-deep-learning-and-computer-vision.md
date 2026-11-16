# Question Bank: Deep Learning and Computer Vision

> A revision bank of 61 interview questions (plus a rapid-fire round) on neural network training, optimization, losses, architectures, mixed precision, distributed training, detection, segmentation, transfer learning, vision in production and vision transformers. **How to use it:** read the question, answer it out loud first (aim for about 2 minutes), and only then open the answer to compare. Every answer links back to the chapter that teaches it. Source chapters: [02 Deep Learning](../02-deep-learning/01-neural-networks-from-scratch.md) (chapters 01 to 10) and [03 Computer Vision](../03-computer-vision/01-image-classification-detection-and-segmentation.md) (chapters 01 to 05). Course overview: [README](../README.md).

## Table of contents

- [1. Neural network fundamentals and backpropagation](#1-neural-network-fundamentals-and-backpropagation)
- [2. Optimizers, learning rate and batch size](#2-optimizers-learning-rate-and-batch-size)
- [3. Loss functions and class imbalance](#3-loss-functions-and-class-imbalance)
- [4. Debugging and regularizing training](#4-debugging-and-regularizing-training)
- [5. CNNs, residual connections, normalization and RNNs](#5-cnns-residual-connections-normalization-and-rnns)
- [6. Mixed precision and distributed training](#6-mixed-precision-and-distributed-training)
- [7. Vision tasks, detection and segmentation metrics](#7-vision-tasks-detection-and-segmentation-metrics)
- [8. Transfer learning, augmentation and small datasets](#8-transfer-learning-augmentation-and-small-datasets)
- [9. Computer vision in production](#9-computer-vision-in-production)
- [10. Vision transformers, CLIP and multimodal models](#10-vision-transformers-clip-and-multimodal-models)
- [Rapid-fire round](#rapid-fire-round)
- [How to practice](#how-to-practice)

## 1. Neural network fundamentals and backpropagation

<details>
<summary><b>Q1. Explain backpropagation and gradient descent.</b></summary>

**Answer:** Training minimizes a scalar loss over the parameters, and gradient descent is the update rule: move every parameter a small step against its gradient, $\theta \leftarrow \theta - \eta \nabla_\theta L$. Backpropagation is how we get that gradient efficiently: the forward pass is a computational graph of simple operations whose intermediate values are cached, and the backward pass applies the chain rule from the loss backward, multiplying each node's upstream gradient by its local derivative and summing gradients where a value fans out. It is reverse-mode automatic differentiation, so one backward pass (about twice the cost of a forward pass) gives the gradient for every parameter, where forward mode or finite differences would need one pass per parameter. For a linear layer $Z = XW$, the backward pass is $dW = X^\top dZ$ and $dX = dZ\,W^\top$. In practice we use mini-batches, which give a noisy but unbiased gradient estimate and many cheap steps. A full training step costs about 3x a forward pass, roughly $6N$ FLOPs per token for an $N$-parameter transformer.

**Key points to mention:**
- Separate the concerns: backprop computes gradients, the optimizer (SGD, momentum, Adam) decides how to use them.
- Reverse mode wins because there are billions of inputs (parameters) and one scalar output; its price is storing activations, which usually sets the micro-batch size.
- Pitfalls: vanishing or exploding gradients through long products, dead ReLUs, non-differentiable ops (argmax, sampling), and a local slope is no guarantee of a global minimum.
- Verify a hand-written backward with central-difference gradient checking in float64 and by overfitting one tiny batch.
- Typical follow-ups: vanishing gradients (Q5), Adam vs SGD (Q7), learning rate and batch size (Q8), a loss that will not decrease (Q16).

**Deep dive:** [Backpropagation and Gradient Descent](../02-deep-learning/02-backpropagation-and-gradient-descent.md), [Neural Networks from Scratch](../02-deep-learning/01-neural-networks-from-scratch.md)

</details>

<details>
<summary><b>Q2. Why do neural networks need non-linear activation functions? What happens if you remove them?</b></summary>

**Answer:** Without non-linearities, a stack of linear layers collapses into a single linear layer: $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$. Depth then adds parameters but no expressive power, so the network can only draw linear decision boundaries and cannot solve even XOR. Non-linear activations let hidden layers re-map the input into a space where the problem becomes simpler; a ReLU network computes a piecewise linear function whose number of linear regions can grow exponentially with depth. The output layer is often left linear: regression outputs raw values, and for classification the softmax or sigmoid lives inside the loss. The universal approximation theorem says one hidden layer with a suitable non-linearity can approximate any continuous function on a bounded domain, but it says nothing about how many units are needed, whether gradient descent finds the weights, or whether the result generalizes.

**Key points to mention:**
- XOR by hand: $h_1 = \text{ReLU}(x_1 + x_2)$, $h_2 = \text{ReLU}(x_1 + x_2 - 1)$, $y = h_1 - 2h_2$.
- Forgetting an activation between two `nn.Linear` layers is a silent bug: no error, just a weaker model.
- Depth is more parameter-efficient for compositional functions but harder to optimize and adds sequential latency; width parallelizes well.
- In practice the bottleneck is optimization and generalization, not representational capacity, which is why architecture design matters.

**Deep dive:** [Neural Networks from Scratch](../02-deep-learning/01-neural-networks-from-scratch.md)

</details>

<details>
<summary><b>Q3. Compare sigmoid, tanh, ReLU and GELU/SwiGLU. What is the dying ReLU problem?</b></summary>

**Answer:** Sigmoid maps to (0, 1) with a maximum slope of 0.25 and saturates at both ends; tanh is zero-centered with maximum slope 1 but also saturates; ReLU, $\max(0, z)$, has slope exactly 1 for positive inputs, is cheap and does not saturate on the positive side. Backprop multiplies by the activation slope at every layer, so sigmoid hidden layers shrink gradients by at least 4x per layer ($0.25^{10} \approx 9.5 \times 10^{-7}$ over 10 layers), which is why ReLU made deep networks practical. A ReLU unit "dies" when its pre-activation is negative for every input: its output and gradient are always zero and its weights never update again, usually because of a too-high learning rate, a large negative bias or bad initialization. Detect it by logging the fraction of units that are zero across a whole batch, and fix it with a lower learning rate, He initialization, normalization, or Leaky ReLU, GELU or SiLU. Transformers use smooth activations: GELU in BERT and GPT-2, and SwiGLU (a SiLU branch multiplied by a linear gate) in Llama-family models, which empirically trains better at equal parameter count.

**Key points to mention:**
- Sigmoid and tanh are still correct for output probabilities, LSTM gates and GLU gates; the problem is using them in deep hidden stacks.
- Distinguish "zero for this input" (healthy sparsity) from "zero for all inputs" (a dead unit).
- SwiGLU has three matrices, so the inner width is cut to about $\frac{8}{3}d$ (11,008 for $d = 4096$ in Llama 2 7B) to keep parameters comparable.
- The gains of GELU/SwiGLU are empirical; saying so honestly is better than inventing a theory.

**Deep dive:** [Neural Networks from Scratch](../02-deep-learning/01-neural-networks-from-scratch.md)

</details>

<details>
<summary><b>Q4. Why not initialize all weights to zero, and how do He and Xavier initialization differ?</b></summary>

**Answer:** With all-zero or all-equal weights, every neuron in a layer computes the same output and receives the same gradient, so they stay identical forever: symmetry is never broken and the layer behaves like a single neuron. Random initialization breaks symmetry, but the scale matters: too large saturates sigmoid/tanh or explodes activations, too small makes signals vanish. For $y = \sum_i w_i x_i$, $\text{Var}(y) = n\,\text{Var}(w)\,\text{Var}(x)$, so keeping the variance constant needs $\text{Var}(w) \approx 1/n$. Xavier (Glorot) assumes a roughly linear activation like tanh and averages the forward and backward requirements, $2/(n_{\text{in}} + n_{\text{out}})$; He (Kaiming) accounts for ReLU zeroing about half its inputs and uses $2/n_{\text{in}}$. For 512 inputs and outputs, He gives a standard deviation of 0.0625 and Xavier about 0.0442, a ratio of $\sqrt{2}$.

**Key points to mention:**
- Biases can start at zero because the weights already break symmetry.
- Xavier in a deep ReLU network shrinks activations by about $\sqrt{2}$ per layer; plain `randn` explodes them.
- Check an init empirically: one forward pass, log per-layer activation standard deviation and gradient norms; they should stay roughly flat across depth.
- Normalization and residuals make modern networks less sensitive to init, not immune; transformers use a small fixed std (about 0.02) plus scaled-down residual output projections.

**Deep dive:** [Neural Networks from Scratch](../02-deep-learning/01-neural-networks-from-scratch.md), [Vanishing and Exploding Gradients, Initialization, Normalization and Residuals](../02-deep-learning/05-vanishing-exploding-gradients-initialization-normalization-and-residuals.md)

</details>

<details>
<summary><b>Q5. What causes vanishing and exploding gradients?</b></summary>

**Answer:** Backpropagation computes the gradient for early layers as a product of one factor per layer, each layer's weights times its activation slope (its Jacobian). If those factors are typically below 1 the product shrinks exponentially with depth (vanishing); if above 1 it grows exponentially (exploding): $0.9^{50} \approx 0.005$ and $1.1^{50} \approx 117$. Vanishing comes from saturating activations (sigmoid's slope is at most 0.25), too-small initialization and long RNN unrolls; exploding comes from too-large initialization, high learning rates, recurrent matrices with largest singular value above 1, and unstable attention logits in large transformers. Vanishing shows up as early layers that barely learn and a plateaued loss; exploding shows up as loss spikes and NaNs. The fixes are ReLU-family activations, He/Xavier initialization, normalization, residual connections, gated RNNs (LSTM/GRU), and gradient clipping, which only helps the exploding case.

**Key points to mention:**
- Measure, do not guess: per-layer gradient norms (early layers orders of magnitude below late ones means vanishing), activation histograms, and the global gradient norm over time (spikes before NaNs).
- Senior nuance: an average gain of 1 is not enough at great depth; the spread of the Jacobians' singular values matters, which is why orthogonal init and residual paths help.
- Adam does not really solve vanishing gradients: it rescales small but informative gradients, but cannot recover a signal that is zero (dead units, underflow) or buried in noise.
- In LLM training, explosion appears as loss spikes and FP16 overflow; vanishing is quieter (early weights barely move).

**Deep dive:** [Vanishing and Exploding Gradients, Initialization, Normalization and Residuals](../02-deep-learning/05-vanishing-exploding-gradients-initialization-normalization-and-residuals.md), [RNNs, LSTMs and the Road to Transformers](../02-deep-learning/08-rnns-lstms-and-the-road-to-transformers.md)

</details>

<details>
<summary><b>Q6. Gradient clipping by norm vs by value: which would you use and why?</b></summary>

**Answer:** Clip by global norm in almost all cases. It rescales the whole gradient vector when its length exceeds a threshold, $g \leftarrow g \cdot \min(1, c/\lVert g \rVert)$, which keeps the update direction and only limits the step size. Clipping by value clamps each element independently and distorts the direction: $[3, 4]$ with $c = 1$ becomes $[0.6, 0.8]$ by norm but $[1, 1]$ by value. Clipping only treats exploding gradients, not vanishing ones, and a threshold of about 1.0 is a common default in LLM recipes. If clipping fires on most steps, the learning rate is probably too high or something is numerically unstable, so clipping is masking a problem.

**Key points to mention:**
- Log the pre-clip norm (`clip_grad_norm_` returns it); spikes are an early warning of divergence.
- With AMP and FP16, unscale the gradients before clipping, or the threshold is compared with values $S$ times too large.
- Clip per optimizer step, not per micro-batch, when accumulating gradients.
- Global norm means one huge layer shrinks everyone's update, which is usually acceptable.

**Deep dive:** [Vanishing and Exploding Gradients, Initialization, Normalization and Residuals](../02-deep-learning/05-vanishing-exploding-gradients-initialization-normalization-and-residuals.md)

</details>

## 2. Optimizers, learning rate and batch size

<details>
<summary><b>Q7. How do Adam and SGD differ?</b></summary>

**Answer:** SGD moves every parameter by the learning rate times its gradient, optionally with momentum, a running sum of past gradients that smooths the direction and damps zig-zagging. Adam keeps two running averages per parameter, of the gradient (first moment, $m$) and of the squared gradient (second moment, $v$), applies bias correction for their zero initialization, and steps by $\eta\,\hat{m}/(\sqrt{\hat{v}} + \epsilon)$, so each parameter gets its own effective step size that is independent of gradient scale. In practice Adam converges faster with less tuning and is the default for transformers, whose parameter blocks have very different curvature, while well-tuned SGD with momentum can generalize as well or better on some vision tasks. Adam also costs more memory: two extra FP32 values per parameter, about 56 GB for a 7B model, versus 28 GB for momentum SGD. Comparisons are only fair with the same tuning budget, since optimizer rankings flip depending on which hyperparameters were tuned.

**Key points to mention:**
- Use AdamW, not `Adam(weight_decay=...)`, which applies coupled L2 (see Q9).
- Adam's defaults ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) rarely change, but LLM recipes use $\beta_2 = 0.95$ so the optimizer reacts faster to gradient bursts.
- Bias correction: without it, the first step is inflated by about $0.1/\sqrt{0.001} \approx 3.16$.
- A from-scratch learning rate like $10^{-3}$ wrecks a pretrained model during fine-tuning regardless of optimizer.

**Deep dive:** [Optimizers: SGD, Momentum, Adam, AdamW and Learning Rate Schedules](../02-deep-learning/03-optimizers-sgd-momentum-adam-and-learning-rate-schedules.md)

</details>

<details>
<summary><b>Q8. How do learning rate and batch size affect training?</b></summary>

**Answer:** The learning rate sets the step size: too small crawls, too large oscillates or diverges (for a quadratic, stability needs $\eta < 2/\lambda_{\max}$). Batch size sets gradient noise (variance falls like $1/B$, so 4x the batch only halves the noise standard deviation), hardware utilization, steps per epoch and activation memory. They are coupled because SGD noise scales roughly with $\eta/B$, so when the batch grows you usually raise the learning rate (linear scaling for SGD, closer to square-root for Adam) and add warmup: Goyal et al. used $\eta = 0.1 \times B/256$, so 3.2 at batch 8,192. Bigger batches cut wall-clock time up to a critical batch size; beyond it, doubling the batch no longer halves the steps and you pay compute for little speed-up. A classic production bug: moving from 1 to 8 data-parallel GPUs grows the effective batch 8x, and if the learning rate and step-based schedule are not rescaled, final accuracy drops.

**Key points to mention:**
- Effective batch = per-device batch x accumulation steps x number of devices; tune the learning rate for that number.
- Schedules defined in steps (warmup, total steps) must be recomputed when the batch changes.
- Compare runs at equal examples or tokens, not equal steps.
- Measure the critical batch size: retune the learning rate per batch size and find where steps-to-target stop halving.
- Large-batch generalization gaps are often closed by longer training, warmup and LARS/LAMB at extreme scale.

**Deep dive:** [Backpropagation and Gradient Descent](../02-deep-learning/02-backpropagation-and-gradient-descent.md), [Optimizers: SGD, Momentum, Adam, AdamW and Learning Rate Schedules](../02-deep-learning/03-optimizers-sgd-momentum-adam-and-learning-rate-schedules.md)

</details>

<details>
<summary><b>Q9. What is the difference between L2 regularization and weight decay, and why does AdamW exist?</b></summary>

**Answer:** L2 regularization adds $\frac{\lambda}{2}\lVert w \rVert^2$ to the loss, so $\lambda w$ is added to the gradient; weight decay shrinks the weights directly by a factor $(1 - \eta\lambda)$ each step. For plain SGD the two are mathematically identical. In Adam they differ, because the $\lambda w$ term is divided by $\sqrt{\hat{v}}$ along with the data gradient, so weights with a large gradient history are barely regularized. AdamW (Loshchilov and Hutter) decouples the decay and applies it outside the adaptive scaling, so all weights decay at the same rate; it generalizes better and is the transformer default. Example with $\eta = 0.1$, $\lambda = 0.01$, $w = 1$: under L2-in-Adam a weight with $\sqrt{\hat{v}} = 10$ decays by 0.0001 while one with $\sqrt{\hat{v}} = 0.1$ decays by 0.01, but AdamW decays both by 0.001.

**Key points to mention:**
- In PyTorch, `Adam(weight_decay=...)` is coupled L2; `AdamW` is decoupled.
- Typical values: about 0.01 for BERT-style fine-tuning, 0.1 for LLM pretraining.
- Exclude biases and normalization weights from decay with parameter groups.
- PyTorch's AdamW multiplies the decay by the current learning rate, so the decay follows the schedule and values do not transfer blindly between codebases.

**Deep dive:** [Optimizers: SGD, Momentum, Adam, AdamW and Learning Rate Schedules](../02-deep-learning/03-optimizers-sgd-momentum-adam-and-learning-rate-schedules.md), [Regularization and Debugging Neural Network Training](../02-deep-learning/06-regularization-and-debugging-neural-network-training.md)

</details>

<details>
<summary><b>Q10. Why do transformers use learning rate warmup, and how would you choose a schedule?</b></summary>

**Answer:** At the start of training the weights are random, gradients are large and erratic, and Adam's second-moment estimates are built from very few samples, so full-size steps can push the model into a bad region, causing loss spikes or divergence. Warmup ramps the learning rate from near zero over the first few hundred to few thousand steps (about 2,000 for LLM pretraining, often 6 to 10% of steps for fine-tuning) so these statistics stabilize first. Post-norm transformers are especially unstable without it; pre-norm helps but warmup remains standard, and large-batch training needs it to use a scaled-up learning rate. To pick a schedule, start from a known recipe (AdamW with linear warmup and cosine or linear decay for transformers, SGD momentum with cosine or step decay for CNNs), then run a learning rate range test or a short log-scale sweep and take the largest stable rate. Warmup does not rescue a peak that is too high: divergence right after warmup ends means the peak is the problem.

**Key points to mention:**
- Starting points: full fine-tuning about $10^{-5}$ to $5 \times 10^{-5}$, LoRA about $10^{-4}$ to $3 \times 10^{-4}$.
- Cosine needs the total step count in advance; warmup-stable-decay suits open-ended runs.
- Search on a log scale, because the useful range spans orders of magnitude.
- At large scale, tune on small proxy models and transfer the settings rather than sweeping the big model.

**Deep dive:** [Optimizers: SGD, Momentum, Adam, AdamW and Learning Rate Schedules](../02-deep-learning/03-optimizers-sgd-momentum-adam-and-learning-rate-schedules.md)

</details>

## 3. Loss functions and class imbalance

<details>
<summary><b>Q11. How do you choose a loss function?</b></summary>

**Answer:** I start from what the model outputs and what the business measures. The output type gives a default likelihood-based loss: MSE for regression, binary cross-entropy for binary, cross-entropy for multi-class, BCE per label for multi-label, and InfoNCE for embeddings. Each standard loss is a negative log-likelihood under a noise assumption: MSE assumes Gaussian noise and estimates the mean, MAE assumes Laplace noise and estimates the median. Then I ask what the real metric cares about that the default ignores (outliers, asymmetric costs, imbalance, overlap, ranking) and adjust with Huber, quantile (pinball) loss, class weights or focal loss, Dice, or contrastive losses. For example, a demand forecast trained with MSE causes stockouts because it predicts the mean; if under-stocking is 9x as costly as over-stocking, pinball loss at $\tau = 0.9$ is the right objective. Finally I validate on the business metric, not on the loss, and keep the simple baseline unless the fancier loss clearly beats it beyond seed-to-seed noise.

**Key points to mention:**
- Practical checks: differentiable, numerically stable (feed logits), comparable scales in multi-task setups, and the expected initial loss ($\ln K$).
- If probabilities feed a cost calculation, prefer plain cross-entropy (keeps calibration) and handle imbalance through the threshold.
- A loss change shifts the score distribution, so re-fit thresholds and calibration before rollout.
- Typical follow-ups: CE vs MSE vs MAE vs Dice (Q12), misleading accuracy (Q14), class imbalance (Q15).

**Deep dive:** [Loss Functions](../02-deep-learning/04-loss-functions.md)

</details>

<details>
<summary><b>Q12. When would you use cross-entropy, MSE, MAE, or Dice loss?</b></summary>

**Answer:** Cross-entropy for classification, because it is the likelihood of a categorical outcome and keeps a strong gradient ($p - y$ with respect to the logit) when the model is confidently wrong. MSE for regression with roughly Gaussian errors where large errors really are disproportionately costly; its gradient grows with the error ($2r$), so a single outlier with residual 14 pulls 28 units hard. MAE for regression with outliers or heavy-tailed noise, or when the reported metric is MAE: it estimates the median and its gradient magnitude is constant, so it needs learning-rate decay, and Huber is the smooth compromise. Dice, $2TP/(2TP + FP + FN)$, for segmentation, especially with small foreground objects, because it ignores true negatives so the huge background does not dominate. Dice alone is noisy on small or empty masks, so the standard recipe is BCE + Dice.

**Key points to mention:**
- MSE through a sigmoid has a gradient about 75x weaker than CE at $z = -5$ for a positive example.
- Box regression classically uses Smooth L1/Huber; modern detectors add IoU-family losses (GIoU, CIoU).
- Dice and IoU are monotonically related ($\text{IoU} = \text{Dice}/(2 - \text{Dice})$), so never compare a Dice number with an IoU number.
- Report segmentation per class and per object size; a good average can hide failures on small structures.

**Deep dive:** [Loss Functions](../02-deep-learning/04-loss-functions.md)

</details>

<details>
<summary><b>Q13. Why is cross-entropy preferred over MSE for classification, and why do PyTorch losses take logits instead of probabilities?</b></summary>

**Answer:** Cross-entropy is the negative log-likelihood of the labels, so minimizing it is maximum likelihood, and its gradient with respect to the logit is simply $p - y$, which stays large when the model is confidently wrong. MSE through a sigmoid has gradient $2(p - y)p(1 - p)$, which vanishes when $p$ is near 0 or 1: for $y = 1$ and $z = -5$, CE gives $-0.993$ and MSE gives $-0.013$, about 75x smaller. For logistic regression CE is also convex in the weights while MSE with a sigmoid is not. Losses such as `BCEWithLogitsLoss` and `CrossEntropyLoss` take raw logits because computing softmax or sigmoid and then a separate log can produce $\log(0)$ or overflow; fusing them lets the library use stable identities (log-sum-exp with max subtraction). This matters even more in FP16, whose range is small.

**Key points to mention:**
- Classic bug: a final `nn.Sigmoid()` plus `BCEWithLogitsLoss`, or a softmax before `CrossEntropyLoss`; training stalls with a loss floor (about 0.551 for 3 classes with double softmax).
- CE penalizes confident mistakes unboundedly ($-\ln 0.01 = 4.6$), great for clean labels but sensitive to label noise.
- The Brier score (MSE on probabilities) is a good evaluation metric, just a weak training signal.
- Softmax for single-label multi-class, independent sigmoids for multi-label.

**Deep dive:** [Loss Functions](../02-deep-learning/04-loss-functions.md), [Neural Networks from Scratch](../02-deep-learning/01-neural-networks-from-scratch.md)

</details>

<details>
<summary><b>Q14. Why can accuracy be misleading during training?</b></summary>

**Answer:** Accuracy is a thresholded count, so it ignores confidence. Loss can drop a lot while accuracy stays flat (correct predictions becoming more confident), and accuracy can rise while loss gets worse (more examples right, but a few confidently wrong). For example, true-class probabilities moving from [0.55, 0.55, 0.45, 0.45] to [0.95, 0.95, 0.95, 0.01] raise accuracy from 50% to 75% but raise mean cross-entropy from 0.698 to 1.190. Accuracy is also misleading under class imbalance (always predicting negative gives 99% with 1% positives), noisy on small validation sets (one example in 200 is half a point), and tied to one threshold. And it cannot be the training objective, because its gradient is zero almost everywhere.

**Key points to mention:**
- Rising validation loss with flat or rising accuracy usually signals overconfidence and early overfitting: check calibration (reliability diagram, ECE).
- For imbalanced problems use PR-AUC, recall at fixed precision, or F1, and compare with the majority baseline.
- A model can rank better (higher PR-AUC) yet show lower accuracy at 0.5.
- Track loss and the business metric together and investigate when they diverge.

**Deep dive:** [Loss Functions](../02-deep-learning/04-loss-functions.md)

</details>

<details>
<summary><b>Q15. How would you handle class imbalance in a neural network?</b></summary>

**Answer:** First measure correctly, with PR-AUC, recall at a fixed precision or expected cost, and compare against the majority-class baseline. The cheapest lever is the decision threshold, tuned on validation data for the business cost: with a false alarm costing 5 dollars and a miss 500, the optimal threshold is $C_{FP}/(C_{FP} + C_{FN}) \approx 0.0099$. If the model still does not learn the minority class, reweight the loss (`pos_weight` or class weights, roughly negatives/positives), use focal loss when a flood of easy negatives dominates, or rebalance batches by oversampling with augmentation. Initializing the output bias to the prior ($b = -\log((1-\pi)/\pi) \approx -4.6$ for 1% positives) stabilizes early training. Getting more minority data is often the biggest win of all.

**Key points to mention:**
- Focal loss multiplies CE by $(1 - p_t)^\gamma$; with $\gamma = 2$ an easy example at $p_t = 0.9$ keeps 1% of its loss and a hard one at 0.1 keeps 81%.
- Class weights and oversampling inflate predicted probabilities, so recalibrate if probabilities are used downstream; after undersampling negatives with keep rate $\beta$, correct the logits by adding $\ln \beta$.
- Pitfalls: tuning the threshold on test data, oversampling without augmentation (memorized positives), hard negative mining amplifying label noise.
- In production the positive rate drifts, so thresholds go stale; re-tune on recent labels and monitor precision and recall per segment.

**Deep dive:** [Loss Functions](../02-deep-learning/04-loss-functions.md), [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

</details>

## 4. Debugging and regularizing training

<details>
<summary><b>Q16. How would you debug a model whose loss is not decreasing?</b></summary>

**Answer:** I go from cheap and local to expensive and global. First I clarify the symptom (training or validation loss, flat from step 0 or plateaued, did this setup ever work) and, if it used to work, diff against the last good run. Then I look at real batches after preprocessing, and check the initial loss against the expected value: $\ln K$ for $K$ classes, so $\ln 2 = 0.693$ for binary and 2.303 for 10 classes; a loss stuck exactly at $\ln K$ means the model outputs a constant. The key test is to turn off regularization and overfit one small batch: if the model cannot memorize 32 examples, the bug is in the model, loss or loop (learning rate, double softmax or sigmoid, missing `zero_grad`, frozen parameters, optimizer created before the new head, detached graph). If it can overfit one batch but not the full data, I look at data quality, input normalization, the schedule and capacity. Throughout, I plot train and validation loss, the learning rate actually read from the optimizer, the global gradient norm and per-layer update-to-weight ratios (about $10^{-3}$ is healthy).

**Key points to mention:**
- Gradient health per layer: `None` means disconnected, zero means dead units or masked paths, tiny in early layers means vanishing, NaN means numerical failure.
- Shape bugs: `(N, 1)` vs `(N,)` in MSE silently broadcasts to an $N \times N$ matrix.
- Compare with a majority baseline and a logistic regression; if a linear model wins, suspect a bug before adding capacity.
- LLM fine-tuning extras: label masking (all-ignored targets give NaN), LoRA adapters actually trainable, each data-parallel rank getting different data.
- Do not add regularization or tune hyperparameters before the model can overfit; change one thing at a time.

**Deep dive:** [Regularization and Debugging Neural Network Training](../02-deep-learning/06-regularization-and-debugging-neural-network-training.md), [Neural Networks from Scratch](../02-deep-learning/01-neural-networks-from-scratch.md)

</details>

<details>
<summary><b>Q17. Your training loss is fine for a few thousand steps and then becomes NaN. How do you debug it?</b></summary>

**Answer:** First decide whether it is instability or a bug by looking at what preceded it. A gradient norm that climbs before the NaN suggests optimization instability (peak learning rate too high, warmup too short, no clipping); a sudden NaN on one batch suggests bad data or a numerical edge case, so I save that batch and inspect it. Common numerical sources are $\log(0)$ in hand-written losses, division by zero (Dice without smoothing, a zero standard deviation), softmax without max subtraction, and FP16 overflow above 65,504 ($e^z$ overflows for $z > 11.09$). I rerun the failing step in BF16 or FP32 to see whether precision is the cause, locate the first non-finite tensor with forward hooks or `torch.autograd.set_detect_anomaly(True)` (slow, debugging only), and check the loss-scale history. Fixes include a lower peak learning rate or longer warmup, clipping at about 1.0, fused stable losses, BF16 instead of FP16, $\beta_2 = 0.95$, and rolling back to a checkpoint while skipping offending batches.

**Key points to mention:**
- Add a finite-loss check every step that saves the batch and state and stops the run, instead of letting NaN corrupt the weights.
- Never "fix" it with `nan_to_num` on the loss or gradients; that hides the cause.
- If the spike follows a resume, check optimizer and scheduler state were restored.
- At pretraining scale (as in PaLM), rollback plus batch skipping is standard practice; checkpoint frequency is a cost trade-off.

**Deep dive:** [Regularization and Debugging Neural Network Training](../02-deep-learning/06-regularization-and-debugging-neural-network-training.md), [Mixed-Precision Training](../02-deep-learning/09-mixed-precision-training.md), [Backpropagation and Gradient Descent](../02-deep-learning/02-backpropagation-and-gradient-descent.md)

</details>

<details>
<summary><b>Q18. Training loss keeps falling but validation loss started rising. What do you do?</b></summary>

**Answer:** That is the classic overfitting signature. First I confirm it is real: same preprocessing for both splits, no leakage, a representative validation set, and a check of the actual task metric, since validation loss can rise from overconfidence on a few wrong examples while accuracy still improves. Then I keep the best checkpoint with early stopping and add regularization roughly in order of typical impact: more or augmented data, weight decay, dropout, label smoothing, a smaller or pretrained model, and a lower learning rate or fewer epochs when fine-tuning. Dropout zeroes each activation with probability $p$ during training and scales survivors by $1/(1-p)$ (inverted dropout), so inference needs no change and runs deterministically. I change one regularizer at a time; good regularization narrows the train-validation gap while keeping training loss reasonably low.

**Key points to mention:**
- For tiny datasets, transfer learning is usually the single biggest win.
- Early stopping selects on validation, so report final numbers on an untouched test set.
- Validation loss below training loss is often fine (dropout and augmentation only in training), but a large persistent gap can indicate leakage or an easier validation set.
- LLM fine-tuning: fewer epochs, lower learning rate, LoRA, held-out prompts, and checks for forgetting of general skills.

**Deep dive:** [Regularization and Debugging Neural Network Training](../02-deep-learning/06-regularization-and-debugging-neural-network-training.md)

</details>

## 5. CNNs, residual connections, normalization and RNNs

<details>
<summary><b>Q19. Explain the purpose of CNNs, residual connections, and normalization.</b></summary>

**Answer:** CNNs process grid data such as images with small learned filters that slide across the input, exploiting locality and reusing the same weights everywhere, which makes them parameter-efficient and data-efficient for vision. Residual connections compute $y = x + F(x)$, so each block only learns a correction and gradients get a direct identity path backward ($\partial y/\partial x = I + \partial F/\partial x$), which made 100+ layer networks and deep transformers trainable. Normalization rescales activations to a stable distribution (BatchNorm per channel in CNNs, LayerNorm or RMSNorm per token in transformers), allowing higher learning rates, faster and more stable training and less sensitivity to initialization. The two are complementary: residuals keep the gradient path healthy across depth, normalization keeps the activation scale healthy at each block, and every transformer block is "normalize, transform, add back the input". Remove either and things break: residuals alone let the stream grow, normalization alone still leaves deep plain networks with the degradation problem.

**Key points to mention:**
- Parameter contrast: a 3-to-64-channel 3x3 conv has 1,792 parameters versus about 150M for a 1,000-unit dense layer on a 224x224x3 image.
- Do not claim BatchNorm works by "removing internal covariate shift"; the smoother-landscape explanation is more accepted, and the mechanism is still debated.
- At inference BN folds into the preceding conv for free; residual adds keep the skip activation in memory until the add.
- Typical follow-ups: why CNNs suit images (Q20), what residuals solve (Q21), BN vs LN (Q22), transfer learning (Q44).

**Deep dive:** [CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md), [Vanishing and Exploding Gradients, Initialization, Normalization and Residuals](../02-deep-learning/05-vanishing-exploding-gradients-initialization-normalization-and-residuals.md)

</details>

<details>
<summary><b>Q20. Why are CNNs effective for image data?</b></summary>

**Answer:** Because their architecture matches the structure of images. Small kernels encode locality (nearby pixels are related), and weight sharing encodes that a useful pattern is useful anywhere, which together give translation equivariance; pooling and global average pooling then give approximate translation invariance for classification. Stacking layers grows the receptive field and builds a hierarchy from edges to textures to parts to objects, and two stacked 3x3 kernels cover a 5x5 field with $18C^2$ instead of $25C^2$ weights. These inductive biases slash the parameter count (1,792 for a 64-filter 3x3 conv versus about 150.5M for a dense layer on a 224x224x3 input), let every weight learn from every position, and work at any resolution. The result is that CNNs need far less data than a generic network, and with small datasets they (or pretrained models) usually beat ViTs trained from scratch.

**Key points to mention:**
- Equivariance $f(T(x)) = T(f(x))$ vs invariance $f(T(x)) = f(x)$: convolution gives the first, global pooling the second.
- Limitations: no built-in rotation or scale invariance (use augmentation), striding breaks exact shift equivariance (aliasing), texture bias, limited long-range interaction early.
- The same biases fit audio spectrograms, time series and genomics, and are wrong for tabular data.
- Show it empirically with learning curves: CNNs lead at small N, ViTs catch up or win at large N.

**Deep dive:** [CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md)

</details>

<details>
<summary><b>Q21. What problem do residual connections solve?</b></summary>

**Answer:** They solve the degradation problem: very deep plain networks are hard to optimize and can have higher *training* error than shallower ones, which is an optimization issue, not overfitting. He et al. showed a 56-layer plain network with higher training error than a 20-layer one on CIFAR-10, even with BatchNorm; with residuals, deeper was better, up to 152 layers on ImageNet. A residual block computes $y = x + F(x)$, so its derivative is $I + \partial F/\partial x$ and the gradient from the top reaches every block without being attenuated. The block only needs to learn a correction, and learning $F = 0$ (identity) is much easier than learning an identity through non-linear layers. When the shapes change, a 1x1 projection convolution is used on the shortcut.

**Key points to mention:**
- Verify by comparing *training* loss of plain vs residual networks at equal depth; per-block gradient norms stay roughly flat with residuals.
- The residual stream's variance grows with depth; mitigate with scaled or zero-initialized branches (for example the last BN $\gamma$ in each block) and pre-norm.
- Robustness: deleting one block from a trained ResNet barely hurts, which also enables stochastic depth.
- Not a regularizer and not a replacement for normalization; ResNets and transformers use both.

**Deep dive:** [Vanishing and Exploding Gradients, Initialization, Normalization and Residuals](../02-deep-learning/05-vanishing-exploding-gradients-initialization-normalization-and-residuals.md), [CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md)

</details>

<details>
<summary><b>Q22. Compare BatchNorm and LayerNorm.</b></summary>

**Answer:** Both normalize activations to zero mean and unit variance and then apply a learnable scale and shift. BatchNorm computes statistics per feature (per channel over N, H, W) across the examples in the batch, so it depends on batch size and composition, keeps running averages for inference, and behaves differently in training and eval mode. LayerNorm computes statistics per example (per token) across its features, so it is batch-independent and identical in training and inference. BatchNorm suits CNNs with reasonable batch sizes and folds into the preceding conv at inference for zero cost; LayerNorm suits transformers and RNNs with variable lengths, padding, and tiny or dynamic batches during autoregressive decoding. BN's typical failure modes are small-batch noise, unusable batch size 1 in training (an error for fully connected BN, statistics from one image's spatial positions only for conv BN), train/eval mismatch and the need for SyncBatchNorm across GPUs.

**Key points to mention:**
- Quick BN bug test: in eval mode, an input must give the same output alone and inside any batch.
- GroupNorm is the batch-independent choice for CNNs with per-device batches of 1 to 8 (detection, segmentation, diffusion U-Nets), at the cost of not folding into convs.
- Modern LLMs often use RMSNorm, which drops mean-centering (see Q23).
- Not purely a modality choice: ConvNeXt uses LayerNorm in a CNN; batch regime and serving constraints decide.

**Deep dive:** [Vanishing and Exploding Gradients, Initialization, Normalization and Residuals](../02-deep-learning/05-vanishing-exploding-gradients-initialization-normalization-and-residuals.md)

</details>

<details>
<summary><b>Q23. What is the difference between pre-norm and post-norm transformers, and why do modern LLMs use RMSNorm?</b></summary>

**Answer:** Post-norm (original Transformer, BERT) computes $\text{LN}(x + F(x))$, placing a normalization on the residual path in every block. Pre-norm (GPT-2 onward, Llama) computes $x + F(\text{LN}(x))$, normalizing only the branch input and leaving the residual highway a clean identity, plus a final norm before the output head. Pre-norm has well-behaved gradients at initialization, so it trains stably at large depth with less dependence on warmup, while post-norm has large gradients near the output at init (Xiong et al., 2020). RMSNorm divides each token's vector by its root mean square and applies a learnable scale, skipping LayerNorm's mean subtraction and bias; it matches LayerNorm quality, has fewer parameters and is slightly cheaper. Llama-family and Mistral models use pre-norm RMSNorm, usually computed in FP32 even when the model runs in BF16.

**Key points to mention:**
- Pre-norm downside: the residual stream grows with depth, so later blocks contribute relatively less.
- Some results suggest a successfully trained post-norm model can be slightly better; hybrids like DeepNorm try to get both.
- Norms are memory-bound, so inference engines fuse "add residual + RMSNorm" into one kernel.
- RMSNorm example: $[1, 2, 3, 6]$ has RMS about 3.536, output about $[0.283, 0.566, 0.849, 1.697]$.

**Deep dive:** [Vanishing and Exploding Gradients, Initialization, Normalization and Residuals](../02-deep-learning/05-vanishing-exploding-gradients-initialization-normalization-and-residuals.md), [The Full Transformer Architecture](../04-transformers/05-the-full-transformer-architecture.md)

</details>

<details>
<summary><b>Q24. A conv layer takes a 224x224x3 image with 64 filters of size 7x7, stride 2, padding 3. What is the output shape, and how many parameters and MACs does it have?</b></summary>

**Answer:** The output spatial size is $\lfloor (224 - 7 + 2 \cdot 3)/2 \rfloor + 1 = 112$, so the output is 64x112x112. Each filter spans the full input depth, so parameters are $64 \times (3 \times 7 \times 7) + 64 = 9{,}408 + 64 = 9{,}472$, or 9,408 without bias as in the ResNet stem, which is followed by BN. MACs are output positions times work per position: $112^2 \times 64 \times 3 \times 49 = 12{,}544 \times 9{,}408 \approx 118$ million. If the input doubles to 448x448, the parameters do not change, the output becomes 224x224 and the MACs grow 4x. The receptive field of one output unit after this layer alone is 7x7.

**Key points to mention:**
- General formulas: output $\lfloor (H - K + 2P)/S \rfloor + 1$; parameters $C_{out}(C_{in}K^2) + C_{out}$, independent of image size.
- Pitfalls: forgetting that padding applies to both sides ($2P$) or that filter depth equals the input channel count.
- Receptive field recurrence: $r_l = r_{l-1} + (k_l - 1)j_{l-1}$ with jump $j_l = j_{l-1}s_l$; the effective receptive field is smaller and center-weighted.
- Compute scales with pixel count, which is why resolution is a key latency knob.

**Deep dive:** [CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md)

</details>

<details>
<summary><b>Q25. What is a depthwise separable convolution, and why might a MobileNet not be much faster than a ResNet on a GPU?</b></summary>

**Answer:** It factorizes a standard convolution into a depthwise conv (one $K \times K$ filter per input channel, no channel mixing) followed by a pointwise 1x1 conv that mixes channels. Cost per position drops from $K^2 C_{in} C_{out}$ to $K^2 C_{in} + C_{in} C_{out}$: for 3x3 with 64 channels in and out that is 36,864 versus 4,672 multiply-adds, about 7.9x fewer (the general ratio is $1/C_{out} + 1/K^2$). But depthwise convolutions have low arithmetic intensity, so on GPUs they are memory-bound and far from peak throughput; FLOP savings do not translate proportionally into latency. On mobile CPUs and NPUs with optimized kernels the savings are real, which is why MobileNet and EfficientNet-style blocks dominate the edge. The takeaway is to choose architectures by measured latency on the target hardware and batch size, not by FLOPs.

**Key points to mention:**
- MobileNetV2 adds inverted residuals and linear bottlenecks; EfficientNet's MBConv builds on them.
- Same lesson as unstructured pruning: fewer nominal operations do not mean lower latency.
- Benchmark p50/p95 latency on the real device, at the production batch size, after export.

**Deep dive:** [CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md)

</details>

<details>
<summary><b>Q26. Why do vanilla RNNs struggle with long-range dependencies, and how does an LSTM help?</b></summary>

**Answer:** RNNs are trained with backpropagation through time, and the gradient from step $T$ back to step 1 is a product of $T - 1$ per-step Jacobians involving $W_{hh}$ and the tanh derivative. If those factors are typically below 1 the gradient vanishes exponentially ($0.5^{20} \approx 10^{-6}$), above 1 it explodes ($1.5^{20} \approx 3{,}325$), so distant inputs get essentially no learning signal; it is an optimization problem, not a capacity problem. The LSTM adds a cell state updated additively, $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, so the direct gradient from $c_t$ to $c_{t-1}$ is just the forget gate $f_t$, with no repeated weight-matrix multiplication. If the network learns $f_t \approx 1$, gradients flow across many steps nearly unchanged ($0.99^{100} \approx 0.37$ versus $10^{-30}$ for a factor of 0.5). It does not prevent exploding gradients through the gate and hidden-state paths, so gradient clipping is still standard.

**Key points to mention:**
- Initialize the forget-gate bias to about 1 so memory is retained early in training; PyTorch does not do this by default.
- GRU: two gates, about 25% fewer parameters than LSTM; neither wins consistently, so let validation decide.
- Truncated BPTT detaches state every $k$ steps: bounded memory, but no gradient for dependencies longer than $k$.
- Same principle as residual connections: an additive, identity-like path for gradients.

**Deep dive:** [RNNs, LSTMs and the Road to Transformers](../02-deep-learning/08-rnns-lstms-and-the-road-to-transformers.md)

</details>

<details>
<summary><b>Q27. Why did transformers replace RNNs, and what did we give up?</b></summary>

**Answer:** Three reasons. Parallelism: an RNN layer needs $n$ sequential steps, while self-attention processes all positions at once with large matrix multiplies, and with teacher forcing plus a causal mask the whole sequence trains in one parallel pass. Path length: any two tokens are one attention hop apart instead of $O(n)$ steps, so long-range dependencies are far easier to learn. Scaling: these two properties made it practical to pretrain huge models on huge data, and quality kept improving with scale. What we gave up is constant memory and constant per-token cost at inference: attention is quadratic in sequence length and the KV cache grows linearly (about 0.5 MiB per token for a Llama-2-7B-style model in FP16). Decoding is sequential for both architectures; the transformer's speedup is in training and prefill.

**Key points to mention:**
- Vaswani et al.: self-attention $O(n^2 d)$ compute with $O(1)$ sequential ops and path length; recurrence $O(nd^2)$ with $O(n)$ for both.
- Self-attention has no notion of order, so positional encodings are needed.
- State-space models (Mamba) recover linear training and constant-state inference, but fixed state makes exact recall harder, hence hybrids.
- RNNs/LSTMs still fit streaming, tiny on-device models and some time-series problems.

**Deep dive:** [RNNs, LSTMs and the Road to Transformers](../02-deep-learning/08-rnns-lstms-and-the-road-to-transformers.md), [KV Cache Explained](../06-llm-inference/02-kv-cache-explained.md)

</details>

## 6. Mixed precision and distributed training

<details>
<summary><b>Q28. What are mixed-precision training and distributed training?</b></summary>

**Answer:** Mixed-precision training runs most computation, mainly the matrix multiplies and stored activations, in a 16-bit format (FP16 or BF16), while keeping numerically sensitive parts in FP32: the master copy of the weights, optimizer states, and softmax, normalization, reductions and the loss. It gives tensor-core speedups (A100: 312 TFLOPS dense FP16/BF16 versus 19.5 TFLOPS plain FP32) and roughly halves activation memory, with FP16 additionally needing loss scaling. Distributed training splits one training job across many GPUs to beat two walls: compute (training costs about $6ND$ FLOPs) and memory (a 7B model needs about 112 GB of model states with Adam). The main strategies are data parallelism (replicate the model, split the batch, all-reduce gradients), model parallelism (tensor parallelism within layers, pipeline parallelism across layers), and sharded data parallelism (ZeRO/FSDP), often combined as 3D parallelism. Both are verified the same way: throughput and memory must improve while the validation curve tracks an FP32 or single-GPU baseline.

**Key points to mention:**
- Mixed precision does not halve total training memory with Adam: weights + grads + states stay at about 16 bytes per parameter.
- Measure distributed efficiency with tokens/s/GPU, scaling efficiency and MFU; communication and stragglers limit it.
- A bigger global batch means the learning rate and schedule may need retuning.
- Typical follow-ups: FP32 vs FP16 vs BF16 (Q29), loss scaling (Q30), gradient accumulation (Q31), data vs model parallelism (Q32).

**Deep dive:** [Mixed-Precision Training](../02-deep-learning/09-mixed-precision-training.md), [Distributed Training and Training Memory](../02-deep-learning/10-distributed-training-and-training-memory.md)

</details>

<details>
<summary><b>Q29. Compare FP32, FP16, and BF16.</b></summary>

**Answer:** FP32 has 1 sign, 8 exponent and 23 mantissa bits: a huge range (up to about $3.4 \times 10^{38}$) and about 7 decimal digits of precision. FP16 has 5 exponent and 10 mantissa bits: more precision (about 3.3 digits, machine epsilon about $9.8 \times 10^{-4}$) but a narrow range, with a maximum of 65,504 and underflow below about $6 \times 10^{-8}$. BF16 has 8 exponent and 7 mantissa bits: the same range as FP32 but only about 2.4 digits (epsilon about $7.8 \times 10^{-3}$). FP16 overflows and underflows easily and needs loss scaling; BF16 is robust and needs no scaling but is coarser, so BF16 is the default for LLM training on Ampere and later GPUs and TPUs. FP32 remains the format for master weights, optimizer states and accumulations.

**Key points to mention:**
- BF16 is the upper half of FP32, so conversion is cheap with no range loss; a BF16-trained model can break when served in FP16 because large values overflow.
- BF16 precision failure: adding 0.01 to a BF16 accumulator stalls at 4.0, where the gap is $2^{-5}$.
- TF32 is a tensor-core math mode (FP32 range, 10 mantissa bits) for FP32 matmuls; tensors stay FP32 in memory.
- FP8 (E4M3 for weights/activations, E5M2 for gradients) needs per-tensor or per-block scales.

**Deep dive:** [Mixed-Precision Training](../02-deep-learning/09-mixed-precision-training.md), [Number Formats and Model Memory](../07-model-compression/01-number-formats-and-model-memory.md)

</details>

<details>
<summary><b>Q30. Why can FP16 require loss scaling?</b></summary>

**Answer:** FP16 has only 5 exponent bits, so its smallest positive value is about $6 \times 10^{-8}$ (and full precision only above about $6 \times 10^{-5}$). Many gradients, especially activation gradients in deep networks, are smaller than that, underflow to zero, and the corresponding parameters stop learning. Multiplying the loss by a factor $S$ multiplies every gradient by $S$ through the chain rule, moving them into the representable range; you divide by $S$ in FP32 before the optimizer step, so the update is unchanged. Example: $g = 2 \times 10^{-8}$ becomes 0 in FP16, but with $S = 2^{16}$ it is 0.00131, comfortably representable. Too large a scale overflows (a gradient of 2.0 times 65,536 exceeds 65,504), so dynamic scaling starts at $2^{16}$, skips the step and halves on inf/NaN, and doubles after 2,000 good steps.

**Key points to mention:**
- "Can" because it depends on the gradient distribution: in the original paper, ImageNet CNNs matched FP32 without scaling while an SSD detector needed it.
- Powers of 2 make scaling and unscaling exact.
- BF16 does not need it because it has FP32's exponent range.
- Unscale before gradient clipping; call `scaler.step` and `scaler.update` once per optimizer step with accumulation; log the scale and skipped-step rate.

**Deep dive:** [Mixed-Precision Training](../02-deep-learning/09-mixed-precision-training.md)

</details>

<details>
<summary><b>Q31. What is gradient accumulation?</b></summary>

**Answer:** Gradient accumulation runs several micro-batches through forward and backward, summing their gradients in the `.grad` buffers, and calls the optimizer step only after $A$ micro-batches. It simulates a large batch with the activation memory of a small one, trading time for memory: there is no speedup, and each optimizer step takes about $A$ times longer. The effective batch is micro-batch x accumulation steps x data-parallel GPUs, for example $8 \times 4 \times 8 = 256$. Each micro-batch loss must be divided by $A$ so the summed gradient equals the large-batch mean: per-example gradients 1, 2, 3, 4 have mean 2.5, and two micro-batch means 1.5 and 3.5 sum to 5.0 unscaled but to 2.5 when each is halved. With DDP, use `no_sync()` on non-final micro-steps so gradients are all-reduced once per optimizer step.

**Key points to mention:**
- Not exactly equivalent to a big batch with BatchNorm (per-micro-batch statistics) or with LLM losses averaged per micro-batch over different token counts (normalize by total tokens).
- Clip, step the scheduler and update the loss scaler per optimizer step, not per micro-step.
- Double-scaling bug: Accelerate and DeepSpeed already apply $1/A$, so dividing again silently shrinks the learning rate.
- Also fills the pipeline in pipeline parallelism.

**Deep dive:** [Distributed Training and Training Memory](../02-deep-learning/10-distributed-training-and-training-memory.md)

</details>

<details>
<summary><b>Q32. How do data parallelism and model parallelism differ?</b></summary>

**Answer:** Data parallelism gives every GPU a full copy of the model and a different slice of the batch, then averages gradients with an all-reduce once per step (overlapped with backward); it speeds up training but does not reduce per-GPU memory. Model parallelism splits the model itself: tensor parallelism splits weight matrices inside each layer and needs activation all-reduces on the critical path (4 per layer in Megatron style), so it needs NVLink and stays within a node; pipeline parallelism splits groups of layers into stages that pass activations point to point, with idle "bubble" time of about $(p-1)/(m+p-1)$ for $p$ stages and $m$ micro-batches. ZeRO/FSDP sit in between: data-parallel compute with sharded optimizer states, gradients and parameters. Large runs combine them: tensor parallelism inside a node, pipeline across nodes, data parallelism (with ZeRO-1 or FSDP) across replicas. To choose, estimate memory first; if it fits on one GPU use DDP, otherwise try FSDP/ZeRO before TP/PP, which require model-code changes.

**Key points to mention:**
- Calling FSDP "model parallelism" is a common mistake: each rank still computes full layers on its own data after all-gathering weights.
- Pipeline bubble example: $p = 4$ with $m = 4$ is 43% idle, with $m = 32$ about 8.6%.
- Failure modes: DP runs out of memory, TP across nodes stalls on the network, PP with an unbalanced stage sets everyone's pace.
- Scaling from 8 GPUs in one node to 64 across 8 nodes often yields far less than 8x; profile communication and check NCCL uses InfiniBand.

**Deep dive:** [Distributed Training and Training Memory](../02-deep-learning/10-distributed-training-and-training-memory.md)

</details>

<details>
<summary><b>Q33. Estimate the GPU memory to fully fine-tune a 7B model with AdamW in mixed precision. Will it fit on 8 x 80 GB GPUs?</b></summary>

**Answer:** Model states cost about 16 bytes per parameter: 2 (BF16 weights) + 2 (BF16 gradients) + 4 (FP32 master weights) + 4 + 4 (Adam $m$ and $v$), so $16 \times 7 \times 10^9 = 112$ GB before activations. That does not fit on one 80 GB GPU, and plain DDP on 8 GPUs does not help because each GPU still holds all 112 GB. With ZeRO-1 it is about 38.5 GB per GPU, with ZeRO-2 about 26 GB, and with ZeRO-3/FSDP about 14 GB, which leaves room for activations. Activations add roughly 18 GB for a 4,096-token sequence over 32 layers in this illustration, under 2 GB with activation checkpointing. Alternatives are 8-bit Adam (moments down to about 2 bytes per parameter), LoRA or QLoRA (optimizer state only for adapters, fits on one GPU), and CPU offload.

**Key points to mention:**
- State assumptions out loud: optimizer, precision, full vs parameter-efficient, sequence length, micro-batch.
- Leave 10 to 20% headroom for CUDA context, NCCL buffers and fragmentation.
- Inference-only weights: 28 GB FP32, 14 GB BF16, 7 GB FP8.
- Resumable checkpoints carry the optimizer state too, which affects storage.

**Deep dive:** [Distributed Training and Training Memory](../02-deep-learning/10-distributed-training-and-training-memory.md), [Optimizers: SGD, Momentum, Adam, AdamW and Learning Rate Schedules](../02-deep-learning/03-optimizers-sgd-momentum-adam-and-learning-rate-schedules.md), [Fine-Tuning: Full, LoRA and QLoRA](../05-llms/07-fine-tuning-full-lora-and-qlora.md)

</details>

<details>
<summary><b>Q34. Explain ZeRO stages 1, 2 and 3 and how FSDP relates. What does each cost in communication?</b></summary>

**Answer:** ZeRO removes the redundant copies that data parallelism keeps on every rank. Stage 1 shards optimizer states, stage 2 also shards gradients, and stage 3 also shards the parameters themselves. Stages 1 and 2 cost about the same communication as DDP ($2\Psi$ per step), because an all-reduce is just a reduce-scatter plus an all-gather; stage 3 costs about $3\Psi$ (1.5x) because weights are all-gathered in both the forward and the backward pass. PyTorch FSDP is the native ZeRO-3 equivalent (`FULL_SHARD`), with `SHARD_GRAD_OP` similar to ZeRO-2 and `HYBRID_SHARD` sharding within a node and replicating across nodes to keep heavy traffic on NVLink. Per layer, ZeRO-3 all-gathers the weights, computes, frees them, and in backward all-gathers again, computes gradients and reduce-scatters them.

**Key points to mention:**
- Per-GPU memory: DDP $(2 + 2 + K)\Psi$, Z1 $4\Psi + K\Psi/N_d$, Z2 $2\Psi + (2+K)\Psi/N_d$, Z3 $(4+K)\Psi/N_d$, with $K = 12$ for mixed-precision Adam.
- Wrap per transformer block and prefetch the next block to overlap communication with compute.
- ZeRO does not shard activations; long sequences still need checkpointing or context parallelism.
- ZeRO-Offload/Infinity move states to CPU or NVMe for few-GPU setups at a speed cost.

**Deep dive:** [Distributed Training and Training Memory](../02-deep-learning/10-distributed-training-and-training-memory.md)

</details>

<details>
<summary><b>Q35. What is activation checkpointing and what does it cost?</b></summary>

**Answer:** Activation (gradient) checkpointing stores only selected activations during the forward pass, usually each transformer block's input, and recomputes the rest during the backward pass when they are needed. It cuts activation memory dramatically, for example from about 18 GB to under 2 GB in the course's 7B illustration. The cost is roughly one extra forward pass, about 33% more compute with full checkpointing, because a training step goes from about 3 units of work (forward 1, backward 2) to about 4. Selective recomputation only recomputes cheap but memory-heavy ops such as the attention softmax and dropout for a few percent overhead. The net effect can even be positive: freed memory allows larger micro-batches and better GPU utilization.

**Key points to mention:**
- Chen et al. (2016): checkpointing every $\sqrt{n}$ layers gives $O(\sqrt{n})$ activation memory.
- APIs: `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`, `model.gradient_checkpointing_enable()` in Hugging Face, FSDP and DeepSpeed integrations.
- It is the standard answer to "OOM in the backward pass" together with smaller micro-batches plus accumulation, BF16, LoRA and sharding.

**Deep dive:** [Distributed Training and Training Memory](../02-deep-learning/10-distributed-training-and-training-memory.md), [Backpropagation and Gradient Descent](../02-deep-learning/02-backpropagation-and-gradient-descent.md)

</details>

## 7. Vision tasks, detection and segmentation metrics

<details>
<summary><b>Q36. Compare image classification, object detection, and image segmentation.</b></summary>

**Answer:** Classification assigns a label (or labels) to the whole image and answers "what". Detection finds each object and returns a class, a bounding box and a confidence score: "what, roughly where, and how many". Segmentation labels every pixel: "what and exactly which pixels", with semantic segmentation merging same-class objects and instance segmentation separating them. Moving down the list, outputs get richer ($K$ numbers, then a variable-length list of boxes, then an $H \times W$ map), labels get more expensive, compute and post-processing grow, and the metric changes from accuracy/F1 to mAP to mIoU/Dice. Choose the cheapest formulation that supports the downstream action: a yes/no decision needs classification, counting needs detection, measuring area needs segmentation.

**Key points to mention:**
- Heads: global pooling + linear for classification; dense predictions plus box regression and NMS or set matching for detection; per-pixel classifier from an encoder-decoder (U-Net) for segmentation.
- You can derive coarser outputs from finer ones (boxes from masks), not the reverse, so over-labeling "just in case" is a cost trade-off to discuss.
- Failure modes: classifiers right for the wrong reason (snow behind a wolf), detectors weak on tiny or crowded objects, segmentation capped by boundary label quality.
- Cascades (cheap model first, finer model on crops) are common because serving cost rises down the list.

**Deep dive:** [Computer Vision Tasks: Classification, Detection and Segmentation](../03-computer-vision/01-image-classification-detection-and-segmentation.md)

</details>

<details>
<summary><b>Q37. What are semantic and instance segmentation, and how do they differ from panoptic segmentation?</b></summary>

**Answer:** Semantic segmentation assigns a class to every pixel but does not distinguish objects of the same class, so two adjacent cars become one "car" region. Instance segmentation detects each object and gives it its own mask, so car 1 and car 2 are separate, but it usually ignores amorphous background "stuff" like road or sky. Panoptic segmentation unifies them: every pixel gets a class, and pixels of countable "things" also get an instance ID. Architectures follow: encoder-decoder per-pixel classifiers (FCN, U-Net, DeepLab) for semantic, detect-then-segment (Mask R-CNN with a 28x28 per-RoI mask) for instance, and mask-classification models such as Mask2Former for panoptic. Metrics also differ: mIoU or Dice for semantic, mask AP for instance, and $\text{PQ} = \text{SQ} \times \text{RQ}$ for panoptic.

**Key points to mention:**
- Use semantic for area or drivable space, instance for counting with shape (cells, grasping), panoptic for full scene understanding.
- Instances from semantic masks only work when objects do not touch (connected components); touching cells merge.
- Mask R-CNN's RoIAlign samples exact fractional positions with bilinear interpolation, fixing RoIPool's misalignment; 28x28 masks are blobby on large objects.
- Instance and panoptic labels are the most expensive; model-assisted (SAM-style) labeling is the main cost lever.

**Deep dive:** [Computer Vision Tasks: Classification, Detection and Segmentation](../03-computer-vision/01-image-classification-detection-and-segmentation.md)

</details>

<details>
<summary><b>Q38. How do IoU, precision, recall, and mAP measure performance?</b></summary>

**Answer:** IoU measures overlap between a predicted and a true region: intersection area over union area, from 0 to 1. In detection, predictions are sorted by score and matched greedily, each to at most one unclaimed ground truth of the same class with IoU above a threshold (say 0.5); matches are true positives, the rest false positives, and unmatched ground truths false negatives. Precision is TP/(TP+FP), the share of predictions that are right, and recall is TP/(TP+FN), the share of objects found. Sweeping the confidence threshold traces a precision-recall curve; AP is the area under its interpolated version for one class, and mAP averages AP over classes, and in COCO also over IoU thresholds 0.5 to 0.95. Worked IoU: boxes (1,1,5,5) and (3,2,7,6) have intersection 6 and union 26, IoU 0.23, so not a match at 0.5.

**Key points to mention:**
- There are no true negatives in detection, so accuracy and ROC are not used; duplicates count as FPs, which is why NMS matters.
- mAP is threshold-free and weights classes equally; in production also report per-class precision and recall at the deployed threshold.
- IoU is harsh on small objects: a 2-pixel shift gives IoU 0.667 on a 10-pixel box but 0.961 on a 100-pixel box, so report AP by size.
- Only compare numbers from the same evaluator (VOC 11-point, all-point, COCO 101-point); TIDE splits the gap by error type.

**Deep dive:** [Computer Vision Tasks: Classification, Detection and Segmentation](../03-computer-vision/01-image-classification-detection-and-segmentation.md), [Object Detection in Depth: Faster R-CNN, YOLO, DETR and mAP](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md)

</details>

<details>
<summary><b>Q39. When would you use YOLO versus Faster R-CNN?</b></summary>

**Answer:** I choose a YOLO-family one-stage detector when latency and deployment simplicity matter most: real-time video, edge devices, many streams per GPU. It does dense prediction in a single pass followed by NMS. I choose Faster R-CNN (usually with FPN, or Mask/Cascade R-CNN) when accuracy on hard cases matters more than speed: precise localization at strict IoU, many small or overlapping objects, offline batch processing, or RoI-level extensions such as instance masks; its RPN proposes regions and the second stage classifies and refines each one with RoIAlign features. At 30 FPS the budget is about 33 ms per frame for everything, including preprocessing and NMS, which a high-resolution two-stage model often cannot meet on edge hardware. Today the benchmark accuracy gap is small, so I compare candidates on my own target-camera data: mAP@[.5:.95], AP_S, per-class recall at the deployed threshold, and p50/p95 end-to-end latency on the real hardware.

**Key points to mention:**
- Do not attribute differences caused by backbone, resolution or training recipe to "one-stage vs two-stage".
- RT-DETR-style models are a real-time, NMS-free alternative.
- Licensing matters: Ultralytics YOLO is AGPL-3.0 unless licensed commercially; torchvision (BSD) and Detectron2 (Apache 2.0) are permissive.
- Failure modes: one-stage at low resolution misses tiny or packed objects; capping two-stage proposals silently lowers recall in busy scenes.

**Deep dive:** [Object Detection in Depth: Faster R-CNN, YOLO, DETR and mAP](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md)

</details>

<details>
<summary><b>Q40. Why is mAP@[0.5:0.95] so much lower than mAP@0.5, and which would you report?</b></summary>

**Answer:** mAP@[.5:.95] averages AP over ten IoU thresholds from 0.5 to 0.95, and at strict thresholds only nearly perfect boxes count, so correct but loose boxes turn into false positives. In the course's worked example, AP fell from 0.683 at IoU 0.5 to 0.40 at IoU 0.75, and the average over all ten thresholds was only about 0.37. The stricter metric therefore also measures localization quality, not just detection. I report both plus AP by object size, and choose the primary metric from the product: mAP@0.5 if we only need to know an object is present, COCO-style AP if downstream steps depend on tight boxes (measuring, cropping for OCR, grasping, tracking). If annotators disagree by a few pixels, AP at 0.9 to 0.95 is capped regardless of the model, worst of all for small objects.

**Key points to mention:**
- Report hygiene: dataset split, input resolution, evaluator, TTA on or off.
- Stakeholders usually care most about per-class precision and recall at the deployed threshold.
- Appending low-confidence detections can never lower AP, which is why COCO caps at 100 detections per image.

**Deep dive:** [Object Detection in Depth: Faster R-CNN, YOLO, DETR and mAP](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md)

</details>

<details>
<summary><b>Q41. Explain NMS. When does it fail, and what are the alternatives?</b></summary>

**Answer:** Dense detectors produce many overlapping boxes for the same object, because neighboring anchors or grid cells all respond to it. Greedy non-maximum suppression sorts boxes by score, keeps the best, deletes others of the same class whose IoU with it exceeds a threshold (often 0.5 to 0.7), and repeats. It fails in crowded scenes: genuinely different objects that overlap heavily (people in a crowd, parked cars) are suppressed if the threshold is low, while a high threshold leaves duplicates that count as false positives. It is also a production concern: it is sequential and data-dependent, often runs on CPU, its cost grows with the number of candidates so crowded frames blow up p95 latency, and it complicates export to some runtimes. Alternatives are Soft-NMS (decay scores instead of deleting), weighted or cluster NMS, class-agnostic NMS for confusable classes, and NMS-free one-to-one detectors such as DETR or YOLOv10.

**Key points to mention:**
- Tune the NMS IoU threshold on validation data per domain, and use a pre-NMS top-k cap to bound cost.
- Batched NMS offsets coordinates per class to run all classes in one call.
- Tiling (SAHI-style) needs a final NMS across tile borders.

**Deep dive:** [Object Detection in Depth: Faster R-CNN, YOLO, DETR and mAP](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md), [Computer Vision Tasks: Classification, Detection and Segmentation](../03-computer-vision/01-image-classification-detection-and-segmentation.md)

</details>

<details>
<summary><b>Q42. Anchor-based vs anchor-free detectors, and how does DETR avoid NMS?</b></summary>

**Answer:** Anchor-based detectors place template boxes of fixed sizes and aspect ratios at every location and regress offsets from them; anchor-free detectors predict boxes directly from points, as distances to the edges (FCOS) or center heatmaps (CenterNet). The field moved to anchor-free because anchors add dataset-specific hyperparameters and many wasted predictions, and ATSS showed that the accuracy difference mostly comes from label assignment, not from the anchors themselves. DETR goes further: it predicts a fixed set of $N$ outputs from learned object queries and trains with a set loss, using the Hungarian algorithm to find the minimum-cost one-to-one matching (class + L1 + GIoU) and training unmatched queries to predict "no object". Because each object has exactly one positive prediction and queries interact through self-attention, the model learns not to emit duplicates, so no NMS is needed. The original DETR converged very slowly and was weak on small objects, largely fixed by Deformable DETR, DINO-style denoising and RT-DETR.

**Key points to mention:**
- NMS exists because of one-to-many assignment; one-to-one assignment removes the need (YOLOv10 borrowed this idea).
- Anchor-free ambiguity when a point lies in several boxes is resolved by assigning to the smallest box or by FPN level.
- An end-to-end graph without data-dependent post-processing exports more easily and has more predictable latency.
- DETR caps detections at $N$ per image; pick $N$ above the maximum expected object count.
- Focal loss ($\gamma = 2$, $\alpha = 0.25$) is what let dense one-stage detectors like RetinaNet cope with about 100,000 mostly easy background anchors per image.

**Deep dive:** [Object Detection in Depth: Faster R-CNN, YOLO, DETR and mAP](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md)

</details>

<details>
<summary><b>Q43. Explain U-Net. Why does it use skip connections, and how are they different from ResNet's?</b></summary>

**Answer:** U-Net is an encoder-decoder for segmentation: the encoder downsamples to learn "what" is in the image, and the decoder upsamples back to full resolution to recover "where". Skip connections concatenate encoder feature maps into the decoder at matching resolutions, handing it the fine spatial detail lost during downsampling, so boundaries stay sharp. ResNet's residual connections instead add a block's input to its output at the same resolution to ease gradient flow; U-Net's skips concatenate across the network to restore spatial information. It was designed for biomedical images with very few training samples (Ronneberger et al., 2015) and remains popular because it is simple, data-efficient, easy to extend to 3-D volumes, and was the standard denoiser backbone in diffusion models for years. Upsampling uses transposed convolution (learned, can cause checkerboard artifacts) or bilinear upsampling followed by a conv.

**Key points to mention:**
- Input size must be divisible by $2^{\text{levels}}$ so shapes align at concatenation.
- Alternatives: FCN, DeepLab (atrous convolution + ASPP), FPN-based decoders, Mask2Former.
- For ragged boundaries, upsample logits bilinearly then argmax, rather than nearest-neighbor upsampling of low-resolution masks.

**Deep dive:** [Computer Vision Tasks: Classification, Detection and Segmentation](../03-computer-vision/01-image-classification-detection-and-segmentation.md)

</details>

## 8. Transfer learning, augmentation and small datasets

<details>
<summary><b>Q44. What are transfer learning and catastrophic forgetting?</b></summary>

**Answer:** Transfer learning reuses a model trained on a large source task (for example ImageNet) as the starting point for a new target task, either by freezing the backbone and training a new head (feature extraction) or by continuing to train some or all layers with small learning rates (fine-tuning). It works because early layers learn general features such as edges and textures, while later layers are more task-specific. Catastrophic forgetting is when training on a new task abruptly destroys performance on what the model knew before, because gradients from the new task overwrite shared weights with nothing protecting the old task. Mitigations, roughly in order of practical reliability, are replaying a sample of old data, parameter isolation (separate heads, adapters, LoRA on a frozen base), small learning rates and freezing, regularization toward the old weights (L2-SP, EWC), and distillation from the old model. Measure forgetting with a held-out old-task evaluation set before and after every retrain.

**Key points to mention:**
- Always establish a linear-probe baseline; then LP-FT (train the head first, then unfreeze) with discriminative learning rates.
- Negative transfer: if source and target differ too much, pretrained init can underperform from-scratch training; compare against a from-scratch baseline.
- Make old-task metrics a release gate; when adding classes, check per-class recall because the head favors the newest classes.
- The same ideas apply to LLM fine-tuning eroding general or safety behavior.

**Deep dive:** [CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md), [Catastrophic Forgetting, Contamination and Memorization](../05-llms/09-catastrophic-forgetting-contamination-and-memorization.md)

</details>

<details>
<summary><b>Q45. How do transfer learning and data augmentation improve computer vision models?</b></summary>

**Answer:** Both attack overfitting when labeled data is limited, from different directions. Transfer learning starts from a backbone pretrained on millions of images, so the model already has general visual features and only has to learn the task-specific mapping, which needs far fewer labels and steps. Data augmentation creates label-preserving variations of the training images (flips, crops, color and lighting changes, scale jitter), effectively enlarging the dataset and encoding invariances the model should have, $f(T(x)) = f(x)$. They interact: augmentation still matters while fine-tuning, and in self-supervised pretraining (SimCLR, DINO) augmentation defines what the representation learns to ignore. Gains depend on domain similarity: ImageNet features help less on X-rays or microscopy, where in-domain self-supervised pretraining or deeper fine-tuning helps more.

**Key points to mention:**
- Measure each contribution by ablation on the same grouped folds and several seeds, comparing per-class recall and fold-to-fold spread.
- Pitfalls: label-breaking augmentations, mismatched preprocessing for pretrained weights, a fine-tuning learning rate that is too high.
- The pretrained weights fix the preprocessing (resolution, normalization, RGB order) that serving must reproduce exactly.
- Mixup and CutMix blend images and labels; they regularize and improve calibration but slow convergence and do not apply directly to detection labels.

**Deep dive:** [Data Augmentation, Transfer Learning and Small Datasets](../03-computer-vision/03-data-augmentation-transfer-learning-and-small-datasets.md)

</details>

<details>
<summary><b>Q46. When would you freeze or fine-tune a pretrained backbone?</b></summary>

**Answer:** It depends on dataset size and domain similarity. Small data and a similar domain: freeze the backbone and train a head (cheap, embeddings can be cached, hard to overfit). Small data and a different domain: start from a strong frozen backbone, then partially fine-tune with strong regularization, or use in-domain self-supervised pretraining. Medium or large data: partial or full fine-tuning, unfreezing top blocks first, with a backbone learning rate 10x to 100x below from-scratch training and below the head's, warmup, and discriminative rates $\eta_\ell = \eta_{\text{head}} d^{L-\ell}$. I always start with the frozen baseline because it sets a floor, and unfreeze more only when grouped validation shows a gain larger than the fold-to-fold spread. A fine-tuned model that is worse than the frozen baseline usually means the learning rate is too high or BatchNorm statistics drifted.

**Key points to mention:**
- LP-FT (Kumar et al., 2022) avoids a random head distorting features and helps out-of-distribution accuracy; full fine-tuning can lose OOD robustness, so evaluate on shifted data.
- Keep frozen BN layers in eval mode; with tiny batches keep BN frozen even while fine-tuning.
- Production: one frozen backbone can serve many heads from shared embeddings; every fully fine-tuned model is a separate artifact to version and monitor.
- For large ViTs, parameter-efficient methods (LoRA) are a middle ground.

**Deep dive:** [Data Augmentation, Transfer Learning and Small Datasets](../03-computer-vision/03-data-augmentation-transfer-learning-and-small-datasets.md), [CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md)

</details>

<details>
<summary><b>Q47. How does BatchNorm cause trouble during fine-tuning or deployment, and what do you do about it?</b></summary>

**Answer:** BatchNorm uses batch statistics in training mode and running statistics in eval mode, and its running mean and variance keep updating in training mode even when its weights are frozen. Common problems are forgetting `model.eval()` at inference (the same image gets different scores depending on the batch), tiny fine-tuning batches giving noisy statistics, and "frozen" layers whose statistics silently drift toward a small dataset because `model.train()` re-enabled updates. The fixes are to switch modes correctly, put BN of frozen blocks in eval mode after every `model.train()` call, use GroupNorm or SyncBatchNorm when per-GPU batches are small (typical in detection and segmentation), and fold BN into the convolutions for deployment, then compare original and exported outputs on the same inputs. Under real domain shift, deliberately re-estimating BN statistics on unlabeled target data with frozen weights is a cheap form of adaptation, as long as it is intentional.

**Key points to mention:**
- A big gap between train-mode and eval-mode accuracy is the symptom.
- Batch size 1 in train mode estimates variance from one image's spatial positions only.
- Folding: $W' = W\gamma/\sqrt{\sigma^2 + \epsilon}$, $b' = \gamma(b - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$.
- ViTs use LayerNorm and avoid the issue entirely.

**Deep dive:** [CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md), [Data Augmentation, Transfer Learning and Small Datasets](../03-computer-vision/03-data-augmentation-transfer-learning-and-small-datasets.md)

</details>

<details>
<summary><b>Q48. When can augmentation harm performance?</b></summary>

**Answer:** When it changes the true label, when it makes training images unlike deployment images, or when it is too strong for the model to fit. Label-changing examples: flipping text, rotating a 6 into a 9, flipping chest X-rays (the heart is on the left), hue jitter when color is the label, and random crops that cut out a small defect while the label still says "defect". Distribution mismatch: random-resized-crop changes apparent object size versus center-cropped test images, and mosaic seams or extreme warps create artifacts never seen at test time, which is why YOLO training turns mosaic off near the end. Strong policies can also cause underfitting with small models or frozen backbones, and average accuracy can rise while some classes drop. Leakage is another trap: augmented copies saved before splitting put near-duplicates on both sides.

**Key points to mention:**
- Augmenting only the oversampled minority class can create artifacts that correlate with the label (a shortcut).
- Detect harm by visualizing augmented samples and ablating each augmentation family on a grouped split, with per-class metrics across seeds.
- Validate targeted augmentation on real images of that condition (real night photos, not darkened day photos).
- For detection and segmentation, geometric transforms must be applied identically to boxes and masks (masks with nearest-neighbor interpolation).

**Deep dive:** [Data Augmentation, Transfer Learning and Small Datasets](../03-computer-vision/03-data-augmentation-transfer-learning-and-small-datasets.md)

</details>

<details>
<summary><b>Q49. How would you evaluate a model trained on a small dataset?</b></summary>

**Answer:** Split by group (patient, video, scene, site or physical object) so correlated images never cross splits, and deduplicate with perceptual hashes or embedding similarity. Use stratified grouped k-fold cross-validation (`StratifiedGroupKFold`) during development and report the mean and spread across folds, with nested CV if hyperparameters are tuned on the same folds. Keep a locked held-out test set from new groups that is scored once, and report confidence intervals: with 100 test images, 85% accuracy is really $0.85 \pm 1.96\sqrt{0.85 \times 0.15/100} \approx 0.85 \pm 0.07$. Use metrics that fit the class balance, such as per-class recall and macro-F1, and compare models with paired tests (McNemar) on the same test set and several seeds; overlapping intervals mean no claim. After launch, treat the small test set as a starting estimate and grow it with labeled production samples.

**Key points to mention:**
- Use a Wilson interval for small $n$, and a bootstrap that resamples whole groups for F1, AUROC or mAP.
- Augmentation only on training folds; evaluation preprocessing must match production.
- Break errors down by class and by group (camera, batch, site); one bad camera can hide behind a good average.
- A 95% CV score that becomes 70% in production usually means leakage across the split or distribution mismatch.

**Deep dive:** [Data Augmentation, Transfer Learning and Small Datasets](../03-computer-vision/03-data-augmentation-transfer-learning-and-small-datasets.md), [ML Workflow, Data Splits and Cross-Validation](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md)

</details>

## 9. Computer vision in production

<details>
<summary><b>Q50. How would you diagnose a vision model that performs well in validation but poorly in production?</b></summary>

**Answer:** I go from cheapest to most expensive. First confirm the gap is real by labeling a random production sample and computing the same metric at the same threshold, because complaint-driven examples are biased and label shift alone changes precision (recall 0.9 and FPR 0.05 give precision about 0.95 at 50% prevalence but about 0.49 at 5%). Then check pipeline parity by pushing identical raw images through the training and serving preprocessing (RGB vs BGR, normalization constants, resize/letterbox, interpolation, EXIF orientation, `model.eval()`, class-index mapping) and diffing tensors and logits. Then check whether validation was honest: near-duplicate frames or random splits instead of group splits inflate scores, so compare against a held-out-site split. Then slice production performance by camera, lighting, object size and time to localize the failure, audit labels in both training and evaluation sets, and only then apply targeted fixes (data collection, realistic augmentation, hard negatives, fine-tuning on production data, recalibrated thresholds) with a safe rollout and monitoring.

**Key points to mention:**
- Production accuracy is a weighted average of slice accuracies, so a change in mix (more night frames) lowers it with an unchanged model.
- A parity test in CI prevents the same skew from recurring.
- Success criterion: after fixing leakage, offline held-out-site scores and online labeled-sample scores should roughly agree.
- Pitfall: jumping to a bigger architecture before knowing the cause.
- Typical follow-ups: domain shift factors (Q51), annotation errors and leakage (Q52), small objects (Q53), latency (Q54).

**Deep dive:** [Computer Vision in Production](../03-computer-vision/04-computer-vision-in-production.md), [Reproducibility and Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md)

</details>

<details>
<summary><b>Q51. How do lighting, occlusion, resolution, and domain shift affect predictions?</b></summary>

**Answer:** CNNs rely heavily on low-level cues such as texture, color and local contrast, so anything that changes those cues hurts. Lighting changes (night, sensor noise, color loss, grayscale IR mode) break color-dependent rules like "yellow means helmet"; occlusion removes the parts the model depends on and makes boxes inconsistent; low resolution and compression erase detail, so a 24-pixel head in a 1920-wide frame shrinks to 8 pixels at a 640 input, a fraction of one stride-32 feature cell. Broader domain shift (new cameras, viewpoints, backgrounds) breaks spurious correlations learned in training and comes as covariate shift, label shift or concept shift. The usual result is lower recall, more false positives, and confidence that stays high even when the model is wrong, because calibration degrades under shift. Mitigate with real data from those conditions, targeted augmentation (brightness, noise, grayscale, random erasing, copy-paste), and measure with stress tests, slice metrics and a domain classifier (AUC near 0.5 means little shift).

**Key points to mention:**
- Augmentation buys invariance by throwing information away: grayscale helps IR frames but hurts if color is the only cue separating two classes, so recheck the day slices.
- Labeling rules for partial objects matter as much as the model under occlusion.
- Train on frames decoded from the real low-bitrate streams.
- Evidence to cite: ImageNet-C and the ImageNet re-collection study show accuracy drops under shift.

**Deep dive:** [Computer Vision in Production](../03-computer-vision/04-computer-vision-in-production.md)

</details>

<details>
<summary><b>Q52. How do you identify annotation errors and data leakage?</b></summary>

**Answer:** For annotation errors I use out-of-fold model predictions, because a network memorizes its own training labels and drives their loss to near zero. I sort by loss, apply confident learning (per-class thresholds equal to the mean predicted probability among examples labeled with that class, as in cleanlab), and audit detection boxes with IoU: confident predictions with no label suggest missing boxes, low-IoU matches suggest loose ones. Flagged items go to human review, and I measure inter-annotator agreement and track label statistics per batch, vendor and guideline version against a gold set. For leakage I check how the split was made (consecutive frames, burst photos, the same site, patient or product on both sides), find near-duplicates across splits with perceptual hashes (aHash/dHash with Hamming distance) and embedding cosine similarity (FAISS at scale), and compare scores on a random split versus a group split; a big drop confirms leakage. The fix is group splits, ideally holding out whole sites or cameras to mimic deployment.

**Key points to mention:**
- Never auto-delete flagged data; review it, since hard but correct examples also have high loss.
- Clean the test set too, or model comparisons become unreliable.
- Augmented copies saved before splitting are a classic hidden leak.
- Label definition drift across annotation batches looks like model regression.

**Deep dive:** [Computer Vision in Production](../03-computer-vision/04-computer-vision-in-production.md), [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md)

</details>

<details>
<summary><b>Q53. How would you improve small-object detection?</b></summary>

**Answer:** First measure: AP_S, the pixel-size distribution of missed objects at the model's input size, and whether small objects are even labeled consistently. Then give the model more pixels: a larger input size, overlapping tiling (SAHI-style), or a cascade that detects a larger context object and re-runs on a full-resolution crop. Architecturally, use feature pyramids (FPN) or add a stride-4 or stride-8 detection head, and make sure label assignment gives small objects enough positives (anchor sizes from k-means, ATSS-style adaptive assignment). On the data side, label more small objects, oversample them and use copy-paste, scale jitter or Mosaic augmentation, plus IoU-based losses (GIoU/CIoU). Every one of these costs latency: doubling the input side roughly quadruples compute, and tiling a 3840x2160 frame into 640 tiles with 20% overlap needs 32 forward passes plus cross-tile NMS.

**Key points to mention:**
- Verify with AP_S and recall by size bin, plus false positives per frame, since more pixels also means more background to fire on.
- Pitfall: lowering input resolution for latency hurts small objects badly while aggregate mAP barely moves.
- INT8 quantization often hurts small objects first, so re-evaluate AP_S after export.
- In video, tracking can confirm tiny detections across frames; a better camera or zoom is sometimes cheaper than ML.

**Deep dive:** [Object Detection in Depth: Faster R-CNN, YOLO, DETR and mAP](../03-computer-vision/02-object-detection-yolo-faster-rcnn-and-map.md), [Computer Vision in Production](../03-computer-vision/04-computer-vision-in-production.md)

</details>

<details>
<summary><b>Q54. How do you balance accuracy against inference latency?</b></summary>

**Answer:** Start from the requirement: a latency budget at p95 or p99, end to end, on the target hardware, for example 33.3 ms per frame per stream at 30 FPS. Profile the whole pipeline, because decode, preprocessing and NMS often cost as much as the model; by Amdahl's law, if the model is 20 of 50 ms, making it 2x faster only reaches 40 ms. Then build an accuracy-latency Pareto frontier of candidates (backbone sizes, input resolutions, precisions) measured on the production evaluation set, and pick the most accurate one within budget. Move the frontier with optimized runtimes (TensorRT, ONNX Runtime), FP16 and INT8 quantization, structured pruning, distillation, batching, and by doing less work (frame skipping, motion gating, tracking between detections, region-of-interest crops). Every optimization is gated on slice metrics, not just aggregate accuracy.

**Key points to mention:**
- Resolution 640 to 480 cuts conv compute to $0.75^2 = 0.5625$, but also shrinks objects.
- Batching raises throughput (illustrative: 100 to about 267 frames/s) while raising per-frame latency; dynamic batching with a short wait is the compromise.
- Unstructured pruning rarely speeds up dense hardware; only structured pruning reliably does.
- Measure correctly: warmup, `torch.cuda.synchronize()`, percentiles under realistic concurrent load, and long runs to catch thermal throttling on edge boxes.

**Deep dive:** [Computer Vision in Production](../03-computer-vision/04-computer-vision-in-production.md), [Pruning, Low-Rank Factorization and Choosing a Compression Method](../07-model-compression/07-pruning-low-rank-factorization-and-choosing-a-compression-method.md)

</details>

<details>
<summary><b>Q55. You have no production labels yet. How do you monitor a deployed vision model?</b></summary>

**Answer:** Monitor proxies at three levels. System health: latency percentiles, throughput, dropped frames and errors. Input health and drift: per-camera brightness, contrast, a blur score (variance of the Laplacian, near zero when blurry), frozen-feed detection (mean absolute difference between sampled frames near zero), and embedding drift or a domain-classifier AUC against a reference set. Output drift: detections per frame, class ratios, and confidence and box-size distributions per camera, which is often the earliest and most actionable signal (a camera whose detections per frame drop 8x). Compare each camera against its own baseline, since global thresholds across very different scenes produce constant false alarms or none. In parallel, start sampling frames for labeling so real quality metrics arrive with a delay, and capture operator feedback on alerts as weak labels.

**Key points to mention:**
- Drift does not always mean worse accuracy; it means "look here" and "sample here for labeling".
- Build the production evaluation set with random or stratified sampling and known weights; uncertainty sampling is great for training data but biased for measurement.
- Never train on the measurement pool.

**Deep dive:** [Computer Vision in Production](../03-computer-vision/04-computer-vision-in-production.md), [Monitoring ML and LLM Systems in Production](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md)

</details>

<details>
<summary><b>Q56. After INT8 quantization, overall mAP dropped only 0.5 points, but customers complain about night performance. What happened and what do you do?</b></summary>

**Answer:** The aggregate hid a slice regression. Post-training quantization picks value ranges from calibration data; if the calibration frames were mostly daytime, night activations (low contrast, different statistics) are poorly represented and get quantized coarsely. I would evaluate the quantized model per slice (day/night, IR, small objects), recalibrate with representative production frames that include night and IR, keep the most sensitive layers in higher precision, and move to quantization-aware training if that is not enough. The calibration method also matters: min-max ranges are widened by outliers and waste resolution, while percentile or entropy-based ranges usually do better. Going forward, slice metrics with explicit per-slice tolerances become a release gate for every optimization, not just aggregate mAP.

**Key points to mention:**
- Small objects and low-contrast inputs are typically the most sensitive to quantization.
- The same "aggregate hides slice" lesson applies to pruning, distillation and resolution changes.
- Keep the FP32/FP16 model available for quick rollback during a staged rollout.

**Deep dive:** [Computer Vision in Production](../03-computer-vision/04-computer-vision-in-production.md), [PTQ vs QAT and Calibration](../07-model-compression/03-ptq-vs-qat-and-calibration.md)

</details>

<details>
<summary><b>Q57. The model is confidently wrong on production images. Why, and how do you handle it?</b></summary>

**Answer:** Modern neural networks are often poorly calibrated, and calibration gets worse under distribution shift, so high confidence on unfamiliar images is common; spurious correlations (a yellow bucket classified as a helmet) make it worse. First measure calibration per slice with reliability diagrams or expected calibration error (ECE, the bin-weighted average gap between confidence and accuracy). Recalibrate on production validation data, for example with temperature scaling, which fixes overall overconfidence without changing accuracy, but fit and check it per slice, since one temperature fitted on day frames will not fix night frames. Add an out-of-distribution signal that does not rely only on softmax confidence (embedding distance from training data, a domain classifier), set per-segment thresholds, and route uncertain or unfamiliar frames to humans or a larger model. Ultimately, add hard negatives and production data from the unfamiliar domain to training.

**Key points to mention:**
- Never use raw softmax confidence as an OOD detector.
- Ensembles tend to give better uncertainty estimates at higher compute cost.
- An edge-plus-cloud hybrid can escalate uncertain frames to a bigger model, which also feeds labeling.

**Deep dive:** [Computer Vision in Production](../03-computer-vision/04-computer-vision-in-production.md)

</details>

## 10. Vision transformers, CLIP and multimodal models

<details>
<summary><b>Q58. Walk me through how a Vision Transformer turns a 224x224 image into tokens. How many tokens, and what happens if you halve the patch size?</b></summary>

**Answer:** ViT cuts the image into non-overlapping 16x16 patches, a 14x14 grid of 196 patches. Each patch is flattened to 768 numbers (16x16x3) and linearly projected to the model width (768 for ViT-B), which is equivalent to a convolution with kernel 16 and stride 16. A learned CLS token is prepended (197 tokens), learned position embeddings are added, and a standard transformer encoder runs. Halving the patch size to 8 gives 784 patch tokens: 4x the tokens and 16x the attention-matrix cost and memory. Total FLOPs rise only about 4.5x at ViT-B scale, because with $N \approx 200$ and $D = 768$ the per-token linear layers dominate; the quadratic term takes over only at high resolution or long video.

**Key points to mention:**
- Formula: $N = HW/P^2$; attention is $O(N^2)$ per layer, MLPs $O(N)$.
- Position embeddings are learned for a fixed grid; for a new resolution, interpolate them in 2D and usually fine-tune.
- Smaller patches help fine detail (text, small objects) at large cost; windowed attention (Swin) or token merging controls it.
- Pool features the way the model was trained (CLS or mean); prefer models with register tokens for dense features on large ViTs.

**Deep dive:** [Vision Transformers, CLIP and Multimodal Models](../03-computer-vision/05-vision-transformers-clip-and-multimodal-models.md)

</details>

<details>
<summary><b>Q59. Why did the original ViT need more data than a ResNet, and when would you still choose a CNN?</b></summary>

**Answer:** CNNs have strong inductive biases, locality and translation equivariance, that encode useful assumptions about images, so they learn efficiently from less data. ViT has weak biases: attention is global from the first layer and spatial structure must be learned, so with ImageNet-1k alone the original ViT lagged ResNets, and only with much larger pretraining did it match or beat them. DeiT later closed much of the gap with augmentation, regularization and distillation, and ConvNeXt showed modernized CNNs remain competitive, so the recipe matters as much as the family. I would still choose a CNN for small datasets trained from scratch, tight edge latency with mature INT8 tooling, or high-resolution dense tasks where global attention cost grows quadratically with the number of patches while conv cost grows linearly with pixels. I would choose a ViT or CLIP-style backbone when strong pretrained weights exist, when I need text alignment, or for the best accuracy at scale.

**Key points to mention:**
- In practice, choose a pretrained backbone by transfer accuracy on your validation set and latency on target hardware, not by architecture family.
- Self-supervised backbones (DINOv2, about 142 million curated images) often give the best frozen features for small labeled datasets.
- Hybrids (conv stems, windowed attention) blur the line.

**Deep dive:** [Vision Transformers, CLIP and Multimodal Models](../03-computer-vision/05-vision-transformers-clip-and-multimodal-models.md), [CNNs and Transfer Learning](../02-deep-learning/07-cnns-and-transfer-learning.md)

</details>

<details>
<summary><b>Q60. Explain CLIP's training objective and how it does zero-shot classification.</b></summary>

**Answer:** CLIP trains an image encoder and a text encoder so that matching image-caption pairs have high cosine similarity and mismatched pairs low. For a batch of $N$ pairs it computes an $N \times N$ similarity matrix, divides by a learned temperature, and applies cross-entropy in both directions (image to text and text to image) with the diagonal as the correct class, so every other item in the batch is a negative. Larger batches give more and harder negatives, and the temperature controls how sharply the softmax separates the correct pair; CLIP learns the logit scale, initialized at $1/0.07$ and capped at 100. For zero-shot classification, each class name becomes a prompt like "a photo of a {class}", the prompts are encoded once into class vectors, and an image is assigned to the class with the highest cosine similarity. If zero-shot accuracy is mediocre, fix ambiguous class names, use domain prompts and ensemble templates, then train a linear probe on a few hundred labeled images before considering fine-tuning.

**Key points to mention:**
- Inference cost is one image encoding plus a small matrix multiply, since class vectors are precomputed.
- Large batches require gathering embeddings across GPUs; the similarity matrix grows as $N^2$. SigLIP replaces the softmax with per-pair sigmoid losses.
- CLIP zero-shot is the wrong tool for counts, negation or fine attributes; use labeled data or a detector.
- Typographic attacks fool CLIP, so do not rely on CLIP-only checks for safety classification.

**Deep dive:** [Vision Transformers, CLIP and Multimodal Models](../03-computer-vision/05-vision-transformers-clip-and-multimodal-models.md), [Loss Functions](../02-deep-learning/04-loss-functions.md)

</details>

<details>
<summary><b>Q61. Describe the architecture and training of a LLaVA-style vision-language model. What is frozen and why?</b></summary>

**Answer:** A pretrained vision encoder (for example CLIP ViT-L/14 at 336 px) produces 576 patch features; a small MLP projector maps them into the LLM's embedding space; those image tokens are spliced into the text sequence, and a decoder-only LLM generates the answer. Stage 1 trains only the projector on image-caption pairs to align the two spaces, which is very cheap (about 21M parameters for a 2-layer 1024-to-4096 MLP). Stage 2 instruction-tunes the projector and the LLM on visual conversations, usually keeping the vision encoder frozen, often with LoRA. Freezing preserves the encoder's general visual features and the LLM's language skills; unfreezing the vision encoder late can help on domain images but risks degrading general features. In production, image tokens dominate cost: with tiling a scanned page can be around 2,900 tokens, so a 20-page PDF means tens of thousands of prefill tokens and a large KV cache, which is why you extract text, retrieve only relevant pages, and cap pages and pixels.

**Key points to mention:**
- Alternatives: BLIP-2's Q-Former compresses to a few query tokens; Flamingo uses a perceiver resampler and cross-attention; native multimodal models train one transformer on both modalities.
- Object hallucination comes from language priors overriding weak visual evidence; measure with POPE-style probes and reduce with higher resolution, crops and grounding checks.
- Treat text inside images as untrusted input: prompt injection through screenshots is a real risk for VLM agents.

**Deep dive:** [Vision Transformers, CLIP and Multimodal Models](../03-computer-vision/05-vision-transformers-clip-and-multimodal-models.md)

</details>

## Rapid-fire round

| Question | Answer |
|---|---|
| Expected initial cross-entropy loss for 10 balanced classes? | $\ln 10 \approx 2.30$; for binary $\ln 2 \approx 0.693$. |
| Binary loss stuck at exactly 0.693? | The model predicts 0.5 for everything: it is not learning (check gradients, labels, double sigmoid). |
| Maximum slope of the sigmoid? | 0.25, so gradient shrinks at least 4x per sigmoid layer. |
| He initialization variance? | $2/n_{\text{in}}$ (Xavier: $2/(n_{\text{in}} + n_{\text{out}})$). |
| Inverted dropout scaling factor? | Multiply survivors by $1/(1-p)$ in training; do nothing at inference. |
| Adam memory overhead per parameter? | Two FP32 moments, 8 bytes (56 GB for 7B). |
| Bytes per parameter for mixed-precision Adam training? | About 16 (2 weights + 2 grads + 4 master + 4 + 4 moments). |
| Linear scaling rule example? | $\eta = 0.1 \times B/256$, so 3.2 at batch 8,192, with warmup. |
| Why AdamW over Adam with L2? | Decay is decoupled from adaptive scaling, so all weights decay equally. |
| Typical gradient clipping threshold for LLMs? | Global norm 1.0. |
| Label smoothing with $\varepsilon = 0.1$, $K = 3$? | Target becomes about [0.933, 0.033, 0.033]; avoid it for distillation teachers. |
| Focal loss with $\gamma = 2$ at $p_t = 0.9$? | Keeps 1% of the CE loss; at $p_t = 0.1$ keeps 81%. |
| Dice in terms of IoU? | $\text{Dice} = 2\text{IoU}/(1 + \text{IoU})$; Dice is always at least IoU. |
| FP16 maximum value? | 65,504. |
| BF16 bit layout? | 1 sign, 8 exponent, 7 mantissa: FP32 range, about 2.4 digits. |
| Default dynamic loss scaling in PyTorch? | Start at $2^{16}$, halve on overflow, double after 2,000 good steps. |
| What is TF32? | Tensor-core mode for FP32 matmuls with 10 mantissa bits and FP32 range. |
| FP8 formats by role? | E4M3 (max 448) for weights and activations, E5M2 (max 57,344) for gradients. |
| Pipeline bubble fraction? | $(p-1)/(m+p-1)$; $p = 4$, $m = 32$ gives about 8.6%. |
| Why keep tensor parallelism inside a node? | Its activation all-reduces sit on the critical path and need NVLink bandwidth. |
| ZeRO-3 per-GPU model states for 7B on 8 GPUs? | $112/8 = 14$ GB. |
| Activation checkpointing compute overhead? | About 33% (one extra forward pass). |
| Conv layer parameter formula? | $C_{out}(C_{in}K^2) + C_{out}$, independent of image size. |
| Depthwise separable cost ratio? | $1/C_{out} + 1/K^2$, about 7.9x fewer MACs for 3x3, 64 channels. |
| Receptive field of two stacked 3x3 convs? | 5x5, with $18C^2$ weights instead of $25C^2$. |
| What does RoIAlign fix? | RoIPool's coordinate rounding, via bilinear sampling at exact positions. |
| Faster R-CNN RPN anchor rule? | IoU of at least 0.7 positive, below 0.3 negative, 256 sampled per image. |
| COCO mAP definition? | AP averaged over classes and 10 IoU thresholds from 0.5 to 0.95. |
| Panoptic quality? | $\text{PQ} = \text{SQ} \times \text{RQ}$ (segmentation quality times recognition quality). |
| Why no accuracy or ROC for detection? | There are no true negatives. |
| Per-frame budget at 30 FPS? | About 33 ms end to end. |
| Pruned 50% of weights, no speedup? | Unstructured sparsity; dense kernels still multiply zeros. Use structured pruning. |
| Test-time augmentation trade-off? | Small accuracy gain for K times the inference cost; fine offline, rarely real-time. |
| ViT-B/16 tokens for a 224x224 image? | 196 patches plus 1 CLS token, 197. |
| Latent diffusion compression for 512x512x3? | 64x64x4 latent, 48x fewer values. |
| Classifier-free guidance formula? | $\hat\epsilon = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})$. |
| Blur score for camera monitoring? | Variance of the Laplacian (near zero means blurry). |
| Bandwidth for 200 cameras at 4 Mbit/s to the cloud? | About 800 Mbit/s of continuous upload, a strong argument for edge or hybrid. |

## How to practice

- **Answer out loud first.** Read only the question, close the answer, and speak a full response before opening it. Writing or thinking silently hides gaps that speaking exposes.
- **Time yourself to about 2 minutes.** Lead with a one-sentence definition, then the mechanism, a trade-off, and one concrete number or example. Stop when time is up and note what you skipped.
- **Explain it to a non-expert.** If you cannot explain backprop, BatchNorm or mAP in plain words to a friend, you do not yet own the idea; go back to the linked chapter.
- **Then go deeper.** Use the "Key points to mention" as the interviewer's follow-ups: answer each one, and practice the debugging questions (Q16, Q17, Q50) as step-by-step walkthroughs.
- **Track weak spots.** Mark every question you could not answer fluently, revisit its deep-dive chapter, and retry it two days later. In the last week, run the rapid-fire table daily until every answer is instant.

