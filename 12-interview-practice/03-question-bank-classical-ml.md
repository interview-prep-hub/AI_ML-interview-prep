# Question Bank: Classical Machine Learning

> A revision bank of 50 interview questions (plus a rapid-fire round) on splits, leakage, metrics, linear models, trees, feature engineering, drift and classic algorithms. **How to use it:** read the question, answer out loud first, then open the answer and compare. Source chapters: [ML Workflow and Cross-Validation](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md), [Linear and Logistic Regression](../01-classical-ml/02-linear-and-logistic-regression.md), [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md), [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md), [Regression Metrics, Baselines and Model Selection](../01-classical-ml/05-regression-metrics-baselines-and-model-selection.md), [Decision Trees, Random Forests and Gradient Boosting](../01-classical-ml/06-decision-trees-random-forests-and-gradient-boosting.md), [Feature Engineering](../01-classical-ml/07-feature-engineering-missing-values-and-encoding.md), [Data Drift and Retraining](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md), [kNN, SVMs and Naive Bayes](../01-classical-ml/09-knn-svm-naive-bayes-and-other-classic-algorithms.md), [Unsupervised Learning](../01-classical-ml/10-unsupervised-learning-clustering-and-dimensionality-reduction.md), plus [Probability and Statistics](../00-foundations/04-probability-and-statistics.md).

## Table of contents

- [Overfitting, leakage and validation](#overfitting-leakage-and-validation)
- [Linear models and regularization](#linear-models-and-regularization)
- [Classification metrics and imbalanced data](#classification-metrics-and-imbalanced-data)
- [Regression metrics and model selection](#regression-metrics-and-model-selection)
- [Trees and ensembles](#trees-and-ensembles)
- [Feature engineering: missing values, encoding and scaling](#feature-engineering-missing-values-encoding-and-scaling)
- [Data drift, concept drift and retraining](#data-drift-concept-drift-and-retraining)
- [Classic algorithms: kNN, SVM and Naive Bayes](#classic-algorithms-knn-svm-and-naive-bayes)
- [Unsupervised learning and anomaly detection](#unsupervised-learning-and-anomaly-detection)
- [Rapid-fire round](#rapid-fire-round)
- [How to practice](#how-to-practice)

## Overfitting, leakage and validation

<details>
<summary><b>Q1. How do you detect and prevent overfitting and data leakage?</b></summary>

**Answer:** Overfitting means the model learned noise: I detect it by comparing training and validation error, ideally with learning curves (error vs data size) and validation curves (error vs complexity), where a large gap that shrinks with more data signals high variance. I prevent it with regularization (L1/L2, tree depth and min-leaf limits, dropout), early stopping, more data, simpler models and cross-validated tuning. Leakage is different: information unavailable at prediction time slips into training, so offline scores look great and the model fails in production. I detect it through suspicious signs (too-good metrics, one dominant feature, an offline/online gap, ID or timestamp overlap between splits, adversarial validation) and prevent it structurally: define the prediction time, audit every feature for point-in-time availability, dedupe, split to mimic production, split first and fit all preprocessing inside a pipeline. For example, a spam model at 0.99 AUC that relies on `user_reported_spam_count` is leaking, not brilliant.

**Key points to mention:**
- Regularization does nothing for leakage; only fixing the feature, split or pipeline does.
- Honest evaluation costs points: grouped and time-based splits report lower numbers, and the lower number is the trustworthy one.
- Automate leakage checks in CI (group and timestamp overlap assertions, shuffled-label test) so they do not depend on memory.
- After launch, compare the live metric with the offline estimate and alert when they diverge.

**Deep dive:** [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md), [ML Workflow and Cross-Validation](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md)

</details>

<details>
<summary><b>Q2. How do you distinguish overfitting from data leakage?</b></summary>

**Answer:** Overfitting shows up offline: a large gap between training and validation performance, so validation is honest but disappointing. Leakage shows up as a validation score that is suspiciously good with a small train-validation gap, and the disappointment comes only in production or after fixing the split. Feature importance helps: overfitting spreads weight over many noisy features, while leakage concentrates it in one or two features, often IDs, timestamps or fields populated after the outcome. The decisive test is to rebuild evaluation with a time-based, group-aware split; if the score collapses toward the production number, it was leakage. Timing also helps: a gap present from launch day suggests leakage or training-serving skew, while a gap that grows over weeks suggests drift.

**Key points to mention:**
- More data and regularization fix overfitting; they do nothing for leakage.
- Fix leakage first: it invalidates every number you would use to tune capacity.
- Hybrid case: overfitting to the validation set through heavy tuning, fixed with an untouched test set or nested CV.
- The two can coexist in the same model.

**Deep dive:** [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md)

</details>

<details>
<summary><b>Q3. When would you use random, stratified, grouped, or time-based splits?</b></summary>

**Answer:** The split must mimic how production data relates to training data. Random splits are fine for independent rows with no time order or entity structure and plenty of data. Stratified splits keep class ratios stable in every fold, so they are my default for classification, especially imbalanced or small datasets: with 1% fraud and 200-row folds, a random fold can hold 0 or 4 positives. Grouped splits (`GroupKFold`, `StratifiedGroupKFold`) are needed when several rows share an entity (patient, user, sender, document) and production will see new entities; otherwise the model learns to recognize patients rather than disease. Time-based splits (`TimeSeriesSplit`, walk-forward backtests) are required whenever the model predicts the future from the past, with a gap (embargo) when labels mature late, such as chargebacks.

**Key points to mention:**
- Combine when needed: time-based outer split plus excluding seen customers ("future periods, new customers").
- The right split depends on the product question (block known spammers vs catch new ones).
- Verify the choice: cross-validate with both a random and the production-like splitter; a large gap means the random split leaked.
- Even after a time-based split, fit preprocessing on the training period only, and keep features point-in-time correct.

**Deep dive:** [ML Workflow and Cross-Validation](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md), [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md)

</details>

<details>
<summary><b>Q4. Why should preprocessing be fitted only on training data?</b></summary>

**Answer:** Fitting is learning: a scaler learns means and standard deviations, an imputer learns medians, an encoder learns vocabularies or target averages, a selector learns which features correlate with the label. If any of those statistics include validation or test rows, the evaluation data has influenced the model, so the score no longer reflects production, where future statistics are unknown at prediction time. Scaling leaks are usually small, but label-aware steps can be huge: selecting 20 of 5,000 pure-noise features on all rows and then cross-validating gives accuracy far above the true 50%, while the same selection inside a pipeline gives about 50%. The fix is to split first and wrap every fitted step in a scikit-learn `Pipeline` (with a `ColumnTransformer`) so it is refit inside each training fold. The fitted pipeline also becomes the deployable artifact, which prevents training-serving skew.

**Key points to mention:**
- Worst offenders: target encoding, feature selection and SMOTE run before splitting.
- A pipeline protects only what is inside it: global aggregates computed in SQL or a feature store over the full table still leak.
- Row-wise, unfitted transforms (log, lowercasing) are safe either way.
- Serving calls `transform`, never `fit`; fitted statistics are versioned with the model.

**Deep dive:** [ML Workflow and Cross-Validation](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md), [Feature Engineering](../01-classical-ml/07-feature-engineering-missing-values-and-encoding.md)

</details>

<details>
<summary><b>Q5. How do regularization and cross-validation help against overfitting?</b></summary>

**Answer:** They play different roles. Regularization changes training: it adds a penalty such as $\lambda\sum w_j^2$ (L2) or $\lambda\sum|w_j|$ (L1) so the model prefers simpler explanations, trading a little bias for lower variance. Cross-validation does not change training; it gives a lower-variance estimate of generalization by averaging over several train/validation splits, and that estimate is what you use to choose $\lambda$ and other hyperparameters. In practice I search $\lambda$ on a log-spaced grid, plot the validation curve, pick the minimum (or the simplest model within one standard error of it), then refit on all training data. Too small a $\lambda$ gives low training error and high CV error; too large gives both high.

**Key points to mention:**
- "Regularization is the knob, CV tells you where to set it."
- The best CV score among many configurations is optimistically biased; confirm on a held-out test set or with nested CV.
- CV must respect structure (stratified, grouped, time-based) and keep preprocessing inside folds.
- Neither helps against leakage; CV can make a leaky score look convincingly stable.

**Deep dive:** [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md), [ML Workflow and Cross-Validation](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md)

</details>

<details>
<summary><b>Q6. Explain the bias-variance trade-off. Does it still hold for large neural networks?</b></summary>

**Answer:** Expected squared error decomposes into bias squared (systematic error from too-simple assumptions), variance (sensitivity to the particular training sample) and irreducible noise: $\mathbb{E}[(y-\hat f)^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$. Increasing capacity usually lowers bias and raises variance, so classical validation error is U-shaped with a sweet spot. A slightly biased model can win: bias $-4$ with variance 11.6 gives error 127.6 (with $\sigma^2 = 100$), far better than an unbiased model with variance 1040. For very large networks the decomposition still holds, but test error can fall again past the interpolation point (double descent), so "bigger always overfits" is wrong for modern deep learning. Implicit regularization from SGD, weight decay and architectural inductive biases explain much of this.

**Key points to mention:**
- Knobs: regularization and early stopping trade variance for bias; bagging cuts variance; boosting cuts bias.
- Learning curves diagnose which side you are on.
- The clean decomposition is for squared error; 0-1 loss has no single agreed version.
- Cite Zhang et al. (2017, memorizing random labels) and Belkin et al. (2019) on double descent.

**Deep dive:** [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md)

</details>

<details>
<summary><b>Q7. You evaluated 300 hyperparameter configurations on one validation set. Is the best score trustworthy, and what is nested cross-validation?</b></summary>

**Answer:** Not fully: the maximum of 300 noisy scores is biased upward because part of the winner's advantage is luck on that particular validation set (winner's curse, selection bias). With 20 equally good configs at 0.80 and estimate noise of 0.015, the expected best score is already about 0.828. Nested CV fixes the estimate by putting tuning (inner CV) inside an outer CV loop used only for evaluation, so each outer fold was never seen by the tuner; in scikit-learn it is `cross_val_score(GridSearchCV(model, grid, cv=inner), X, y, cv=outer)`. It estimates the whole "tune then train" procedure; to deploy, you rerun the inner search on all data. I use it when data is too small for a separate test set; with large data an untouched test set does the same job more cheaply.

**Key points to mention:**
- The optimism grows with more configurations and smaller validation sets.
- Nested CV is an evaluation method, not a selection method: do not pick hyperparameters from the best outer fold.
- Cost multiplies: 5 outer x (3 inner x 20 configs + 1 refit) = 305 fits.
- Reused public benchmarks suffer the same effect at community scale.

**Deep dive:** [ML Workflow and Cross-Validation](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md), [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md)

</details>

<details>
<summary><b>Q8. Your model shows 0.99 AUC offline but behaves like 0.70 in production. Walk me through the debugging.</b></summary>

**Answer:** I treat 0.99 as a bug report and assume the offline evaluation did not mirror production until proven otherwise. First I reproduce the online number offline by scoring logged production traffic. Then I check in order: the split (re-split time-based and group-aware; a big drop pinpoints leakage through time or entities), target leakage (single-feature AUC scan and ablation of top features, looking for fields like "days since account closed"), preprocessing fitted outside the pipeline, duplicates across splits, and training-serving skew (log serving features and diff them against the training pipeline's values for the same entities). Finally I check genuine drift and label definition changes. The timing of the gap separates causes: present from launch day points to leakage or skew, gradual decay points to drift.

**Key points to mention:**
- Metric mismatch: AUC measures ranking, while production judges a thresholded decision; base-rate or calibration shifts hurt that even at equal AUC.
- Adversarial validation between training data and production logs shows which features shifted.
- Feedback loops: blocked transactions never produce chargebacks, so production labels are biased.
- Communicate the corrected, honest estimate; prevent recurrence with point-in-time feature joins, split assertions and shadow deployment.

**Deep dive:** [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md), [ML Workflow and Cross-Validation](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md), [Reproducibility and Training-Serving Skew](../10-mlops/03-reproducibility-and-training-serving-skew.md)

</details>

<details>
<summary><b>Q9. Compare grid search, random search, Bayesian optimization and Hyperband.</b></summary>

**Answer:** Grid search tries every combination; it is fine for one or two hyperparameters, but cost multiplies (5 values for 6 hyperparameters is 15,625 trials) and it wastes trials on unimportant dimensions. Random search is my default because only a few hyperparameters usually matter, and for the same budget it explores many more distinct values of them: if 5% of the space is good, 60 trials hit it with probability about 95%, regardless of dimensionality (Bergstra and Bengio, 2012). Bayesian optimization (Gaussian processes, TPE in Optuna) fits a surrogate of score versus configuration and picks promising points, which pays off when each trial is expensive, though it is more sequential. Successive halving and Hyperband stop bad configurations early, assuming early performance predicts final performance. In practice I often use TPE with pruning.

**Key points to mention:**
- Sample learning rates and regularization strengths on a log scale.
- Hyperband can kill configurations that only pay off late (small learning rates).
- Give every compared model the same tuning budget; record every trial in an experiment tracker.
- Tuning is often worth less than better features or data.

**Deep dive:** [Regression Metrics, Baselines and Model Selection](../01-classical-ml/05-regression-metrics-baselines-and-model-selection.md), [ML Workflow and Cross-Validation](../01-classical-ml/01-ml-workflow-data-splits-and-cross-validation.md)

</details>

## Linear models and regularization

<details>
<summary><b>Q10. L1 vs L2 regularization: what is the difference, and why does L1 give sparse models?</b></summary>

**Answer:** L2 (ridge) adds $\lambda\sum w_j^2$ and shrinks all weights smoothly toward zero; L1 (lasso) adds $\lambda\sum|w_j|$ and sets many weights exactly to zero, performing feature selection. From the gradient view, L2's pull is $2\lambda w$, which fades as the weight shrinks, while L1's pull has constant magnitude $\lambda$, so weak weights get pushed all the way to zero. Geometrically, the L1 constraint region is a diamond with corners on the axes, and loss contours usually touch it at a corner. In 1-D the lasso solution is a soft threshold: once $\lambda$ exceeds the feature's evidence, the weight is exactly 0, while ridge only divides it down. With correlated features, ridge spreads weight stably, lasso picks one somewhat arbitrarily, and elastic net mixes both to keep groups.

**Key points to mention:**
- Bayesian view: L2 is a Gaussian prior, L1 a Laplace prior.
- Always standardize first, leave the intercept unpenalized, and tune strength with CV (in `LogisticRegression`, `C` is the inverse).
- L1 is not differentiable at 0; solvers use coordinate descent or proximal methods.
- Use L1 for interpretability and selection, L2 as the default when many features carry small effects.

**Deep dive:** [Linear and Logistic Regression](../01-classical-ml/02-linear-and-logistic-regression.md), [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md)

</details>

<details>
<summary><b>Q11. Why is logistic regression trained with log loss instead of mean squared error?</b></summary>

**Answer:** Log loss is the negative log-likelihood of a Bernoulli model, so minimizing it is maximum likelihood estimation. With a linear model it is convex (one global optimum), and its gradient with respect to the logit is simply $p - y$, so the gradient is $(p - y)\mathbf{x}$, "error times input", the same shape as linear regression. MSE through a sigmoid is non-convex and its gradient contains $p(1-p)$, which vanishes when the model is confidently wrong. For a positive example with logit $-5$ ($p \approx 0.0067$), the log-loss gradient is about $-0.99$ versus about $-0.013$ for MSE, roughly 75 times stronger. Log loss is also a proper scoring rule, which is why logistic regression tends to be well calibrated.

**Key points to mention:**
- Same reason every neural classifier uses cross-entropy.
- Confident errors are punished hard: predicting 0.01 for a positive costs $-\ln 0.01 \approx 4.6$.
- Compute from logits for numerical stability, not `log(sigmoid(z))`.
- Calibration can still break with misspecification, heavy regularization, reweighting or drift.

**Deep dive:** [Linear and Logistic Regression](../01-classical-ml/02-linear-and-logistic-regression.md), [Softmax, Entropy, Cross-Entropy and KL Divergence](../00-foundations/05-softmax-entropy-cross-entropy-and-kl-divergence.md)

</details>

<details>
<summary><b>Q12. A logistic regression coefficient for "has_link" is 2.0. How do you explain it to a product manager?</b></summary>

**Answer:** Coefficients live on the log-odds scale, so I exponentiate: holding the other features fixed, having a link multiplies the odds of spam by $e^{2.0} \approx 7.4$. It is not a fixed increase in probability, because the sigmoid is flat near 0 and 1: adding a link moves one email from 0.27 to about 0.73 (+0.46) but another from 0.80 to only about 0.97 (+0.17). Magnitudes are comparable across features only if features are standardized, and the coefficient describes association, not causation. Correlated features make individual coefficients unstable, and regularization biases them toward zero. For uncertainty I would report a confidence interval from `statsmodels` or the bootstrap.

**Key points to mention:**
- Odds ratio, not probability change; give one concrete before/after example.
- A very large coefficient on standardized data (say above 10) hints at perfect separation or a leaking feature.
- "Holding others fixed" may be unrealistic when features are correlated.
- Intercept and base rate determine where on the sigmoid a typical example sits.

**Deep dive:** [Linear and Logistic Regression](../01-classical-ml/02-linear-and-logistic-regression.md)

</details>

<details>
<summary><b>Q13. What is multicollinearity, why does it matter, and what do you do about it?</b></summary>

**Answer:** Multicollinearity is strong correlation among features, which makes $X^\top X$ nearly singular, so small changes in the data produce large swings in the weights. It usually does not hurt predictions on stable data, but individual coefficients become high-variance, may flip sign and become uninterpretable; with perfect collinearity there is no unique solution. I detect it with a correlation matrix or the variance inflation factor $\text{VIF}_j = 1/(1 - R_j^2)$, where $R_j^2 = 0.9$ gives VIF 10, a common warning threshold. Fixes are dropping or combining features, ridge (which adds $\lambda I$ and stabilizes the solution), elastic net to keep correlated groups, or PCA.

**Key points to mention:**
- Matters for interpretation, causal-style analysis and importance reports; much less for pure prediction.
- Lasso picks one of the correlated features arbitrarily, and the choice can flip between retrains.
- Production risk: if the correlation structure changes after deployment, a model that split credit arbitrarily can degrade.
- The normal equation fails outright under perfect collinearity; use `lstsq` or ridge.

**Deep dive:** [Linear and Logistic Regression](../01-classical-ml/02-linear-and-logistic-regression.md)

</details>

## Classification metrics and imbalanced data

<details>
<summary><b>Q14. How do you evaluate a classification model on an imbalanced dataset?</b></summary>

**Answer:** I never rely on accuracy. I start with the confusion matrix and the prevalence $\pi$, then use threshold-free metrics that focus on the rare class (PR-AUC or average precision, with ROC-AUC for comparison across segments), and then evaluate at the real operating point with business-tied metrics: precision and recall at the chosen threshold, precision@k for a review budget, or expected cost. Every number sits next to its baseline: null accuracy $1 - \pi$, random PR-AUC $\pi$, random ROC-AUC 0.5. I use stratified or time-based splits, keep validation and test sets at the natural class balance, check calibration if probabilities are consumed, and report confidence intervals, since with 100 positives a recall of 0.80 carries roughly $\pm 0.08$ uncertainty.

**Key points to mention:**
- Slice metrics by segment: a good average can hide a segment with near-zero recall.
- Never resample validation or test data; never tune the threshold on the test set.
- In production precision moves with prevalence and labels arrive late; add alert rate and analyst-confirmed precision.
- Use paired comparisons (same folds, bootstrap) when comparing models.

**Deep dive:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

</details>

<details>
<summary><b>Q15. When would you use precision, recall, F1, ROC-AUC, or PR-AUC?</b></summary>

**Answer:** Precision, $TP/(TP+FP)$, when false alarms are costly, for example each alert consumes analyst time or blocks a customer. Recall, $TP/(TP+FN)$, when misses are costly, as in disease screening or safety filters. F1 when I need one number and both errors matter about equally, F-beta when one matters more. ROC-AUC, the probability a random positive outranks a random negative, to compare rankers threshold-free when I need prevalence invariance across periods or segments. PR-AUC when positives are rare and the cost lives in the alerts, because it exposes false positives that ROC hides: ROC-AUC can be 0.98 while precision at every useful threshold is below 10% at 0.1% prevalence. If dollar costs are known, expected cost at the chosen threshold is the final metric.

**Key points to mention:**
- Precision depends on prevalence; recall does not.
- F1 ignores true negatives and silently assumes a cost ratio; for a calibrated model the F1-optimal threshold is half the best F1.
- PR-AUC's baseline is $\pi$, so it is not comparable across datasets with different prevalence.
- Report several: AP for model selection, ROC-AUC for stability, precision/recall at the operating point for launch.

**Deep dive:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

</details>

<details>
<summary><b>Q16. Why can accuracy be misleading?</b></summary>

**Answer:** Accuracy counts all correct predictions equally, so on imbalanced data the majority class dominates it: a fraud model that always says "legit" on data with 0.1% fraud is 99.9% accurate and catches nothing. It also assumes every error costs the same, which is rarely true, and it depends on a single, often default, threshold. Two models with identical accuracy can have very different error mixes (20 FN plus 30 FP versus 50 FN plus 0 FP) and therefore very different business costs. Accuracy can even rise while the model gets worse: if fraud prevalence halves, accuracy improves even as recall drops. Balanced accuracy, MCC, precision/recall and expected cost are better choices.

**Key points to mention:**
- Always state null accuracy $1 - \pi$ as the baseline.
- Balanced accuracy gives 0.5 for the do-nothing model.
- Accuracy is acceptable for balanced classes with symmetric costs, and even then inspect the confusion matrix.
- Dashboards built on accuracy can show "improvement" during a real degradation.

**Deep dive:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

</details>

<details>
<summary><b>Q17. How do you select the classification threshold based on business costs?</b></summary>

**Answer:** Assign a cost to each error type and pick the threshold that minimizes total expected cost on validation data. With calibrated probabilities the optimal rule follows from flagging when $(1-p)C_{FP} < p\,C_{FN}$, which gives the threshold $C_{FP}/(C_{FP} + C_{FN})$: with a 10-dollar false alarm and a 500-dollar missed fraud it is about 0.02, not 0.5. In practice I sweep thresholds on validation data, compute $FN \cdot C_{FN} + FP \cdot C_{FP}$ for each, take the minimum and confirm once on the test set; the sweep works even without calibration because it uses real counts. Then I add constraints such as analyst capacity (top-k per day), a regulatory recall floor or a customer-friction limit. In the course example the cost-optimal threshold cost 20,620 versus 50,400 at 0.5, despite precision of only about 16%.

**Key points to mention:**
- Costs are rough: run a sensitivity analysis and pick a threshold in the flat part of the cost curve.
- Costs can be per-example (transaction amount), turning the threshold into a per-transaction rule.
- Store the threshold as versioned config; re-tune when costs, prevalence or the model change.
- Tune on validation, never test.

**Deep dive:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

</details>

<details>
<summary><b>Q18. When would you use class weights versus oversampling?</b></summary>

**Answer:** I default to class weights when the learner supports them: they shift the loss toward the minority class without duplicating data, so there is no extra training time and no exact copies to memorize. In expectation, duplicating a row $k$ times equals giving it weight $k$ under gradient-based training; the differences are variance, speed and learner-specific behavior such as how trees count samples in leaves. I use oversampling when the algorithm or pipeline cannot take sample weights, or in mini-batch deep learning, where balanced batch sampling keeps minority examples in every batch and gives steadier gradients than huge weights. SMOTE is reasonable only for low-dimensional continuous features, and undersampling (ideally an ensemble of undersamples) when the majority class is huge and redundant. Often the biggest win is simply moving the threshold.

**Key points to mention:**
- Both methods distort probabilities; correct with prior correction or Platt/isotonic on natural-distribution data.
- Resample only inside the training fold (`imblearn.pipeline.Pipeline`).
- Compare fairly: each variant gets its own tuned threshold; comparing everything at 0.5 measures the threshold, not the method.
- Oversampling risks memorizing duplicates in deep trees and kNN.

**Deep dive:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

</details>

<details>
<summary><b>Q19. Your fraud model has ROC-AUC 0.98, but operations says most alerts are false. How can both be true?</b></summary>

**Answer:** Because fraud is rare. ROC uses the false positive rate, which divides false positives by the huge number of legitimate transactions, so even a tiny FPR means many false alarms in absolute terms. Precision is $\frac{\text{TPR}\,\pi}{\text{TPR}\,\pi + \text{FPR}(1-\pi)}$: at prevalence 0.1%, TPR 0.8 and FPR 1%, precision is only about 7%. Improving FPR from 1% to 0.1% looks negligible on the ROC curve but lifts precision from about 7% to about 44%. The fix is to select models on PR-AUC and precision@k, and to set the threshold for the team's capacity and costs.

**Key points to mention:**
- Check whether prevalence differs between validation and production; precision moves with it.
- Options: raise the threshold, add a second-stage model or rules, add features targeting the false-positive pattern.
- ROC-AUC is still useful for stability across segments because it is prevalence-invariant.
- Partial AUC focuses on the low-FPR region that matters operationally.

**Deep dive:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

</details>

<details>
<summary><b>Q20. What is calibration, how do you measure it, and how do you fix it?</b></summary>

**Answer:** A model is calibrated if, among examples scored at 0.3, about 30% are actually positive. I measure it with a reliability diagram, expected calibration error (with quantile bins on imbalanced data), and proper scoring rules such as log loss and the Brier score compared with the base-rate Brier $\pi(1-\pi)$. I fix it post hoc on held-out data: Platt scaling (a two-parameter sigmoid, data-efficient), isotonic regression (a flexible monotonic step function that needs more data), or temperature scaling for neural networks. Discrimination and calibration are separate: monotonic calibration leaves AUC essentially unchanged. Calibration matters whenever probability values are consumed, for cost thresholds, expected loss, pricing or combining models.

**Key points to mention:**
- Typical tendencies: Naive Bayes overconfident, random forests underconfident at the extremes, deep nets often overconfident, reweighted models inflated toward the minority.
- Fit the calibrator on data the model did not train on (`CalibratedClassifierCV`).
- Calibrate on the natural class balance.
- Logistic regression is often well calibrated out of the box because it minimizes log loss.

**Deep dive:** [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md), [Linear and Logistic Regression](../01-classical-ml/02-linear-and-logistic-regression.md)

</details>

<details>
<summary><b>Q21. You trained with class_weight="balanced" on 1% positives and production probabilities look far too high. Why, and how do you fix it?</b></summary>

**Answer:** Reweighting, like undersampling, makes the model train as if positives were far more common, mostly by shifting the intercept, so probabilities no longer reflect the real base rate even though ranking (AUC) may be fine. For uniform undersampling the shift is exact and reversible: rescale the odds by the ratio of true to training prior odds, $\text{odds}_{\text{true}} = \text{odds}_{\text{model}} \times \frac{\pi/(1-\pi)}{\pi'/(1-\pi')}$. Trained at 50/50 with a true prevalence of 1%, a model output of 0.9 corrects to about 0.083. Class weights behave similarly, but the correction is exact only if the model is well specified, so I recalibrate with Platt or isotonic on an untouched validation set at the natural distribution. Alternatively, train unweighted and handle imbalance purely through the threshold.

**Key points to mention:**
- If you only need a decision, skip correction and re-tune the threshold on natural-distribution data.
- Forgetting the correction breaks cost thresholds, expected-loss math and downstream consumers of the probability.
- Check with a reliability diagram and by comparing the mean predicted probability with the observed rate.
- Same issue applies to `scale_pos_weight` in XGBoost and to SMOTE.

**Deep dive:** [Linear and Logistic Regression](../01-classical-ml/02-linear-and-logistic-regression.md), [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md)

</details>

## Regression metrics and model selection

<details>
<summary><b>Q22. When would you use MSE or MAE, as a metric and as a loss?</b></summary>

**Answer:** As a metric, MAE gives a robust "typical error" in the target's units when every unit of error costs the same; MSE (reported as RMSE) punishes large errors disproportionately, which fits cases where a 2-hour delivery miss is far worse than two 1-hour misses. As a loss, MSE makes the model predict the conditional mean and has smooth gradients proportional to the error; MAE predicts the conditional median, is robust to label outliers, but has a constant gradient undefined at zero. On prices {200, 300, 400, 500, 2000}, the MSE-optimal constant is the mean 680 and the MAE-optimal constant is the median 400. Probabilistically, MSE is maximum likelihood under Gaussian noise and MAE under Laplace noise. Huber is the compromise, and quantile loss handles asymmetric costs.

**Key points to mention:**
- Report both MAE and RMSE; a large gap reveals heavy-tailed errors.
- Training with MSE on $\log(y)$ predicts roughly the median, so summed predictions under-shoot totals without a correction (Duan's smearing).
- MAE failure mode: on intermittent demand the median can be 0, so "always zero" scores well and is useless.
- Match the loss to the metric the business actually cares about.

**Deep dive:** [Regression Metrics, Baselines and Model Selection](../01-classical-ml/05-regression-metrics-baselines-and-model-selection.md)

</details>

<details>
<summary><b>Q23. What are the problems with MAPE, and what would you use instead?</b></summary>

**Answer:** MAPE is undefined when an actual is zero, explodes when actuals are small (actual 1, forecast 3 is 200% error), and is asymmetric: under-forecasts are capped at 100% while over-forecasts are unbounded, so model selection on MAPE biases toward predicting low. The same 100-unit miss is 100% error for actual 100 and forecast 200, but 50% for actual 200 and forecast 100. Better options are WAPE, $\sum|e| / \sum|y|$, for aggregate relative error that tolerates individual zeros; MASE, which divides MAE by the in-sample MAE of a naive forecast (below 1 beats naive), for scale-free comparison across series; and an explicit cost or pinball loss when costs are asymmetric.

**Key points to mention:**
- sMAPE has several definitions and still misbehaves near zero.
- scikit-learn's `mean_absolute_percentage_error` returns a fraction and uses a tiny epsilon for zero actuals, producing huge values instead of an error.
- WAPE weights big items more, which usually matches business impact.
- For asymmetric costs, forecast a quantile: under-cost 3, over-cost 1 means the 0.75 quantile.

**Deep dive:** [Regression Metrics, Baselines and Model Selection](../01-classical-ml/05-regression-metrics-baselines-and-model-selection.md)

</details>

<details>
<summary><b>Q24. Your new model improves cross-validated MAE from 14.5 to 14.2. Is it real, and is it worth shipping?</b></summary>

**Answer:** First I make the comparison fair: same folds, same pipelines, comparable tuning budgets. Then I look at paired differences per fold or per example, because shared example difficulty cancels and gives much tighter intervals than two independent confidence intervals. I compute a paired bootstrap interval or a corrected paired test: CV training sets overlap, so the naive t-test is overconfident, and the Nadeau-Bengio correction inflates the variance (for 5 folds the standard error grows about 1.5x). With five fold differences of 0.3, 0.3, -0.1, 0.5, 0.4 the naive t is about 2.75, already borderline, and about 1.83 after correction. Finally I compare the gain with a minimum worthwhile improvement agreed in advance, check slices and latency, and confirm online with a canary or A/B test.

**Key points to mention:**
- Practical significance beats statistical significance: a 2% gain may not justify a heavier pipeline.
- Multiple comparisons: after trying 30 variants, some winners are false; confirm on a fresh test set.
- Alternatives: repeated k-fold, 5x2cv, Wilcoxon signed-rank on per-example errors, Diebold-Mariano for forecasts.
- Always compare against baselines (mean, seasonal naive, current production model).

**Deep dive:** [Regression Metrics, Baselines and Model Selection](../01-classical-ml/05-regression-metrics-baselines-and-model-selection.md), [Probability and Statistics](../00-foundations/04-probability-and-statistics.md)

</details>

## Trees and ensembles

<details>
<summary><b>Q25. Compare Random Forest, XGBoost, and Logistic Regression.</b></summary>

**Answer:** Logistic regression is a linear model of the log-odds: fast, interpretable through odds ratios, usually well calibrated, excellent on small or very high-dimensional sparse data, but it only captures nonlinearity and interactions you engineer, and it needs scaling and imputation. Random forest averages many deep, decorrelated trees trained on bootstrap samples with random feature subsets: it mainly cuts variance, needs little tuning, gives a free out-of-bag estimate, but probabilities are compressed toward the middle and the model is large. XGBoost builds shallow trees sequentially, each fitting the gradient of the loss with second-order information and explicit regularization: usually the most accurate on tabular data, handles missing values natively, but needs early stopping and more tuning. Both tree models are flat outside the training range, while logistic regression extrapolates linearly. I start with logistic regression as the baseline and leakage check, and ship XGBoost only if its gain is outside fold noise and worth the extra latency, monitoring and explainability cost.

**Key points to mention:**
- Compare on identical leak-free folds with the business metric, a calibration measure (log loss or Brier) and latency.
- "XGBoost beat LR by 0.5% AUC, do you ship it?": depends on noise, business value and operational cost.
- On bag-of-words text or tiny datasets, regularized LR is often competitive.
- Hybrids: boosted-tree leaf indices as features for LR, or LR as a stacker or calibrator.

**Deep dive:** [Decision Trees, Random Forests and Gradient Boosting](../01-classical-ml/06-decision-trees-random-forests-and-gradient-boosting.md), [Linear and Logistic Regression](../01-classical-ml/02-linear-and-logistic-regression.md)

</details>

<details>
<summary><b>Q26. When would you choose gradient-boosted trees over a neural network?</b></summary>

**Answer:** On typical tabular data (mixed numeric and categorical columns, missing values, thousands to millions of rows), gradient-boosted trees are my default: they are usually as accurate or more accurate, need less preprocessing and tuning, train fast on CPUs and are easier to explain. Grinsztajn et al. (2022) explain why: tabular targets are irregular with sharp thresholds, trees ignore uninformative features, and columns carry individual meaning that axis-aligned splits exploit while MLPs are rotation-invariant. I choose a neural network for unstructured data (text, images, audio), very large data with high-cardinality IDs that benefit from embeddings, sequences, multimodal or end-to-end setups, or when transfer learning is available. A common hybrid feeds neural or LLM text embeddings as features into a GBDT.

**Key points to mention:**
- Operational wins: cheap training and inference, fewer sensitive hyperparameters, SHAP, monotonic constraints for regulated domains.
- Trees cannot extrapolate; trending targets may need linear or NN components.
- Compare over several seeds (NNs on tables vary more run to run), plus cost and latency.
- Tabular deep learning and tabular foundation models are active research; benchmark if the stakes justify it.

**Deep dive:** [Decision Trees, Random Forests and Gradient Boosting](../01-classical-ml/06-decision-trees-random-forests-and-gradient-boosting.md)

</details>

<details>
<summary><b>Q27. How do bagging and boosting differ?</b></summary>

**Answer:** Bagging trains many models independently, in parallel, on bootstrap resamples and averages them; it mainly reduces variance, so it uses deep, high-variance trees and does not overfit as you add more of them. The variance of an average of correlated trees is $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$, so more trees help until the correlation floor, which random forests lower by subsampling features at each split. Boosting trains models sequentially, $F_m = F_{m-1} + \eta h_m$, with each new tree fit to the negative gradient (pseudo-residuals) of the current ensemble; it mainly reduces bias, uses shallow trees, and will overfit with too many rounds unless you use shrinkage and early stopping. Plotting validation error against the number of models shows it: a forest plateaus, boosting falls, flattens, then rises.

**Key points to mention:**
- Bagging is embarrassingly parallel; boosting is sequential across trees (parallel within a tree).
- Boosting, especially AdaBoost, can chase noisy labels; bagging is robust to them.
- Bagging gives OOB error for free; boosting needs a validation set.
- Stochastic gradient boosting borrows bagging's row and column subsampling.

**Deep dive:** [Decision Trees, Random Forests and Gradient Boosting](../01-classical-ml/06-decision-trees-random-forests-and-gradient-boosting.md), [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md)

</details>

<details>
<summary><b>Q28. How do you control overfitting in XGBoost?</b></summary>

**Answer:** Five levers. Number of steps: a small learning rate with a large `n_estimators` and early stopping on a validation set. Tree complexity: lower `max_depth`, higher `min_child_weight` (minimum hessian sum per child), higher `gamma` (minimum gain to split). Leaf-weight regularization: `lambda` (L2) and `alpha` (L1), which shrink leaf values via $w^* = -G/(H+\lambda)$. Randomness: `subsample` and `colsample_bytree` below 1. Data: a leak-free feature set and a validation split that mirrors production, time-based if the data is temporal. My recipe: fix the learning rate at 0.05 to 0.1, tune depth and `min_child_weight`, then subsampling, then gamma, lambda and alpha, and finally lower the learning rate for the last gain.

**Key points to mention:**
- Diagnose with train vs validation loss per round; a widening gap is overfitting.
- Never early-stop on the test set; never use random K-fold on time series.
- If validation looks too good, suspect leakage before hyperparameters.
- Domain knowledge regularizes too: `monotone_constraints` and `interaction_constraints`; `scale_pos_weight` distorts probabilities.

**Deep dive:** [Decision Trees, Random Forests and Gradient Boosting](../01-classical-ml/06-decision-trees-random-forests-and-gradient-boosting.md)

</details>

<details>
<summary><b>Q29. How do you interpret feature importance, and what are its limitations?</b></summary>

**Answer:** First I clarify which importance: split count, gain (impurity decrease), permutation importance or SHAP. Gain and split count are free but computed on training data and biased toward high-cardinality, ID-like features that find spurious splits. Permutation importance on held-out data measures real predictive contribution, though it creates unrealistic rows and has run-to-run variance. SHAP decomposes each prediction additively, $\hat y(x) = \phi_0 + \sum_j \phi_j(x)$, and TreeSHAP is exact and fast for trees. Limitations: correlated features share or hide credit (permutation can mark both near-duplicates as unimportant), rankings differ between methods, importance reflects the model including its leaks, and importance is never causality.

**Key points to mention:**
- Sanity check: add a random column and see where it ranks.
- Fix correlation issues by permuting feature groups or clustering correlated features.
- Check ranking stability across CV folds; validate feature removal by retraining.
- "used_coupon is important for churn" does not mean coupons reduce churn; causal questions need experiments.
- Per-decision SHAP values double as reason codes in regulated domains.

**Deep dive:** [Decision Trees, Random Forests and Gradient Boosting](../01-classical-ml/06-decision-trees-random-forests-and-gradient-boosting.md)

</details>

<details>
<summary><b>Q30. Walk me through how a decision tree picks a split. Gini or entropy?</b></summary>

**Answer:** For each feature and each candidate threshold (midpoints of sorted values), the tree computes the weighted impurity of the two children and greedily picks the split with the largest impurity decrease. Gini impurity is $1 - \sum p_k^2$ and entropy is $-\sum p_k\log_2 p_k$. For a root with 4 spam and 6 ham, Gini is 0.48 and entropy 0.971 bits; splitting into (3,1) and (1,5) gives a Gini decrease of 0.163 and an information gain of 0.256 bits. The two criteria almost always choose the same split; Gini is slightly cheaper, which is why it is scikit-learn's default. For regression, the tree minimizes the children's squared error and leaves predict the mean.

**Key points to mention:**
- Greedy splitting is not globally optimal; finding an optimal tree is NP-hard.
- Cost is $O(n \log n)$ per feature per node; histogram binning (LightGBM, HistGradientBoosting) reduces it.
- Information gain favors many-valued features; C4.5's gain ratio corrects this.
- Unpruned trees memorize; control with depth, min samples per leaf or cost-complexity pruning.

**Deep dive:** [Decision Trees, Random Forests and Gradient Boosting](../01-classical-ml/06-decision-trees-random-forests-and-gradient-boosting.md)

</details>

<details>
<summary><b>Q31. Can tree models extrapolate? Why does it matter?</b></summary>

**Answer:** No. A tree predicts a constant per leaf, so any input beyond the training range lands in an edge leaf and gets that leaf's value; forests and boosted ensembles inherit this flat behavior. A house-price model trained on houses up to 110 m² predicts the same price for 150 m² and for 500 m². It matters for time trends (sales growth, inflation), new regimes and any drift that pushes inputs beyond what was seen. Fixes include modeling the trend separately (predict a ratio or a detrended target, or fit a linear base model and boost the residuals), LightGBM's `linear_tree` option, more frequent retraining, and monitoring input ranges in production.

**Key points to mention:**
- Linear models extrapolate linearly, which can be right or dangerously wrong.
- Out-of-range inputs are a cheap, useful monitoring signal.
- This is a key reason to prefer a linear component for trending targets like prices.
- Retrain cadence matters more for trees when the target trends.

**Deep dive:** [Decision Trees, Random Forests and Gradient Boosting](../01-classical-ml/06-decision-trees-random-forests-and-gradient-boosting.md), [Regression Metrics, Baselines and Model Selection](../01-classical-ml/05-regression-metrics-baselines-and-model-selection.md)

</details>

## Feature engineering: missing values, encoding and scaling

<details>
<summary><b>Q32. How do you handle missing values, categorical features, and distribution drift?</b></summary>

**Answer:** For missing values I first ask why they are missing (MCAR, MAR or MNAR) and how often; my default is median or constant imputation plus a missing indicator for linear models and neural nets, or native missing handling for gradient boosting, with KNN or iterative imputation only when it measurably helps. For categoricals I choose by cardinality and model: one-hot for low cardinality, out-of-fold smoothed target encoding or native categorical support for high cardinality with trees, hashing or learned embeddings for huge vocabularies, always with a plan for unseen and rare categories. For drift, I make encoders robust to unseen values, monitor feature distributions (PSI, KS), missing rates and new-category rates, and refit encoders or retrain when the shift actually hurts performance. Everything is fitted inside a pipeline on training data only.

**Key points to mention:**
- A sudden spike in missing rate is usually an upstream bug, not real drift; validate schemas first.
- Treat imputer and encoder choices as pipeline hyperparameters compared under the same CV.
- Fitted state (medians, category lists, encoding maps) is a snapshot of the training period; version it.
- Alert on PSI above about 0.25, but retrain only when label-based or business metrics confirm harm.

**Deep dive:** [Feature Engineering](../01-classical-ml/07-feature-engineering-missing-values-and-encoding.md), [Data Drift and Retraining](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md)

</details>

<details>
<summary><b>Q33. When would you use imputation, one-hot encoding, or target encoding?</b></summary>

**Answer:** Imputation is for missing values in models that cannot accept them (linear models, SVMs, KNN, neural nets): median for skewed numerics, mode or a `"missing"` category for categoricals, usually with an indicator; XGBoost, LightGBM, CatBoost and HistGradientBoosting learn where to send missing values, so they often skip it. One-hot is for nominal categoricals with low to moderate cardinality, especially for linear models and neural nets, because it adds no fake order; its width equals cardinality, and it is weak for trees at high cardinality because each split isolates one category. Target encoding replaces a high-cardinality category with its smoothed mean target, giving one dense column that tree models use well, but only with out-of-fold fitting. Example: in a fraud model, one-hot `card_type` (4 values), target-encode or embed `merchant_id` (500k values), and impute `device_age` with median plus indicator for the logistic baseline.

**Key points to mention:**
- Alternatives: frequency encoding (no label use), hashing (unbounded vocabulary), embeddings, native categorical handling.
- Target encoding compresses a category to one number and goes stale when category behavior drifts.
- Define fallbacks for unseen values: all-zeros, the prior, or an "unknown" bucket.
- Drop one column or regularize to avoid the dummy-variable trap in unregularized linear regression.

**Deep dive:** [Feature Engineering](../01-classical-ml/07-feature-engineering-missing-values-and-encoding.md)

</details>

<details>
<summary><b>Q34. What is target encoding leakage, and how do you avoid it?</b></summary>

**Answer:** Target encoding replaces a category with the mean label for that category. Leakage happens when a row's encoding is computed using its own label (or validation and test labels, if you encode before splitting), so the feature partially contains the answer, the model over-trusts it, and offline scores are inflated. It is worst for rare categories: a category seen once with label 1 is encoded as exactly 1.0, a perfect but fake signal. The fixes are out-of-fold encoding (each fold encoded from the other folds), smoothing toward the global mean, $\text{enc}(c) = \frac{n_c \bar{y}_c + m \bar{y}}{n_c + m}$, optional noise, and fitting the encoder inside the CV loop. Out-of-fold works because each training row's encoding is then exactly as noisy as it will be on new data, so the model learns the right amount of trust.

**Key points to mention:**
- scikit-learn's `TargetEncoder.fit_transform` cross-fits internally; `fit().transform()` on the same data does not.
- Leave-one-out encoding is a trap: with binary labels it can perfectly separate classes within a category.
- CatBoost uses ordered target statistics for the same reason.
- On temporal data, encode from past events only; detect leakage with a shuffled-label run that still beats chance.

**Deep dive:** [Feature Engineering](../01-classical-ml/07-feature-engineering-missing-values-and-encoding.md), [Overfitting, Regularization and Data Leakage](../01-classical-ml/03-overfitting-regularization-and-data-leakage.md)

</details>

<details>
<summary><b>Q35. Explain MCAR, MAR and MNAR. Why does the distinction matter in practice?</b></summary>

**Answer:** MCAR (missing completely at random) means missingness is unrelated to anything, such as random sensor glitches. MAR (missing at random) means it depends on other observed features, for example square footage missing for old houses when we know the year built. MNAR (missing not at random) means it depends on the missing value itself, such as high earners declining to report income. The distinction decides what is safe: deletion is guaranteed unbiased only under MCAR, imputation from other features handles MAR, and MNAR needs a missing indicator plus domain knowledge or new data. For prediction, informative missingness is an asset, and the indicator often becomes a useful feature.

**Key points to mention:**
- MAR versus MNAR cannot be proven from observed data, because the evidence is the hidden values.
- Diagnostics: compare distributions of missing vs non-missing rows, train a classifier to predict missingness, ask data owners.
- Structural missingness (no garage) gets a domain fill plus an indicator.
- For statistical inference, multiple imputation with Rubin's rules is the principled tool.

**Deep dive:** [Feature Engineering](../01-classical-ml/07-feature-engineering-missing-values-and-encoding.md)

</details>

<details>
<summary><b>Q36. Which models need feature scaling, and which scaler would you choose?</b></summary>

**Answer:** Distance-based models (KNN, k-means, RBF SVM), PCA, regularized linear models and neural networks need scaling; tree-based models do not, because splits are per-feature thresholds. For regularized linear models the reason is fairness: the penalty on $\sum w_j^2$ depends on units, so unscaled features are penalized unequally. I default to standardization, $(x - \mu)/\sigma$; robust scaling, $(x - \text{median})/\text{IQR}$, when there are real outliers; min-max for naturally bounded features; and a log or Yeo-Johnson transform first for heavily right-skewed data. With square footage 1200, 1500, 1600, 1800, 4000, min-max squeezes the four normal houses into 0 to 0.21, while robust scaling keeps them spread and leaves the mansion at 8.

**Key points to mention:**
- Fit scalers on training data only, inside the pipeline.
- Min-max outputs can exceed [0, 1] at serving time; consider clipping.
- Gradient descent converges much faster on scaled features; closed-form OLS predictions do not depend on scaling.
- Unscaled kNN lets the largest-range feature dominate the distance.

**Deep dive:** [Feature Engineering](../01-classical-ml/07-feature-engineering-missing-values-and-encoding.md), [kNN, SVMs and Naive Bayes](../01-classical-ml/09-knn-svm-naive-bayes-and-other-classic-algorithms.md)

</details>

<details>
<summary><b>Q37. You have a feature with 10 million distinct user IDs. How would you encode it?</b></summary>

**Answer:** Never one-hot; the choice depends on the model and on how much data each ID has. With a neural model and many interactions per user, learned embeddings, often with hashing to bound table size: 10M users at 32 dimensions in FP32 is about 1.28 GB, rare users barely get trained, and cold-start users need an "unknown" vector. With gradient boosting, out-of-fold smoothed target encoding or frequency encoding, where smoothing is mandatory because most IDs are rare. For a streaming linear model, the hashing trick with a fixed number of buckets, accepting collisions. Often the best answer is to replace the raw ID with features about the user (tenure, average order value, recent activity counts), which generalize to new users.

**Key points to mention:**
- An ID-memorizing model fails on new users; aggregates generalize.
- Hashing: with $m$ buckets, a given pair collides with probability $1/m$; use a stable hash, not Python's randomized `hash()`.
- Plan cold-start and unseen-ID behavior explicitly.
- Compute aggregates point-in-time correctly to avoid leakage.

**Deep dive:** [Feature Engineering](../01-classical-ml/07-feature-engineering-missing-values-and-encoding.md)

</details>

## Data drift, concept drift and retraining

<details>
<summary><b>Q38. How do you detect data drift versus concept drift?</b></summary>

**Answer:** Using $P(X, Y) = P(X)\,P(Y \mid X)$: data (covariate) drift is a change in $P(X)$, so it can be detected from inputs alone, without labels, by comparing feature distributions between a reference and a current window with PSI, KS, chi-square or JS divergence, and with a domain classifier or embedding-based tests for high-dimensional inputs. Concept drift is a change in $P(Y \mid X)$: inputs can look identical while the right answers change, so it needs labels. I detect it with windowed performance on backfilled labels by cohort, streaming detectors on the error rate (DDM, ADWIN, Page-Hinkley) when labels are fast, and proxies such as user feedback or sampled human review when labels are slow. The telltale pattern: performance drops while feature drift is low points to concept drift; feature drift is high while performance holds is harmless covariate shift.

**Key points to mention:**
- Prior (label) shift is a third type: $P(Y)$ changes with $P(X \mid Y)$ fixed; fix by recalibration, not retraining.
- Label-free performance estimation (CBPE) assumes $P(Y \mid X)$ is unchanged, so it is blind to concept drift.
- Rule out label pipeline changes (new review guidelines) that mimic concept drift.
- Localize by segment; keep a randomized holdout when the model's own decisions hide labels.

**Deep dive:** [Data Drift and Retraining](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md), [Monitoring ML and LLM Systems in Production](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md)

</details>

<details>
<summary><b>Q39. When should drift trigger retraining?</b></summary>

**Answer:** Drift should trigger investigation; retraining is justified when the drift is real (not a pipeline bug), practically significant and persistent on features that matter, is hurting or likely to hurt performance beyond an agreed floor, and enough labeled data from the new regime exists for retraining to help. Even then, the retrained model is promoted only if it beats the current one on a recent time-based holdout, usually via champion-challenger and a canary. Do not retrain for a pipeline bug (fix the data), pure prior shift (recalibrate or move the threshold), a concept change with no labels yet (rules, fallback, human review), or features that lost their signal (redesign). Most teams combine a scheduled baseline retrain with performance-triggered retrains and drift-triggered investigations. Example: PSI 0.3 on `merchant_category` for 3 days, data checks pass, and mature AUC on that segment fell from 0.91 to 0.86, which justifies retraining on the last 8 weeks.

**Key points to mention:**
- Set the scheduled cadence by plotting a frozen model's metric against model age (decay curve).
- Training window: sliding after sudden drift, recency weighting for incremental drift, multiple seasons for recurring drift.
- Retraining can make things worse: corrupted data, immature labels, feedback loops; gate it.
- Thresholds are business decisions; backtest the trigger policy on historical incidents.

**Deep dive:** [Data Drift and Retraining](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md)

</details>

<details>
<summary><b>Q40. Explain the Population Stability Index. How do you compute it, and what are its weaknesses?</b></summary>

**Answer:** PSI measures how much a feature's distribution moved between a reference and a current window. Bin the feature using reference quantiles, compute the reference share $e_i$ and current share $a_i$ per bin, and sum $(a_i - e_i)\ln(a_i / e_i)$; it is a symmetric KL divergence. Rules of thumb from credit scoring: below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 significant. For reference quartiles of 25% each and current shares 10%, 20%, 30%, 40%, PSI is about 0.228, moderate drift. Weaknesses: it is sensitive to binning, empty bins need an epsilon, changes within a bin are invisible, the thresholds are not calibrated to your business, and it says nothing about impact on the model.

**Key points to mention:**
- Unlike KS p-values, PSI does not grow automatically with sample size, so it suits alerting on large data.
- Store bin edges with the model version; merge rare categories.
- Weight drift by feature importance and pair it with performance metrics.
- Very small windows make PSI noisy.

**Deep dive:** [Data Drift and Retraining](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md)

</details>

<details>
<summary><b>Q41. You run a KS test on 200 features daily and 40 are flagged as drifting. What do you do?</b></summary>

**Answer:** I assume most are statistically significant but practically irrelevant: with a million rows per side, the KS critical gap is about 0.002, so even a 0.01-standard-deviation shift is "significant", and 200 daily tests at $\alpha = 0.05$ produce about 10 false positives by chance. But 40 is four times that, so something real is likely in there, often one shared upstream change. I triage by grouping flagged features by upstream source and start time and ranking by drift times feature importance, and I check whether prediction distributions and performance moved. Long term, I switch alerting to effect sizes (PSI, JS, normalized Wasserstein), require persistence over several windows, and correct for multiple testing (Bonferroni or Benjamini-Hochberg) if p-values remain.

**Key points to mention:**
- Significance is not impact; big data makes every test fire.
- One broken table often explains many features at once.
- Tier alerts so only data-quality failures and performance drops page someone.
- Calibrate alert thresholds by replaying history against known incidents.

**Deep dive:** [Data Drift and Retraining](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md)

</details>

<details>
<summary><b>Q42. How would you monitor a fraud model whose labels (chargebacks) arrive up to 60 days later?</b></summary>

**Answer:** I layer fast label-free signals with slow label-based ones. Immediately: data quality (schema, nulls, ranges), feature drift, score distribution and flag rate. Within days: proxies such as analyst-confirmed precision on reviewed alerts and customer complaints. Over weeks: backfill chargebacks onto the prediction log and compute recall and cost by prediction cohort, always reporting label coverage and comparing cohorts at the same label age (for example AUC at 14 days maturity). A small random sample reviewed regardless of score keeps recall estimates unbiased, because blocked transactions never produce chargebacks.

**Key points to mention:**
- Early labels are biased toward fast-reported fraud types.
- Prevalence shifts move precision with no model change; monitor alert volume against review capacity.
- Fraudsters adapt, so expect sudden concept drift; keep fast-to-update rules alongside the model.
- Dashboards show operating-point metrics (precision@k, daily cost), not just AUC.

**Deep dive:** [Data Drift and Retraining](../01-classical-ml/08-data-drift-concept-drift-and-retraining.md), [Classification Metrics and Imbalanced Data](../01-classical-ml/04-classification-metrics-and-imbalanced-data.md), [Design a Real-Time Fraud Detection System](../11-system-design/05-design-a-real-time-fraud-detection-system.md)

</details>

## Classic algorithms: kNN, SVM and Naive Bayes

<details>
<summary><b>Q43. Explain how kNN works, how you choose k, and why feature scaling is critical.</b></summary>

**Answer:** kNN stores the training set; to predict, it computes the distance from the query to every training point, takes the $k$ closest and votes (classification) or averages (regression). Training is essentially free and inference is $O(nd)$ per query with brute force, so memory and latency grow with the data. $k$ controls bias versus variance: $k = 1$ memorizes the training set and is sensitive to label noise, while $k = n$ always predicts the global majority. I choose $k$ by cross-validation, with odd values for binary problems and distance weighting when neighbor distances vary. Because it relies on raw distances, features with large ranges dominate: house size in square meters would swamp bedroom count unless I standardize inside the pipeline.

**Key points to mention:**
- Speedups: KD-trees or ball trees in low dimensions, ANN indexes (HNSW, IVF, PQ) in high dimensions; measure recall@k against exact search.
- Under imbalance, large $k$ swamps the minority; use weighted voting and PR-based metrics.
- Cover and Hart (1967): asymptotically, 1-NN error is at most twice the Bayes error.
- kNN over embeddings is the retrieval core of RAG and semantic search.

**Deep dive:** [kNN, SVMs and Naive Bayes](../01-classical-ml/09-knn-svm-naive-bayes-and-other-classic-algorithms.md), [Vector Search: ANN, HNSW, IVF and PQ](../08-rag/06-vector-search-ann-hnsw-ivf-and-product-quantization.md)

</details>

<details>
<summary><b>Q44. What is the curse of dimensionality? If it is real, why does vector search over 1,536-dimensional embeddings work?</b></summary>

**Answer:** As dimensions grow, data becomes sparse, neighborhoods must stretch across most of every feature's range, and distances concentrate so the nearest and farthest points look similar. To capture 1% of uniformly distributed data, a sub-cube needs a side of $0.01^{1/d}$: 0.01 in 1-D, about 0.63 in 10-D and about 0.955 in 100-D. Beyer et al. (1999) showed the max/min distance ratio tends to 1 for many distributions. Embedding search still works because embeddings are not uniform random vectors: they are trained so semantic similarity maps to geometric closeness, and the data lies near a much lower-dimensional manifold. The real test is retrieval quality (recall@k on labeled query-document pairs), not the dimension count.

**Key points to mention:**
- Mitigations for classic kNN: feature selection, PCA, learned metrics.
- Hubness: a few points appear in many neighbor lists in high dimensions.
- Use the similarity the embedding model was trained with (often cosine on normalized vectors).
- Measure relative contrast (nearest vs random-pair distance) on your own data.

**Deep dive:** [kNN, SVMs and Naive Bayes](../01-classical-ml/09-knn-svm-naive-bayes-and-other-classic-algorithms.md), [Text Embeddings and Similarity Search](../08-rag/02-text-embeddings-and-similarity-search.md)

</details>

<details>
<summary><b>Q45. Explain SVMs: the margin, support vectors, the C parameter and the kernel trick.</b></summary>

**Answer:** An SVM finds the hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$ that separates classes with the widest margin, of width $2/\lVert \mathbf{w} \rVert$; the support vectors are the points on or inside the margin, and they alone determine the boundary. The soft-margin objective $\frac{1}{2}\lVert \mathbf{w} \rVert^2 + C\sum_i \max(0, 1 - y_i f(\mathbf{x}_i))$ uses hinge loss, and $C$ prices violations: large $C$ gives a narrow margin and low bias, small $C$ a wider, more regularized margin, so $C$ acts like $1/\lambda$. The kernel trick exploits the fact that the dual only uses dot products: $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x}) \cdot \phi(\mathbf{z})$ gives a linear boundary in a rich feature space without building $\phi$. The RBF kernel $\exp(-\gamma\lVert \mathbf{x} - \mathbf{z} \rVert^2)$ sets an influence radius: large $\gamma$ memorizes, small $\gamma$ is nearly linear. Kernel SVMs need $O(n^2)$ kernel evaluations, so they fit small to medium data.

**Key points to mention:**
- Standardize first: the margin is a Euclidean distance in feature units.
- Tune $C$ and $\gamma$ jointly on a log grid.
- SVM scores are not probabilities; use Platt scaling or `CalibratedClassifierCV`.
- At millions of rows switch to linear SVM, logistic regression, Nystroem or random Fourier features, or GBDT.

**Deep dive:** [kNN, SVMs and Naive Bayes](../01-classical-ml/09-knn-svm-naive-bayes-and-other-classic-algorithms.md)

</details>

<details>
<summary><b>Q46. Why does Naive Bayes work despite its false independence assumption, and how does it compare with logistic regression?</b></summary>

**Answer:** Naive Bayes scores each class as $\log P(c) + \sum_w \log P(w \mid c)$ from simple counts, with Laplace smoothing so an unseen word does not zero out a class. Classification only needs the right class to score highest, and independence errors often distort all classes similarly, so the argmax survives even when the probabilities are wrong (Domingos and Pazzani, 1997); it also has few parameters and low variance, so it works with little data. Its probabilities are usually overconfident, because correlated words like "New" and "York" are counted as independent evidence, so calibrate before thresholding. It is generative (models $P(\mathbf{x} \mid c)$), while logistic regression is discriminative (models $P(c \mid \mathbf{x})$); Ng and Jordan (2001) showed NB approaches its asymptotic error with fewer examples, while logistic regression reaches lower error with more data. So NB is a great first model on hundreds of labeled texts, and logistic regression or a linear SVM usually wins at scale.

**Key points to mention:**
- Without smoothing, one unseen word vetoes a class; $\alpha$ is a Dirichlet prior and a tunable hyperparameter.
- Compute in log space to avoid underflow.
- Training is one counting pass: incremental (`partial_fit`) and trivially parallel.
- Plot learning curves for both models to find the crossover at your label volume.

**Deep dive:** [kNN, SVMs and Naive Bayes](../01-classical-ml/09-knn-svm-naive-bayes-and-other-classic-algorithms.md), [Probability and Statistics](../00-foundations/04-probability-and-statistics.md)

</details>

## Unsupervised learning and anomaly detection

<details>
<summary><b>Q47. Walk me through k-means, k-means++ and how you choose k.</b></summary>

**Answer:** k-means picks $k$ centroids, assigns each point to the nearest one, moves each centroid to the mean of its points, and repeats until assignments stop changing. It minimizes inertia, $\sum_i \lVert x_i - \mu_{c_i} \rVert^2$; each step can only lower it, so it always converges, but only to a local minimum, at $O(nkd)$ per iteration. k-means++ spreads initial seeds by picking each next centroid with probability proportional to its squared distance from the nearest chosen one, giving an $O(\log k)$ approximation guarantee in expectation, and I still use several restarts (`n_init`). To choose $k$, inertia always decreases, so I use the elbow method, the silhouette score $(b - a)/\max(a, b)$, or BIC for Gaussian mixtures, then check stability across seeds and whether humans can name and act on the clusters. Often the downstream use constrains $k$ more than any metric.

**Key points to mention:**
- k-means is the hard-assignment, equal-spherical-covariance limit of EM for a Gaussian mixture.
- Silhouette is $O(n^2)$; compute it on a sample.
- Scale features first; remove extreme outliers before k-means++ seeding.
- For text embeddings, L2-normalize first so Euclidean distance matches cosine.

**Deep dive:** [Unsupervised Learning](../01-classical-ml/10-unsupervised-learning-clustering-and-dimensionality-reduction.md)

</details>

<details>
<summary><b>Q48. When does k-means fail, and what would you use instead (DBSCAN, HDBSCAN, GMM)?</b></summary>

**Answer:** k-means boundaries are straight hyperplanes halfway between centroids, so it fails on non-spherical shapes (moons, rings), clusters with very different sizes or densities, data with outliers (means get dragged), unscaled features, and when $k$ is unknown. DBSCAN marks points with at least `min_samples` neighbors within radius `eps` as core points, grows clusters through chains of core points and labels the rest as noise, so it handles arbitrary shapes and gives an explicit noise label. HDBSCAN removes the single-`eps` limitation by considering all density levels and keeping the most stable clusters, with `min_cluster_size` as the main knob. A Gaussian mixture handles elliptical or overlapping clusters with soft membership, chosen by BIC. Hierarchical clustering suits small data where a taxonomy is wanted, and k-medoids handles non-Euclidean distances.

**Key points to mention:**
- Choose DBSCAN's `eps` with a k-distance plot; it struggles with varying density and high dimensions.
- Reduce dimensions first (for example UMAP) before density clustering of embeddings.
- Full-covariance GMMs overfit in high dimensions; use diagonal covariance or reduce dimensions.
- DBSCAN does not natively predict new points; assign to the nearest core point or refit.

**Deep dive:** [Unsupervised Learning](../01-classical-ml/10-unsupervised-learning-clustering-and-dimensionality-reduction.md)

</details>

<details>
<summary><b>Q49. Explain PCA, how many components to keep, and how it compares with t-SNE and UMAP.</b></summary>

**Answer:** PCA finds orthogonal directions of maximum variance: the eigenvectors of the covariance matrix of centered data, ordered by eigenvalue, where each eigenvalue is the variance along its direction; projecting onto the top $m$ gives the best $m$-dimensional linear approximation in squared error. With covariance $\begin{pmatrix}2.5 & 2\\2 & 2.5\end{pmatrix}$ the eigenvalues are 4.5 and 0.5, so PC1 explains 90%. I keep enough components for a target cumulative explained variance (say 95%) or choose $m$ by downstream validation, and I standardize first so large-unit features do not dominate. t-SNE and UMAP are nonlinear and preserve local neighborhoods, so they are for visualizing embeddings, not for compression; UMAP is faster and can transform new points. In t-SNE plots, cluster sizes and inter-cluster distances are not meaningful and results change with perplexity and seed, so treat them as hypotheses to verify in the original space.

**Key points to mention:**
- Variance is not relevance to the label; PCA is unsupervised and linear.
- Computed via SVD in practice; fit on training data only.
- Common pipeline: PCA to about 50 dimensions, then t-SNE or UMAP to 2.
- For embedding compression in RAG, measure recall@k, not explained variance.

**Deep dive:** [Unsupervised Learning](../01-classical-ml/10-unsupervised-learning-clustering-and-dimensionality-reduction.md), [Linear Algebra for ML](../00-foundations/02-linear-algebra-for-ml.md)

</details>

<details>
<summary><b>Q50. How does isolation forest work, and how would you evaluate an anomaly detector with almost no labels?</b></summary>

**Answer:** Isolation forest builds many random trees on small subsamples, splitting on random features at random values. Anomalies are few and different, so they get isolated in very few splits; the score is $s = 2^{-E[h(x)]/c(n)}$, where $E[h(x)]$ is the average path length and $c(n)$ the expected path length for the subsample size, so short paths mean anomalous (with subsample 256, an average path of 4 gives about 0.76, while about 10.24 gives 0.5). It runs in linear time with little memory and no distance computations, but its axis-parallel splits miss some patterns and it struggles with local anomalies inside dense regions, where LOF does better. Without labels, I inject synthetic anomalies to check detection, have analysts review the top-ranked cases to compute precision@k, and set the threshold from review capacity rather than a guessed contamination rate.

**Key points to mention:**
- Remove ID-like and timestamp features, which make every point look unique.
- Once some labels exist, add a supervised model and track precision and recall on confirmed cases.
- Monitor alert volume over time as a drift signal.
- Extended Isolation Forest addresses the axis-parallel bias.

**Deep dive:** [Unsupervised Learning](../01-classical-ml/10-unsupervised-learning-clustering-and-dimensionality-reduction.md)

</details>

## Rapid-fire round

| Question | Answer |
|---|---|
| Which model do you deploy after cross-validation? | Refit the chosen configuration on all training data; the CV score is its performance estimate. |
| Typical k in k-fold CV? | 5 or 10; repeated k-fold for small data, a single holdout for huge data. |
| How large should a test set be? | Large enough for the needed precision: at 90% accuracy, 1,000 examples give about plus or minus 1.9 points. |
| SMOTE lifted CV recall from 0.55 to 0.95 but production is 0.5. Why? | Resampling before splitting leaked synthetic copies into validation; resample only inside training folds. |
| Can test $R^2$ be negative? | Yes: the model is worse than predicting the mean, a sign of overfitting, bugs or shift. |
| Null accuracy? | $1 - \pi$, the accuracy of always predicting the majority class. |
| Baseline PR-AUC of a random model? | The prevalence $\pi$. |
| Probabilistic meaning of ROC-AUC? | P(random positive scores higher than random negative), ties counted as half. |
| AUC clearly below 0.5? | Inverted ranking: usually flipped labels, wrong `predict_proba` column or a sign error. |
| Cost-optimal threshold with calibrated probabilities? | $C_{FP}/(C_{FP} + C_{FN})$. |
| F1-optimal threshold for a calibrated model? | Half the best achievable F1. |
| Macro or weighted F1 for imbalanced multiclass? | Macro as headline, since it exposes weak rare classes; always add the per-class report. |
| Platt or isotonic calibration? | Platt for small calibration sets; isotonic when you have more data and need flexibility. |
| Brier score? | Mean squared error between predicted probability and the 0/1 label; compare with $\pi(1-\pi)$. |
| Optimal constant prediction under MSE and under MAE? | The mean and the median. |
| RMSE much larger than MAE means? | A few very large errors; inspect the worst residuals and segments. |
| Under-stocking costs 3x over-stocking: what to predict? | The 0.75 quantile, trained with pinball loss. |
| Gradient of log loss with respect to the logit? | $p - y$. |
| VIF warning threshold? | About 10, which corresponds to $R_j^2 = 0.9$. |
| Normal equation or gradient descent? | Normal equation (QR/SVD) for modest $d$ in memory; gradient descent for huge or streaming data. |
| What fraction of rows is out-of-bag? | About 37% ($1/e$). |
| When is OOB error untrustworthy? | Time series, grouped rows or duplicates, and when preprocessing was fit on all data. |
| XGBoost optimal leaf weight? | $w^* = -G/(H+\lambda)$. |
| What do XGBoost's gamma and lambda do? | Gamma is the minimum gain to split; lambda shrinks leaf weights. |
| Halve the learning rate in boosting? | Roughly double the trees; usually slightly better generalization, slower inference. |
| Do trees need feature scaling? | No, splits are per-feature thresholds. |
| How to encode hour of day for a linear model? | Sine and cosine of $2\pi t/24$ so hour 23 sits next to hour 0. |
| Hashing trick main risk? | Collisions and lost interpretability; use a stable hash shared by training and serving. |
| PSI rule-of-thumb thresholds? | Below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 significant. |
| Correct for label shift without retraining? | Multiply calibrated class probabilities by $\pi_{\text{new}}/\pi_{\text{old}}$ and renormalize. |
| Shapes of concept drift? | Sudden, gradual, incremental and recurring. |
| What breaks Naive Bayes without smoothing? | A word unseen in a class gives probability 0 and vetoes that class. |
| What does large RBF gamma do? | Tiny influence radius, a wiggly boundary that memorizes training data. |
| kNN at k = 1? | Zero training error, high variance, sensitive to label noise. |
| Cluster cosine-trained embeddings with k-means? | L2-normalize first; then squared Euclidean distance equals $2 - 2\cos$. |
| Is k-means guaranteed to find the global optimum? | No, only a local minimum; use k-means++ and several restarts. |
| Silhouette score range? | -1 to 1; higher means tighter, better-separated clusters. |
| No free lunch theorem in one line? | Averaged over all problems no algorithm wins, so validate on your own data. |

## How to practice

- **Answer out loud first.** Say your answer before opening the collapsible; silent recognition feels like knowledge but is not.
- **Time yourself to 2 minutes.** Lead with a one-sentence definition, then mechanism, trade-off and one concrete number. Stop at 2 minutes and check what you skipped.
- **Explain it to a non-expert.** If you cannot explain target encoding leakage or the ROC-vs-precision paradox to a product manager, you do not own it yet.
- **Then go deeper.** Use the "Key points" bullets as the follow-ups an interviewer will ask next, and practice the second and third layer of each answer.
- **Close the loop.** For every question you fumble, reread the linked chapter's interview section, and redo the rapid-fire table until you clear it in under 10 minutes.
- **Mix in code.** Pair this bank with [Coding Exercises: Classical ML and Deep Learning from Scratch](./01-coding-exercises-ml-from-scratch.md) so you can implement what you explain.
