# From logits to softmax: statistical foundations

> **Status:** introductory English draft. The full, separately developed Italian tutorial on generalized linear models, categorical exponential families, multinomial regression, cross-entropy, and probability calibration still requires a complete editorial translation and integration.

## 1. Scores and probabilities

For $K$ possible classes, a network produces a vector of *logits* $z=(z_1,\dots,z_K)\in\mathbb R^K$. The softmax operation maps these unconstrained real values onto the probability simplex:

$$
p_j=\mathrm{softmax}(z)_j=\frac{e^{z_j}}{\sum_{k=1}^{K}e^{z_k}}.
$$

Therefore $p_j>0$ for every class with finite logits and $\sum_j p_j=1$ (in exact arithmetic).

## 2. Connection to categorical and multinomial logistic models

If an observation belongs to exactly one of $K$ categories, represent it using a one-hot vector $y$. Its categorical likelihood is

$$
P(Y=y)=\prod_{j=1}^{K}p_j^{y_j},\qquad \sum_j y_j=1.
$$

Choose category $K$ as a reference and define the $K-1$ natural log-odds parameters

$$
\theta_j=\log\frac{p_j}{p_K},\qquad j=1,\dots,K-1.
$$

Solving these equations together with $\sum_jp_j=1$ yields

$$
p_j=\frac{e^{\theta_j}}{1+\sum_{k=1}^{K-1}e^{\theta_k}},\quad
p_K=\frac{1}{1+\sum_{k=1}^{K-1}e^{\theta_k}}.
$$

Now write $\theta_j=z_j-z_K$. The previous equations become the full $K$-logit softmax. This is the inverse of the canonical link under the reference-category parameterization of the categorical distribution.

## 3. Why a reference category is enough

For any constant $c$:

$$
\mathrm{softmax}(z+c\mathbf 1)=\mathrm{softmax}(z).
$$

Thus the $K$ logits have one redundant degree of freedom: only their differences matter. Fixing $z_K=0$ is an **identifiability convention**, not a restriction on the resulting probability vector. A positive affine transformation with scaling $a\ne1$ generally **does change** the probabilities, so softmax is invariant to common **translations**, not to all affine transformations.

## 4. Likelihood and cross-entropy

For one one-hot target $y$ and predictions $p$:

$$
-\log P(Y=y)=-\sum_{j=1}^{K}y_j\log p_j.
$$

This is categorical cross-entropy. Differentiating with respect to the logits gives the important result

$$
\frac{\partial\mathcal L}{\partial z_j}=p_j-y_j.
$$

The network propagates these derivatives back through the vocabulary projection and the decoder layers. In PyTorch, `F.cross_entropy(logits, target_indices)` computes a numerically stable version of this objective without an explicit prior call to softmax.

## 5. Two roles of softmax in a Transformer

- **Output head:** normalizes logits over the vocabulary and defines a categorical distribution over next tokens.
- **Self-attention:** normalizes compatibility scores over allowed sequence positions and produces aggregation weights. These weights are not automatically causal explanations or probabilities of vocabulary tokens.

A valid distribution does not by itself imply calibrated probabilities or trustworthy predictions. Those questions require separate empirical evaluation.

**Next development milestone:** incorporate the complete GLM-to-softmax derivation and Bayesian/frequentist interpretation from the longer tutorial, with worked examples and references.
