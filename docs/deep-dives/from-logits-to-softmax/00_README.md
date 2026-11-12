# From Logits to Softmax

Statistical Foundations and Applications in Neural Networks

## Introduction

**Softmax** is a mathematical function widely used in machine learning to transform a vector of real-valued scores, commonly called *logits*, into a probability distribution over a finite set of categories.

Given a vector of $K$ logits $z_1,\ldots,z_K$, softmax is defined as

$$
\mathrm{softmax}(\mathbf z)_j=
\frac{e^{z_j}}{\sum_{h=1}^{K}e^{z_h}}.
$$

For finite logits in exact arithmetic, the transformation ensures that each resulting value is strictly positive and that the components sum to one.

This property makes softmax particularly natural for **multiclass classification** and **autoregressive language models**, where it represents the probability distribution of the next token. The same function also appears in Transformer *attention* mechanisms, where it serves a different purpose: it normalizes compatibility scores between elements of a sequence, producing weights used to aggregate their representations.

Softmax is not, however, a required component of every neural architecture. Its use depends on the nature of the problem and the chosen parameterization.

Why this particular normalized exponential transformation? What are its statistical foundations, and why does it combine so naturally with *cross-entropy* when training neural networks?

To answer these questions, we will begin with **Generalized Linear Models (GLMs)** and **exponential families of distributions**. We will show how the canonical link of the Bernoulli distribution leads to the *logit* function, and how its generalization to the multinomial distribution produces a vector of log-probability ratios. By inverting this relationship, we will derive softmax.

Finally, we will examine the relationship between the multinomial likelihood, cross-entropy, and gradient calculation, highlighting the mathematical continuity between estimation methods in classical statistics and the optimization procedures used in deep learning.


## Contents of the complete tutorial

- **[1. Generalized Linear Models](01-generalized-linear-models.md):** from the classical linear model to the three components of a GLM.
- **[2. The exponential family and the canonical link](02-exponential-families-and-canonical-link.md):** general form, cumulant function, mean, variance, and choice of link.
- **[3. From Bernoulli to logistic regression](03-bernoulli-logistic-regression.md):** natural parameter, logit, sigmoid, odds, and interpretation of coefficients.
- **[4. Generalization to the multinomial distribution](04-multinomial-distribution.md):** one-hot encoding, $K-1$ natural parameters, exponential-family form, and cumulant function.
- **[5. Deriving the softmax function](05-softmax-derivation,md):** inversion of the link, reference category, invariance, and examples.
- **[6. Softmax in neural networks](06-softmax-neural-networks.md):** classification, attention, maximum likelihood, cross-entropy, and gradients.
- **[7. Statistical inference and uncertainty](07-statistical-inference.md):** frequentist and Bayesian approaches, and the limitations of softmax probabilities.
- **[8. Conclusions](08-conclusions.md):** theoretical synthesis, applications, and final considerations.

To follow: bibliographic references and, if appropriate, an appendix with PyTorch implementations.
