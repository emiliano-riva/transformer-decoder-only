# From Logits to Softmax

Statistical Foundations and Applications in Neural Networks

## Contents of the complete tutorial

- **Introduction:** what softmax is, where it is used, and the aims of this tutorial.
- **1. Generalized Linear Models:** from the classical linear model to the three components of a GLM.
- **2. The exponential family and the canonical link:** general form, cumulant function, mean, variance, and choice of link.
- **3. From Bernoulli to logistic regression:** natural parameter, logit, sigmoid, odds, and interpretation of coefficients.
- **4. Generalization to the multinomial distribution:** one-hot encoding, $K-1$ natural parameters, exponential-family form, and cumulant function.
- **5. Deriving the softmax function:** inversion of the link, reference category, invariance, and examples.
- **6. Softmax in neural networks:** classification, attention, maximum likelihood, cross-entropy, and gradients.
- **7. Statistical inference and uncertainty:** frequentist and Bayesian approaches, and the limitations of softmax probabilities.
- **8. Conclusions:** theoretical synthesis, applications, and final considerations.

To follow: bibliographic references and, if appropriate, an appendix with PyTorch implementations.

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

# 1. Generalized Linear Models (GLMs)

## 1.1 From the classical linear model to GLMs

The **classical linear regression model** describes the relationship between a response variable $Y$ and a set of explanatory variables through a linear combination of the model coefficients.

For a generic observation $i$, we can write

$$
Y_i=\beta_0+\beta_1x_{i1}+\cdots+\beta_px_{ip}+\varepsilon_i,
$$

where:

- $Y_i$ is the response variable;
- $x_{ij}$ is the value of the $j$th explanatory variable for observation $i$;
- $\beta_j$ is the coefficient associated with $x_{ij}$;
- $\beta_0$ is the intercept;
- $\varepsilon_i$ is the error term.

Introducing the covariate vector $x_i$, including an initial component equal to $1$, and the coefficient vector $\beta$, we can express the model compactly as

$$
\boxed{Y_i=x_i^\top\beta+\varepsilon_i.}
$$

### Assumption about the conditional expectation

One of the fundamental assumptions of the linear model concerns the conditional mean of the errors:

$$
\mathbb E[\varepsilon_i\mid x_i]=0.
$$

This assumption states that, conditional on the explanatory variables, the error term has an expected value of zero.

Taking the conditional expectation of both sides of the model gives

$$
\mathbb E[Y_i\mid x_i]
=\mathbb E[x_i^\top\beta+\varepsilon_i\mid x_i].
$$

By linearity of expectation,

$$
\mathbb E[Y_i\mid x_i]
=x_i^\top\beta+\mathbb E[\varepsilon_i\mid x_i].
$$

Since we have assumed that the conditional expectation of the errors is zero,

$$
\boxed{\mu_i=\mathbb E[Y_i\mid x_i]=x_i^\top\beta.}
$$

We therefore obtain a linear relationship between the **conditional expectation of the response variable** and the coefficient vector.

This result does not require the errors to be normally distributed.

### Normality and constant variance assumptions

In the classical Gaussian linear model, we may additionally assume that the errors are conditionally normally distributed, with zero mean and constant variance:

$$
\varepsilon_i\mid x_i\sim\mathcal N(0,\sigma^2).
$$

This implies

$$
\mathbb E[\varepsilon_i\mid x_i]=0
$$

and

$$
\mathrm{Var}(\varepsilon_i\mid x_i)=\sigma^2.
$$

Since $Y_i$ is obtained by adding a deterministic component $x_i^\top\beta$ to a normally distributed error, it follows that

$$
\boxed{Y_i\mid x_i\sim\mathcal N(\mu_i,\sigma^2),}
$$

where

$$
\mu_i=x_i^\top\beta.
$$

We have thus defined a probabilistic model in which the conditional distribution of the response is normal and its mean depends linearly on the covariates.

The usual joint-likelihood formulation also assumes conditional independence of the observations; alternatively, a dependence structure must be specified explicitly.

### Limitations of the Gaussian linear model

This formulation is appropriate for many phenomena, but it may become unsuitable when the response variable has characteristics incompatible with a normal distribution with constant variance.

Consider the following examples:

- A **binary variable** can take only the values $0$ and $1$, representing, for example, the absence or presence of an event.
- A **count variable** takes nonnegative integer values, such as the number of events observed during a time interval.
- A **categorical variable** belongs to one of several mutually exclusive categories.

At least two problems arise in these cases.

The first concerns the **range of the expected value**. For a binary variable, for example,

$$
\mu_i=P(Y_i=1\mid x_i),
$$

so necessarily

$$
0\leq\mu_i\leq1.
$$

A linear combination $x_i^\top\beta$, however, can take any real value and does not guarantee that this constraint is satisfied.

The second problem concerns the **variance structure**. For a Bernoulli variable, for example,

$$
\mathrm{Var}(Y_i\mid x_i)=p_i(1-p_i).
$$

The variance therefore depends on the probability of success and is generally not constant.

These characteristics suggest the need for models that can represent distributions other than the normal distribution and appropriately connect the expected response to the explanatory variables.

**Generalized Linear Models (GLMs)** address this need by extending the linear model through an exponential-family distribution and a function that connects the expected response to the linear predictor.

## 1.2 The three components of a GLM

A Generalized Linear Model is defined by three fundamental components: a random component, a systematic component, and a link function.

### 1. Random component

The random component specifies the conditional distribution of the response variable.

In standard GLMs, this distribution belongs to the **exponential family of distributions**, which includes the normal, Bernoulli, and Poisson distributions, among others.

We can write this assumption generically as

$$
Y_i\mid x_i\sim\mathcal F(\theta_i,\phi),
$$

where:

- $\mathcal F$ denotes an exponential-family distribution;
- $\theta_i$ is the natural parameter of the distribution;
- $\phi$, where applicable, is a dispersion parameter.

We will denote the associated probability mass function or probability density function by

$$
f(y_i;\theta_i,\phi).
$$

The mathematical structure of this family will be examined in the next chapter.

### 2. Systematic component

The systematic component defines the **linear predictor**:

$$
\boxed{\eta_i=x_i^\top\beta,}
$$

where $x_i$ is the covariate vector and $\beta$ is the coefficient vector to be estimated.

The word *linear* refers to the predictor's dependence on the coefficients $\beta$, not necessarily to the original form of the explanatory variables.

The predictor may therefore include transformations of the covariates and interaction terms, provided that the relationship remains linear in the coefficients.

### 3. Link function

The link function connects the conditional expectation of the response to the linear predictor.

Define

$$
\mu_i=\mathbb E[Y_i\mid x_i].
$$

We introduce a function $g$ such that

$$
\boxed{g(\mu_i)=\eta_i=x_i^\top\beta.}
$$

The link function allows us to express the expected value, which may be subject to constraints, through a linear predictor.

If the function is invertible, we can recover the expected value using its inverse:

$$
\boxed{\mu_i=g^{-1}(x_i^\top\beta).}
$$

This relationship is central to GLMs: the inverse link transforms the linear predictor into a value compatible with the mean's domain under the chosen distribution.

For example, the traditional normal linear model can use the **identity link**:

$$
g(\mu_i)=\mu_i,
$$

which gives

$$
\mu_i=x_i^\top\beta.
$$

Gaussian linear regression can therefore be interpreted as a special case of a GLM.

For a Bernoulli distribution, the canonical link is the **logit**, whose inverse is the logistic function, also called the sigmoid.

The choice of link function is not arbitrary: it must be compatible with the domain of the mean and justified by the characteristics of the model. One particularly important choice is the **canonical link**, which directly relates the linear predictor to the natural parameter of the distribution.

We will see how this choice can be derived from the structure of the exponential family and why it is particularly useful for parameter estimation.

---

The next step is to study the exponential family and identify the roles of the natural parameter and the cumulant function. These elements will allow us to derive the canonical link formally and then obtain the logit function from the Bernoulli distribution.
