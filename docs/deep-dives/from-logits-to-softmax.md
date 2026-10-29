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


# 2. The exponential family and the canonical link

In the previous chapter, we saw how Generalized Linear Models (GLMs) extend classical linear regression to response variables with different probability distributions.

A fundamental property of standard GLMs is that the conditional response distribution belongs to the **exponential family of distributions**. This family includes many distributions used in statistical modeling, including the normal, Bernoulli, binomial, Poisson, and gamma distributions.

The exponential-family representation identifies a particular parameterization from which we can derive the expected value, variance, and a distinguished link function, called the *canonical link*.

## 2.1 Defining the exponential family

In the parameterization commonly used in GLMs, an exponential-family distribution can be expressed as follows, for a scalar response and $a(\phi)>0$:

$$
\boxed{
f(y;\theta,\phi)=
\exp\left[
\frac{y\theta-b(\theta)}{a(\phi)}
+c(y,\phi)
\right].
}
$$

The terms have different roles:

| Term | Meaning |
| --- | --- |
| $y$ | Observed response value |
| $\theta$ | Natural or canonical parameter |
| $\phi$ | Dispersion parameter, where applicable |
| $b(\theta)$ | Cumulant function, related to normalization and the distribution's moments |
| $a(\phi)$ | Scale function, which enters the variance |
| $c(y,\phi)$ | Term independent of $\theta$, contributing to the distribution's form and normalization |

The dependence on the natural parameter $\theta$ appears through

$$
y\theta-b(\theta).
$$

The function $b(\theta)$ plays a central role: besides contributing to normalization, it contains information about the mean and variance. Under the usual regularity conditions,

$$
\boxed{\mathbb E[Y]=b'(\theta)}
$$

and

$$
\boxed{\mathrm{Var}(Y)=a(\phi)b''(\theta).}
$$

The mean and variance can therefore be obtained from the first and second derivatives of the cumulant function, respectively. Let us derive these results.

## 2.2 Deriving the mean and variance

### The normalization constraint

By definition, a probability mass function or density must be normalized. For a continuous random variable,

$$
\int_{\mathcal Y}f(y;\theta,\phi)\,dy=1,
$$

where $\mathcal Y$ is the distribution's support. In the discrete case, integration is replaced by summation:

$$
\sum_{y\in\mathcal Y}f(y;\theta,\phi)=1.
$$

We will use the continuous case for the proof; the discrete argument is analogous. Assume that the support does not depend on $\theta$ and that the regularity conditions needed to interchange differentiation and integration hold.

### Deriving the expected value

Starting from

$$
\int_{\mathcal Y}f(y;\theta,\phi)\,dy=1,
$$

differentiate both sides with respect to $\theta$, holding $\phi$ fixed:

$$
\frac{\partial}{\partial\theta}
\int_{\mathcal Y}f(y;\theta,\phi)\,dy=0.
$$

Interchanging differentiation and integration gives

$$
\int_{\mathcal Y}
\frac{\partial f(y;\theta,\phi)}{\partial\theta}\,dy=0.
$$

Recall the exponential-family form:

$$
f(y;\theta,\phi)=
\exp\left[
\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi)
\right].
$$

Applying the chain rule,

$$
\frac{\partial f(y;\theta,\phi)}{\partial\theta}
=f(y;\theta,\phi)
\frac{\partial}{\partial\theta}
\left[
\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi)
\right].
$$

Since $a(\phi)$ and $c(y,\phi)$ do not depend on $\theta$,

$$
\boxed{
\frac{\partial f(y;\theta,\phi)}{\partial\theta}
=f(y;\theta,\phi)\frac{y-b'(\theta)}{a(\phi)}.
}
$$

Substitute this expression into the integral:

$$
\int_{\mathcal Y}
f(y;\theta,\phi)\frac{y-b'(\theta)}{a(\phi)}\,dy=0.
$$

Since $a(\phi)$ is constant with respect to $y$, multiply both sides by it:

$$
\int_{\mathcal Y}[y-b'(\theta)]f(y;\theta,\phi)\,dy=0.
$$

Expanding,

$$
\int_{\mathcal Y}y f(y;\theta,\phi)\,dy
-b'(\theta)\int_{\mathcal Y}f(y;\theta,\phi)\,dy=0.
$$

The first integral is $\mathbb E[Y]$; the second equals $1$ by normalization. Therefore,

$$
\mathbb E[Y]-b'(\theta)=0,
$$

which gives

$$
\boxed{\mathbb E[Y]=b'(\theta).}
$$

Thus **the first derivative of the cumulant function equals the distribution's expected value**.

### Deriving the variance

Use the normalization constraint again, now differentiating twice with respect to $\theta$:

$$
\int_{\mathcal Y}
\frac{\partial^2 f(y;\theta,\phi)}{\partial\theta^2}\,dy=0.
$$

We have already obtained

$$
\frac{\partial f(y;\theta,\phi)}{\partial\theta}
=f(y;\theta,\phi)\frac{y-b'(\theta)}{a(\phi)}.
$$

Differentiate again using the product rule, with $\phi$ fixed:

$$
\frac{\partial^2 f}{\partial\theta^2}
=\frac{\partial f}{\partial\theta}
\frac{y-b'(\theta)}{a(\phi)}
-f(y;\theta,\phi)\frac{b''(\theta)}{a(\phi)}.
$$

Substituting the first derivative,

$$
\boxed{
\frac{\partial^2 f}{\partial\theta^2}
=f(y;\theta,\phi)
\left[
\frac{(y-b'(\theta))^2}{a(\phi)^2}
-\frac{b''(\theta)}{a(\phi)}
\right].
}
$$

Integrating with respect to $y$ gives

$$
\int_{\mathcal Y}f(y;\theta,\phi)
\left[
\frac{(y-b'(\theta))^2}{a(\phi)^2}
-\frac{b''(\theta)}{a(\phi)}
\right]\,dy=0.
$$

Recognizing $b'(\theta)=\mu=\mathbb E[Y]$, and using normalization and the definition of expectation,

$$
\frac{\mathbb E[(Y-\mu)^2]}{a(\phi)^2}
-\frac{b''(\theta)}{a(\phi)}=0.
$$

By definition,

$$
\mathrm{Var}(Y)=\mathbb E[(Y-\mu)^2].
$$

Therefore,

$$
\frac{\mathrm{Var}(Y)}{a(\phi)^2}
=\frac{b''(\theta)}{a(\phi)}.
$$

Multiplying by $a(\phi)^2$ yields

$$
\boxed{\mathrm{Var}(Y)=a(\phi)b''(\theta).}
$$

We have proved both fundamental properties:

$$
\boxed{
\begin{aligned}
\mathbb E[Y]&=b'(\theta),\\
\mathrm{Var}(Y)&=a(\phi)b''(\theta).
\end{aligned}
}
$$

These results directly connect the natural parameter, cumulant function, and first two moments. In particular, since $a(\phi)>0$ and variance is nonnegative,

$$
b''(\theta)\geq0.
$$

Hence $b(\theta)$ is convex in the natural parameter. This will have important consequences for likelihood optimization.

## 2.3 The canonical link function

Recall the linear predictor and link:

$$
\eta_i=x_i^\top\beta,\qquad g(\mu_i)=\eta_i.
$$

Exponential-family theory gives

$$
\mu=b'(\theta).
$$

If this relationship is invertible, the natural parameter can be expressed in terms of the mean:

$$
\theta=(b')^{-1}(\mu).
$$

A GLM uses the **canonical link** when its linear predictor equals the natural parameter:

$$
\boxed{\eta=\theta.}
$$

Consequently, $g(\mu)=\theta$, and

$$
\boxed{g(\mu)=(b')^{-1}(\mu).}
$$

This derives the canonical link directly from the response distribution. Its inverse is

$$
\boxed{\mu=b'(\eta).}
$$

Once we identify $b(\theta)$, we can derive both the canonical link and the relationship that maps the linear predictor to the mean. The canonical link is optional, but it has particularly favorable mathematical properties.

## 2.4 Why choose the canonical link?

One advantage is that it simplifies **maximum-likelihood estimation** of the coefficients.

Suppose we have $n$ conditionally independent observations with exponential-family distributions. The log-likelihood is

$$
\ell(\beta)=\sum_{i=1}^{n}
\left[
\frac{y_i\theta_i-b(\theta_i)}{a(\phi)}
+c(y_i,\phi)
\right].
$$

To find the maximizing coefficients, calculate its gradient. By the chain rule,

$$
\frac{\partial\ell(\beta)}{\partial\beta}
=\sum_{i=1}^{n}
\frac{\partial\ell_i}{\partial\theta_i}
\frac{\partial\theta_i}{\partial\eta_i}
\frac{\partial\eta_i}{\partial\beta}.
$$

From the log-likelihood expression,

$$
\frac{\partial\ell_i}{\partial\theta_i}
=\frac{y_i-b'(\theta_i)}{a(\phi)}.
$$

Since $b'(\theta_i)=\mu_i$ and $\partial\eta_i/\partial\beta=x_i$,

$$
\frac{\partial\ell(\beta)}{\partial\beta}
=\sum_{i=1}^{n}
\frac{y_i-\mu_i}{a(\phi)}
\frac{\partial\theta_i}{\partial\eta_i}\,x_i.
$$

A general link therefore introduces the factor $\partial\theta_i/\partial\eta_i$. For the canonical link,

$$
\theta_i=\eta_i,\qquad
\frac{\partial\theta_i}{\partial\eta_i}=1.
$$

The gradient simplifies to

$$
\boxed{
\nabla_\beta\ell(\beta)
=\sum_{i=1}^{n}\frac{y_i-\mu_i}{a(\phi)}\,x_i.
}
$$

It depends directly on the differences between observations and expected values, weighted by their covariates. This structure is useful in numerical estimation methods.

### Concavity of the log-likelihood

A second advantage concerns second derivatives. With a canonical link and fixed dispersion,

$$
\ell(\beta)=\sum_{i=1}^{n}
\left[
\frac{y_i x_i^\top\beta-b(x_i^\top\beta)}{a(\phi)}
+c(y_i,\phi)
\right].
$$

Differentiating again gives the Hessian:

$$
\boxed{
H(\beta)=
-\sum_{i=1}^{n}
\frac{b''(\theta_i)}{a(\phi)}x_ix_i^\top.
}
$$

Since $b''(\theta_i)\geq0$ and $a(\phi)>0$, the Hessian is negative semidefinite. Under these conditions, the log-likelihood is concave in the coefficients, so any local maximum is also global.

Concavity alone does not guarantee a finite, unique estimator. Additional conditions concern parameter identifiability and the existence of a maximum.

The canonical choice therefore offers two main advantages:

1. It simplifies the maximum-likelihood estimating equations.
2. It gives the log-likelihood a concave structure in the coefficients under the stated conditions.

These properties do not mean that the canonical link always best describes the observed phenomenon.

## 2.5 Why use a different link?

The link function is a modeling choice specifying the relationship between the expected response and the explanatory variables. Mathematical convenience does not necessarily mean the best fit to the relationship in the data.

For a Bernoulli response,

$$
Y\sim\mathrm{Bernoulli}(p),
$$

the canonical link, as we will derive next, is the logit:

$$
g(p)=\ln\left(\frac{p}{1-p}\right).
$$

An alternative is the *probit* link:

$$
g(p)=\Phi^{-1}(p),
$$

where $\Phi$ is the standard normal cumulative distribution function. Inverting it gives

$$
p=\Phi(\eta).
$$

The conditional response distribution remains Bernoulli, but the dependence of the success probability on the predictor changes.

A noncanonical link may be motivated by the phenomenon being studied, the desired interpretation of coefficients, or the model's ability to represent observed relationships. It must remain compatible with the mean's domain and the model assumptions.

**The canonical link is therefore a mathematically privileged choice, not a requirement for defining a GLM.**

---

Next, we apply these properties to the Bernoulli distribution. We will express its probability mass function in exponential-family form, identify the cumulant function, and derive

$$
\theta=\ln\left(\frac{p}{1-p}\right).
$$

Setting $\eta=\theta$ gives the logit link and, by inversion, the logistic or sigmoid function. This provides the foundation for the subsequent generalization to the multinomial distribution and softmax.
