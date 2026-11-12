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
