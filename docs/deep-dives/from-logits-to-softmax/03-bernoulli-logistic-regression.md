# 3. From the Bernoulli Distribution to Logistic Regression

In the previous chapter, we showed how representing a distribution as a member of the exponential family allows us to identify its natural parameter, derive its expected value and variance through the cumulant function, and define a particular link function known as the *canonical link*.

We now apply these results to the **Bernoulli distribution**, which is used to model binary random variables.

We will show how the natural parameter of the Bernoulli distribution leads to the *logit* function and how choosing the canonical link gives rise to the logistic regression model, whose inverse link function is the *sigmoid*.

## 3.1 The Bernoulli Distribution

Consider a random variable $Y$ that can take only two values:

$$
Y\in\{0,1\}
$$

The Bernoulli distribution is particularly well suited to modeling phenomena with two possible outcomes, conventionally called *success* and *failure*.

We write:

$$
Y\sim\mathrm{Bernoulli}(p)
$$

where $p$ denotes the probability of success:

$$
P(Y=1)=p
$$

$$
P(Y=0)=1-p
$$

The Bernoulli probability mass function can be written compactly as:

$$
\boxed{
P(Y=y)=p^y(1-p)^{1-y}
}
$$

where $y\in\{0,1\}$.

We can verify this expression by considering the two possible values separately.

If $y=1$:

$$
P(Y=1)=p^1(1-p)^0=p
$$

If $y=0$:

$$
P(Y=0)=p^0(1-p)^1=1-p
$$

The distribution is therefore completely determined by the parameter $p$. In general, $0\leq p\leq1$; for the logarithmic derivations that follow, we assume $0<p<1$. The boundary cases $p=0$ and $p=1$ are degenerate distributions and can be obtained as limits of the canonical parameterization.

In regression, however, we want this probability to depend on the characteristics of an observation.

For observation $i$, we write:

$$
Y_i\mid x_i\sim\mathrm{Bernoulli}(p_i)
$$

where:

$$
p_i=P(Y_i=1\mid x_i)
$$

Our goal is to build a model that describes the relationship between the covariates $x_i$ and the probability $p_i$.

## 3.2 Representing the Bernoulli Distribution in the Exponential Family

To identify the canonical link, we first need to express the Bernoulli probability mass function in exponential-family form.

We start with:

$$
P(Y=y)=p^y(1-p)^{1-y}
$$

Taking the natural logarithm of both sides:

$$
\ln P(Y=y)=y\ln p+(1-y)\ln(1-p)
$$

Expanding:

$$
\ln P(Y=y)=y\ln p+\ln(1-p)-y\ln(1-p)
$$

Collecting the terms that depend on $y$:

$$
\ln P(Y=y)=y[\ln p-\ln(1-p)]+\ln(1-p)
$$

Using the properties of logarithms:

$$
\ln P(Y=y)=y\ln\left(\frac{p}{1-p}\right)+\ln(1-p)
$$

Returning to exponential form:

$$
\boxed{
P(Y=y)=
\exp\left[
y\ln\left(\frac{p}{1-p}\right)+\ln(1-p)
\right]
}
$$

We compare this expression with the general form of the exponential family:

$$
f(y;\theta,\phi)=
\exp\left[
\frac{y\theta-b(\theta)}{a(\phi)}
+c(y,\phi)
\right]
$$

The term multiplying $y$ allows us to identify the **natural parameter**:

$$
\boxed{
\theta=\ln\left(\frac{p}{1-p}\right)
}
$$

For $0<p<1$, the natural parameter is finite and can take any real value. As $p$ approaches $0$ or $1$, $\theta$ approaches $-\infty$ or $+\infty$, respectively.

We have therefore found a transformation that maps the probability $p$ to a parameter defined over the entire real line.

To complete the exponential-family representation, we need to determine the cumulant function $b(\theta)$.

## 3.3 Deriving the Cumulant Function

We begin with the relationship obtained above:

$$
\theta=\ln\left(\frac{p}{1-p}\right)
$$

Exponentiating both sides:

$$
e^\theta=\frac{p}{1-p}
$$

Multiplying both sides by $1-p$:

$$
e^\theta(1-p)=p
$$

Expanding:

$$
e^\theta-e^\theta p=p
$$

Factoring out $p$:

$$
e^\theta=p(1+e^\theta)
$$

We obtain:

$$
\boxed{
p=\frac{e^\theta}{1+e^\theta}
}
$$

We can therefore also express the complementary probability in terms of $\theta$:

$$
1-p=1-\frac{e^\theta}{1+e^\theta}
$$

Hence:

$$
1-p=\frac{1}{1+e^\theta}
$$

Taking the natural logarithm:

$$
\ln(1-p)=-\ln(1+e^\theta)
$$

Let us return to the probability mass function derived in the previous section:

$$
P(Y=y)=
\exp\left[
y\ln\left(\frac{p}{1-p}\right)+\ln(1-p)
\right]
$$

Substituting the expressions in terms of $\theta$, we obtain:

$$
\boxed{
P(Y=y)=
\exp\left[
y\theta-\ln(1+e^\theta)
\right]
}
$$

Comparing this with the canonical form:

$$
f(y;\theta)=\exp[y\theta-b(\theta)]
$$

we can identify:

$$
\boxed{
b(\theta)=\ln(1+e^\theta)
}
$$

Under the parameterization considered here, the Bernoulli distribution also satisfies:

$$
a(\phi)=1
$$

$$
c(y,\phi)=0
$$

The Bernoulli distribution has no free dispersion parameter, because its variance is completely determined by the probability $p$.

We have thus identified the two fundamental elements:

$$
\boxed{
\begin{aligned}
\theta &= \ln\left(\frac{p}{1-p}\right) \\
b(\theta) &= \ln(1+e^\theta)
\end{aligned}
}
$$

## 3.4 Verifying the Expected Value and Variance

In the previous chapter, we demonstrated that, for a distribution in the exponential family under the usual regularity conditions:

$$
\mathbb{E}[Y]=b'(\theta)
$$

$$
\mathrm{Var}(Y)=a(\phi)b''(\theta)
$$

Let us verify that these relationships reproduce the moments of the Bernoulli distribution.

### Expected Value

The cumulant function is:

$$
b(\theta)=\ln(1+e^\theta)
$$

Differentiating with respect to $\theta$:

$$
b'(\theta)=\frac{e^\theta}{1+e^\theta}
$$

But from the previous section we know that:

$$
p=\frac{e^\theta}{1+e^\theta}
$$

Therefore:

$$
\boxed{
\mathbb{E}[Y]=b'(\theta)=p
}
$$

### Variance

Differentiating a second time:

$$
b''(\theta)=
\frac{\partial}{\partial\theta}
\left(
\frac{e^\theta}{1+e^\theta}
\right)
$$

Applying the quotient rule:

$$
b''(\theta)=
\frac{
e^\theta(1+e^\theta)-e^{2\theta}
}{
(1+e^\theta)^2
}
$$

Simplifying:

$$
b''(\theta)=
\frac{e^\theta}{(1+e^\theta)^2}
$$

Since:

$$
p=\frac{e^\theta}{1+e^\theta}
$$

and:

$$
1-p=\frac{1}{1+e^\theta}
$$

we obtain:

$$
b''(\theta)=p(1-p)
$$

Since $a(\phi)=1$:

$$
\boxed{
\mathrm{Var}(Y)=p(1-p)
}
$$

Representing the Bernoulli distribution in the exponential family therefore allows us to recover both its expected value and its variance correctly.

These properties also have an important consequence: the variance of the Bernoulli distribution depends on the probability of success and reaches its maximum at $p=0.5$. Indeed:

$$
p(1-p)=\frac14-\left(p-\frac12\right)^2\leq\frac14
$$

The maximum variance is therefore $1/4$.

## 3.5 The Canonical Link: The Logit Function

We can now apply the theory of generalized linear models (GLMs) directly.

For the Bernoulli distribution, we have shown that:

$$
\mu=\mathbb{E}[Y]=p
$$

and:

$$
\theta=\ln\left(\frac{p}{1-p}\right)
$$

Recall that the canonical link is obtained by setting the natural parameter equal to the linear predictor:

$$
\boxed{
\eta=\theta
}
$$

Substituting $\mu=p$, we obtain:

$$
\boxed{
g(\mu)=\ln\left(\frac{\mu}{1-\mu}\right)
}
$$

This function is called the **logit**:

$$
\boxed{
\mathrm{logit}(p)=
\ln\left(\frac{p}{1-p}\right)
}
$$

The logit function maps the open interval of probabilities onto the real line:

$$
g:(0,1)\longrightarrow\mathbb R
$$

This connection is fundamental because the linear predictor of a GLM can take any real value, whereas a probability must lie in the interval $[0,1]$.

The logit function therefore lets us model an unconstrained quantity on the real scale while retaining a correspondence with a valid probability. For every finite value of the predictor $\eta$, the logistic function returns a probability strictly between $0$ and $1$; the boundary values are reached only in the limit.

### The Logistic Function, or Sigmoid

Consider a model with a single explanatory variable:

$$
\eta=\beta_0+\beta_1x
$$

Choosing the canonical link gives:

$$
\ln\left(\frac{p}{1-p}\right)=\beta_0+\beta_1x
$$

We have already shown that:

$$
p=\frac{e^\theta}{1+e^\theta}
$$

Since $\theta=\eta$, substituting the linear predictor gives:

$$
p=\frac{e^{\beta_0+\beta_1x}}
{1+e^{\beta_0+\beta_1x}}
$$

Dividing the numerator and denominator by $e^{\beta_0+\beta_1x}$:

$$
\boxed{
p=\frac{1}{1+e^{-(\beta_0+\beta_1x)}}
}
$$

This is the **logistic function**, often called the *logistic sigmoid* (the term *sigmoid* can also refer to other S-shaped functions):

$$
\boxed{
\sigma(\eta)=\frac{1}{1+e^{-\eta}}
}
$$

The logistic function is therefore the inverse of the logit:

$$
\boxed{
\sigma=\mathrm{logit}^{-1}
}
$$

The sigmoid is a strictly increasing function whose values lie between $0$ and $1$, excluding the endpoints.

In particular:

$$
\lim_{\eta\to-\infty}\sigma(\eta)=0
$$

$$
\sigma(0)=0.5
$$

$$
\lim_{\eta\to+\infty}\sigma(\eta)=1
$$

We can thus express the conditional probability of the binary response as:

$$
\boxed{
P(Y_i=1\mid x_i)=
\sigma(x_i^\top\beta)
}
$$

This is the **binary logistic regression** model.

In the vector notation $x_i^\top\beta$, the intercept $\beta_0$ is included by assuming that the first component of the covariate vector equals $1$. For example, with one explanatory variable, $x_i=(1,x_i)^\top$ and $\beta=(\beta_0,\beta_1)^\top$; to avoid ambiguity between the vector and the scalar, we can write more precisely $\mathbf{x}_i=(1,x_i)^\top$ and $\mathbf{x}_i^\top\beta=\beta_0+\beta_1x_i$.

The key distinction is that the model is linear on the log-odds scale, but not on the probability scale.

## 3.6 Statistical Interpretation of the Coefficients

One of the main advantages of logistic regression is that its coefficients can be interpreted through a measure of association between the explanatory variables and the response.

To understand this property, we first need to introduce the concept of *odds*.

### Odds

Given a probability of success $p$, we define the odds as:

$$
\boxed{
\mathrm{odds}(p)=\frac{p}{1-p}
}
$$

The odds are the ratio of the probability that an event occurs to the probability that it does not occur.

For example, if:

$$
p=0.8
$$

then:

$$
\mathrm{odds}(p)=\frac{0.8}{0.2}=4
$$

This means that the event is four times as likely to occur as not to occur.

The logit function is simply the natural logarithm of the odds:

$$
\mathrm{logit}(p)=
\ln(\mathrm{odds}(p))
$$

### Interpreting the Coefficient $\beta_1$

Consider again the model:

$$
\mathrm{logit}(p(x))=\beta_0+\beta_1x
$$

Consider an initial value $x$ and a one-unit increase to $x+1$.

At the initial value:

$$
\mathrm{logit}(p(x))=\beta_0+\beta_1x
$$

At the increased value:

$$
\mathrm{logit}(p(x+1))
=\beta_0+\beta_1(x+1)
$$

Expanding:

$$
\mathrm{logit}(p(x+1))
=\beta_0+\beta_1x+\beta_1
$$

Subtracting the two expressions:

$$
\mathrm{logit}(p(x+1))-\mathrm{logit}(p(x))=\beta_1
$$

Substituting the definition of the logit:

$$
\ln\left(\frac{p(x+1)}{1-p(x+1)}\right)-\ln\left(\frac{p(x)}{1-p(x)}\right)=\beta_1
$$

Using the properties of logarithms:

$$
\ln\left[\frac{\mathrm{odds}(p(x+1))}{\mathrm{odds}(p(x))}\right]=\beta_1
$$

Exponentiating:

$$
\boxed{\frac{\mathrm{odds}(p(x+1))}{\mathrm{odds}(p(x))}=e^{\beta_1}}
$$

The resulting ratio is called the **odds ratio (OR)**.

Thus, $e^{\beta_1}$ represents the multiplicative change in the odds associated with a one-unit increase in the explanatory variable $x$.

We can distinguish three cases:

| Coefficient | Odds ratio | Interpretation |
|---|---|---|
| $\beta_1>0$ | $e^{\beta_1}>1$ | The odds increase |
| $\beta_1=0$ | $e^{\beta_1}=1$ | The odds remain unchanged |
| $\beta_1<0$ | $e^{\beta_1}<1$ | The odds decrease |

The interpretation is therefore multiplicative on the odds scale and additive on the log-odds scale.

### Interpreting the Intercept

In the model:

$$
\mathrm{logit}(p(x))=\beta_0+\beta_1x
$$

setting $x=0$:

$$
\mathrm{logit}(p(0))=\beta_0
$$

Consequently:

$$
\boxed{p(0)=\frac{1}{1+e^{-\beta_0}}}
$$

The intercept determines the predicted probability when the explanatory variable equals zero.

This interpretation is particularly useful when $x=0$ represents a meaningful condition in the phenomenon being studied.

### Extension to Multiple Covariates

For a model with multiple explanatory variables:

$$
\mathrm{logit}(p(x))=\beta_0+\beta_1x_1+\cdots+\beta_kx_k
$$

the interpretation generalizes directly.

The coefficient $\beta_j$ represents the change in log-odds associated with a one-unit increase in $x_j$, holding the other covariates constant.

Exponentiating the coefficient:

$$
\boxed{OR_j=e^{\beta_j}}
$$

gives the corresponding odds ratio conditional on the other explanatory variables.

This property allows us to quantify the direction and strength of the modeled association between each covariate and the response variable. In the additive log-odds model considered here, with no interaction terms, $OR_j$ is constant across values of the other covariates. When interactions are present, however, the effect of $x_j$ may depend on the other variables.

This is nevertheless a statistical association: a causal interpretation requires additional assumptions about the data-generating process.

## 3.7 Numerical Example: Odds Ratios and Probabilities

Consider a logistic regression model with one explanatory variable:

$$
\mathrm{logit}(p(x))=\beta_0+\beta_1x
$$

Suppose the coefficients take the following values:

$$
\beta_0=\ln\left(\frac14\right)
$$

$$
\beta_1=\ln 2
$$

The corresponding odds ratio is:

$$
OR=e^{\beta_1}=2
$$

This means that a one-unit increase in $x$ doubles the odds of the event.

Let us see what happens to the probabilities.

### Initial Probability

For $x=0$:

$$
\mathrm{logit}(p(0))=\beta_0
$$

Applying the logistic function:

$$
p(0)=\frac{1}{1+e^{-\beta_0}}
$$

Since:

$$
e^{\beta_0}=\frac14
$$

we obtain:

$$
p(0)=\frac{1/4}{1+1/4}=0.2
$$

The initial odds are:

$$
\mathrm{odds}(p(0))=\frac{0.2}{0.8}=0.25
$$

### Probability After a One-Unit Increase

For $x=1$:

$$
\mathrm{logit}(p(1))=\beta_0+\beta_1
$$

The odds become:

$$
\mathrm{odds}(p(1))
=e^{\beta_0+\beta_1}
$$

Using the properties of the exponential function:

$$
\mathrm{odds}(p(1))
=e^{\beta_0}e^{\beta_1}
$$

Therefore:

$$
\mathrm{odds}(p(1))=
0.25\cdot2=0.5
$$

To recover the probability, we use:

$$
p=\frac{\mathrm{odds}(p)}{1+\mathrm{odds}(p)}
$$

We obtain:

$$
p(1)=\frac{0.5}{1+0.5}=\frac13\approx0.333
$$

Let us summarize the results:

| Variable $x$ | Probability $p(x)$ | Odds |
|---|---|---|
| $0$ | $0.200$ | $0.25$ |
| $1$ | $0.333$ | $0.50$ |

A one-unit increase in $x$ therefore doubles the odds, while the probability rises from $20\%$ to approximately $33.3\%$, an increase of about $13.3$ percentage points.

**An odds ratio of $2$ does not mean that the probability doubles.**

This is because the relationship between probabilities and odds is nonlinear.

In general, for a one-unit increase in $x$, we can write:

$$
\boxed{p(x+1)=\frac{e^{\beta_1}p(x)}{1-p(x)+e^{\beta_1}p(x)}}
$$

The change in probability therefore depends on both the odds ratio and the initial probability.

This distinguishes logistic regression from ordinary linear regression: while $\beta_1$ represents a constant change on the log-odds scale, it does not imply a constant change in probability.

## 3.8 Remarks on Logistic Regression

We have derived binary logistic regression directly from the Bernoulli distribution.

The derivation can be summarized by the following relationships.

**Response distribution:**

$$
Y_i\mid x_i\sim\mathrm{Bernoulli}(p_i)
$$

**Natural parameter:**

$$
\theta_i=\ln\left(\frac{p_i}{1-p_i}\right)
$$

**Canonical link:**

$$
\mathrm{logit}(p_i)=\eta_i=x_i^\top\beta
$$

**Inverse link function:**

$$
\boxed{p_i=\sigma(x_i^\top\beta)=\frac{1}{1+e^{-x_i^\top\beta}}}
$$

We have therefore shown that the logistic function is not introduced arbitrarily; it arises from choosing the canonical link for the Bernoulli distribution.

This choice is mathematically convenient and makes it possible to interpret the coefficients using odds ratios.

However, it is not the only link function that can be used for a Bernoulli response. As discussed in the previous chapter, other links, such as the probit link, allow us to specify different relationships between probabilities and predictors.

The result of greatest interest here is that **the Bernoulli distribution has a single natural parameter, which can be interpreted as the logarithm of the ratio between two complementary probabilities**.

In the next chapter, we will generalize this construction to a response variable with $K$ mutually exclusive categories.

In that case, instead of a single natural parameter, we will obtain a vector of $K-1$ log probability ratios. This generalization will be the key step toward deriving the softmax function.
